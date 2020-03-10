/******************************************************************************
 * Copyright 2017 Baidu Robotic Vision Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
#include "feature_utils.h"
#include "image_utils.h"  // for sampleBrightnessHistogram
#include "timer.h"
#include "xppyramid.hpp"

#include <brisk/scale-space-feature-detector.h>
#include <brisk/internal/uniformity-enforcement.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <boost/lexical_cast.hpp>

// #define USE_OPENCV_OPTICAL_FLOW

namespace XP {

inline void init_mask(const cv::Size mask_size,
                      const cv::Mat_<uchar>& mask_in,
                      cv::Mat_<uchar>* mask_out) {
  if (mask_in.rows != 0) {
    *mask_out = mask_in.clone();
  } else {
    mask_out->create(mask_size);
    mask_out->setTo(0xff);
  }
}
//非极大值抑制以后得到了在原始图像上的特征点
inline bool suppress_sort_kp_in_larger_img(const cv::Mat& img_in_smooth/*当前帧*/,
                                           const cv::Mat_<uchar>& mask_in/*掩码*/,
                                           const std::vector<cv::KeyPoint>& kp_in_small_img/*第1层中提到的特征点*/,
                                           std::vector<cv::KeyPoint>* kp_in_large_img_ptr/*非极大值抑制以后的在第0层上的点*/,
                                           cv::Mat_<uchar>* mask_out) {
  CHECK_NOTNULL(kp_in_large_img_ptr);
  CHECK_NOTNULL(mask_out);
  init_mask(img_in_smooth.size(), mask_in, mask_out);//先用原始掩码
  std::vector<cv::KeyPoint>& kp_in_large_img = *kp_in_large_img_ptr;
  kp_in_large_img.clear();
  kp_in_large_img.reserve(kp_in_small_img.size());

  constexpr int compress_ratio = 2;
  std::vector<std::pair<int, float> > response_pair;//记录下每个点的响应,降序排列
  response_pair.reserve(kp_in_small_img.size());
  for (int i = 0; i < kp_in_small_img.size(); ++i) {
    response_pair.push_back({i, kp_in_small_img[i].response});
  }
  // Sort the response_pair with descending response
  std::sort(response_pair.begin(), response_pair.end(),
            [](const std::pair<int, float>& lhs, const std::pair<int, float>& rhs) {
              return lhs.second > rhs.second;
            });
//从响应大的点开始,设置半径为4的掩码
  // Suppress and refine propagated features starting from keypoints with stronger response
  constexpr int of_half_mask_size = 4;
  for (const auto& it : response_pair) {
    cv::KeyPoint kp = kp_in_small_img[it.first];
    kp.pt.x *= compress_ratio;
    kp.pt.y *= compress_ratio;
    int x = static_cast<int>(kp.pt.x + 0.5f);
    int y = static_cast<int>(kp.pt.y + 0.5f);
    if ((*mask_out)(y, x) == 0x00) {
#ifndef __FEATURE_UTILS_NO_DEBUG__
      if (VLOG_IS_ON(3)) {
        LOG(ERROR) << "of kp is masked out. ft id = " << kp.class_id;
      }
#endif
      continue;
    }

    // We generate acc mask to ensure the propagated keypoints won't cluster together.
    // The suppression radius here is intended to be small.
    for (int i = y - of_half_mask_size; i <= y + of_half_mask_size; ++i) {
      for (int j = x - of_half_mask_size; j <= x + of_half_mask_size; ++j) {
        (*mask_out)(i, j) = 0x00;
      }
    }
    kp_in_large_img.push_back(kp);
  }

  // Construct mask_out with acc larger suppresion radius for re-detection
  constexpr int det_half_mask_size = 8;
  for (const auto& kp : kp_in_large_img) {
    int x = static_cast<int>(kp.pt.x + 0.5f);
    int y = static_cast<int>(kp.pt.y + 0.5f);
    for (int i = y - det_half_mask_size; i <= y + det_half_mask_size; ++i) {
      for (int j = x - det_half_mask_size; j <= x + det_half_mask_size; ++j) {
        (*mask_out)(i, j) = 0x00;
      }
    }
  }
  return true;
}
//将特征点分配到网格中,网格中特征点数量满足阈值,就把网格整个mask掉,计算网格占用率
// This function will return
// 1) feat_min_num_thre:  The required minimum features based on the grids
//                        and the request_feat_num
// 2) grid_occ_ratio: The grid occupancy ratio
inline void compute_grid_mask(const std::vector<cv::KeyPoint>& keypoints,
                              const int request_feat_num,
                              cv::Mat_<uchar>* mask,
                              int* feat_min_num_thres/*最少需要的特征点数量*/,
                              float* grid_occ_ratio/*网格占用率*/) {
  CHECK_NOTNULL(mask);
  CHECK_NOTNULL(feat_min_num_thres);
  CHECK_NOTNULL(grid_occ_ratio);
  if (mask->rows == 0) {
    LOG(ERROR) << "Mask is not initialized";
    return;
  }
  const int grid_pixel = 80;//网格按80pixel * 80 pixel来划分  // 640 x 480 = 8 x 6 grid_pixel
  const int xbins = mask->cols / grid_pixel;//
  const int ybins = mask->rows / grid_pixel;
  cv::Mat_<int> grid_occ(ybins, xbins);//每个网格里特征点的个数
  grid_occ.setTo(0);

  for (const cv::KeyPoint& kp : keypoints) {
    int c = static_cast<int>(kp.pt.x) / grid_pixel;
    int r = static_cast<int>(kp.pt.y) / grid_pixel;
    if (c == xbins) --c;  //落在边界上的情况 Take care of the remainder bins
    if (r == ybins) --r;
    grid_occ(r, c)++;
  }

  // Add grid mask on top.  The grid_occ_threshold is needed in case the mask
  // suppression is too aggressive.  grid_occ_threshold can go to zero.  In that case,
  // single feature point can make its grid occupied.
  const int grid_occ_threshold = request_feat_num / xbins / ybins;  // 70 / (8 * 6)每个网格份应该分配多少特征点
  int grid_occ_num = 0;
  for (int r = 0; r < grid_occ.rows; ++r) {
    for (int c = 0; c < grid_occ.cols; ++c) {
      if (grid_occ(r, c) > grid_occ_threshold) {//如果这个网格已经被占据了,那么就将这个网格mask掉
        for (int i = r * grid_pixel; i < (r + 1) * grid_pixel; ++i) {
          for (int j = c * grid_pixel; j < (c + 1) * grid_pixel; ++j) {
            (*mask)(i, j) = 0x00;
          }
        }
        ++grid_occ_num;//网格被占据的数量
      }
    }
  }
  *grid_occ_ratio = static_cast<float>(grid_occ_num) / (xbins * ybins);//网格占用率
  *feat_min_num_thres = grid_occ_threshold * xbins * ybins;//最小需要的特征点
}

void optical_flow_stats(const std::vector<cv::Point2f>& small_img_pt_init,
                        const std::vector<cv::Point2f>& small_img_pt_flow,
#ifndef USE_OPENCV_OPTICAL_FLOW
                        const std::vector<bool>& status) {
#else
  const std::vector<uchar>& status) {
#endif
  std::vector<float> dists_sorted;
  dists_sorted.reserve(status.size());
  for (size_t i = 0; i < status.size(); ++i) {
    if (status[i]) {
      float diff_x = small_img_pt_init[i].x - small_img_pt_flow[i].x;
      float diff_y = small_img_pt_init[i].y - small_img_pt_flow[i].y;
      float dist = sqrt(diff_x * diff_x + diff_y * diff_y);
      dists_sorted.push_back(dist);
    }
  }

  std::sort(dists_sorted.begin(), dists_sorted.end());
  float medium = dists_sorted[dists_sorted.size() / 2];
  LOG(ERROR) << "flow dist medium = " << medium << " max = " << dists_sorted.back();
}

// Heuristically filter flows that differ too much from medium
void filter_outlier_flows(const std::vector<cv::Point2f>& small_img_pt_init,
                          const std::vector<cv::Point2f>& small_img_pt_flow,
                          const std::vector<cv::KeyPoint>& pre_small_img_kp,
                          const float threshold_multiplier,
#ifndef USE_OPENCV_OPTICAL_FLOW
                          std::vector<bool>* status) {
#else
  std::vector<uchar>* status) {
#endif
  std::vector<float> dists(status->size(), std::numeric_limits<float>::max());
  std::vector<float> dists_sorted;
  dists_sorted.reserve(status->size());//求一下初值和光流优化后之间变动的值,升序排列
  for (size_t i = 0; i < status->size(); ++i) {
    if ((*status)[i]) {
      float diff_x = small_img_pt_init[i].x - small_img_pt_flow[i].x;
      float diff_y = small_img_pt_init[i].y - small_img_pt_flow[i].y;
      dists[i] = sqrt(diff_x * diff_x + diff_y * diff_y);
      dists_sorted.push_back(dists[i]);
    }
  }

  std::sort(dists_sorted.begin(), dists_sorted.end());
  float medium = dists_sorted[dists_sorted.size() / 2];//中位数
  if (VLOG_IS_ON(2)) {
    LOG(ERROR) << "flow dist medium = " << medium << " max = " << dists_sorted.back();
  } else {
    VLOG(1) << "flow dist medium = " << medium << " max = " << dists_sorted.back();
  }

  // Apply acc heuristic lower bound for threshold (to avoid falsely removing correct flow
  // that is large due to very close distance)
  const float threshold = std::max(15.f, medium * threshold_multiplier);
  if (dists_sorted.back() > threshold) {//将变化距离过多的剔除,怕是错误点
    for (size_t i = 0; i < status->size(); ++i) {
      if ((*status)[i] && dists[i] > threshold) {
#ifndef USE_OPENCV_OPTICAL_FLOW
        (*status)[i] = false;
#else
        (*status)[i] = 0;
#endif
        if (VLOG_IS_ON(3)) {
          LOG(ERROR) << "filter ft_id: " << pre_small_img_kp[i].class_id
                     << " flow_dist = " << dists[i];
        }
      }
    }
  }
}

//// [NOTE] We keep this NON-pyramid general interface to support slave_det_mode = OF
//// This function is acc wrapper of the pyramid version.
//void propagate_with_optical_flow(const cv::Mat& img_in_smooth,
//                                 const cv::Mat_<uchar>& mask,
//                                 const cv::Mat& pre_image,
//                                 const cv::Mat& pre_image_orb_feature,
//                                 const std::vector<cv::KeyPoint>& pre_image_keypoints,
//                                 FeatureTrackDetector* feat_track_detector,
//                                 std::vector<cv::KeyPoint>* key_pnts_ptr,
//                                 cv::Mat_<uchar>* mask_with_of_out_ptr,
//                                 cv::Mat* orb_feat_OF_ptr,
//                                 const cv::Vec2f& init_pixel_shift,
//                                 const cv::Matx33f* K_ptr,
//                                 const cv::Mat_<float>* dist_ptr,
//                                 const cv::Matx33f* old_R_new_ptr,
//                                 const bool absolute_static) {
//  // Construct image pyramids first
//  std::vector<cv::Mat> img_in_smooth_pyramids, pre_image_pyramids;
//  build_pyramids(img_in_smooth, FeatureTrackDetector::kMaxPyraLevelOF, &img_in_smooth_pyramids);
//  build_pyramids(pre_image, FeatureTrackDetector::kMaxPyraLevelOF, &pre_image_pyramids);
//
//  bool fisheye = false;
//  propagate_with_optical_flow(img_in_smooth_pyramids,
//                              mask,
//                              pre_image_pyramids,
//                              pre_image_orb_feature,
//                              pre_image_keypoints,
//                              feat_track_detector,
//                              key_pnts_ptr,
//                              mask_with_of_out_ptr,
//                              orb_feat_OF_ptr,
//                              fisheye,
//                              init_pixel_shift,
//                              K_ptr,
//                              dist_ptr,
//                              old_R_new_ptr,
//                              absolute_static);
//}
//先过滤响应弱的前一帧的特征点,然后imu预测的特征点作为初值进行光流追踪,如果追踪效果不好,调整图像亮度再进行一次光流,最后极大值抑制,并且更新观测到的地图点的轨迹长度和坐标
void propagate_with_optical_flow(const std::vector<cv::Mat>& img_in_smooth_pyramids/*当前帧的所有金字塔图片*/,
                                 const cv::Mat_<uchar>& mask/*左相机掩码*/,
                                 const std::vector<cv::Mat>& pre_image_pyramids/*之前帧的所有金字塔图片*/,
                                 const cv::Mat& pre_image_orb_feature/*左相机上一帧检测到的特征点的描述子*/,
                                 const std::vector<cv::KeyPoint>& pre_image_keypoints/*左相机上一帧检测到的特征点*/,
                                 FeatureTrackDetector* feat_track_detector/*判断是否要存储上一帧点的观测信息*/,
                                 std::vector<cv::KeyPoint>* key_pnts_ptr/*左相机当前帧提取到的特征点*/,
                                 cv::Mat_<uchar>* mask_with_of_out_ptr,
                                 cv::Mat* orb_feat_OF_ptr/*左相机当前帧提取到的特征点对应的描述子*/,
                                 const bool fisheye,
                                 const cv::Vec2f& init_pixel_shift,
                                 const cv::Matx33f* K_ptr/*cv形式的左右相机内参*/,
                                 const cv::Mat_<float>* dist_ptr/*左右相机的畸变参数*/,
                                 const cv::Matx33f* old_R_new_ptr/*Rcl(pre)_cl(cur)左相机当前帧到前一帧的旋转*/,
                                 const bool absolute_static)
{
#ifndef __FEATURE_UTILS_NO_DEBUG__
    int t_harris_us = 0;
    int t_of_us = 0;
    int t_of_refine_us = 0;
    int t_of_orb_us = 0;
    int t_of_total_us = 0;
    MicrosecondTimer of_total_timer("propagate_with_optical_flow total time", 2);
    MicrosecondTimer harris_timer("propagate_with_optical_flow harris time", 2);
#endif

    // select the strong features from the previous image
    std::vector<cv::KeyPoint> pre_image_strong_kp_small;//Harris响应大于阈值的前一帧的特征点
    pre_image_strong_kp_small.reserve(pre_image_keypoints.size());

    std::vector<cv::Point2f> pre_image_strong_pt_small;//Harris响应大于阈值的前一帧的特征点像素坐标

    pre_image_strong_pt_small.reserve(pre_image_keypoints.size());
    std::vector<int> pre_image_strong_feature_ids;//记录过滤后的满足条件的特征点,这里的id是和pre_image_keypoints对应上的
    pre_image_strong_feature_ids.reserve(pre_image_keypoints.size());
    std::vector<cv::Point2f> cur_small_img_pt, cur_small_img_pt_init/*Harris响应大于阈值的,经过旋转预测后的前一帧的特征点在当前帧的像素坐标*/;
    cur_small_img_pt_init.reserve(pre_image_keypoints.size());
    // do optical flow
    const cv::Mat& img_in_smooth_small = img_in_smooth_pyramids.at(1);//当前帧第1层金字塔图片
    const cv::Mat& pre_image_small = pre_image_pyramids.at(1);//前一帧的第1层金字塔图片
    const cv::Mat& img_in_smooth = img_in_smooth_pyramids.at(0);//当前帧第0层金字塔图片,原始图片
    // always do OF on level 1，在第1层金字塔进行操作,缩放1倍就可以
    constexpr int pyra_level_OF = 2;
    constexpr int compress_ratio_OF = 1 << (pyra_level_OF - 1);
//利用imu数据算出的旋转对当前帧这些老点在的位置进行一个预测,同时过滤响应弱的老点
    if (pre_image_keypoints.size() > 0)
    {
        // compute harris response
        auto pre_image_keypoints_small = pre_image_keypoints;//上一帧特征点
        for (auto& kp : pre_image_keypoints_small) {//因为是在第1层进行操作,所以要对坐标进行缩放
            kp.pt.x /= compress_ratio_OF;
            kp.pt.y /= compress_ratio_OF;
        }//计算Harris响应
        ORBextractor::HarrisResponses(pre_image_small, 7, 0.04f, &pre_image_keypoints_small);
        std::vector<cv::Point2f> pre_image_pt_rotated;//经过旋转预测后的前一帧的特征点在当前帧的像素坐标
        std::vector<cv::Point2f> pre_feat_distorted(pre_image_keypoints_small.size());//带畸变的前一帧的原始特征点
        for (size_t i = 0; i < pre_image_keypoints_small.size(); i++) {
            pre_feat_distorted[i] = pre_image_keypoints_small[i].pt;
        }//如果内参,畸变,imu预测的旋转都有的时候,对前一帧的特征点在当前帧的像素坐标进行一个预测
        if (K_ptr != nullptr && dist_ptr != nullptr && old_R_new_ptr != nullptr) {
            // TODO(mingyu): Use vio project & backProject (faster / more accurate)
            cv::Matx33f K = *K_ptr;
            // let K fit small image//内参因为分辨率的变化,内参也要相应的变化
            K(0, 0) /= compress_ratio_OF;
            K(1, 1) /= compress_ratio_OF;
            K(0, 2) /= compress_ratio_OF;
            K(1, 2) /= compress_ratio_OF;
            const cv::Matx33f& old_R_new = *old_R_new_ptr;//Rcl(pre)_cl(cur)
            std::vector<cv::Point2f> pre_feat_undistorted;//去完畸变后的特征点的归一化坐标//TODO equi wya
            // TODO(mingyu): use vio undistort with NEON
            // wya:fisheye
            if(fisheye)
            {
                cv::Vec4f distortion_coeffs;
                for (int i = 0; i < distortion_coeffs.rows; ++i)
                    distortion_coeffs[i] = (*dist_ptr)(i);
                cv::fisheye::undistortPoints(pre_feat_distorted,//去畸变
                                             pre_feat_undistorted,
                                             K, distortion_coeffs);


                std::vector<cv::Point3f> feat_new_rays(pre_feat_undistorted.size());//经过旋转预测的当前帧的特征点的归一化坐标
                // reset kp position//pre_image_pt_rotated经过旋转预测后的前一帧的特征点在当前帧的像素坐标
                for (size_t i = 0; i < pre_feat_undistorted.size(); i++) {
                    cv::Vec3f pre_ray(pre_feat_undistorted[i].x, pre_feat_undistorted[i].y, 1);//前一帧下这个特征点的归一话坐标
                    cv::Vec3f predicted_ray = old_R_new.t() * pre_ray;// Pcur = Rcl(cur)_cl(pre) * Ppre //用旋转预测当前特征点的归一化坐标
                    feat_new_rays[i].x = predicted_ray[0] / predicted_ray[2];
                    feat_new_rays[i].y = predicted_ray[1] / predicted_ray[2];
                    feat_new_rays[i].z = 1;
                }

                cv::fisheye::projectPoints(feat_new_rays,pre_image_pt_rotated,
                                  cv::Matx31d::zeros(),
                                  cv::Matx31d::zeros(),
                                  K, distortion_coeffs);
            }
            else
            {
                cv::undistortPoints(pre_feat_distorted,//去畸变
                                    pre_feat_undistorted,
                                    K, *dist_ptr);

                std::vector<cv::Point3f> feat_new_rays(pre_feat_undistorted.size());//经过旋转预测的当前帧的特征点的归一化坐标
                for (size_t i = 0; i < pre_feat_undistorted.size(); i++) {
                    cv::Vec3f pre_ray(pre_feat_undistorted[i].x, pre_feat_undistorted[i].y, 1);//前一帧下这个特征点的归一话坐标
                    cv::Vec3f predicted_ray = old_R_new.t() * pre_ray;// Pcur = Rcl(cur)_cl(pre) * Ppre //用旋转预测当前特征点的归一化坐标
                    feat_new_rays[i].x = predicted_ray[0] / predicted_ray[2];
                    feat_new_rays[i].y = predicted_ray[1] / predicted_ray[2];
                    feat_new_rays[i].z = 1;
                }
                // reset kp position//pre_image_pt_rotated经过旋转预测后的前一帧的特征点在当前帧的像素坐标

                cv::projectPoints(feat_new_rays,
                                  cv::Matx31d::zeros(),
                                  cv::Matx31d::zeros(),
                                  K, *dist_ptr,
                                  pre_image_pt_rotated);
            }


            CHECK_EQ(pre_image_keypoints_small.size(), pre_image_pt_rotated.size());
        } else {
            pre_image_pt_rotated = pre_feat_distorted;
        }

        // Apply init_pixel_shift//图像坐标原点的偏移如果预设了,那就加上
        for (size_t i = 0; i < pre_image_pt_rotated.size(); ++i) {
            pre_image_pt_rotated[i].x += init_pixel_shift(0);
            pre_image_pt_rotated[i].y += init_pixel_shift(1);
        }//对小响应,图像外的点进行过滤
        for (size_t i = 0; i < pre_image_keypoints_small.size(); i++) {
            if (pre_image_keypoints_small[i].response > 1e-7 &&
                pre_image_keypoints_small[i].pt.x > 0 &&
                pre_image_keypoints_small[i].pt.y > 0 &&
                pre_image_keypoints_small[i].pt.x < img_in_smooth_small.cols - 1 &&
                pre_image_keypoints_small[i].pt.y < img_in_smooth_small.rows - 1) {
                pre_image_strong_pt_small.push_back(pre_image_keypoints_small[i].pt);//将满足条件的老点进行记录
                pre_image_strong_kp_small.push_back(pre_image_keypoints_small[i]);
                pre_image_strong_feature_ids.push_back(i);
                // OF init location
                cur_small_img_pt_init.push_back(pre_image_pt_rotated[i]);//将满足条件的老点进行旋转后的预测点进行记录
            } else {
#ifndef __FEATURE_UTILS_NO_DEBUG__
                if (VLOG_IS_ON(3)) {
                    if (pre_image_keypoints_small[i].response <= 1e-7) {
                        LOG(ERROR) << "pre_image_kpts_small[" << i << "] id = "
                                   << pre_image_keypoints_small[i].class_id << " has weak response: "
                                   << pre_image_keypoints_small[i].response;
                    }
                }
#endif
            }
        }
    }

#ifndef __FEATURE_UTILS_NO_DEBUG__
    CHECK_NOTNULL(key_pnts_ptr);
    t_harris_us = harris_timer.end();
    MicrosecondTimer of_timer("propagate_with_optical_flow OF time", 2);
#endif
    key_pnts_ptr->clear();//清楚一下数据,后续还要用
    CHECK_EQ(pre_image_strong_feature_ids.size(), pre_image_strong_kp_small.size());
    if (pre_image_strong_feature_ids.size() > 0) {//如果上一帧有满足要求的老点
        // We intentionally keep the initial guess clean in case we need to redo optical flow
        // with brightness adjusted images
        cur_small_img_pt = cur_small_img_pt_init;

        // Window size can only be 7 or 8
        cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 20, 0.01);//设置一下停止迭代条件
        std::vector<float> err;
        err.reserve(pre_image_strong_pt_small.size());
        constexpr int kMaxLevel = 3;
        constexpr int kStartLevel = 1;
        constexpr float kStrictPixelErr = 7.5;
        constexpr float kLoosePixelErr = 15.0;
        const cv::Size kWinSize(7, 7);

#ifndef USE_OPENCV_OPTICAL_FLOW
        // Convert to XP_OPTICAL_FLOW::XPKeyPoint first,首先转换一下响应大于阈值的老点数据结构
        std::vector<XP_OPTICAL_FLOW::XPKeyPoint> pre_xp_kp_small;
        pre_xp_kp_small.reserve(pre_image_strong_kp_small.size());
        for (const cv::KeyPoint& cv_kp : pre_image_strong_kp_small) {
            pre_xp_kp_small.push_back(XP_OPTICAL_FLOW::XPKeyPoint(cv_kp));
        }
        std::vector<bool> status;
        status.reserve(pre_image_strong_pt_small.size());
        //光流追踪,把cv的源码抄过来了,只支持winsize 7*7的光流
        XP_OPTICAL_FLOW::XPcalcOpticalFlowPyrLK(pre_image_pyramids/*之前帧的所有金字塔图片*/,
                                                img_in_smooth_pyramids/*当前帧的所有金字塔图片*/,
                                                &pre_xp_kp_small/*响应大于阈值的老点*/,
                                                &cur_small_img_pt/*Harris响应大于阈值的,经过旋转预测后的前一帧的特征点在当前帧的像素坐标*/,
                                                &status/*记录了每个点有没有被光流追踪上*/,
                                                &err/*误差函数*/,
                                                kWinSize/*每个金字塔等级的搜索窗口的winSize大小*/,
                                                kMaxLevel/*金字塔层数*/,
                                                kStartLevel/*从第几层开始*/,
                                                criteria/*终止条件*/,
                                                cv::OPTFLOW_USE_INITIAL_FLOW);
#else
        std::vector<uchar> status;
    status.reserve(pre_image_strong_pt_small.size());
    cv::calcOpticalFlowPyrLK(pre_image_small,
                             img_in_smooth_small,
                             pre_image_strong_pt_small,
                             cur_small_img_pt,
                             status,
                             err,
                             kWinSize,
                             kMaxLevel - 1,  // cv OpticalFlow starts from pyr1 (pre_image_small)
                             criteria,
                             cv::OPTFLOW_USE_INITIAL_FLOW);
#endif  // USE_OPENCV_OPTICAL_FLOW

        // Check the optical flow propagation result to heuristically test if
        // there is acc abrupt gain/exposure change
        CHECK_EQ(status.size(), err.size());
        CHECK_EQ(status.size(), cur_small_img_pt.size());

        // Heuristically remove outlier flows，如果光流前imu预测的初值和光流优化后差得很多,大于了阈值,那么也要剔除
        filter_outlier_flows(cur_small_img_pt_init/*光流前只用imu旋转修正的右目特征点坐标*/,
                             cur_small_img_pt/*光流优化后的右目特征点坐标*/,
                             pre_image_strong_kp_small/*Harris响应大于阈值的前一帧的特征点*/,  // need the class_id
                             4.f/*阈值*/,
                             &status/*是否追踪上*/);

        size_t of_feat_num_in = pre_image_strong_kp_small.size();//上一帧所有要去光流追踪的老点
        size_t of_feat_num_out = 0;//最终确定下来的点
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i] &&  err[i] < kStrictPixelErr) {
                ++of_feat_num_out;
            }
        }
        float of_prop_ratio = static_cast<float>(of_feat_num_out) / of_feat_num_in;
        bool redo_of = of_feat_num_in > 10 && of_prop_ratio < 0.4;//就说明追踪效果不好呗,要重新调整图像亮度然后LK追踪
        if (redo_of) {
            if (VLOG_IS_ON(2)) {
                LOG(ERROR) << "of prop ratio = " << of_prop_ratio << " try scale brightness";
            } else {
                VLOG(1) << "of prop ratio = " << of_prop_ratio << " try scale brightness";
            }
           //如果追踪不好,那就需要对之前帧的图像进行增益,达到亮度一致
            std::vector<int> hist_cur, hist_pre;//当前帧和之前帧的像素值分布直方图
            int avg_cur = 1, avg_pre = 1;//当前帧和之前帧的像素值分布直方图
            //计算当前帧的像素值分布直方图以及平均像素值
            sampleBrightnessHistogram(img_in_smooth_pyramids.at(0), &hist_cur, &avg_cur);
            //计算之前帧的像素值分布直方图以及平均像素值
            sampleBrightnessHistogram(pre_image_pyramids.at(0), &hist_pre, &avg_pre);
            //暴力采样算出最佳增益
            float pre_scale = matchingHistogram(hist_pre/*之前帧像素值分布直方图*/,
                                                hist_cur/*当前帧像素值分布直方图*/,
                                                static_cast<float>(avg_cur) / avg_pre/*平均像素值比值(c/p)作为初始增益*/);

            // [NOTE] cv::Mat_<uchar> automatically handles the clipping of uchar
            std::vector<cv::Mat> pre_image_pyramids_clone(pre_image_pyramids.size());
            for (size_t i = 0; i < pre_image_pyramids.size(); ++i) {
                pre_image_pyramids_clone[i] = pre_image_pyramids[i] * pre_scale;//增益
            }

            // Try optical flow again
            status.clear();
            err.clear();
            cur_small_img_pt = cur_small_img_pt_init;  // restore init guess for optical flow
#ifndef USE_OPENCV_OPTICAL_FLOW//一样还是用光流跟踪
            XP_OPTICAL_FLOW::XPcalcOpticalFlowPyrLK(pre_image_pyramids_clone,
                                                    img_in_smooth_pyramids,
                                                    &pre_xp_kp_small,
                                                    &cur_small_img_pt,
                                                    &status,
                                                    &err,
                                                    kWinSize,
                                                    kMaxLevel,
                                                    kStartLevel,
                                                    criteria,
                                                    cv::OPTFLOW_USE_INITIAL_FLOW);
#else
            cv::calcOpticalFlowPyrLK(pre_image_pyramids_clone.at(1),
                               img_in_smooth_small,
                               pre_image_strong_pt_small,
                               cur_small_img_pt,
                               status,
                               err,
                               kWinSize,
                               kMaxLevel - 1,  // cv OpticalFlow starts from pyr1 (pre_image_small)
                               criteria,
                               cv::OPTFLOW_USE_INITIAL_FLOW);
#endif  // USE_OPENCV_OPTICAL_FLOW

            // Filter flow(s) that differ too much from the medium
            // [NOTE] Use the init guess to compute the flow distance
            filter_outlier_flows(cur_small_img_pt_init,//同上
                                 cur_small_img_pt,
                                 pre_image_strong_kp_small,  // need the class_id
                                 4.f,
                                 &status);
            int re_of_feat_num_out = 0;
            for (size_t i = 0; i < status.size(); ++i) {
                if (status[i] && err[i] < kLoosePixelErr) {
                    ++re_of_feat_num_out;
                }
            }
            float re_of_prop_ratio = static_cast<float>(re_of_feat_num_out) / of_feat_num_in;//这次重新算一下追踪成功率
            if (VLOG_IS_ON(2)) {
                LOG(ERROR) << "After pre_scale = " << pre_scale << " of prop ratio = " << re_of_prop_ratio;
            } else {
                VLOG(1) << "After pre_scale = " << pre_scale << " of prop ratio = " << re_of_prop_ratio;
            }
        } else {
            if (VLOG_IS_ON(2)) {
                LOG(ERROR) << "of prop ratio = " << of_prop_ratio;
            } else {
                VLOG(1) << "of prop ratio = " << of_prop_ratio;
            }
        }

#ifndef __FEATURE_UTILS_NO_DEBUG__
        t_of_us = of_timer.end();
        if (VLOG_IS_ON(2)) {//输出一下追踪信息
            cv::Mat img_color;
            cv::cvtColor(img_in_smooth_small, img_color, CV_GRAY2BGR);
            for (size_t i = 0; i < cur_small_img_pt.size(); i++) {
                if (status[i]) {
                    // Draw the optical flow in blue//光流蓝色
                    cv::line(img_color, cur_small_img_pt_init[i], cur_small_img_pt[i],
                             cv::Scalar(255, 0, 0));
                    // Draw the init point from the prev point location in green//前后点关系绿色
                    cv::line(img_color, cur_small_img_pt_init[i], pre_image_strong_pt_small[i],
                             cv::Scalar(0, 0, 255));
                    // Draw the final point in green
                    img_color.at<cv::Vec3b>(cur_small_img_pt[i].y, cur_small_img_pt[i].x) =
                            cv::Vec3b(0, 255, 0);
                }
            }
            cv::putText(img_color, "flow", cv::Point2i(10, 35),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            cv::putText(img_color, "prev to init", cv::Point2i(10, 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            cv::putText(img_color, "final", cv::Point2i(10, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            cv::imwrite("/tmp/img_OF_after.png", img_color);
            cv::imshow("after OF", img_color);
        }
        MicrosecondTimer of_refine_timer("propagate_with_optical_flow OF refine time", 2);
#endif
        // refine OF results.
        // [NOTE] We impose non-max suppression on OF features for the following reasons:
        // 1) two different points come to the same location after OF
        // 2) OF features may cluster together and hence the feature distribution is sub-optimal
        CHECK_EQ(pre_image_strong_kp_small.size(), status.size());
        std::vector<cv::KeyPoint> cur_small_img_kp;//剔除大误差后追踪后的当前帧特征点
        cur_small_img_kp.reserve(status.size());
        const float pixel_err = redo_of ? kLoosePixelErr : kStrictPixelErr;//如果之前追踪效果不好,就允许像素误差大一些
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i] && err[i] < pixel_err) {
                // Need to preserve margin for computing orb descriptor (20 pixels at pyramid 0)
                if (cur_small_img_pt[i].x - 10 >= 0 &&
                    cur_small_img_pt[i].y - 10 >= 0 &&
                    cur_small_img_pt[i].x + 10 < img_in_smooth_small.cols &&
                    cur_small_img_pt[i].y + 10 < img_in_smooth_small.rows) {
                    cv::KeyPoint kp_cur_small(pre_image_strong_kp_small[i]);  //主要是为了保存金字塔层和class_id,id就对应于全局地图点id copy all info
                    kp_cur_small.pt = cur_small_img_pt[i];
                    cur_small_img_kp.push_back(kp_cur_small);
                } else {
#ifndef __FEATURE_UTILS_NO_DEBUG__
                    if (VLOG_IS_ON(3)) {
                        LOG(ERROR) << "OF rejects ft id = " << pre_image_strong_kp_small[i].class_id
                                   << " with boundary check";
                    }
#endif
                }
#ifndef __FEATURE_UTILS_NO_DEBUG__
            } else {
                if (VLOG_IS_ON(3)) {
                    LOG(ERROR) << "OF rejects ft id = " << pre_image_strong_kp_small[i].class_id
                               << ": status = " << static_cast<int>(status[i]) << " err = " << err[i];
                }
#endif
            }
        }
        if (true) {//非极大值抑制以后得到了在原始图像上的特征点
            // Suppress OF features and generate mask_with_of_out
            suppress_sort_kp_in_larger_img(img_in_smooth/*当前帧原始图片*/, mask/*左相机掩码*/, cur_small_img_kp/*剔除大误差后追踪后的当前帧特征点(1层)*/,
                                           key_pnts_ptr/*最终当前帧(0层)提取到的描述子*/, mask_with_of_out_ptr);
        } else {
            // Do not refine corner location.  Do not suppress OF features, either.
            key_pnts_ptr->resize(cur_small_img_kp.size());
            for (size_t i = 0; i < cur_small_img_kp.size(); ++i) {
                (*key_pnts_ptr)[i] = cur_small_img_kp[i];
                (*key_pnts_ptr)[i].pt.x *= 2;
                (*key_pnts_ptr)[i].pt.y *= 2;
            }
            init_mask(img_in_smooth.size(), mask, mask_with_of_out_ptr);
        }
        // 进行点管理
        // Update active feature tracks of this frame.  Also perform the random drop out if necessary.
        if (feat_track_detector) {
            if (absolute_static) {
                feat_track_detector->filter_static_features(key_pnts_ptr);
            }
            feat_track_detector->update_feature_tracks(key_pnts_ptr);//更新feature_tracks_map_点管理信息,对看到的地图点设成active = true
        }

#ifndef __FEATURE_UTILS_NO_DEBUG__
        t_of_refine_us = of_refine_timer.end();
        if (VLOG_IS_ON(2)) {
            cv::Mat img_color;
            cv::cvtColor(img_in_smooth, img_color, CV_GRAY2BGR);
            if (mask_with_of_out_ptr->rows != 0) {
                // Plot the mask with dark gray
                for (int i = 0; i < img_in_smooth.rows; ++i) {
                    for (int j = 0; j < img_in_smooth.cols; ++j) {
                        if ((*mask_with_of_out_ptr)(i, j) == 0x00) {
                            img_color.at<cv::Vec3b>(i, j)[0] = 0x09;
                            img_color.at<cv::Vec3b>(i, j)[1] = 0x09;
                            img_color.at<cv::Vec3b>(i, j)[2] = 0x09;
                        }
                    }
                }
            }
            // Overlay OF features on top with yellow
            for (size_t i = 0; i < key_pnts_ptr->size(); i++) {
                cv::circle(img_color, (*key_pnts_ptr)[i].pt, 2, cv::Vec3b(0, 255, 255));
            }
            cv::putText(img_color,
                        "feat# " + boost::lexical_cast<std::string>(key_pnts_ptr->size()),
                        cv::Point2i(10, 20),
                        cv::FONT_HERSHEY_PLAIN, 1, cv::Vec3b(0, 0, 255), 1);
            cv::imwrite("/tmp/img_OF_compress_w_mask.png", img_color);
        }
#endif
        if (orb_feat_OF_ptr != nullptr) {
            // compute ORB feat in level 0
            if (key_pnts_ptr->size() > 0) {
#ifndef __FEATURE_UTILS_NO_DEBUG__
                MicrosecondTimer of_orb_timer("propagate_with_optical_flow orb desc time", 2);
#endif
#ifdef __ARM_NEON__
                ORBextractor::computeDescriptorsN512(img_in_smooth, *key_pnts_ptr, orb_feat_OF_ptr);
#else//计算当前帧的特征点描述子
                ORBextractor::computeDescriptors(img_in_smooth, *key_pnts_ptr, orb_feat_OF_ptr);
#endif  // __ARM_NEON__
#ifndef __FEATURE_UTILS_NO_DEBUG__
                t_of_orb_us = of_orb_timer.end();
                CHECK_EQ(orb_feat_OF_ptr->rows, key_pnts_ptr->size());
#endif
            }
        }
    }
#ifndef __FEATURE_UTILS_NO_DEBUG__
    t_of_total_us = of_total_timer.end();
    VLOG(1) << "propagate_with_optical_flow time (imu_measures)"
            << " harris:" << t_harris_us
            << " of:" << t_of_us
            << " of_refine:" << t_of_refine_us
            << " of_orb:" << t_of_orb_us
            << " of_total:" << t_of_total_us;
#endif
}

FeatureTrackDetector::FeatureTrackDetector(const int length_thres,
                                           const float drop_rate,
                                           const bool use_fast,
                                           const int uniform_radius,
                                           const cv::Size& img_size) :
    length_threshold_(length_thres), drop_rate_(drop_rate),
    use_fast_(use_fast), uniform_radius_(uniform_radius)
{
    generator_.seed(0);
    distribution_ = std::uniform_real_distribution<float>(0.f, 1.f);//0-1之间随机数
    // Allocated buffer for Optical flow pyramids
    //扩容图像存储
    prev_pyramids_buffer_ = std::shared_ptr<uchar>(
            new uchar[img_size.width * img_size.height * 2]);
    curr_pyramids_buffer_ = std::shared_ptr<uchar>(
            new uchar[img_size.width * img_size.height * 2]);
    if (prev_pyramids_buffer_ == nullptr ||
        curr_pyramids_buffer_ == nullptr) {
        LOG(FATAL) << "Allocat buffer for optical flow pyramids failed.";
  }
}

int FeatureTrackDetector::add_new_feature_track(const cv::Point2f pt) {
  int new_ft_id = id_generator_.get();//得到一个新的id(这个是特征点的id还是地图点的id不太确定)
  feature_tracks_map_.emplace(new_ft_id, FeatureTrack(pt));
  return new_ft_id;
}

void FeatureTrackDetector::mark_all_feature_tracks_dead() {
  for (std::map<int, FeatureTrack>::iterator ft_it = feature_tracks_map_.begin();
       ft_it != feature_tracks_map_.end(); ++ft_it) {
    ft_it->second.isActive = false;
  }
}

void FeatureTrackDetector::filter_static_features(std::vector<cv::KeyPoint>* key_pnts) {
  std::vector<cv::KeyPoint> static_key_pnts;
  static_key_pnts.reserve(key_pnts->size());
  for (const cv::KeyPoint& kp : *key_pnts) {
    std::map<int, FeatureTrack>::iterator ft_it = feature_tracks_map_.find(kp.class_id);
    CHECK(ft_it != feature_tracks_map_.end())
        << "feature track id: " << kp.class_id << " doesn't exist!";
    cv::Point2f diff = ft_it->second.point - kp.pt;
    if (std::abs(diff.x) < 0.2 && std::abs(diff.y) < 0.2) {
      static_key_pnts.push_back(kp);
    }
  }
#ifndef __FEATURE_UTILS_NO_DEBUG__
  VLOG(2) << " filter_static_features: " << key_pnts->size() << " -> " << static_key_pnts.size();
#endif
  *key_pnts = static_key_pnts;
}
//输入当前帧追踪到的上一帧的特征点
void FeatureTrackDetector::update_feature_tracks(std::vector<cv::KeyPoint>* key_pnts) {
  for (cv::KeyPoint& kp : *key_pnts) {
      std::map<int, FeatureTrack>::iterator ft_it = feature_tracks_map_.find(kp.class_id);
    CHECK(ft_it != feature_tracks_map_.end())//只是对上一帧的特征点做追踪所以肯定会有这个点的
        << "feature track id: " << kp.class_id << " doesn't exist!";
    ft_it->second.length++;//追踪长度加1

    // Random drop out and re-assign to acc new one if this feature track is too long.
    // Otherwise, mark this feature track active.//这里是不是论文里说的分段处理,不过好像没用到
    if (ft_it->second.length > length_threshold_ && distribution_(generator_) < drop_rate_) {
      kp.class_id = id_generator_.get();  // Re-assign acc new feature track id
      feature_tracks_map_.emplace(kp.class_id, FeatureTrack(kp.pt));//如果轨迹过长,就按drop_rate_的概率,生成新的轨迹,不过drop_rate_也太低了吧,就0.05概率?
#ifndef __FEATURE_UTILS_NO_DEBUG__
      VLOG(2) << " insert ftId = " << kp.class_id << " dropout " << ft_it->first
              << " with length " << ft_it->second.length << "!";
#endif
    } else {
      ft_it->second.isActive = true;
      ft_it->second.point = kp.pt;  // Update the pixel location
    }

  }
}

void FeatureTrackDetector::flush_feature_tracks(const std::vector<cv::KeyPoint>& key_pnts) {
  int max_feature_track_id = 0;
  for (const cv::KeyPoint kp : key_pnts) {
    std::map<int, FeatureTrack>::iterator ft_it = feature_tracks_map_.find(kp.class_id);
    if (ft_it == feature_tracks_map_.end()) {
      feature_tracks_map_.emplace(kp.class_id, FeatureTrack(kp.pt));
    } else {
      ft_it->second.length++;
      ft_it->second.isActive = true;
      ft_it->second.point = kp.pt;
    }
  }

  // Force update the id_generator_ to the maximum feat_id we've seen
  if (!feature_tracks_map_.empty()) {
    int max_feat_id = feature_tracks_map_.rbegin()->first;
    id_generator_.reset(max_feat_id);
  }
}

void FeatureTrackDetector::ComputeDescriptors(const cv::Mat& img_in_smooth,//相机图片
                            std::vector<cv::KeyPoint>* key_pnts_ptr,cv::Mat* orb_feat_ptr)
    {
        if (orb_feat_ptr != nullptr) {
            // detect orb at pyra 0
            // since orb radius is 15, which is alraedy pretty big
#ifdef __ARM_NEON__
            ORBextractor::computeDescriptorsN512(img_in_smooth, *key_pnts_ptr, orb_feat_ptr);
#else
            ORBextractor::computeDescriptors(img_in_smooth, *key_pnts_ptr, orb_feat_ptr);
#endif
        }
    }


//输入左相机图片,图像掩码,最大提取的特征点数量,金字塔层数,fast点阈值,特征点,描述子
//step1在第1层金字塔上提取fast点,算Harris响应
//step2对提取的点进行过滤,让特征点enforce_uniformity_radius范围点无其他点
//step3计算方向,然后将点从1层到第0层细化坐标
//step4将在1层Harris响应小的过滤以后,进行点管理并且计算描述子
bool FeatureTrackDetector::detect(const cv::Mat& img_in_smooth,
                                  const cv::Mat_<uchar>& mask,
                                  int request_feat_num,
                                  int pyra_level,  // Total pyramid levels, including the base image
                                  int fast_thresh,
                                  std::vector<cv::KeyPoint>* key_pnts_ptr,
                                  cv::Mat* orb_feat_ptr) {
  return XP::detect_orb_features(img_in_smooth,//相机图片
                                 mask,//图像掩码
                                 request_feat_num,//最大提取的特征数目
                                 pyra_level,//金字塔层数
                                 fast_thresh,//fast角点提取阈值
                                 use_fast_,//是否是提取fast
                                 uniform_radius_,//点多少像素范围内不要有其他点
                                 key_pnts_ptr,//特征点
                                 orb_feat_ptr,//orb描述子
                                 this,
                                 1e-7 /*refine_harris_threshold*/);
}

    bool FeatureTrackDetector::detect_for_loop(const cv::Mat& img_in_smooth,
                                      const cv::Mat_<uchar>& mask,
                                      int request_feat_num,
                                      int pyra_level,  // Total pyramid levels, including the base image
                                      int fast_thresh,
                                      std::vector<cv::KeyPoint>* key_pnts_ptr,
                                      cv::Mat* orb_feat_ptr)
{

//                      ORBextractor orb(request_feat_num , 2, 1,
//                                       ORBextractor::HARRIS_SCORE,
//                                       fast_thresh,
//                                       true);
//                      //提取fast(但是是用harris响应值),但没有计算256位的描述子,注意,这里只在det_pyra_level这层提取了特征点
//                      orb.detectgf_compute(img_in_smooth,
//                                           mask,
//                                  key_pnts_ptr);
//    ComputeDescriptors(img_in_smooth,key_pnts_ptr,orb_feat_ptr);
//    return true;
//一致的话表现比较好
        return XP::detect_orb_features(img_in_smooth,//相机图片
                                       mask,//图像掩码
                                       request_feat_num,//最大提取的特征数目
                                       pyra_level,//金字塔层数
                                       fast_thresh,//fast角点提取阈值
                                       use_fast_,//是否是提取fast
                                       10,//点多少像素范围内不要有其他点
                                       key_pnts_ptr,//特征点
                                       orb_feat_ptr,//orb描述子
                                       nullptr,
                                       1e-9 /*refine_harris_threshold*/);



 }

// TODO(mingyu): store pre_image_orb_feature, pre_image_keypoints in
// FeatureTrackDetector member variables
    //step1:先过滤响应弱的前一帧的特征点,然后imu预测的特征点作为初值进行光流追踪,如果追踪效果不好,
    // 调整图像亮度再进行一次光流,最后极大值抑制,并且更新观测到的地图点的轨迹长度和坐标
    //step2:将特征点分配到网格中,并且判断是否要检测新的特征点
    //step3:如果要检测新的点，就检测完了将新点加进点管理中
bool FeatureTrackDetector::optical_flow_and_detect(const cv::Mat_<uchar>& mask/*左相机掩码*/,
                                                   const cv::Mat& pre_image_orb_feature/*左相机上一帧检测到的特征点的描述子*/,
                                                   const std::vector<cv::KeyPoint>& prev_img_kpts/*左相机上一帧检测到的特征点*/,
                                                   int request_feat_num/*最大要求提取的特征点数量*/,
                                                   int pyra_level_det/*金字塔层*/,
                                                   int fast_thresh/*fast阈值*/,
                                                   std::vector<cv::KeyPoint>* key_pnts_ptr/*左相机当前帧提取到的特征点*/,
                                                   cv::Mat* orb_feat_ptr/*左相机当前帧提取到的特征点对应的描述子*/,
                                                   const bool fisheye,
                                                   const cv::Vec2f& init_pixel_shift,
                                                   const cv::Matx33f* K_ptr/*cv形式的左右相机内参*/,
                                                   const cv::Mat_<float>* dist_ptr/*左右相机的畸变参数*/,
                                                   const cv::Matx33f* old_R_new_ptr/*Rcl(pre)_cl(cur)左相机当前帧到前一帧的旋转*/,
                                                   const bool absolute_static) {
  // Mark all feature tracks dead first!
  //feature_tracks_map_存储的是上一帧左目看到的地图点,应该是和上一帧看到的左目的特征点是一致的
//  if (prev_img_kpts.size() != this->feature_tracks_number()) {
//    LOG(ERROR) << "Inconsistent prev keypoints (" << prev_img_kpts.size()
//               << ") vs feature_tracks_map ("
//               << this->feature_tracks_number() << ")";
//  }
  //将所有的上一帧观测到的地图点状态设成不能看到的,真能看到的话就把active设成false
  this->mark_all_feature_tracks_dead();//将所有的
  const cv::Mat& img_in_smooth = curr_img_pyramids_.at(0);//原始图片
  cv::Mat orb_feat_OF;
    //先过滤响应弱的前一帧的特征点,然后imu预测的特征点作为初值进行光流追踪,如果追踪效果不好,
    // 调整图像亮度再进行一次光流,最后极大值抑制,并且更新观测到的地图点的轨迹长度和坐标
    XP::propagate_with_optical_flow(curr_img_pyramids_/*当前帧的所有金字塔图片*/,
                                  mask/*左相机掩码*/,
                                  prev_img_pyramids_/*之前帧的所有金字塔图片*/,
                                  pre_image_orb_feature/*左相机上一帧检测到的特征点的描述子*/,
                                  prev_img_kpts/*左相机上一帧检测到的特征点*/,
                                  this,
                                  key_pnts_ptr/*左相机当前帧提取到的特征点*/,
                                  &mask_with_of_out_/*非极大值以后的掩码,原始mask加点的mask*/,
                                  (orb_feat_ptr ? &orb_feat_OF : nullptr)/*左相机当前帧提取到的特征点对应的描述子*/,
                                  fisheye,
                                  init_pixel_shift,
                                  K_ptr/*cv形式的左右相机内参*/,
                                  dist_ptr/*左右相机的畸变参数*/,
                                  old_R_new_ptr/*Rcl(pre)_cl(cur)左相机当前帧到前一帧的旋转*/,
                                  absolute_static);

#ifndef __FEATURE_UTILS_NO_DEBUG__
  int t_redet_decision_us = 0;
  int t_redet_us = 0;
  int t_redet_total_us = 0;
  MicrosecondTimer redet_total_timer("detect_features_with_optical_flow redet total time", 2);
#endif

  int propagated_feat_num = static_cast<int>(key_pnts_ptr->size());//追踪到的点

  // Determine if we need to detect new feature
#ifndef __FEATURE_UTILS_NO_DEBUG__
  MicrosecondTimer redet_decision_timer("detect_features_with_optical_flow redet decision time", 2);
#endif
//将特征点分配到网格中,网格中特征点数量满足阈值,就把网格整个mask掉,更新mask,需要的特征点,计算网格占用率
  // Check re-detect criteria:
  // 1) Features are too concentrated: the grid occupancy ratio is too low 网格占用率太低
  // or
  // 2) Features are too few: depending on the threshold for grid_occ_ratio and request_feat_num,
  //    it is possible that grid_occ_ratio exceeds the bar, but the total feature number is still
  //    not enough.需要的最小特征点数量不足
  int feat_min_num_thres = 0;
  float grid_occ_ratio = 0;
  compute_grid_mask(*key_pnts_ptr/*当前帧追踪到的特征点*/, request_feat_num/*需要的特征点数量*/,
                    &mask_with_of_out_/*非极大值以后的掩码,原始mask加点的mask*/, &feat_min_num_thres, &grid_occ_ratio);
  bool re_detect = grid_occ_ratio < 0.4 || propagated_feat_num < feat_min_num_thres;//判断是否需要检测新的特征点,条件就是以上的两点

#ifndef __FEATURE_UTILS_NO_DEBUG__
  t_redet_decision_us = redet_decision_timer.end();
  if (VLOG_IS_ON(2)) {
    cv::Mat img_color;
    cv::cvtColor(img_in_smooth, img_color, CV_GRAY2BGR);
    // Plot the mask with dark gray
    for (int i = 0; i < img_in_smooth.rows; ++i) {
      for (int j = 0; j < img_in_smooth.cols; ++j) {
        if (mask_with_of_out_(i, j) == 0x00) {
          img_color.at<cv::Vec3b>(i, j)[0] = 0x09;
          img_color.at<cv::Vec3b>(i, j)[1] = 0x09;
          img_color.at<cv::Vec3b>(i, j)[2] = 0x09;
        }
      }
    }
    // Draw OF features in yellow
    for (int i = 0; i < orb_feat_OF.rows; i++) {
      cv::circle(img_color, (*key_pnts_ptr)[i].pt, 2, cv::Vec3b(0, 255, 255));
    }
    cv::putText(img_color,
                "feat# " + boost::lexical_cast<std::string>(propagated_feat_num),
                cv::Point2i(10, 20),
                cv::FONT_HERSHEY_PLAIN, 1, cv::Vec3b(0, 0, 255), 1);
    cv::imshow("img_OF_and_occ_grid", img_color);
    cv::imwrite("/tmp/img_OF_and_occ_grid.png", img_color);
  }
  VLOG(1) << "OF gets " << propagated_feat_num << " pnts "
          << " request_feat_num " << request_feat_num;
  MicrosecondTimer redet_timer("detect_features_with_optical_flow redet time", 2);
#endif
  cv::Mat orb_feat_new;
  if (re_detect) {
    // [NOTE] detect function handles the bookkeeping of these newly created feature tracks
    // [NOTE] In the rare case re_detect is triggered and propagated_feat_num == request_feat_num,
    // (feature distribution is too concentrated), we enforce to detect up to 10 more features.
    int request_new_feat_num = (request_feat_num > propagated_feat_num) ?//需要检测的特征点的数量
                               request_feat_num - propagated_feat_num : 10;
    std::vector<cv::KeyPoint> key_pnts_new;//新提取到的特征点


//    检测新的特征点,在detect中会对新点做点管理,存在feature_tracks_map_中
    this->detect(img_in_smooth,//一样还是orb的代码检测过程
                 mask_with_of_out_/*只在没有mask的地方生成新点*/,
                 request_new_feat_num,
                 pyra_level_det,
                 fast_thresh,
                 &key_pnts_new,
                 &orb_feat_new);
    key_pnts_ptr->insert(key_pnts_ptr->end(), key_pnts_new.begin(), key_pnts_new.end());//将新点存到当前帧的特征点存储中
#ifndef __FEATURE_UTILS_NO_DEBUG__
    VLOG(1) << "After OF gets " << key_pnts_new.size() << " new pnts "
            << " request_new_feat_num " << request_new_feat_num;
#endif
  }
  if (orb_feat_ptr != nullptr) {
    orb_feat_ptr->create(orb_feat_OF.rows + orb_feat_new.rows, 32, CV_8U);
    if (orb_feat_OF.rows > 0) {
      orb_feat_OF.copyTo(orb_feat_ptr->rowRange(0, orb_feat_OF.rows));
#ifndef __FEATURE_UTILS_NO_DEBUG__
      CHECK_EQ(orb_feat_ptr->at<uchar>(0, 0), orb_feat_OF.at<uchar>(0, 0));
#endif
    }
    if (orb_feat_new.rows > 0) {
      orb_feat_new.copyTo(orb_feat_ptr->rowRange(orb_feat_OF.rows, orb_feat_ptr->rows));
#ifndef __FEATURE_UTILS_NO_DEBUG__
      CHECK_EQ(orb_feat_ptr->at<uchar>(orb_feat_OF.rows, 0), orb_feat_new.at<uchar>(0, 0));
#endif
    }
  }

  // Clean up dead feature tracks//遍历特征管理器
  std::map<int, FeatureTrack>::iterator ft_it = feature_tracks_map_.begin();
  while (ft_it != feature_tracks_map_.end()) {
    if (!ft_it->second.isActive) {//将看不到的点从feature_tracks_map_去掉
#ifndef __FEATURE_UTILS_NO_DEBUG__
      if (VLOG_IS_ON(3)) {
        LOG(ERROR) << "dead ft id = " << ft_it->first
                   << " length = " << ft_it->second.length;
      }
#endif
      ft_it = feature_tracks_map_.erase(ft_it);
    } else {
      ++ft_it;
    }
  }

#ifndef __FEATURE_UTILS_NO_DEBUG__
  t_redet_us = redet_timer.end();
  t_redet_total_us = redet_total_timer.end();
  if (re_detect) {
    VLOG(1) << " grid occ ratio = " << grid_occ_ratio * 100 << "%"
            << " OF feat#: " << orb_feat_OF.rows
            << " final feat# " << key_pnts_ptr->size() << " redet!";
  } else {
    VLOG(1) << " grid occ ratio = " << grid_occ_ratio * 100 << "%"
            << " OF feat#: " << orb_feat_OF.rows
            << " final feat# " << key_pnts_ptr->size();
  }
  VLOG(1) << "detect_features_with_optical_flow time (imu_measures)"
          << " redet_decision: " << t_redet_decision_us
          << " redet: " << t_redet_us
          << " redet total: " << t_redet_total_us;
  if (VLOG_IS_ON(2)) {
    CHECK_EQ(key_pnts_ptr->size(), orb_feat_OF.rows + orb_feat_new.rows);
    cv::Mat img_color;
    cv::cvtColor(img_in_smooth, img_color, CV_GRAY2BGR);
    // Draw OF features in blue
    for (int i = 0; i < orb_feat_OF.rows; i++) {
      cv::circle(img_color, (*key_pnts_ptr)[i].pt, 2, cv::Vec3b(255, 0, 0));
    }
    // Draw newly detected features in red
    for (int i = orb_feat_OF.rows; i < orb_feat_OF.rows + orb_feat_new.rows; ++i) {
      cv::circle(img_color, (*key_pnts_ptr)[i].pt, 2, cv::Vec3b(0, 0, 255));
    }
    cv::putText(img_color, "OF feats: " + boost::lexical_cast<std::string>(orb_feat_OF.rows),
                cv::Point2i(10, 20), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 0, 0), 1);
    cv::putText(img_color, "redet feats: " + boost::lexical_cast<std::string>(orb_feat_new.rows),
                cv::Point2i(10, 35), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 255), 1);
    cv::putText(img_color,
                "Total feat# " + boost::lexical_cast<std::string>(key_pnts_ptr->size()),
                cv::Point2i(10, 50), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 255, 0), 1);
    cv::imwrite("/tmp/img_OF_and_redet.png", img_color);
  }
#endif
  return true;
}
}  // namespace XP
