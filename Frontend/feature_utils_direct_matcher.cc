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
#ifndef __DEVELOPMENT_DEBUG_MODE__
#define __FEATURE_UTILS_NO_DEBUG__
#endif
#include "feature_utils.h"
#include "patch_score.h"

#include "cameras/PinholeCamera.hpp"
#include "cameras/RadialTangentialDistortion8.hpp"
#include "cameras/RadialTangentialDistortion.hpp"
#include "cameras/EquidistantDistortion.hpp"

#include <boost/lexical_cast.hpp>
#ifndef __FEATURE_UTILS_NO_DEBUG__
#include <opencv2/highgui.hpp>
#endif
#include <opencv2/core/eigen.hpp>
// define this MACROS to enable profing and verifying align2D_NEON
#undef _ALIGN2D_TIMING_AND_VERIFYING
#ifdef _ALIGN2D_TIMING_AND_VERIFYING
#include <vio/timing/ProfilingTimer.hpp>
using vio::timing::MicrosecondStopwatch;
MicrosecondStopwatch timer_align2d("align2d", 0);
MicrosecondStopwatch timer_align2d_neon("align2d neon", 0);
float align2d_total_time = 0.f;
float align2d_total_time_neon = 0.f;
#endif

namespace XP {

using Eigen::Vector2i;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Matrix2f;
using Eigen::Matrix3f;
using Eigen::Matrix4f;

// The triangulated point at the current frame coordinate can be represented as:
// R_cur_ref * f_ref * d_ref + t_cur_ref =  f_cur * d_cur,
// where d_ref and d_cur are scales, respetively
// We can organize the equation into acc linear system to solve d in the form of Ax = b:
// [R_cur_ref * f_ref  f_cur] * [d_ref; - d_cur] = - t_cur_ref
//   A = [R_cur_ref * f_ref f_cur],  3 x 2
//   x = [d_ref; -d_cur],  2 x 1
//   b = - t_cur_ref,  3 x 1
// x = (AtA)^-1 * At * b
//
// The returned depth is based on the reference frame
//如上所示,构造Ax = b直接求线性解,返回的是在左目坐标系中的深度
bool depthFromTriangulation(const Matrix3f& R_cur_ref,
                            const Vector3f& t_cur_ref,
                            const Vector3f& f_ref,
                            const Vector3f& f_cur,
                            float* depth) {
  Eigen::Matrix<float, 3, 2> A;
  A << R_cur_ref * f_ref, f_cur;
  const Matrix2f AtA = A.transpose() * A;
  if (AtA.determinant() < 1e-6) {
    // TODO(mingyu): figure the right threshold for float
    *depth = 1000;  // acc very far point
    return false;
  }

  const Vector2f depth2 = - AtA.inverse() * A.transpose() * t_cur_ref;
  *depth = fabs(depth2[0]);//左目深度
  return true;
}

void DirectMatcher::createPatchFromPatchWithBorder() {
  uint8_t* ref_patch_ptr = patch_;
  for (int y = 1; y < patch_size_ + 1; ++y, ref_patch_ptr += patch_size_) {
    uint8_t* ref_patch_border_ptr = patch_with_border_ + y * (patch_size_ + 2) + 1;
    memcpy(ref_patch_ptr, ref_patch_border_ptr, patch_size_);
    /*
    for (int x = 0; x < patch_size_; ++x) {
      ref_patch_ptr[x] = ref_patch_border_ptr[x];
    }
    */
  }
}

bool DirectMatcher::findMatchDirect(const vio::cameras::CameraBase& cam_ref,
                                    const vio::cameras::CameraBase& cam_cur,
                                    const Vector2f& px_ref,
                                    const Vector3f& f_ref,
                                    const Matrix3f& R_cur_ref,
                                    const Vector3f& t_cur_ref,
                                    const int level_ref,
                                    const float depth_ref,
                                    const std::vector<cv::Mat>& pyrs_ref,
                                    const std::vector<cv::Mat>& pyrs_cur,
                                    const bool edgelet_feature,
                                    Vector2f* px_cur) {
  CHECK_NEAR(f_ref[2], 1.f, 1e-6);
  CHECK_EQ(pyrs_ref.size(), pyrs_cur.size());

  // TODO(mingyu): check return boolean of getWarpMatrixAffine
  // warp affine
  warp::getWarpMatrixAffine(cam_ref,
                            cam_cur,
                            px_ref,
                            f_ref,
                            depth_ref,
                            R_cur_ref,
                            t_cur_ref,
                            level_ref,
                            &A_cur_ref_);
  const int max_level = pyrs_ref.size() - 1;
  const int search_level = warp::getBestSearchLevel(A_cur_ref_, max_level);

  // TODO(mingyu): check return boolean of warpAffine
  warp::warpAffine(A_cur_ref_,
                   pyrs_ref[level_ref],
                   px_ref,
                   level_ref,
                   search_level,
                   halfpatch_size_ + 1,
                   patch_with_border_);
  createPatchFromPatchWithBorder();

  // px_cur should be set
  Vector2f px_scaled = *px_cur / (1 << search_level);

  bool success = false;
  if (edgelet_feature) {
    // TODO(mingyu): currently not used until we further refine the feature type
    //               with gradient direction.
    //               Fast features do contain edgelet features.
    /*
    Vector2f dir_cur(A_cur_ref_ * ref_ftr_->grad);
    dir_cur.normalize();
    success = align::align1D(pyrs_cur[search_level],
                             dir_cur,
                             patch_with_border_,
                             patch_,
                             options_.max_iter,
                             &px_scaled,
                             &h_inv_);
    */
    LOG(ERROR) << "findMatchDirect for edgelet feature is NOT rimplemented yet";
    success = false;
  } else {
#ifndef _ALIGN2D_TIMING_AND_VERIFYING
#ifndef __ARM_NEON__
    success = align::align2D(pyrs_cur[search_level],
                             patch_with_border_,
                             patch_,
                             options_.max_iter,
                             &px_scaled);
#else
    success = align::align2D_NEON(pyrs_cur[search_level],
                                  patch_with_border_,
                                  patch_,
                                  options_.max_iter,
                                  &px_scaled);
#endif  // __ARM_NEON__
#else
    // timing and verifying code.
#ifdef __ARM_NEON__
    Vector2f px_scaled_neon = px_scaled;
#endif  // __ARM_NEON__
    timer_align2d.start();
    success = align::align2D(pyrs_cur[search_level],
                             patch_with_border_,
                             patch_,
                             options_.max_iter,
                             &px_scaled);
    timer_align2d.stop();
    align2d_total_time = timer_align2d.stop();

#ifdef __ARM_NEON__
    timer_align2d_neon.start();
    bool success_neon = align::align2D_NEON(pyrs_cur[search_level],
                                            patch_with_border_,
                                            patch_,
                                            options_.max_iter,
                                            &px_scaled_neon);
    timer_align2d_neon.stop();
    align2d_total_time_neon = timer_align2d_neon.elapse_ms();
    CHECK_EQ(success, success_neon);
    if (success_neon) {
      CHECK_NEAR(px_scaled[0], px_scaled_neon[0], 0.05);
      CHECK_NEAR(px_scaled[1], px_scaled_neon[1], 0.05);
      std::cout << "[NEON : NON-NEON --->" << "[" << align2d_total_time_neon
                << " : " << align2d_total_time << "]" << std::endl;
    }
#endif  // __ARM_NEON__
#endif  // _ALIGN2D_TIMING_AND_VERIFYING
  }
  *px_cur = px_scaled * (1 << search_level);
  return success;
}


//这里用的evo的代码,evo原来是用来做求得初始相对位姿后细化位姿用参考帧和当前帧的块匹配以得到更精准的特征点,现在立体相机左右外参已知，ref->left相机,cur->right相机
//如果极线短就不用搜索,直接优化图片块和左目warp过来的图像的强度误差,如果需要搜索,就用ZMSSD评判出一个最相似的patch，然后再优化最终的右目特征点像素坐标
bool DirectMatcher::findEpipolarMatchDirect(const vio::cameras::CameraBase& cam_ref,//左相机类
                                            const vio::cameras::CameraBase& cam_cur,//右相机类
                                            const Vector2f& px_ref,//左相机特征点2d坐标
                                            const Vector3f& f_ref,//左相机特征点的归一话坐标
                                            const Matrix3f& R_cur_ref,//外参
                                            const Vector3f& t_cur_ref,//外参
                                            const int level_ref,//左特征点所在的金字塔层
                                            const float d_estimate,//深度估计的初值(左目的)
                                            const float d_min,//深度值范围最小值
                                            const float d_max,//深度值范围最大值
                                            const std::vector<cv::Mat>& pyrs_ref,//左图像金字塔层所有的图片
                                            const std::vector<cv::Mat>& pyrs_cur,//右图像金字塔层所有的图片
                                            const cv::Mat_<uchar>& mask_cur,//右相机掩码
                                            const bool edgelet_feature,
                                            Vector2f* px_cur,//右相机的特征点
                                            float* depth,//左相机特征点深度
                                            int* level_cur,
                                            cv::Mat* dbg_cur) {
  CHECK_NEAR(f_ref[2], 1.f, 1e-6);
  CHECK_EQ(pyrs_ref.size(), pyrs_cur.size());

  // Compute start and end of epipolar line in old_kf for match search, on unit plane!
  // i.e., A & B are the first two elements of unit rays.
  // We will search from far to near
  Vector3f ray_A, ray_B;//极线的起点和终点在相机坐标系中
  Vector2f px_A, px_B;//极线的起点和终点在图像坐标系中
  ray_B = R_cur_ref * (f_ref * d_max) + t_cur_ref;  //最大深度对应的极线端点
  ray_B /= ray_B(2);
  if (vio::cameras::CameraBase::ProjectionStatus::Successful !=
      cam_cur.project(ray_B, &px_B)) {
#ifndef __FEATURE_UTILS_NO_DEBUG__
    VLOG(2) << "ray_A (far) cannot be reprojected in cam_cur";
#endif
    return false;
  }

  bool invalid_ray_A = true;
  for (float d = d_min; d < d_estimate; d *= 10) {
    ray_A = R_cur_ref * (f_ref * d) + t_cur_ref;  // near
    ray_A /= ray_A(2);
    if (vio::cameras::CameraBase::ProjectionStatus::Successful ==
        cam_cur.project(ray_A, &px_A)) {// 最小深度向上采样最近的有效深度对应的极线端点
      invalid_ray_A = false;
      break;
    }
  }
  if (invalid_ray_A) {
#ifndef __FEATURE_UTILS_NO_DEBUG__
    VLOG(2) << "ray_B (near) cannot be reprojected in cam_cur "
            << " d_min = " << d_min
            << " d_estimate = " << d_estimate;
#endif
    return false;
  }

  // Compute warp affine matrix
  // 计算初始的仿射变化矩阵 A_r_l_2×2
       //输入左右相机类,左相机像素坐标,归一化坐标,估计的深度,特征点所在的金字塔层,外参
       //输出仿射矩阵
  if (!warp::getWarpMatrixAffine(cam_ref,
                                 cam_cur,
                                 px_ref,
                                 f_ref,
                                 d_estimate,
                                 R_cur_ref,
                                 t_cur_ref,
                                 level_ref,
                                 &A_cur_ref_)) {
    LOG(WARNING) << "warp::getWarpMatrixAffine fails";
    return false;
  }
#ifndef __FEATURE_UTILS_NO_DEBUG__
  VLOG(2) << "A_cur_ref_ =\n" << A_cur_ref_;
#endif

  const int max_level = pyrs_ref.size() - 1;
  //找到在右相机上最佳的搜索金字塔层
  const int search_level = warp::getBestSearchLevel(A_cur_ref_, max_level);
  epi_dir_ = ray_A.head<2>() - ray_B.head<2>();  // far to near, B to A极线段向量
  epi_length_ = (px_A - px_B).norm() / (1 << search_level);

  // feature pre-selection//暂时没用到这个
  if (edgelet_feature) {
    /*
    const Vector2f grad_cur = (A_cur_ref_ * ref_ftr.grad).normalized();
    const float cosangle = fabs(grad_cur.dot(epi_dir_.normalized()));
    if (cosangle < options_.epi_search_edgelet_max_angle) {
      return false;
    }
    */
    LOG(ERROR) << "findEpipolarMatchDirect for edgelet feature is NOT implemented yet";
    return false;
  }



        //将左相机图像特征点中心的图像块warp到右相机图像坐标系中
        //输入之前得到的粗略的仿射矩阵,左相机特征点所在所在金字塔图像,左特征点像素坐标,左特征的金字塔层，右相机需要搜索的金字塔层,
  if (!warp::warpAffine(A_cur_ref_,
                        pyrs_ref[level_ref],
                        px_ref,
                        level_ref,
                        search_level,
                        halfpatch_size_ + 1,
                        patch_with_border_)) {
    return false;
  }
  //patch_with_border 生成 patch
  createPatchFromPatchWithBorder();
#ifndef __FEATURE_UTILS_NO_DEBUG__
  VLOG(2) << " search_level = " << search_level << " epi_length_ = " << epi_length_;
#endif
  //如果基线足够小,那么不同再基线搜索了,直接图片对齐
  if (epi_length_ < options_.max_epi_length_optim)
  {
    // The epipolar search line is short enough (< 2 pixels)
    // to perform direct alignment
    *px_cur = (px_A + px_B) * 0.5f; // 取平均值
    Vector2f px_scaled(*px_cur / (1 << search_level));//变换到对应的最佳金字塔层
    bool success;
    if (options_.align_1d) {
      Vector2f direction = (px_A - px_B).normalized();
      success = align::align1D(pyrs_cur[search_level],
                               direction,
                               patch_with_border_,
                               patch_,
                               options_.max_iter,
                               &px_scaled,
                               &h_inv_);
    } else {
#ifndef _ALIGN2D_TIMING_AND_VERIFYING
#ifndef __ARM_NEON__
        //默认是这个
        //输入右相机的图像,从左相机变换到右相机上的patch_border,从左相机变换到右相机上的patch,最大迭代次数
        //输出最终块匹配残差最小的右目中特征点的像素坐标
      success = align::align2D(pyrs_cur[search_level],
                               patch_with_border_,
                               patch_,
                               options_.max_iter,
                               &px_scaled);
#else
      success = align::align2D_NEON(pyrs_cur[search_level],
                                    patch_with_border_,
                                    patch_,
                                    options_.max_iter,
                                    &px_scaled);
#endif  // __ARM_NEON__
#else
      // verifying and timing code
#ifdef __ARM_NEON__
      Vector2f px_scaled_neon = px_scaled;
#endif  // __ARM_NEON__
      timer_align2d.start();
      success = align::align2D(pyrs_cur[search_level],
                               patch_with_border_,
                               patch_,
                               options_.max_iter,
                               &px_scaled);
      timer_align2d.stop();
      align2d_total_time = timer_align2d.elapse_ms();
#ifdef __ARM_NEON__
      timer_align2d_neon.start();
      bool success_neon = align::align2D_NEON(pyrs_cur[search_level],
                                              patch_with_border_,
                                              patch_,
                                              options_.max_iter,
                                              &px_scaled_neon);
      timer_align2d_neon.stop();
      align2d_total_time_neon = timer_align2d_neon.elapse_ms();
      CHECK_EQ(success, success_neon);
      if (success_neon) {
        CHECK_NEAR(px_scaled[0], px_scaled_neon[0], 0.05);
        CHECK_NEAR(px_scaled[1], px_scaled_neon[1], 0.05);
        std::cout << "[NEON : NON-NEON --->" << "[" << align2d_total_time_neon
                  << " : " << align2d_total_time << "]" << std::endl;
      }
#endif  // __ARM_NEON__
#endif  // _ALIGN2D_TIMING_AND_VERIFYING
    }

    //如果迭代收敛的话
    if (success)
    {
      *px_cur = px_scaled * (1 << search_level);//最终这个特征点在右目中的位置
      Vector3f f_cur;
      if (cam_cur.backProject(*px_cur, &f_cur))
      {

        CHECK_NEAR(f_cur[2], 1.f, 1e-6);
        //输入外参,左相机这个特征点的归一化坐标,右相机归一化坐标
        //构造Ax=b,得到左目坐标系下特征点的深度
        if (!depthFromTriangulation(R_cur_ref,
                                    t_cur_ref,
                                    f_ref,
                                    f_cur,
                                    depth)) {
          LOG(WARNING) << "depthFromTriangulation fails, set depth to d_max";
          *depth = d_max;//求解失败就给最大深度
        }
      }
    }

    //如果需要debug,就把点圈出来
    if (dbg_cur != nullptr) {
      if (success) {
        // green: subpix alignment is good
        cv::circle(*dbg_cur, cv::Point2f((*px_cur)(0), (*px_cur)(1)), 2, cv::Scalar(0, 255, 0));
      } else {
        // red: subpix alignment fails
        cv::circle(*dbg_cur, cv::Point2f((*px_cur)(0), (*px_cur)(1)), 2, cv::Scalar(0, 0, 255));
      }
    }
    return success;
  }

  //极线不够近,所以极线搜索
  // Determine the steps to Search along the epipolar line
  // [NOTE] The epipolar line can be curvy, so we slightly increase it
  //        to roughly have one step per pixel (heuristically).
  size_t n_steps = epi_length_ / 0.7;//搜索步数
  Vector3f step;
  step << epi_dir_ / n_steps, 0;//步长
  if (n_steps > options_.max_epi_search_steps) {//需要的步数太多,那就不搜索了
    LOG(ERROR) << "Skip epipolar search: evaluations = " << n_steps
               << "epi length (px) = " << epi_length_;
    return false;
  }

  // Search along the epipolar line (on unit plane) with patch matching
  // for matching, precompute sum and sum2 of warped reference patch
  // [heuristic] The ssd from patch mean difference can be up to 50% of the resulting ssd.
  //  ssd = zmssd + N * (a_bar - b_bar)^2
  typedef patch_score::ZMSSD<halfpatch_size_> PatchScore;
  PatchScore patch_score(patch_);//左相机warp到右相机以后的patch
  int zmssd_best = PatchScore::threshold();
  int ssd_corr = PatchScore::threshold() * 2;
  Vector3f ray_best;
  Vector3f ray = ray_B;//从极线远段往近端搜索
  Eigen::Vector2i last_checked_pxi(0, 0);
  const int search_img_rows = pyrs_cur[search_level].rows;
  const int search_img_cols = pyrs_cur[search_level].cols;
  ++n_steps;

  for (size_t i = 0; i < n_steps; ++i, ray += step)
  {
    Vector2f px;
    //投影到右目像素坐标系,并且转换到对应金字塔层
    if (vio::cameras::CameraBase::ProjectionStatus::Successful != cam_cur.project(ray, &px)) {
      // We have already checked the valid projection of starting and ending rays.  However,
      // under very rare circumstance, cam_cur.project may still fail:
      // close to zero denominator for radial tangential 8 distortion:
      // 1 + k4 * r^2 + k5 * r^4 + k6 * r^6 < 1e-6
      continue;
    }
    Vector2i pxi(px[0] / (1 << search_level) + 0.5,
                 px[1] / (1 << search_level) + 0.5);  // round to closest int

      // 和上一个相同则下一个
    if (pxi == last_checked_pxi) {
      continue;
    }
    last_checked_pxi = pxi;
// 检查patch是否都在其内
    // check if the patch is full within the new frame
    if (pxi[0] >= halfpatch_size_ && pxi[0] < search_img_cols - halfpatch_size_ &&
        pxi[1] >= halfpatch_size_ && pxi[1] < search_img_rows - halfpatch_size_ &&
        mask_cur(pxi(1), pxi(0)) > 0)
    {
      // TODO(mingyu): Interpolation instead?
      //得到右目patch的头指针
      uint8_t* cur_patch_ptr = pyrs_cur[search_level].data
          + (pxi[1] - halfpatch_size_) * search_img_cols
          + (pxi[0] - halfpatch_size_);
      int ssd, zmssd;
// 计算极线上的patch与ref仿射变换得到的patch_之间的ZMSSD得分
//这个我没具体关注这个评分怎么算的,应该和ZNCC,SSD差不多吧,都是强度相似度
      patch_score.computeScore(cur_patch_ptr, search_img_cols, &zmssd, &ssd);
      if (zmssd < zmssd_best) {
        // We store the best zmssd and its corresponding ssd score.  Usually,
        // zmssd and ssd have good correlation if the *matching* is reasonable.
        //保存下最佳的评分以及对应的点
        zmssd_best = zmssd;
        ssd_corr = ssd;
        ray_best = ray;
      }
#ifndef __FEATURE_UTILS_NO_DEBUG__
      VLOG(2) << "search pxi=[" << pxi[0] << ", " << pxi[1] << "] zmssd = " << zmssd
              << " ssd = " << ssd;
#endif
      if (dbg_cur != nullptr) {
        dbg_cur->at<cv::Vec3b>(pxi[1], pxi[0]) = cv::Vec3b(255, 0, 0);
      }
    } else {
      // The patch contains out of bound pixels
      continue;
    }
  }

  VLOG(2) << "zmssd_best = " << zmssd_best << " ssd_corr = " << ssd_corr
          << " zmssd / ssd = " << static_cast<float>(zmssd_best) / ssd_corr;
  //如果得分小于阈值,说明可以接受,再进行优化,找到更准的右目中的这个点的像素坐标,同上
  if (zmssd_best < PatchScore::threshold())
  {
    cam_cur.project(ray_best, px_cur);
    if (options_.subpix_refinement) {
      Vector2f px_scaled(*px_cur / (1 << search_level));
      bool success;
      if (options_.align_1d) {
        Vector2f direction = (px_A - px_B).normalized();
        success = align::align1D(pyrs_cur[search_level],
                                 direction,
                                 patch_with_border_,
                                 patch_,
                                 options_.max_iter,
                                 &px_scaled,
                                 &h_inv_);
      } else {
#ifndef _ALIGN2D_TIMING_AND_VERIFYING
#ifndef __ARM_NEON__
        success = align::align2D(pyrs_cur[search_level],
                                 patch_with_border_,
                                 patch_,
                                 options_.max_iter,
                                 &px_scaled);
#else
        success = align::align2D_NEON(pyrs_cur[search_level],
                              patch_with_border_,
                              patch_,
                              options_.max_iter,
                              &px_scaled);
#endif  // __ARM_NEON__
#else
        #ifdef __ARM_NEON__
        Vector2f px_scaled_neon = px_scaled;
#endif  // __ARM_NEON__
        timer_align2d.start();
        success = align::align2D(pyrs_cur[search_level],
                             patch_with_border_,
                             patch_,
                             options_.max_iter,
                             &px_scaled);
        timer_align2d.stop();
        align2d_total_time = timer_align2d.elapse_ms();
#ifdef __ARM_NEON__
        timer_align2d_neon.start();
        bool success_neon = align::align2D_NEON(pyrs_cur[search_level],
                                            patch_with_border_,
                                            patch_,
                                            options_.max_iter,
                                            &px_scaled_neon);
        timer_align2d_neon.stop();
        align2d_total_time_neon = timer_align2d_neon.elapse_ms();
        CHECK_EQ(success, success_neon);
        if (success_neon) {
          CHECK_NEAR(px_scaled[0], px_scaled_neon[0], 0.05);
          CHECK_NEAR(px_scaled[1], px_scaled_neon[1], 0.05);
          std::cout << "[NEON : NON-NEON --->" << "[" << align2d_total_time_neon
                  << " : " << align2d_total_time << "]" << std::endl;
        }
#endif  // __ARM_NEON__
#endif  // _ALIGN2D_TIMING_AND_VERIFYING
      }

      if (success) {
        *px_cur = px_scaled * (1 << search_level);
        Vector3f f_cur;
        if (cam_cur.backProject(*px_cur, &f_cur)) {
          CHECK_NEAR(f_cur[2], 1.f, 1e-6);
          if (!depthFromTriangulation(R_cur_ref,
                                      t_cur_ref,
                                      f_ref,
                                      f_cur,
                                      depth)) {
            LOG(WARNING) << "depthFromTriangulation fails, set depth to d_max";
            *depth = d_max;
          }
        }
      }

      if (dbg_cur != nullptr) {
        if (success) {
          // green: subpix alignment is good
          cv::circle(*dbg_cur, cv::Point2f((*px_cur)(0), (*px_cur)(1)), 2, cv::Scalar(0, 255, 0));
        } else {
          // red: subpix alignment fails
          cv::circle(*dbg_cur, cv::Point2f((*px_cur)(0), (*px_cur)(1)), 2, cv::Scalar(0, 0, 255));
        }
      }
      return success;
    } else {
      // No subpix refinement
      CHECK_NEAR(ray_best[2], 1.f, 1e-6);
      if (!depthFromTriangulation(R_cur_ref,
                                  t_cur_ref,
                                  f_ref,
                                  ray_best,
                                  depth)) {
        LOG(WARNING) << "depthFromTriangulation fails, set depth to d_max";
        *depth = d_max;
      }
      return true;
    }
  }

  // No patch qualifiess acc match
#ifndef __FEATURE_UTILS_NO_DEBUG__
  VLOG(1) << "No matching patch found for this feature";
#endif
  return false;
}

// Prepare all the shared variabls for the whole tracking pipeline
//图像特征处理器的构造函数
//输入右左相机内参,右左相机畸变,右相机的掩码,金字塔层数
ImgFeaturePropagator::ImgFeaturePropagator(
        const Eigen::Matrix3f& right_camK,
        const Eigen::Matrix3f& left_camK,
        const cv::Mat_<float>& right_cv_dist_coeff,
        const cv::Mat_<float>& left_cv_dist_coeff,
        const bool fisheye,
        const cv::Mat_<uchar>& right_mask,
        int feat_det_pyramid_level,
        float min_feature_distance_over_baseline_ratio,
        float max_feature_distance_over_baseline_ratio) :
        mask_right_(right_mask),
        min_feature_distance_over_baseline_ratio_(min_feature_distance_over_baseline_ratio),
        max_feature_distance_over_baseline_ratio_(max_feature_distance_over_baseline_ratio),
        feat_det_pyramid_level_(feat_det_pyramid_level)
    {
        CHECK_GT(mask_right_.rows, 0);
        CHECK_GT(mask_right_.cols, 0);

        if(fisheye)
        {
            cam_right_.reset(new vio::cameras::PinholeCamera<
                    vio::cameras::EquidistantDistortion>(
                    mask_right_.cols,
                    mask_right_.rows,
                    right_camK(0, 0),  // focalLength[0],
                    right_camK(1, 1),  // focalLength[1],
                    right_camK(0, 2),  // principalPoint[0],
                    right_camK(1, 2),  // principalPoint[1],
                    vio::cameras::EquidistantDistortion(
                            right_cv_dist_coeff(0),
                            right_cv_dist_coeff(1),
                            right_cv_dist_coeff(2),
                            right_cv_dist_coeff(3))));
        }
        else if (right_cv_dist_coeff.rows == 8)//初始化右相机的针孔投影、rantan畸变模型
        {
            cam_right_.reset(new vio::cameras::PinholeCamera<
                    vio::cameras::RadialTangentialDistortion8>(
                    mask_right_.cols,
                    mask_right_.rows,
                    right_camK(0, 0),  // focalLength[0], fu
                    right_camK(1, 1),  // focalLength[1], fv
                    right_camK(0, 2),  // principalPoint[0], cx
                    right_camK(1, 2),  // principalPoint[1], cy
                    vio::cameras::RadialTangentialDistortion8(//目前只有randtan模型
                            right_cv_dist_coeff(0),
                            right_cv_dist_coeff(1),
                            right_cv_dist_coeff(2),
                            right_cv_dist_coeff(3),
                            right_cv_dist_coeff(4),
                            right_cv_dist_coeff(5),
                            right_cv_dist_coeff(6),
                            right_cv_dist_coeff(7))));
        } else if (right_cv_dist_coeff.rows == 4) {
            cam_right_.reset(new vio::cameras::PinholeCamera<
                    vio::cameras::RadialTangentialDistortion>(
                    mask_right_.cols,
                    mask_right_.rows,
                    right_camK(0, 0),  // focalLength[0],
                    right_camK(1, 1),  // focalLength[1],
                    right_camK(0, 2),  // principalPoint[0],
                    right_camK(1, 2),  // principalPoint[1],
                    vio::cameras::RadialTangentialDistortion(
                            right_cv_dist_coeff(0),
                            right_cv_dist_coeff(1),
                            right_cv_dist_coeff(2),
                            right_cv_dist_coeff(3))));
        } else {
            LOG(FATAL) << "Dist model unsupported for cam_right_";
        }
        cam_right_->setMask(right_mask);

        //初始化左相机的针孔投影、rantan畸变模型
        if(fisheye)
        {
            cam_left_.reset(new vio::cameras::PinholeCamera<
                    vio::cameras::EquidistantDistortion>(
                    mask_right_.cols,
                    mask_right_.rows,
                    left_camK(0, 0),  // focalLength[0],
                    left_camK(1, 1),  // focalLength[1],
                    left_camK(0, 2),  // principalPoint[0],
                    left_camK(1, 2),  // principalPoint[1],
                    vio::cameras::EquidistantDistortion(
                            left_cv_dist_coeff(0),
                            left_cv_dist_coeff(1),
                            left_cv_dist_coeff(2),
                            left_cv_dist_coeff(3))));
        }
        else if (left_cv_dist_coeff.rows == 8) {
            cam_left_.reset(new vio::cameras::PinholeCamera<
                    vio::cameras::RadialTangentialDistortion8>(
                    mask_right_.cols,
                    mask_right_.rows,
                    left_camK(0, 0),  // focalLength[0],
                    left_camK(1, 1),  // focalLength[1],
                    left_camK(0, 2),  // principalPoint[0],
                    left_camK(1, 2),  // principalPoint[1],
                    vio::cameras::RadialTangentialDistortion8(
                            left_cv_dist_coeff(0),
                            left_cv_dist_coeff(1),
                            left_cv_dist_coeff(2),
                            left_cv_dist_coeff(3),
                            left_cv_dist_coeff(4),
                            left_cv_dist_coeff(5),
                            left_cv_dist_coeff(6),
                            left_cv_dist_coeff(7))));
        } else if (left_cv_dist_coeff.rows == 4) {
            cam_left_.reset(new vio::cameras::PinholeCamera<
                    vio::cameras::RadialTangentialDistortion>(
                    mask_right_.cols,
                    mask_right_.rows,
                    left_camK(0, 0),  // focalLength[0],
                    left_camK(1, 1),  // focalLength[1],
                    left_camK(0, 2),  // principalPoint[0],
                    left_camK(1, 2),  // principalPoint[1],
                    vio::cameras::RadialTangentialDistortion(
                            left_cv_dist_coeff(0),
                            left_cv_dist_coeff(1),
                            left_cv_dist_coeff(2),
                            left_cv_dist_coeff(3))));
        } else {
            LOG(FATAL) << "Dist model unsupported for cam_left_";
        }

        cam_left_->setMask(right_mask);
    }
//输入右左目图片，左目提取的特征点,左右目外参,右目提取的特征点,描述子,是否要输出debug信息
bool ImgFeaturePropagator::PropagateFeatures(
    const cv::Mat& right_img,
    const cv::Mat& left_img,  // TODO(mingyu): store image pyramids
    const std::vector<cv::KeyPoint>& left_keypoints,
    const Matrix4f& T_l_r,
    std::vector<cv::KeyPoint>* right_keypoints,
    cv::Mat* right_orb_features,
    const bool draw_debug) {
  right_keypoints->clear();
  right_keypoints->reserve(left_keypoints.size());

  //算一下SE3的逆,分成R，t存储
    R_r_l_ = T_l_r.topLeftCorner<3, 3>().transpose();
    t_r_l_ = - R_r_l_ * T_l_r.topRightCorner<3, 1>();//if(false)
    if(cam_left_->distortionType() == "EquidistantDistortion" || cam_right_->distortionType() == "EquidistantDistortion")
    {
        std::vector<unsigned char> inlier_markers;

        cv::Matx33f Rrcam_lcam_cv;
        cv::Vec3f trcam_lcam_cv;

        cv::eigen2cv(R_r_l_,Rrcam_lcam_cv);//因为已经有了左右两目相机之间的外参
        cv::eigen2cv(t_r_l_,trcam_lcam_cv);//因为已经有了左右两目相机之间的外参

        std::vector<cv::Point2f> rcam_points_undistorted;
        std::vector<cv::Point2f> pre_matched_lpoint;
        std::vector<int> idxs;
        std::vector<cv::Point2f> pre_matched_rpoint;

        for (int i = 0; i < left_keypoints.size(); ++i)
        {
            Vector2f px_right, px_left(left_keypoints[i].pt.x, left_keypoints[i].pt.y);
            Vector3f f_left;//左相机的归一化坐标
            if (!cam_left_->backProject(px_left, &f_left))
            {
                continue;
            }
            Eigen::Vector3f rightCam_bearing_vec = R_r_l_ * f_left;//利用旋转进行一个坐标对齐
            Eigen::Vector2f right_cam_measure;
            vio::cameras::CameraBase::ProjectionStatus result = cam_right_->project(rightCam_bearing_vec,&right_cam_measure);
            if(result == vio::cameras::CameraBase::ProjectionStatus::Successful)
            {
                pre_matched_lpoint.push_back(left_keypoints[i].pt);
                idxs.push_back(i);
                pre_matched_rpoint.push_back(cv::Point2f(right_cam_measure[0],right_cam_measure[1]));
            }
        }
        cv::calcOpticalFlowPyrLK(left_img, right_img,
                                 pre_matched_lpoint, pre_matched_rpoint,inlier_markers, cv::noArray(),
                                 cv::Size(15, 15),3,cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,30,0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);

        for (int i = 0; i < pre_matched_rpoint.size(); ++i)
        {
            if (inlier_markers[i] == 0)
                continue;

            if (pre_matched_rpoint[i].y < 0 ||
                pre_matched_rpoint[i].y > right_img.rows-1 || pre_matched_rpoint[i].x < 0 ||
                pre_matched_rpoint[i].x > right_img.cols-1)
            {
                inlier_markers[i] = 0;
            }

        }
        // Compute the essential matrix.
        //用外参算出一个E矩阵
        const cv::Matx33f t_rcam_lcam_hat(
                0.0, -trcam_lcam_cv[2], trcam_lcam_cv[1],
                trcam_lcam_cv[2], 0.0, -trcam_lcam_cv[0],
                -trcam_lcam_cv[1], trcam_lcam_cv[0], 0.0);
        const cv::Matx33f E = t_rcam_lcam_hat * Rrcam_lcam_cv;


        Eigen::VectorXd lcam_intrinsics,rcam_intrinsics;
        cam_left_->getIntrinsics(&lcam_intrinsics);
        cam_right_->getIntrinsics(&rcam_intrinsics);
        double norm_pixel_unit = 4.0 / (lcam_intrinsics[0] + lcam_intrinsics[1] + rcam_intrinsics[0] + rcam_intrinsics[1]);

        CHECK_EQ(pre_matched_lpoint.size() , inlier_markers.size());
        CHECK_EQ(pre_matched_lpoint.size() , pre_matched_rpoint.size());

        for (int i = 0; i < pre_matched_lpoint.size(); ++i)
        {
            if (inlier_markers[i] == 0) continue;
            Eigen::Vector2f left_cam_measure{pre_matched_lpoint[i].x,pre_matched_lpoint[i].y};
            Eigen::Vector3f leftCam_bearing_vec;
            if (!cam_left_->backProject(left_cam_measure, &leftCam_bearing_vec))
            {
                inlier_markers[i] = 0;
                continue;
            }

            Eigen::Vector2f right_cam_measure{pre_matched_rpoint[i].x,pre_matched_rpoint[i].y};
            Eigen::Vector3f rightCam_bearing_vec;
            if (!cam_right_->backProject(right_cam_measure, &rightCam_bearing_vec))
            {
                inlier_markers[i] = 0;
                continue;
            }
            cv::Vec3d pt_left(leftCam_bearing_vec[0],//左特征点的极线
                              leftCam_bearing_vec[1], 1.0);
            cv::Vec3d pt_right(rightCam_bearing_vec[0],
                               rightCam_bearing_vec[1], 1.0);
            cv::Vec3d epipolar_line = E * pt_left;
            double error = fabs((pt_right.t() * epipolar_line)[0]) / sqrt(
                    epipolar_line[0]*epipolar_line[0]+
                    epipolar_line[1]*epipolar_line[1]);
            double stereo_threshold = 5;
            if (error > stereo_threshold*norm_pixel_unit)
                inlier_markers[i] = 0;
        }


        //把不好的匹配都去掉
        for (int j = 0; j < inlier_markers.size(); ++j)
        {
            if(inlier_markers[j])
            {
                const int orb_desc_margin = 20;
                if (pre_matched_rpoint[j].x < orb_desc_margin || pre_matched_rpoint[j].x > mask_right_.cols - orb_desc_margin ||
                    pre_matched_rpoint[j].y < orb_desc_margin || pre_matched_rpoint[j].y > mask_right_.rows - orb_desc_margin) {
                    continue;
                }

                cv::KeyPoint cur_kp = left_keypoints[idxs[j]];//先保证一下class_id相同
//                    cur_kp.octave = 0;
                cur_kp.pt.x = pre_matched_rpoint[j].x;
                cur_kp.pt.y = pre_matched_rpoint[j].y;
                right_keypoints->push_back(cur_kp);

            }
        }

    } else
    {
        // TODO(mingyu): Make shared variables of DirectMatcher into member variables
        //               instead of passing as input arguments

        // Heuristically determine the d_max, d_min, and d_estimate based on the baseline,
        // d_min = baseline x 3
        // d_max = baseline x 3000
        // inv_d_estimate is the average of inv_d_min and inv_d_max
        const float baseline = t_r_l_.norm();//基线
        const float d_min = baseline * min_feature_distance_over_baseline_ratio_;
        const float d_max = baseline * max_feature_distance_over_baseline_ratio_;
        const float d_estimate = 2.f / (1.f / d_min + 1.f / d_max);

        // TODO(mingyu): feed the backend results back for d_estimate if available
        // TODO(mingyu): feed the pyramids in directly
        std::vector<cv::Mat> pyrs_right(feat_det_pyramid_level_);
        std::vector<cv::Mat> pyrs_left(feat_det_pyramid_level_);
        pyrs_right[0] = right_img;
        pyrs_left[0] = left_img;
        //降采样采样
        for (int pyr_lv = 1; pyr_lv < feat_det_pyramid_level_; ++pyr_lv) {
            pyrs_right[pyr_lv] = fast_pyra_down(pyrs_right[pyr_lv - 1]);
            pyrs_left[pyr_lv] = fast_pyra_down(pyrs_left[pyr_lv - 1]);
        }

        //遍历左相机提取的特征点
        int totol_count = 0;
        int valid_count = 0;

        for (const cv::KeyPoint& left_kp : left_keypoints)
        {
            totol_count ++ ;
            int level_right = 0;
            float depth = -1.f;
            Vector2f px_right, px_left(left_kp.pt.x, left_kp.pt.y);
            Vector3f f_left;//左相机的归一化坐标
            if (!cam_left_->backProject(px_left, &f_left)) {
                continue;
            }
            valid_count ++ ;

            // Visualization (re-draw for every keypoint)
            //debug,
            if (draw_debug) {
                dbg_img_.create(mask_right_.rows * 2, mask_right_.cols, CV_8UC3);
                dbg_left_ = dbg_img_(cv::Rect(0, 0, mask_right_.cols, mask_right_.rows));
                dbg_right_ = dbg_img_(cv::Rect(0, mask_right_.rows, mask_right_.cols, mask_right_.rows));
                dbg_right_ptr_ = &dbg_right_;
                cv::cvtColor(left_img, dbg_left_, CV_GRAY2RGB);
                cv::cvtColor(right_img, dbg_right_, CV_GRAY2RGB);
                cv::circle(dbg_left_, left_kp.pt, 2, cv::Scalar(0, 255, 0));
                cv::putText(dbg_left_, boost::lexical_cast<std::string>(left_kp.class_id),
                            left_kp.pt, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                for (int i = 0; i < mask_right_.rows; ++i) {
                    for (int j = 0; j < mask_right_.cols; ++j) {
                        if (mask_right_(i, j) == 0x00) {
                            dbg_right_.at<cv::Vec3b>(i, j)[0] = 0;
                            dbg_right_.at<cv::Vec3b>(i, j)[1] = 0;
                        }
                    }
                }
            }
#ifndef __FEATURE_UTILS_NO_DEBUG__
            VLOG(1) << "findEpipolarMatchDirect for of_id = " << left_kp.class_id;
#endif
//这里用的evo的代码,evo原来是用来做求得初始相对位姿后细化位姿用参考帧和当前帧的块匹配以得到更精准的特征点,现在立体相机左右外参已知，ref->left相机,cur->right相机
//如果极线短就不用搜索,直接优化图片块和左目warp过来的图像的强度误差,如果需要搜索,就用ZMSSD评判出一个最相似的patch，然后再优化最终的右目特征点像素坐标
            if (!direct_matcher_.findEpipolarMatchDirect(*cam_left_,//左相机类
                                                         *cam_right_,//右相机类
                                                         px_left,//左相机特征点2d坐标
                                                         f_left,//左相机特征点的归一话坐标
                                                         R_r_l_,//外参
                                                         t_r_l_,//外参
                                                         left_kp.octave,  // double check here左特征点所在的金字塔层
                                                         d_estimate,//深度估计的初值
                                                         d_min,//深度最小值
                                                         d_max,//深度最大值
                                                         pyrs_left,//左图像金字塔层所有的图片
                                                         pyrs_right,//右图像金字塔层所有的图片
                                                         mask_right_,//右相机掩码
                                                         false, /*edgelet_feature, not supported yet*/
                                                         &px_right,//右相机的特征点
                                                         &depth,//左相机特征点深度
                                                         &level_right,//右目中特征点所在的金字塔层
                                                         dbg_right_ptr_)) {
#ifndef __FEATURE_UTILS_NO_DEBUG__
                if (draw_debug) {
        cv::imwrite("/tmp/per_feat_dbg/det_" + boost::lexical_cast<std::string>(left_kp.class_id)
                    + ".png", dbg_img_);
      }
      VLOG(1) << "findEpipolarMatchDirect fails for of_id: " << left_kp.class_id;
#endif
                continue;
            }
#ifndef __FEATURE_UTILS_NO_DEBUG__
            if (draw_debug) {
      cv::imwrite("/tmp/per_feat_dbg/det_" + boost::lexical_cast<std::string>(left_kp.class_id)
                  + ".png", dbg_img_);
      VLOG(1) << "findEpipolarMatchDirect succeeds for of_id: " << left_kp.class_id;
    }
#endif

            // check boundary condition (set to 20 pixels for computing ORB features)
            // [NOTE] Returned px_right should be within mask_right_ already
            const int orb_desc_margin = 20;
            if (px_right(0) < orb_desc_margin || px_right(0) > mask_right_.cols - orb_desc_margin ||
                px_right(1) < orb_desc_margin || px_right(1) > mask_right_.rows - orb_desc_margin) {
                continue;
            }

            cv::KeyPoint cur_kp = left_kp;//先保证一下class_id相同
            cur_kp.octave = level_right;
            cur_kp.pt.x = px_right(0);
            cur_kp.pt.y = px_right(1);
            right_keypoints->push_back(cur_kp);
        }
    }
//  std::cout<<valid_count<<" "<<totol_count<<std::endl;
  //计算右目特征点的描述子
  if (right_orb_features != nullptr && right_keypoints->size() > 0) {
    right_orb_features->create(right_keypoints->size(), 32, CV_8U);
#ifdef __ARM_NEON__
    ORBextractor::computeDescriptorsN512(right_img, *right_keypoints, right_orb_features);
#else
    ORBextractor::computeDescriptors(right_img, *right_keypoints, right_orb_features);
#endif
  }
  return true;
}

}  // namespace XP
