/**
* This file is part of ORB-SLAM.
*
* Copyright (C) 2014 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <http://webdiis.unizar.es/~raulmur/orbslam/>
*
* ORB-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBEXTRACTOR_H_    // NOLINT
#define ORBEXTRACTOR_H_    // NOLINT

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <list>
#include <chrono>
using std::vector;
using cv::KeyPoint;


/**
 * @brief 提取器节点
 * @details 用于在特征点的分配过程中。
 *
 */
class ExtractorNode
{
public:
    /** @brief 构造函数 */
    ExtractorNode():bNoMore(false){}

    /**
     * @brief 在八叉树分配特征点的过程中，实现一个节点分裂为4个节点的操作
     *
     * @param[out] n1   分裂的节点1
     * @param[out] n2   分裂的节点2
     * @param[out] n3   分裂的节点3
     * @param[out] n4   分裂的节点4
     */
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    ///保存有当前节点的特征点
    std::vector<cv::KeyPoint> vKeys;
    ///当前节点所对应的图像坐标边界
    cv::Point2i UL, UR, BL, BR;
    //存储提取器节点的列表（其实就是双向链表）的一个迭代器,可以参考[http://www.runoob.com/cplusplus/cpp-overloading.html]
    //这个迭代器提供了访问总节点列表的方式，需要结合cpp文件进行分析
    std::list<ExtractorNode>::iterator lit;

    ///如果节点中只有一个特征点的话，说明这个节点不能够再进行分裂了，这个标志置位
    //如果你要问这个节点中如果没有特征点的话怎么办，我只想说那样的话这个节点就直接被删除了
    bool bNoMore;
};





class ORBextractor {
 public:
  enum {HARRIS_SCORE = 0, FAST_SCORE = 1};

  struct SinCosAngleVal {
    float sin_val;
    float cos_val;
  };

  // Utility functions to help get access to sin_cos_0_360_deg_look_up table
  static SinCosAngleVal get_sin_cos_0_360_deg_look_up(int i);
  static float travel_sin_cos_lookup_table(const std::vector<int>& angle);

  /*
   * Note: computeDescriptorsN512 and computeDescriptors are not exactly the same due
   * to the cvRound Function. But the difference is at most acc couple of bits.
   */
  static void computeDescriptorsN512(
      const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, cv::Mat* descriptors);

  static void computeDescriptors(
      const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, cv::Mat* descriptors);

  static void HarrisResponses(
      const cv::Mat& img, int blockSize, float harris_k, vector<KeyPoint>* pts);
  static void HarrisResponses_original(
      const cv::Mat& img, int blockSize, float harris_k, vector<KeyPoint>* pts);

#ifdef __ARM_NEON__
  // this function only handles block size 7 & 8
    static inline void accumulate_matrix_coeff(const uint8x16_t* top_row,
                                 const uint8x16_t* mid_row,
                                 const uint8x16_t* bot_row,
                                 const int block_size,
                                 int32x4_t* acc,
                                 int32x4_t* b,
                                 int32x4_t* c) {
      union {
        int16x8_t ix;
        int16_t ixc[8];
      } __attribute__((aligned(16)));
      union {
        int16x8_t iy;
        int16_t iyc[8];
      } __attribute__((aligned(16)));
      int16x8_t diff0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(*bot_row),
                                                       vget_low_u8(*top_row)));
      int16x8_t diff1 = vreinterpretq_s16_u16(
                      vsubl_u8(vget_low_u8(vextq_u8(*bot_row, vdupq_n_u8(0), 1)),
                      vget_low_u8(vextq_u8(*top_row, vdupq_n_u8(0), 1))));
      int16x8_t diff2 = vreinterpretq_s16_u16(
                      vsubl_u8(vget_low_u8(vextq_u8(*bot_row, vdupq_n_u8(0), 2)),
                      vget_low_u8(vextq_u8(*top_row, vdupq_n_u8(0), 2))));

      iy = vaddq_s16(vaddq_s16(diff0, diff2), vmulq_n_s16(diff1, 2));

      diff0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vextq_u8(*top_row, vdupq_n_u8(0), 2)),
                               vget_low_u8(*top_row)));
      diff1 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vextq_u8(*mid_row, vdupq_n_u8(0), 2)),
                               vget_low_u8(*mid_row)));
      diff2 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(vextq_u8(*bot_row, vdupq_n_u8(0), 2)),
                               vget_low_u8(*bot_row)));

      ix = vaddq_s16(vmulq_n_s16(diff1, 2), vaddq_s16(diff0, diff2));
      // set the last lane to zero
      if (__builtin_expect(block_size == 7, 1)) {
        ixc[7] = 0;
        iyc[7] = 0;
      }
      *acc = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(ix), vget_low_s16(ix)),
                               vmull_s16(vget_high_s16(ix), vget_high_s16(ix))), *acc);
      *b = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(iy), vget_low_s16(iy)),
                               vmull_s16(vget_high_s16(iy), vget_high_s16(iy))), *b);
      *c = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(ix), vget_low_s16(iy)),
                               vmull_s16(vget_high_s16(ix), vget_high_s16(iy))), *c);
    }


    // [NOTE] This function does NOT check boundary condition when computing harris response!
    // [NOTE] This function only handles block size 7, 8 and 9.
    static void HarrisResponses_neon(const cv::Mat& img,
                                     int blockSize,
                                     float harris_k,
                                     vector<KeyPoint>* pts) {
      CV_Assert(img.type() == CV_8UC1 && blockSize * blockSize <= 2048);
      CV_Assert(blockSize == 7 || blockSize == 8 || blockSize == 9);
      uint8x16_t raw_buffer[3] __attribute__((aligned(16)));
      size_t ptidx, ptsize = pts->size();

      const uchar* ptr00 = img.ptr<uchar>();
      int step = static_cast<int>(img.step / img.elemSize1());
      int r = blockSize / 2;

      float scale = (1 << 2) * blockSize * 255.0f;
      scale = 1.0f / scale;
      float scale_sq_sq = scale * scale * scale * scale;

      uint8x16_t* row_top = &raw_buffer[0];
      uint8x16_t* row_mid = &raw_buffer[1];
      uint8x16_t* row_bot = &raw_buffer[2];
      uint8x16_t* tmp_ptr = NULL;

      for (ptidx = 0; ptidx < ptsize; ptidx++) {
        int acc = 0, b = 0, c = 0;
        int x0 = cvRound((*pts)[ptidx].pt.x - r);
        int y0 = cvRound((*pts)[ptidx].pt.y - r);
        const uchar* ptr0 = ptr00 + y0 * step + x0;
        *row_mid = vld1q_u8(ptr0 - step - 1);  // top row
        *row_bot = vld1q_u8(ptr0 - 1);  // middle row

        int32x4_t va __attribute__((aligned(16))) = vmovq_n_s32(0);
        int32x4_t vb __attribute__((aligned(16))) = vmovq_n_s32(0);
        int32x4_t vc __attribute__((aligned(16))) = vmovq_n_s32(0);
        for (int k = 0; k < blockSize; k++, ptr0 += step) {
          // shifting
          tmp_ptr = row_top;
          row_top = row_mid;
          row_mid = row_bot;
          row_bot = tmp_ptr;
          *row_bot = vld1q_u8(ptr0 + step - 1);  // bottom row
          accumulate_matrix_coeff(row_top, row_mid, row_bot, blockSize,
                                  &va, &vb, &vc);
          if (blockSize == 9) {
            const uchar* rtop_ptr = reinterpret_cast<const uchar*>(row_top);
            const uchar* rmid_ptr = reinterpret_cast<const uchar*>(row_mid);
            const uchar* rbot_ptr = reinterpret_cast<const uchar*>(row_bot);
            int ix = (rmid_ptr[10] - rmid_ptr[8]) * 2 +
                     (rtop_ptr[10] - rtop_ptr[8]) +
                     (rbot_ptr[10] - rbot_ptr[8]);
            int iy = (rbot_ptr[9] - rtop_ptr[9]) * 2 +
                     (rbot_ptr[8] - rtop_ptr[8]) +
                     (rbot_ptr[10] - rtop_ptr[10]);
            acc += ix * ix;
            b += iy * iy;
            c += ix * iy;
          }
        }
        acc += (vgetq_lane_s32(va, 0) + vgetq_lane_s32(va, 1) +
              vgetq_lane_s32(va, 2) + vgetq_lane_s32(va, 3));
        b += (vgetq_lane_s32(vb, 0) + vgetq_lane_s32(vb, 1) +
              vgetq_lane_s32(vb, 2) + vgetq_lane_s32(vb, 3));
        c += (vgetq_lane_s32(vc, 0) + vgetq_lane_s32(vc, 1) +
              vgetq_lane_s32(vc, 2) + vgetq_lane_s32(vc, 3));
        (*pts)[ptidx].response = (static_cast<float>(acc) * b - static_cast<float>(c) * c -
          harris_k * (static_cast<float>(acc) + b) * (static_cast<float>(acc) + b)) * scale_sq_sq;
      }
    }
#endif

  static void computeOrientation(const cv::Mat& image,
                                 const std::vector<int>& umax,
                                 vector<KeyPoint>* keypoints);
  ORBextractor(int nfeatures = 1000,
               float scaleFactor = 1.2f,
               int nlevels = 8,
               int scoreType = FAST_SCORE,
               int fastTh = 20,
               bool use_fast = true);  // or use TomasShi

  ~ORBextractor() {}

    void detectgf_compute(const cv::Mat& image,
                                        const cv::Mat& mask,
                                        std::vector<cv::KeyPoint>* keypoints);

  // Compute the ORB features and descriptors on an image
  void detect(const cv::Mat& image,
              const cv::Mat& mask,
              std::vector<cv::KeyPoint>* keypoints,
              cv::Mat* descriptors = nullptr);

  int inline GetLevels() {
      return nlevels;
  }

  float inline GetScaleFactor() {
      return scaleFactor;
  }

 protected:


    /**
     * @brief 以八叉树分配特征点的方式，计算图像金字塔中的特征点
     * @detials 这里两层vector的意思是，第一层存储的是某张图片中的所有特征点，而第二层则是存储图像金字塔中所有图像的vectors of keypoints
     * @param[out] allKeypoints 提取得到的所有特征点
     */
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >* allKeypoints);

    /**
     * @brief 对于某一图层，分配其特征点，通过八叉树的方式
     * @param[in] vToDistributeKeys         等待分配的特征点
     * @param[in] minX                      分发的图像范围
     * @param[in] maxX                      分发的图像范围
     * @param[in] minY                      分发的图像范围
     * @param[in] maxY                      分发的图像范围
     * @param[in] nFeatures                 设定的、本图层中想要提取的特征点数目
     * @param[in] level                     要提取的图像所在的金字塔层
     * @return std::vector<cv::KeyPoint>
     */
    std::vector<cv::KeyPoint> DistributeOctTree(
            const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX, const int &maxX, const int &minY, const int &maxY,
            const int &nFeatures, const int &level);
  void ComputePyramid(const cv::Mat& image, cv::Mat Mask = cv::Mat());
  void ComputeKeyPoints(std::vector<std::vector<cv::KeyPoint> >* allKeypoints);

  int nfeatures;
  double scaleFactor;
  int nlevels;
  int scoreType;
  int fastTh;
  bool use_fast_;

  std::vector<int> mnFeaturesPerLevel;

  std::vector<int> umax;//计算质心时要用到

  std::vector<float> mvScaleFactor;
  std::vector<float> mvInvScaleFactor;

  std::vector<cv::Mat> mvImagePyramid;
  std::vector<cv::Mat> mvMaskPyramid;
};
#endif  // ORBEXTRACTOR_H_    // NOLINT

