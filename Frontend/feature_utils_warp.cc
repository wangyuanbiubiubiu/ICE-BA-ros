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
#include <Eigen/Dense>

namespace XP {
namespace warp {

using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Matrix2f;
using Eigen::Matrix3f;

// Compute affine warp matrix A_ref_cur
// The warping matrix is warping the ref patch (at level_ref) to the current frame (at pyr 0)
//输入左右相机类,左相机像素坐标,归一化坐标,估计的深度,特征点所在的金字塔层,外参
bool getWarpMatrixAffine(const vio::cameras::CameraBase& cam_ref,
                         const vio::cameras::CameraBase& cam_cur,
                         const Vector2f& px_ref,  // distorted pixel at pyr0
                         const Vector3f& f_ref,   // undist ray in unit plane
                         const float depth_ref,
                         const Matrix3f& R_cur_ref,
                         const Vector3f& t_cur_ref,
                         const int level_ref,
                         Eigen::Matrix2f* A_cur_ref) {
  CHECK_NOTNULL(A_cur_ref);
  // TODO(mingyu): tune the *d_unit* size in pixel for different 1st order approximation
  const int halfpatch_size = 5;
  const Vector3f xyz_ref(f_ref * depth_ref);// 特征点在左相机坐标中的位置
  //这个是一半块的大小,在不同的金子塔层patch的大小也是需要缩放的
  float d_unit = halfpatch_size * (1 << level_ref);
  //这里在算以px_ref为原点,uv的方向
  Vector2f du_ref(px_ref + Vector2f(d_unit, 0));
  Vector2f dv_ref(px_ref + Vector2f(0, d_unit));
  Vector3f xyz_du_ref, xyz_dv_ref;
  //反投影
  if (cam_ref.backProject(du_ref, &xyz_du_ref) && cam_ref.backProject(dv_ref, &xyz_dv_ref))
  {
    // Make sure the back project succeed for both du_ref & dv_ref
    //初始深度
    xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
    xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
    Vector2f px_cur, du_cur, dv_cur;
      // 利用外参把这三点变换到右相机坐标系下
    if (vio::cameras::CameraBase::ProjectionStatus::Successful ==
        cam_cur.project(R_cur_ref * xyz_ref + t_cur_ref, &px_cur) &&
        vio::cameras::CameraBase::ProjectionStatus::Successful ==
            cam_cur.project(R_cur_ref * xyz_du_ref + t_cur_ref, &du_cur) &&
        vio::cameras::CameraBase::ProjectionStatus::Successful ==
            cam_cur.project(R_cur_ref * xyz_dv_ref + t_cur_ref, &dv_cur)) {
        //如果都投影成功的话,计算仿射变换(每列就是某轴变换以后的方向)
      A_cur_ref->col(0) = (du_cur - px_cur) / halfpatch_size;
      A_cur_ref->col(1) = (dv_cur - px_cur) / halfpatch_size;
      return true;
    }
  }
  A_cur_ref->setIdentity();  // No warping
  return false;
}
// 找到合适金字塔层
// Compute patch level in other image (based on pyramid level 0)
int getBestSearchLevel(const Eigen::Matrix2f& A_cur_ref,
                       const int max_level) {
  int search_level = 0;
  float D = A_cur_ref.determinant();
  //行列式小于3为止
  while (D > 3.f && search_level < max_level) {
    ++search_level;
    D *= 0.25;
  }
  return search_level;
}

namespace {
// Return value between 0 and 255
// [NOTE] Does not check whether the x/y is within the border
inline float interpolateMat_8u(const cv::Mat& mat, float u, float v) {
  CHECK_EQ(mat.type(), CV_8U);
  int x = floor(u);
  int y = floor(v);
  float subpix_x = u - x;
  float subpix_y = v - y;

  float w00 = (1.0f - subpix_x) * (1.0f - subpix_y);
  float w01 = (1.0f - subpix_x) * subpix_y;
  float w10 = subpix_x * (1.0f - subpix_y);
  float w11 = 1.0f - w00 - w01 - w10;

  const int stride = mat.step.p[0];
  uint8_t* ptr = mat.data + y * stride + x;
  return w00 * ptr[0] + w01 * ptr[stride] + w10 * ptr[1] + w11 * ptr[stride+1];
}
}  // namespace

//将左相机图像特征点中心的图像块warp到右相机图像坐标系中
// Compute acc squared patch that is *warperd* from img_ref with A_cur_ref.
//输入之前得到的粗略的仿射矩阵,左相机特征点所在所在金字塔图像,左特征点像素坐标,左特征的金字塔层，右相机需要搜索的金字塔层,
bool warpAffine(const Eigen::Matrix2f& A_cur_ref,
                const cv::Mat& img_ref,         // at pyramid level_ref
                const Eigen::Vector2f& px_ref,  // at pyramid 0
                const int level_ref,
                const int level_cur,
                const int halfpatch_size,
                uint8_t* patch) {
  const int patch_size = halfpatch_size * 2;
  const Matrix2f A_ref_cur = A_cur_ref.inverse();
  if (std::isnan(A_ref_cur(0, 0))) {
    // TODO(mingyu): Use looser criteria for invalid affine warp?
    //               I suspect A_ref_cur can barely hit NaN.
    LOG(ERROR) << "Invalid affine warp matrix (NaN)";
    return false;
  }

  // px_ref is at pyr0, img_ref is at level_ref pyr already
  CHECK_NOTNULL(patch);
  uint8_t* patch_ptr = patch;
  const Vector2f px_ref_pyr = px_ref / (1<< level_ref);  // pixel at pyramid level_ref//变换到对应的金字塔层坐标上
  for (int y = 0; y < patch_size; ++y)
  {
    for (int x = 0; x < patch_size; ++x, ++patch_ptr)// // 以建立patch坐标系
    {
      Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
      px_patch *= (1 << level_cur);//缩放
      const Vector2f px(A_ref_cur * px_patch + px_ref_pyr);  // pixel at pyramid level_ref
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1) {
        *patch_ptr = 0;
      } else {
        *patch_ptr = interpolateMat_8u(img_ref, px[0], px[1]);//将左相机图像warp到右相机图像坐标系中
      }
    }
  }
  return true;
}

}  // namespace warp
}  // namespace XP
