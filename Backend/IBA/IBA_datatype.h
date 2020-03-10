/******************************************************************************
 * Copyright 2017-2018 Baidu Robotic Vision Authors. All Rights Reserved.
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
#ifndef _IBA_DATATYPE_H_
#define _IBA_DATATYPE_H_

#include <IBA_config.h>

#define IBA_SERIAL_NONE   0x00000000
#define IBA_SERIAL_LBA    0x00000001
#define IBA_SERIAL_GBA    0x00000100

#define IBA_VERBOSE_NONE  0x00000000
#define IBA_VERBOSE_LBA   0x00000001
#define IBA_VERBOSE_GBA   0x00000100

#define IBA_DEBUG_NONE    0x00000000
#define IBA_DEBUG_LBA     0x00000001
#define IBA_DEBUG_GBA     0x00000100

#define IBA_HISTORY_NONE  0x00000000
#define IBA_HISTORY_LBA   0x00000001
#define IBA_HISTORY_GBA   0x00000100

namespace IBA {

struct Intrinsic {
  float fx, fy;   // focal length
  float cx, cy;   // optic center
  float ds[8];    // distortion parameters
};


//用于保存成dat文件所用
struct Calibration {
  int w, h;       // image resolution
  bool fishEye;   // fish eye distortion model
  float Tu[3][4]; // X_cam = Tu * X_imu 存储的是Tc0_i
  float ba[3];    // initial acceleration bias
  float bw[3];    // initial gyroscope bias
  //float sa[3];
  Intrinsic K;    // intrinsic parameters
#ifdef CFG_STEREO
  float Tr[3][4]; // X_left = Tr * X_right  //Tc0_c1的外参
        Intrinsic Kr;   // intrinsic parameters for right camera //右相机的内参
#endif
};

//相机的位姿
struct CameraPose {
  float R[3][3];  //Rc0(观测关键帧)c0(参考关键帧) rotation matrix, R[0][0] = FLT_MAX for unknown camera pose
  float p[3];     //tc0(参考关键帧)c0(观测关键帧) position
};                // for acc 3D point in world frame X, its coordinate in camera frame is obtained by R * (X - p)

struct CameraPoseCovariance {
  float S[6][6];  // position + rotation
                  // p = \hat p + \tilde p
                  // R = \hat R * exp(\tilde\theta)
};

//位姿因子和imu相关的运动状态的因子
struct CameraIMUState {
  CameraPose Cam_pose;   //原始名称C camera pose
  float v[3];     // velocity, v[0] = FLT_MAX for unknown velocity
  float ba[3];    // acceleration bias, ba[0] = FLT_MAX for unknown acceleration bias
  float bw[3];    //gyroscope bias, bw[0] = FLT_MAX for unknown gyroscope bias
};
//逆深度,其中d = 0是未知的深度
struct Depth {
  float d;   // inverse depth, d = 0 for unknown depth
  float s2;  // variance
};

//像素坐标数据结构
struct Point2D {
  float x[2];     //像素坐标 feature location in the original image
  float S[2][2];  //协方差 covariance matrix in the original image
};

struct Point3D {
  int idx;    // 地图点全局id global point index
  float X[3]; // 位置3D position, X[0] = FLT_MAX for unknown 3D position
};

//地图点测量的数据结构
struct MapPointMeasurement {
  union {//注意,这里如果是作为一个普通帧的观测结构来用的话,即在CF.feat_measures中时,iFrm,idx都表示的是全局的地图点id
      //当被当作关键帧首次观测到的地图点的观测来用的时候,即在KF.Xs中的mp.feat_measures中时,iFrm,idx都表示的是这个关键帧的id

    int iFrm; //首次看到这个地图点的关键帧的id// frame ID
    int idx;  //全局地图点id,这个观测是观测的哪个地图点 global point ID
  };
  inline bool operator < (const MapPointMeasurement &X) const {
    return iFrm < X.iFrm
#ifdef CFG_STEREO
//#if 1
        || iFrm <= X.iFrm && !right && X.right
#endif
        ;
  }
  Point2D x;//特征点的像素坐标
#ifdef CFG_STEREO
//#if 1
  ubyte right;//是否是右目相机看到的地图点
#endif
};

struct MapPoint {
  Point3D X;//地图点在世界坐标系中的位置
  std::vector<MapPointMeasurement> feat_measures;//原始名称zs 这个地图点的所有观测,这里左目右目的观测是各算一个观测的,分开算的
};

struct FeatureTrack {
  int idx;
  Point2D x;
};
//imu测量
struct IMUMeasurement {
  float acc[3];     //原始名称c acceleration
  float gyr[3];     //原始名称w gyroscope
  float t;        // timestamp
};

struct CurrentFrame {
  int iFrm;//当前帧的id                             //frame index
  CameraIMUState Cam_state;                     //原始名称C initial camera/IMU state of current frame
  std::vector<MapPointMeasurement> feat_measures;  //原始名称zs 这里存储老地图点的观测,左右两目都观测到了地图点也是分开push进去的 feature measurements of current frame
  std::vector<IMUMeasurement> imu_measures;       //原始名称us IMU measurements between last frame and current frame;
                                        // the timestamp of first IMU must be the same as last frame
  float t;                              //当前帧的时间戳,用的左相机的时间戳 timestamp of current frame, should be greater than the timestamp of last IMU
  Depth d;                              // acc rough depth estimate for current frame
                                        // (e.g. average over all visible points in current frame)
  std::string fileName;                 // image file name, just for visualization
#ifdef CFG_STEREO
//#if 1
  std::string fileNameRight;
#endif
};

struct KeyFrame {
  int iFrm;                             // frame index, -1 for invalid keyframe(这帧不是关键帧的时候)
  CameraPose Cam_pose;                         //原始名称C initial camera pose of keyframe
  std::vector<MapPointMeasurement> feat_measures;  //原始名称zs feature measurements of keyframe 存储的是对于老地图点的观测
  std::vector<MapPoint> Xs;             // new map points 注意,这里的Xs只存储首次被这个关键帧看到的新的地图点,老的地图点它是不会存储的
  Depth d;                              // acc rough depth estimate
};
//GBA中更新了的地图点以及关键帧pose
struct Global_Map
{
    std::vector<int> iFrmsKF;         //关键帧中的帧id frame indexes of those keyframes whose camera pose
    std::vector<std::vector<int>>  CovisibleKFs;//更新的关键帧可能会有新的共视,所以这里也获取一下
    // has been updated since last call
    std::vector<CameraPose> CsKF;     //所有关键帧的pose camera poses corresponding to iFrmsKF
    std::vector<Point3D> Xs;          //上一次call之后更新的地图点,世界坐标系下 updated 3D points since last call

};

struct SlidingWindow {
  std::vector<int> iFrms;           //滑窗中帧id frame indexes of those sliding window frames whose
                                    // camera/IMU state has been updated since last call
  std::vector<CameraIMUState> CsLF; //滑窗中所有帧的状态 camera/IMU states corresponding to iFrms  //
  std::vector<int> iFrmsKF;         //关键帧中的帧id frame indexes of those keyframes whose camera pose
                                    // has been updated since last call
  std::vector<CameraPose> CsKF;     //所有关键帧的pose camera poses corresponding to iFrmsKF
  std::vector<Point3D> Xs;          //上一次call之后更新的地图点,世界坐标系下 updated 3D points since last call
#ifdef CFG_CHECK_REPROJECTION
  std::vector<std::pair<float, float> > esLF, esKF;
#endif
};

struct RelativeConstraint {
  int iFrm1, iFrm2;//参考关键帧id和观测关键帧id
  CameraPose T;//Tc0(观测关键帧)c0(参考关键帧)       // X2 = T * X1 = R * (X1 - p)
  CameraPoseCovariance S;
};

struct Error {
  float ex;                         // feature reprojection error
  float eur, eup, euv, euba, eubw;  // IMU delta error
  float edr, edp;                   // drift error compared to ground truth
};

struct Time {
  float t;
  int n;
};

}  // namespace IBA

#endif  //  _IBA_DATATYPE_H_
