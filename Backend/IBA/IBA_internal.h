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
#ifndef _IBA_INTERNAL_H_
#define _IBA_INTERNAL_H_

#include "IBA.h"
#include "LocalBundleAdjustor.h"
#include "GlobalBundleAdjustor.h"

class ViewerIBA;

namespace IBA {
//ICEBA后端
class Internal {

 public:
//对于地图点的观测
  class FeatureMeasurement {
   public:
    inline bool operator < (const FeatureMeasurement &z) const {
      return m_id < z.m_id
#ifdef CFG_STEREO
          || (m_id == z.m_id && !m_right && z.m_right)
#endif
        ;
    }
    inline bool operator < (const int id) const {
      return m_id < id;
    }
   public:
    int m_id;//地图点在所有关键帧地图点中的全局id
    ::Point2D m_z;//左目没有畸变的归一化坐标,或者右目特征点去畸变后的归一化坐标(左乘了Rc0_c1以后进行了归一化)
    LA::SymmetricMatrix2x2f m_W;//畸变部分信息矩阵
    ubyte m_right;
  };

  //地图点索引数据结构
  class MapPointIndex {
   public:
    inline MapPointIndex() {}
    inline MapPointIndex(const int iFrm, const int idx, const int ix) :
            m_iFrm(iFrm), m_G_idx(idx), m_L_idx(ix) {}
    inline bool operator < (const MapPointIndex &idx) const {
      return m_iFrm < idx.m_iFrm || (m_iFrm == idx.m_iFrm && m_G_idx < idx.m_G_idx);
    }
      //输入的是当前帧的id,地图点的全局id,在当前新地图点中的id
    inline void Set(const int iFrm, const int global_idx, const int local_idx) {
      m_iFrm = iFrm;
          m_G_idx = global_idx;
          m_L_idx = local_idx;
    }
   public:
    int m_iFrm/*frame_id 这帧的id*/, m_G_idx/*原始:m_idx,全局id(从1开始),global_idx*/, m_L_idx;//原始:m_ix,局部id就是在这帧的新地图点中的id,local_idx
  };

 public:

  void* operator new(std::size_t count) {
    // [NOTE] : we don't use count here, count is equal to sizeof(Internal)
    return SIMD::Malloc<Internal>();
  }

  void operator delete(void* p) {
    SIMD::Free<Internal>(static_cast<Internal*>(p));
  }

 protected:

  const LocalBundleAdjustor::InputLocalFrame& PushCurrentFrame(const CurrentFrame &CF);
  const GlobalMap::InputKeyFrame& PushKeyFrame(const KeyFrame &KF, const Camera *C = NULL);
  void ConvertFeatureMeasurements(const std::vector<MapPointMeasurement> &cur_feat_measures, FRM::Frame *F);
#ifdef CFG_GROUND_TRUTH
  void PushDepthMeasurementsGT(const FRM::Frame &F);
#endif

  bool SavePoints(const AlignedVector<Rigid3D> &CsKF,
                  const std::vector<::Depth::InverseGaussian> &ds,
                  const std::string fileName, const bool append = true);

  void AssertConsistency();

 protected:

  friend Solver;
  friend LocalBundleAdjustor;
  friend GlobalBundleAdjustor;
  friend ViewerIBA;

  LocalMap m_LM;//局部地图
  GlobalMap m_GM; //全局地图
  LocalBundleAdjustor m_LBA; //局部地图优化器
  GlobalBundleAdjustor m_GBA; //全局地图优化器
  int m_debug;
  int m_nFrms;
  std::vector<int> m_idx2stereo;//记录一下所有的地图点是否是双目,size是和前端地图点id一致,-1是没有这个点,0是不是双目,1是是双目
  std::vector<int> m_idx2iKF;//记录一下所有的地图点是否是双目,size是和前端地图点id一致
  std::vector<FRM::Tag> m_Ts;//所有非关键帧的信息,每一帧进来都会有一个Tag,当这帧同时还是关键帧时,会从m_Ts中把这帧剔除掉
  std::vector<int> m_iKF2d/*记录的是这个关键帧来之前所有的关键帧数量,比如[1]=70，意思就是第1帧来之前有70个特征点*/,
  m_id2iX,//原始名称:m_id2X 有三种id,一种是地图点在这个关键帧中的id
  m_iX2id,//原始名称:m_iX2d一种是地图点在所有关键帧地图点中的全局id(id,iX都是表示的这个),也就是第几个加入到IBA里的还有一个是前端的全局id
  m_id2idx,//还有一个是前端的全局id(idx)
  m_idx2iX;//resize的时候会多加1位,即它的size就是接下来最新的地图点的起始id
  std::vector<::Point2D> m_xs;//对应的m_IKF.m_Xs中地图点的左目观测归一化坐标
  std::list<LocalMap::CameraLF> m_CsLF;//用来获取滑窗中的pose
  std::vector<LocalMap::CameraKF> m_CsKF;//所有的关键帧pose
  std::vector<::Depth::InverseGaussian> m_ds;//所有地图点的逆深度
  std::vector<ubyte> m_uds;
#ifdef CFG_GROUND_TRUTH
  AlignedVector<IMU::Measurement> m_usGT;
  std::vector<int> m_iusGT;
  std::vector<float> m_tsGT;
  std::vector<::Depth::InverseGaussian> m_dsGT;
  AlignedVector<Rotation3D> m_RsGT;
  AlignedVector<LA::AlignedVector3f> m_TsGT;
  std::vector<std::vector<::Depth::Measurement> > m_zsGT;
#endif

  Camera::Calibration m_K; //标定参数
  ::Intrinsic::UndistortionMap m_UM;//左目畸变对照表,key是畸变像素坐标,value是无畸变归一化坐标
#ifdef CFG_STEREO
  ::Intrinsic::UndistortionMap m_UMr;//右目畸变对照表,key是畸变像素坐标,value是无畸变归一化坐标
#endif
  AlignedVector<Camera> m_CsGT;
  std::vector<::Depth::InverseGaussian> m_DsGT;
  std::string m_dir;

  std::vector<FeatureMeasurement> m_zsSortTmp;//暂时存储当前帧对于老地图点去畸变后的观测
  std::vector<FTR::Measurement> m_zsTmp;//当前帧对老地图点的观测
  std::vector<MapPointIndex> m_idxsSortTmp;//当前关键帧新地图点的id管理,每一个元素存储帧的id,全局地图点id,局部地图点id
  std::vector<int> m_idxsTmp;//下标对应于m_IKF.m_Xs的下标,值对应于KF.Xs的下标

  LocalBundleAdjustor::InputLocalFrame m_ILF;//当前输入的普通帧
  GlobalMap::InputKeyFrame m_IKF;//当前输入的关键帧

  std::vector<GlobalMap::InputCamera> m_ICs;

  IMU::Delta m_D;
  AlignedVector<IMU::Measurement> m_us;
  AlignedVector<float> m_work;
};

}

#endif
