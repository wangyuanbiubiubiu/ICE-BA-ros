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
#ifndef _GLOBAL_MAP_H_
#define _GLOBAL_MAP_H_

#include "Frame.h"
#include "Rigid.h"
#include "Depth.h"
#include "MultiThread.h"

#define GM_FLAG_FRAME_DEFAULT       0
#define GM_FLAG_FRAME_UPDATE_CAMERA 1
#define GM_FLAG_FRAME_UPDATE_DEPTH  2
//全局地图
class GlobalMap {

 public:
   
  class InputCamera {
   public:
    inline InputCamera() {}
    inline InputCamera(const Rigid3D &C, const int iFrm) : m_Cam_pose(C), m_iFrm(iFrm) {}
    inline bool operator < (const int iFrm) const { return m_iFrm < iFrm; }
    inline bool operator < (const InputCamera &C) const { return m_iFrm < C.m_iFrm; }
   public:
    Rigid3D m_Cam_pose;//相机的pose,
    int m_iFrm;//这帧图像的id
  };

  class Camera : public InputCamera {
   public:
    inline Camera() : InputCamera() {}
    inline Camera(const Rigid3D &C, const int iFrm,
                  const ubyte uc = GM_FLAG_FRAME_DEFAULT
#ifdef CFG_HANDLE_SCALE_JUMP
                , const float d = 0.0f
#endif
                ) : InputCamera(C, iFrm), m_uc(uc)
#ifdef CFG_HANDLE_SCALE_JUMP
                  , m_d(d)
#endif
    {}
   public:
    ubyte m_uc;
#ifdef CFG_HANDLE_SCALE_JUMP
    float m_d;
#endif
  };

  class Point {
   public:
    inline void SaveB(FILE *fp) const {
      UT::SaveB(m_iKF, fp);
      UT::SaveB(m_x, fp);
      UT::SaveB(m_W, fp);
      UT::VectorSaveB(m_zs, fp);
      UT::SaveB(m_d, fp);
    }
    inline void LoadB(FILE *fp) {
      UT::LoadB(m_iKF, fp);
      UT::LoadB(m_x, fp);
      UT::LoadB(m_W, fp);
      UT::VectorLoadB(m_zs, fp);
      UT::LoadB(m_d, fp);
    }
    inline void AssertConsistency() const {
      const int Nz = static_cast<int>(m_zs.size());
      for (int iz = 0; iz < Nz; ++iz) {
        const int iKF = m_zs[iz].m_iKF;
        UT_ASSERT(iKF > m_iKF);
        if (iz > 0) {
          UT_ASSERT(iKF > m_zs[iz - 1].m_iKF);
        }
      }
    }
   public:
    int m_iKF;//在哪个关键帧被首次观测到的,存的是关键帧的局部id
    FTR::Source m_x;//首次被观测到时的左右目的无畸变归一化坐标以及其他东西
    LA::SymmetricMatrix2x2f m_W;//左目观测畸变部分H矩阵
    std::vector<FTR::Measurement> m_zs;//这个地图点除首次观测到它的那帧以外其他帧的观测
    Depth::InverseGaussian m_d;//左相机坐标系下特征点的逆深度以及协方差
  };
//构造输入关键帧
  class InputKeyFrame : public FRM::Frame {
   public:
    //inline InputKeyFrame() {}
    //inline InputKeyFrame(const FRM::Frame &F, const Camera &Cam_state, const std::vector<Point> &Xs) :
    //                     FRM::Frame(F), m_Cam_pose(Cam_state), m_Xs(Xs) {}
    inline void AssertConsistency() const {
      const int NX = static_cast<int>(m_Xs.size());
      for (int iX = 0; iX < NX; ++iX) {
        const Point &X = m_Xs[iX];
        X.AssertConsistency();
        if (iX > 0) {
          UT_ASSERT(X.m_iKF >= m_Xs[iX - 1].m_iKF);
        }
      }
    }
   public:
    ::Camera m_Cam_state;//相机的状态,pvq,bias
    std::vector<Point> m_Xs;//关键帧新观测到的地图点
  };

  class KeyFrame : public FRM::Frame {
   public:
    inline KeyFrame() : FRM::Frame() {}
    inline KeyFrame(const KeyFrame &KF) { *this = KF; }
    inline void operator = (const KeyFrame &KF) {
      *((FRM::Frame *) this) = KF;
      m_xs = KF.m_xs;
    }
    inline bool operator == (const int iFrm) const { return m_T.m_iFrm == iFrm; }
    inline bool operator < (const int iFrm) const { return m_T.m_iFrm < iFrm; }
    inline void Initialize(const FRM::Frame &F) {
      FRM::Frame::Initialize(F);
      m_xs.resize(0);
    }
    inline void PushFeatures(const std::vector<FTR::Source> &xs) {
      m_xs.insert(m_xs.end(), xs.begin(), xs.end());
    }
    inline void SaveB(FILE *fp) const {
      FRM::Frame::SaveB(fp);
      UT::VectorSaveB(m_xs, fp);
    }
    inline void LoadB(FILE *fp) {
      FRM::Frame::LoadB(fp);
      UT::VectorLoadB(m_xs, fp);
    }
    inline void AssertConsistency(const int iKF) const {
      Frame::AssertConsistency();
      const int NZ = static_cast<int>(m_Zs.size());
      for (int iZ = 0; iZ < NZ; ++iZ) {
        UT_ASSERT(m_Zs[iZ].m_iKF < iKF);
      }
    }
   public:
    std::vector<FTR::Source> m_xs;/*所有新地图点（由当前关键帧产生）的观测*/
  };

  class KeyFrameBA : public KeyFrame {
   public:
    inline KeyFrameBA() : KeyFrame() {}
    inline KeyFrameBA(const KeyFrameBA &KF) { *this = KF; }
    inline void operator = (const KeyFrameBA &KF) {
      *((KeyFrame *) this) = KF;
      m_Apds.Set(KF.m_Apds);
#ifdef CFG_STEREO
      m_Ards.Set(KF.m_Ards);
#endif
    }
    inline void Initialize(const FRM::Frame &F) {
      KeyFrame::Initialize(F);
      m_Apds.Resize(0);
#ifdef CFG_STEREO
      m_Ards.Resize(0);
#endif
    }
    inline void PushFeatures(const std::vector<FTR::Source> &xs/*所有新地图点（由当前关键帧产生）的观测*/) {
      KeyFrame::PushFeatures(xs/*所有新地图点（由当前关键帧产生）的观测*/);//将xs保存到m_xs里
      const int ix = m_Apds.Size(), Nx = static_cast<int>(m_xs.size()) - ix;
      m_Apds.InsertZero(ix, Nx, NULL);//左目因子扩容
#ifdef CFG_STEREO
      m_Ards.InsertZero(ix, Nx, NULL);//右目扩容
#endif
    }
    inline void InvalidateFeatures(const ubyte *mxs) {
      const int Nx = static_cast<int>(m_xs.size());
      for (int ix = 0; ix < Nx; ++ix) {
        if (!mxs[ix]) {
          continue;
        }
        m_Apds[ix].MakeZero();
#ifdef CFG_STEREO
        m_Ards[ix].MakeZero();
#endif
      }
    }
    inline void MakeZero() {
      m_Apds.MakeZero();
#ifdef CFG_STEREO
      m_Ards.MakeZero();
#endif
    }
    inline void SaveB(FILE *fp) const {
      KeyFrame::SaveB(fp);
      m_Apds.SaveB(fp);
#ifdef CFG_STEREO
      m_Ards.SaveB(fp);
#endif
    }
    inline void LoadB(FILE *fp) {
      KeyFrame::LoadB(fp);
      m_Apds.LoadB(fp);
#ifdef CFG_STEREO
      m_Ards.LoadB(fp);
#endif
    }
    inline void AssertConsistency(const int iKF) const {
      KeyFrame::AssertConsistency(iKF);
	  const int Nx = static_cast<int>(m_xs.size());
      UT_ASSERT(m_Apds.Size() == Nx);
#ifdef CFG_STEREO
      UT_ASSERT(m_Ards.Size() == Nx);
#endif
    }
   public://下面两个因子都是关于新地图点的
    AlignedVector<Depth::Prior::Factor> m_Apds;//只有左目观测时的因子 costfun(Uc0)：||地图点逆深度 - 关键帧平均逆深度||^2 马氏
#ifdef CFG_STEREO
    AlignedVector<FTR::Factor::Stereo> m_Ards;//两目都有观测时的因子,cost_F(Uc0) = ||归一化(Pnc0 - Uc0 * tc0c1) - 归一化(Rc0c1 *Pnc1)||^2（马氏距离下）
#endif
  };

 public:

  void LBA_Reset();
  void LBA_PushKeyFrame(const Camera &C);
  void LBA_DeleteKeyFrame(const int iFrm, const int iKF);
  std::vector<GlobalMap::Camera> Get_Total_KFs();

  ubyte LBA_Synchronize(const int iFrm, AlignedVector<Rigid3D> &Cs, AlignedVector<Rigid3D> &CsBkp,
                        std::vector<ubyte> &ucs
#ifdef CFG_HANDLE_SCALE_JUMP
                      , std::vector<float> &ds, std::vector<float> &dsBkp
#endif
                      );
  void GBA_Update(const std::vector<int> &iFrms, const AlignedVector<Rigid3D> &Cs,
                  const std::vector<ubyte> &ucs
#ifdef CFG_HANDLE_SCALE_JUMP
                , const std::vector<float> &ds
#endif
                );

  void SaveB(FILE *fp);
  void LoadB(FILE *fp);
  void AssertConsistency();

 protected:

  std::vector<Camera> m_Cs;//存储了所有关键帧的pose(还有这帧的id),会在优化完以后被更新
  ubyte m_Uc;//全局地图是否要更新状态,确定一下这个啥时候更新
  boost::shared_mutex m_MT;

};

#endif
