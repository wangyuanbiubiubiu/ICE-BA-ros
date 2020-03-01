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
#include "stdafx.h"
//#ifndef CFG_DEBUG
//#define CFG_DEBUG
//#endif
#include "GlobalMap.h"

void GlobalMap::LBA_Reset() {
  MT_WRITE_LOCK_BEGIN(m_MT, MT_TASK_NONE, MT_TASK_GM_LBA_Reset);
  m_Cs.resize(0);
  m_Uc = GM_FLAG_FRAME_DEFAULT;
  MT_WRITE_LOCK_END(m_MT, MT_TASK_NONE, MT_TASK_GM_LBA_Reset);
}

void GlobalMap::LBA_PushKeyFrame(const Camera &C) {
  MT_WRITE_LOCK_BEGIN(m_MT, C.m_iFrm, MT_TASK_GM_LBA_PushKeyFrame);
  m_Cs.push_back(C);
  MT_WRITE_LOCK_END(m_MT, C.m_iFrm, MT_TASK_GM_LBA_PushKeyFrame);
}

void GlobalMap::LBA_DeleteKeyFrame(const int iFrm, const int iKF) {
  MT_WRITE_LOCK_BEGIN(m_MT, iFrm, MT_TASK_GM_LBA_DeleteKeyFrame);
  const ubyte uc = m_Cs[iKF].m_uc;
  m_Cs.erase(m_Cs.begin() + iKF);
  if (uc & GM_FLAG_FRAME_UPDATE_CAMERA) {
    bool found = false;
    const int N = static_cast<int>(m_Cs.size());
    for (int i = 0; i < N && !found; ++i) {
      found = (m_Cs[i].m_uc & GM_FLAG_FRAME_UPDATE_CAMERA) != 0;
    }
    if (!found) {
      m_Uc &= ~GM_FLAG_FRAME_UPDATE_CAMERA;
    }
  }
  if (uc & GM_FLAG_FRAME_UPDATE_DEPTH) {
    bool found = false;
    const int N = static_cast<int>(m_Cs.size());
    for (int i = 0; i < N && !found; ++i) {
      found = (m_Cs[i].m_uc & GM_FLAG_FRAME_UPDATE_DEPTH) != 0;
    }
    if (!found) {
      m_Uc &= ~GM_FLAG_FRAME_UPDATE_DEPTH;
    }
  }
  MT_WRITE_LOCK_END(m_MT, iFrm, MT_TASK_GM_LBA_DeleteKeyFrame);
}

std::vector<GlobalMap::Camera> GlobalMap::Get_Total_KFs()
{
    return m_Cs;
}
//
ubyte GlobalMap::LBA_Synchronize(const int iFrm/*最新的关键帧对应的普通帧的id*/, AlignedVector<Rigid3D> &Cs,/*关键帧左相机位姿*/
                                 AlignedVector<Rigid3D> &CsBkp/*关键帧左相机位姿备份*/, std::vector<ubyte> &ucs/*最新的共视关键帧中地图点是否有子轨迹生成*/
#ifdef CFG_HANDLE_SCALE_JUMP
                               , std::vector<float> &ds, std::vector<float> &dsBkp
#endif
                               )
{
  ubyte ret;
  MT_WRITE_LOCK_BEGIN(m_MT, iFrm, MT_TASK_GM_LBA_Synchronize);
  if (m_Uc == GM_FLAG_FRAME_DEFAULT)
  {
    ret = GM_FLAG_FRAME_DEFAULT;
  } else
  {
      m_Uc = GM_FLAG_FRAME_DEFAULT;
      const int N = static_cast<int>(m_Cs.size());
#ifdef CFG_DEBUG
      UT_ASSERT(Cs.Size() == N);
#endif
      CsBkp.Resize(N);
      ucs.assign(N, GM_FLAG_FRAME_DEFAULT);
#ifdef CFG_HANDLE_SCALE_JUMP
      dsBkp.resize(N);
#endif
      for (int i = 0; i < N; ++i)
      {
          Camera &C = m_Cs[i];
          if (C.m_uc == GM_FLAG_FRAME_DEFAULT)
          {
              continue;
          }
          if (C.m_uc & GM_FLAG_FRAME_UPDATE_CAMERA)
          {
              CsBkp[i] = Cs[i];
              Cs[i] = C.m_Cam_pose;
          }
          ucs[i] = C.m_uc;//将需要更新的关键帧记录下来
#ifdef CFG_HANDLE_SCALE_JUMP
          if (Cam_state.m_uc & GM_FLAG_FRAME_UPDATE_DEPTH) {
        dsBkp[i] = ds[i];
        ds[i] = Cam_state.m_d;
      }
#endif
          C.m_uc = GM_FLAG_FRAME_DEFAULT;//设置成不需要更新
      }
      ret = GM_FLAG_FRAME_UPDATE_CAMERA;
  }
  MT_WRITE_LOCK_END(m_MT, iFrm, MT_TASK_GM_LBA_Synchronize);
    return ret;
}
//更新GlobalMap里m_Cs的pose
void GlobalMap::GBA_Update(const std::vector<int> &iFrms/*关键帧和普通帧之间的id对应*/, const AlignedVector<Rigid3D> &Cs/*更新以后的相机状态*/,
                           const std::vector<ubyte> &ucs
#ifdef CFG_HANDLE_SCALE_JUMP
                         , const std::vector<float> &ds
#endif
                         ) {
  MT_WRITE_LOCK_BEGIN(m_MT, iFrms.back(), MT_TASK_GM_GBA_Update);
  std::vector<Camera>::iterator i = m_Cs.begin();
  const int N = static_cast<int>(iFrms.size());
  for (int j = 0; j < N; ++j) {//遍历所有关键帧
    const ubyte uc = ucs[j];
    if (uc == GM_FLAG_FRAME_DEFAULT) {
      continue;
    }
    const int iFrm = iFrms[j];//关键帧id对应的全局帧id
    i = std::lower_bound(i, m_Cs.end(), iFrm);
    if (i == m_Cs.end()) {
      break;
    } else if (i->m_iFrm != iFrm) {
      continue;
    }
    if (uc & GM_FLAG_FRAME_UPDATE_CAMERA) {
      i->m_Cam_pose = Cs[j];
      i->m_uc |= GM_FLAG_FRAME_UPDATE_CAMERA;
      m_Uc |= GM_FLAG_FRAME_UPDATE_CAMERA;
    }
#ifdef CFG_HANDLE_SCALE_JUMP
    if (uc & GM_FLAG_FRAME_UPDATE_DEPTH) {
      i->m_d = ds[j];
      i->m_uc |= GM_FLAG_FRAME_UPDATE_DEPTH;
      m_Uc |= GM_FLAG_FRAME_UPDATE_DEPTH;
    }
#endif
  }
  MT_WRITE_LOCK_END(m_MT, iFrms.back(), MT_TASK_GM_GBA_Update);
}

void GlobalMap::SaveB(FILE *fp) {
  MT_READ_LOCK_BEGIN(m_MT, MT_TASK_NONE, MT_TASK_NONE);
  UT::VectorSaveB(m_Cs, fp);
  UT::SaveB(m_Uc, fp);
  MT_READ_LOCK_END(m_MT, MT_TASK_NONE, MT_TASK_NONE);
}

void GlobalMap::LoadB(FILE *fp) {
  MT_WRITE_LOCK_BEGIN(m_MT, MT_TASK_NONE, MT_TASK_NONE);
  UT::VectorLoadB(m_Cs, fp);
  UT::LoadB(m_Uc, fp);
  MT_WRITE_LOCK_END(m_MT, MT_TASK_NONE, MT_TASK_NONE);
}

void GlobalMap::AssertConsistency() {
  MT_READ_LOCK_BEGIN(m_MT, MT_TASK_NONE, MT_TASK_NONE);
  ubyte Uc = GM_FLAG_FRAME_DEFAULT;
  const int N = static_cast<int>(m_Cs.size());
  for (int i = 0; i < N; ++i) {
    const Camera &C = m_Cs[i];
    C.m_Cam_pose.AssertOrthogonal();
    Uc |= C.m_uc;
  }
  UT_ASSERT(m_Uc == Uc);
  MT_READ_LOCK_END(m_MT, MT_TASK_NONE, MT_TASK_NONE);
}
