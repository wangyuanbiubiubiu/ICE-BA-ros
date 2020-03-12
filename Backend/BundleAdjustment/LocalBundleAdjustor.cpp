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
#include "LocalBundleAdjustor.h"
#include "GlobalBundleAdjustor.h"
#include "IBA_internal.h"
#include "Vector12.h"
#include <glog/logging.h>
#include <Eigen/Core>
#ifdef CFG_DEBUG
#ifdef CFG_DEBUG_EIGEN
#define LBA_DEBUG_EIGEN
#endif
#if WIN32
#define LBA_DEBUG_CHECK
//#define LBA_DEBUG_PRINT
//#define LBA_DEBUG_PRINT_STEP
#ifdef CFG_GROUND_TRUTH
//#define LBA_DEBUG_GROUND_TRUTH_MEASUREMENT
#endif
//#define LBA_DEBUG_PRINT_MARGINALIZATION
#define LBA_DEBUG_VIEW
#ifdef LBA_DEBUG_VIEW
#include "ViewerIBA.h"
static ViewerIBA *g_viewer = NULL;
#endif
#endif
#endif

#ifdef LBA_DEBUG_CHECK
static const int g_ic = 0;
//static const int g_ic = INT_MAX;
static const int g_iKF = 1;
//static const int g_iKF = INT_MAX;
static const int g_ix = 8;
static const int g_ist = 41;
static const int g_izKF = 0;
static const int g_izLF = 22;
static const Rigid3D *g_CKF = NULL;
static const LocalBundleAdjustor::KeyFrame *g_KF = NULL;
static const Depth::InverseGaussian *g_d = NULL;
static const FTR::Factor::FixSource::Source::A *g_Ax = NULL;
static const FTR::Factor::FixSource::Source::A *g_AxST = NULL;
static const FTR::Factor::FixSource::Source::M *g_MxST = NULL;
static const FTR::Factor::FixSource::Source::M *g_Mx = NULL;
static const FTR::Factor::Depth *g_AzKF = NULL;
static const Camera *g_CLF = NULL;
static const LocalBundleAdjustor::LocalFrame *g_LF = NULL;
static const Camera::Factor::Unitary::CC *g_SAcuLF = NULL;
static const Camera::Factor::Unitary::CC *g_SMcuLF = NULL;
static const Camera::Factor *g_SAcmLF = NULL;
static const FTR::Factor::FixSource::A1 *g_Az1LF = NULL;
static const FTR::Factor::FixSource::A2 *g_Az2LF = NULL;
static const FTR::Factor::FixSource::M1 *g_Mz1LF = NULL;
static const FTR::Factor::FixSource::M2 *g_Mz2LF = NULL;
static const FTR::Factor::DD *g_SmddST = NULL;
#endif
//1 0 0 1
void LocalBundleAdjustor::Initialize(IBA::Solver *solver, const int serial, const int verbose,
                                     const int debug, const int history) {
  MT::Thread::Initialize(serial, 4, "LBA");
  m_solver = solver;
  m_LM = &solver->m_internal->m_LM;
  m_GM = &solver->m_internal->m_GM;
  m_GBA = &solver->m_internal->m_GBA;
  m_K = solver->m_internal->m_K;
  m_verbose = verbose;
  m_debug = debug;
  m_history = history;
  m_dir = solver->m_internal->m_dir;
#ifdef CFG_GROUND_TRUTH
  m_CsGT = solver->m_internal->m_CsGT.Data();
#ifdef LBA_DEBUG_GROUND_TRUTH_MEASUREMENT
  if (m_CsGT) {
    LA::AlignedVector3f ba, bw;
    ba.MakeZero();
    bw.MakeZero();
    const int N = solver->m_internal->m_CsGT.Size();
    for (int i = 0; i < N; ++i) {
      const Camera &Cam_state = m_CsGT[i];
      ba += Cam_state.m_ba;
      bw += Cam_state.m_bw;
    }
    const float s = 1.0f / N;
    ba *= s;
    bw *= s;
    Camera *Cs = (Camera *) m_CsGT;
    for (int i = 0; i < N; ++i) {
      Camera &Cam_state = Cs[i];
      Cam_state.m_ba = ba;
      Cam_state.m_bw = bw;
    }
  }
#endif
  m_dsGT = solver->m_internal->m_DsGT.empty() ? NULL : &solver->m_internal->m_dsGT;
#ifdef CFG_HISTORY
  if (m_history >= 3 && (!m_CsGT || !m_dsGT)) {
    m_history = 2;
  }
#endif
#endif
}

void LocalBundleAdjustor::Reset() {
  MT::Thread::Reset();
  MT_WRITE_LOCK_BEGIN(m_MT, MT_TASK_NONE, MT_TASK_LBA_Reset);
  m_ITs1.resize(0);
  m_ITs2.resize(0);
  m_ILFs1.resize(0);
  m_ILFs2.resize(0);
  m_IKFs1.resize(0);
  m_IKFs2.resize(0);
  m_IDKFs1.resize(0);
  m_IDKFs2.resize(0);
  m_IDMPs1.resize(0);
  m_IDMPs2.resize(0);
  m_IUCs1.resize(0);
  m_IUCs2.resize(0);
  MT_WRITE_LOCK_END(m_MT, MT_TASK_NONE, MT_TASK_LBA_Reset);

  m_GM->LBA_Reset();

  for (int i = 0; i < TM_TYPES; ++i) {
    m_ts[i].Reset(TIME_AVERAGING_COUNT);
  }
#ifdef CFG_HISTORY
  m_hists.resize(0);
#endif

  m_delta2 = BA_DL_RADIUS_INITIAL;

  m_ic2LF.reserve(LBA_MAX_LOCAL_FRAMES);    m_ic2LF.resize(0);
  m_LFs.reserve(LBA_MAX_LOCAL_FRAMES);      m_LFs.resize(0);
  m_CsLF.Reserve(LBA_MAX_LOCAL_FRAMES);     m_CsLF.Resize(0);
#ifdef CFG_GROUND_TRUTH
  m_CsLFGT.Reserve(LBA_MAX_LOCAL_FRAMES);   m_CsLFGT.Resize(0);
#endif
  m_ucsLF.reserve(LBA_MAX_LOCAL_FRAMES);    m_ucsLF.resize(0);
  m_ucmsLF.reserve(LBA_MAX_LOCAL_FRAMES);   m_ucmsLF.resize(0);
#ifdef CFG_INCREMENTAL_PCG
  m_xcsLF.Reserve(LBA_MAX_LOCAL_FRAMES);    m_xcsLF.Resize(0);
  m_xmsLF.Reserve(LBA_MAX_LOCAL_FRAMES);    m_xmsLF.Resize(0);
#endif
  m_DsLF.Reserve(LBA_MAX_LOCAL_FRAMES);     m_DsLF.Resize(0);
#ifdef CFG_GROUND_TRUTH
  m_DsLFGT.Reserve(LBA_MAX_LOCAL_FRAMES);   m_DsLFGT.Resize(0);
#endif
  m_AdsLF.Reserve(LBA_MAX_LOCAL_FRAMES);    m_AdsLF.Resize(0);
  m_AfpsLF.Reserve(LBA_MAX_LOCAL_FRAMES);   m_AfpsLF.Resize(0);
  m_AfmsLF.Reserve(LBA_MAX_LOCAL_FRAMES);   m_AfmsLF.Resize(0);
  m_SAcusLF.Reserve(LBA_MAX_LOCAL_FRAMES);  m_SAcusLF.Resize(0);
  m_SMcusLF.Reserve(LBA_MAX_LOCAL_FRAMES);  m_SMcusLF.Resize(0);
  m_SAcmsLF.Reserve(LBA_MAX_LOCAL_FRAMES);  m_SAcmsLF.Resize(0);

  m_KFs.resize(0);
  m_iFrmsKF.resize(0);
  m_CsKF.Resize(0);
#ifdef CFG_GROUND_TRUTH
  m_CsKFGT.Resize(0);
#endif
  m_ucsKF.resize(0);
#ifdef CFG_HANDLE_SCALE_JUMP
  m_dsKF.resize(0);
#endif
  m_usKF.Resize(0);
  m_usKFLast.Resize(0);

  m_iKF2d.assign(1, 0);
  m_ds.resize(0);
  m_uds.resize(0);

#ifdef CFG_CHECK_REPROJECTION
  m_esLF.resize(0);
  m_esKF.resize(0);
#endif

  m_ZpLF.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL, BA_VARIANCE_PRIOR_VELOCITY_FIRST,
                    BA_VARIANCE_PRIOR_BIAS_ACCELERATION_FIRST,
                    BA_VARIANCE_PRIOR_BIAS_GYROSCOPE_FIRST);
#ifdef LBA_DEBUG_GROUND_TRUTH_MEASUREMENT
  if (m_CsGT) {
    m_ZpLF.DebugSetMeasurement(m_CsGT[0]);
  }
#endif
  m_Zp.Initialize(m_ZpLF);
  m_ApLF.MakeZero();
  m_ZpKF.Invalidate();
  //m_F = 0.0f;
}

void LocalBundleAdjustor::PushLocalFrame(const InputLocalFrame &ILF) {
  MT_WRITE_LOCK_BEGIN(m_MT, ILF.m_T.m_iFrm, MT_TASK_LBA_PushLocalFrame);
  m_ITs1.push_back(IT_LOCAL_FRAME);
  m_ILFs1.push_back(ILF);
  MT_WRITE_LOCK_END(m_MT, ILF.m_T.m_iFrm, MT_TASK_LBA_PushLocalFrame);
}

void LocalBundleAdjustor::PushKeyFrame(const GlobalMap::InputKeyFrame &IKF, const bool serial) {
  MT_WRITE_LOCK_BEGIN(m_MT, IKF.m_T.m_iFrm, MT_TASK_LBA_PushKeyFrame);
  if (m_serial) {
    m_ITs1.push_back(IT_KEY_FRAME_SERIAL);
  } else {
    m_ITs1.push_back(IT_KEY_FRAME);
  }
  m_IKFs1.push_back(IKF);
  MT_WRITE_LOCK_END(m_MT, IKF.m_T.m_iFrm, MT_TASK_LBA_PushKeyFrame);
}

void LocalBundleAdjustor::PushDeleteKeyFrame(const int iFrm, const int iKF){
  MT_WRITE_LOCK_BEGIN(m_MT, iFrm, MT_TASK_LBA_PushDeleteKeyFrame);
  m_ITs1.push_back(IT_DELETE_KEY_FRAME);
  m_IDKFs1.push_back(iKF);
  MT_WRITE_LOCK_END(m_MT, iFrm, MT_TASK_LBA_PushDeleteKeyFrame);
}

void LocalBundleAdjustor::PushDeleteMapPoints(const int iFrm, const std::vector<int> &ids) {
  MT_WRITE_LOCK_BEGIN(m_MT, iFrm, MT_TASK_LBA_PushDeleteMapPoints);
  m_ITs1.push_back(IT_DELETE_MAP_POINTS);
  m_IDMPs1.push_back(ids);
  MT_WRITE_LOCK_END(m_MT, iFrm, MT_TASK_LBA_PushDeleteMapPoints);
}

void LocalBundleAdjustor::PushUpdateCameras(const int iFrm,
                                            const std::vector<GlobalMap::InputCamera> &Cs,
                                            const bool serial) {
  MT_WRITE_LOCK_BEGIN(m_MT, iFrm, MT_TASK_LBA_PushUpdateCameras);
  if (m_serial) {
    m_ITs1.push_back(IT_UPDATE_CAMERAS_SERIAL);
  } else {
    m_ITs1.push_back(IT_UPDATE_CAMERAS);
  }
  m_IUCs1.push_back(Cs);
  MT_WRITE_LOCK_END(m_MT, iFrm, MT_TASK_LBA_PushUpdateCameras);
}

void LocalBundleAdjustor::GetCamera(FRM::Tag &T, Camera &C) {
  MT_READ_LOCK_BEGIN(m_MTC, MT_TASK_NONE, MT_TASK_LBA_GetCamera);
  T = m_C.m_T;
  C = m_C.m_C;
  MT_READ_LOCK_END(m_MTC, MT_TASK_NONE, MT_TASK_LBA_GetCamera);
}
//LBA优化
void LocalBundleAdjustor::Run() {
#if 0
//#if 1
  if (m_debug) {
    AssertConsistency();
  }
#endif
#if 0
  if (!m_ic2LF.empty()) {
    const int iFrm = m_LFs[m_ic2LF.back()].m_Cam_pose.m_iFrm;
    UT::PrintSeparator('!');
    UT::Print("[%d]\n", iFrm);
  }
#endif
  m_delta2 = BA_DL_RADIUS_INITIAL;//信赖域半径
  m_ts[TM_TOTAL].Start();//记录电脑unix时间戳
  m_ts[TM_SYNCHRONIZE].Start();
  SynchronizeData();//同步数据,如果有新的关键帧的话,还要唤醒GBA,这里会做很多事情,具体看注释
  m_ts[TM_SYNCHRONIZE].Stop();
  m_ts[TM_TOTAL].Stop();
#if 0
//#if 1
  if (m_debug) {
    AssertConsistency();
  }
#endif
//#ifdef CFG_DEBUG
#if 0
#ifdef WIN32
  const std::string fileName = "D:/tmp/test/cons.txt";
#else
  const std::string fileName = "/tmp/test/cons.txt";
#endif
#if 1
  FILE *fp = fopen(fileName.c_str(), "rb");
  if (fp) {
    AlignedVector<Camera> CsLF;
    CsLF.LoadB(fp);
    const int nLFs = static_cast<int>(m_LFs.size());
    UT_ASSERT(CsLF.Size() == nLFs);
    for (int iLF = 0; iLF < nLFs; ++iLF) {
      m_CsLF[iLF].AssertEqual(CsLF[iLF], 1, UT::String("CsLF[%d]", iLF), -1.0f, -1.0f);
    }
    AlignedVector<Rigid3D> CsKF;
    CsKF.LoadB(fp);
    const int nKFs = static_cast<int>(m_KFs.size());
    UT_ASSERT(CsKF.Size() == nKFs);
    for (int iKF = 0; iKF < nKFs; ++iKF) {
      m_CsKF[iKF].AssertEqual(CsKF[iKF], 1, UT::String("CsKF[%d]", iKF), -1.0f, -1.0f);
    }
    std::vector<Depth::InverseGaussian> ds;
    UT::VectorLoadB(ds, fp);
    const int Nd = static_cast<int>(m_ds.size());
    UT_ASSERT(static_cast<int>(ds.size()) == Nd);
    for (int id = 0; id < Nd; ++id) {
      m_ds[id].AssertEqual(ds[id], 1, UT::String("ds[%d]", id), -1.0f);
    }
    fclose(fp);
    UT::PrintLoaded(fileName);
  }
#else
  FILE *fp = fopen(fileName.c_str(), "wb");
  if (fp) {
    m_CsLF.SaveB(fp);
    m_CsKF.SaveB(fp);
    UT::VectorSaveB(m_ds, fp);
    fclose(fp);
    UT::PrintSaved(fileName);
  }
#endif
#endif
#ifdef CFG_DEBUG
  if (m_debug < 0) {
    return;
  }
#endif
//#ifdef CFG_DEBUG
#if 0
//#if 1
  if (m_debug >= 2) {
    AssertConsistency(false);
  }
#endif
#ifdef LBA_DEBUG_EIGEN
    DebugGenerateTracks();
#endif
#ifdef LBA_DEBUG_CHECK
  {
    const int nKFs = static_cast<int>(m_KFs.size());
    if (nKFs != 0) {
      const int iKF = g_iKF < nKFs ? g_iKF : nKFs - 1;
      g_CKF = &m_CsKF[iKF];
      g_KF = &m_KFs[iKF];
      if (g_ix >= 0 && g_ix < int(g_KF->m_xs.size())) {
        g_d = m_ds.data() + m_iKF2d[iKF] + g_ix;
        g_Ax = &g_KF->m_Axs[g_ix];
        g_Mx = &g_KF->m_Mxs[g_ix];
        if (g_ist >= 0 && g_ist < g_KF->m_ix2ST[g_ix + 1] - g_KF->m_ix2ST[g_ix]) {
          const int iST = g_KF->m_ix2ST[g_ix] + g_ist;
          g_AxST = &g_KF->m_AxsST[iST];
          g_MxST = &g_KF->m_MxsST[iST];
        }
      }
      if (g_izKF >= 0 && g_izKF < g_KF->m_Azs.Size()) {
        g_AzKF = &g_KF->m_Azs[g_izKF];
      }
    }
    const int nLFs = int(m_LFs.size());
    if (nLFs != 0) {
      const int ic = g_ic >= 0 && g_ic < nLFs ? g_ic : nLFs - 1;
      const int iLF = m_ic2LF[ic];
      g_CLF = m_CsLF.Data() + iLF;
      g_LF = &m_LFs[iLF];
      g_SAcuLF = m_SAcusLF.Data() + iLF;
      g_SMcuLF = m_SMcusLF.Data() + iLF;
      g_SAcmLF = m_SAcmsLF.Data() + iLF;
      if (g_izLF >= 0 && g_izLF < int(g_LF->m_zs.size())) {
        g_Az1LF = &g_LF->m_Azs1[g_izLF];
        g_Az2LF = &g_LF->m_Azs2[g_izLF];
        g_Mz1LF = &g_LF->m_Mzs1[g_izLF];
        g_Mz2LF = &g_LF->m_Mzs2[g_izLF];
        g_SmddST = &g_LF->m_SmddsST[g_izLF];
      }
    }
  }
#endif
  const int iFrm = m_LFs[m_ic2LF.back()].m_T.m_iFrm;//最新的一帧的id
#ifdef CFG_VERBOSE
  if (m_verbose == 1) {
    UT::PrintSeparator();
    UT::Print("[%d] LBA\n", iFrm);
  } else if (m_verbose >= 2) {
    const int NzLF = CountMeasurementsFeatureLF(), NzKF = CountMeasurementsFeatureKF();
    const int NsLF = CountSchurComplements();
    const int NST = CountSlidingTracks(), Nd = CountLocalTracks();
    UT::PrintSeparator('*');
    UT::Print("[%d] [LocalBundleAdjustor::Run]\n", iFrm);
    UT::Print("  FrameLF = %d\t\t\tMeasurement = %d\tSchur = %d\n", m_LFs.size(), NzLF, NsLF);
    UT::Print("  FrameKF = %d\t\t\tMeasurement = %d\n", m_KFs.size(), NzKF);
    UT::Print("  TrackST = %d * %.2f = %d\n", Nd, UT::Percentage(NST, Nd), NST);
#ifdef CFG_GROUND_TRUTH
    if (!m_CsLFGT.Empty() && !m_CsKFGT.Empty() && !m_DsLFGT.Empty() && m_dsGT) {
      UT::PrintSeparator();
      PrintErrorStatistic("*GT: ", m_CsLFGT, m_CsKFGT, *m_dsGT, m_DsLFGT, true);
    }
#endif
  }
#endif
//#ifdef CFG_DEBUG
#if 0
//#if 1
  {
    UT::Check("Noise\n");
    //const float erMax = 1.0f;
    const float erMax = 10.0f;
    //const float epMax = 0.0f;
    const float epMax = 0.1f;
    //const float epMax = 1.0f;
    //const float evMax = 0.0f;
    const float evMax = 0.01f;
    //const float ebaMax = 0.0f;
    const float ebaMax = 0.001f;
    //const float ebwMax = 0.0f;
    const float ebwMax = 1.0f * UT_FACTOR_DEG_TO_RAD;
    //const float edMax = 0.0f;
    const float edMax = 0.1f;
    const ubyte ucmFlag = LBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION |
                          LBA_FLAG_CAMERA_MOTION_UPDATE_POSITION |
                          LBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY |
                          LBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_ACCELERATION |
                          LBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_GYROSCOPE;
    const int nLFs = static_cast<int>(m_LFs.size());
    for (int iLF = 0; iLF < nLFs; ++iLF) {
      Camera &Cam_state = m_CsLF[iLF];
      Cam_state.m_Cam_pose = Rotation3D::GetRandom(erMax * UT_FACTOR_DEG_TO_RAD) * Cam_state.m_Cam_pose;
      Cam_state.m_p += LA::AlignedVector3f::GetRandom(epMax);
      Cam_state.m_Cam_pose.SetPosition(Cam_state.m_p);
      m_ucsLF[iLF] |= LBA_FLAG_FRAME_UPDATE_CAMERA;
      Cam_state.m_v += LA::AlignedVector3f::GetRandom(evMax);
      Cam_state.m_ba += LA::AlignedVector3f::GetRandom(ebaMax);
      Cam_state.m_bw += LA::AlignedVector3f::GetRandom(ebwMax);
      m_ucmsLF[iLF] |= ucmFlag;
      m_UcsLF[iLF] |= LM_FLAG_FRAME_UPDATE_CAMERA_LF;
    }
    const int nKFs = static_cast<int>(m_KFs.size());
    for (int iKF = 0; iKF < nKFs; ++iKF) {
      const KeyFrame &KF = m_KFs[iKF];
      Depth::InverseGaussian *ds = m_ds.data() + KF.m_id;
      ubyte *uds = m_uds.data() + KF.m_id, *Uds = m_Uds.data() + KF.m_id;
      const int Nx = static_cast<int>(KF.m_xs.size());
      for (int ix = 0; ix < Nx; ++ix) {
        ds[ix].u() += UT::Random<float>(edMax);
        uds[ix] |= LBA_FLAG_TRACK_UPDATE_DEPTH;
        Uds[ix] |= LM_FLAG_TRACK_UPDATE_DEPTH;
      }
      m_ucsKF[iKF] |= LBA_FLAG_FRAME_UPDATE_DEPTH;
      m_UcsKF[iKF] |= LM_FLAG_FRAME_UPDATE_DEPTH;
    }
  }
#endif
#ifdef LBA_DEBUG_VIEW
  if (m_verbose >= 2) {
    //if (iFrm == 15)
    if (!g_viewer) {
      g_viewer = new ViewerIBA();
      g_viewer->Create(m_solver);
      g_viewer->m_keyPause = true;
      //g_viewer->m_keyPause = m_verbose >= 3;
      g_viewer->m_keyDrawCamTypeLF = ViewerIBA::DRAW_CAM_LF_LBA;
      g_viewer->m_keyDrawCamTypeKF = ViewerIBA::DRAW_CAM_KF_LBA;
      g_viewer->m_keyDrawDepType = ViewerIBA::DRAW_DEP_LBA;
    }
    if (g_viewer) {
      g_viewer->m_iLF = m_ic2LF.back();
      g_viewer->ActivateFrame(g_viewer->GetKeyFrames() + g_viewer->GetLocalFrames() - 1);
    }
  }
#endif
#if 0
//#if 1
  {
    UT::Check("Noise\n");
    UpdateFactors();
    UT::PrintSeparator();
    PrintErrorStatistic("    ", m_CsLF, m_CsKF, m_ds, m_DsLF, false);
    UT::Print("\n");
    const float edMax = 1.0f;
    const int Nd = int(m_ds.size());
    for (int id = 0; id < Nd; ++id) {
      if (m_uds[id] & LBA_FLAG_TRACK_UPDATE_DEPTH) {
        m_ds[id].u() += UT::Random<float>(edMax);
      }
    }
    UT::PrintSeparator();
    PrintErrorStatistic("--> ", m_CsLF, m_CsKF, m_ds, m_DsLF, false);
    UT::Print("\n");
#ifdef LBA_DEBUG_VIEW
    if (g_viewer) {
      g_viewer->Run();
    }
#endif
    EmbeddedPointIteration();
    UT::PrintSeparator();
    PrintErrorStatistic("--> ", m_CsLF, m_CsKF, m_ds, m_DsLF, false);
    UT::Print("\n");
#ifdef LBA_DEBUG_VIEW
    if (g_viewer) {
      g_viewer->Run();
    }
#endif
  }
#endif
  m_ts[TM_TOTAL].Start();
  for (m_iIter = 0; m_iIter < BA_MAX_ITERATIONS; ++m_iIter) {
//#ifdef CFG_DEBUG
#if 0
    if (m_iIter == 2) {
      m_debug = 3;
    } else {
      m_debug = 0;
    }
#endif
//#ifdef CFG_DEBUG
#if 0
//#if 1
    if (m_iIter == 0) {
      UT::Check("Update\n");
//#ifdef CFG_DEBUG
#if 0
      ((Depth::InverseGaussian *) g_d)->u() = 0.0f;
#endif
      const int Nc = int(m_LFs.size());
      for (int ic = 0; ic < Nc; ++ic) {
        const int iLF = m_ic2LF[ic];
        m_ucsLF[iLF] |= LBA_FLAG_FRAME_UPDATE_CAMERA;
        m_ucmsLF[iLF] |= LBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION | LBA_FLAG_CAMERA_MOTION_UPDATE_POSITION |
                         LBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY |
                         LBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_ACCELERATION | LBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_GYROSCOPE;
        m_UcsLF[iLF] |= LM_FLAG_FRAME_UPDATE_CAMERA_LF;
        m_LFs[iLF].MakeZero();
        if (ic == 0) {
          continue;
        }
        IMU::Delta &D = m_DsLF[iLF];
        const LocalFrame &LF = m_LFs[iLF];
        const int _iLF = m_ic2LF[ic - 1];
        IMU::PreIntegrate(LF.m_imu_measures, m_LFs[_iLF].m_Cam_pose.m_t, LF.m_Cam_pose.m_t, m_CsLF[_iLF], &D, &m_work,
                          true, D.m_u1.Valid() ? &D.m_u1 : NULL, D.m_u2.Valid() ? &D.m_u2 : NULL,
                          BA_ANGLE_EPSILON);
      }
      m_AdsLF.MakeZero();
      m_AfpsLF.MakeZero();
      m_AfmsLF.MakeZero();
      const int nKFs = static_cast<int>(m_KFs.size());
      for (int iKF = 0; iKF < nKFs; ++iKF) {
        m_ucsKF[iKF] |= LBA_FLAG_FRAME_UPDATE_CAMERA | LBA_FLAG_FRAME_UPDATE_DEPTH;
        m_UcsKF[iKF] |= LM_FLAG_FRAME_UPDATE_CAMERA_KF | LM_FLAG_FRAME_UPDATE_DEPTH;
        m_KFs[iKF].MakeZero();
      }
      const int Nd = static_cast<int>(m_ds.size());
      for (int id = 0; id < Nd; ++id) {
        m_uds[id] |= LBA_FLAG_TRACK_UPDATE_DEPTH | LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;
        m_Uds[id] |= LM_FLAG_TRACK_UPDATE_DEPTH;
      }
      m_SAcusLF.MakeZero();
      m_SMcusLF.MakeZero();
      m_SAcmsLF.MakeZero();
    }
#endif
#ifdef CFG_VERBOSE
    if (m_verbose) {
      if (m_verbose >= 2) {
        UT::PrintSeparator('*');
      } else if (m_iIter == 0) {
        UT::PrintSeparator();
      }
      PrintErrorStatistic(UT::String("*%2d: ", m_iIter), m_CsLF, m_CsKF, m_ds, m_DsLF,
                          m_verbose >= 2);
    }
#endif
#ifdef LBA_DEBUG_VIEW
    if (g_viewer) {
      g_viewer->Run(true, false);
    }
#endif
#ifdef LBA_DEBUG_PRINT_STEP
    UT::Print("\r[%d] UpdateFactors Start\t\t\t", iFrm);
#endif
    m_ts[TM_FACTOR].Start();
    //更新因子对于H|-b的影响
    //视觉约束 + imu约束,LBA里的KF的位姿是固定不动的,所以涉及KF位姿的雅克比都设成了0
    UpdateFactors();
    m_ts[TM_FACTOR].Stop();
#ifdef LBA_DEBUG_EIGEN
    DebugUpdateFactors();
#endif
#ifdef LBA_DEBUG_PRINT_STEP
    UT::Print("\r[%d] UpdateFactors Stop\t\t\t", iFrm);
#endif
#if 0
//#if 1
    UT::DebugStart();
    const ES ES1 = ComputeErrorStatistic(m_CsLF, m_CsKF, m_ds, m_DsLF);
    SolveSchurComplementGT(&m_xs);
    SolveBackSubstitutionGT(&m_xs);
    const ES ES2 = ComputeErrorStatistic(m_CsLF, m_xs, false);
    UT::DebugStop();
#endif
#ifdef LBA_DEBUG_PRINT_STEP
    UT::Print("\r[%d] UpdateSchurComplement Start\t\t\t", iFrm);
#endif
    m_ts[TM_SCHUR_COMPLEMENT].Start();
    UpdateSchurComplement();//要将H|-b中的地图点部分先边缘化得到[S|-g],这里在求的就是W*V^-1*W.t g 和 W*V^-1*v
    m_ts[TM_SCHUR_COMPLEMENT].Stop();
#ifdef LBA_DEBUG_EIGEN
    DebugUpdateSchurComplement();
#endif
#ifdef LBA_DEBUG_PRINT_STEP
    UT::Print("\r[%d] UpdateSchurComplement Stop\t\t\t", iFrm);
#endif

#ifdef LBA_DEBUG_PRINT_STEP
    UT::Print("\r[%d] SolveSchurComplement Start\t\t\t", iFrm);
#endif
    m_ts[TM_CAMERA].Start();
    const bool scc = SolveSchurComplement();//PCG加速S*x = -g
    m_ts[TM_CAMERA].Stop();
#ifdef LBA_DEBUG_EIGEN
    DebugSolveSchurComplement();
#endif
#ifdef LBA_DEBUG_PRINT_STEP
    UT::Print("\r[%d] SolveSchurComplement Stop\t\t\t", iFrm);
#endif
    //if (!scc) {
    //  m_update = false;
    //  m_converge = false;
    //  break;
    //}
#ifdef LBA_DEBUG_PRINT_STEP
    UT::Print("\r[%d] SolveBackSubstitution Start\t\t\t", iFrm);
#endif
    m_ts[TM_DEPTH].Start();
 //du =-Wcu.t * Huu^-1 * dxc +  Huu^-1*bu (Huu就是V,bu就是v) = Huu^-1*(bu -Wcu.t * dxc )也就求出了地图点逆深度的增量
//利用相机pose的增量反求出地图点逆深度的增量,将所有所有需要反求增量的du push进m_xsGN
    SolveBackSubstitution();
    m_ts[TM_DEPTH].Stop();
#ifdef LBA_DEBUG_EIGEN
    DebugSolveBackSubstitution();
#endif
#ifdef LBA_DEBUG_PRINT_STEP
    UT::Print("\r[%d] SolveBackSubstitution Stop\t\t\t", iFrm);
#endif

#ifdef CFG_VERBOSE
    const int N = UT::PrintStringWidth();
#endif
    m_xsGD.Resize(0);   m_x2GD = 0.0f;
    m_xsDL.Resize(0);   m_x2DL = 0.0f;
    m_rho = FLT_MAX;
    const int nItersDL = std::max(BA_DL_MAX_ITERATIONS, 1);
    for (m_iIterDL = 0; m_iIterDL < nItersDL; ++m_iIterDL) {//dogleg迭代
      if (m_x2GN > m_delta2 && m_xsGD.Empty() && BA_DL_MAX_ITERATIONS > 0) {//如果GN求出的解在可信域外,就需要求pU点
#ifdef LBA_DEBUG_PRINT_STEP
        UT::Print("\r[%d] SolveGradientDescent Start\t\t\t", iFrm);
#endif
        m_ts[TM_UPDATE].Start();
        SolveGradientDescent();//求解pU点,方向是负梯度方向,步长g.t*g/g.t*A*g
        m_ts[TM_UPDATE].Stop();
#ifdef LBA_DEBUG_EIGEN
        DebugSolveGradientDescent();
#endif
#ifdef LBA_DEBUG_PRINT_STEP
        UT::Print("\r[%d] SolveGradientDescent Stop\t\t\t", iFrm);
#endif
      }
#ifdef LBA_DEBUG_PRINT_STEP
      UT::Print("\r[%d] SolveDogLeg Start\t\t\t", iFrm);
#endif
      m_ts[TM_UPDATE].Start();
      SolveDogLeg();//判断GN求出的增量和pU点与信赖半径的关系,然后求解dogleg方法的增量
      UpdateStatesPropose();//备份状态以及更新状态(m_CsLF)和逆深度(m_ds)
      m_ts[TM_UPDATE].Stop();
#ifdef LBA_DEBUG_PRINT_STEP
      UT::Print("\r[%d] SolveDogLeg Stop\t\t\t", iFrm);
#endif
#ifdef CFG_VERBOSE
      if (m_verbose) {
        if (m_verbose >= 3) {
          UT::PrintSeparator();
        }
        const std::string str = m_verbose >= 2 ? " --> " :
                                (m_iIterDL == 0 ? " > " : std::string(N + 3, ' '));
        PrintErrorStatistic(str, m_CsLFBkp, m_CsKF, m_dsBkp, m_xsDL, m_verbose >= 2);
      }
#endif
      if (BA_DL_MAX_ITERATIONS == 0) {
#ifdef CFG_VERBOSE
        if (m_verbose) {
          UT::Print("\n");
        }
#endif
        break;
      }
#ifdef LBA_DEBUG_PRINT_STEP
      UT::Print("\r[%d] ComputeReduction Start\t\t\t", iFrm);
#endif
      m_ts[TM_UPDATE].Start();
      ComputeReduction();//计算一下所有因子的实际下降和理论下降,然后计算rho = 实际/理论
      m_ts[TM_UPDATE].Stop();
#ifdef CFG_VERBOSE
      if (m_verbose) {
        if (m_verbose >= 2) {
          UT::Print(" %f * (%e %e) <= %e %f\n", m_beta, sqrtf(m_x2GN), sqrtf(m_x2GD), sqrtf(m_delta2), m_rho);
        } else {
          UT::Print(" %.3f %.2e <= %.2e %.1f\n", m_beta, sqrtf(m_x2DL), sqrtf(m_delta2), m_rho);
        }
      }
#endif
#ifdef LBA_DEBUG_EIGEN
      DebugComputeReduction();
#endif
#ifdef LBA_DEBUG_PRINT_STEP
      UT::Print("\r[%d] ComputeReduction Stop\t\t\t", iFrm);
#endif
#ifdef LBA_DEBUG_PRINT_STEP
      UT::Print("\r[%d] UpdateStatesDecide Start\t\t\t", iFrm);
#endif
      m_ts[TM_UPDATE].Start();
      const bool accept = UpdateStatesDecide();//通过rho来判断近似效果的好坏,近似的不好需要回滚更新
      m_ts[TM_UPDATE].Stop();
#ifdef LBA_DEBUG_PRINT_STEP
      UT::Print("\r[%d] UpdateStatesDecide Stop\t\t\t", iFrm);
#endif
      if (accept) {//接受这次更新就跳出迭代
        break;
      }
    }
    if (m_iIterDL == BA_DL_MAX_ITERATIONS) {
#ifdef LBA_DEBUG_PRINT_STEP
      UT::Print("\r[%d] UpdateStatesDecide Start\t\t\t", iFrm);
#endif
      m_ts[TM_UPDATE].Start();
      UpdateStatesDecide();//通过rho来判断近似效果的好坏,近似的不好需要回滚更新
      m_ts[TM_UPDATE].Stop();
#ifdef LBA_DEBUG_PRINT_STEP
      UT::Print("\r[%d] UpdateStatesDecide Stop\t\t\t", iFrm);
#endif
    }
//#ifdef CFG_DEBUG
#if 1
    if (m_debug >= 2) {
      AssertConsistency();
    }
#endif
    if (!m_update || m_converge) {//如果没有需要更新的或者收敛了就退出更新
      break;
    }
    if (LBA_EMBEDDED_POINT_ITERATION) {
      m_ts[TM_UPDATE].Start();
      EmbeddedPointIteration(m_CsLF, m_CsKF, m_ucsKF, m_uds, &m_ds);
      m_ts[TM_UPDATE].Stop();
    }
    MT_READ_LOCK_BEGIN(m_MT, iFrm, MT_TASK_LBA_BufferDataEmpty);
    m_empty = BufferDataEmpty();
    MT_READ_LOCK_END(m_MT, iFrm, MT_TASK_LBA_BufferDataEmpty);
    if (!m_empty) {
      break;
    }
  }
#ifdef CFG_VERBOSE
  if (m_verbose) {
    if (m_verbose >= 2) {
      UT::PrintSeparator('*');
    }
    PrintErrorStatistic(UT::String("*%2d: ", m_iIter), m_CsLF, m_CsKF, m_ds, m_DsLF,
                        m_verbose >= 2);
    if (m_verbose < 2) {
      UT::Print("\n");
    }
  }
#endif
  m_ts[TM_TOTAL].Stop();
  for (int i = 0; i < TM_TYPES; ++i) {
    m_ts[i].Finish();
  }
#ifdef CFG_HISTORY//一些debug的东西
  const int iLFn = m_ic2LF.back();
  const FRM::Tag &T = m_LFs[iLFn].m_T;
  if (m_history >= 1 && (m_hists.empty() || T.m_iFrm > m_hists.back().m_iFrm)) {
    History hist;
    hist.MakeZero();
    hist.m_iFrm = T.m_iFrm;
    hist.m_C = m_CsLF[iLFn];
    //hist.m_Cam_pose = m_CsLFGT[iLFn];
    hist.m_t = T.m_t;
    for (int i = 0; i < TM_TYPES; ++i) {
      hist.m_ts[i] = m_ts[i].GetAverageSeconds() * 1000.0;
    }
    //hist.m_Nd = CountLocalTracks();
    if (m_history >= 2) {
      hist.m_ESa = ComputeErrorStatistic(m_CsLF, m_CsKF, m_ds, m_DsLF);
      hist.m_ESb = ComputeErrorStatistic(m_CsLFBkp, m_CsKF, m_dsBkp, m_DsLF);
      hist.m_ESp = ComputeErrorStatistic(m_CsLFBkp, m_CsKF, m_dsBkp, m_xsDL, false);
#ifdef CFG_GROUND_TRUTH
      if (m_CsGT) {
        SolveSchurComplementGT(m_CsLFBkp, &m_xsGT);
        if (m_dsGT) {
          if (m_history >= 3) {
            EmbeddedPointIteration(m_CsLFGT, m_CsKFGT, m_ucsKFGT, m_udsGT, m_dsGT);
          }
          SolveBackSubstitutionGT(m_dsBkp, &m_xsGT);
          hist.m_ESaGT = ComputeErrorStatistic(m_CsLFGT, m_CsKFGT, *m_dsGT, m_DsLFGT);
          hist.m_ESpGT = ComputeErrorStatistic(m_CsLFBkp, m_CsKFBkp, m_dsBkp, m_xsGT, false);
        }
      }
#endif
      const int N = m_bs.Size();
      m_xsGN.Resize(N);
      hist.m_R = ComputeResidual(m_xsGN);
#ifdef CFG_GROUND_TRUTH
      if (m_CsGT) {
        m_xsGT.Resize(N);
        hist.m_RGT = ComputeResidual(m_xsGT);
        const int iLF0 = m_ic2LF.front();
        const Camera &C = m_CsLFGT[iLF0];
        m_CsKFGT.Push(C.m_Cam_pose);
        const float eps = 0.0f;
        const float epsr = UT::Inverse(BA_VARIANCE_MAX_ROTATION, BA_WEIGHT_FEATURE, eps);
        const float epsp = UT::Inverse(BA_VARIANCE_MAX_POSITION, BA_WEIGHT_FEATURE, eps);
        const float epsv = UT::Inverse(BA_VARIANCE_MAX_VELOCITY, BA_WEIGHT_FEATURE, eps);
        const float epsba = UT::Inverse(BA_VARIANCE_MAX_BIAS_ACCELERATION, BA_WEIGHT_FEATURE, eps);
        const float epsbw = UT::Inverse(BA_VARIANCE_MAX_BIAS_GYROSCOPE, BA_WEIGHT_FEATURE, eps);
        const float _eps[] = {epsp, epsp, epsp, epsr, epsr, epsr, epsv, epsv, epsv,
                              epsba, epsba, epsba, epsbw, epsbw, epsbw};
        LA::AlignedMatrixXf S;
        LA::AlignedVectorXf x;
        float xTb;
        if (m_ZpKF.Valid()) {
          if (m_ZpKF.GetPriorMeasurement(BA_WEIGHT_FEATURE, &S, &x, &xTb, &m_work, _eps)) {
            CameraPrior::Pose::Error e;
            m_ZpKF.GetError(m_CsKFGT, &e, BA_ANGLE_EPSILON);
            hist.m_PSKF.m_F = m_ZpKF.GetCost(1.0f / BA_WEIGHT_FEATURE, e) - xTb;
            ComputePriorStatisticPose(m_ZpKF, m_CsKFGT, S, x, &hist.m_PSKF);
          } else {
            hist.m_PSKF.Invalidate();
          }
        }
        //if (m_ZpLF.Valid()) {
        if (m_Zp.Pose::Valid()) {
          if (m_Zp.GetPriorMotion(&m_ZpLF, &m_work, _eps) &&
              m_ZpLF.GetPriorMeasurement(BA_WEIGHT_FEATURE, &S, &x, &xTb, &m_work, _eps + 6)) {
            CameraPrior::Motion::Error e;
            m_ZpLF.GetError(C, &e);
            hist.m_PSLF.m_F = m_ZpLF.GetCost(1.0f / BA_WEIGHT_FEATURE, e) - xTb;
            ComputePriorStatisticMotion(m_ZpLF, C, S, x, &hist.m_PSLF);
          } else {
            hist.m_PSLF.Invalidate();
          }
        }
        if (m_Zp.Pose::Valid()) {
#ifdef CFG_DEBUG
          UT_ASSERT(m_Zp.m_iKFs.back() == INT_MAX);
#endif
          if (m_Zp.GetPriorMeasurement(BA_WEIGHT_FEATURE, &S, &x, &xTb, &m_work, _eps)) {
            CameraPrior::Joint::Error e;
            m_Zp.GetError(m_CsKFGT, C, &e, BA_ANGLE_EPSILON);
            hist.m_PS.m_F = m_Zp.GetCost(1.0f  / BA_WEIGHT_FEATURE, e) - xTb;
            ComputePriorStatisticJoint(m_Zp, m_CsKFGT, C, S, x, &hist.m_PS);
            if (hist.m_PSKF.Valid()) {
              hist.m_PSKF.SetMotion(hist.m_PS);
            }
            if (hist.m_PSLF.Valid()) {
              hist.m_PSLF.SetPose(hist.m_PS);
            }
          } else {
            hist.m_PS.Invalidate();
          }
        }
        m_CsKFGT.Resize(m_CsKFGT.Size() - 1);
      }
#endif
      if (m_Zp.Pose::Valid()) {
        hist.m_MSLP = ComputeMarginalizationStatistic();
        if (m_MH.SolveLDL(&m_work)) {
          hist.m_MSEM = ComputeMarginalizationStatistic(&m_MH.m_x);
        } else {
          hist.m_MSEM.Invalidate();
        }
#ifdef CFG_GROUND_TRUTH
        if (m_CsGT) {
          hist.m_MSGT = ComputeMarginalizationStatistic(&m_MH.m_xGT);
        } else {
          hist.m_MSGT.Invalidate();
        }
#endif
      }
    }
    m_hists.push_back(hist);
  }
#endif
#if 0
//#if 1
  UT::PrintSeparator();
  UT::Print("Time = %f\n", m_ts[TM_TOTAL].GetAverageMilliseconds() * 1000.0);
#endif
#ifdef LBA_DEBUG_VIEW
  if (g_viewer) {
    //g_viewer->m_keyPause = true;
    //g_viewer->m_keyPause = m_verbose >= 3;
    g_viewer->Run();
    //delete g_viewer;
  }
#endif
#if 0
//#if 1
  {
    FILE *fp = fopen(UT::String("D:/tmp/%04d.txt", iFrm).c_str(), "rb");
    m_CsLF.LoadB(fp);
    m_CsLFLP.LoadB(fp);
    UT::VectorLoadB(m_ucsLF, fp);
    UT::VectorLoadB(m_ucmsLF, fp);
    UT::VectorLoadB(m_UcsLF, fp);
    m_CsKF.LoadB(fp);
    m_CsKFLP.LoadB(fp);
    UT::VectorLoadB(m_ucsKF, fp);
    UT::VectorLoadB(m_UcsKF, fp);
    UT::VectorLoadB(m_ds, fp);
    UT::VectorLoadB(m_dsLP, fp);
    UT::VectorLoadB(m_uds, fp);
    UT::VectorLoadB(m_Uds, fp);
    fclose(fp);
  }
#endif
  UpdateData();//更新m_CsLF中存储的滑窗中的相机pose,更新m_CsKF中存储的相机pose,以及地图点逆深度
//#ifdef CFG_DEBUG
#if 1
  //if (!m_serial && m_debug)
  if (m_debug) {
    AssertConsistency();
  }
#endif
#if 0
//#if 1
  {
    FILE *fp = fopen(UT::String("D:/tmp/%04d.txt", iFrm).c_str(), "wb");
    m_CsLF.SaveB(fp);
    m_CsLFLP.SaveB(fp);
    UT::VectorSaveB(m_ucsLF, fp);
    UT::VectorSaveB(m_ucmsLF, fp);
    UT::VectorSaveB(m_UcsLF, fp);
    m_CsKF.SaveB(fp);
    m_CsKFLP.SaveB(fp);
    UT::VectorSaveB(m_ucsKF, fp);
    UT::VectorSaveB(m_UcsKF, fp);
    UT::VectorSaveB(m_ds, fp);
    UT::VectorSaveB(m_dsLP, fp);
    UT::VectorSaveB(m_uds, fp);
    UT::VectorSaveB(m_Uds, fp);
    fclose(fp);
  }
#endif
//#ifdef CFG_DEBUG
#if 0
  if (m_KFs.size() == 2) {
    UT::PrintSeparator();
    m_CsKF[0].Print(true);
    UT::PrintSeparator();
    m_CsKF[1].Print(true);
  }
#endif
#ifdef LBA_DEBUG_PRINT
  {
    Camera Cc;
    Rigid3D Ck;
    Camera::Factor::Unitary::CC A;
    Camera::Factor::Binary::CC M;
    double Su, Ss2, Sadd, SmddST;
    UT::PrintSeparator('*');
    for (int i = 0; i < 2; ++i) {
      if (i == 0) {
        const int Nc = static_cast<int>(m_LFs.size());
        const int ic = Nc - 1, iLF = m_ic2LF[ic];
        const LocalFrame &LF = m_LFs[iLF];
        const int iKFNearest = LF.m_iKFNearest == -1 ? LF.m_iKFsMatch.front() : LF.m_iKFNearest;
        UT::Print("[%d] <-- [%d]\n", LF.m_Cam_pose.m_iFrm, m_KFs[iKFNearest].m_Cam_pose.m_iFrm);
        Cc = m_CsLF[iLF];
        Ck.MakeIdentity();
        const int NkKF = static_cast<int>(LF.m_iKFsMatch.size());
        for (int ik = 0; ik < NkKF; ++ik)
          Ck = m_CsKF[LF.m_iKFsMatch[ik]] * Ck;
        A = m_SAcusLF[iLF] - m_SMcusLF[iLF];
        M.MakeZero();
        const int NkLF = std::min(Nc, LBA_MAX_SLIDING_TRACK_LENGTH) - 1;
        xp128f sm; sm.vdup_all_lane(1.0f / NkLF);
        for (int ik = 0, _ic = ic - 1; ik < NkLF; ++ik, --_ic) {
          const int _iLF = m_ic2LF[_ic];
          const LocalFrame &_LF = m_LFs[_iLF];
#ifdef CFG_DEBUG
          UT_ASSERT(_LF.m_iLFsMatch.back() == iLF);
#endif
          M += _LF.m_Zm.m_SMczms.Back() * sm;
        }
        Su = Ss2 = Sadd = SmddST = 0.0;
        const int NZ = static_cast<int>(LF.m_Zs.size());
        for (int iZ = 0; iZ < NZ; ++iZ) {
          const FRM::Measurement &Z = LF.m_Zs[iZ];
          const KeyFrame &KF = m_KFs[Z.m_iKF];
          const Depth::InverseGaussian *ds = m_ds.data() + KF.m_id;
          for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
            const int ix = LF.m_zs[iz].m_L_idx;
            const Depth::InverseGaussian &d = ds[ix];
            Su = d.u() + Su;
            Ss2 = d.s2() + Ss2;
            Sadd = KF.m_Axps[ix].m_Sadd.m_a + Sadd;
            SmddST = LF.m_SmddsST[iz].m_a + SmddST;
#ifdef CFG_DEBUG
            UT::Print("%d %.10e %.10e\n", iz, d.u(), Su);
#endif
//#ifdef CFG_DEBUG
#if 0
            UT::Print("%d %.10e %.10e\n", iz, SmddST, LF.m_SmddsST[iz].m_a);
#endif
          }
        }
      } else {
        Cc.MakeIdentity();
        Ck.MakeIdentity();
        A.MakeZero();
        M.MakeZero();
        Su = Ss2 = Sadd = SmddST = 0.0;
        const xp128f sc = xp128f::get(1.0f / nLFs);
        const xp128f sm = xp128f::get(1.0f / (CountSchurComplements() - nLFs));
        for (int iLF = 0; iLF < nLFs; ++iLF) {
          const Camera &Cam_state = m_CsLF[iLF];
          Cc.m_Cam_pose = Cam_state.m_Cam_pose * Cc.m_Cam_pose;
          Cc.m_v += Cam_state.m_v * sc;
          Cc.m_ba += Cam_state.m_ba * sc;
          Cc.m_bw += Cam_state.m_bw * sc;
          A += (m_SAcusLF[iLF] - m_SMcusLF[iLF]) * sc;
//#ifdef CFG_DEBUG
#if 0
          UT::Print("%d %.10e\n", iLF, A.m_A.m00());
#endif
          const LocalFrame &LF = m_LFs[iLF];
          const int Nk = static_cast<int>(LF.m_iLFsMatch.size());
          for (int ik = 0; ik < Nk; ++ik) {
            M += LF.m_Zm.m_SMczms[ik] * sm;
          }
        }
        const int nKFs = static_cast<int>(m_KFs.size());
        for (int iKF = 0; iKF < nKFs; ++iKF) {
          Ck = m_CsKF[iKF] * Ck;
          const KeyFrame &KF = m_KFs[iKF];
          const Depth::InverseGaussian *ds = m_ds.data() + KF.m_id;
          const int Nx = static_cast<int>(KF.m_xs.size());
          for (int ix = 0; ix < Nx; ++ix) {
            const Depth::InverseGaussian &d = ds[ix];
            Su = d.u() + Su;
            Ss2 = d.s2() + Ss2;
            Sadd = KF.m_Axps[ix].m_Sadd.m_a + Sadd;
          }
        }
        for (int iLF = 0; iLF < nLFs; ++iLF) {
          const LocalFrame &LF = m_LFs[iLF];
          const int Nz = static_cast<int>(LF.m_zs.size());
          for (int iz = 0; iz < Nz; ++iz) {
            SmddST = LF.m_SmddsST[iz].m_a + SmddST;
          }
        }
      }
      UT::Print("Cam_state:\n");
      Cc.m_Cam_pose.Print(true);
      Cc.m_v.Print("", true, false);
      Cc.m_ba.Print(" ", true, false);
      Cc.m_bw.Print(" ", true, true);
      Ck.Print(true);
      UT::Print("A:\n");
      A.Print(true);
      M.Print(true);
      UT::Print("d: %.10e %.10e %.10e %.10e\n", Su, Ss2, Sadd, SmddST);
    }
    UT::PrintSeparator('*');
  }
  const float xGN = m_xsGN.Mean(), s2GN = m_xsGN.Variance(xGN);
  const float xGD = m_xsGD.Mean(), s2GD = m_xsGD.Variance(xGD);
  const float xDL = m_xsDL.Mean(), s2DL = m_xsDL.Variance(xDL);
  UT::Print("x: %e %e %e %e %e %e\n", xGN, xGD, xDL, s2GN, s2GD, s2DL);
#endif
//#ifndef WIN32
#if 0
  UT::Print("[%d]\n", iFrm);
#endif
//#ifdef CFG_DEBUG
#if 0
  UT::Print("%.10e\n", m_SAcusLF[m_ic2LF.back()].m_b.v0());
#endif
//#ifdef CFG_DEBUG
#if 0
  {
    const LocalFrame &LF = m_LFs[m_ic2LF.back()];
    const int NZ = static_cast<int>(LF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ) {
      const FRM::Measurement &Z = LF.m_Zs[iZ];
      const Depth::InverseGaussian *ds = m_ds.data() + m_KFs[Z.m_iKF].m_id;
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
        const int ix = LF.m_zs[iz].m_L_idx;
        UT::Print("%d (%d %d) %.10e\n", iz, Z.m_iKF, ix, ds[ix].u());
      }
    }
  }
#endif
//#ifdef CFG_DEBUG
#if 0
  const LA::AlignedVector3f &v = m_CsLF[m_ic2LF.back()].m_v;
  //const LA::AlignedVector3f &v = m_CsLF[m_ic2LF.back()].m_bw;
  UT::Print("[%d] %.10e %.10e %.10e\n", iFrm, v.x(), v.y(), v.z());
#endif
//#ifdef CFG_DEBUG
#if 0
  for (int iLF = 0; iLF < nLFs; ++iLF) {
    const int _iFrm = m_LFs[iLF].m_Cam_pose.m_iFrm;
    //if (_iFrm == 1500) {
    //  UT::Print("[%d] %.10e\n", iFrm, m_SAcusLF[iLF].m_b.v4());
    if (_iFrm == 1506)
      UT::Print("[%d] %.10e\n", iFrm, m_DsLF[iLF].m_Jpbw.m00());
  }
#endif

  // Trigger LBA callback function if set
  if (m_callback) {
    //UT::Print("Trigger LBA callback [iFrm, ts]  = [%d, %f]\n", iFrm,
    //          m_LFs[m_ic2LF.back()].m_Cam_pose.m_t);
    m_callback(iFrm/*最新一帧的id*/, m_LFs[m_ic2LF.back()].m_T.m_t/*最新一帧的时间戳*/);
  }
//#ifdef CFG_DEBUG
#if 0
  static int g_iFrm = -1, g_cnt = 0;
  if (iFrm == g_iFrm) {
    ++g_cnt;
  } else {
    g_iFrm = iFrm;
    g_cnt = 0;
  }
  const std::string suffix = g_cnt == 0 ? "" : UT::String("_%d", g_cnt);
#ifdef WIN32
  const std::string fileName = UT::String("D:/tmp/test/cons/%04d%s.txt", iFrm, suffix.c_str());
#else
  const std::string fileName = UT::String("/tmp/test/%04d%s.txt", iFrm, suffix.c_str());
#endif
#if 0
//#if 1
  UT::FilesStartSaving(fileName);
  FILE *fp = fopen(fileName.c_str(), "wb");
  if (fp) {
    m_SAcusLF.SaveB(fp);
    m_SMcusLF.SaveB(fp);
    m_SAcmsLF.SaveB(fp);
    m_CsKF.SaveB(fp);
    m_CsLF.SaveB(fp);
    UT::VectorSaveB(m_ds, fp);
    UT::VectorSaveB(m_uds, fp);
    fclose(fp);
    UT::PrintSaved(fileName);
  }
#else
  FILE *fp = fopen(fileName.c_str(), "rb");
  if (fp) {
    AlignedVector<Camera::Factor::Unitary::CC> SAcusLF, SMcusLF;
    AlignedVector<Camera::Factor> SAcmsLF;
    SAcusLF.LoadB(fp);
    SMcusLF.LoadB(fp);
    SAcmsLF.LoadB(fp);
    const int nLFs = static_cast<int>(m_LFs.size());
    UT_ASSERT(SAcusLF.Size() == nLFs && SMcusLF.Size() == nLFs && SAcmsLF.Size() == nLFs);
    for (int iLF = 0; iLF < nLFs; ++iLF) {
      m_SAcusLF[iLF].AssertEqual(SAcusLF[iLF], 1, UT::String("SAcusLF[%d]", iLF), -1.0f, -1.0f);
      m_SMcusLF[iLF].AssertEqual(SMcusLF[iLF], 1, UT::String("SMcusLF[%d]", iLF), -1.0f, -1.0f);
      m_SAcmsLF[iLF].AssertEqual(SAcmsLF[iLF], 1, UT::String("SAcmsLF[%d]", iLF), -1.0f, -1.0f);
    }
    AlignedVector<Rigid3D> CsKF;
    CsKF.LoadB(fp);
    const int nKFs = static_cast<int>(m_KFs.size());
    UT_ASSERT(CsKF.Size() == nKFs);
    for (int iKF = 0; iKF < nKFs; ++iKF) {
      m_CsKF[iKF].AssertEqual(CsKF[iKF], 1, UT::String("CsKF[%d]", iKF), -1.0f, -1.0f);
    }
    AlignedVector<Camera> CsLF;
    CsLF.LoadB(fp);
    UT_ASSERT(CsLF.Size() == nLFs);
    for (int iLF = 0; iLF < nLFs; ++iLF) {
      m_CsLF[iLF].AssertEqual(CsLF[iLF], 1, UT::String("CsLF[%d]", iLF), -1.0f, -1.0f);
    }
    std::vector<Depth::InverseGaussian> ds;
    std::vector<ubyte> uds;
    UT::VectorLoadB(ds, fp);
    UT::VectorLoadB(uds, fp);
    const int Nd = static_cast<int>(m_ds.size());
    UT_ASSERT(static_cast<int>(ds.size()) == Nd);
    for (int id = 0; id < Nd; ++id) {
      m_ds[id].AssertEqual(ds[id], 1, UT::String("ds[%d]", id), -1.0f);
      UT_ASSERT(m_uds[id] == uds[id]);
    }
    fclose(fp);
    UT::PrintLoaded(fileName);
  }
#endif
#endif
}

void LocalBundleAdjustor::SetCallback(const IBA::Solver::IbaCallback& iba_callback) {
  m_callback = iba_callback;
}

//int LocalBundleAdjustor::GetTotalPoints(int *N) {
//  int SN = 0;
//  const int _N = static_cast<int>(m_hists.size());
//  for (int i = 0; i < _N; ++i) {
//    SN += m_hists[i].m_Nd;
//  }
//  if (N) {
//    *N = _N;
//  }
//  return SN;
//}

float LocalBundleAdjustor::GetTotalTime(int *N) {
#ifdef CFG_HISTORY
  const int _N = static_cast<int>(m_hists.size());
  m_work.Resize(_N);
  LA::AlignedVectorXf ts(m_work.Data(), _N, false);
  for (int i = 0; i < _N; ++i) {
    ts[i] = static_cast<float>(m_hists[i].m_ts[TM_TOTAL]);
  }
  if (N) {
    *N = _N;
  }
  return ts.Sum();
#else
  if (N) {
    *N = 0;
  }
  return 0.0f;
#endif
}

bool LocalBundleAdjustor::SaveTimes(const std::string fileName) {
  FILE *fp = fopen(fileName.c_str(), "w");
  if (!fp) {
    return false;
  }
#ifdef CFG_HISTORY
  const int N = static_cast<int>(m_hists.size());
  for (int i = 0; i < N; ++i) {
    const double *ts = m_hists[i].m_ts;
    for (int j = 0; j < TM_TYPES; ++j) {
      fprintf(fp, "%f ", ts[j]);
    }
    fprintf(fp, "\n");
  }
#endif
  fclose(fp);
  UT::PrintSaved(fileName);
  return true;
}

bool LocalBundleAdjustor::SaveCameras(const std::string fileName, const bool poseOnly) {
  FILE *fp = fopen(fileName.c_str(), "w");
  if (!fp) {
    return false;
  }
#ifdef CFG_HISTORY
  Point3D p;
  Quaternion q;
  Rotation3D R;
  LA::AlignedVector3f ba, bw;
  const Rotation3D RuT = m_K.m_Ru.GetTranspose();
  const int N = static_cast<int>(m_hists.size());
  for (int i = 0; i < N; ++i) {
    const History &hist = m_hists[i];
    const Rigid3D &T = hist.m_C.m_Cam_pose;
    p = hist.m_C.m_p + T.GetAppliedRotationInversely(m_K.m_pu);
    Rotation3D::AB(RuT, T, R);
    R.GetQuaternion(q);
    fprintf(fp, "%f %f %f %f %f %f %f %f", hist.m_t, p.x(), p.y(), p.z(),
                                           q.x(), q.y(), q.z(), q.w());
    if (!poseOnly) {
      RuT.Apply(hist.m_C.m_ba, ba);
      RuT.Apply(hist.m_C.m_bw, bw);
      ba += m_K.m_ba;
      bw += m_K.m_bw;
      const LA::AlignedVector3f &v = hist.m_C.m_v;
      fprintf(fp, " %f %f %f %f %f %f %f %f %f", v.x(), v.y(), v.z(),
                                                 ba.x(), ba.y(), ba.z(),
                                                 bw.x(), bw.y(), bw.z());
    }
    fprintf(fp, "\n");
  }
#endif
  fclose(fp);
  UT::PrintSaved(fileName);
  return true;
}

bool LocalBundleAdjustor::SaveCosts(const std::string fileName, const int type) {
  FILE *fp = fopen(fileName.c_str(), "w");
  if (!fp) {
    return false;
  }
#ifdef CFG_HISTORY
  const int N = static_cast<int>(m_hists.size());
  for (int i = 0; i < N; ++i) {
    const History &hist = m_hists[i];
    switch (type) {
    case 0: hist.m_ESa.Save(fp);    break;
    case 1: hist.m_ESb.Save(fp);    break;
    case 2: hist.m_ESp.Save(fp);    break;
#ifdef CFG_GROUND_TRUTH
    case 3: hist.m_ESaGT.Save(fp);  break;
    case 4: hist.m_ESpGT.Save(fp);  break;
#endif
    }
  }
#endif
  fclose(fp);
  UT::PrintSaved(fileName);
  return true;
}

bool LocalBundleAdjustor::SaveResiduals(const std::string fileName, const int type) {
  FILE *fp = fopen(fileName.c_str(), "w");
  if (!fp) {
    return false;
  }
#ifdef CFG_HISTORY
  const int N = static_cast<int>(m_hists.size());
  for (int i = 0; i < N; ++i) {
    const History &hist = m_hists[i];
    switch (type) {
    case 0: hist.m_R.Save(fp);    break;
#ifdef CFG_GROUND_TRUTH
    case 1: hist.m_RGT.Save(fp);  break;
#endif
    }
  }
#endif
  fclose(fp);
  UT::PrintSaved(fileName);
  return true;
}

bool LocalBundleAdjustor::SavePriors(const std::string fileName, const int type) {
  FILE *fp = fopen(fileName.c_str(), "w");
  if (!fp) {
    return false;
  }
#ifdef CFG_HISTORY
  const int N = static_cast<int>(m_hists.size());
  for (int i = 0; i < N; ++i) {
    const History &hist = m_hists[i];
    switch (type) {
#ifdef CFG_GROUND_TRUTH
    case 0: hist.m_PS.Save(fp);   break;
    case 1: hist.m_PSKF.Save(fp); break;
    case 2: hist.m_PSLF.Save(fp); break;
#endif
    }
  }
#endif
  fclose(fp);
  UT::PrintSaved(fileName);
  return true;
}

bool LocalBundleAdjustor::SaveMarginalizations(const std::string fileName, const int type) {
  FILE *fp = fopen(fileName.c_str(), "w");
  if (!fp) {
    return false;
  }
#ifdef CFG_HISTORY
  const int N = static_cast<int>(m_hists.size());
  for (int i = 0; i < N; ++i) {
    const History &hist = m_hists[i];
    switch (type) {
    case 0: hist.m_MSLP.Save(fp); break;
    case 1: hist.m_MSEM.Save(fp); break;
#ifdef CFG_GROUND_TRUTH
    case 2: hist.m_MSGT.Save(fp); break;
#endif
    }
  }
#endif
  fclose(fp);
  UT::PrintSaved(fileName);
  return true;
}

void LocalBundleAdjustor::ComputeErrorFeature(float *ex) {
  float exi;
  *ex = 0.0f;
  const int nLFs = static_cast<int>(m_LFs.size());
  for (int iLF = 0; iLF < nLFs; ++iLF) {
    ComputeErrorFeature(&m_LFs[iLF], m_CsLF[iLF].m_Cam_pose, m_CsKF, m_ds, &exi);
    *ex = std::max(exi, *ex);
  }
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    ComputeErrorFeature(&m_KFs[iKF], m_CsKF[iKF], m_CsKF, m_ds, &exi, iKF);
    *ex = std::max(exi, *ex);
  }
}
//计算一下重投影误差
void LocalBundleAdjustor::ComputeErrorFeature(const FRM::Frame *F/*当前帧*/, const Rigid3D &C,//当前帧Tc0w
                                              const AlignedVector<Rigid3D> &CsKF,//所有关键帧的位姿
                                              const std::vector<Depth::InverseGaussian> &ds,//逆深度信息
                                              float *ex, const int iKF) {
  Rigid3D Tr[2];
  FTR::Error e;
//#if 0
#if 1
  float Se2 = 0.0f;//地图点首次被左目的观测和当前左目的归一化的残差欧式距离和
#ifdef CFG_STEREO
  float Se2r = 0.0f;//地图点首次被左目的观测和当前右目的归一化的残差欧式距离和
#endif
  int SN = 0;
#else
  const int Nz = static_cast<int>(F->m_zs.size());
#ifdef CFG_STEREO
  m_work.Resize(Nz * 2);
  LA::AlignedVectorXf e2s(m_work.Data(), Nz, false);
  LA::AlignedVectorXf e2rs(e2s.BindNext(), Nz, false);
  e2s.Resize(0);
  e2rs.Resize(0);
#else
  m_work.Resize(Nz);
  LA::AlignedVectorXf e2s(m_work.Data(), Nz, false);
  e2s.Resize(0);
#endif
#endif
  const int NZ = static_cast<int>(F->m_Zs.size());
  for (int iZ = 0; iZ < NZ; ++iZ) {//遍历当前帧对于关键帧的观测
    const FRM::Measurement &Z = F->m_Zs[iZ];
    *Tr = C / CsKF[Z.m_iKF];// Tc0(当前帧)_c0(关键帧)=Tc0w(当前帧)*Tc0w(关键帧).inv
#ifdef CFG_STEREO
    Tr[1] = Tr[0];
    Tr[1].SetTranslation(m_K.m_br + Tr[0].GetTranslation());/*Tc1(当前帧)_c0(关键帧)*/
#endif
    const Depth::InverseGaussian *_ds = ds.data() + m_iKF2d[Z.m_iKF];
    const KeyFrame &KF = m_KFs[Z.m_iKF];
    for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
      const FTR::Measurement &z = F->m_zs[iz];//当前帧对这个地图点的观测
      const int ix = z.m_ix;//这个地图点的局部id
      FTR::GetError(Tr/*Tc0(当前帧)_c0(关键帧)*/, KF.m_xs[ix]/*地图点在关键帧中的观测*/, _ds[ix]/*地图点的逆深度*/, z/*当前帧对这个地图点的观测*/, e);
#ifdef CFG_STEREO
      if (z.m_z.Valid()) {
        const float e2 = e.m_e.SquaredLength();
#if 1
//#if 0
        Se2 = e2 + Se2;
        ++SN;
#else
        e2s.Push(e2);
#endif
      }
      if (z.m_zr.Valid()) {
        const float e2r = e.m_er.SquaredLength();
#if 1
//#if 0
        Se2r = e2r + Se2r;
        ++SN;
#else
        e2rs.Push(e2r);
#endif
      }
#else
      const float e2 = e.m_e.SquaredLength();
#if 1
//#if 0
      Se2 = e2 * m_K.m_K.fxy() + Se2;
      ++SN;
#else
      e2s.Push(e2);
#endif
#endif
    }
  }
#ifdef CFG_STEREO
  if (iKF != -1) {//说明输入的是F是关键帧,所以需要算一下新地图点的
    const Depth::InverseGaussian *_ds = ds.data() + m_iKF2d[iKF];//当前关键帧所能看到的第一个新的地图点的指针
    const KeyFrame &KF = *((KeyFrame *) F);
    const int Nx = static_cast<int>(KF.m_xs.size());
    for (int ix = 0; ix < Nx; ++ix) {//遍历所有新地图点,左右目观测都有的计算重投影误差
      if (KF.m_xs[ix].m_xr.Invalid()) {
        continue;
      }
        // 投影 Pc0 Pc1代表相机坐标下的特征点 Pnc0 和 Pnc1代表了归一化以后的点,齐次转换就省略了 Uc0,Uc1表示两个坐标系下的逆深度
        //m_er= 归一化(Pnc0 - Uc0 * tc0c1) - 归一化(Rc0c1 *Pnc1) //具体看Project注释
      FTR::GetError(m_K.m_br/*-tc0_c1*/, _ds[ix]/*当前MP对应的c0深度*/, KF.m_xs[ix]/*MP对应的观测信息*/, e.m_er/*重投影误差*/);
      const float e2r = e.m_er.SquaredLength();
#if 1
//#if 0
      Se2r = e2r + Se2r;
      ++SN;
#else
      e2rs.Push(e2r);
#endif
    }
  }
#endif
//#if 0
#if 1
#ifdef CFG_STEREO
  *ex = SN == 0 ? 0.0f : sqrtf((Se2 * m_K.m_K.fxy() + Se2r * m_K.m_Kr.fxy()) / SN);//转到像素量程的残差
#else
  *ex = SN == 0 ? 0.0f : sqrtf(Se2 * m_K.m_K.fxy() / SN);
#endif
#else
  if (!e2s.Empty()) {
    const int ith = e2s.Size() >> 1;
    std::nth_element(e2s.Data(), e2s.Data() + ith, e2s.End());
    *ex = sqrtf(e2s[ith] * m_K.m_K.fxy());
  } else
    *ex = 0.0f;
#ifdef CFG_STEREO
  if (!e2rs.Empty()) {
    const int ith = e2rs.Size() >> 1;
    std::nth_element(e2rs.Data(), e2rs.Data() + ith, e2rs.End());
    *ex = std::max(sqrtf(e2rs[ith] * m_K.m_K.fxy()), *ex);
  }
#endif
#endif
}

void LocalBundleAdjustor::ComputeErrorIMU(float *er, float *ep, float *ev,
                                          float *eba, float *ebw) {
  *er = *ep = *ev = *eba = *ebw = 0.0f;
  const int nLFs = static_cast<int>(m_LFs.size());
  for (int ic1 = 0, ic2 = 1; ic2 < nLFs; ic1 = ic2++) {
    const int iLF1 = m_ic2LF[ic1], iLF2 = m_ic2LF[ic2];
    const IMU::Delta::Error e = m_DsLF[iLF2].GetError(m_CsLF[iLF1], m_CsLF[iLF2], m_K.m_pu,
                                                      BA_ANGLE_EPSILON);
    *er = std::max(e.m_er.SquaredLength(), *er);
    *ep = std::max(e.m_ep.SquaredLength(), *ep);
    *ev = std::max(e.m_ev.SquaredLength(), *ev);
    *eba = std::max(e.m_eba.SquaredLength(), *eba);
    *ebw = std::max(e.m_ebw.SquaredLength(), *ebw);
  }
  *er *= UT_FACTOR_RAD_TO_DEG;
  *ebw *= UT_FACTOR_RAD_TO_DEG;
}

void LocalBundleAdjustor::ComputeErrorDrift(float *er, float *ep) {
  *er = 0.0f;
  *ep = 0.0f;
#ifdef CFG_GROUND_TRUTH
  if (m_CsLFGT.Empty() || m_CsKFGT.Empty()) {
    return;
  }
  Rigid3D Tr, TrGT, Te;
  LA::AlignedVector3f _er, _ep;
  const int nLFs = static_cast<int>(m_LFs.size());
  for (int iLF = 0; iLF < nLFs; ++iLF) {
    const int iKFNearest = m_LFs[iLF].m_iKFNearest;
    Tr = m_CsLF[iLF].m_Cam_pose / m_CsKF[iKFNearest];
    TrGT = m_CsLFGT[iLF].m_Cam_pose / m_CsKFGT[iKFNearest];
    Te = Tr / TrGT;
    Te.GetRodrigues(_er, BA_ANGLE_EPSILON);
    Te.GetPosition(_ep);
    *er = std::max(_er.SquaredLength(), *er);
    *ep = std::max(_ep.SquaredLength(), *ep);
  }
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    const int iKFNearest = m_KFs[iKF].m_iKFNearest;
    if (iKFNearest == -1) {
      continue;
    }
    Tr = m_CsKF[iKF] / m_CsKF[iKFNearest];
    TrGT = m_CsKFGT[iKF] / m_CsKFGT[iKFNearest];
    Te = Tr / TrGT;
    Te.GetRodrigues(_er, BA_ANGLE_EPSILON);
    Te.GetPosition(_ep);
    *er = std::max(_er.SquaredLength(), *er);
    *ep = std::max(_ep.SquaredLength(), *ep);
  }
  *er *= UT_FACTOR_RAD_TO_DEG;
#endif
}

float LocalBundleAdjustor::ComputeRMSE() {
  float Se2 = 0.0f;
  const int N = static_cast<int>(m_hists.size());
#ifdef CFG_GROUND_TRUTH
  if (m_CsGT) {
    for (int i = 0; i < N; ++i) {
      const History &hist = m_hists[i];
      const Point3D p1 = hist.m_C.m_p, p2 = m_CsGT[hist.m_iFrm].m_p;
      Se2 += (p1 - p2).SquaredLength();
    }
  }
#endif
  return sqrtf(Se2 / N);
}

float LocalBundleAdjustor::GetTotalDistance() {
  float S = 0.0f;
#ifdef CFG_HISTORY
  const int N = static_cast<int>(m_hists.size());
#ifdef CFG_GROUND_TRUTH
  if (m_CsGT) {
    for (int i1 = 0, i2 = 1; i2 < N; i1 = i2++) {
      S += sqrtf((m_CsGT[m_hists[i1].m_iFrm].m_p -
                  m_CsGT[m_hists[i2].m_iFrm].m_p).SquaredLength());
//#ifdef CFG_DEBUG
#if 0
      UT::Print("[%d, %d] %f\n", i1, i2, S);
#endif
    }
  } else
#endif
  {
    for (int i1 = 0, i2 = 1; i2 < N; i1 = i2++) {
      S += sqrtf((m_hists[i1].m_C.m_p - m_hists[i2].m_C.m_p).SquaredLength());
    }
  }
#endif
  return S;
}
//同步数据
//step1:先进行一下前后数据的swap
//step2:遍历所有的输入普通帧(只有一个)根据当前imu测量,用上一帧的pose积分出当前帧的pose,注意存的pose都是Tc0w,位置单独存的时候是twc0.
// 用的JPL的四元数,所以是反着的,可以看Indirect Kalman Filter for 3D Attitude Estimation
//step3:如果当前帧同时还是关键帧的话,会先把当前输入关键帧里的有双目观测的新地图点进行三角化,初始化深度和协方差.
//step4:遍历m_ITs2,这个就是个记录器,存储了输入LBA的东西以及GBA的东西,(关键帧时就是输入普通帧 + 输入关键帧)
//step4.1:当输入是普通帧时
//step4.1.1:先初始化一下这帧的地图点平均深度(用上一帧的平均深度).然后再初始化一下地图点(利用和地图点所在关键帧的相对位姿)的深度，
// 如果同时是关键帧的话,还要对新的地图点也更新深度,然后再重新算一下这帧的地图点平均深度和协方差
//step4.1.2:对普通帧进行共视关键帧的关联,包括在共视关键帧的共视关键帧去寻找可能的共视,然后从这些共视里找出图像运动最小的,作为它的最近关键帧
//step4.1.3:向LBA加进普通帧帧信息,滑窗处理(如果最老的帧是关键帧merge了,或者参考关键帧变了,都会给GBA里对应关键帧先验motion,和先验pose,同时每次滑窗会给LBA最老帧
// 运动先验),与前5帧的共视关联,生成新的子轨迹,进行预积分的协方差,状态变化,对i时刻ba,bw雅克比等
//step4.2:当输入还有关键帧时,去看这个函数我的注释吧
//step4.3:当有关键帧时需要进行全局BA优化,唤醒GBA
void LocalBundleAdjustor::SynchronizeData()
{
#ifdef CFG_DEBUG
  UT_ASSERT(m_ITs2.empty());
#endif
  const int iFrm = m_ic2LF.empty() ? MT_TASK_NONE : m_LFs[m_ic2LF.back()].m_T.m_iFrm;
  MT_WRITE_LOCK_BEGIN(m_MT, iFrm, MT_TASK_LBA_SynchronizeData);//写锁
  m_ITs1.swap(m_ITs2);//1的东西暂时存到2
  m_ILFs1.swap(m_ILFs2);//
  m_IKFs1.swap(m_IKFs2);//关键帧
  m_IDKFs1.swap(m_IDKFs2);//
  m_IDMPs1.swap(m_IDMPs2);
  m_IUCs1.swap(m_IUCs2);
  MT_WRITE_LOCK_END(m_MT, iFrm, MT_TASK_LBA_SynchronizeData);
  m_UcsLF.assign(m_LFs.size(), LM_FLAG_FRAME_DEFAULT);//初始化一下相关flags
  m_UcsKF.assign(m_KFs.size(), LM_FLAG_FRAME_DEFAULT);
  m_Uds.assign(m_ds.size(), LM_FLAG_TRACK_DEFAULT);
#if defined CFG_GROUND_TRUTH && defined CFG_HISTORY
  if (m_history >= 3) {//和真值有关的，应该是用来debug的吧
    m_ucsKFGT.assign(m_KFs.size(), LBA_FLAG_FRAME_DEFAULT);
    m_udsGT.assign(m_ds.size(), LBA_FLAG_TRACK_DEFAULT);
  }
#endif//遍历所有当前输入的普通帧
  for (std::list<InputLocalFrame>::iterator ILF = m_ILFs2.begin(); ILF != m_ILFs2.end(); ++ILF)
  {
    if (ILF->m_Cam_state.m_Cam_pose.Valid() && ILF->m_Cam_state.m_v.Valid())
    {
      continue;
    }//从第二帧开始,用imu预积分出状态量
    Camera C;//如果有imu数据(已经被转到了左相机坐标系(Rc0_i*)下的观测),就用加速度数据求出东北天坐标系下左相机的一个初始朝向,其他的p,v,bias初始化为0
    if (m_LFs.empty())
    {//说明是第一帧
      IMU::InitializeCamera(ILF->m_imu_measures/*imu的原始测量,但是测量值是转到了左相机坐标系(Rc0_i*)下的*/, C);
#ifdef LBA_DEBUG_GROUND_TRUTH_MEASUREMENT
//#if 0
      if (m_CsGT) {
        const LA::AlignedVector3f g = m_CsGT[ILF->m_Cam_pose.m_iFrm].m_Cam_pose.GetGravity();
        Cam_state.MakeIdentity(&g);
      }
#endif
    } else
    {
        float _t;
        Camera _C;
        const IMU::Measurement *_u;
        if (ILF == m_ILFs2.begin())
        {//只可能是这种情况,一次输入一个关键帧
            const int iLF = m_ic2LF.back();//前一帧的局部帧id
            const LocalFrame &LF = m_LFs[iLF];//前一帧
            _t = LF.m_T.m_t;//上一帧的时间戳
            _C = m_CsLF[iLF];//上一帧的pose
            _u = &LF.m_us.Back();//上一帧最后一个imu测量
        } else
        {
            std::list<InputLocalFrame>::const_iterator _ILF = ILF;
            --_ILF;
            _t = _ILF->m_T.m_t;
            _C = _ILF->m_Cam_state;
            _u = &_ILF->m_imu_measures.Back();
        }
        if (LBA_PROPAGATE_CAMERA)
        {//imu预积分一下
            IMU::Delta D;//预积分pvR，存储在D中,相对于i时刻的c0坐标系的增量pvR
            IMU::PreIntegrate(ILF->m_imu_measures/*当前帧和之前帧之间的imu测量*/, _t/*上一帧的时间戳*/,
                              ILF->m_T.m_t/*当前帧的时间戳*/, _C/*上一帧的状态*/, &D/*预积分部分*/, &m_work,
                              false/*是否要输出雅克比*/, _u/*上一帧最后一个imu测量*/, NULL, BA_ANGLE_EPSILON);
            IMU::Propagate(m_K.m_pu/*外参tc0_i*/, D/*预积分部分*/, _C/*上一帧的状态*/, C/*当前帧的状态*/, BA_ANGLE_EPSILON);//传播
        } else
        {
            if(m_LFs.size() > 1)
            {
//                const int pre_preLF = m_ic2LF[m_ic2LF.size()-2];//前前帧的局部帧id
//                Camera pp_C = m_CsLF[pre_preLF];//上一帧的pose
//                Rigid3D uniform_pose =  _C.m_Cam_pose / pp_C.m_Cam_pose; //Tc1w * Tc0w.inv = Tc1c0 = Tc2c1
              //solve pnp
              //
                C = _C;
//                C.m_Cam_pose = uniform_pose * _C.m_Cam_pose;  //Tc2w = Tc2c1 * Tc1w
            }
            else
            {
                C = _C;
            }
        }
    }
    if (ILF->m_Cam_state.m_Cam_pose.Invalid()) {
      ILF->m_Cam_state.m_Cam_pose = C.m_Cam_pose;//Tc0w
      ILF->m_Cam_state.m_p = C.m_p;//twc0
    }
    if (ILF->m_Cam_state.m_v.Invalid()) {
      ILF->m_Cam_state.m_v = C.m_v;//v_w
      ILF->m_Cam_state.m_ba = C.m_ba;
      ILF->m_Cam_state.m_bw = C.m_bw;
    }
  }
  const bool newKF = !m_IKFs2.empty()/*是否有新关键帧?*/, delKF = !m_IDKFs2.empty(), updCams = !m_IUCs2.empty() ;
  bool serialGBA = false;
  if (newKF)
  {//如果有关键帧的话
    const float w = 1.0f;
    //const float gyr = BA_WEIGHT_FEATURE;
    m_CsKFBkp.Set(m_CsKF);//备份一下关键帧的位姿
    std::list<int>::const_iterator IDK = m_IDKFs2.begin();
    std::list<GlobalMap::InputKeyFrame>::iterator IKF = m_IKFs2.begin();//当前输入关键帧
    for (std::list<InputType>::const_iterator IT = m_ITs2.begin()/*输入帧的类型(无关键帧时就是当前帧，有关键帧时就是当前帧+关键帧)*/;
    IT != m_ITs2.end(); ++IT)
    {
      if (*IT == IT_DELETE_KEY_FRAME)
      {
        m_CsKFBkp.Erase(*IDK);
        ++IDK;
      }
      if (*IT != IT_KEY_FRAME && *IT != IT_KEY_FRAME_SERIAL)
      {//只需要关键帧
        continue;
      }
      LA::Vector3f Rx;
      LA::SymmetricMatrix2x2f W;
      const int NX = static_cast<int>(IKF->m_Xs.size());//新观测到的地图点的数量
#ifdef CFG_DEBUG
      for (int iX = 0; iX < NX; ++iX) {
        const GlobalMap::Point &X = IKF->m_Xs[iX];
        if (iX > 0) {
          UT_ASSERT(X.m_iKF >= IKF->m_Xs[iX - 1].m_iKF);
        }
        X.AssertConsistency();
      }
#endif
#ifdef LBA_DEBUG_GROUND_TRUTH_MEASUREMENT
      if (m_CsGT && m_dsGT) {
        const std::vector<int> &iKF2d = m_solver->m_internal->m_iKF2d;
        const Rigid3D &Cam_state = m_CsGT[IKF->m_Cam_pose.m_iFrm].m_Cam_pose;
        for (int iX1 = 0, iX2 = 0; iX1 < NX; iX1 = iX2) {
          const int iKF = IKF->m_Xs[iX1].m_iKF;
          for (iX2 = iX1 + 1; iX2 < NX && IKF->m_Xs[iX2].m_iKF == iKF; ++iX2) {}
          const int Nx = iKF < m_CsKFGT.Size() ? m_iKF2d[iKF + 1] - m_iKF2d[iKF] : 0;
          const Depth::InverseGaussian *ds = m_dsGT->data() + iKF2d[iKF] + Nx - iX1;
          for (int iX = iX1; iX < iX2; ++iX) {
            GlobalMap::Point &X = IKF->m_Xs[iX];
            const Depth::InverseGaussian &d = ds[iX];
            if (X.m_x.m_xr.Valid()) {
              FTR::DebugSetMeasurement(m_K.m_br, d, X.m_x);
            }
            if (!X.m_zs.empty()) {
              DebugSetFeatureMeasurements(Cam_state, m_CsKFGT, d, &X);
            }
          }
        }
        DebugSetFeatureMeasurements(Cam_state, m_CsKFGT, *m_dsGT, iKF2d, &(*IKF));
      }
#endif
      if (IKF->m_Cam_state.Invalid())
      {//如果关键帧的相机状态并没有初始化,那么就从普通帧中找到相机的状态,因为刚才普通帧相机状态已经初始化过了
        for (std::list<InputLocalFrame>::iterator ILF = m_ILFs2.begin(); ILF != m_ILFs2.end(); ++ILF)
        {
          if (ILF->m_T == IKF->m_T)
          {//和关键帧对应的普通帧
            IKF->m_Cam_state = ILF->m_Cam_state;
            break;
          }
        }
      }
      if (IKF->m_Cam_state.Invalid())
      {//这种情况就是普通帧的状态也是不合法的,就用m_CsLF里的来初始化
        const int nLFs = static_cast<int>(m_LFs.size());
        for (int iLF = 0; iLF < nLFs; ++iLF)
        {
          if (m_LFs[iLF].m_T == IKF->m_T)
          {
            IKF->m_Cam_state = m_CsLF[iLF];
          }
        }
      }
      m_CsKFBkp.Push(IKF->m_Cam_state.m_Cam_pose);//m_CsKFBkp的m_data记录所有关键帧的左相机位姿
      const int nKFs = m_CsKFBkp.Size();
      m_R12s.Resize(nKFs);
#ifdef CFG_STEREO
      m_t12s.Resize(nKFs << 1);
#else
      m_t12s.Resize(nKFs);
#endif
      for (int iX1 = 0, iX2 = 0; iX1 < NX; iX1 = iX2)
      {//遍历所有的地图点
        const int iKF = IKF->m_Xs[iX1].m_iKF;//关键帧帧id
        const Rigid3D &C = m_CsKFBkp[iKF];
        m_marksTmp1.assign(nKFs, 0);
        for (iX2 = iX1 + 1; iX2 < NX && IKF->m_Xs[iX2].m_iKF == iKF; ++iX2) {}//没啥作用,就是迭代一次
        for (int iX = iX1; iX < iX2; ++iX)
        {//遍历所有地图点,如果左右两目都有观测的话,就进行三角化,算出左相机坐标系下的逆深度,协方差
            GlobalMap::Point &X = IKF->m_Xs[iX];//当前这个地图点
            m_zds.resize(0);
#ifdef CFG_STEREO
            if (X.m_x.m_xr.Valid())
            {//有右目的观测的话就说明可以三角化
                Rx.Set(X.m_x.m_x.x(), X.m_x.m_x.y(), 1.0f);
                X.m_x.m_Wr.GetScaled(w, W);
                m_zds.push_back(Depth::Measurement(m_K.m_br, Rx, X.m_x.m_xr, W));
            }
#endif
          const int Nz = static_cast<int>(X.m_zs.size());//除首次观测以外的其他观测
          for (int i = 0; i < Nz; ++i)
          {
            const FTR::Measurement &z = X.m_zs[i];
            Rotation3D &R = m_R12s[z.m_iKF];
#ifdef CFG_STEREO
            LA::AlignedVector3f *t = m_t12s.Data() + (z.m_iKF << 1);
#else
            LA::AlignedVector3f *t = m_t12s.Data() + z.m_iKF;
#endif
            if (!m_marksTmp1[z.m_iKF])
            {
              const Rigid3D &_C = m_CsKFBkp[z.m_iKF];
              if (_C.Valid())
              {
                const Rigid3D T = _C / C;
                R = T;
                T.GetTranslation(*t);
#ifdef CFG_STEREO
                LA::AlignedVector3f::apb(t[0], m_K.m_br, t[1]);
#endif
              } else
              {
                  R.Invalidate();
                  t->Invalidate();
#ifdef CFG_STEREO
                  t[1].Invalidate();
#endif
              }
              m_marksTmp1[z.m_iKF] = 1;
            }
            if (R.Valid())
            {
              R.Apply(X.m_x.m_x, Rx);
#ifdef CFG_STEREO
              if (z.m_z.Valid())
              {
                z.m_W.GetScaled(w, W);
                m_zds.push_back(Depth::Measurement(t[0], Rx, z.m_z, W));
              }
              if (z.m_zr.Valid())
              {
                z.m_Wr.GetScaled(w, W);
                m_zds.push_back(Depth::Measurement(t[1], Rx, z.m_zr, W));
              }
#else
              z.m_W.GetScaled(gyr, W);
              m_zds.push_back(Depth::Measurement(*t, Rx, z.m_z, W));
#endif
            }
          }
//#ifdef CFG_DEBUG
#if 0
          const KeyFrame &KF = m_KFs[iKF];
          const int ix = static_cast<int>(KF.m_xs.size()) + iX - iX1;
          if (KF.m_Cam_pose.m_iFrm == 1041 && ix == 431) {
            UT::DebugStart();
            X.m_d.Print();
          }
#endif
          float eAvg;
          const int Nzd = static_cast<int>(m_zds.size());//左右两目观测都有时这里的size就是1
            //三角化对X.m_d的逆深度和斜方差进行更新,通过r(Uc0) = 归一化(Pnc0 - Uc0 * tc0c1) - 归一化(Rc0c1 *Pnc1),min F(Uc0) = 0.5*||r(Uc0)||^2（马氏距离下）求解左目特征点的深度
            if (!Depth::Triangulateinit(w, Nzd, m_zds.data()/*观测*/, &X.m_d/*逆深度以及协方差*/, &m_work/*雅克比*/,m_K.m_Rr,m_K.m_br, X.m_d.Valid(), &eAvg) ||
              m_K.m_K.fx() * eAvg > DEPTH_TRI_MAX_ERROR)//,m_K.m_Rr,m_K.m_br
            {//如果无法三角化或者误差过大
            //X.m_d.Invalidate();
            //X.m_d = m_KFs[iKF].m_d;
            //if (!Depth::Triangulate(gyr, Nzd, m_zds.data(), &X.m_d, &m_work)) {
            //  X.m_d.Invalidate();
            //}
            if (Nzd == 0)
            {//如果没有其他观测
              X.m_d.Invalidate();
            } else {
              X.m_d.Initialize();
              for (int n = 1; n <= Nzd; ++n)
              {
                Depth::Triangulate(w, n, m_zds.data(), &X.m_d, &m_work);
              }
              if (!X.m_d.Valid())
              {
                X.m_d.Invalidate();
              }
            }
          }
          //UT::Print("%d %f %f\n", iX, X.m_d.u(), X.m_d.s2());
//#ifdef CFG_DEBUG
#if 0
          if (UT::Debugging()) {
            UT::DebugStop();
            X.m_d.Print();
            Depth::ComputeError(Nzd, m_zds.data(), X.m_d);
          }
#endif
        }
      }
      ++IKF;
    }
  }
  while (!m_ITs2.empty())
  {//遍历m_ITs2记录的帧信息,比如只有普通帧,或者普通帧+关键帧
    const int nKFs = static_cast<int>(m_KFs.size());
    const InputType IT = m_ITs2.front();
    m_ITs2.pop_front();//从m_ITs2删除这帧的记录
    if (IT == IT_LOCAL_FRAME)
    {//普通帧需要初始化一下,对所有地图点的深度做一个加权平均,以及寻找最近的关键帧(共视么)
      InputLocalFrame &ILF = m_ILFs2.front();//当前输入普通帧
#ifdef LBA_DEBUG_GROUND_TRUTH_MEASUREMENT
      const std::vector<Depth::InverseGaussian> &dsGT = m_solver->m_internal->m_dsGT;
      if (m_CsGT && !dsGT.empty()) {
        DebugSetFeatureMeasurements(m_CsGT[ILF.m_Cam_pose.m_iFrm].m_Cam_pose, m_CsKFGT, dsGT,
                                    m_solver->m_internal->m_iKF2d, &ILF);
      }
#endif
      if (ILF.m_d.Invalid())
      {//第一帧的时候左相机坐标系下特征点平均分布的逆深度和协方差,要用关键帧的所有有深度的地图点来求
        if (m_LFs.empty())
        {
          ILF.m_d.Initialize();
        } else
        {//用前一帧的深度来做初始化
            const LocalFrame &LF = m_LFs[m_ic2LF.back()];//滑窗里最新的一阵帧,那么应该就是前一帧
            ILF.m_d = LF.m_d;//用前一帧的来初始化一下深度
            ILF.m_d.Propagate(ILF.m_T.m_t - LF.m_T.m_t/*两帧之间dt*/);//协方差也更新下
        }
        const int Nz = static_cast<int>(ILF.m_zs.size())/*观测到关键帧中地图点的数量*/, NzC = SIMD_FLOAT_CEIL(Nz);
        m_work.Resize(NzC * 3);
        LA::AlignedVectorXf us(m_work.Data(), Nz, false);     us.Resize(0);//所有特征点的逆深度
        LA::AlignedVectorXf ws(us.Data() + NzC, Nz, false);   ws.Resize(0);//
        LA::AlignedVectorXf wus(ws.Data() + NzC, Nz, false);
        for (std::list<GlobalMap::InputKeyFrame>::iterator IKF = m_IKFs2.begin();//如果有关键帧的话找到当前对应的关键帧
             IKF != m_IKFs2.end(); ++IKF)
        {
          if (IKF->m_T != ILF.m_T)
          {//再检查一下是不是同一帧
            continue;
          }
          const int NX = static_cast<int>(IKF->m_Xs.size()), N = NX + Nz, NC = SIMD_FLOAT_CEIL(N);
          m_work.Resize(NC * 3);
          us.Bind(m_work.Data(), N);     us.Resize(0);
          ws.Bind(us.Data() + NC, N);    ws.Resize(0);
          wus.Bind(ws.Data() + NC, N);
          for (int iX = 0; iX < NX; ++iX)
          {//遍历所有特征点,将有效深度信息记录下来
            const GlobalMap::Point &X = IKF->m_Xs[iX];
            if (X.m_iKF != nKFs || !X.m_d.Valid())
            {//如果有深度信息就push进去
              continue;
            }
            us.Push(X.m_d.u());
            ws.Push(X.m_d.s2());
          }
          break;
        }
        Depth::InverseGaussian dz;
        Rigid3D::Row Crz;
        const Rigid3D C = ILF.m_Cam_state.m_Cam_pose;//当前imu积分后的pose Tc0w
        const Rigid3D::Row Cz = C.GetRowZ();//第三行
        const int NZ = static_cast<int>(ILF.m_Zs.size());
        for (int iZ = 0; iZ < NZ; ++iZ)
        {//遍历所有对于关键帧的观测信息
          const FRM::Measurement &Z = ILF.m_Zs[iZ];
          if (Z.m_iKF == nKFs)
          {//是本帧关键帧的观测信息就跳过,这种情况不会发生
            continue;
          }
          const Depth::InverseGaussian *ds = m_ds.data() + m_iKF2d[Z.m_iKF];//这个关键帧对应的首个地图点地址
          Rigid3D::ABI(Cz, m_CsKF[Z.m_iKF]/*这个关键帧的Tc0w*/, Crz);//Crz就是(Tc0w(当前帧) * Twc0(关键帧)).row(z)
          const GlobalMap::KeyFrame &KF = m_KFs[Z.m_iKF];
          for (int iz = Z.m_iz1/*起始位置*/; iz < Z.m_iz2/*终止位置*/; ++iz)
          {//遍历所有观测到的当前关键帧的地图点,算出这些观测到的地图点在当前帧中的逆深度
            const int ix = ILF.m_zs[iz].m_ix;//当前普通帧对这个关键帧观测索引,观测的这个地图点在关键帧中的局部id
            const Depth::InverseGaussian &d = ds[ix];
            if (!d.Valid() || !d.ProjectD(Crz/*(Tc0w(当前帧) * Twc0(关键帧)).row(z)*/, KF.m_xs[ix].m_x/*首次被观测到的归一化坐标*/
                    , dz/*地图点在当前帧中的观测信息*/))
            {
              continue;//利用Pc(当前帧)[2]= u * (Tc0w(当前帧) * Twc0(关键帧)).row(2) * Pc0(关键帧)算出这个地图点在当前帧中的逆深度和协方差
            }
            us.Push(dz.u());
            ws.Push(dz.s2());
          }
        }
        ws += DEPTH_VARIANCE_EPSILON;//防止对角线为0吧?
        ws.MakeInverse();//信息矩阵
        const float Sw = ws.Sum();//所有的有深度的地图点的信息矩阵的和
        if (Sw < FLT_EPSILON)
        {
          ws.Set(1.0f / Nz);
        } else
        {
            ws *= 1.0f / Sw;//对所有地图点的信息矩阵进行归一化
        }
        LA::AlignedVectorXf::AB(ws, us, wus);//wus = ws * us  这个是归一化以后的权重*点的逆深度
        const float u = wus.Sum();//这个是算了一个加权的地图点平均逆深度
        us -= u;//us = x-u u为平均深度，x为每个地图点自己的深度
        us.MakeSquared();
        LA::AlignedVectorXf::AB(ws, us, wus);//(x-u)*ws*(x-u).t 归一化后的协方差
        const float s2 = wus.Sum();//由所有地图点的观测算出目前场景深度的总的协方差
        if (s2 > 0.0f)
        {//高斯分布更新
          ILF.m_d.Update(Depth::InverseGaussian(u, s2));
        }
      }
      //if (!ILF.m_Zs.empty() && ILF.m_iKFsMatch.empty()) {
        SearchMatchingKeyFrames(ILF);//输入普通帧,进行共视帧的关联,包括在共视关键帧的共视关键帧去寻找可能的共视
      //}
      const int Nk = static_cast<int>(ILF.m_iKFsMatch.size());//所有的共视帧数量
      if (ILF.m_iKFNearest == -1 && Nk != 0)//如果还没有设置最近关键帧,并且有共视帧时
      {
        if (LBA_MARGINALIZATION_REFERENCE_NEAREST)//算出和当前帧之间图像运动最小的这个关键帧,作为最近的关键帧m_iKFNearest
        {
          const int iKFMax = ILF.m_iKFsMatch.back();//共视帧中最新的那帧
#ifdef CFG_DEBUG
          if (iKFMax >= nKFs) {
            UT_ASSERT(iKFMax == nKFs);
          }
#endif
          ubyte first = 1;
          int iKFNearest = m_Zp.m_iKFr == -1 ? nKFs - 1 : m_Zp.m_iKFr;//如果m_Zp.m_iKFr还没有赋值的话,就直接用最近的那个关键帧
          float imgMotionNearest = FLT_MAX;
          const Rigid3D C = ILF.m_Cam_state.m_Cam_pose;//当前帧pose
          const float z = 1.0f / ILF.m_d.u();//当前帧地图点的平均深度
          m_marksTmp1.assign(iKFMax + 1, 0);
          for (int ik = 0; ik < Nk; ++ik)
          {//遍历所有共视帧,将其在m_marksTmp1中设成1
            m_marksTmp1[ILF.m_iKFsMatch[ik]] = 1;
          }
          const int _nKFs = std::min(nKFs/*所有关键帧的个数*/, iKFMax + 1/*共视帧中最新的关键帧的id+1*/);//只用遍历到共视里最新的就好了
          for (int iKF = 0; iKF < _nKFs; ++iKF)//遍历所有关键帧,跳过非共视帧,算出和当前帧之间图像运动最小的这个关键帧,作为最近的关键帧
          {
            if (!m_marksTmp1[iKF])//跳过非共视帧
            {
              continue;
            }
            const Rigid3D _C = m_CsKF[iKF];//当前共视帧的pose
            const float imgMotion = ComputeImageMotion(z/*当前帧地图点的平均深度*/, C/*Tc0w(ILF)*/, _C/*Tc0w(KF)*/, &first);//算一下图片的运动,像素级的残差
            if (imgMotion > imgMotionNearest)//
            {
              continue;
            }
            imgMotionNearest = imgMotion;
            iKFNearest = iKF;
          }
          ILF.m_iKFNearest = iKFNearest;
        } else
        {//不用边缘化的信息，那么最近的帧就是离它时间最近的一个关键帧
            ILF.m_iKFNearest = nKFs - 1;
        }
      } else if (ILF.m_iKFNearest == -1 && Nk == 0)//如果没共视帧的话,那么最近帧要么是m_iKFr要么就是最近的关键帧的id
      {
        ILF.m_iKFNearest = m_Zp.m_iKFr == -1 ? nKFs - 1 : m_Zp.m_iKFr;
      }
        //向LBA加进普通帧帧信息
        //step1:判断是否大于最大窗口size,如果大于了需要滑窗处理
        //step2:对 比当前帧新5帧以内的帧 和 当前帧 进行共视的关联,存在较新帧的m_Zm中
        //step3:遍历当前帧对于每个关键帧的观测,如果这个关键帧中的地图点的轨迹起点比现在5帧窗口的最老帧还要老,就需要将关键帧里这个点再生成一条起点在当前5帧小窗口的新的子轨迹
        //step4:第2帧开始需要利用当前的imu测量进行预积分,算出前状态为本体坐标系下的当前状态的变动以及协方差。第3帧开始还会对前状态重新预积分,前前状态为本体坐标系下的前状态的协方差。
        //step5:计算一下这帧对所有关键帧的观测的重投影误差,当关键帧时还会对新的地图点的左右目(如果都有的话)之间的观测做重投影误差
      _PushLocalFrame(ILF);//向LBA加进普通帧信息
      m_ILFs2.pop_front();//m_ILFs2删除当前普通帧
    } else if (IT == IT_KEY_FRAME || IT == IT_KEY_FRAME_SERIAL)
    {//处理关键帧
      if (IT == IT_KEY_FRAME_SERIAL)
      {
        serialGBA = true;
      }
      GlobalMap::InputKeyFrame &IKF = m_IKFs2.front();
      const bool v1 = IKF.m_Cam_state.m_Cam_pose.Valid(), v2 = IKF.m_Cam_state.m_v.Valid();//之前已经用imu测量初始化过了,应该都是true
      if (!v1 || !v2)
      {
          const int nLFs = static_cast<int>(m_LFs.size());
        for (int ic = nLFs - 1; ic >= 0; --ic)
        {
          const int iLF = m_ic2LF[ic];
          if (m_LFs[iLF].m_T != IKF.m_T)
          {
            continue;
          }
          const Camera &C = m_CsLF[iLF];
          if (!v1)
          {
            IKF.m_Cam_state.m_Cam_pose = C.m_Cam_pose;
            IKF.m_Cam_state.m_p = C.m_p;
          }
          if (!v2)
          {
            IKF.m_Cam_state.m_v = C.m_v;
            IKF.m_Cam_state.m_ba = C.m_ba;
            IKF.m_Cam_state.m_bw = C.m_bw;
          }
          break;
        }
      }
      if (IKF.m_d.Invalid())
      {//如果关键帧的平均点云深度还没有赋值的话,就用普通帧的赋值一下
        const int nLFs = static_cast<int>(m_LFs.size());//在滑窗内的普通帧,最后一帧应该就是和关键帧对应的普通帧
        for (int ic = nLFs - 1; ic >= 0; --ic)
        {//ic 为 普通帧索引
          const int iLF = m_ic2LF[ic];
          const LocalFrame &LF = m_LFs[iLF];//与当前关键帧对应的局部普通帧
          if (LF.m_T != IKF.m_T)
          {
            continue;
          }
          IKF.m_d = LF.m_d;
          break;
        }
      }
        // step1:m_idxsTmp1扩容成所有关键帧size，存储一下F这帧都共视到了哪些老的关键帧 [关键帧id] = m_Zs观测中这个关键帧的索引 -1代表没有共视
        // step2:遍历F观测到的所有关键帧,再遍历这些关键帧自己的共视共视帧,寻找次共视,如果找到了(即这个共视帧和F观测到了同一个地图点),那么m_idxsTmp1对应的关键帧设为-2
        // step3:直接共视的需要对m_Zs构建索引,次共视和直接共视的都会作为F的共视关键帧存储在m_iKFsMatch中
        SearchMatchingKeyFrames(IKF);

      const int NX = static_cast<int>(IKF.m_Xs.size());//关键帧新观测到的地图点
      for (int iX = 0; iX < NX; ++iX)
      {//遍历所有的新地图点,将所有没有深度的地图点用第一次观测到它的关键帧的平均深度初始化
        GlobalMap::Point &X = IKF.m_Xs[iX];
#ifdef CFG_DEBUG
        UT_ASSERT(X.m_iKF <= nKFs);
#endif
        if (X.m_d.Invalid()) {//深度没有的话,就用平均深度当初始深度赋值一下
          X.m_d.Initialize(X.m_iKF == nKFs ? IKF.m_d.u() : m_KFs[X.m_iKF].m_d.u());
        }
      }

//step1:相关矩阵和KF的初始化
//step2:遍历所有的共视关键帧,也需要在他们自己的共视关键帧数据关联内加上当前这个关键帧
//step3:遍历所有观测到的地图点,将地图点的逆深度备份在m_dsBkp中
//step4:构造m_KFs里的当前关键帧,将它的所有新地图点（由当前关键帧产生）的观测保存下来
//step5:新来一个关键帧,需要和当前的滑窗内所有的帧建立共视关系。
//step5.1:如果不是当前关键帧所对应的滑窗帧,遍历滑窗帧观测的所有的关键帧中的地图点，只要有一个地图点是和当前关键帧有共视,就在这个滑窗帧中更新共视关键帧关联
//step5.2:如果是当前关键帧所对应的滑窗帧,这个关键帧所对应的滑窗普通帧也需要更新对于这个关键帧的观测信息,关键帧的最近关键帧设置成滑窗帧的最近关键帧,
//而滑窗的最近关键帧要改成当前这个最近关键帧,LF的子轨迹和观测,KF的子轨迹都需要初始化
//step5.3:向m_GM的m_Cs中放进这这个关键帧
//step5.4:向m_ITs1push进IT_KEY_FRAME GBA里的输入关键帧(m_us:在当前帧时间戳之前的imu测量,转到了(Rc0_i*)左相机坐标系下,m_dzs观测到的地图点的逆深度)
      _PushKeyFrame(IKF);//向LBA加进关键帧信息
      m_IKFs2.pop_front();//m_IKFs2中删除
    } else if (IT == IT_DELETE_KEY_FRAME)
    {
      DeleteKeyFrame(m_IDKFs2.front());
      m_IDKFs2.pop_front();
    } else if (IT == IT_DELETE_MAP_POINTS)
    {
      DeleteMapPoints(m_IDMPs2.front());
      m_IDMPs2.pop_front();
    } else if (IT == IT_UPDATE_CAMERAS || IT == IT_UPDATE_CAMERAS_SERIAL)
    {
      if (IT == IT_UPDATE_CAMERAS_SERIAL)
      {
        serialGBA = true;
      }
      UpdateCameras(m_IUCs2.front());
      m_IUCs2.pop_front();
    }
  }
  if (newKF || delKF || updCams)//运动先验和位姿先验只能等下一个关键帧来的时候再进行push
  {//如果新的关键帧
    m_ts[TM_TOTAL].Stop();
    m_ts[TM_SYNCHRONIZE].Stop();
    m_GBA->WakeUp(serialGBA);//唤醒GBA线程,进行globalBA优化
    m_ts[TM_SYNCHRONIZE].Start();
    m_ts[TM_TOTAL].Start();
  }
#if 0
//#if 1
  m_IT->RunView();
#endif
  if (!m_iFrmsKF.empty() &&//当有新的关键帧来的时候GBA更新了,LBA也需要数据同步
      m_GM->LBA_Synchronize(m_iFrmsKF.back()/*最新的关键帧对应的普通帧的id*/, m_CsKF/*关键帧左相机位姿*/,//更新LBA储存的m_CsKF里的位姿
              m_CsKFBkp/*关键帧左相机位姿备份*/, m_marksTmp1/*最新的共视关键帧中地图点是否有子轨迹生成*/
#ifdef CFG_HANDLE_SCALE_JUMP
                          , m_dsKF, m_dsKFBkp
#endif
                          )) {
#ifdef CFG_HANDLE_SCALE_JUMP
    AlignedVector<float> &ss = m_work;
    ss.Resize(0);
    const ubyte ucmFlag = LBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION |
                          LBA_FLAG_CAMERA_MOTION_UPDATE_POSITION |
                          LBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY;
    const int nKFs = static_cast<int>(m_KFs.size());
    for (int iKF = 0; iKF < nKFs; ++iKF) {
      if ((m_marksTmp1[iKF] & GM_FLAG_FRAME_UPDATE_DEPTH) &&
          fabs(m_dsKF[iKF] - m_dsKFBkp[iKF]) > BA_UPDATE_FRAME_DEPTH) {
        ss.Push(m_dsKF[iKF] / m_dsKFBkp[iKF]);
      }
    }
    if (ss.Size() > static_cast<int>(nKFs * BA_UPDATE_FRAME_DEPTH_RATIO + 0.5f)) {
      const int ith = ss.Size() >> 1;
      std::nth_element(ss.Data(), ss.Data() + ith, ss.End());
      const float s = ss[ith], sI = 1.0f / s;
      const xp128f _sI = xp128f::get(s);
      const int nLFs = static_cast<int>(m_LFs.size());
      for (int iLF = 0; iLF < nLFs; ++iLF) {
        Camera &Cam_state = m_CsLF[iLF];
        Cam_state.m_Cam_pose.ScaleTranslation(sI);
        Cam_state.m_p *= _sI;
        Cam_state.m_v *= _sI;
        m_ucsLF[iLF] |= LBA_FLAG_FRAME_UPDATE_CAMERA;
        m_ucmsLF[iLF] |= ucmFlag;
        m_UcsLF[iLF] = LM_FLAG_FRAME_UPDATE_CAMERA_LF;
      }
      const float s2 = s * s;
      for (int iKF = 0; iKF < nKFs; ++iKF) {
        if (m_marksTmp1[iKF] & GM_FLAG_FRAME_UPDATE_CAMERA) {
          m_CsKFBkp[iKF].ScaleTranslation(sI);
        }/* else {
          m_CsKF[iKF].ScaleTranslation(sI);
          m_ucsKF[iKF] |= LBA_FLAG_FRAME_UPDATE_CAMERA;
          m_UcsKF[iKF] |= LM_FLAG_FRAME_UPDATE_CAMERA_KF;
        }*/
        const int id = m_iKF2d[iKF], Nx = m_iKF2d[iKF + 1] - id;
        Depth::InverseGaussian *ds = m_ds.data() + id;
        ubyte *uds = m_uds.data() + id;
        for (int ix = 0; ix < Nx; ++ix) {
          Depth::InverseGaussian &d = ds[ix];
          d.u() *= s;
          d.s2() *= s2;
          uds[ix] |= LBA_FLAG_TRACK_UPDATE_DEPTH;
        }
        if (Nx > 0) {
          m_ucsKF[iKF] |= LBA_FLAG_FRAME_UPDATE_DEPTH;
          m_UcsKF[iKF] |= LM_FLAG_TRACK_UPDATE_DEPTH;
        }
      }
    }
#endif
    //关键帧的位姿如果变化了,那么以这个关键帧参考关键帧的滑窗帧的pose也要相应的调整,Tw(更新后)w(更新前)来修正
    UpdateCameras(m_marksTmp1/*记录需要更新的关键帧*/, m_CsKFBkp/*LBA更新前的关键帧pose*/, m_CsKF/*LBA更新后的关键帧pose*/);
  }
//#ifdef CFG_DEBUG
#if 0
  if (m_KFs.size() == 2) {
    UT::PrintSeparator();
    m_CsKF[0].Print(true);
    UT::PrintSeparator();
    m_CsKF[1].Print(true);
  }
#endif
}
//更新m_CsLF中存储的滑窗中的相机pose,是否更新了的flags和更新m_CsKF中存储的相机pose,以及地图点逆深度,是否更新了的flags
void LocalBundleAdjustor::UpdateData() {
  const int iFrm1 = m_LFs[m_ic2LF.front()].m_T.m_iFrm;//滑窗中最老的帧的全局帧id
  const int iFrm2 = m_LFs[m_ic2LF.back()].m_T.m_iFrm;//滑窗中最新的帧的全局帧id
#ifdef CFG_CHECK_REPROJECTION
  const int nLFs = static_cast<int>(m_LFs.size());
#ifdef CFG_DEBUG
  UT_ASSERT(static_cast<int>(m_esLF.size()) == nLFs);
#endif//遍历所有滑窗中的需要更新状态的帧，计算一下普通帧和它观测到的地图点所属的关键帧之间的重投影误差
  for (int iLF = 0; iLF < nLFs; ++iLF) {
    if (m_UcsLF[iLF] & LM_FLAG_FRAME_UPDATE_CAMERA_LF) {
      ComputeErrorFeature(&m_LFs[iLF]/*当前帧*/, m_CsLF[iLF].m_Cam_pose/*Tc0w*/, m_CsKF/*关键帧位姿*/, m_ds,/*所有地图点逆深度*/
              &m_esLF[iLF].second);//计算一下普通帧和它观测到的地图点之间的重投影误差(转到像素量程的残差)
    }
  }
  const int nKFs = static_cast<int>(m_KFs.size());
#ifdef CFG_DEBUG
  UT_ASSERT(static_cast<int>(m_esKF.size()) == nKFs);
#endif//遍历所有需要更新状态的关键帧，计算一下它观测到的地图点所属的关键帧之间的重投影误差,以及自己地图点左右目之间的重投影误差
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    if (m_UcsKF[iKF] & LM_FLAG_FRAME_UPDATE_CAMERA_KF) {
      ComputeErrorFeature(&m_KFs[iKF], m_CsKF[iKF], m_CsKF, m_ds, &m_esKF[iKF].second, iKF);
    }
  }
#endif
//更新m_CsLF中存储的滑窗中的相机pose,是否更新了的flags和更新m_CsKF中存储的相机pose,以及地图点逆深度,是否更新了的flags
  m_LM->LBA_Update(iFrm1/*滑窗中最老的帧的全局帧id*/, iFrm2/*滑窗中最新的帧的全局帧id*/, m_ic2LF/*索引*/, m_CsLF//滑窗内的相机状态
   , m_UcsLF/*局部地图普通帧是否更新*/, m_iFrmsKF/*关键帧对应的普通帧id,下标是关键帧索引,值是普通帧索引*/, m_CsKF/*关键帧左相机位姿*/,
   m_UcsKF,/*局部地图关键帧是否更新*/m_iKF2d/*kf和地图点之间的索引*/, m_ds/*所有地图点*/, m_Uds/*地图点是否要更新*/
#ifdef CFG_CHECK_REPROJECTION
                 , m_esLF/*滑窗帧和共视的关键帧之间的重投影误差*/,
                 m_esKF/*所有的关键帧的地图点的平均重投影误差还有与它共视的关键帧之间的重投影误差*/
#endif
                 );
#ifdef CFG_VERBOSE
  if (m_verbose >= 2) {
    const int Ncu = UT::VectorCountFlag<ubyte>(m_UcsLF, LM_FLAG_FRAME_UPDATE_CAMERA_LF);
    const int Nc = static_cast<int>(m_LFs.size());
    const int Nku = UT::VectorCountFlag<ubyte>(m_UcsKF, LM_FLAG_FRAME_UPDATE_CAMERA_KF);
    const int Nk = static_cast<int>(m_KFs.size());
    const int Ndu = UT::VectorCountFlag<ubyte>(m_Uds, LM_FLAG_TRACK_UPDATE_DEPTH);
    const int Nd = static_cast<int>(m_ds.size());
    UT::PrintSeparator();
    UT::Print("[%d] [LocalBundleAdjustor::UpdateData]\n", iFrm2);
    UT::Print("  CameraLF = %d / %d = %.2f%%\n", Ncu, Nc, UT::Percentage(Ncu, Nc));
    UT::Print("  CameraKF = %d / %d = %.2f%%\n", Nku, Nk, UT::Percentage(Nku, Nk));
    UT::Print("  Depth    = %d / %d = %.2f%%\n", Ndu, Nd, UT::Percentage(Ndu, Nd));
  }
#endif
  const int iLF = m_ic2LF.back(), iFrm = m_LFs[iLF].m_T.m_iFrm;
  MT_WRITE_LOCK_BEGIN(m_MT, iFrm, MT_TASK_LBA_UpdateData);
  m_C.m_T = m_LFs[iLF].m_T;//记录一下滑窗中最新的这帧
  m_C.m_C = m_CsLF[iLF];
  //m_ILFs1.resize(0);
  //m_IKFs1.resize(0);
  MT_WRITE_LOCK_END(m_MT, iFrm, MT_TASK_LBA_UpdateData);
}

bool LocalBundleAdjustor::BufferDataEmpty() {
  return m_ITs1.empty();
}

#ifdef LBA_DEBUG_PRINT_MARGINALIZATION
static inline void PrintReduction(const IMU::Delta::Error &e1, const IMU::Delta::Error *e2) {
  float er[3], ev[3], ep[3], eba[3], ebw[3];
  for (int i = 0; i < 3; ++i) {
    const IMU::Delta::Error &e = e2[i];
    if (e.Valid()) {
      er[i] = sqrtf(e.m_er.SquaredLength()) * UT_FACTOR_RAD_TO_DEG;
      ev[i] = sqrtf(e.m_ev.SquaredLength());
      ep[i] = sqrtf(e.m_ep.SquaredLength());
      eba[i] = sqrtf(e.m_eba.SquaredLength());
      ebw[i] = sqrtf(e.m_ebw.SquaredLength()) * UT_FACTOR_RAD_TO_DEG;
    } else {
      er[i] = FLT_MAX;
      ev[i] = FLT_MAX;
      ep[i] = FLT_MAX;
      eba[i] = FLT_MAX;
      ebw[i] = FLT_MAX;
    }
  }
  UT::PrintReduction(sqrtf(e1.m_er.SquaredLength()) * UT_FACTOR_RAD_TO_DEG, er, 3, "  r = ");
  UT::PrintReduction(sqrtf(e1.m_ev.SquaredLength()), ev, 3, "  v = ");
  UT::PrintReduction(sqrtf(e1.m_ep.SquaredLength()), ep, 3, "  p = ");
  UT::PrintReduction(sqrtf(e1.m_eba.SquaredLength()), eba, 3, " ba = ");
  UT::PrintReduction(sqrtf(e1.m_ebw.SquaredLength()) * UT_FACTOR_RAD_TO_DEG, ebw, 3, " bw = ");
}
static inline void PrintReduction(const Camera::Calibration &K, const int ix,
                                  const LA::Vector2f &e1,
#ifdef CFG_STEREO
                                  const LA::Vector2f &e1r,
#endif
                                  const FTR::Error *e2) {
  float e[3];
#ifdef CFG_STEREO
  UT::Print("  %2d\t", ix);
  if (e1.Valid()) {
    for (int i = 0; i < 3; ++i) {
      if (e2[i].m_ex.Valid()) {
        e[i] = sqrtf(e2[i].m_ex.SquaredLength() * K.m_K.fxy());
      } else {
        e[i] = FLT_MAX;
      }
    }
    UT::PrintReduction(sqrtf(e1.SquaredLength() * K.m_K.fxy()), e, 3,
                       "", false, e1r.Invalid());
  }
  if (e1r.Valid()) {
    for (int i = 0; i < 3; ++i) {
      if (e2[i].m_exr.Valid()) {
        e[i] = sqrtf(e2[i].m_exr.SquaredLength() * K.m_Kr.fxy());
      } else {
        e[i] = FLT_MAX;
      }
    }
    UT::PrintReduction(sqrtf(e1r.SquaredLength() * K.m_Kr.fxy()), e, 3,
                       e1.Valid() ? UT::String("\t\t") : UT::String("\t\t\t\t\t"));
  }
#else
  for (int i = 0; i < 3; ++i) {
    e[i] = sqrtf(e2[i].m_ex.SquaredLength() * K.m_K.fxy());
  }
  UT::PrintReduction(sqrtf(e1.m_ex.SquaredLength() * K.m_K.fxy()), e, 3, UT::String("  %d ", ix));
#endif
}
static inline void PrintDifference(const LA::Vector2f *xg, const int i1 = 0, const int i2 = 1) {
  UT::Print(" eg = %f\n", sqrtf((xg[i1] - xg[i2]).SquaredLength()) * UT_FACTOR_RAD_TO_DEG);
}
static inline void PrintDifference(const CameraPrior::Element::EC *xc, const int i1 = 0, const int i2 = 1) {
  UT::Print(" ep = %f\n", sqrtf((xc[i1].m_ep - xc[i2].m_ep).SquaredLength()));
  UT::Print(" er = %f\n", sqrtf((xc[i1].m_er - xc[i2].m_er).SquaredLength()) * UT_FACTOR_RAD_TO_DEG);
}
static inline void PrintDifference(const CameraPrior::Element::EC *xc1,
                                   const CameraPrior::Element::EC *xc2,
                                   const int i1 = 0, const int i2 = 1) {
  UT::Print(" ep = %f %f\n", sqrtf((xc1[i1].m_ep - xc1[i2].m_ep).SquaredLength()),
                             sqrtf((xc2[i1].m_ep - xc2[i2].m_ep).SquaredLength()));
  UT::Print(" er = %f %f\n", sqrtf((xc1[i1].m_er - xc1[i2].m_er).SquaredLength() * UT_FACTOR_RAD_TO_DEG),
                             sqrtf((xc2[i1].m_er - xc2[i2].m_er).SquaredLength()) * UT_FACTOR_RAD_TO_DEG);
}
static inline void PrintDifference(const std::vector<CameraPrior::Element::EC> *xks,
                                   const int i1 = 0, const int i2 = 1) {
  const int N = static_cast<int>(xks[i1].size());
#ifdef CFG_DEBUG
  UT_ASSERT(static_cast<int>(xks[i2].size()) == N);
#endif
  UT::Print(" ep =");
  for (int i = 0; i < N; ++i) {
    UT::Print(" %f", sqrtf((xks[i1][i].m_ep - xks[i2][i].m_ep).SquaredLength()));
  }
  UT::Print("\n");
  UT::Print(" er =");
  for (int i = 0; i < N; ++i) {
    UT::Print(" %f", sqrtf((xks[i1][i].m_er - xks[i2][i].m_er).SquaredLength()) * UT_FACTOR_RAD_TO_DEG);
  }
  UT::Print("\n");
}
static inline void PrintDifference(const CameraPrior::Element::EM *xm1,
                                   const CameraPrior::Element::EM *xm2,
                                   const int i1 = 0, const int i2 = 1) {
  UT::Print(" ev = %f %f\n", sqrtf((xm1[i1].m_ev - xm1[i2].m_ev).SquaredLength()),
                             sqrtf((xm2[i1].m_ev - xm2[i2].m_ev).SquaredLength()));
  UT::Print("eba = %f %f\n", sqrtf((xm1[i1].m_eba - xm1[i2].m_eba).SquaredLength()),
                             sqrtf((xm2[i1].m_eba - xm2[i2].m_eba).SquaredLength()));
  UT::Print("ebw = %f %f\n", sqrtf((xm1[i1].m_ebw - xm1[i2].m_ebw).SquaredLength()) * UT_FACTOR_RAD_TO_DEG,
                             sqrtf((xm2[i1].m_ebw - xm2[i2].m_ebw).SquaredLength()) * UT_FACTOR_RAD_TO_DEG);
}
static inline void PrintDifference(const LA::Vector2f *xg,
                                   const std::vector<CameraPrior::Element::EC> *xks,
                                   const CameraPrior::Element::EC *xc1,
                                   const CameraPrior::Element::EM *xm1,
                                   const CameraPrior::Element::EC *xc2,
                                   const CameraPrior::Element::EM *xm2) {
  const std::string str[3] = {"GT", "ET", "HT"};
  for (int i1 = 0; i1 < 3; ++i1) {
    if (xg[i1].Invalid()) {
      continue;
    }
    for (int i2 = i1 + 1; i2 < 3; ++i2) {
      if (xg[i2].Invalid()) {
        continue;
      }
      UT::PrintSeparator();
      UT::Print("%s vs %s\n", str[i1].c_str(), str[i2].c_str());
      PrintDifference(xg, i1, i2);
      if (xks) {
        PrintDifference(xks, i1, i2);
      }
      if (xc1) {
        PrintDifference(xc1, xc2, i1, i2);
      } else {
        PrintDifference(xc2, i1, i2);
      }
      PrintDifference(xm1, xm2, i1, i2);
    }
  }
}
#endif
//两种边缘化策略,最老帧是关键帧时,那么边缘化最老帧的motion,如果不是关键帧时,边缘化最老帧的pose和motion
//先验部分不仅仅会存储g最老帧的cm还会存储各个时刻最老帧观测到的观测关键帧(非当前的参考关键帧,同时下一次更新参考关键帧后,观测关键帧需更新关联)
//状态量Rcik,Rcjk,p_kci,p_kcj,,v_i,v_j,bai,baj,bwi,bwj,gw(东北天下的重力) k是参考关键帧,vi就是i坐标系下的速度
//视觉约束 在滑窗中,重投影误差由原来的Pcl = Rclw * (Rckw.t*Pck + twck - twcl) cl是局部普通帧,ck是这个地图点所属于的关键帧
//改成了 Pcl = Rclrk * (Rckrk.t *Pck + trkck - trkcl ) 从world坐标系转到了参考关键帧坐标系下 ,rk是参考关键帧
//
//imu约束
//e_r = -ln{预积分的Rij * exp[Jrbw *(bwi - z_bw)]x * Rcjk * Rcik.t}v
//e_v = Rcik*(Rcjk.t * v_j(是Rcjw*vw_j) - Rckw * gw*dt) - v_i(是Rciw*vw_i) - (m_v + m_Jvba * (bai - m_ba) + m_Jvbw * (bwi - m_bw))
//e_p = - tc0i  + Rcik *(Rcjk.t*tc0i + (pkj(是Rkw*pwi + tkw) - pki)(这里代码里是Rkw*(pwi-pwj)因为tkw的部分消去了) - Rckw *
// gw*0.5*dt^2) - v_i*dt- (m_p + m_Jpba * (bai - m_ba) + m_Jpbw * (bwi - m_bw))
//e_ba = bai - baj
//e_bw = bwi - bwj 这两个的雅克比就是i是1,j是-1
//
//A第一次滑窗的时候是没有运动先验的,所以运动先验都是0,
//step1:m_Zp初始化,存储当前参考关键帧id,m_Zps里[0]存当前参考关键帧的Rwc0
//step2:计算imu预积分对H|-b的影响，fix最老帧pose
//step3:PropagateKF中将当前最老帧的motion状态进行边缘化,即只存g,次老帧的pose,motion之间的先验约束,m_Zps里的最新的参考关键
// 帧pose放到[1]中(Rwck,但是t是0,0，0),而[0]处的设置成Tc0(参考关键帧)_c0(次老帧)
//更新m_Zp 中存储的motion状态量,其中m_v存储的是次老帧坐标系下的速度,更新m_Arr,m_Arc,m_br,m_Acc,m_bc,m_Arm,m_Acm,m_Amm,m_bm存储的先验约束
//step4:m_ZpLF存储这次边缘化时最后的次老帧的Rc0w*vw(次老帧坐标系下的v),ba,bw,下一次取的时候就是最老帧的Rc0w*vw,ba,bw了,获取g,观测关键帧,
// 最老帧c,最老帧m的先验矩阵,除了这次的次老帧的motion,其他的全部都merge掉,保存在m_ZpLF中
//
//B下一次滑窗时如果最老帧是关键帧
//step1:先重新获取g,观测关键帧,最老帧c,最老帧m的先验矩阵,除了最老帧的motion,其他的全部都merge掉,保存在m_ZpLF中,如果满秩，
// 就向GBA里m_IZpLM1的保存运动先验,同时m_ITs1保存了向GBA输入先验的标签
//step2:计算imu预积分对H|-b的影响，fix最老帧pose
//step3:m_ZpKF中信息的导入以及边缘化最老帧的运动状态,重新获取g,观测关键帧,最老帧c,最老帧m的先验矩阵,将最老帧的motion部分边缘化,(因为最老帧是关键帧,所以不用merge pose)
// 向m_ZpKF中存储g和pose的先验约束,然后再求一下Ax=b以后的x.t*b,向GBA里m_ZpKF的保存位姿先验,同时m_ITs1保存了向GBA输入先验位姿的标签
//step4:m_Zp初始化,存储当前参考关键帧id,m_Zps里[0]存当前参考关键帧的Rwc0,以及之前的运动约束，以前存储的观测关键帧全部清零(因为
// 存的都是Tc0(观测到的关键帧)c0(参考关键帧))现在参考关键帧变了,存储的这些自然要变
//step5:PropagateKF中将当前最老帧的motion状态进行边缘化,即只存g,次老帧的pose,motion之间的先验约束,m_Zps里的最新的参考关键帧pose放
// 到[1]中(Rwck,但是t是0,0，0),而[0]处的设置成Tc0(参考关键帧)_c0(次老帧)
//更新m_Zp 中存储的motion状态量,其中m_v存储的是次老帧坐标系下的速度,更新m_Arr,m_Arc,m_br,m_Acc,m_bc,m_Arm,m_Acm,m_Amm,m_bm存
// 储的先验约束
//step6:m_ZpLF存储这次边缘化时最后的次老帧的Rc0w*vw(次老帧坐标系下的v),ba,bw,下一次取的时候就是最老帧的Rc0w*vw,ba,bw了,获取g,观测关
// 键帧,最老帧c,最老帧m的先验矩阵,除了这次的次老帧的motion,其他的全部都merge掉,保存在m_ZpLF中
//
//C下一次滑窗时如果最老帧不是关键帧
//step1:遍历最老帧观测到的关键帧
//step1.1如果最老帧的观测到的关键帧是参考关键帧,遍历所有地图点,那么投影前的这帧的位姿就会fix,只算投影后和逆深度的影响,将地图点边缘化以后,
// m_Zp中更新由地图点边缘化引起的最老帧的pose和pose之间的约束
//step1.2如果最老帧的观测到的关键帧不是参考关键帧,遍历所有地图点,那么就需要将观测关键帧k1也加入约束,原来比如说是g,最老帧,那么现在就变
// 成g,ck1,最老帧的c，m,如果再来一个观测参考帧,那么先验矩阵就变成了g,ck1,ck2,c,m
//step1.2.1计算投影后,投影前,逆深度的影响,将地图点边缘化以后,地图点边缘化以后,会在观测关键帧和最老帧之间产生约束
//step1.2.2向m_Zp中插入这个没有存储过的观测关键帧:m_iKFs存储这个新加的关键帧,m_Zps存储Tc0(观测到的关键帧)c0(参考关键帧),这两种的索引是一致的，
//m_Zp中更新由地图点边缘化引起的关键帧和最老帧pose(当然还有它们自己的影响)的约束
//step1.2.3当新来了一个观测关键帧时,会将它放在H矩阵的最老帧前面,之前的观测关键帧之后,顺序即为g,N个观测关键帧,最老帧,更新他们之前地图点边缘化
// 以后产生的约束
//step2:判断当前最老帧的最近关键帧是否是当前的参考关键帧,如果不是那么就要切换参考关键帧
//step2.1先重新获取g,观测关键帧,最老帧c,最老帧m的先验矩阵,除了最老帧的motion,其他的全部都merge掉,保存在m_ZpLF中
//step2.2m_ZpKF中信息的导入以及边缘化最老帧的运动状态,重新获取g,观测关键帧,最老帧c,最老帧m的先验矩阵,将最老帧的pose,motion部分边缘化,
//step2.3向m_ZpKF中存储g和观测关键帧的pose的先验约束,边缘化最老帧的pose和motion,向GBA里m_ZpKF的保存位姿先验,同时m_ITs1保存了向GBA输入先验位姿的标签
//step2.4重置m_Zp,m_Zp初始化,存储当前参考关键帧id,m_Zps里[0]存当前参考关键帧的Rwc0,以及之前的运动约束，以前存储的观测关键帧全部清零
// (因为存的都是Tc0(观测到的关键帧)c0(参考关键帧))现在参考关键帧变了,存储的这些自然要变
//step3:为了保持和前一次滑窗的一致性,都是用前一次保存的相对位姿来算前后帧的pose,因为前后的pose可能会在前后滑窗中的LBA优化中产生变动,
// 计算imu预积分对H|-b的影响
//step4:PropagateLF中将m_Zps[i]更新为Tc0(参考关键帧)_c0(次老帧) i为当前最后一个参考关键帧,更新m_Zp 中存储的motion状态量,
// 其中m_v存储的是次老帧坐标系下的速度,将当前最老帧的pose,motion状态进行边缘化,即只存g,次老帧的pose,motion之间的先验约束
//step5:m_ZpLF存储这次边缘化时最后的次老帧的Rc0w*vw(次老帧坐标系下的v),ba,bw,下一次取的时候就是最老帧的Rc0w*vw,ba,bw了,获取g,
// 观测关键帧,最老帧c,最老帧m的先验矩阵,除了这次的次老帧的motion,其他的全部都merge掉,保存在m_ZpLF中
void LocalBundleAdjustor::MarginalizeLocalFrame()
{
  if (static_cast<int>(m_LFs.size()) < LBA_MAX_LOCAL_FRAMES)
  {
    return;
  }

//#ifdef CFG_DEBUG
#if 0
  UT::PrintSeparator();
  //m_Zp.Print();
  m_Zp.PrintDiagonal();
#endif
  const float eps = 0.0f;
  const float epsr = UT::Inverse(BA_VARIANCE_MAX_ROTATION, BA_WEIGHT_FEATURE, eps);
  const float epsp = UT::Inverse(BA_VARIANCE_MAX_POSITION, BA_WEIGHT_FEATURE, eps);
  const float epsv = UT::Inverse(BA_VARIANCE_MAX_VELOCITY, BA_WEIGHT_FEATURE, eps);
  const float epsba = UT::Inverse(BA_VARIANCE_MAX_BIAS_ACCELERATION, BA_WEIGHT_FEATURE, eps);
  const float epsbw = UT::Inverse(BA_VARIANCE_MAX_BIAS_GYROSCOPE, BA_WEIGHT_FEATURE, eps);
  const float _eps[] = {epsp, epsp, epsp, epsr, epsr, epsr, epsv, epsv, epsv,
                        epsba, epsba, epsba, epsbw, epsbw, epsbw};
  const int iLF1 = m_ic2LF[0]/*滑窗中最老的这帧索引*/, iLF2 = m_ic2LF[1]/*滑窗中次老的这帧索引*/;
  const LocalFrame &LF1 = m_LFs[iLF1];//最老的这帧要被边缘化掉
  const int iKFr = LF1.m_iKFNearest == -1 ? m_Zp.m_iKFr : LF1.m_iKFNearest;//和最老的这帧的图像运动最小的关键帧作为参考关键帧,如果没有最近关键帧,就用最新的参考关键帧

  const int iFrm = m_LFs[m_ic2LF.back()].m_T.m_iFrm;//最新的帧的id


  m_ZpKF.Invalidate();
  if (LF1.m_T.m_iFrm == m_KFs[iKFr].m_T.m_iFrm)//如果这帧的参考关键帧就是它自己(就是它当时也是个关键帧)
  {
    const bool v = m_Zp.Pose::Valid();//第一次要merge老帧才会出现v == false的情况

    if (v)//不是merge第一个关键帧
    {//m_ZpLF存储最老帧(就是对应于这个关键帧)的Rc0w*vw(最老帧坐标系下的v),ba,bw
     //获取g,c,m的先验矩阵,除了最老帧的motion,其他的全部都merge掉,保存在m_ZpLF中
      if (m_Zp.GetPriorMotion(&m_ZpLF/*最老帧motion的先验*/, &m_work, _eps) ||//
          !LBA_MARGINALIZATION_CHECK_INVERTIBLE)
      {//如果满秩
        //就向GBA里m_IZpLM1的保存运动先验,同时m_ITs1保存了向GBA输入先验的标签
        m_GBA->PushCameraPriorMotion(iFrm/*这个关键帧对应的全局帧id*/, iKFr/*对应的关键帧*/, m_ZpLF/*最老帧motion的先验*/);

      } else
      {
        //m_ZpLF.MakeZero();
        m_ZpLF.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL,
                          BA_VARIANCE_PRIOR_VELOCITY_RESET,
                          BA_VARIANCE_PRIOR_BIAS_ACCELERATION_RESET,
                          BA_VARIANCE_PRIOR_BIAS_GYROSCOPE_RESET, &m_CsLF[iLF1]);

      }
      //m_ZpKF中信息的导入以及边缘化最老帧的运动状态,将最老帧的motion部分边缘化,向m_ZpKF中存储g和观测关键帧以及参考关键帧的先验约束,然后再求一下Ax=b以后的x.t*b
      if ((m_Zp.GetPriorPose(iKFr/*当前要merge的老帧对应的关键帧id*/, &m_ZpKF/*关键帧的先验*/, &m_work, _eps) ||
           !LBA_MARGINALIZATION_CHECK_INVERTIBLE) &&
           m_ZpKF.MarginalizeUninformative(BA_WEIGHT_FEATURE,
                                           BA_VARIANCE_PRIOR_POSITION_INFORMATIVE,
                                           BA_VARIANCE_PRIOR_ROTATION_INFORMATIVE,
                                           &m_idxsTmp1/**/, &m_work, _eps))
      {
        m_GBA->PushCameraPriorPose(iFrm, m_ZpKF);//向GBA里m_ZpKF的保存运动先验,同时m_ITs1保存了向GBA输入先验位姿的标签
      } else
      {
        m_ZpKF.Invalidate();
      }
    }
    const Rigid3D &Tr = m_CsKF[iKFr];//参考关键帧对应Tc0w(kf)
    //const Rigid3D &Tr = m_CsLF[iLF1].m_Cam_pose;
    const float s2g = v ? BA_VARIANCE_PRIOR_GRAVITY_NEW : BA_VARIANCE_PRIOR_GRAVITY_FIRST/*第一次需要给一个重力的先验*/;
    //m_ZpLF.m_Amm *= sp;
    //m_ZpLF.m_bm *= sp;
    //清空m_Zp里所有的数据,m_Zp.m_iKFr存储当前参考关键帧的id,m_Zps里[0]存当前参考关键帧的Rwc0
    m_Zp.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL/* * sp*/, iKFr/*参考关键帧的id*/, Tr/*参考关键帧对应Tc0w(kf)*/,
            s2g/*重力对应的协方差*/, m_ZpLF/*运动先验*/, true/*是一个新的KF*/);

#ifdef CFG_HISTORY
    if (m_history >= 2) {
      m_MH.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL, s2g, BA_VARIANCE_PRIOR_POSITION_NEW,
                      BA_VARIANCE_PRIOR_ROTATION_NEW, BA_VARIANCE_PRIOR_VELOCITY_NEW,
                      BA_VARIANCE_PRIOR_BIAS_ACCELERATION_NEW, BA_VARIANCE_PRIOR_BIAS_GYROSCOPE_NEW,
                      m_ZpLF, &m_work, _eps);
#ifdef CFG_GROUND_TRUTH
      if (m_CsGT) {
        m_MH.PushGT(m_Zp, m_CsKFGT[iKFr], m_CsLFGT[iLF1]);
      }
#endif
    }
#endif
    //C1先存一下KF的东西,这里实际就是最老帧的状态
    Camera C1;
    C1.m_Cam_pose = Tr;//Tc0w(kf)
    Tr.GetPosition(C1.m_p);//twc0
    m_Zp.GetMotion(Tr, &C1.m_v, &C1.m_ba, &C1.m_bw);//C1.m_v = Rc0w.t*v_c0，ba,bw v转到了这帧自己的系下

    IMU::Delta::Error e;
    IMU::Delta::Jacobian::RelativeKF J;
    IMU::Delta::Factor::Auxiliary::RelativeKF A;
    const Camera &C2 = m_CsLF[iLF2];//最老帧之后的这帧
    const IMU::Delta &D = m_DsLF[iLF2];//最老帧和次老帧之间的预积分
    //这里没有求对最老帧pose的雅克比,因为这帧同时也是关键帧
    if(!IMU_GRAVITY_EXCLUDED)
        D.GetFactor(BA_WEIGHT_IMU/* * sd*/, C1/*最老帧的状态*/, C2/*次老帧的相机状态*/, m_K.m_pu/*tc0_i*/, &e, &J, &A/*因子*/, BA_ANGLE_EPSILON);

//#ifdef CFG_DEBUG
#if 0
    const bool scc1 = m_Zp.Invertible(&m_work, _eps);
#endif
//当前要merge的帧同时也是KF时,视觉约束仍然在KF中,所以这里的变量只有g,mi,cj,mj(重力,最老帧的motion,次老帧的pose,次老帧的motion)
//将当前帧的motion状态进行边缘化,m_Zps里的最新的参考关键帧pose放到[1]中(Rwck,但是t是0,0，0),而[0]处的设置成Tc0(参考关键帧)_c0(次老帧)
//更新m_Zp 中存储的motion状态量,其中m_v存储的是次老帧坐标系下的速度
//更新m_Arr,m_Arc,m_br,m_Acc,m_bc,m_Arm,m_Acm,m_Amm,m_bm存储的先验约束
    if (!m_Zp.PropagateKF(Tr/*参考关键帧Tc0w*/, C2/*次老帧的相机状态*/, A/*预积分的约束*/, &m_work, _eps) && LBA_MARGINALIZATION_CHECK_INVERTIBLE)
    {
      m_ZpLF.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL,
                        BA_VARIANCE_PRIOR_VELOCITY_RESET,
                        BA_VARIANCE_PRIOR_BIAS_ACCELERATION_RESET,
                        BA_VARIANCE_PRIOR_BIAS_GYROSCOPE_RESET, &C2);

      m_Zp.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL, iKFr, Tr,
                      BA_VARIANCE_PRIOR_GRAVITY_RESET, m_ZpLF, false, &C2.m_Cam_pose,
                      BA_VARIANCE_PRIOR_POSITION_RESET, BA_VARIANCE_PRIOR_ROTATION_RESET);

#ifdef CFG_HISTORY
      if (m_history >= 2) {
        m_MH.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL, s2g, BA_VARIANCE_PRIOR_POSITION_NEW,
                        BA_VARIANCE_PRIOR_ROTATION_NEW, BA_VARIANCE_PRIOR_VELOCITY_NEW,
                        BA_VARIANCE_PRIOR_BIAS_ACCELERATION_NEW, BA_VARIANCE_PRIOR_BIAS_GYROSCOPE_NEW,
                        m_ZpLF, &m_work, _eps, false);
#ifdef CFG_GROUND_TRUTH
        m_MH.PushGT(m_Zp, m_CsKFGT[iKFr], m_CsLFGT[iLF1]);
#endif
      }
#endif
    }
#ifdef CFG_HISTORY
    else if (m_history >= 2) {
      m_MH.PropagateKF(D, J, e, A);
#ifdef CFG_GROUND_TRUTH
      if (m_CsGT) {
        m_MH.PushGT(m_Zp, m_CsKFGT[iKFr], m_CsLFGT[iLF2]);
      }
#endif
    }
#endif
//#ifdef CFG_DEBUG

  } else
  {//如果这个老帧的最近关键帧不是它自己
    //Camera C1;
    Rigid3D /*Tr, TrI, */Tr1/*上一次边缘化算出的Tc0(最老帧)_c0(参考关键帧)*/, Trk, Tk1[2];
    /*const */int Nk = static_cast<int>(m_Zp.m_iKFs.size()) - 1;
#ifdef CFG_DEBUG
    UT_ASSERT(m_Zp.m_iKFs[Nk] == INT_MAX);
#endif
    //m_Zp.GetReferencePose(m_CsKF[m_Zp.m_iKFr], &Tr, &TrI);
    //m_Zp.GetPose(TrI, Nk, &Tr1, &C1.m_Cam_pose);
    m_Zp.m_Zps[Nk].GetInverse(Tr1);//上一次边缘化算出的Tc0(次老帧)_c0(参考关键帧),这里的次老帧是上一次边缘化时的次老帧,现在已经变成了最老帧
    const float s2d = UT::Inverse(BA_VARIANCE_PRIOR_DEPTH_NEW, BA_WEIGHT_PRIOR_CAMERA_INITIAL); 
    const float epsd = UT::Inverse(BA_VARIANCE_MAX_DEPTH, BA_WEIGHT_FEATURE, eps);
    const float epsc[12] = {epsp, epsp, epsp, epsr, epsr, epsr,
                            epsp, epsp, epsp, epsr, epsr, epsr};
    //const float s2pMax = BA_VARIANCE_PRIOR_POSITION_INSERT_MAX / BA_WEIGHT_FEATURE;
    //const float s2rMax = BA_VARIANCE_PRIOR_ROTATION_INSERT_MAX / BA_WEIGHT_FEATURE;
    //const float r2Max = ME_VARIANCE_HUBER;
    const float r2Max = FLT_MAX;
    const int NZ = static_cast<int>(LF1.m_Zs.size());//最老帧观测到的所有关键帧
    for (int iZ = 0; iZ < NZ; ++iZ)
    {//遍历对关键帧的所有地图点观测

      const FRM::Measurement &Z = LF1.m_Zs[iZ];
      const int iKF = Z.m_iKF;
      const Depth::InverseGaussian *ds = m_ds.data() + m_iKF2d[iKF];//这个点对应的逆深度
      //const Depth::InverseGaussian *ds = m_dsGT->data() + m_solver->m_internal->m_iKF2d[iKF];
      const KeyFrame &KF = m_KFs[iKF];
        //在滑窗中,重投影误差由原来的Pcl = Rclw * (Rckw.t*Pck + twck - twcl) cl是局部普通帧,ck是这个地图点所属于的关键帧
        //改成了 Pcl = Rclrk * (Rckrk.t *Pck + trkck - trkcl ) 从world坐标系转到了参考关键帧坐标系下 ,rk是参考关键帧
      if (iKF == m_Zp.m_iKFr)//如果这个地图点所在的关键帧和当前的参考关键帧是同一帧
      {
        *Tk1 = Tr1;//Tc0(最老帧)_c0(参考关键帧),这里的次老帧是上一次边缘化时的次老帧,现在已经变成了最老帧
#ifdef CFG_STEREO
        Tk1[1] = Tk1[0];
        Tk1[1].SetTranslation(m_K.m_br + Tk1[0].GetTranslation());//Tc1(最老帧)_c0(参考关键帧)
#endif
        FTR::Factor::FixSource::L L;
        FTR::Factor::FixSource::A1 A1;
        FTR::Factor::FixSource::A2 A2;
        FTR::Factor::FixSource::U U;
        FTR::Factor::FixSource::M2 M;
        FTR::Factor::DD mdd;
        LA::AlignedVector6f mdcz;
        Camera::Factor::Unitary::CC SAczz;
        SAczz.MakeZero();
          //遍历所有对这个关键帧的观测到的地图点,构建最老帧和逆深度对应的H|-b，并且边缘化地图点,SAczz存储边缘化以后的S|-g
        for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz)
        {
          const FTR::Measurement &z = LF1.m_zs[iz];
          const int ix = z.m_ix;
          //在滑窗中,重投影误差由原来的Pcl = Rclw * (Rckw.t*Pck + twck - twcl) cl是局部普通帧,ck是这个地图点所属于的关键帧
          //改成了 Pcl = Rclrk * (Rckrk.t *Pck + trkck - trkcl ) 从world坐标系转到了参考关键帧坐标系下 ,rk是参考关键帧
           // 如果这个地图点所在的关键帧和当前的参考关键帧是同一帧,那么现在地图点所在的关键帧(参考关键帧就不需要优化,即FixSource)
          FTR::GetFactor<LBA_ME_FUNCTION>(BA_WEIGHT_FEATURE, Tk1/*Tc(最老帧)_c0(参考关键帧)*/, KF.m_xs[ix]/*关键帧对这个地图点的观测*/,
                  ds[ix]/*逆深度*/, Tr1/*Tc0(最老帧)_c0(参考关键帧)*/, z,/*最老帧对这个地图点的观测*/
                                          &L/*最老帧中对这个地图点观测的重投影误差e,J(对当前帧的pose,对关键帧点的逆深度),cost*/,
                                          &A1,/*m_adcz存的是当前这个最老帧pose和观测到的地图点在kf中逆深度的H*/
                                          &A2,//A2->m_add里的m_a就存的是逆深度和逆深度的H,逆深度的-b，A2->m_Aczz里的m_A存储最老帧pose和最老帧pose的H,m_b存储最老帧pose的-b
                                          &U,/*这个地图点从关键帧投影到当前帧上的重投影误差的因子,存了H|-b,信息矩阵*/
#ifdef CFG_STEREO
                                          m_K.m_br,
#endif
                                          r2Max);
          SAczz += A2.m_Aczz;//m_A存储最老帧pose和最老帧pose的H,m_b存储最老帧pose的-b
          A2.m_add.m_a += s2d;
#ifdef CFG_HISTORY
          if (m_history >= 2)
          {
            m_MH.Update1(z, L, A2.m_add, A1);
          }
#endif
          if (A2.m_add.m_a < epsd)//说明是一个无效的观测,不过无效观测基本就是H等于0了,对于SAczz也影响不大,所以也没减去
          {
            continue;
          }
          mdd.m_a = 1.0f / A2.m_add.m_a;//Huu^-1^-1 u代表地图点逆深度
          mdd.m_b = mdd.m_a * A2.m_add.m_b;//Huu^-1*bu
          FTR::Factor::FixSource::Marginalize(mdd, A1.m_adczA/*最老帧pose和观测到的地图点在kf中逆深度对应的Hpose_u*/,
                  &mdcz/*Hpose_u *Huu^-1|Huu^-1 * -bu*/, &M/*Hpose_u *Huu^-1 * Hpose_u.t|Hpose_u *Huu^-1 * -bu*/);
          SAczz -= M.m_Mczz;//边缘化这个观测点
        }
        //merge掉最老帧对于这个关键帧观测的地图点以后,将舒尔补以后的约束加到cc中
        m_Zp.Update(Nk, SAczz);//这个最老帧作为上一次边缘化时的次老帧,所以此时应是最后一个C的状态,之前有它和g,motion的先验,这里要加上merge了地图点以后的约束
#ifdef CFG_HISTORY
        if (m_history >= 2)
        {
          m_MH.Update2(SAczz);
        }
#endif
      } else
      {//如果观测到的这个关键帧不是参考关键帧
        const std::vector<int>::const_iterator it = std::lower_bound(m_Zp.m_iKFs.begin(),//在已经存在的关键帧里找找看能不能找到这个关键帧
                                                                     m_Zp.m_iKFs.end(), iKF);
        const int ik = static_cast<int>(it - m_Zp.m_iKFs.begin());
        if (it == m_Zp.m_iKFs.end() || *it != iKF)//如果没有找到这个关键的话,因为m_iKFs最后一维设了一个大数,所以它肯定返回的是m_Zp.m_iKFs.size()-2
        {
          //Rigid3D::ABI(m_CsKF[iKF], Tr, Trk);
          Rigid3D::ABI(m_CsKF[iKF]/*Tc0w(观测到的关键帧)*/, m_CsKF[m_Zp.m_iKFr]/*Tc0w(参考关键帧)*/, Trk/*Tc0(观测到的关键帧)c0(参考关键帧)*/);
        } else {
          m_Zp.m_Zps[ik].GetInverse(Trk);//如果找到的话就直接取出存储的Tc0(观测到的关键帧)_c0(参考关键帧)
        }
        Rigid3D::ABI(Tr1/*上一次边缘化算出的Tc0(最老帧)_c0(参考关键帧)*/, Trk/*Tc0(观测到的关键帧)_c0(参考关键帧)*/, *Tk1/*上一次边缘化算出的Tc0(最老帧)_c0(观测到的关键帧)*/);
#ifdef CFG_STEREO
        Tk1[1] = Tk1[0];/*上一次边缘化算出的Tc1(最老帧)_c0(观测到的关键帧)*/
        Tk1[1].SetTranslation(m_K.m_br + Tk1[0].GetTranslation());
#endif
        xp128f mdd;
        FTR::Factor::Full::L L;
        FTR::Factor::Full::A1 A1;
        FTR::Factor::Full::A2 A2;
        FTR::Factor::Full::U U;
        FTR::Factor::Full::Source::M1 M1;
        FTR::Factor::Full::Source::M2 M2;
        FTR::Factor::Full::M1 M3;
        FTR::Factor::Full::M2 M4;
        LA::ProductVector6f adcz;
        Camera::Factor::Unitary::CC SAcxx, SAczz;
        Camera::Factor::Binary::CC SAcxz;
        SAcxx.MakeZero();
        SAcxz.MakeZero();
        SAczz.MakeZero();
        for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz)
        {
          const FTR::Measurement &z = LF1.m_zs[iz];
          const int ix = z.m_ix;
            //在滑窗中,重投影误差由原来的Pcl = Rclw * (Rckw.t*Pck + twck - twcl) cl是局部普通帧,ck是这个地图点所属于的关键帧
            //改成了 Pcl = Rclrk * (Rckrk.t *Pck + trkck - trkcl ) 从world坐标系转到了参考关键帧坐标系下 ,rk是参考关键帧
            // 就是所有的pose都是相对于参考关键帧的,因为观测到的帧不是参考关键帧,所以这里对投影前后的帧的pose都要优化
            //A1,A2存储的如下:
            //A2->m_adx.m_add.m_a//逆深度x逆深度
            //A2->m_adx.m_adc//逆深度 x 投影前pose
            //A1->m_adcz//逆深度 x 投影后pose
            //A2->m_adx.m_add.m_b//逆深度的-b
            //A2->m_Acxx.m_A//投影前pose x 投影前pose
            //A2->m_Acxz//投影前pose x 投影后pose
            //A2->m_Acxx.m_b//投影前pose的-b
            //A2->m_Aczz.m_A//投影后pose x 投影后pose
            //A2->m_Aczz.m_b//投影后pose的-b

          FTR::GetFactor<LBA_ME_FUNCTION>(BA_WEIGHT_FEATURE, Tk1/*上一次边缘化算出的Tc(最老帧)_c0(观测到的关键帧)*/,
                  KF.m_xs[ix], ds[ix], Tr1/*上一次边缘化算出的Tc0(最老帧)_c0(参考关键帧)*/, z,
                                          &L, &A1, &A2, &U,
#ifdef CFG_STEREO
                                          m_K.m_br,
#endif
                                          r2Max);
          SAcxx += A2.m_Acxx;//投影前pose x 投影前pose的H 和投影前pose的-b
          SAcxz += A2.m_Acxz;//投影前pose x 投影后pose
          SAczz += A2.m_Aczz;//投影后pose x 投影后pose的H 和投影后pose的-b
          A2.m_adx.m_add.m_a += s2d;
#ifdef CFG_HISTORY
          if (m_history >= 2)
          {
//#ifdef CFG_DEBUG
#if 0
            const float addBkp = A2.m_adx.m_add.m_a;
            A2.m_adx.m_add.m_a = 0.0f;
#endif
            m_MH.Update1(iKF, ik, z, L, A2.m_adx, A1);
//#ifdef CFG_DEBUG
#if 0
            A2.m_adx.m_add.m_a = addBkp;
#endif
          }
#endif
          if (A2.m_adx.m_add.m_a < epsd)
          {
            continue;
          }
          mdd.vdup_all_lane(1.0f / A2.m_adx.m_add.m_a);
          FTR::Factor::Full::Source::Marginalize(mdd, A2.m_adx, &M1, &M2);//边缘化地图点
          FTR::Factor::Full::Marginalize(mdd, M1, A1, &M3, &M4, &adcz);
          SAcxx -= M2.m_Mcxx;//边缘化所有这帧的观测地图点以后的投影前pose x 投影前pose的H 和投影前pose的-b
          SAcxz -= M4.m_Mcxz;//边缘化所有这帧的观测地图点以后的投影前pose x 投影后pose
          SAczz -= M4.m_Mczz;//边缘化所有这帧的观测地图点以后的投影后pose x 投影后pose的H 和投影后pose的-b
        }
        if (it == m_Zp.m_iKFs.end() || *it != iKF)
        {
          if (LBA_MARGINALIZATION_CHECK_RANK)
          {//检查一下是否满秩
//#if 0
#if 1
            LA::AlignedMatrix12x12f A;
            A.Set(SAcxx.m_A, SAcxz, SAczz.m_A);
            //if (!A.DecomposeLDL(epsc)) {
            //  continue;
            //}
            const int rank = A.RankLDL(epsc);
            //UT::Print("[%d] --> [%d] %d\n", m_KFs[Z.m_iKF].m_Cam_pose.m_iFrm, LF1.m_Cam_pose.m_iFrm, rank);
            if (rank < 6) {
            //if (rank < 12) {
              continue;
            }
#else
            LA::AlignedMatrix6x6f S;
            LA::Vector3f s2p, s2r;
            SAcxx.m_A.GetAlignedMatrix6x6f(S);
            if (!S.InverseLDL(epsc)) {
              continue;
            }
            S.GetDiagonal(s2p, s2r);
            if (s2p.Maximal() > s2pMax || s2r.Maximal() > s2rMax) {
              continue;
            }
            SAczz.m_A.GetAlignedMatrix6x6f(S);
            if (!S.InverseLDL(epsc)) {
              continue;
            }
            S.GetDiagonal(s2p, s2r);
            if (s2p.Maximal() > s2pMax || s2r.Maximal() > s2rMax) {
              continue;
            }
#endif
          }
//#ifdef CFG_DEBUG
#if 0
          UT::DebugStart();
          const bool scc1 = m_Zp.Invertible(&m_work, _eps);
          LA::AlignedMatrixXf S;
          const bool scc2 = m_Zp.GetPriorMeasurement(1.0f, &S, NULL, &m_work, _eps);
          UT_ASSERT(scc1 == scc2);
          UT::DebugStop();
#endif
          const CameraPrior::Element::CC AczzBkp = m_Zp.m_Acc[Nk][Nk];//备份一下H的最老帧 x 最老帧
          const CameraPrior::Element::C bczBkp = m_Zp.m_bc[Nk];//备份一下b的最老帧

          //向m_Zp中插入这个没有存储过的观测关键帧:m_iKFs存储这个新加的关键帧,m_Zps存储Tc0(观测到的关键帧)c0(参考关键帧),这两种的索引是一致的
          m_Zp.Insert(BA_WEIGHT_PRIOR_CAMERA_INITIAL, ik/*要插入这个关键帧的位置*/, iKF/*观测到的关键帧的id*/, Trk,//Tc0(观测到的关键帧)c0(参考关键帧)
                      BA_VARIANCE_PRIOR_POSITION_NEW,
                      BA_VARIANCE_PRIOR_ROTATION_NEW, &m_work);
          ++Nk;
          //当新来了一个观测关键帧时,会将它放在H矩阵的最老帧前面,之前的观测关键帧之后,顺序即为g,N个观测关键帧,最老帧,更新他们之前地图点边缘化以后产生的约束
          m_Zp.Update(ik/*插入的位置*/, Nk/*当前H里所有的pose数量,会比观测的关键数量多1,就是最老帧的pose*/, SAcxx/*投影前pose x 投影前pose的H 和投影前pose的-b*/, SAcxz/*投影前pose x 投影后pose*/, SAczz/*投影后pose x 投影后pose的H 和投影后pose的-b*/);
#if 0
          if (!m_Zp.Invertible(&m_work, _eps)) {
//#ifdef CFG_DEBUG
#if 0
            UT_ASSERT(!scc1 && !scc2);
#endif
            m_Zp.Erase(ik);
            --Nk;
            m_Zp.m_Acc[Nk][Nk] = AczzBkp;
            m_Zp.m_bc[Nk] = bczBkp;
            continue;
          }
#endif
#ifdef CFG_HISTORY
          if (m_history >= 2)
          {
            m_MH.Insert(&m_work);
#ifdef CFG_GROUND_TRUTH
            if (m_CsGT)
            {
              m_MH.InsertGT(m_Zp, m_CsKFGT[m_Zp.m_iKFr], m_CsKFGT[iKF], &m_work);
            }
#endif
          }
#endif
        } else
        {
          m_Zp.Update(ik, Nk, SAcxx, SAcxz, SAczz);
        }
#ifdef CFG_HISTORY
        if (m_history >= 2)
        {
          m_MH.Update2(iKF, ik, SAcxx, SAcxz, SAczz);
        }
#endif
      }
    }
    if (iKFr == m_Zp.m_iKFr)//当前这帧的最近关键帧是之前的参考关键帧
    {
      //C1.m_Cam_pose.GetPosition(C1.m_p);
    } else//当前最老帧的最近关键帧不是当前的参考关键帧,那么就要切换参考关键帧
    {
      //Tr = m_CsKF[iKFr];
      //C1 = m_CsLF[iLF1];
      const Camera &C1 = m_CsLF[iLF1];//最老滑窗帧的pose Tc0w(lf)
      //m_ts[TM_TEST_2].Start();
      if (!m_Zp.GetPriorMotion(&m_ZpLF, &m_work, _eps) && LBA_MARGINALIZATION_CHECK_INVERTIBLE)
      {
        m_ZpLF.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL,
                          BA_VARIANCE_PRIOR_VELOCITY_RESET,
                          BA_VARIANCE_PRIOR_BIAS_ACCELERATION_RESET,
                          BA_VARIANCE_PRIOR_BIAS_GYROSCOPE_RESET, &C1);

      }
      //m_ts[TM_TEST_2].Stop();
      if ((m_Zp.GetPriorPose(INT_MAX, &m_ZpKF, &m_work, _eps) ||//边缘化最老帧的pose和motion,只保留g和观测关键帧的先验
           !LBA_MARGINALIZATION_CHECK_INVERTIBLE) &&
           m_ZpKF.MarginalizeUninformative(BA_WEIGHT_FEATURE,
                                           BA_VARIANCE_PRIOR_POSITION_INFORMATIVE,
                                           BA_VARIANCE_PRIOR_ROTATION_INFORMATIVE,
                                           &m_idxsTmp1, &m_work, _eps))
      {
        m_GBA->PushCameraPriorPose(iFrm, m_ZpKF);//向GBA里m_ZpKF的保存运动先验,同时m_ITs1保存了向GBA输入先验位姿的标签
      } else
      {
        m_ZpKF.Invalidate();
      }
      const Rigid3D &Tr = m_CsKF[iKFr];//重置m_Zp

      m_Zp.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL, iKFr, Tr,//重置m_zp
                      BA_VARIANCE_PRIOR_GRAVITY_NEW, m_ZpLF, false, &C1.m_Cam_pose,
                      BA_VARIANCE_PRIOR_POSITION_NEW, BA_VARIANCE_PRIOR_ROTATION_NEW);
      Rigid3D::ABI(C1.m_Cam_pose, Tr, Tr1);

#ifdef CFG_HISTORY
      if (m_history >= 2) {
        m_MH.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL, BA_VARIANCE_PRIOR_GRAVITY_NEW,
                        BA_VARIANCE_PRIOR_POSITION_NEW, BA_VARIANCE_PRIOR_ROTATION_NEW,
                        BA_VARIANCE_PRIOR_VELOCITY_NEW, BA_VARIANCE_PRIOR_BIAS_ACCELERATION_NEW,
                        BA_VARIANCE_PRIOR_BIAS_GYROSCOPE_NEW, m_ZpLF, &m_work, _eps, false);
#ifdef CFG_GROUND_TRUTH
        if (m_CsGT) {
          m_MH.PushGT(m_Zp, m_CsKFGT[iKFr], m_CsLFGT[iLF1]);
        }
#endif
      }
#endif
    }
    //const Camera &C2 = m_CsLF[iLF2];
    const Rigid3D &Tr = m_CsKF[iKFr];//当前最老帧的最近关键帧的Tc0w
    Rigid3D _Tr, TrI, Tr2;
    Camera C1/*最老帧*/, C2/*次老帧*/;//C1,2里的速度都是v_w(世界坐标系下)
    LA::AlignedVector3f v2;
    //为了保持滑窗的一致性,当前的参考帧位姿可能已经变化了,所以C1,C2都要用前一次的滑窗的相对位姿来计算
      //得到上一次滑窗时的参考关键帧Tc0w
    m_Zp.GetReferencePose(Tr/*当前最老帧的最近关键帧的Tc0w*/, &_Tr/*老的Tc0w(参考关键帧)*/, &TrI/*老的Twc0(参考关键帧)*/);
    //是用上一次边缘化算出的Tc0(最老帧)_c0(参考关键帧)*上一次的参考关键帧pose算出的C1.m_Cam_pose
    Rigid3D::ABI(Tr1/*上一次的边缘化求出的Tc0(最老帧)_c0(参考关键帧)*/, TrI/*上一次的Twc0(参考关键帧)*/, C1.m_Cam_pose/*最老帧pose*/);//C1.m_Cam_pose = Tc0w(最老帧)
    C1.m_Cam_pose.GetPosition(C1.m_p);//twc0(最老帧)
    //上一次边缘化的时候用次老帧算了在次老帧坐标系下的速度,这次边缘化,次老帧变成了最老帧,再用它的pose将速度转回到w坐标系下,存在C1.m_v中
    m_Zp.GetMotion(C1.m_Cam_pose/*Tc0w(最老帧)*/, &C1.m_v, &C1.m_ba, &C1.m_bw);
    C2 = m_CsLF[iLF2];//这次边缘化中次老帧的 Tc0w(次老帧)
    Rigid3D::ABI(C2.m_Cam_pose, Tr, Tr2);//Tr2 用当前的数据得到相对pose Tc0(次老帧)_c0(参考关键帧)

    C2.m_Cam_pose.ApplyRotation(C2.m_v, v2);//v2 = Rc0w * v_w (次老帧)
   //用上一次滑窗的参考帧pose算出与上次滑窗一致坐标系的次老帧pose
    Rigid3D::ABI(Tr2/*Tc0(次老帧)_c0(参考关键帧)*/, TrI/*上一次的的Twc0(参考关键帧)*/, C2.m_Cam_pose/*次老帧pose*/);

    C2.m_Cam_pose.GetPosition(C2.m_p);//twc0(次老帧)
    C2.m_Cam_pose.ApplyRotationInversely(v2, C2.m_v);//C2.m_v再用修正后的R转回w系 C2.m_v =Rc0w(次老帧).t * v2(次老帧坐标系下)

    IMU::Delta::Error e;
    IMU::Delta::Jacobian::RelativeLF J;
    IMU::Delta::Factor::Auxiliary::RelativeLF A;
    const IMU::Delta &D = m_DsLF[iLF2];//次老帧和最老帧之间的imu约束
    //当最老帧不是关键帧的时候,优化变量是g,C1,M1,C2,M2
    D.GetFactor(BA_WEIGHT_IMU/* * sd*/, C1/*最老帧的状态*/, C2/*次老帧的相机状态*/, m_K.m_pu/*tc0_i*/, _Tr/*上一次滑窗时的Tc0w(参考关键帧)*/
            , &e, &J, &A/*因子*/, BA_ANGLE_EPSILON);


      if (!m_Zp.PropagateLF(_Tr/*上一次滑窗时的Tc0w(参考关键帧)*/, C2/*次老帧的相机状态*/, A/*imu约束因子*/, &m_work, _eps) && LBA_MARGINALIZATION_CHECK_INVERTIBLE) {
      const Camera &_C2 = m_CsLF[iLF2];
      m_ZpLF.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL,
                        BA_VARIANCE_PRIOR_VELOCITY_RESET,
                        BA_VARIANCE_PRIOR_BIAS_ACCELERATION_RESET,
                        BA_VARIANCE_PRIOR_BIAS_GYROSCOPE_RESET, &_C2);

      m_Zp.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL, iKFr, Tr,
                      BA_VARIANCE_PRIOR_GRAVITY_RESET, m_ZpLF, false, &_C2.m_Cam_pose,
                      BA_VARIANCE_PRIOR_POSITION_RESET, BA_VARIANCE_PRIOR_ROTATION_RESET);

#ifdef CFG_HISTORY
      if (m_history >= 2) {
        m_MH.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL, BA_VARIANCE_PRIOR_GRAVITY_NEW,
                        BA_VARIANCE_PRIOR_POSITION_NEW, BA_VARIANCE_PRIOR_ROTATION_NEW,
                        BA_VARIANCE_PRIOR_VELOCITY_NEW, BA_VARIANCE_PRIOR_BIAS_ACCELERATION_NEW,
                        BA_VARIANCE_PRIOR_BIAS_GYROSCOPE_NEW, m_ZpLF, &m_work, _eps, false);
#ifdef CFG_GROUND_TRUTH
        m_MH.PushGT(m_Zp, m_CsKFGT[iKFr], m_CsLFGT[iLF2]);
#endif
      }
#endif
    }
#ifdef CFG_HISTORY
    else if (m_history >= 2) {
      m_MH.PropagateLF(D, J, e, A);
#ifdef CFG_GROUND_TRUTH
      if (m_CsGT) {
        m_MH.PushGT(m_Zp, m_CsKFGT[iKFr], m_CsLFGT[iLF2]);
      }
#endif
    }
#endif

  }//参考关键帧不是其自己
//#if 0
#if 1
  const int Nk = static_cast<int>(m_Zp.m_iKFs.size() - 1);
#ifdef CFG_DEBUG
  UT_ASSERT(m_Zp.m_iKFs[Nk] == INT_MAX);
#endif//为了保证正定么
  m_Zp.m_Acc[Nk][Nk].IncreaseDiagonal(UT::Inverse(BA_VARIANCE_PRIOR_POSITION_NEW,
                                                  BA_WEIGHT_PRIOR_CAMERA_INITIAL),
                                      UT::Inverse(BA_VARIANCE_PRIOR_ROTATION_NEW,
                                                  BA_WEIGHT_PRIOR_CAMERA_INITIAL));
  m_Zp.m_Amm.IncreaseDiagonal(UT::Inverse(BA_VARIANCE_PRIOR_VELOCITY_NEW,
                                          BA_WEIGHT_PRIOR_CAMERA_INITIAL),
                              UT::Inverse(BA_VARIANCE_PRIOR_BIAS_ACCELERATION_NEW,
                                          BA_WEIGHT_PRIOR_CAMERA_INITIAL),
                              UT::Inverse(BA_VARIANCE_PRIOR_BIAS_GYROSCOPE_NEW,
                                          BA_WEIGHT_PRIOR_CAMERA_INITIAL));
#endif
  //m_ts[TM_TEST_2].Start();
//每一次边缘化的最后,都会给m_ZpLF存储这次边缘化时最后的次老帧的Rc0w*vw(次老帧坐标系下的v),ba,bw,下一次取的时候就是最老帧的Rc0w*vw,ba,bw了
//获取g,c,m的先验矩阵,除了这次的次老帧的motion,其他的全部都merge掉,保存在m_ZpLF中
  if (!m_Zp.GetPriorMotion(&m_ZpLF, &m_work, _eps) && LBA_MARGINALIZATION_CHECK_INVERTIBLE) {
    m_ZpLF.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL,
                      BA_VARIANCE_PRIOR_VELOCITY_RESET,
                      BA_VARIANCE_PRIOR_BIAS_ACCELERATION_RESET,
                      BA_VARIANCE_PRIOR_BIAS_GYROSCOPE_RESET, &m_CsLF[iLF2]);

  }

  m_ApLF.MakeZero();
  m_ucmsLF[iLF2] |= LBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY |//次老帧在lba中需要更新速度,bias
                    LBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_ACCELERATION |
                    LBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_GYROSCOPE;
//#ifdef CFG_DEBUG
#if 0
  UT::PrintSeparator();
  m_ZpLF.m_Amm.Print();
  m_ZpLF.m_bm.Print();
#endif
//#ifdef CFG_DEBUG
#if 0
  const bool e = true;
  UT::PrintSeparator();
  m_Zp.m_Arr.Print(e);
  UT::PrintSeparator();
  m_Zp.m_Acc[0][0].Print(e);
  UT::PrintSeparator();
  m_Zp.m_Acm[0].Print(e);
  UT::PrintSeparator();
  m_Zp.m_Amm.Print(e);
#endif
}
//step1边缘化处理,注释看MarginalizeLocalFrame实现部分
//step2因为现在最老帧边缘化出去了,需要将它的观测信息删除,它的观测信息存在与它的子轨迹中,并且它后面STL的帧数内的子轨迹也可能存储了这帧的观测
//遍历最老帧观测到的所有的关键帧中的地图点,先找一下最老帧就近的那几帧有没有也具有这个地图点所在的关键帧观测信息的
//1如果在次老帧里找到了共视这个关键帧的，同时还共视这个地图点,那么就需要把最老帧为起点的子轨迹从KF和这几个次老帧的这个点的子轨迹中去掉
//并且要在这个地图点对应的KF里的H|-b和次老帧子轨迹里的H|b中减去约束
//2如果找到了共视帧,但并不是共视的这地图点,那么就将最老的这条子轨迹的起点设成这帧的ic索引,并在KF里的H|-b中减去约束
//step3遍历所有和最老帧相关的次老帧,遍历它的共视信息,因为它的共视信息会存储要计算舒尔补的东西，当它的共视信息里涉及到了pop老帧时的子轨迹的点
// ,那么就要在LF1.m_Zm.m_SmddsST[i]里减去最老帧的子轨迹所占的约束
//step4遍历最老帧追踪轨迹窗口以后的滑窗帧,遍历滑窗观测到的关键帧,如果这个关键帧里有被pop出去的子轨迹,
// 就遍历所有观测,如果观测的点有子轨迹被pop出去,就要整体前移一个索引
//step5遍历最老帧对关键帧的所有观测,如果这个关键帧里有要pop的子轨迹的话就这个点所有在这个子轨迹后面的子轨迹整体前移
//step6将次老帧的H|-b减去与最老帧imu约束所影响的H|-b约束,并更新所有关键帧的子轨迹起始和终止索引
void LocalBundleAdjustor::PopLocalFrame() {
#ifdef LBA_DEBUG_EIGEN
  m_ZpBkp = m_Zp;
#endif
  //m_ts[TM_TEST_1].Start();
  MarginalizeLocalFrame();//边缘化处理
  //m_ts[TM_TEST_1].Stop();
#ifdef LBA_DEBUG_EIGEN
  DebugMarginalizeLocalFrame();
#endif
  const int nLFs = static_cast<int>(m_LFs.size());//滑窗中的帧个数
  const int STL = std::min(nLFs, LBA_MAX_SLIDING_TRACK_LENGTH);
  const int iLF = m_ic2LF.front();//最老的这帧在m_LFs中的索引
  LocalFrame &LF = m_LFs[iLF];
  const int NZ = static_cast<int>(LF.m_Zs.size());//滑窗中最老的这帧观测到的关键帧
  //因为现在最老帧边缘化出去了,需要将它的观测信息删除,它的观测信息存在与它的子轨迹中,并且它后面STL的帧数内的子轨迹也可能存储了这帧的观测
  //遍历最老帧观测到的所有的关键帧中的地图点,先找一下最老帧就近的那几帧有没有也具有这个地图点所在的关键帧观测信息的
  //1如果在次老帧里找到了共视这个关键帧的，同时还共视这个地图点,那么就需要把最老帧为起点的子轨迹从KF和这几个次老帧的这个点的子轨迹中去掉
  //并且要在这个地图点对应的KF里的H|-b和次老帧子轨迹里的H|b中减去约束
  //2如果找到了共视帧,但并不是共视的这地图点,那么就将最老的这条子轨迹的起点设成这帧的ic索引,并在KF里的H|-b中减去约束
  for (int iZ = 0; iZ < NZ; ++iZ) {//遍历观测到的所有关键帧
    const FRM::Measurement &Z = LF.m_Zs[iZ];
    m_idxsListTmp.resize(STL);
    for (int i = 0; i < STL; ++i) {
      m_idxsListTmp[i].resize(0);
    }
    bool popST = false;
    const int iKF = Z.m_iKF;
    ubyte *uds = m_uds.data() + m_iKF2d[iKF];
    const ubyte pushFrm = m_ucsKF[iKF] & LBA_FLAG_FRAME_PUSH_TRACK;
    KeyFrame &KF = m_KFs[iKF];
    for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {//遍历对于关键帧的观测
      const int ix = LF.m_zs[iz].m_ix, iSTMin = KF.m_ix2ST[ix];//该地图点的第一个子轨迹id
      int icMin = -1;
      for (int _ic = 1; _ic < STL && icMin == -1; ++_ic) {//和滑窗中最老的帧在子窗口范围(5)里的,找和最老帧都共视到这个关键帧的次老帧id
        std::vector<int> &_iz2x = m_idxsListTmp[_ic];
        if (_iz2x.empty()) {//在较老帧中找到对于这个kf的观测
          MarkFeatureMeasurements(m_LFs[m_ic2LF[_ic]]/*和滑窗中最老的帧在子窗口范围(5)里的*/, iKF/*老帧观测的关键帧id*/, _iz2x);
        }
        if (_iz2x[ix] != -1) {
          icMin = _ic;//找到次老帧中和最老帧都共视到这个关键帧的次老帧id
        }
      }
      FTR::Factor::FixSource::A2 &Az = LF.m_Azs2[iz];
      Az.m_add.MakeMinus();//m_add里的是逆深度和逆深度的H,逆深度的-b取负,减去这个因子
      FTR::Factor::FixSource::A3 &AzST = LF.m_AzsST[iz];
      AzST.m_add.MakeMinus();//子轨迹的m_add里的是逆深度和逆深度的H,逆深度的-b取负
#ifdef CFG_DEBUG
      //UT_ASSERT(LF.m_Nsts[iz] == 1);
      UT_ASSERT(LF.m_STs[iz].m_ist1 == 0 && LF.m_STs[iz].m_ist2 == 1);
#endif
      if (icMin == -1 ||//如果没有共视或者(这个特征点的子轨迹数量大于1条并且这个点第二条子轨迹的起点是这个共视的次老帧(因为之前只是在找有没有和这关键帧共视的,不一定是刚好和这点共视的))
          (KF.m_ix2ST[ix + 1] - iSTMin > 1 && KF.m_STs[iSTMin + 1].m_icMin == icMin)) {
        popST = true;
        uds[ix] |= LBA_FLAG_TRACK_POP;//这个地图点的flags设成需要pop子轨迹
        KF.m_usST[iSTMin] |= LBA_FLAG_TRACK_POP;//最小的这条轨迹(起点是对应于最老帧)设成需要pop子轨迹
        FTR::Factor::DD &mddST = KF.m_MxsST[iSTMin].m_mdd;
        mddST.MakeMinus();//逆深度和逆深度的ST_Huu^-1,逆深度的ST_Huu^-1*-ST_bu取负,就是要减去这个因子
        const bool nonZero = !(KF.m_usST[iSTMin] & LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO);//nonZero代表H>0切已经更新过的老的有效因子
        if (icMin != -1) {//如果共视次老帧存在
          for (int _ic = 1; _ic < STL; ++_ic) {
            const int _iLF = m_ic2LF[_ic];
            LocalFrame &_LF = m_LFs[_iLF];
            std::vector<int> &_ix2z = m_idxsListTmp[_ic];
            if (_ix2z.empty()) {
              MarkFeatureMeasurements(_LF, iKF, _ix2z);
            }
            const int _iz = _ix2z[ix];//在m_ic2LF[_ic]这帧中的观测
            if (_iz == -1) {//如果没有观测到这个点
              continue;
            }
#ifdef CFG_DEBUG
            UT_ASSERT(_LF.m_STs[_iz].m_ist1 == 0 && _LF.m_STs[_iz].m_ist2 > 1);
#endif
            --_LF.m_STs[_iz].m_ist2;//老帧的所在轨迹要push出去,所以在这个最老帧小窗口中共视这个点的帧也要把起点为最老帧的删除
            if (nonZero) {
              _LF.m_SmddsST[_iz] += mddST;//减少了一条轨迹,那么就要在自轨迹里减去这个因子的影响
              _LF.m_ms[_iz] |= LBA_FLAG_MARGINALIZATION_UPDATE;
            }
          }
        }
        if (nonZero) {
//#ifdef CFG_DEBUG
#if 0
          if (iKF == 187 && ix == 1) {
            UT::PrintSeparator();
            const float dm = mdxST.m_add.m_a, Sm = KF.m_SmdxsST[ix].m_add.m_a;
            UT::Print("[-] %f + %f = %f\n", dm, Sm, dm + Sm);
          }
#endif
          KF.m_ms[ix] |= LBA_FLAG_MARGINALIZATION_UPDATE;
        }
      } else {//如果有共视的次老帧,但不是共视到这个地图点,那么就将这个点的最早轨迹设成当前这个共视的最早帧
        KF.m_STs[iSTMin].m_icMin = icMin;
        if (!pushFrm || !(KF.m_usST[iSTMin] & LBA_FLAG_TRACK_PUSH)) {
          KF.m_usST[iSTMin] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
          FTR::Factor::DD &SaddST = KF.m_AxsST[iSTMin].m_Sadd;//这个点的最早的轨迹需要减去最老帧的因子
//#ifdef CFG_DEBUG
#if 0
          if (iKF == 0 && ix == 4) {
            UT::Print("  SaddST = %e + %e = %e [%d]\n", AzST.m_add.m_a, SaddST.m_a,
                                                        AzST.m_add.m_a + SaddST.m_a, LF.m_Cam_pose.m_iFrm);
          }
#endif
          SaddST += AzST.m_add;
          if (SaddST.m_a < 0.0f) {//如果减完以后H小于0了,则需要更新
            //SaddST.MakeZero();
            uds[ix] |= LBA_FLAG_TRACK_UPDATE_DEPTH;
            m_ucsKF[iKF] |= LBA_FLAG_FRAME_UPDATE_DEPTH;
          }
        }
      }
      FTR::Factor::DD &Sadd = KF.m_Axs[ix].m_Sadd;//这个地图点的逆深度和逆深度的H,逆深度的-b
//#ifdef CFG_DEBUG
#if 0
      if (iKF == 21 && ix == 316) {
        UT::Print("-[%d] %d: [%d] %e + %e = %e\n", m_LFs[m_ic2LF.back()].m_Cam_pose.m_iFrm, -1, LF.m_Cam_pose.m_iFrm,
                  Sadd.m_a, Az.m_add.m_a, Sadd.m_a + Az.m_add.m_a);
      }
#endif
      Sadd += Az.m_add;//在总的地图点H|-b中减去这个最老帧的因子信息
      if (Sadd.m_a < 0.0f) {//需要更新深度
        //Sadd.MakeZero();
        uds[ix] |= LBA_FLAG_TRACK_UPDATE_DEPTH;
        m_ucsKF[iKF] |= LBA_FLAG_FRAME_UPDATE_DEPTH;
      }
      //m_F = -Az.m_F + m_F;
      uds[ix] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
    }
    if (Z.m_iz1 < Z.m_iz2) {
      m_ucsKF[iKF] |= LBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION;//这个关键帧的地图点更新了追踪信息
    }
    if (popST) {
      m_ucsKF[iKF] |= LBA_FLAG_FRAME_POP_TRACK;//这个关键帧里有地图点需要pop
    }
  }
  //遍历所有和最老帧相关的次老帧,遍历它的共视信息,因为它的共视信息会存储要计算舒尔补的东西，当它的共视信息里涉及到了pop老帧时的子轨迹的点
  // ,那么就要在LF1.m_Zm.m_SmddsST[i]里减去最老帧的子轨迹所占的约束
  const int icMax = STL - 1;
  for (int _ic = 1; _ic < icMax; ++_ic) {//遍历所有的次老帧
    const int iLF1 = m_ic2LF[_ic];
    LocalFrame &LF1 = m_LFs[iLF1];
    const int NI = int(LF1.m_Zm.m_Is.size());//次老帧的共视信息
    for (int iI = 0; iI < NI; ++iI) {//遍历次老帧的所有观测
      const MeasurementMatchLF::Index &I = LF1.m_Zm.m_Is[iI];
      if (!(m_ucsKF[I.m_iKF] & LBA_FLAG_FRAME_POP_TRACK)) {//KF里有需要pop的子轨迹的话刚才已经设成需要pop了
        continue;
      }
      const KeyFrame &KF = m_KFs[I.m_iKF];//次老帧和共视帧观测到的关键帧
      const int iLF2 = LF1.m_iLFsMatch[I.m_ik];
      const LocalFrame &LF2 = m_LFs[iLF2];//和次老帧的共视帧
      const int i1 = LF1.m_Zm.m_iI2zm[iI]/**/, i2 = LF1.m_Zm.m_iI2zm[iI + 1];//这两帧的所有的观测
      for (int i = i1; i < i2; ++i) {//遍历所有共视观测
        const int iz2 = LF1.m_Zm.m_izms[i].m_iz2, ix = LF2.m_zs[iz2].m_ix/*次老帧的共视帧的观测索引*/, iST = KF.m_ix2ST[ix];
        if (LF2.m_STs[iz2].m_ist1 != 0 || !(KF.m_usST[iST] & LBA_FLAG_TRACK_POP) ||//起点不是0或者不需要pop,或者是无效的观测,那么就不是要pop的老帧中的子轨迹
            (KF.m_usST[iST] & LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO)) {
          continue;
        }
        //LF1.m_Zm.m_SmddsST[i] = -KF.m_MxsST[iST].m_mdd.m_a + LF1.m_ZmLF.m_SmddsST[i];
        LF1.m_Zm.m_SmddsST[i] = KF.m_MxsST[iST].m_mdd.m_a + LF1.m_Zm.m_SmddsST[i];//需要减去老帧占有的着一份观测的约束
        LF1.m_Zm.m_ms[i] |= LBA_FLAG_MARGINALIZATION_UPDATE;
      }
    }
  }
  //遍历最老帧追踪轨迹窗口以后的滑窗帧,遍历滑窗观测到的关键帧,如果这个关键帧里有被pop出去的子轨迹,
  // 就遍历所有观测,如果观测的点有子轨迹被pop出去,就要整体前移一个索引
  for (int _ic = STL; _ic < nLFs; ++_ic)
  {
    LocalFrame &_LF = m_LFs[m_ic2LF[_ic]];
    const int _NZ = int(_LF.m_Zs.size());//滑窗的观测
    for (int _iZ = 0; _iZ < _NZ; ++_iZ)
    {
      const FRM::Measurement &_Z = _LF.m_Zs[_iZ];
      if (!(m_ucsKF[_Z.m_iKF] & LBA_FLAG_FRAME_POP_TRACK))
      {
        continue;
      }
      const KeyFrame &KF = m_KFs[_Z.m_iKF];//如果这个关键帧里有被pop出去的子轨迹,就遍历观测,如果观测的点有子轨迹被pop出去,就要整体前移一个索引
      const int _iz1 = _Z.m_iz1, _iz2 = _Z.m_iz2;
      for (int _iz = _iz1; _iz < _iz2; ++_iz) {
        if (KF.m_usST[KF.m_ix2ST[_LF.m_zs[_iz].m_ix]] & LBA_FLAG_TRACK_POP)
        {
          _LF.m_STs[_iz].Step();
        }
      }
    }
  }
  //遍历最老帧对关键帧的所有观测,如果这个关键帧里有要pop的子轨迹的话就这个点所有在这个子轨迹后面的子轨迹整体前移
  for (int iZ = 0; iZ < NZ; ++iZ) {
    const FRM::Measurement &Z = LF.m_Zs[iZ];
    if (!(m_ucsKF[Z.m_iKF] & LBA_FLAG_FRAME_POP_TRACK))//如果这个关键帧内的地图点的子轨迹没有pop的就跳过
    {
      continue;
    }
    KeyFrame &KF = m_KFs[Z.m_iKF];
    const int Nx = static_cast<int>(KF.m_xs.size());
    m_ix2STTmp.swap(KF.m_ix2ST);
    KF.m_ix2ST.resize(Nx + 1);//因为有子轨迹会被pop出去,需要重新建立索引
    KF.m_ix2ST[0] = 0;
    for (int ix = 0, iST = 0; ix < Nx; ++ix)//遍历关键帧的地图点
    {
      const int iST1 = m_ix2STTmp[ix]/*这个地图点对应的第一条子轨迹*/, iST2 = m_ix2STTmp[ix + 1];/*这个地图点对应的最后子轨迹截止位*/
      const bool popST = iST1 < iST2 && (KF.m_usST[iST1] & LBA_FLAG_TRACK_POP);//如果这个子轨迹需要被pop
      for (int _iST = popST ? iST1 + 1 : iST1; _iST < iST2; ++iST, ++_iST)
      {//m_ix2STTmp 和m_ix2ST之间的更新关系我有点忘记了
        KF.m_STs[iST] = KF.m_STs[_iST];//如果要pop的话就这个点所有在这个子轨迹后面的子轨迹整体前移
        KF.m_usST[iST] = KF.m_usST[_iST];
        KF.m_AxsST[iST] = KF.m_AxsST[_iST];
        KF.m_MxsST[iST] = KF.m_MxsST[_iST];
      }
      KF.m_ix2ST[ix + 1] = iST;
      if (popST && iST == KF.m_ix2ST[ix])
      {
        KF.m_Axs[ix] = KF.m_Axps[ix];
        KF.m_ms[ix] &= ~LBA_FLAG_MARGINALIZATION_NON_ZERO;
      }
    }
    const int NST = KF.m_ix2ST.back();
    if (NST == 0) {
      KF.m_STs.clear();
      KF.m_usST.clear();
      KF.m_AxsST.Clear();
      KF.m_MxsST.Clear();
    } else {
      KF.m_STs.resize(NST);
      KF.m_usST.resize(NST);
      KF.m_AxsST.Resize(NST);
      KF.m_MxsST.Resize(NST);
    }
  }

  //将次老帧的H|-b减去与最老帧imu约束所影响的H|-b约束,并更新所有关键帧的子轨迹起始和终止索引
  const int _iLF = m_ic2LF[1];//次老帧
  IMU::Delta::Factor &Ad = m_AdsLF[_iLF];//次老帧和最老帧之间的imu约束
  Ad.m_A22.MakeMinus();
  m_SAcusLF[_iLF] += Ad.m_A22.m_Acc;//后一帧pose自己和自己的H以及自己对应的-b减去最老的imu约束引起的cc部分的约束
  Camera::Factor &SAcm = m_SAcmsLF[_iLF];
  SAcm.m_Au.m_Acm += Ad.m_A22.m_Acm;//减去cm的约束
  SAcm.m_Au.m_Amm += Ad.m_A22.m_Amm;//减去mm的约束
  SAcm.m_Ab.MakeZero();//前后相关的约束随着最老帧的merge也没有了
  Ad.MakeZero();
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    KeyFrame &KF = m_KFs[iKF];
    const int NST = static_cast<int>(KF.m_STs.size());
    for (int iST = 0; iST < NST; ++iST) {
      KF.m_STs[iST].Step();
    }
  }
#if defined CFG_GROUND_TRUTH && defined CFG_HISTORY
  if (m_history >= 3) {
    MarkFeatureMeasurementsUpdateDepth(LF, m_ucsKFGT, m_udsGT);
  }
#endif
}
//向LBA加进普通帧帧信息
//step1:判断是否大于最大窗口size,如果大于了需要滑窗处理
//step2:对 比当前帧新5帧以内的帧 和 当前帧 进行共视的关联,存在较新帧的m_Zm中
//step3:遍历当前帧对于每个关键帧的观测,如果这个关键帧中的地图点的轨迹起点比现在5帧窗口的最老帧还要老,就需要将关键帧里这个点再生成一条起点在当前5帧小窗口的新的子轨迹
//step4:第2帧开始需要利用当前的imu测量进行预积分,算出前状态为本体坐标系下的当前状态的变动以及协方差。第3帧开始还会对前状态重新预积分,前前状态为本体坐标系下的前状态的协方差。
//step5:计算一下这帧对所有关键帧的观测的重投影误差,当关键帧时还会对新的地图点的左右目(如果都有的话)之间的观测做重投影误差
void LocalBundleAdjustor::_PushLocalFrame(const InputLocalFrame &ILF/*当前普通帧*/) {
#if 0
//#if 1
  if (!m_KFs.empty()) {
    const int iKF = 0;
    const KeyFrame &KF = m_KFs[iKF];
    if (!KF.m_xs.empty()) {
      UT::PrintSeparator('*');
      const int ix = 206;
      const int iST1 = KF.m_ix2ST[ix], iST2 = KF.m_ix2ST[ix + 1];
      for (int iST = iST1; iST < iST2; ++iST) {
        const KeyFrame::SlidingTrack &ST = KF.m_STs[iST];
        const int iFrmMin = m_LFs[m_ic2LF[ST.m_icMin]].m_Cam_pose.m_iFrm;
        const int iFrmMax = m_LFs[m_ic2LF[ST.m_icMax]].m_Cam_pose.m_iFrm;
        UT::Print("[%d, %d]\n", iFrmMin, iFrmMax);
      }
    }
  }
#endif
  const int nLFs1 = static_cast<int>(m_LFs.size());//滑窗内已经有的帧的数量
  if (static_cast<int>(m_LFs.size()) < LBA_MAX_LOCAL_FRAMES)
  {//如果滑窗中帧的数量小于滑床size,说明是刚才开始
    const int nLFs2 = nLFs1 + 1;//滑窗里的一些东西需要扩容
    m_ic2LF.push_back(nLFs1);
    m_LFs.resize(nLFs2);
    m_CsLF.Resize(nLFs2, true);
#ifdef CFG_GROUND_TRUTH
    if (m_CsGT) {
      m_CsLFGT.Resize(nLFs2, true);
    }
#endif
    m_ucsLF.resize(nLFs2);
    m_ucmsLF.resize(nLFs2);
#ifdef CFG_INCREMENTAL_PCG
    m_xcsLF.Resize(nLFs2, true);
    m_xmsLF.Resize(nLFs2, true);
#endif
    m_DsLF.Resize(nLFs2, true);
#ifdef CFG_GROUND_TRUTH
    if (m_CsGT) {
      m_DsLFGT.Resize(nLFs2, true);
    }
#endif
    m_AdsLF.Resize(nLFs2, true);
    m_AfpsLF.Resize(nLFs2, true);
    m_AfmsLF.Resize(nLFs2, true);
    m_SAcusLF.Resize(nLFs2, true);
    m_SMcusLF.Resize(nLFs2, true);
    m_SAcmsLF.Resize(nLFs2, true);
    m_UcsLF.resize(nLFs2, LM_FLAG_FRAME_DEFAULT);
#ifdef CFG_CHECK_REPROJECTION
    m_esLF.resize(nLFs2, std::make_pair(FLT_MAX, FLT_MAX));
#endif
  } else
  {//如果是大于了滑窗最大size,就需要边缘化老帧
      PopLocalFrame();
      const int iLF = m_ic2LF.front();//就是0-49循环在用,比如滑窗满了,那么0就跑到49的索引位去了,那么最新的局部滑窗帧的id就是0
      m_ic2LF.erase(m_ic2LF.begin());
      m_ic2LF.push_back(iLF);
  }

  const int iLF = m_ic2LF.back();//最新帧的id
  LocalFrame &LF = m_LFs[iLF];//构造滑窗中的当前帧
#if 0
//#if 1
  UT::Print("+ [%d]\n", LF.m_Cam_pose.m_iFrm);
#endif//保存imu测量,将m_zs每个关键帧内的地图点按局部id进行排序,扩容相关矩阵
  LF.Initialize(ILF/*当前普通帧*/, ILF.m_imu_measures/*左相机坐标系(Rc0_i*)中imu的测量值*/);
  while (!LF.m_Zs.empty() && LF.m_Zs.back().m_iKF >= static_cast<int>(m_KFs.size()))
  {
    LF.PopFrameMeasurement();//这种情况应该不会发生吧
  }
#ifdef CFG_DEBUG
  if (!LF.m_Zs.empty() && LF.m_Zs.back().m_iKF == static_cast<int>(m_KFs.size())) {
    UT_ASSERT(!m_IKFs2.empty() && m_IKFs2.front().m_Cam_pose == LF.m_Cam_pose);
  }
#endif
  std::vector<int> &iKF2X = m_idxsTmp1/*存储的是当前帧和关键帧的共视关系*/, &iX2z = m_idxsTmp2;
  PushFeatureMeasurementMatchesFirst(LF/*滑窗中的当前帧*/, iKF2X, iX2z);//存储一下数据关联,具体看注释吧还有m_idxsTmp1,m_idxsTmp2的注释
  const int nLFs2 = int(m_LFs.size())/*包含当前帧总共有多少帧*/, STL = std::min(nLFs2, LBA_MAX_SLIDING_TRACK_LENGTH/*5*/);//
  const int ic1 = nLFs2 - STL/*当帧数超过5次以后,会找前5新的局部帧*/, ic2 = nLFs2 - 1/*最新的一个局部帧*/;
  //对 比当前帧新5帧以内的帧 和 当前帧 进行共视的关联,存在较新帧的m_Zm中
  for (int _ic = ic1; _ic < ic2; ++_ic)
  {
    LocalFrame &_LF = m_LFs[m_ic2LF[_ic]];//较新的局部帧
    _LF.m_iLFsMatch.push_back(iLF);//会将它最近的LBA_MAX_SLIDING_TRACK_LENGTH个普通帧的滑窗内的id记录下来
    PushFeatureMeasurementMatchesNext(_LF/*较新帧*/, LF/*当前帧*/, iKF2X/*[i]非-1时表示第i个关键帧之前的共视帧有多少个地图点*/,
            iX2z/*储存的是当前这帧的共视帧的所有地图点数量,地图点被观测到,就记录这个地图点在m_zs存储的起始位置*/, _LF.m_Zm/*较新帧和之后几帧之间的匹配关联*/);
  }
  m_ucsLF[iLF] = LBA_FLAG_FRAME_UPDATE_CAMERA;//都先初始化为需要更新
  m_ucmsLF[iLF] = LBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION | LBA_FLAG_CAMERA_MOTION_UPDATE_POSITION |
                  LBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY |
                  LBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_ACCELERATION | LBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_GYROSCOPE;
  m_UcsLF[iLF] = LM_FLAG_FRAME_UPDATE_CAMERA_LF;
#if 0
  if (nLFs2 > 1) {
    m_ucsLF[m_ic2LF[ic2 - 1]] |= LBA_FLAG_FRAME_UPDATE_CAMERA;
  }
#endif
  const ubyte udFlagST = LBA_FLAG_TRACK_PUSH | LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;
  const int NZ = static_cast<int>(LF.m_Zs.size());//共视的关键帧个数
    //遍历对于每个关键帧的观测,如果这个关键帧中的地图点的轨迹起点比现在5帧窗口的最老帧还要老,就需要将
    //关键帧里这个点再生成一条起点在当前5帧小窗口的新的子轨迹,防止舒尔补以后的S过于稠密,以及重新线性化时的变动范围变小
  for (int iZ = 0; iZ < NZ; ++iZ)
  {
    const FRM::Measurement &Z = LF.m_Zs[iZ];//当前关键帧的观测
    bool pushST = false;
    ubyte *uds = m_uds.data() + m_iKF2d[Z.m_iKF];//这个关键帧逆深度相关的flags
    KeyFrame &KF = m_KFs[Z.m_iKF];
    m_marksTmp1.assign(KF.m_xs.size(), 0);//size==这个关键帧中新地图点的个数,如果追踪轨迹长度超过了LBA_MAX_SLIDING_TRACK_LENGTH,那么就设置为1
    for (int iz/*观测在LF的局部id*/ = Z.m_iz1; iz < Z.m_iz2; ++iz)//遍历输入这个关键帧的地图点观测,更新轨迹追踪
    {
      const int ix = LF.m_zs[iz].m_ix/*观测到的地图点的局部id*/, Nst = KF.CountSlidingTracks(ix)/*这个地图点有几条子轨迹*/, iSTMax = KF.m_ix2ST[ix + 1] - 1;//下一个地图点其实轨迹的前一条,那就是这个地图点最后一条子轨迹
      if (Nst > 0 && KF.m_STs[iSTMax].m_icMin >= ic1/*这个地图点的最新的一条子轨迹如果起点不在5帧的小窗口内*/)
      {
        LF.m_STs[iz].Set(Nst - 1, Nst);//设置一下普通帧对地图点的追踪轨迹,也就是这个LF的地图点的观测是从属于KF.m_STs中这个地图点的哪几条轨迹的
        KF.m_STs[iSTMax].m_icMax = ic2;//设置一下关键帧中地图点的追踪轨迹
      } else
      {//这个地图点的最新的一条子轨迹如果起点不在5帧的小窗口内,那么就需要增加子轨迹
          LF.m_STs[iz].Set(Nst, Nst + 1);// 当前追踪轨迹是属于要新建的子轨迹里的一个观测
          pushST = true;
          uds[ix] |= LBA_FLAG_TRACK_PUSH;//更新这个地图状态,需要子轨迹插入
          //uds[ix] |= LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;
          m_marksTmp1[ix] = 1;
      }
    }
    if (!pushST)//只要是m_icMin比ic要老就需要加一个新的子轨迹
    {//比如关键帧中一个地图点追踪轨迹是 0,1 ,那么当第6个LF来,即ic=1时还能追踪上时,就需要一条新的轨迹,从ic = 5时开始,这时候比它早5帧的是ic=1
      continue;//即从1-5之间再生成一个新的子轨迹,那就是1,5
    }
    m_ucsKF[Z.m_iKF] |= LBA_FLAG_FRAME_PUSH_TRACK;
    m_idxsListTmp.resize(STL);
    for (int i = 0; i < STL; ++i)
    {
      m_idxsListTmp[i].resize(0);
    }
    const int Nx = static_cast<int>(KF.m_xs.size());//这个关键帧所管理的地图点的数量
    m_ix2STTmp.swap(KF.m_ix2ST);  KF.m_ix2ST.resize(Nx + 1);//先存在中间变量中
    m_STsTmp.swap(KF.m_STs);      KF.m_STs.resize(0);
    m_usSTTmp.swap(KF.m_usST);    KF.m_usST.resize(0);
    m_AxsTmp.Swap(KF.m_AxsST);    KF.m_AxsST.Resize(0);
    m_MxsTmp.Swap(KF.m_MxsST);    KF.m_MxsST.Resize(0);
    for (int ix = 0; ix < Nx; ++ix)//遍历每一个地图点
    {
      const int iST1/*当前这个地图点的轨迹id*/ = m_ix2STTmp[ix], iST2 = m_ix2STTmp[ix + 1]/*后这个地图点的轨迹id*/, Nst/*0说明是一个轨迹*/ = iST2 - iST1;
      KF.m_ix2ST[ix] = static_cast<int>(KF.m_STs.size());
      KF.m_STs.insert(KF.m_STs.end(), m_STsTmp.begin() + iST1, m_STsTmp.begin() + iST2);
      KF.m_usST.insert(KF.m_usST.end(), m_usSTTmp.begin() + iST1, m_usSTTmp.begin() + iST2);
      KF.m_AxsST.Push(m_AxsTmp.Data() + iST1, Nst);
      KF.m_MxsST.Push(m_MxsTmp.Data() + iST1, Nst);
      //if (!(uds[ix] & LBA_FLAG_TRACK_PUSH))
      if (!m_marksTmp1[ix])//如果不需要拆分成子轨迹,就跳过
      {
        continue;
      }//针对这个地图点要增加一条新的子轨迹,索引顺序接在老的轨迹之后
      const int NST1 = static_cast<int>(KF.m_STs.size())/*新的轨迹所在的索引*/, NST2 = NST1 + 1;//需要扩容的size
      KF.m_STs.resize(NST2);
      KF.m_usST.push_back(udFlagST);
      KF.m_AxsST.Resize(NST2, true);  KF.m_AxsST[NST1].MakeZero();
      KF.m_MxsST.Resize(NST2, true);  KF.m_MxsST[NST1].MakeZero();
#ifdef CFG_DEBUG
      KF.m_MxsST[NST1].m_mdd.Invalidate();
#endif
      KeyFrame::SlidingTrack &ST = KF.m_STs[NST1];//新的子轨迹
      ST.Set(ic2);//现在是一个新的轨迹,所以终点就先设为当前帧的ic,下面要确定起点
      for (int _ic = ic1; _ic < ic2; ++_ic)//遍历之前的4帧,确定这个子轨迹在当前最新的5帧中的起点
      {
        const int _iLF = m_ic2LF[_ic];
        LocalFrame &_LF = m_LFs[_iLF];//对应的滑窗内较新帧
        std::vector<int> &_ix2z = m_idxsListTmp[_ic - ic1];
        if (_ix2z.empty())
        {//将这帧对这个关键帧中地图点的观测信息记录下来[地图点局部id] = 地图点的观测在帧中m_zs的位置
          MarkFeatureMeasurements(_LF/*滑窗内较新帧*/, Z.m_iKF/*当前地图点所属关键帧id*/, _ix2z);
        }
        const int _iz = _ix2z[ix];
        if (_iz == -1)//如果等于-1说明这个较新帧没有看到这个地图点
        {
          continue;
        }
        ST.m_icMin = std::min(ST.m_icMin, _ic);//确定这个子轨迹的起点
#ifdef CFG_DEBUG
        const LocalFrame::SlidingTrack &_ST = _LF.m_STs[_iz];
        UT_ASSERT(_ST.m_ist2 == Nst);
#endif
        ++_LF.m_STs[_iz].m_ist2;//这个应该就是更新这个局部帧它对这个观测都出现在哪几个轨迹里,比如原来轨迹是1条,那么它这里就是m_ist1=0,m_ist2=1
      }//当新增了一条这个地图掉子轨迹,这个观测还含在这条子轨迹里,那么它的终点轨迹id就会加1 变成m_ist1=0,m_ist2=2
    }
    KF.m_ix2ST[Nx] = static_cast<int>(KF.m_STs.size());//这个实际上是没有这个地图点的,但是还是赋值一下
  }
  m_CsLF[iLF] = ILF.m_Cam_state;//滑窗中保存这帧的Tc0w
#ifdef CFG_GROUND_TRUTH
  if (m_CsGT)
  {
    m_CsLFGT[iLF] = m_CsGT[LF.m_T.m_iFrm];
  }
#endif
#ifdef CFG_INCREMENTAL_PCG
  m_xcsLF[iLF].MakeZero();
  m_xmsLF[iLF].MakeZero();
#endif
  IMU::Delta &D = m_DsLF[iLF];//当前帧对应的预积分,保存了预积分以后的状态量,协方差,信息矩阵,以及对ba,bw的雅克比
  if (nLFs2 > 1 && !IMU_GRAVITY_EXCLUDED)
  {
    const int _iLF = m_ic2LF[nLFs2 - 2];//当前帧的前一帧
    const LocalFrame &_LF = m_LFs[_iLF];//前一帧
    const float _t = _LF.m_T.m_t;//前一帧时间戳
    //UT::DebugStart();
    IMU::PreIntegrate(LF.m_us/*当前帧和之前帧之间的imu测量*/, _t/*上一帧的时间戳*/, LF.m_T.m_t/*当前帧的时间戳*/,
            m_CsLF[_iLF]/*上一帧的状态*/, &D/*预积分部分*/, &m_work, true/*是否要输出雅克比*/,
                      &_LF.m_us.Back()/*上一帧最后一个imu测量*/, NULL, BA_ANGLE_EPSILON);
    //UT::DebugStop();
    if (nLFs2 > 2)
    {//这里就是重新算了上一帧的预积分,这次的a,w的测量均值不再是k k+1取均值,而是i,j取中指
      const int __iLF = m_ic2LF[nLFs2 - 3];//当前帧的前前帧
      IMU::Delta &_D = m_DsLF[_iLF];//前一帧对应的预积分,保存了预积分以后的状态量,协方差,信息矩阵,以及对ba,bw的雅克比
      IMU::PreIntegrate(_LF.m_us, m_LFs[__iLF].m_T.m_t, _t, m_CsLF[__iLF], &_D, &m_work, true,
                        &_D.m_u1/*前前帧的最后一个imu测量*/, &LF.m_us.Front()/*当前帧第一个imu测量*/, BA_ANGLE_EPSILON);
      m_ucsLF[_iLF] |= LBA_FLAG_FRAME_UPDATE_CAMERA;
      m_ucmsLF[_iLF] |= LBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION | LBA_FLAG_CAMERA_MOTION_UPDATE_POSITION |
                        LBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY |
                        LBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_ACCELERATION | LBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_GYROSCOPE;
    }
  } else
      {
    D.Invalidate();
  }
#ifdef CFG_GROUND_TRUTH
  if (m_CsGT)
  {
    if (nLFs2 > 1 && !IMU_GRAVITY_EXCLUDED)
    {
      const int _iLF = m_ic2LF[nLFs2 - 2];
      const LocalFrame &_LF = m_LFs[_iLF];
      const float _t = _LF.m_T.m_t;
      const Camera &_C = m_CsLFGT[_iLF];
//#ifdef CFG_DEBUG
#if 0
      UT::DebugStart();
#endif
      IMU::PreIntegrate(LF.m_us, _t, LF.m_T.m_t, _C, &m_DsLFGT[iLF],
                        &m_work, true, &_LF.m_us.Back(), NULL, BA_ANGLE_EPSILON);
//#ifdef CFG_DEBUG
#if 0
      UT::DebugStop();
      UT::PrintSeparator();
      const int N = LF.m_imu_measures.Size();
      for (int i = 0; i < N; ++i) {
        UT::Print("%d: %f\n", i, LF.m_imu_measures[i].t());
      }
      m_DsLFGT[iLF].Print();
#endif
#ifdef LBA_DEBUG_GROUND_TRUTH_MEASUREMENT
      D.DebugSetMeasurement(_C, m_CsLFGT[iLF], m_K.m_pu, BA_ANGLE_EPSILON);
      m_DsLFGT[iLF].DebugSetMeasurement(_C, m_CsLFGT[iLF], m_K.m_pu, BA_ANGLE_EPSILON);
#endif
      if (nLFs2 > 2)
      {
        const int __iLF = m_ic2LF[nLFs2 - 3];
        IMU::Delta &_D = m_DsLFGT[_iLF];
        IMU::PreIntegrate(_LF.m_us, m_LFs[__iLF].m_T.m_t, _t, m_CsLFGT[__iLF], &_D,
                          &m_work, true, &_D.m_u1, &LF.m_us.Front(), BA_ANGLE_EPSILON);
#ifdef LBA_DEBUG_GROUND_TRUTH_MEASUREMENT
        m_DsLF[_iLF].DebugSetMeasurement(m_CsLFGT[__iLF], _C, m_K.m_pu, BA_ANGLE_EPSILON);
        _D.DebugSetMeasurement(m_CsLFGT[__iLF], _C, m_K.m_pu, BA_ANGLE_EPSILON);
#endif
      }
    } else
        {
      m_DsLFGT[iLF] = D;
    }
  }
#ifdef CFG_HISTORY
  if (m_history >= 3)
  {
    MarkFeatureMeasurementsUpdateDepth(LF, m_ucsKFGT, m_udsGT);
  }
#endif
#endif
  if (LF.m_T.m_iFrm == 0)
  {   //如果是第一帧的需要設置一下pose,固定首幀的pose
    const LA::Vector3f s2r = LA::Vector3f::Get(BA_VARIANCE_FIX_ORIGIN_ROTATION_X,
                                               BA_VARIANCE_FIX_ORIGIN_ROTATION_Y,
                                               BA_VARIANCE_FIX_ORIGIN_ROTATION_Z);//yaw没有被固定
    const float s2p = BA_VARIANCE_FIX_ORIGIN_POSITION;
    m_Zo.Set(BA_WEIGHT_FIX_ORIGIN, s2r/*首帧R初始值*/, s2p/*首帧t初始值*/, m_CsLF[iLF].m_Cam_pose/*imu测出的左相机的Rc0w,但是twc0是000*/);
    m_Ao.MakeZero();
  }
  m_AdsLF[iLF].MakeZero();
  m_AfpsLF[iLF].MakeZero();
  m_AfmsLF[iLF].MakeZero();
  m_SAcusLF[iLF].MakeZero();
  m_SMcusLF[iLF].MakeZero();
  m_SAcmsLF[iLF].MakeZero();
  m_usKF.Push(ILF.m_imu_measures);//保存imu测量
#ifdef CFG_CHECK_REPROJECTION//计算一下这帧对所有关键帧的观测的重投影误差,当关键帧时还会对新的地图点的左右目(如果都有的话)之间的观测做重投影误差
  ComputeErrorFeature(&LF/*滑窗中的当前帧*/, m_CsLF[iLF].m_Cam_pose/*当前Tc0w*/, m_CsKF/*键帧左相机位姿*/, m_ds/*所有地图点的逆深度信息*/, &m_esLF[iLF].first);
#endif
}
//向LBA加进关键帧信息
void LocalBundleAdjustor::_PushKeyFrame(const GlobalMap::InputKeyFrame &IKF) {
  //Timer timer;
  //timer.Start();
  const int nKFs1/*关键帧的局部id,第几个关键帧,确认一下这个的id*/ = static_cast<int>(m_KFs.size()), nKFs2 = nKFs1 + 1;
  m_KFs.resize(nKFs2);//新来了一个关键帧,相关矩阵的扩容
  m_iFrmsKF.resize(nKFs2);
  m_CsKF.Resize(nKFs2, true);
  m_ucsKF.resize(nKFs2, LBA_FLAG_FRAME_UPDATE_CAMERA);
  m_UcsKF.resize(nKFs2, LM_FLAG_FRAME_UPDATE_CAMERA_KF);
#ifdef CFG_GROUND_TRUTH
  if (m_CsGT) {
    m_CsKFGT.Resize(nKFs2, true);
  }
#ifdef CFG_HISTORY
  if (m_history >= 3) {
    m_ucsKFGT.resize(nKFs2, LBA_FLAG_FRAME_DEFAULT);
  }
#endif
#endif
#ifdef CFG_HANDLE_SCALE_JUMP
  m_dsKF.resize(nKFs2, 0.0f);
#endif
  const int iKF = nKFs1;
  KeyFrame &KF = m_KFs[iKF];//构造最新的关键帧
  KF.Initialize(IKF);//m_KFs中KF的初始化,基本和普通帧的差不多,需要的某些矩阵的扩容
  m_iFrmsKF[iKF] = KF.m_T.m_iFrm;
  //KF.m_zs[368].m_z.Print(true);
  const int Nk = static_cast<int>(KF.m_iKFsMatch.size());//当前关键帧的共视的关键帧
#ifdef CFG_DEBUG
  for (int ik = 0; ik < Nk; ++ik) {
    UT_ASSERT(KF.m_iKFsMatch[ik] < iKF);
  }
#endif
  for (int ik = 0; ik < Nk; ++ik) {//遍历所有的共视关键帧,也需要在他们自己的共视关键帧数据关联内加上当前这个关键帧
    KeyFrame &_KF = m_KFs[KF.m_iKFsMatch[ik]];
    //_KF.InsertMatchKeyFrame(iKF);
#ifdef CFG_DEBUG
    UT_ASSERT(_KF.m_iKFsMatch.empty() || _KF.m_iKFsMatch.back() < iKF);
#endif
    _KF.m_iKFsMatch.push_back(iKF);
  }
  m_iKF2d.push_back(m_iKF2d.back());//
  m_dsBkp.resize(KF.m_zs.size());//观测到的地图点的逆深度备份
  const int NZ1 = static_cast<int>(KF.m_Zs.size());//这个关键帧直接观测到了多少个关键帧
  for (int iZ = 0; iZ < NZ1; ++iZ) {//遍历所有观测到的地图点,将地图点的逆深度备份在m_dsBkp中
    const FRM::Measurement &Z = KF.m_Zs[iZ];
    const Depth::InverseGaussian *ds = m_ds.data() + m_iKF2d[Z.m_iKF];
    for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
      m_dsBkp[iz] = ds[KF.m_zs[iz].m_ix];
    }
  }
  const ubyte udFlag1 = LBA_FLAG_TRACK_UPDATE_DEPTH | LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;//新的地图点需要更新深度,并且之前的更新为0
  const int NX = static_cast<int>(IKF.m_Xs.size());//新观测到的地图点
  for (int iX1 = 0, iX2 = 0; iX1 < NX; iX1 = iX2)
  {
    const int _iKF = IKF.m_Xs[iX1].m_iKF;//当前关键帧的id
    for (iX2 = iX1 + 1; iX2 < NX && IKF.m_Xs[iX2].m_iKF == _iKF; ++iX2) {}
    const int id = m_iKF2d[_iKF + 1], Nx/*地图点个数*/ = iX2 - iX1;//因为m_iKF2d存的是老地图点的个数,所以新地图的id就是从个数开始
    for (int jKF = _iKF; jKF <= iKF; ++jKF)
    {
      m_iKF2d[jKF + 1] += Nx;//用完了以后就加上新的地图点个数,为的是下一次的m_iKF2d.push_back(m_iKF2d.back());
    }
    m_ds.insert(m_ds.begin() + id, Nx, Depth::InverseGaussian());//地图点以及地图点所在关键帧都需要设置要更新深度的flags
    m_uds.insert(m_uds.begin() + id, Nx, udFlag1);
    m_Uds.insert(m_Uds.begin() + id, Nx, LM_FLAG_TRACK_UPDATE_DEPTH);
    m_ucsKF[_iKF] |= LBA_FLAG_FRAME_UPDATE_DEPTH;
    m_UcsKF[_iKF] |= LM_FLAG_FRAME_UPDATE_DEPTH;
#if defined CFG_GROUND_TRUTH && defined CFG_HISTORY
    if (m_history >= 3) {
      m_ucsKFGT[_iKF] |= LBA_FLAG_FRAME_UPDATE_DEPTH;
      m_udsGT.insert(m_udsGT.begin() + id, Nx, LBA_FLAG_TRACK_UPDATE_DEPTH);
    }
#endif
    const GlobalMap::Point *Xs = IKF.m_Xs.data() + iX1;//遍历地图点前的准备工作
    Depth::InverseGaussian *ds = m_ds.data() + id;//从当前第一个地图点id开始
    ubyte *uds = m_uds.data() + id;
    m_xsTmp.resize(Nx);
    std::vector<ubyte> &mcs = m_marksTmp1;
    mcs.assign(nKFs2, 0);//清空上一次的标识,赋值0
    for (int i = 0; i < Nx; ++i)
    {//遍历所有地图点将所有地图点的逆深度存到m_ds里,以及m_xsTmp里新地图点观测信息的存储
      const GlobalMap::Point &X = Xs[i];//当前地图点
      ds[i] = X.m_d;//深度信息赋值
      m_xsTmp[i] = X.m_x;//新地图点的观测信息
      const int Nz = static_cast<int>(X.m_zs.size());//应该是恒为0的,新的地图点不会有别的关键帧的观测
#ifdef LBA_FLAG_TRACK_MEASURE_KF
      if (Nz > 0) {
        uds[i] |= LBA_FLAG_TRACK_MEASURE_KF;
      }
#endif
      for (int j = 0; j < Nz; ++j)//如果有别的关键帧的观测,就在对应位置设成1
      {
        mcs[X.m_zs[j].m_iKF] = 1;
      }
    }
#ifdef CFG_HANDLE_SCALE_JUMP
    m_dsKF[_iKF] = AverageDepths(m_ds.data() + m_iKF2d[_iKF], m_iKF2d[_iKF + 1] - m_iKF2d[_iKF]);
#endif//这部分操作没看懂,并不会发生啊
    std::vector<int> &ik2KF = m_idxsTmp1, &iKF2k = m_idxsTmp2;
    ik2KF.resize(0);
    iKF2k.assign(nKFs2, -1);
    for (int jKF = 0; jKF < nKFs2; ++jKF)//这里好像没啥用吧,遍历所有的关键帧,如果也观测到当前关键帧的新地图点的话,就记录到ik2KF中
    {
      if (!mcs[jKF]) {
        continue;
      }
      iKF2k[jKF] = static_cast<int>(ik2KF.size());
      ik2KF.push_back(jKF);
    }
    const int Nk = static_cast<int>(ik2KF.size());
    m_zsListTmp.resize(Nk);
    for (int ik = 0; ik < Nk; ++ik)
    {
      m_zsListTmp[ik].resize(0);
    }
    KeyFrame &_KF = m_KFs[_iKF];//当前关键帧
    const int Nx1 = static_cast<int>(_KF.m_xs.size());
    for (int i = 0, ix = Nx1; i < Nx; ++i, ++ix)
    {//遍历所有地图点
      const GlobalMap::Point &X = Xs[i];
      const int Nz = static_cast<int>(X.m_zs.size());
      for (int j = 0; j < Nz; ++j)
      {
        const FTR::Measurement &z = X.m_zs[j];
        const int ik = iKF2k[z.m_iKF];
        m_zsListTmp[ik].push_back(z);
        m_zsListTmp[ik].back().m_ix = ix;
      }
    }
    std::vector<int> &iks = m_idxsTmp2;
    iks.resize(Nk);
    std::vector<ubyte> &mxs = m_marksTmp2;
    for (int ik2 = 0; ik2 < Nk; ++ik2)
    {
      const int iKF2 = ik2KF[ik2];
      KeyFrame &KF2 = m_KFs[iKF2];
      int &jk2 = iks[ik2];
      const std::vector<FTR::Measurement> &zs2 = m_zsListTmp[ik2];
      const int Nk2 = static_cast<int>(KF2.m_iKFsMatch.size());
      KF2.PushFeatureMeasurements(_iKF, zs2, &jk2, &m_work);
      if (static_cast<int>(KF2.m_iKFsMatch.size()) > Nk2)
      {
        _KF.InsertMatchKeyFrame(iKF2);
      }
      mxs.assign(Nx, -1);
      ubyte *_mxs = mxs.data() - Nx1;
      const int Nz2 = static_cast<int>(zs2.size());
      for (int i = 0; i < Nz2; ++i)
      {
        const int ix = zs2[i].m_ix;
#ifdef CFG_DEBUG
        UT_ASSERT(ix >= Nx1 && ix < Nx1 + Nx);
#endif
        _mxs[ix] = 1;
      }
      for (int ik1 = 0; ik1 < ik2; ++ik1)
      {
        const int iKF1 = ik2KF[ik1];
        const std::vector<FTR::Measurement> &zs1 = m_zsListTmp[ik1];
        const int Nz1 = static_cast<int>(zs1.size());
        bool found = false;
        for (int i = 0; i < Nz1; ++i)
        {
          const int ix = zs1[i].m_ix;
#ifdef CFG_DEBUG
          UT_ASSERT(ix >= Nx1 && ix < Nx1 + Nx);
#endif
          if (_mxs[ix]) {
            found = true;
            break;
          }
        }
        if (!found)
        {
          continue;
        }
        const std::vector<int>::iterator _jk2 = std::lower_bound(KF2.m_iKFsMatch.begin() + jk2,
                                                                 KF2.m_iKFsMatch.end(), iKF1);
        jk2 = static_cast<int>(_jk2 - KF2.m_iKFsMatch.begin());
        if (_jk2 != KF2.m_iKFsMatch.end() && *_jk2 == iKF1) {
          continue;
        }
        KF2.InsertMatchKeyFrame(iKF1, &_jk2);
        KeyFrame &KF1 = m_KFs[iKF1];
        int &jk1 = iks[ik1];
        const std::vector<int>::iterator _jk1 = std::lower_bound(KF1.m_iKFsMatch.begin() + jk1,
                                                                 KF1.m_iKFsMatch.end(), iKF2);
        jk1 = static_cast<int>(_jk1 - KF1.m_iKFsMatch.begin());
        KF1.InsertMatchKeyFrame(iKF2, &_jk1);
      }
    }
    _KF.PushFeatures(m_xsTmp/*所有新地图点（由当前关键帧产生）的观测*/);
  }
  std::vector<int> &iKF2X = m_idxsTmp1, &iX2z = m_idxsTmp2;
  PushFeatureMeasurementMatchesFirst(KF/*当前关键帧*/, iKF2X/*[i]表示第i个关键帧之前的共视帧有多少个地图点*/, iX2z/*地图点被观测到,就记录这个地图点在m_zs存储的起始位置*/);
  m_CsKF[iKF] = IKF.m_Cam_state.m_Cam_pose;//存储关键帧的pose
#ifdef CFG_GROUND_TRUTH
  if (m_CsGT) {
    m_CsKFGT[iKF] = m_CsGT[KF.m_T.m_iFrm].m_Cam_pose;
  }
#endif

//新来一个关键帧,需要和当前的滑窗内所有的帧建立共视关系。
//如果不是当前关键帧所对应的滑窗帧,遍历滑窗帧观测的所有的关键帧中的地图点，只要有一个地图点是和当前关键帧有共视,就在这个滑窗帧中更新共视关键帧关联
//如果是当前关键帧所对应的滑窗帧,这个关键帧所对应的滑窗普通帧也需要更新对于这个关键帧的观测信息,关键帧的最近关键帧设置成滑窗帧的最近关键帧,
//而滑窗的最近关键帧要改成当前这个最近关键帧,LF的子轨迹和观测,KF的子轨迹都需要初始化
  const ubyte udFlag2 = LBA_FLAG_TRACK_PUSH | LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;
  const int nLFs = static_cast<int>(m_ic2LF.size());//滑窗中普通帧的个数
  for (int ic = 0; ic < nLFs; ++ic) //遍历当前滑窗内的所有帧,从最老帧开始遍历
  {//遍历所有滑窗中的普通帧
    const int iLF = m_ic2LF[ic];//
    LocalFrame &LF = m_LFs[iLF];//滑窗中的第ic老的普通帧
#ifdef CFG_DEBUG
    UT_ASSERT(LF.m_Zs.empty() || LF.m_Zs.back().m_iKF < nKFs1);
#endif
    if (LF.m_T == KF.m_T)//这个应该是最后一个滑窗的帧了,就是当前这个关键帧对应的滑窗中的这帧
    {//如果这个普通帧是和关键帧相对应的这帧,如果有新地图点,这个关键帧所对应的滑窗普通帧也需要更新对于这个关键帧的观测信息
      const int Nx/*新地图点数*/ = static_cast<int>(KF.m_xs.size()), Nz/*当前关键帧观测到的地图点的数量*/ = static_cast<int>(LF.m_zs.size());
//#ifdef CFG_DEBUG
#if 0
      UT_ASSERT(static_cast<int>(KF.m_zs.size()) >= Nz);
#endif
      if (Nx > 0)
      {//如果有新地图点,这个关键帧所对应的滑窗普通帧也需要更新对于这个关键帧的观测信息
        LF.PushFrameMeasurement(nKFs1/*当前这个关键帧是第几个关键帧*/, Nx/*新地图点数*/);
      } else {
        bool found = false;
        const int NZ = static_cast<int>(LF.m_Zs.size());
        for (int iZ = 0; iZ < NZ && !found; ++iZ) {
          const FRM::Measurement &Z = LF.m_Zs[iZ];
          const int iX = iKF2X[Z.m_iKF];
          if (iX == -1)
          {
            continue;
          }
          const int *_ix2z = iX2z.data() + iX;
          for (int iz = Z.m_iz1; iz < Z.m_iz2 && !found; ++iz) {
            found = _ix2z[LF.m_zs[iz].m_ix] != -1;
          }
        }
        if (found)
        {
          LF.m_iKFsMatch.push_back(nKFs1);
        }
      }
      KF.m_iKFNearest = LF.m_T.m_iFrm == 0 ? 0 : LF.m_iKFNearest;//当关键帧和普通帧是同一帧时,那自然普通帧的关联最强的关键帧也就是当前关键帧关联最强的关键帧了
      LF.m_iKFNearest = nKFs1;//之前 LF.m_iKFNearest是不包含当前关键帧的,现在有了最强关键帧,那关联最强的就是本身作为关键帧的这帧了
      m_ucsKF[nKFs1] |= LBA_FLAG_FRAME_PUSH_TRACK;//有新增的点轨迹
      ubyte *uds = m_uds.data() + m_iKF2d[nKFs1];
      const GlobalMap::Point *Xs = IKF.m_Xs.data() + IKF.m_Xs.size() - Nx;
#ifdef CFG_DEBUG
      for (int ix = 0; ix < Nx; ++ix) {
        UT_ASSERT(Xs[ix].m_iKF == nKFs1);
      }
      if (static_cast<int>(IKF.m_Xs.size()) > Nx) {
        UT_ASSERT(Xs[-1].m_iKF < nKFs1);
      }
#endif
      for (int ix/*新地图点局部id*/ = 0, iz = Nz/*m_zs中新的地图点对应的索引*/; ix < Nx; ++ix, ++iz) {//遍历所有新地图点，设置观测信息,追踪信息
        const FTR::Source &x = KF.m_xs[ix];//为当前这个新的观测设置一下观测信息
        LF.m_zs[iz].Set(ix/*新地图点局部id*/, x.m_x/*Pc0归一化坐标*/, Xs[ix].m_W/*Pc0信息矩阵*/
#ifdef CFG_STEREO
                      , x.m_xr/*Pc1归一化坐标*/, x.m_Wr/*Pc1信息矩阵*/
#endif
                      );
        LF.m_STs[iz].Set(0, 1);//设置一下这个地图点的子轨迹,刚开始追踪
        KF.m_ix2ST[ix] = ix;
        uds[ix] |= udFlag2;
      }
      KF.m_ix2ST[Nx] = Nx;//更新这个地图点对应的子轨迹索引
      KF.m_STs.assign(Nx, KeyFrame::SlidingTrack(ic));//这个关键帧新的地图点所对应的子轨迹
      KF.m_usST.assign(Nx, udFlag2);
      KF.m_AxsST.Resize(Nx);  KF.m_AxsST.MakeZero();
      KF.m_MxsST.Resize(Nx);  KF.m_MxsST.MakeZero();
#ifdef CFG_DEBUG
      for (int ix = 0; ix < Nx; ++ix) {
        KF.m_MxsST[ix].m_mdd.Invalidate();
      }
#endif
    } else
    {
      bool found = false;
      const int NZ = static_cast<int>(LF.m_Zs.size());//这个滑窗帧观测到的所有关键帧
      for (int iZ = 0; iZ < NZ && !found; ++iZ)//遍历滑窗帧观测的所有的关键帧中的地图点，只要有一个地图点是和当前关键帧有共视,就在这个滑窗帧中更新共视关键帧关联
      {
        const FRM::Measurement &Z = LF.m_Zs[iZ];
        const int iX = iKF2X[Z.m_iKF];
        if (iX == -1)//等于-1时说明当前关键帧没有观测到这个
        {
          continue;
        }
        const int *_ix2z = iX2z.data() + iX;
        for (int iz = Z.m_iz1; iz < Z.m_iz2 && !found; ++iz)
        {
          found = _ix2z[LF.m_zs[iz].m_ix] != -1;
        }
      }
      if (found)
      {
        LF.m_iKFsMatch.push_back(nKFs1);
      }
    }
  }
  //const ubyte udFlag3 = LBA_FLAG_TRACK_MEASURE_KF;
  const bool ud = LBA_RESET_DEPTH_INFORMATION;//是否需要重置深度信息
  const ubyte udFlag3 = (ud ? LBA_FLAG_TRACK_UPDATE_DEPTH : LBA_FLAG_TRACK_DEFAULT)
#ifdef LBA_FLAG_TRACK_MEASURE_KF
                       | LBA_FLAG_TRACK_MEASURE_KF
#endif
                       ;
  const int NZ2 = static_cast<int>(KF.m_Zs.size());//遍历关键帧所有观测到的地图点,设置初始的轨迹flag
  for (int iZ = 0; iZ < NZ2; ++iZ)
  {
    const FRM::Measurement &Z = KF.m_Zs[iZ];
    if (ud)
    {
      m_ucsKF[Z.m_iKF] |= LBA_FLAG_FRAME_UPDATE_DEPTH;
    }
    ubyte *uds = m_uds.data() + m_iKF2d[Z.m_iKF];
    for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz)
    {
      uds[KF.m_zs[iz].m_ix] |= udFlag3;
    }
  }//向全局地图的m_Cs中放进这这个关键帧
  m_GM->LBA_PushKeyFrame(GlobalMap::Camera(IKF.m_Cam_state.m_Cam_pose/*当前帧左相机pose*/, KF.m_T.m_iFrm/*帧的id*/, GM_FLAG_FRAME_DEFAULT
#ifdef CFG_HANDLE_SCALE_JUMP
                                         , m_dsKF[iKF]
#endif
  ));
  const int N = static_cast<int>(std::lower_bound(m_usKF.Data(), m_usKF.End(), KF.m_T.m_t) -//从imu测量里找到比当前帧时间戳之后(>=)的imu数据索引
                                                  m_usKF.Data());
  m_usKF.Erase(N, m_usKFLast);//比当前帧还晚的imu数据不需要(不过前端已经进行过剔除了,如上一行所示),并且将imu测量给m_usKFLast
  m_GBA->PushKeyFrame(IKF/*当前关键帧*/, m_usKFLast/*在当前帧时间戳之前的imu测量*/, m_dsBkp//当前关键帧观测到的地图点逆深度的备份
#ifdef CFG_HANDLE_SCALE_JUMP
                    , m_dsKF[iKF]
#endif
  );
#if defined CFG_GROUND_TRUTH && defined CFG_HISTORY
  if (m_history >= 3) {
    MarkFeatureMeasurementsUpdateDepth(KF, m_ucsKFGT, m_udsGT);
  }
#endif
#ifdef CFG_CHECK_REPROJECTION
  m_esKF.resize(nKFs2);
  ComputeErrorFeature(&KF, m_CsKF[nKFs1], m_CsKF/*所有关键帧的pose*/, m_ds/*所有地图点的逆深度*/, &m_esKF[nKFs1].first/*当前关键帧的地图点平均重投影误差*/, nKFs1/*关键帧的局部id*/);
#endif
#ifdef LBA_DEBUG_VIEW
  if (g_viewer) {
    g_viewer->UpdateCurrentFrame();
  }
#endif
  //timer.Stop(true);
  //static double g_St = 0.0;
  //static int g_N = 0;
  //const double t = timer.GetAverageMilliseconds();
  //g_St += t;
  //++g_N;
  //UT::Print("[%d] LBA::PushKeyFrame = %f %f\n", IKF.m_Cam_pose.m_iFrm, t, g_St / g_N);
}

void LocalBundleAdjustor::DeleteKeyFrame(const int iKF) {
  //Timer timer;
  //timer.Start();
  const int iFrm = m_iFrmsKF[iKF];
  const int nKFs1 = static_cast<int>(m_KFs.size()), nKFs2 = nKFs1 - 1;
#ifdef CFG_DEBUG
  UT_ASSERT(iKF < nKFs2);
#endif
  const int Nd = static_cast<int>(m_KFs[iKF].m_xs.size());
  const int id1 = m_iKF2d[iKF], id2 = id1 + Nd;
  {
    FTR::Factor::DD daddST;
    KeyFrame &KF = m_KFs[iKF];
    const ubyte ucFlag = LBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION |
                         LBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION_KF;
    const ubyte udFlag = LBA_FLAG_TRACK_UPDATE_INFORMATION | LBA_FLAG_TRACK_UPDATE_INFORMATION_KF;
    const int NZ = static_cast<int>(KF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ) {
      const FRM::Measurement &Z = KF.m_Zs[iZ];
      const int _iKF = Z.m_iKF;
      ubyte *_uds = m_uds.data() + m_iKF2d[_iKF];
      KeyFrame &_KF = m_KFs[_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
        const int ix = KF.m_zs[iz].m_ix;
        FTR::Factor::Depth &A = KF.m_Azs[iz];
        A.m_add.MakeMinus();
        _KF.m_Axps[ix].m_Sadd += A.m_add;
        _KF.m_Axs[ix].m_Sadd += A.m_add;
        _uds[ix] |= udFlag;
        FTR::Factor::FixSource::Source::A &AST = _KF.m_AxpsST[ix];
        daddST = AST.m_Sadd;
        AST = _KF.m_Axps[ix];
        const int iST1 = _KF.m_ix2ST[ix], iST2 = _KF.m_ix2ST[ix + 1], Nst = iST2 - iST1;
        if (Nst > 1) {
          _KF.m_AxpsST[ix] *= 1.0f / Nst;
        }
        FTR::Factor::DD::amb(AST.m_Sadd, daddST, daddST);
        for (int iST = iST1; iST < iST2; ++iST) {
          _KF.m_AxsST[iST].m_Sadd += daddST;
          _KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
        }
      }
      if (Z.m_iz1 < Z.m_iz2) {
        m_ucsKF[_iKF] |= ucFlag;
      }
    }
    //const int Nk = static_cast<int>(KF.m_iKFsMatch.size());
    //for (int ik = 0; ik < Nk; ++ik) {
    //  const int jKF = KF.m_iKFsMatch[ik];
    //  if (jKF > iKF) {
    //    break;
    //  }
    //  m_KFs[jKF].DeleteMatchKeyFrame(iKF);
    //}
  }
  for (int jKF = 0; jKF < iKF; ++jKF) {
    m_KFs[jKF].DeleteMatchKeyFrame(iKF);
  }
  for (int jKF = iKF + 1; jKF < nKFs1; ++jKF) {
    m_KFs[jKF].DeleteKeyFrame(iKF);
  }
  m_KFs.erase(m_KFs.begin() + iKF);
  m_iFrmsKF.erase(m_iFrmsKF.begin() + iKF);
  for (int jKF = iKF + 1; jKF <= nKFs1; ++jKF) {
    m_iKF2d[jKF] -= Nd;
  }
  m_iKF2d.erase(m_iKF2d.begin() + iKF);
  m_CsKF.Erase(iKF);
  m_ucsKF.erase(m_ucsKF.begin() + iKF);
  m_UcsKF.erase(m_UcsKF.begin() + iKF);
#ifdef CFG_GROUND_TRUTH
  if (m_CsGT) {
    m_CsKFGT.Erase(iKF);
  }
#ifdef CFG_HISTORY
  if (m_history >= 3) {
    m_ucsKFGT.erase(m_ucsKFGT.begin() + iKF);
  }
#endif
#endif
#ifdef CFG_HANDLE_SCALE_JUMP
  m_dsKF.erase(m_dsKF.begin() + iKF);
#endif
  m_ds.erase(m_ds.begin() + id1, m_ds.begin() + id2);
  m_uds.erase(m_uds.begin() + id1, m_uds.begin() + id2);
  //m_Uds.erase(m_Uds.begin() + id1, m_Uds.begin() + id2);
  const int nLFs = static_cast<int>(m_LFs.size());
  m_iLF2Z.resize(nLFs);
  for (int iLF = 0; iLF < nLFs; ++iLF) {
    LocalFrame &LF = m_LFs[iLF];
    m_iLF2Z[iLF] = std::lower_bound(LF.m_Zs.begin(), LF.m_Zs.end(), iKF);
  }
  for (int iLF1 = 0; iLF1 < nLFs; ++iLF1) {
    LocalFrame &LF1 = m_LFs[iLF1];
    const std::vector<FRM::Measurement>::iterator iZ1 = m_iLF2Z[iLF1];
    const bool z1 = iZ1 != LF1.m_Zs.end() && iZ1->m_iKF == iKF;
    const int Nk = static_cast<int>(LF1.m_iLFsMatch.size());
    for (int ik = 0; ik < Nk; ++ik) {
      const int iLF2 = LF1.m_iLFsMatch[ik];
      const std::vector<FRM::Measurement>::iterator iZ2 = m_iLF2Z[iLF2];
      const bool z2 = iZ2 != m_LFs[iLF2].m_Zs.end() && iZ2->m_iKF == iKF;
      if (!z1 && !z2) {
        continue;
      }
      std::vector<FTR::Measurement::Match> &izms = LF1.m_Zm.m_izms;
      const int i1 = LF1.m_Zm.m_ik2zm[ik], i2 = LF1.m_Zm.m_ik2zm[ik + 1];
      if (z1) {
        const int Nz1 = iZ1->CountFeatureMeasurements();
        for (int i = i2 - 1; i >= i1 && izms[i].m_iz1 >= iZ1->m_iz2; --i) {
          izms[i].m_iz1 -= Nz1;
        }
      }
      if (z2) {
        const int Nz2 = iZ2->CountFeatureMeasurements();
        for (int i = i2 - 1; i >= i1 && izms[i].m_iz2 >= iZ2->m_iz2; --i) {
          izms[i].m_iz2 -= Nz2;
        }
      }
    }
  }
  for (int iLF = 0; iLF < nLFs; ++iLF) {
    LocalFrame &LF = m_LFs[iLF];
    const std::vector<FRM::Measurement>::iterator iZ = m_iLF2Z[iLF];
    if (iZ != LF.m_Zs.end() && iZ->m_iKF == iKF) {
      Camera::Factor::Unitary::CC &SAczz = m_SAcusLF[iLF], &SMczz = m_SMcusLF[iLF];
      for (int iz = iZ->m_iz1; iz < iZ->m_iz2; ++iz) {
        Camera::Factor::Unitary::CC &Aczz = LF.m_Azs2[iz].m_Aczz;
        Aczz.MakeMinus();
        SAczz += Aczz;
        if (!(LF.m_ms[iz] & LBA_FLAG_MARGINALIZATION_NON_ZERO)) {
          continue;
        }
        Camera::Factor::Unitary::CC &Mczz = LF.m_Mzs2[iz].m_Mczz;
        Mczz.MakeMinus();
        SMczz += Mczz;
      }
    }
    LF.DeleteKeyFrame(iKF, &iZ);
    if (LF.m_iKFNearest != -1) {
      continue;
    }
    if (LBA_MARGINALIZATION_REFERENCE_NEAREST) {
      ubyte first = 1;
      int iKFNearest = -1;
      float imgMotionNearest = FLT_MAX;
      const Rigid3D &C = m_CsLF[iLF].m_Cam_pose;
      const float z = 1.0f / LF.m_d.u();
      m_marksTmp1.assign(nKFs2, 0);
      const int Nk = static_cast<int>(LF.m_iKFsMatch.size());
      for (int i = 0; i < Nk; ++i) {
        m_marksTmp1[LF.m_iKFsMatch[i]] = 1;
      }
      for (int jKF = 0; jKF < nKFs2 && m_KFs[jKF].m_T < LF.m_T; ++jKF) {
        if (Nk > 0 && !m_marksTmp1[jKF]) {
          continue;
        }
        const Rigid3D _C = m_CsKF[jKF];
        const float imgMotion = ComputeImageMotion(z, C, _C, &first);
        if (imgMotion > imgMotionNearest) {
          continue;
        }
        imgMotionNearest = imgMotion;
        iKFNearest = jKF;
      }
      LF.m_iKFNearest = iKFNearest;
    } else {
      LF.m_iKFNearest = iKF - 1;
    }
  }
  m_Zp.DeleteKeyFrame(iKF);
  if (m_Zp.Pose::Invalid()) {
    const int iLF = m_ic2LF.front(), iKFr = m_LFs[iLF].m_iKFNearest;
#ifdef LBA_DEBUG_GROUND_TRUTH_MEASUREMENT
    if (m_CsGT) {
      m_ZpLF.DebugSetMeasurement(m_CsLFGT[iLF]);
    }
#endif
    if (m_LFs[iLF].m_T.m_iFrm == m_KFs[iKFr].m_T.m_iFrm) {
      m_Zp.Initialize(m_ZpLF);
    } else {
      m_Zp.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL, iKFr, m_CsKF[iKFr],
                      BA_VARIANCE_PRIOR_GRAVITY_NEW, m_ZpLF, false, &m_CsLF[iLF].m_Cam_pose,
                      BA_VARIANCE_PRIOR_POSITION_NEW, BA_VARIANCE_PRIOR_ROTATION_NEW);
#ifdef LBA_DEBUG_GROUND_TRUTH_MEASUREMENT
      if (m_CsGT) {
        m_Zp.Pose::DebugSetMeasurement(m_CsKFGT[iKFr], false, &m_CsLFGT[iLF].m_Cam_pose);
      }
#endif
    }
  }
#if defined CFG_GROUND_TRUTH && defined CFG_HISTORY
  if (m_history >= 3) {
    m_udsGT.erase(m_udsGT.begin() + id1, m_udsGT.begin() + id2);
  }
#endif
  m_GM->LBA_DeleteKeyFrame(iFrm, iKF);
  m_GBA->PushDeleteKeyFrame(iFrm, iKF);
  //if (iKF == nKFs2) {
  //  m_usKF.Insert(0, m_usKFLast, &m_work);
  //}
#ifdef CFG_CHECK_REPROJECTION
  m_esKF.erase(m_esKF.begin() + iKF);
#endif
#ifdef LBA_DEBUG_VIEW
  if (g_viewer) {
    g_viewer->DeleteKeyFrame(iFrm);
  }
#endif
//#ifdef CFG_DEBUG
#if 0
  if (m_debug) {
    AssertConsistency(true, false);
  }
#endif
  //timer.Stop(true);
  //UT::Print("[%d] LBA::DeleteKeyFrame = %f\n", iFrm, timer.GetAverageMilliseconds());
}

void LocalBundleAdjustor::DeleteMapPoints(const std::vector<int> &ids) {
  //Timer timer;
  //timer.Start();
//#ifdef CFG_DEBUG
#if 0
  if (m_debug) {
    AssertConsistency();
  }
#endif
  const int nKFs = static_cast<int>(m_KFs.size());
  std::vector<ubyte> &mcs = m_marksTmp1;
  mcs.assign(nKFs, 0);
  std::vector<ubyte> &mds = m_marksTmp2;
  mds.assign(m_ds.size(), 0);
  const int N = static_cast<int>(ids.size());
#ifdef CFG_DEBUG
  UT_ASSERT(N > 0);
  for (int i = 1; i < N; ++i) {
    UT_ASSERT(ids[i - 1] < ids[i]);
  }
#endif
  for (int i1 = 0, i2 = 0, iKF = 0; i1 < N; i1 = i2) {
    iKF = static_cast<int>(std::upper_bound(m_iKF2d.begin() + iKF, m_iKF2d.end(), ids[i1]) -
                                            m_iKF2d.begin()) - 1;
    const int id2 = m_iKF2d[iKF + 1];
#ifdef CFG_DEBUG
    UT_ASSERT(ids[i1] >= m_iKF2d[iKF] && ids[i1] < id2);
#endif
    mcs[iKF] = 1;
    for (i2 = i1 + 1; i2 < N && ids[i2] < id2; ++i2);
    for (int i = i1; i < i2; ++i) {
      const int id = ids[i];
#ifdef CFG_DEBUG
      UT_ASSERT((m_uds[id] & LBA_FLAG_TRACK_INVALID) == 0);
#endif
      if (m_uds[id] & LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO) {
        m_uds[id] = LBA_FLAG_TRACK_INVALID | LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;
      } else {
        m_uds[id] = LBA_FLAG_TRACK_INVALID;
      }
      mds[id] = 1;
    }
  }
  std::vector<int> &izsDel = m_idxsTmp1, &izs = m_idxsTmp2;
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    KeyFrame &KF = m_KFs[iKF];
    const int id = m_iKF2d[iKF], Nx = static_cast<int>(KF.m_xs.size());
#ifdef CFG_DEBUG
    UT_ASSERT((mcs[iKF] != 0) == UT::VectorExistFlag<ubyte>(mds.data() + id, Nx, 1));
#endif
    if (mcs[iKF]) {
      KF.InvalidateFeatures(mds.data() + id);
    }
    izsDel.resize(0);
    const int NZ = static_cast<int>(KF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ) {
      const FRM::Measurement &Z = KF.m_Zs[iZ];
      if (!mcs[Z.m_iKF]) {
        continue;
      }
      const ubyte *mxs = mds.data() + m_iKF2d[Z.m_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
        if (mxs[KF.m_zs[iz].m_ix]) {
          izsDel.push_back(iz);
        }
      }
    }
    if (izsDel.empty()) {
      continue;
    }
    KF.DeleteFeatureMeasurementsPrepare(izsDel, &izs);
    KF.DeleteFeatureMeasurements(izs);
  }

  const int nLFs = static_cast<int>(m_LFs.size());
  const int STL = std::min(nLFs, LBA_MAX_SLIDING_TRACK_LENGTH);
  std::vector<std::vector<int> > &izsList = m_idxsListTmp;
  izsList.resize(STL);
//#ifdef CFG_DEBUG
#if 0
  std::vector<int> &ics = m_idxsTmp2;
  ics.resize(STL);
#endif
  for (int ic = nLFs - 1, il = 0; ic >= 0; --ic, il = (il + STL - 1) % STL) {
    const int iLF = m_ic2LF[ic];
    LocalFrame &LF = m_LFs[iLF];
    izsDel.resize(0);
    Camera::Factor::Unitary::CC &SAczz = m_SAcusLF[iLF], &SMczz = m_SMcusLF[iLF];
    const int NZ = static_cast<int>(LF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ) {
      const FRM::Measurement &Z = LF.m_Zs[iZ];
      if (!mcs[Z.m_iKF]) {
        continue;
      }
      const ubyte *mxs = mds.data() + m_iKF2d[Z.m_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
        if (!mxs[LF.m_zs[iz].m_ix]) {
          continue;
        }
        izsDel.push_back(iz);
        Camera::Factor::Unitary::CC &Aczz = LF.m_Azs2[iz].m_Aczz;
        Aczz.MakeMinus();
        SAczz += Aczz;
        if (!(LF.m_ms[iz] & LBA_FLAG_MARGINALIZATION_NON_ZERO)) {
          continue;
        }
        Camera::Factor::Unitary::CC &Mczz = LF.m_Mzs2[iz].m_Mczz;
        Mczz.MakeMinus();
        SMczz += Mczz;
      }
    }
//#ifdef CFG_DEBUG
#if 0
    ics[il] = ic;
#endif
    std::vector<int> &izs = izsList[il];
    if (izsDel.empty()) {
      izs.resize(0);
    } else {
      LF.DeleteFeatureMeasurementsPrepare(izsDel, &izs);
      LF.DeleteFeatureMeasurements(izs);
    }
    const int NI = static_cast<int>(LF.m_Zm.m_Is.size());
    for (int iI = 0; iI < NI; ++iI) {
      const MeasurementMatchLF::Index &I = LF.m_Zm.m_Is[iI];
      //if (!mcs[I.m_iKF]) {
      //  continue;
      //}
      const int ik = I.m_ik, _il = (il + ik + 1) % STL;
//#ifdef CFG_DEBUG
#if 0
      UT_ASSERT(ics[_il] == ic + ik + 1);
#endif
      LF.m_Zm.DeleteFeatureMeasurementMatches(iI, izs, izsList[_il]);
//#ifdef CFG_DEBUG
#if 0
      const int _iI = iI + 1;
      if (_iI < NI && LF.m_Zm.m_Is[_iI].m_ik == ik + 1) {
        LF.m_Zm.AssertConsistency(ik, LF, m_LFs[LF.m_iLFsMatch[ik]], m_izmsTmp);
      }
#endif
    }
//#ifdef CFG_DEBUG
#if 0
    if (ic == 1) {
      UT::DebugStart();
      const int ik = 0;
      const int iLF = LF.m_iLFsMatch[ik];
      LF.m_Zm.AssertConsistency(ik, LF, m_LFs[iLF], m_izmsTmp);
      UT::DebugStop();
    }
#endif
  }
  m_GBA->PushDeleteMapPoints(m_LFs[m_ic2LF.back()].m_T.m_iFrm, ids);
#if 0
  UT::DebugStart();
  m_GBA->WakeUp();
  UT::DebugStop();
#endif
//#ifdef CFG_DEBUG
#if 0
  if (m_debug) {
    AssertConsistency();
  }
#endif
  //timer.Stop(true);
  //UT::Print("LBA::DeleteMapPoints = %f\n", timer.GetAverageMilliseconds());
}

void LocalBundleAdjustor::MergeMapPoints(const std::vector<std::pair<int, int> > &ids) {
}

void LocalBundleAdjustor::UpdateCameras(const std::vector<GlobalMap::InputCamera> &Cs) {
  std::vector<int>::iterator i = m_iFrmsKF.begin();
  const int N = static_cast<int>(Cs.size()), nKFs = static_cast<int>(m_KFs.size());
  m_CsKFBkp.Resize(nKFs);
  m_marksTmp1.assign(nKFs, GM_FLAG_FRAME_DEFAULT);
  for (int j = 0; j < N; ++j) {
    const GlobalMap::InputCamera &C = Cs[j];
    i = std::lower_bound(i, m_iFrmsKF.end(), C.m_iFrm);
    if (i == m_iFrmsKF.end()) {
      break;
    } else if (*i != C.m_iFrm) {
      continue;
    }
    const int iKF = static_cast<int>(i - m_iFrmsKF.begin());
    m_CsKFBkp[iKF] = m_CsKF[iKF];
    m_CsKF[iKF] = C.m_Cam_pose;
    m_marksTmp1[iKF] = GM_FLAG_FRAME_UPDATE_CAMERA;
  }
  m_GBA->PushUpdateCameras(m_LFs[m_ic2LF.back()].m_T.m_iFrm, Cs);
}


//关键帧的位姿如果变化了,那么以这个关键帧参考关键帧的滑窗帧的pose也要相应的调整,Tw(更新后)w(更新前)来修正
void LocalBundleAdjustor::UpdateCameras(const std::vector<ubyte> &ucs,
                                        const AlignedVector<Rigid3D> &CsKF1,/*LBA更新前的关键帧pose*/
                                        const AlignedVector<Rigid3D> &CsKF2/*LBA更新后的关键帧pose*/) {
  Rigid3D TI;
  int iKFNearest = -1;
  const ubyte ucmFlag = LBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION |
                        LBA_FLAG_CAMERA_MOTION_UPDATE_POSITION |
                        LBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY;
  const int nLFs = static_cast<int>(m_LFs.size());
  for (int ic = 0; ic < nLFs; ++ic) {//遍历所有滑窗中的帧
    const int iLF = m_ic2LF[ic];
    const LocalFrame &LF = m_LFs[iLF];
    const int _iKFNearest = LF.m_iKFNearest;
    if (/*_iKFNearest == -1 || */!(ucs[_iKFNearest] & GM_FLAG_FRAME_UPDATE_CAMERA)) {//如果它的最近参考帧的位姿变化了
      continue;
    }
    if (_iKFNearest != iKFNearest) {
      iKFNearest = _iKFNearest;
      TI = CsKF2[iKFNearest].GetInverse() * CsKF1[iKFNearest];//Ti = Tw(更新后)w(更新前)
    }
    Camera &C = m_CsLF[iLF];//滑窗帧的pose Tc0w
    /*if (LF.m_Cam_pose.m_iFrm == m_iFrmsKF[iKFNearest]) {
      Cam_state.m_Cam_pose = CsKF2[iKFNearest];
    } else*/ {
      C.m_Cam_pose = C.m_Cam_pose / TI; //Tc0w*Tw(更新后)w(更新前).inv = Tc0w(更新后)
    }
    C.m_Cam_pose.GetPosition(C.m_p);
    C.m_v = TI.GetAppliedRotation(C.m_v);
    m_ucsLF[iLF] |= LBA_FLAG_FRAME_UPDATE_CAMERA;
    m_ucmsLF[iLF] |= ucmFlag;
    m_UcsLF[iLF] = LM_FLAG_FRAME_UPDATE_CAMERA_LF;
  }
#ifdef CFG_DEBUG
  UT_ASSERT(GM_FLAG_FRAME_DEFAULT == LM_FLAG_FRAME_DEFAULT);
#endif
  const bool ud = LBA_RESET_DEPTH_INFORMATION;
  const ubyte ucFlag = LBA_FLAG_FRAME_UPDATE_CAMERA |
                       (ud ? LBA_FLAG_FRAME_UPDATE_DEPTH : LBA_FLAG_FRAME_DEFAULT);
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {//遍历所有的关键帧,如果这个关键帧位姿更新了,则将m_ucsKF,m_UcsKF设置成更新
    if (!(ucs[iKF] & GM_FLAG_FRAME_UPDATE_CAMERA)) {
      continue;
    }
    m_ucsKF[iKF] |= ucFlag;
    m_UcsKF[iKF] |= LM_FLAG_FRAME_UPDATE_CAMERA_KF;
    if (!ud) {
      continue;
    }
    const KeyFrame &KF = m_KFs[iKF];//没用到
    const int NZ = static_cast<int>(KF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ) {
      const FRM::Measurement &Z = KF.m_Zs[iZ];
      m_ucsKF[Z.m_iKF] |= LBA_FLAG_FRAME_UPDATE_DEPTH;
      ubyte *uds = m_uds.data() + m_iKF2d[Z.m_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
        uds[KF.m_zs[iz].m_ix] |= LBA_FLAG_TRACK_UPDATE_DEPTH;
      }
    }
  }
}
// step1:m_idxsTmp1扩容成所有关键帧size，存储一下F这帧都共视到了哪些老的关键帧 [关键帧id] = m_Zs观测中这个关键帧的索引 -1代表没有共视
// step2:遍历F观测到的所有关键帧,再遍历这些关键帧自己的共视共视帧,寻找次共视,如果找到了(即这个共视帧和F观测到了同一个地图点),那么m_idxsTmp1对应的关键帧设为-2
// step3:直接共视的需要对m_Zs构建索引,次共视和直接共视的都会作为F的共视关键帧存储在m_iKFsMatch中
void LocalBundleAdjustor::SearchMatchingKeyFrames(FRM::Frame &F) {
  std::vector<int> &iKF2Z = m_idxsTmp1;
  const int nKFs1 = static_cast<int>(m_KFs.size());//关键帧个数
  const int nKFs2 = !F.m_Zs.empty() && F.m_Zs.back().m_iKF < nKFs1 ? nKFs1 : nKFs1 + 1;//如果有对老关键帧的观测时,nKFs2 = nKFs1
  iKF2Z.assign(nKFs2, -1);//先全赋值成-1,如果有观测,就把对应的索引给m_idxsTmp1
  const int NZ = static_cast<int>(F.m_Zs.size());//观测到的所有的关键帧的个数
  for (int iZ = 0; iZ < NZ; ++iZ) {//构建KF_id和m_Zs之前的对应关系
    iKF2Z[F.m_Zs[iZ].m_iKF] = iZ;
  }
  for (int iZ = 0; iZ < NZ; ++iZ) {//遍历所有的对关键帧的观测,通过关键帧的共视找到别的共视帧,比如都观测到了6号点,那么这两个就算共视帧
    const FRM::Measurement &Z = F.m_Zs[iZ];
    const GlobalMap::KeyFrame &KF = *(Z.m_iKF == nKFs1//为什么会出现等于的情况?等于就是它观测到的关键帧不在关键帧存储里
       ? (GlobalMap::KeyFrame *) &m_IKFs2.front()//如果等于就用最早的关键帧
       : (GlobalMap::KeyFrame *) &m_KFs[Z.m_iKF]);//否则就找到对应的关键帧
    m_marksTmp1.assign(KF.m_xs.size(), 0);//观测到的这个关键帧的新地图点
    for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {//这帧对这个关键帧所存的地图点进行标记,如果观测到了,就设成1
      m_marksTmp1[F.m_zs[iz].m_ix] = 1;
    }
    const int nKFsMatch = static_cast<int>(KF.m_iKFsMatch.size());//这个关键帧的共视帧很有可能也是当前帧的共视
    for (int i = 0; i < nKFsMatch; ++i) {//遍历这个关键帧的共视
      const int _iKF = KF.m_iKFsMatch[i];//当前F的观测到的关键帧的共视关键帧
      if (iKF2Z[_iKF] != -1) {
        continue;
      }
      const KeyFrame &_KF = m_KFs[_iKF];//找到和当前帧共视的这个关键帧
      const int _iZ = _KF.SearchFrameMeasurement(Z.m_iKF);//在共视关键帧中找当前对这个关键帧的观测
      if (_iZ == -1) {
        continue;
      }
      const FRM::Measurement &_Z = _KF.m_Zs[_iZ];
      const int _iz1 = _Z.m_iz1, _iz2 = _Z.m_iz2;
      int _iz;
      for (_iz = _iz1; _iz < _iz2 && !m_marksTmp1[_KF.m_zs[_iz].m_ix]; ++_iz);//当_iz < _iz2意思就是当前这个输入帧和这个
      // 关键帧的共视帧都对同一个关键帧上的点有观测
      if (_iz < _iz2) {
        iKF2Z[_iKF] = -2;//-2说明是通过共视找到了另一个关键帧的共视
      }
    }
  }
  F.m_iKFsMatch.resize(0);
  for (int iKF = 0; iKF < nKFs2; ++iKF) {
    const int iZ = iKF2Z[iKF];
    if (iZ == -1) {//说明对这帧没有共视
      continue;
    } else if (iZ >= 0) {
      F.m_Zs[iZ].m_ik = static_cast<int>(F.m_iKFsMatch.size());
    }//-2的情况就是直接push了
    F.m_iKFsMatch.push_back(iKF);
  }
}
//处理一下共视之间的关系
void LocalBundleAdjustor::PushFeatureMeasurementMatchesFirst(const FRM::Frame &F/*滑窗中的当前帧*/,
                                                             std::vector<int> &iKF2X/*存储的是当前帧和关键帧的共视关系*/, std::vector<int> &iX2z) {
  int SNx = 0;
  const int NZ = int(F.m_Zs.size());//共视关键帧的个数
  iKF2X.assign(m_KFs.size(), -1);
  for (int iZ = 0; iZ < NZ; ++iZ) {
    const int iKF = F.m_Zs[iZ].m_iKF;//当前的共视关键帧id
    iKF2X[iKF] = SNx;
    SNx += int(m_KFs[iKF].m_xs.size());//第i个关键帧之前的共视帧有多少个地图点
  }
  iX2z.assign(SNx, -1);
  for (int iZ = 0; iZ < NZ; ++iZ) {
    const FRM::Measurement &Z = F.m_Zs[iZ];
    int *ix2z = iX2z.data() + iKF2X[Z.m_iKF];//跳转到这个关键帧的首个地图点
    for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
      ix2z[F.m_zs[iz].m_ix] = iz;//如果这个地图点被观测到,就记录这个地图点在m_zs存储的起始位置
    }
  }
}

void LocalBundleAdjustor::PushFeatureMeasurementMatchesNext(const FRM::Frame &F1,/*较新帧*/
                                                            const FRM::Frame &F2,/*当前帧*/
                                                            const std::vector<int> &iKF2X,/*[i]表示第i个关键帧之前的共视帧有多少个地图点*/
                                                            const std::vector<int> &iX2z2,/*共视帧的地图点被观测到,就记录这个地图点在m_zs存储的起始位置*/
                                                            MeasurementMatchLF &Zm) {
  ubyte firstKF = 1;
  const int NZ1 = int(F1.m_Zs.size());//较新帧的共视关键帧数量
  for (int iZ1 = 0; iZ1 < NZ1; ++iZ1)
  {//遍历较新帧对于关键帧的观测
    const FRM::Measurement &Z1 = F1.m_Zs[iZ1];//较新帧对某一个关键帧的观测
    const int iX = iKF2X[Z1.m_iKF];//这里存的value是为了方便所以到每一个关键帧里的点
    if (iX == -1)
    {
      continue;
    }
    m_izmsTmp.resize(0);
    const int *ix2z2 = iX2z2.data() + iX;//当前这个观测点首个地图点
    const int iz11 = Z1.m_iz1, iz12 = Z1.m_iz2;//现在这个是较近帧对这帧关键帧的观测在F1.m_zs里的索引,
    for (int iz1 = iz11; iz1 < iz12; ++iz1)
    {
      const int iz2 = ix2z2[F1.m_zs[iz1].m_ix/*这个地图点的局部id*/];//这个是当前帧对这同一个点的观测
      if (iz2 != -1)
      {//如果不等于-1,那就说明这个点是被当前帧和较新帧同时观测的,记录下匹配,这里的匹配分别是在他们各自的m_zs的索引
        m_izmsTmp.push_back(FTR::Measurement::Match(iz1/*较新帧*/, iz2/*当前帧*/));
      }
    }//m_izmsTmp存储的是这个关键帧中所有可以被这两帧同时观测到的地图点,存的是各自的m_zs的索引
    if (!m_izmsTmp.empty())
    {//开始向_LF.m_Zm里添加和后几帧的匹配
      Zm.PushFeatureMeasurementMatches(m_izmsTmp/*关键帧中所有可以被这两帧同时观测到的地图点,存的是各自的m_zs的索引*/, Z1.m_iKF/*都共视的这个关键帧的id*/, &firstKF);
    }
  }
  if (firstKF) {
    m_izmsTmp.resize(0);
    Zm.FRM::MeasurementMatch::PushFeatureMeasurementMatches(m_izmsTmp);
  }
}
//在LF中找到对于这个kf的观测,记录到ix2z里,[地图点局部id] = 在LF中的观测索引
void LocalBundleAdjustor::MarkFeatureMeasurements(const LocalFrame &LF/*滑窗内较新帧*/, const int iKF,/*当前地图点所属关键帧id*/
                                                  std::vector<int> &ix2z) {
  ix2z.assign(m_KFs[iKF].m_xs.size(), -1);//和这个关键帧所含有的地图点一致
  const int iZ = LF.SearchFrameMeasurement(iKF);//在这个较新帧的m_Zs中寻找这个关键帧的观测
  if (iZ == -1) {
    return;
  }
  const FRM::Measurement &Z = LF.m_Zs[iZ];//对这个关键帧的观测信息
  for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {//将对这个关键帧中地图点的观测信息记录下来[地图点局部id] = 地图点的观测在帧中m_zs的位置
    ix2z[LF.m_zs[iz].m_ix] = iz;
  }
}

void LocalBundleAdjustor::MarkFeatureMeasurementsUpdateDepth(const FRM::Frame &F,
                                                             std::vector<ubyte> &ucsKF,
                                                             std::vector<ubyte> &uds) {
  const int NZ = static_cast<int>(F.m_Zs.size());
  for (int iZ = 0; iZ < NZ; ++iZ) {
    const FRM::Measurement &Z = F.m_Zs[iZ];
    const int iKF = Z.m_iKF;
    ucsKF[iKF] |= LBA_FLAG_FRAME_UPDATE_DEPTH;
    ubyte *uds = m_udsGT.data() + m_iKF2d[iKF];
    for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
      uds[F.m_zs[iz].m_ix] |= LBA_FLAG_TRACK_UPDATE_DEPTH;
    }
  }
}

#ifdef CFG_DEBUG
void LocalBundleAdjustor::DebugSetFeatureMeasurements(const Rigid3D &Cam_state,
                                                      const AlignedVector<Rigid3D> &CsKF,
                                                      const Depth::InverseGaussian &d,
                                                      GlobalMap::Point *X) {
  Rigid3D Tr[2];
  const int Nz = static_cast<int>(X->m_zs.size());
  for (int i = 0; i < Nz; ++i) {
    FTR::Measurement &z = X->m_zs[i];
    const Rigid3D &Cz = z.m_iKF == CsKF.Size() ? Cam_state : CsKF[z.m_iKF];
    *Tr = Cz / CsKF[X->m_iKF];
#ifdef CFG_STEREO
    Tr[1] = Tr[0];
    Tr[1].SetTranslation(m_K.m_br + Tr[0].GetTranslation());
#endif
    FTR::DebugSetMeasurement(Tr, X->m_x, d, z);
  }
}
void LocalBundleAdjustor::DebugSetFeatureMeasurements(const Rigid3D &Cam_state,
                                                      const AlignedVector<Rigid3D> &CsKF,
                                                      const std::vector<Depth::InverseGaussian> &ds,
                                                      const std::vector<int> &iKF2d,
                                                      FRM::Frame *F) {
  Rigid3D Tr[2];
  const int NZ = static_cast<int>(F->m_Zs.size()), nKFs = static_cast<int>(m_KFs.size());
  for (int iZ = 0; iZ < NZ; ++iZ) {
    const FRM::Measurement &Z = F->m_Zs[iZ];
    const int iKF = Z.m_iKF;
    if (iKF >= nKFs) {
      continue;
    }
    *Tr = Cam_state / CsKF[iKF];
#ifdef CFG_STEREO
    Tr[1] = Tr[0];
    Tr[1].SetTranslation(m_K.m_br + Tr[0].GetTranslation());
#endif
    const Depth::InverseGaussian *_ds = ds.data() + iKF2d[iKF];
    const KeyFrame &KF = m_KFs[iKF];
    for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
      FTR::Measurement &z = F->m_zs[iz];
      FTR::DebugSetMeasurement(Tr, KF.m_xs[z.m_L_idx], _ds[z.m_L_idx], z);
    }
  }
}
#endif

int LocalBundleAdjustor::CountMeasurementsFrameLF() {
  int SN = 0;
  const int nLFs = static_cast<int>(m_LFs.size());
  for (int iLF = 0; iLF < nLFs; ++iLF) {
    SN += static_cast<int>(m_LFs[iLF].m_Zs.size());
  }
  return SN;
}

int LocalBundleAdjustor::CountMeasurementsFrameKF() {
  int SN = 0;
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    SN += static_cast<int>(m_KFs[iKF].m_Zs.size());
  }
  return SN;
}

int LocalBundleAdjustor::CountMeasurementsFeatureLF() {
  int SN = 0;
  const int nLFs = static_cast<int>(m_LFs.size());
  for (int iLF = 0; iLF < nLFs; ++iLF) {
    SN += static_cast<int>(m_LFs[iLF].m_zs.size());
  }
  return SN;
}

int LocalBundleAdjustor::CountMeasurementsFeatureKF() {
  int SN = 0;
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    SN += static_cast<int>(m_KFs[iKF].m_zs.size());
  }
  return SN;
}

int LocalBundleAdjustor::CountLocalTracks() {
  int SN = 0;
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    const KeyFrame &KF = m_KFs[iKF];
    const int Nx = static_cast<int>(KF.m_xs.size());
    for (int ix = 0; ix < Nx; ++ix) {
      if (KF.m_ix2ST[ix] != KF.m_ix2ST[ix + 1]) {
        ++SN;
      }
    }
  }
  return SN;
}

int LocalBundleAdjustor::CountSlidingTracks() {
  int SN = 0;
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    SN += static_cast<int>(m_KFs[iKF].m_STs.size());
  }
  return SN;
}

int LocalBundleAdjustor::CountSchurComplements() {
  return m_SAcusLF.Size() + CountSchurComplementsOffDiagonal();
}

int LocalBundleAdjustor::CountSchurComplementsOffDiagonal() {
  int SN = 0;
  const int nLFs = int(m_LFs.size());
  for (int iLF = 0; iLF < nLFs; ++iLF) {
    SN += int(m_LFs[iLF].m_iLFsMatch.size());
  }
  return SN;
}

float LocalBundleAdjustor::ComputeImageMotion(const float z1/*地图点的平均深度*/, const Rigid3D &C1/*Tc0w(ILF)*/, const Rigid3D &C2,/*Tc0w(KF)*/
                                              ubyte *first /* = NULL */) {
  Rigid3D C12;
  C1.ApplyRotation(C2.r_20_21_22_x(), C12.r_20_21_22_x());//
  if (fabs(C12.r20()) < fabs(C12.r21())) {
    C12.r_10_11_12_x().vset_all_lane(0.0f, C12.r22(), -C12.r21(), 0.0f);
    C12.r_10_11_12_x().normalize012();
    SIMD::Cross012(C12.r_10_11_12_x(), C12.r_20_21_22_x(), C12.r_00_01_02_x());
  } else {
    C12.r_00_01_02_x().vset_all_lane(C12.r22(), 0.0f, -C12.r20(), 0.0f);
    C12.r_00_01_02_x().normalize012();
    SIMD::Cross012(C12.r_20_21_22_x(), C12.r_00_01_02_x(), C12.r_10_11_12_x());
  }
  C12.SetPosition(C1.GetApplied(C2.GetPosition()));//SetPosition(tc0ILF_c0KF),但是在set过程中又会把变换反转,即tc0KF_c0ILF,
  // 其中tc0ILF_c0KF = (Tc0w(ILF)*(Tc0w(KF))^1).block<3,1>(0,3)
    //C12可以理解为 (Tc0w(ILF)*Tc0w(KF))^1 = Tc0KF_c0ILF
  m_work.Resize(4 * (sizeof(Point3D) + 2 * sizeof(Point2D)) / sizeof(float));
  AlignedVector<Point3D> X1s((Point3D *) m_work.Data(), 4, false);
  AlignedVector<Point2D> x1s((Point2D *) (X1s.Data() + 4), 4, false);
  AlignedVector<Point2D> x2s(x1s.Data() + 4, 4, false);
  if (!first || *first) {
    if (first) {
      *first = 0;
    }
    const Intrinsic &K = m_K.m_K;
    x1s[0].Set(-K.fxIcx(), -K.fyIcy());//这里是预设了4个归一化坐标
    x1s[1].Set(-K.fxIcx(), K.fyIcy());
    x1s[2].Set(K.fxIcx(), K.fyIcy());
    x1s[3].Set(K.fxIcx(), -K.fyIcy());
    for (int i = 0; i < 4; ++i) {
      X1s[i].Set(x1s[i], z1);
    }
  }
  for (int i = 0; i < 4; ++i) {//Pc0_KF = Tc0KF_c0ILF* Pc0_ILF,这里的Project不是投影到像素平面,而是归一化平面上
    C12.GetApplied(X1s[i]).Project(x2s[i]);
  }
  LA::AlignedVectorXf e2s((float *) x2s.Data(), 4 * 2, false);
  e2s -= (float *) x1s.Data();
  return sqrtf(e2s.SquaredLength() * m_K.m_K.fxy() * 0.25f);//归一化坐标的运动反映了像素的运动,再转到像素的运动残差
}
