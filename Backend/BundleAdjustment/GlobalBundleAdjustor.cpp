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
#include "GlobalBundleAdjustor.h"
#include "IBA_internal.h"

#ifdef CFG_DEBUG
#ifdef CFG_DEBUG_EIGEN
#define GBA_DEBUG_EIGEN
#endif
#if WIN32
//#define GBA_DEBUG_CHECK
//#define GBA_DEBUG_PRINT
#ifdef CFG_GROUND_TRUTH
//#define GBA_DEBUG_GROUND_TRUTH_MEASUREMENT
#endif
#define GBA_DEBUG_VIEW
#ifdef GBA_DEBUG_VIEW
#include "ViewerIBA.h"
#endif
#endif
#endif

#ifdef GBA_DEBUG_CHECK
//static const int g_iKF = 87;
static const int g_iKF = INT_MAX;
static const int g_ix = 44;
static const Rigid3D *g_C = NULL;
static const Camera *g_CLM = NULL;
static const GlobalBundleAdjustor::KeyFrame *g_KF = NULL;
static const Camera::Factor::Unitary::CC *g_SAcu = NULL;
static const Camera::Factor *g_SAcmLM = NULL;
static const Depth::InverseGaussian *g_d = NULL;
static const FTR::Factor::Full::Source::A *g_Ax = NULL;
static const FTR::Factor::Full::Source::M1 *g_Mx1 = NULL;
static const FTR::Factor::Full::Source::M2 *g_Mx2 = NULL;
#endif

void GlobalBundleAdjustor::Initialize(IBA::Solver *solver, const int serial, const int verbose,
                                      const int debug, const int history) {
  MT::Thread::Initialize(serial, 5, "GBA");
  m_solver = solver;
  m_LM = &solver->m_internal->m_LM;
  m_GM = &solver->m_internal->m_GM;
  m_LBA = &solver->m_internal->m_LBA;
  m_K = solver->m_internal->m_K;
  m_verbose = verbose;
  m_debug = debug;
  m_history = history;
  m_dir = solver->m_internal->m_dir;
#ifdef CFG_GROUND_TRUTH
  m_CsGT = solver->m_internal->m_CsGT.Data();
  m_dsGT = solver->m_internal->m_DsGT.empty() ? NULL : &solver->m_internal->m_dsGT;
#ifdef CFG_HISTORY
  if (m_history >= 3 && (!m_CsGT || !m_dsGT)) {
    m_history = 2;
  }
#endif
#endif
}

void GlobalBundleAdjustor::Reset() {
  MT::Thread::Reset();
  MT_WRITE_LOCK_BEGIN(m_MT, MT_TASK_NONE, MT_TASK_GBA_Reset);
  m_ITs1.resize(0);
  m_ITs2.resize(0);
  m_IKFs1.resize(0);
  m_IKFs2.resize(0);
  m_IDKFs1.resize(0);
  m_IDKFs2.resize(0);
  m_IDMPs1.resize(0);
  m_IDMPs2.resize(0);
  m_IUCs1.resize(0);
  m_IUCs2.resize(0);
  m_IZps1.resize(0);
  m_IZps2.resize(0);
  m_IZpLM1.Invalidate();
  m_IZpLM2.Invalidate();
  MT_WRITE_LOCK_END(m_MT, MT_TASK_NONE, MT_TASK_GBA_Reset);

  for (int i = 0; i < TM_TYPES; ++i) {
    m_ts[i].Reset(TIME_AVERAGING_COUNT);
  }
#ifdef CFG_HISTORY
  m_hists.resize(0);
#endif
  m_CsDel.resize(0);

  m_delta2 = BA_DL_RADIUS_INITIAL;

  m_Zps.resize(0);
  m_Aps.resize(0);
  //m_ZpLM.Invalidate();
  m_ZpLM.m_iKF = 0;//初始化m_Amm先验部分权重
  m_ZpLM.Initialize(BA_WEIGHT_PRIOR_CAMERA_INITIAL,//初始权重
                    BA_VARIANCE_PRIOR_VELOCITY_FIRST,
                    BA_VARIANCE_PRIOR_BIAS_ACCELERATION_FIRST,
                    BA_VARIANCE_PRIOR_BIAS_GYROSCOPE_FIRST);
#ifdef GBA_DEBUG_GROUND_TRUTH_MEASUREMENT
  if (m_CsGT) {
    m_ZpLM.DebugSetMeasurement(m_CsGT[0]);
  }
#endif
  m_ApLM.MakeZero();

  m_KFs.resize(0);
  m_iFrms.resize(0);
  m_Cs.Resize(0);
  m_CsLM.Resize(0);
#ifdef CFG_GROUND_TRUTH
  m_CsKFGT.Resize(0);
  m_CsLMGT.Resize(0);
#endif
  m_ucs.resize(0);
  m_ucmsLM.resize(0);
#ifdef CFG_HANDLE_SCALE_JUMP
  m_dsKF.resize(0);
#endif
#ifdef CFG_INCREMENTAL_PCG
  m_xcs.Resize(0);
  m_xmsLM.Resize(0);
#endif
  m_DsLM.Resize(0);
#ifdef CFG_GROUND_TRUTH
  m_DsLMGT.Resize(0);
#endif
  m_AdsLM.Resize(0);

  m_Afps.Resize(0);
  m_AfmsLM.Resize(0);

  m_iKF2d.assign(1, 0);
  m_ds.resize(0);
  m_uds.resize(0);

  m_SAcus.Resize(0);
  m_SMcus.Resize(0);
  m_SAcmsLM.Resize(0);

  m_iKF2cb.assign(1, 0);

  //m_F = 0.0f;
}

void GlobalBundleAdjustor::PushKeyFrame(const GlobalMap::InputKeyFrame &IKF,/*当前关键帧*/
                                        const AlignedVector<IMU::Measurement> &us,/*在当前帧时间戳之前的imu测量*/
                                        const std::vector<Depth::InverseGaussian> &dzs//当前关键帧观测到的地图点逆深度的备份
#ifdef CFG_HANDLE_SCALE_JUMP
                                      , const float d
#endif
                                      ) {
  MT_WRITE_LOCK_BEGIN(m_MT, IKF.m_T.m_iFrm, MT_TASK_GBA_PushKeyFrame);
#ifdef CFG_DEBUG
  UT_ASSERT(dzs.size() >= IKF.m_zs.size());
#endif
  m_ITs1.push_back(IT_KEY_FRAME);//记录一下m_IKFs1里push的是什么东西
  m_IKFs1.push_back(InputKeyFrame(IKF/*当前关键帧*/, us/*在当前帧时间戳之前的imu测量*/, dzs//当前关键帧观测到的地图点逆深度的备份
#ifdef CFG_HANDLE_SCALE_JUMP
                                , d
#endif
                                ));
  MT_WRITE_LOCK_END(m_MT, IKF.m_T.m_iFrm, MT_TASK_GBA_PushKeyFrame);
}

void GlobalBundleAdjustor::PushDeleteKeyFrame(const int iFrm, const int iKF) {
  MT_WRITE_LOCK_BEGIN(m_MT, iFrm, MT_TASK_GBA_PushDeleteKeyFrame);
  m_ITs1.push_back(IT_DELETE_KEY_FRAME);
  m_IDKFs1.push_back(iKF);
  MT_WRITE_LOCK_END(m_MT, iFrm, MT_TASK_GBA_PushDeleteKeyFrame);
}

void GlobalBundleAdjustor::PushDeleteMapPoints(const int iFrm, const std::vector<int> &ids) {
  MT_WRITE_LOCK_BEGIN(m_MT, iFrm, MT_TASK_GBA_PushDeleteMapPoints);
  m_ITs1.push_back(IT_DELETE_MAP_POINTS);
  m_IDMPs1.push_back(ids);
  MT_WRITE_LOCK_END(m_MT, iFrm, MT_TASK_GBA_PushDeleteMapPoints);
}

void GlobalBundleAdjustor::PushUpdateCameras(const int iFrm,
                                             const std::vector<GlobalMap::InputCamera> &Cs) {
  MT_WRITE_LOCK_BEGIN(m_MT, iFrm, MT_TASK_GBA_PushUpdateCameras);
  m_ITs1.push_back(IT_UPDATE_CAMERAS);
  m_IUCs1.push_back(Cs);
  MT_WRITE_LOCK_END(m_MT, iFrm, MT_TASK_GBA_PushUpdateCameras);
}

void GlobalBundleAdjustor::PushCameraPriorPose(const int iFrm, const CameraPrior::Pose &IZp) {
  MT_WRITE_LOCK_BEGIN(m_MT, iFrm, MT_TASK_GBA_PushCameraPriorPose);
  m_ITs1.push_back(IT_CAMERA_PRIOR_POSE);
  m_IZps1.push_back(IZp);
  MT_WRITE_LOCK_END(m_MT, iFrm, MT_TASK_GBA_PushCameraPriorPose);
}

void GlobalBundleAdjustor::PushCameraPriorMotion(const int iFrm, const int iKF, const CameraPrior::Motion &IZp) {
  MT_WRITE_LOCK_BEGIN(m_MT, iFrm, MT_TASK_GBA_PushCameraPriorMotion);
  std::list<InputType>::iterator IT1, IT2 = m_ITs1.begin();
  while (IT2 != m_ITs1.end()) {
    IT1 = IT2++;
    if (*IT1 == IT_CAMERA_PRIOR_MOTION) {
      m_ITs1.erase(IT1);
    }
  }
  m_ITs1.push_back(IT_CAMERA_PRIOR_MOTION);
  m_IZpLM1.Set(iKF, IZp);//
  MT_WRITE_LOCK_END(m_MT, iFrm, MT_TASK_GBA_PushCameraPriorMotion);
}

void GlobalBundleAdjustor::Run() {


#if 0
//#if 1
  AssertConsistency();
#endif
  m_delta2 = BA_DL_RADIUS_INITIAL;
  m_ts[TM_TOTAL].Start();
  m_ts[TM_SYNCHRONIZE].Start();
  SynchronizeData();//做一些数据GBA数据的关联
  m_ts[TM_SYNCHRONIZE].Stop();
  m_ts[TM_TOTAL].Stop();
#if 0
//#if 1
  AssertConsistency();
#endif
//#ifdef CFG_DEBUG
#if 0
  const std::string fileName = "D:/tmp/cons.txt";
#if 0
  FILE *fp = fopen(fileName.c_str(), "wb");
  m_Cs.SaveB(fp);
  m_CsLM.SaveB(fp);
  UT::VectorSaveB(m_ds, fp);
  fclose(fp);
  UT::PrintSaved(fileName);
#else
  FILE *fp = fopen(fileName.c_str(), "rb");
  AlignedVector<Rigid3D> Cs;
  Cs.LoadB(fp);
  const int nKFs = static_cast<int>(m_KFs.size());
  UT_ASSERT(Cs.Size() == nKFs);
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    m_Cs[iKF].AssertEqual(Cs[iKF], 1, UT::String("Cs[%d]", iKF), -1.0f, -1.0f);
  }
  AlignedVector<Camera> CsLM;
  CsLM.LoadB(fp);
  const int Nm = m_CsLM.Size();
  UT_ASSERT(CsLM.Size() == Nm);
  for (int im = 0; im < Nm; ++im) {
    m_CsLM[im].AssertEqual(CsLM[im], 1, UT::String("CsLM[%d]", im), -1.0f, -1.0f);
  }
  std::vector<Depth::InverseGaussian> ds;
  UT::VectorLoadB(ds, fp);
  const int Nd = static_cast<int>(m_ds.size());
  UT_ASSERT(static_cast<int>(ds.size()) == Nd);
  //for (int id = 0; id < Nd; ++id) {
  //  m_ds[id].AssertEqual(ds[id], 1, UT::String("ds[%d]", id), -1.0f);
  //}
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    const int id1 = m_iKF2d[iKF], id2 = m_iKF2d[iKF + 1];
    for (int id = id1; id < id2; ++id) {
      if (!m_ds[id].AssertEqual(ds[id], 1, UT::String("ds[%d]", id), -1.0f)) {
        UT::Print("iKF = %d [%d], ix = %d\n", iKF, m_iFrms[iKF], id - id1);
      }
    }
  }
  fclose(fp);
  UT::PrintLoaded(fileName);
#endif
#endif
#ifdef CFG_DEBUG
  if (m_debug < 0) {
    return;
  }
#endif
//#ifdef CFG_DEBUG
#if 0
  {
    std::vector<float> ts;
    const int Nm = m_CsLM.Size();
    for (int im1 = 0, im2 = 1; im2 < Nm; im1 = im2++)
      ts.push_back(m_DsLM[im2].m_Tpv);
    UT::SaveValues("D:/tmp/test.txt", ts);
    exit(0);
  }
#endif
#ifdef GBA_DEBUG_EIGEN
  DebugGenerateTracks();
#endif
#ifdef GBA_DEBUG_CHECK
  {
    const int nKFs = int(m_KFs.size());
    if (nKFs != 0) {
      const int iKF = g_iKF >= 0 && g_iKF < nKFs ? g_iKF : nKFs - 1;
      g_C = &m_Cs[iKF];
      g_KF = &m_KFs[iKF];
      g_SAcu = &m_SAcus[iKF];
      const int im = iKF - (nKFs - m_CsLM.Size());
      if (im >= 0) {
        g_CLM = &m_CsLM[im];
        g_SAcmLM = &m_SAcmsLM[im];
      }
      if (g_ix >= 0 && g_ix < static_cast<int>(g_KF->m_xs.size())) {
        g_d = m_ds.data() + m_iKF2d[iKF] + g_ix;
        g_Ax = &g_KF->m_Axs[g_ix];
        g_Mx1 = &g_KF->m_Mxs1[g_ix];
        g_Mx2 = &g_KF->m_Mxs2[g_ix];
      }
    }
  }
#endif
  const int iFrm = m_iFrms.back();
#ifdef CFG_VERBOSE
  if (m_verbose == 1) {
    UT::PrintSeparator();
    UT::Print("[%d] GBA\n", iFrm);
  } else if (m_verbose >= 2) {
    UT::PrintSeparator('*');
    UT::Print("[%d] [GlobalBundleAdjustor::Run]\n", iFrm);
    UT::Print("  Frame = %d\tMeasurement = %d + %d\tSchur = %d\tTrack = %d\n", m_Cs.Size(),
              CountMeasurementsFeature(), CountMeasurementsPriorCameraPose(),
              CountSchurComplements(), m_ds.size());
#ifdef CFG_GROUND_TRUTH
    if (!m_CsKFGT.Empty() && m_dsGT) {
      UT::PrintSeparator();
      PrintErrorStatistic("*GT: ", m_CsKFGT, m_CsLMGT, *m_dsGT, m_DsLMGT, true);
    }
#endif
  }
#endif
#if 0
//#if 1
  const int Nc = m_Cs.Size();
  //if (Nc == 2) {
  if (Nc >= 2) {
    UT::Check("Noise\n");
    //const float erMax1 = 0.0f;
    const float erMax1 = 0.5f;
    //const float erMax2 = 0.0f;
    const float erMax2 = 0.5f;
    //const float erMax2 = 10.0f;
    //const float epMax1 = 0.0f;
    const float epMax1 = 0.05f;
    //const float epMax2 = 0.0f;
    const float epMax2 = 0.05f;
    //const float epMax2 = 1.0f;
    const float evMax = 0.0f;
    //const float evMax = 1.0f;
    const float ebaMax = 0.0f;
    //const float ebaMax = 0.001f;
    const float ebwMax = 0.0f;
    const float edMax = 0.0f;
    //const float edMax = 0.1f;
    //const float ebwMax = 1.0f * UT_FACTOR_DEG_TO_RAD;
    Rotation3D dR;
    LA::AlignedVector3f dp;
    dR.MakeIdentity();
    dp.MakeZero();
    for (int ic = 0, im = m_CsLM.Size() - Nc; ic < Nc; ++ic, ++im) {
      Rigid3D &T = m_Cs[ic];
      Point3D p = T.GetPosition();
//#if 0
#if 1
      if (ic > 0)
#endif
      {
        dR = Rotation3D::GetRandom(erMax1 * UT_FACTOR_DEG_TO_RAD) * dR;
        T = dR * Rotation3D::GetRandom(erMax1 * UT_FACTOR_DEG_TO_RAD) * T;
      }
      //dp.Random(epMax);
      dp += LA::AlignedVector3f::GetRandom(epMax1);
      p += dp + LA::AlignedVector3f::GetRandom(epMax2);
      T.SetPosition(p);
      m_ucs[ic] |= GBA_FLAG_FRAME_UPDATE_CAMERA | GBA_FLAG_FRAME_UPDATE_DEPTH;
      if (im == -1) {
        continue;
      }
      Camera &Cam_state = m_CsLM[im];
      Cam_state.m_Cam_pose = T;
      Cam_state.m_p = p;
      Cam_state.m_v += LA::AlignedVector3f::GetRandom(evMax);
      Cam_state.m_ba += LA::AlignedVector3f::GetRandom(ebaMax);
      Cam_state.m_bw += LA::AlignedVector3f::GetRandom(ebwMax);
#ifdef CFG_DEBUG
      Cam_state.m_v.MakeZero();
      Cam_state.m_ba.MakeZero();
      Cam_state.m_bw.MakeZero();
#endif
      m_ucmsLM[im] |= GBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION | GBA_FLAG_CAMERA_MOTION_UPDATE_POSITION;
    }
    const int Nd = int(m_ds.size());
    for (int id = 0; id < Nd; ++id) {
      m_ds[id].u() += UT::Random<float>(edMax);
      m_uds[id] |= GBA_FLAG_TRACK_UPDATE_DEPTH;
    }
  }
#endif
#ifdef GBA_DEBUG_VIEW
  ViewerIBA *viewer = NULL;
  if (m_verbose >= 2) {
    viewer = new ViewerIBA();
    viewer->Create(m_solver/*, "D:/tmp/backend.png", 2*/);
    //viewer->Start("D:/tmp/view.txt");
    viewer->ActivateFrame(viewer->GetKeyFrames() + viewer->GetLocalFrames() - 1);
    viewer->ActivateFrame(viewer->GetKeyFrames() - 1);
    viewer->m_keyPause = true;
    viewer->m_keyDrawCamTypeKF = ViewerIBA::DRAW_CAM_KF_GBA;
    viewer->m_keyDrawDepType = ViewerIBA::DRAW_DEP_GBA;
  }
#endif
#if 0
//#if 1
  {
    UT::Check("Noise\n");
    UpdateFactors();
    UT::PrintSeparator();
    PrintErrorStatistic("    ", m_Cs, m_CsLM, m_ds, m_DsLM, false);
    UT::Print("\n");
    const float edMax = 1.0f;
    const int Nd = int(m_ds.size());
    for (int id = 0; id < Nd; ++id) {
      if (m_uds[id] & GBA_FLAG_TRACK_UPDATE_DEPTH) {
        m_ds[id].u() += UT::Random<float>(edMax);
      }
    }
    UT::PrintSeparator();
    PrintErrorStatistic("--> ", m_Cs, m_CsLM, m_ds, m_DsLM, false);
    UT::Print("\n");
#ifdef GBA_DEBUG_VIEW
    if (viewer) {
      viewer->Run();
    }
#endif
    EmbeddedPointIteration();
    UT::PrintSeparator();
    PrintErrorStatistic("--> ", m_Cs, m_CsLM, m_ds, m_DsLM, false);
    UT::Print("\n");
#ifdef GBA_DEBUG_VIEW
    if (viewer) {
      viewer->Run();
    }
#endif
  }
#endif
  m_ts[TM_TOTAL].Start();
  for (m_iIter = 0; m_iIter < BA_MAX_ITERATIONS; ++m_iIter) {
//#ifdef CFG_DEBUG
#if 0
//#if 1
    if (m_iIter == 0) {
      UT::Check("Update\n");
      m_Ao.MakeZero();
      const int NZp = static_cast<int>(m_Aps.size());
      for (int iZp = 0; iZp < NZp; ++iZp) {
        m_Aps[iZp].MakeZero();
      }
      if (m_ZpLM.Valid()) {
        m_ApLM.MakeZero();
      }
      const int nKFs = static_cast<int>(m_KFs.size());
      for (int iKF = 0; iKF < nKFs; ++iKF) {
        m_ucs[iKF] |= GBA_FLAG_FRAME_UPDATE_CAMERA | GBA_FLAG_FRAME_UPDATE_DEPTH;
        m_Ucs[iKF] |= GM_FLAG_FRAME_UPDATE_CAMERA/* | GM_FLAG_FRAME_UPDATE_DEPTH*/;
        m_KFs[iKF].MakeZero();
      }
      for (int ic = nKFs - m_CsLM.Size(), im = 0; ic < nKFs; ++ic, ++im) {
        m_ucmsLM[im] |= GBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION | GBA_FLAG_CAMERA_MOTION_UPDATE_POSITION |
                        GBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY |
                        GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_ACCELERATION | GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_GYROSCOPE;
        if (im == 0) {
          continue;
        }
        IMU::Delta &D = m_DsLM[im];
        const KeyFrame &KF = m_KFs[ic];
        IMU::PreIntegrate(KF.m_imu_measures, m_KFs[ic - 1].m_Cam_pose.m_t, KF.m_Cam_pose.m_t, m_CsLM[im - 1], &D,
                          &m_work, true, D.m_u1.Valid() ? &D.m_u1 : NULL, D.m_u2.Valid() ? &D.m_u2 : NULL);
      }
      m_AdsLM.MakeZero();
      m_Afps.MakeZero();
      m_AfmsLM.MakeZero();
      const int Nd = static_cast<int>(m_ds.size());
      for (int id = 0; id < Nd; ++id) {
        m_uds[id] |= GBA_FLAG_TRACK_UPDATE_DEPTH | GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;
        //m_Uds[id] |= GM_FLAG_TRACK_UPDATE_DEPTH;
      }
      m_SAcus.MakeZero();
      m_SMcus.MakeZero();
      m_SAcmsLM.MakeZero();
//#ifdef CFG_GROUND_TRUTH
#if 0
      m_Cs.Set(m_CsKFGT);
      m_CsLM.Set(m_CsLMGT);
      //const int ic = nKFs - 1;
      const int ic = nKFs - 2;
      Rigid3D &T = m_Cs[ic];
      const LA::AlignedVector3f dr(0.0f, 0.0f, 0.0f);
      //const LA::AlignedVector3f dr(0.1f, 0.2f, 0.3f);
      const LA::AlignedVector3f dp(0.0f, 0.0f, 0.0f);
      //const LA::AlignedVector3f dp(0.1f, 0.2f, 0.3f);
      //const LA::AlignedVector3f dv(0.0f, 0.0f, 0.0f);
      const LA::AlignedVector3f dv(0.1f, 0.2f, 0.3f);
      const LA::AlignedVector3f dba(0.0f, 0.0f, 0.0f);
      //const LA::AlignedVector3f dba(0.1f, 0.2f, 0.3f);
      const LA::AlignedVector3f dbw(0.0f, 0.0f, 0.0f);
      //const LA::AlignedVector3f dbw(0.1f, 0.2f, 0.3f);
      const Point3D p = T.GetPosition() + dp;
      const Rotation3D dR(dr * UT_FACTOR_DEG_TO_RAD);
      T = Rotation3D(T) * dR;
      T.SetPosition(p);
      const int im = m_CsLM.Size() - nKFs + ic;
      if (im >= 0) {
        Camera &Cam_state = m_CsLM[im];
        Cam_state.m_Cam_pose = T;
        Cam_state.m_p = p;
        Cam_state.m_v += dv;
        Cam_state.m_ba += dba;
        Cam_state.m_bw += dbw * UT_FACTOR_DEG_TO_RAD;
      }
#endif
    }
#endif
#ifdef CFG_VERBOSE
    if (m_verbose) {
      if (m_verbose >= 2) {
        UT::PrintSeparator('*');
      } else if (m_iIter == 0) {
        UT::PrintSeparator();
      }
      PrintErrorStatistic(UT::String("*%2d: ", m_iIter), m_Cs, m_CsLM, m_ds, m_DsLM,
                          m_verbose >= 2);
    }
#endif
#ifdef GBA_DEBUG_VIEW
    if (viewer) {
      viewer->Run(true, false);
    }
#endif
    m_ts[TM_FACTOR].Start();
    UpdateFactors();//更新因子
    m_ts[TM_FACTOR].Stop();
#ifdef GBA_DEBUG_EIGEN
    DebugUpdateFactors();
#endif
    m_ts[TM_SCHUR_COMPLEMENT].Start();
    UpdateSchurComplement();//要将H|-b中的地图点部分先边缘化得到[S|-g],这里在求的就是W*V^-1*W.t g 和 W*V^-1*v
    m_ts[TM_SCHUR_COMPLEMENT].Stop();
#ifdef GBA_DEBUG_EIGEN
    DebugUpdateSchurComplement();
#endif
    m_ts[TM_CAMERA].Start();
    const bool scc = SolveSchurComplement();//PCG加速S*x = -g
    m_ts[TM_CAMERA].Stop();
#ifdef GBA_DEBUG_EIGEN
    DebugSolveSchurComplement();
#endif
    //if (!scc) {
    //  m_update = false;
    //  m_converge = false;
    //  break;
    //}
    m_ts[TM_DEPTH].Start();
      //du =-Wcu.t * Huu^-1 * dxc +  Huu^-1*bu (Huu就是V,bu就是v) = Huu^-1*(bu -Wcu.t * dxc )也就求出了地图点逆深度的增量
//利用相机pose的增量反求出地图点逆深度的增量,将所有所有需要反求增量的du push进m_xsGN
    SolveBackSubstitution();
    m_ts[TM_DEPTH].Stop();
#ifdef GBA_DEBUG_EIGEN
    DebugSolveBackSubstitution();
#endif

#ifdef CFG_VERBOSE
    const int N = UT::PrintStringWidth();
#endif
    m_xsGD.Resize(0);   m_x2GD = 0.0f;
    m_xsDL.Resize(0);   m_x2DL = 0.0f;
    m_rho = FLT_MAX;
    const int nItersDL = std::max(BA_DL_MAX_ITERATIONS, 1);//dogleg迭代次数
    for (m_iIterDL = 0; m_iIterDL < nItersDL; ++m_iIterDL) {
      if (m_x2GN > m_delta2 && m_xsGD.Empty() && BA_DL_MAX_ITERATIONS > 0) {//如果大于半径的话,
        m_ts[TM_UPDATE].Start();
        SolveGradientDescent();//求解pU点,方向是负梯度方向,步长g.t*g/g.t*A*g
        m_ts[TM_UPDATE].Stop();
#ifdef GBA_DEBUG_EIGEN
        DebugSolveGradientDescent();
#endif
      }
      m_ts[TM_UPDATE].Start();
      SolveDogLeg();//判断GN求出的增量和pU点与信赖半径的关系,然后求解dogleg方法的增量
      UpdateStatesPropose();//备份状态以及更新状态(pose更新在m_Cs中,motion更新在m_CsLM)和逆深度(m_ds)
      m_ts[TM_UPDATE].Stop();
#ifdef CFG_VERBOSE
      if (m_verbose) {
        if (m_verbose >= 3) {
          UT::PrintSeparator();
        }
        const std::string str = m_verbose >= 2 ? " --> " :
                                (m_iIterDL == 0 ? " > " : std::string(N + 3, ' '));
        PrintErrorStatistic(str, m_CsBkp, m_CsLMBkp, m_dsBkp, m_DsLM, m_xsDL, m_verbose >= 2);
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
#ifdef GBA_DEBUG_EIGEN
      DebugComputeReduction();
#endif
      m_ts[TM_UPDATE].Start();
      const bool accept = UpdateStatesDecide();
      m_ts[TM_UPDATE].Stop();
      if (accept) {
        break;
      }
    }
    if (m_iIterDL == BA_DL_MAX_ITERATIONS) {
      m_ts[TM_UPDATE].Start();
      UpdateStatesDecide();//通过rho来判断近似效果的好坏,近似的不好需要回滚更新
      m_ts[TM_UPDATE].Stop();
    }
#ifdef CFG_DEBUG
    if (m_debug >= 2) {
      AssertConsistency();
    }
#endif
    if (!m_update || m_converge) {
      break;
    }
    if (GBA_EMBEDDED_POINT_ITERATION) {
      m_ts[TM_UPDATE].Start();
      EmbeddedPointIteration(m_Cs, m_ucs, m_uds, &m_ds);
      m_ts[TM_UPDATE].Stop();
    }
    MT_READ_LOCK_BEGIN(m_MT, iFrm, MT_TASK_GBA_BufferDataEmpty);
    m_empty = BufferDataEmpty();
    MT_READ_LOCK_END(m_MT, iFrm, MT_TASK_GBA_BufferDataEmpty);
    if (!m_empty) {
      break;
    }
  }
#ifdef CFG_VERBOSE
  if (m_verbose) {
    if (m_verbose >= 2) {
      UT::PrintSeparator('*');
    }
    PrintErrorStatistic(UT::String("*%2d: ", m_iIter), m_Cs, m_CsLM, m_ds, m_DsLM, m_verbose >= 2);
    if (m_verbose < 2) {
      UT::Print("\n");
    }
  }
#endif
  m_ts[TM_TOTAL].Stop();
  for (int i = 0; i < TM_TYPES; ++i) {
    m_ts[i].Finish();
  }
#ifdef CFG_HISTORY
  if (m_history >= 1) {
    History hist;
    hist.MakeZero();
    for (int i = 0; i < TM_TYPES; ++i) {
      hist.m_ts[i] = m_ts[i].GetAverageSeconds() * 1000.0;
    }
    //hist.m_Nd = static_cast<int>(m_ds.size());
    if (m_history >= 2) {
      hist.m_ESa = ComputeErrorStatistic(m_Cs, m_CsLM, m_ds, m_DsLM);
      hist.m_ESb = ComputeErrorStatistic(m_CsBkp, m_CsLMBkp, m_dsBkp, m_DsLM);
      hist.m_ESp = ComputeErrorStatistic(m_CsBkp, m_CsLMBkp, m_dsBkp, m_DsLM, m_xsDL, false);
#ifdef CFG_GROUND_TRUTH
      if (m_CsGT) {
        SolveSchurComplementGT(m_CsBkp, m_CsLMBkp, &m_xsGT);
        if (m_dsGT) {
          if (m_history >= 3) {
            EmbeddedPointIteration(m_CsKFGT, m_ucsGT, m_udsGT, m_dsGT);
          }
          SolveBackSubstitutionGT(m_dsBkp, &m_xsGT);
          hist.m_ESaGT = ComputeErrorStatistic(m_CsKFGT, m_CsLMGT, *m_dsGT, m_DsLMGT);
          hist.m_ESpGT = ComputeErrorStatistic(m_CsBkp, m_CsLMBkp, m_dsBkp, m_DsLM, m_xsGT, false);
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
      }
#endif
    }
    m_hists.push_back(hist);
  }
#endif
#ifdef GBA_DEBUG_VIEW
  if (viewer) {
    viewer->m_keyPause = true;
    viewer->Run();
    //viewer->Stop("D:/tmp/view.txt");
    delete viewer;
  }
#endif
  UpdateData();//更新GlobalMap里m_Cs的pose
#ifdef CFG_DEBUG
  //if (!m_serial && m_debug)
  if (m_debug) {
    AssertConsistency();
  }
#endif
#ifdef GBA_DEBUG_PRINT
  {
    Rigid3D Cam_state;
    Camera::Factor::Unitary::CC A;
    Camera::Factor::Binary::CC M;
    double Su, Ss2;
    UT::PrintSeparator('*');
    for (int i = 0; i < 2; ++i) {
      const int nKFs = static_cast<int>(m_KFs.size());
      if (i == 0) {
        const int iKF = nKFs - 1;
        const KeyFrame &KF = m_KFs[iKF];
        UT::Print("[%d]\n", KF.m_Cam_pose.m_iFrm);
        Cam_state = m_Cs[iKF];
        A = m_SAcus[iKF] - m_SMcus[iKF];
        M.MakeZero();
        //const int Nk = static_cast<int>(KF.m_iKFsMatch.size());
        const int Nk = KF.m_Zm.m_SMczms.Size();
        const xp128f sm = xp128f::get(1.0f / Nk);
        for (int ik = 0; ik < Nk; ++ik) {
          M += KF.m_Zm.m_SMczms[ik] * sm;
        }
        const int NZ = static_cast<int>(KF.m_Zs.size());
        for (int iZ = 0; iZ < NZ; ++iZ) {
          M -= KF.m_SAcxzs[iZ] * sm;
        }

        Su = Ss2 = 0.0;
        for (int iZ = 0; iZ < NZ; ++iZ) {
          const FRM::Measurement &Z = KF.m_Zs[iZ];
          const Depth::InverseGaussian *ds = m_ds.data() + m_iKF2d[Z.m_iKF];
          for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
            const Depth::InverseGaussian &d = ds[KF.m_zs[iz].m_L_idx];
            Su = d.u() + Su;
            Ss2 = d.s2() + Ss2;
          }
        }
      } else {
        Cam_state.MakeIdentity();
        A.MakeZero();
        M.MakeZero();
        Su = Ss2 = 0.0;
        int SNZ = 0;
        for (int iKF = 0; iKF < nKFs; ++iKF) {
          SNZ += static_cast<int>(m_KFs[iKF].m_Zs.size());
        }
        const xp128f sa = xp128f::get(1.0f / nKFs);
        const xp128f sm = xp128f::get(1.0f / (CountSchurComplements() - nKFs));
        for (int iKF = 0; iKF < nKFs; ++iKF) {
          Cam_state = m_Cs[iKF] * Cam_state;
          A += (m_SAcus[iKF] - m_SMcus[iKF]) * sa;
          const KeyFrame &KF = m_KFs[iKF];
          //const int Nk = static_cast<int>(KF.m_iKFsMatch.size());
          const int Nk = KF.m_Zm.m_SMczms.Size();
          for (int ik = 0; ik < Nk; ++ik) {
            M += KF.m_Zm.m_SMczms[ik] * sm;
          }
          const int NZ = static_cast<int>(KF.m_Zs.size());
          for (int iZ = 0; iZ < NZ; ++iZ) {
            M -= KF.m_SAcxzs[iZ] * sm;
          }
        }
        const int Nd = m_iKF2d[nKFs - 1];
        for (int id = 0; id < Nd; ++id) {
          const Depth::InverseGaussian &d = m_ds[id];
          Su = d.u() + Su;
          Ss2 = d.s2() + Ss2;
        }
      }
      UT::Print("Cam_state:\n");
      Cam_state.Print(true);
      UT::Print("A:\n");
      A.Print(true);
      M.Print(true);
      UT::Print("d: %.10e %.10e\n", Su, Ss2);
    }
    UT::PrintSeparator('*');
  }
#endif
//#ifdef CFG_DEBUG
#if 0
  const std::string fileName = UT::String("D:/tmp/xsGN/x_%03d_%03d.txt", m_KFs.size(), m_Zps.size());
  LA::AlignedVectorXf xsGN;
  xsGN.LoadB(fileName);
  const bool scc = m_xsGN.AssertEqual(xsGN, 1, "", -1.0f, -1.0f);
  UT_ASSERT(scc);
#endif
//#ifdef CFG_DEBUG
#if 0
  //const int iKF = 100;
  //const int ix = 2;
  //if (iKF < static_cast<int>(m_KFs.size()))
  //  UT::Print("%.10e\n", m_ds[m_KFs[iKF].m_id].u());
  const int im = 83;
  const int ipc = 5, ipm = 8;
  if (im < m_SAcmsLM.Size())
    UT::Print("%.10e\n", m_SAcmsLM[im].m_Au.m_Acm[ipc][ipm]);
#endif

  // Trigger GBA callback function if set
  if (m_callback) {
    const FRM::Tag &T = m_KFs.back().m_T;
    m_callback(T.m_iFrm, T.m_t);
  }
}

void GlobalBundleAdjustor::SetCallback(const IBA::Solver::IbaCallback& iba_callback) {
  m_callback = iba_callback;
}

//int GlobalBundleAdjustor::GetTotalPoints(int *N) {
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


float GlobalBundleAdjustor::GetTotalTime(int *N) {
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

bool GlobalBundleAdjustor::SaveTimes(const std::string fileName) {
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
  UT::PrintSaved(fileName);
  fclose(fp);
  return true;
}

bool GlobalBundleAdjustor::SaveCameras(const std::string fileName, const bool poseOnly) {
  FILE *fp = fopen(fileName.c_str(), "w");
  if (!fp) {
    return false;
  }
  const int Nc1 = static_cast<int>(m_CsDel.size()), Nc2 = m_Cs.Size(), Nc = Nc1 + Nc2;
  AlignedVector<HistoryCamera> Cs;
  m_work.Resize(Cs.BindSize(Nc) / sizeof(float));
  Cs.Bind(m_work.Data(), Nc);
  Cs.Set(m_CsDel.data(), Nc1);
  for (int ic = 0; ic < Nc2; ++ic) {
    Cs.Push(HistoryCamera(m_KFs[ic].m_T, m_Cs[ic]));
  }
  std::sort(Cs.Data(), Cs.Data() + Nc);
  std::vector<int> &iFrm2m = m_idxsTmp1;
  if (!poseOnly) {
    iFrm2m.assign(Cs.Back().m_iFrm + 1, -1);
    for (int ic = Nc2 - m_CsLM.Size(), im = 0; ic < Nc2; ++ic, ++im) {
      iFrm2m[m_iFrms[ic]] = im;
    }
  }

  Point3D p;
  Quaternion q;
  Rotation3D R;
  LA::AlignedVector3f ba, bw;
  const Rotation3D RuT = m_K.m_Ru.GetTranspose();//Ri_c0
  for (int ic = 0; ic < Nc2; ++ic) {
    const HistoryCamera &C = Cs[ic];
    const Rigid3D &T = C.m_C;
    T.GetPosition(p);
    p += T.GetAppliedRotationInversely(m_K.m_pu);
    Rotation3D::AB(RuT, T, R);//Ri_w = Ri_c0* Rc0w
    R.GetQuaternion(q);
    fprintf(fp, "%f %f %f %f %f %f %f %f", C.m_t, p.x(), p.y(), p.z(),
                                           q.x(), q.y(), q.z(), q.w());
    if (!poseOnly && iFrm2m[C.m_iFrm] != -1) {
      const Camera &_C = m_CsLM[iFrm2m[C.m_iFrm]];
      RuT.Apply(_C.m_ba, ba);
      RuT.Apply(_C.m_bw, bw);
      ba += m_K.m_ba;
      bw += m_K.m_bw;
      fprintf(fp, " %f %f %f %f %f %f %f %f %f", _C.m_v.x(), _C.m_v.y(), _C.m_v.z(),
                                                 ba.x(), ba.y(), ba.z(),
                                                 bw.x(), bw.y(), bw.z());
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  UT::PrintSaved(fileName);
  return true;
}

bool GlobalBundleAdjustor::SaveCosts(const std::string fileName, const int type) {
  FILE *fp = fopen(fileName.c_str(), "w");
  if (!fp) {
    return false;
  }
#ifdef CFG_HISTORY
  const int N = static_cast<int>(m_hists.size());
  for (int i = 0; i < N; ++i) {
    const History &hist = m_hists[i];
    switch (type) {
    case 0: hist.m_ESa.Save(fp);     break;
    case 1: hist.m_ESb.Save(fp);     break;
    case 2: hist.m_ESp.Save(fp);     break;
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

bool GlobalBundleAdjustor::SaveResiduals(const std::string fileName, const int type) {
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

void GlobalBundleAdjustor::ComputeErrorFeature(float *ex) {
  Rigid3D Tr[2];
  FTR::Error e;
  float exi;
#if 0
  float Se2;
#ifdef CFG_STEREO
  float Ser2;
#endif
  int SN;
#endif
  *ex = 0.0f;
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {
#if 0
    Se2 = 0.0f;
#ifdef CFG_STEREO
    Ser2 = 0.0f;
#endif
    SN = 0;
#else
    const KeyFrame &KF = m_KFs[iKF];
    const int Nz = static_cast<int>(KF.m_zs.size());
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
    const Rigid3D C = m_Cs[iKF];
    const int NZ = static_cast<int>(KF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ) {
      const FRM::Measurement &Z = KF.m_Zs[iZ];
      *Tr = C / m_Cs[Z.m_iKF];
#ifdef CFG_STEREO
      Tr[1] = Tr[0];
      Tr[1].SetTranslation(m_K.m_br + Tr[0].GetTranslation());
#endif
      const Depth::InverseGaussian *_ds = m_ds.data() + m_iKF2d[Z.m_iKF];
      const KeyFrame &_KF = m_KFs[Z.m_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
        const FTR::Measurement &z = KF.m_zs[iz];
        const int ix = z.m_ix;
        FTR::GetError(Tr, _KF.m_xs[ix], _ds[ix], z, e);
#ifdef CFG_STEREO
        if (z.m_z.Valid()) {
          const float e2 = e.m_e.SquaredLength();
#if 0
          Se2 = e2 + Se2;
          ++SN;
#else
          e2s.Push(e2);
#endif
        }
        if (z.m_zr.Valid()) {
          const float e2r = e.m_er.SquaredLength();
#if 0
          Ser2 = e2r + Ser2;
          ++SN;
#else
          e2rs.Push(e2r);
#endif
        }
#else
        const float e2 = e.m_e.SquaredLength();
#if 0
        Se2 = e2 * m_K.m_K.fxy() + Se2;
        ++SN;
#else
        e2s.Push(e2);
#endif
#endif
      }
    }
#ifdef CFG_STEREO
    const Depth::InverseGaussian *ds = m_ds.data() + m_iKF2d[iKF];
    const int Nx = static_cast<int>(KF.m_xs.size());
    for (int ix = 0; ix < Nx; ++ix) {
      if (KF.m_xs[ix].m_xr.Invalid()) {
        continue;
      }
      FTR::GetError(m_K.m_br, ds[ix], KF.m_xs[ix], e.m_er);
      const float e2r = e.m_er.SquaredLength();
#if 0
      Ser2 = e2r + Ser2;
      ++SN;
#else
      e2rs.Push(e2r);
#endif
    }
#endif
#if 0
#ifdef CFG_STEREO
    exi = sqrtf((Se2 * m_K.m_K.fxy() + Ser2 * m_K.m_Kr.fxy()) / SN);
#else
    exi = sqrtf(Se2 * m_K.m_K.fxy() / SN);
#endif
#else
    if (!e2s.Empty()) {
      const int ith = e2s.Size() >> 1;
      std::nth_element(e2s.Data(), e2s.Data() + ith, e2s.End());
      exi = sqrtf(e2s[ith] * m_K.m_K.fxy());
    } else
      exi = 0.0f;
#ifdef CFG_STEREO
    if (!e2rs.Empty()) {
      const int ith = e2rs.Size() >> 1;
      std::nth_element(e2rs.Data(), e2rs.Data() + ith, e2rs.End());
      exi = std::max(sqrtf(e2rs[ith] * m_K.m_K.fxy()), exi);
    }
#endif
#endif
    *ex = std::max(exi, *ex);
  }
}

void GlobalBundleAdjustor::ComputeErrorIMU(float *er, float *ep, float *ev,
                                           float *eba, float *ebw) {
  *er = *ep = *ev = *eba = *ebw = 0.0f;
  const int Nc = m_Cs.Size();
  for (int ic1 = Nc - m_CsLM.Size(), ic2 = ic1 + 1, im1 = 0, im2 = 1; ic2 < Nc; 
       ic1 = ic2++, im1 = im2++) {
    if (m_KFs[ic2].m_us.Empty()) {
      continue;
    }
    const IMU::Delta::Error e = m_DsLM[im2].GetError(m_CsLM[im1], m_CsLM[im2], m_K.m_pu,
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

void GlobalBundleAdjustor::ComputeErrorDrift(float *er, float *ep) {
  *er = 0.0f;
  *ep = 0.0f;
#ifdef CFG_GROUND_TRUTH
  if (m_CsKFGT.Empty())
    return;
  Rigid3D Tr, TrGT, Te;
  LA::AlignedVector3f _er, _ep;
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    const int iKFNearest = m_KFs[iKF].m_iKFNearest;
    if (iKFNearest == -1) {
      continue;
    }
    Tr = m_Cs[iKF] / m_Cs[iKFNearest];
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

float GlobalBundleAdjustor::ComputeRMSE() {
  float Se2 = 0.0f;
  const int Nc1 = static_cast<int>(m_CsDel.size()), Nc2 = m_Cs.Size(), Nc = Nc1 + Nc2;
#ifdef CFG_GROUND_TRUTH
  if (m_CsGT) {
    for (int ic = 0; ic < Nc1; ++ic) {
      const HistoryCamera &C = m_CsDel[ic];
      const Point3D p1 = C.m_C.GetPosition(), p2 = m_CsGT[C.m_iFrm].m_p;
      Se2 += (p1 - p2).SquaredLength();
    }
  }
  if (!m_CsKFGT.Empty()) {
    for (int ic = 0; ic < Nc2; ++ic) {
      const Point3D p1 = m_Cs[ic].GetPosition(), p2 = m_CsKFGT[ic].GetPosition();
      Se2 += (p1 - p2).SquaredLength();
    }
  }
#endif
  return sqrtf(Se2 / Nc2);
}
//同步数据,比如当新的关键帧老的时候需要进行数据关联,当滑窗merge的时候,需要将滑窗对先验获取
void GlobalBundleAdjustor::SynchronizeData() {
  const int iFrm = m_iFrms.empty() ? MT_TASK_NONE : m_iFrms.back();//当前帧id
  MT_WRITE_LOCK_BEGIN(m_MT, iFrm, MT_TASK_GBA_SynchronizeData);
  m_ITs1.swap(m_ITs2);
  m_IKFs1.swap(m_IKFs2);
  m_IDKFs1.swap(m_IDKFs2);
  m_IDMPs1.swap(m_IDMPs2);
  m_IUCs1.swap(m_IUCs2);
  m_IZps1.swap(m_IZps2);
  if (m_IZpLM1.Valid()) {
    m_IZpLM2 = m_IZpLM1;
    m_IZpLM1.Invalidate();
  }
  MT_WRITE_LOCK_END(m_MT, iFrm, MT_TASK_GBA_SynchronizeData);
  m_Ucs.assign(m_KFs.size(), GM_FLAG_FRAME_DEFAULT);
#if defined CFG_GROUND_TRUTH && defined CFG_HISTORY
  if (m_history >= 3) {
    m_ucsGT.assign(m_KFs.size(), GBA_FLAG_FRAME_DEFAULT);
  }
#endif

  while (!m_ITs2.empty()) {//遍历所有的输入帧
    const InputType IT = m_ITs2.front();//输入类型
    m_ITs2.pop_front();
    if (IT == IT_KEY_FRAME) {
      PushKeyFrame(m_IKFs2.front());//向GBA中加入关键帧
      m_IKFs2.pop_front();
    } else if (IT == IT_DELETE_KEY_FRAME) {
      DeleteKeyFrame(m_IDKFs2.front());
      m_IDKFs2.pop_front();
    } else if (IT == IT_DELETE_MAP_POINTS) {
      DeleteMapPoints(m_IDMPs2.front());
      m_IDMPs2.pop_front();
    } else if (IT == IT_UPDATE_CAMERAS) {
      UpdateCameras(m_IUCs2.front());
      m_IUCs2.pop_front();
    } else if (IT == IT_CAMERA_PRIOR_POSE) {//滑窗时最老帧是关键帧或者参考关键帧变化时滑窗会给GBA位姿先验
      PushCameraPriorPose(m_IZps2.front());
      m_IZps2.pop_front();
    } else if (IT == IT_CAMERA_PRIOR_MOTION) {//滑窗时最老帧是关键帧滑窗会给GBA motion先验
      if (m_IZpLM2.Valid()) {
        SetCameraPriorMotion(m_IZpLM2);//如果滑窗边缘化了一个关键帧对应的普通帧,就会给关键帧一个先验约束
        m_IZpLM2.Invalidate();
      }
    }
  }
}

void GlobalBundleAdjustor::UpdateData() {
  m_GM->GBA_Update(m_iFrms/*关键帧和普通帧之间的id对应*/, m_Cs/*优化完更新以后的左相机的位姿*/, m_Ucs
#ifdef CFG_HANDLE_SCALE_JUMP
                 , m_dsKF
#endif
                 );
#ifdef CFG_VERBOSE
  if (m_verbose >= 2) {
    const int Nc = static_cast<int>(m_KFs.size()), Nd = static_cast<int>(m_ds.size());
    const int Ncu = UT::VectorCountFlag<ubyte>(m_Ucs, GM_FLAG_FRAME_UPDATE_CAMERA);
    //const int Ndu = UT::VectorCountFlag<ubyte>(m_Uds, GM_FLAG_TRACK_UPDATE_DEPTH);
    UT::PrintSeparator();
    UT::Print("[%d] [GlobalBundleAdjustor::UpdateData]\n", m_iFrms.back());
    UT::Print("  Camera = %d / %d = %.2f%%\n", Ncu, Nc, UT::Percentage(Ncu, Nc));
    //UT::Print("  Depth  = %d / %d = %.2f%%\n", Ndu, Nd, UT::Percentage(Ndu, Nd));
  }
#endif
}

bool GlobalBundleAdjustor::BufferDataEmpty() {
  return m_IKFs1.empty() && m_IDKFs1.empty() && m_IUCs1.empty();
}
//向GBA中加入关键帧,流程和LBA的差不多
//step1:初始化GBA的m_KFs中的当前关键帧
//step2:遍历所有共视关键帧,将当前共视关键帧和当前关键帧之间的共视信息进行关联,都观测到了同一个点就存储下来,m_iz1是F1对这个地图点
// 的观测索引,m_iz2是F1对这个地图点的观测索引
//step3:将观测的匹配信息加入Zm中,并进行相关H矩阵的扩容,当前共视关键帧也要更新当前关键帧作为自己作为自己的共视帧
//step4:处理一下imu非法的情况,因为前后两个关键帧可能会隔的比较久
//step5:向m_Cs中存储这个关键帧的左相机pose,向m_CsLM中存储这个关键帧的左相机状态
//step6:计算当前关键帧和之前关键帧之间的imu约束,以及重新计算之前帧和前前帧之间的imu约束(此时最后一次a,w的测量均值用的是当前帧
// 第一个imu测量和之前的imu测量做中值,保证了数据的一致性)
//step7:遍历所有的观测关键帧然后遍历观测关键帧中的地图点,如果地图点有变动,则更新地图点
//step8:将所有地图点的逆深度存到m_ds里,以及m_xsTmp里新地图点观测信息的存储,最终存储到m_KFs中这帧的m_xs中
// (中间有一步我没看懂在干啥,感觉也并不会出现这种情况啊,就是这帧观测的新地图点已经被别的关键帧观测了,难道是闭环时用的?)
void GlobalBundleAdjustor::PushKeyFrame(const InputKeyFrame &IKF)
{
  //Timer timer;
  //timer.Start();
  const int nKFs1 = static_cast<int>(m_KFs.size())/*之前有多少帧关键帧，也可以用作关键帧id*/, nKFs2 = nKFs1 + 1;/*需要扩容的size*/
  const int Nm1 = m_CsLM.Size()/*关键帧id*/, Nm2 = Nm1 + 1;

  m_KFs.resize(nKFs2);//和LBA一样，需要的数据扩容
  m_iFrms.resize(nKFs2);
  m_Cs.Resize(nKFs2, true);
  m_CsLM.Resize(Nm2, true);
#ifdef CFG_GROUND_TRUTH
  if (m_CsGT)
  {
    m_CsKFGT.Resize(nKFs2, true);
    m_CsLMGT.Resize(Nm2, true);
  }
#endif
  m_ucs.resize(nKFs2, GBA_FLAG_FRAME_UPDATE_CAMERA);
  m_Ucs.resize(nKFs2, GM_FLAG_FRAME_DEFAULT);
#ifdef CFG_INCREMENTAL_PCG
  m_xcs.Resize(nKFs2, true);
  m_xmsLM.Resize(Nm2, true);
#endif
  const ubyte ucmFlag1 = GBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION |
                         GBA_FLAG_CAMERA_MOTION_UPDATE_POSITION;
  const ubyte ucmFlag2 = GBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY |
                         GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_ACCELERATION |
                         GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_GYROSCOPE;
  m_ucmsLM.resize(Nm2, ucmFlag1 | ucmFlag2);
  m_DsLM.Resize(Nm2, true);
#ifdef CFG_GROUND_TRUTH
  if (m_CsGT)
  {
    m_DsLMGT.Resize(Nm2, true);
  }
#ifdef CFG_HISTORY
  if (m_history >= 3)
  {
    m_ucsGT.resize(m_KFs.size(), GBA_FLAG_FRAME_DEFAULT);
  }
#endif
#endif
#ifdef CFG_HANDLE_SCALE_JUMP
  m_dsKF.resize(nKFs2, 0.0f);
#endif
  const ubyte udFlag1 = GBA_FLAG_TRACK_UPDATE_DEPTH | GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;
  const ubyte udFlag2 = GBA_FLAG_TRACK_UPDATE_DEPTH
#ifdef GBA_FLAG_TRACK_MEASURE
                      | GBA_FLAG_TRACK_MEASURE
#endif
                      ;
  const int iKF = nKFs1, im/*关键帧id*/ = Nm1;
  KeyFrame &KF = m_KFs[iKF];//构造GBA的m_KFs中的当前关键帧
  KF.Initialize(IKF);//构造关键帧
  const int iFrm = KF.m_T.m_iFrm;//帧id
  m_iFrms[iKF] = iFrm;//全局帧id和关键帧id的对应
  std::vector<int> &iKF2X = m_idxsTmp1, &iX2z = m_idxsTmp2;
  PushFeatureMeasurementMatchesFirst(KF/*当前帧*/, iKF2X, iX2z/*地图点被观测到,就记录这个地图点在m_zs存储的起始位置*/);
  const int Nk = static_cast<int>(KF.m_iKFsMatch.size());//这个关键帧的共视的老关键帧
#ifdef CFG_DEBUG
  for (int ik = 0; ik < Nk; ++ik) {
    UT_ASSERT(KF.m_iKFsMatch[ik] < iKF);
  }
#endif

//遍历所有共视关键帧,将当前共视关键帧和当前关键帧之间的共视信息进行关联,都观测到了同一个点就存储下来,m_iz1是F1对这个地图点的观测索引,m_iz2是F1对这个地图点的观测索引
//将观测的匹配信息加入Zm中,并进行相关H矩阵的扩容,当前共视关键帧也要更新当前关键帧作为自己作为自己的共视帧
  for (int ik = 0; ik < Nk; ++ik)
  {
    KeyFrame &_KF = m_KFs[KF.m_iKFsMatch[ik]];
      //将_KF和KF之间的共视信息进行关联,都观测到了同一个点就存储下来,m_iz1是F1对这个地图点的观测索引,m_iz2是F1对这个地图点的观测索引
//将观测的匹配信息加入Zm中,并进行相关H矩阵的扩容
    PushFeatureMeasurementMatchesNext(_KF/*观测到的关键帧*/, KF/*当前关键帧*/, iKF2X/*[i]非-1时表示第i个关键帧之前的共视帧有多少个地图点*/,
            iX2z/*储存的是当前这帧的共视帧的所有地图点数量,地图点被观测到,就记录这个地图点在m_zs存储的起始位置*/, KF.m_Zm/*较新帧和之后几帧之间的匹配关联*/);

#ifdef CFG_DEBUG
    UT_ASSERT(_KF.m_iKFsMatch.empty() || _KF.m_iKFsMatch.back() < iKF);
#endif
    _KF.m_iKFsMatch.push_back(iKF);
  }


  if (iKF > 0)//处理一下imu非法的情况
  {
    const int _iKF = iKF - 1;
    if (KF.m_us.Empty())//如果当前帧没有imu数据,flag设为非法
    {
      m_ucmsLM[im] |= GBA_FLAG_CAMERA_MOTION_INVALID;
    } else
    {
      if ((KF.m_T.m_t - m_KFs[_iKF].m_T.m_t) > GBA_MAX_IMU_TIME_INTERVAL)//两帧之间时间间隔过长,清空imu,flag设为非法
      {
        KF.m_us.Clear();
        m_ucmsLM[im] |= GBA_FLAG_CAMERA_MOTION_INVALID;
      }
      else if (KF.m_iKFsMatch.empty() || KF.m_iKFsMatch.back() != _iKF)
      {
        KF.InsertMatchKeyFrame(_iKF);
        m_KFs[_iKF].m_iKFsMatch.push_back(iKF);
      }
      if (!KF.m_us.Empty()) {//如果imu数据不为空,
        const int _im = im - 1;
        if (_im >= 0 && (m_ucmsLM[_im] & GBA_FLAG_CAMERA_MOTION_INVALID))
        {
          m_ucmsLM[_im] &= ~GBA_FLAG_CAMERA_MOTION_INVALID;
          m_ucmsLM[_im] |= ucmFlag2;
        }
      }
    }
  }
  const bool v1 = IKF.m_Cam_state.m_Cam_pose.Valid(), v2 = IKF.m_Cam_state.m_v.Valid();//之前已经用imu测量初始化过了,应该都是true
  if (im > 0 && !KF.m_us.Empty() && (GBA_PROPAGATE_CAMERA || !v1 || !v2)) {//这里不进行预积分
    IMU::Delta D;
    Camera C;
    const KeyFrame &_KF = m_KFs[iKF - 1];
    const Camera &_C = m_CsLM[im - 1];
    IMU::PreIntegrate(KF.m_us, _KF.m_T.m_t, KF.m_T.m_t, _C, &D, &m_work, false,
                      _KF.m_us.Empty() ? NULL : &_KF.m_us.Back(), NULL, BA_ANGLE_EPSILON);
    IMU::Propagate(m_K.m_pu, D, _C, C, BA_ANGLE_EPSILON);
    if (GBA_PROPAGATE_CAMERA) {
      m_CsLM[im] = C;
      m_Cs[iKF] = m_CsLM[im].m_Cam_pose;
    } else {
      if (!v1) {
        m_Cs[iKF] = C.m_Cam_pose;
        m_CsLM[im].m_Cam_pose = C.m_Cam_pose;
        m_CsLM[im].m_p = C.m_p;
      }
      if (!v2) {
        m_CsLM[im].m_v = C.m_v;
        m_CsLM[im].m_ba = C.m_ba;
        m_CsLM[im].m_bw = C.m_bw;
      }
    }
  } else {
    m_Cs[iKF] = IKF.m_Cam_state.m_Cam_pose;//相机pose和相机状态的设置
    m_CsLM[im] = IKF.m_Cam_state;
  }
#ifdef CFG_GROUND_TRUTH
  if (m_CsGT) {
    m_CsKFGT[iKF] = m_CsGT[iFrm].m_Cam_pose;
    m_CsLMGT[im] = m_CsGT[iFrm];
  }
#endif
#ifdef CFG_HANDLE_SCALE_JUMP
  m_dsKF[iKF] = IKF.m_d;
#endif
#ifdef CFG_INCREMENTAL_PCG
  m_xcs[iKF].MakeZero();//pose和motion的增量置0
  m_xmsLM[im].MakeZero();
#endif
  IMU::Delta &D = m_DsLM[im];
  if (KF.m_us.Empty() || im == 0)
  {//如果没有imu测量或者是第一帧,就设成非法
    D.Invalidate();
  } else
  {
      //前后两个关键帧之间的imu约束,算一下imu预积分
    const int _iKF = iKF - 1;
    const int _im = im - 1;
    const KeyFrame &_KF = m_KFs[_iKF];
    const float _t = _KF.m_T.m_t;//前一帧关键很
    IMU::PreIntegrate(KF.m_us/*当前关键帧和之前关键帧之间的imu测量*/, _t/*上一关键帧的时间戳*/, KF.m_T.m_t/*当前关键帧的时间戳*/,
            m_CsLM[_im]/*上一关键帧的状态*/, &D/*预积分部分*/, &m_work, true,/*是否要输出雅克比*/
                      _KF.m_us.Empty() ? NULL : &_KF.m_us.Back()/*上一关键帧最后一个imu测量*/, NULL, BA_ANGLE_EPSILON);
    if (_im > 0 && !_KF.m_us.Empty())
    {
        //这里就是重新算了上一帧的预积分,这次的a,w的测量均值不再是k k+1取均值,而是i,j取中值(为了imu数据的连续性)
      IMU::Delta &_D = m_DsLM[_im];//前一帧对应的预积分(前一帧和前前帧之间的),保存了预积分以后的状态量,协方差,信息矩阵,以及对ba,bw的雅克比
      IMU::PreIntegrate(_KF.m_us/*之前关键帧和前前关键帧之间的imu测量*/, m_KFs[_iKF - 1].m_T.m_t, _t, m_CsLM[_im - 1], &_D, &m_work, true,
                        _D.m_u1.Valid() ? &_D.m_u1/*前前帧的最后一个imu测量*/ : NULL, &KF.m_us.Front()/*当前帧第一个imu测量*/, BA_ANGLE_EPSILON);
      m_ucs[_iKF] |= GBA_FLAG_FRAME_UPDATE_CAMERA;
      m_ucmsLM[_im] |= ucmFlag1 | ucmFlag2;//

    }
  }
  if (iFrm == 0) {//如果是第一帧,进行固定
    const LA::Vector3f s2r = LA::Vector3f::Get(BA_VARIANCE_FIX_ORIGIN_ROTATION_X,/*0*/
                                               BA_VARIANCE_FIX_ORIGIN_ROTATION_Y,/*0*/
                                               BA_VARIANCE_FIX_ORIGIN_ROTATION_Z/*(0.01*pi/180)^2*/);//yaw没有被固定
    const float s2p = BA_VARIANCE_FIX_ORIGIN_POSITION;//1.0e-6f 0.001^2
    m_Zo.Set(BA_WEIGHT_FIX_ORIGIN/*1.0e-5f*/, s2r/*首帧R初始值*/, s2p/*首帧t初始值*/, m_Cs[iKF]/*首帧pose*/);
    m_Ao.MakeZero();
  }
#ifdef CFG_GROUND_TRUTH
  if (m_CsGT) {
    if (KF.m_us.Empty() || im == 0) {
      m_DsLMGT[im] = D;
    } else {
      const int _iKF = iKF - 1, _im = im - 1;
      const KeyFrame &_KF = m_KFs[_iKF];
      const float _t = _KF.m_T.m_t;
      const Camera &_C = m_CsLMGT[_im];
      IMU::PreIntegrate(KF.m_us, _t, KF.m_T.m_t, _C, &m_DsLMGT[im], &m_work, true,
                        _KF.m_us.Empty() ? NULL : &_KF.m_us.Back(), NULL, BA_ANGLE_EPSILON);

      if (_im > 0 && !_KF.m_us.Empty()) {
        IMU::Delta &_D = m_DsLMGT[_im];
        IMU::PreIntegrate(_KF.m_us, m_KFs[_iKF - 1].m_T.m_t, _t, m_CsLMGT[_im - 1], &_D, &m_work,
                          true, _D.m_u1.Valid() ? &_D.m_u1 : NULL, &KF.m_us.Front(),
                          BA_ANGLE_EPSILON);

      }
    }
  }
#endif
//遍历所有的观测关键帧然后遍历观测关键帧中的地图点,如果地图点有变动,则更新地图点
  m_iKF2d.push_back(m_iKF2d.back());
  m_iKF2cb.push_back(m_iKF2cb.back() + static_cast<int>(KF.m_ikp2KF.size()));//KF.m_ikp2KF.size())是这个关键帧和之前的多少个关键帧有共视
  //m_iKF2cb存储这个关键帧来之前其他的关键帧总共有多少个共视,为的是方便索引
  const int NZ = static_cast<int>(IKF.m_Zs.size());//当前关键帧的所有观测
  for (int iZ = 0; iZ < NZ; ++iZ)
  {
    const FRM::Measurement &Z = IKF.m_Zs[iZ];
    const int _iKF = Z.m_iKF, id = m_iKF2d[_iKF];
    Depth::InverseGaussian *ds = m_ds.data() + id;
    ubyte *uds = m_uds.data() + id/*, *Uds = m_Uds.data() + id*/;
#ifdef CFG_GROUND_TRUTH
    ubyte *udsGT = m_udsGT.data() + id;
#endif
    bool ud = false;
    const KeyFrame &_KF = m_KFs[_iKF];
    for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz)//遍历所有的观测关键帧然后遍历观测关键帧中的地图点,如果地图点有变动,则更新地图点
    {
      const int ix = IKF.m_zs[iz].m_ix;//逆深度不是地图点平均逆深度(说明是三角化出来的点的逆深度)
      if (IKF.m_dzs[iz].Valid() && fabs(IKF.m_dzs[iz].u() - ds[ix].u()) >= BA_UPDATE_DEPTH)
      {
        ds[ix] = IKF.m_dzs[iz];//那么就更新地图点的逆深度并且flags设置为需要更新深度
        m_ucs[_iKF] |= GBA_FLAG_FRAME_UPDATE_DEPTH;
        uds[ix] |= udFlag2;
        //m_Ucs[_iKF] |= GM_FLAG_FRAME_UPDATE_DEPTH;
        //Uds[ix] |= GM_FLAG_TRACK_UPDATE_DEPTH;
        ud = true;
      }

#if defined CFG_GROUND_TRUTH && defined CFG_HISTORY
      if (m_history >= 3) {
        m_ucsGT[_iKF] |= GBA_FLAG_FRAME_UPDATE_DEPTH;
        udsGT[ix] |= GBA_FLAG_TRACK_UPDATE_DEPTH;
      }
#endif
    }

  }
  const int NX = static_cast<int>(IKF.m_Xs.size());//关键帧新看到的地图点
  for (int iX1 = 0, iX2 = 0; iX1 < NX; iX1 = iX2)
  {
    const int _iKF = IKF.m_Xs[iX1].m_iKF;//当前关键帧id
    for (iX2 = iX1 + 1; iX2 < NX && IKF.m_Xs[iX2].m_iKF == _iKF; ++iX2) {}
    const int id = m_iKF2d[_iKF + 1]/*新的地图点在全局地图点中的id*/, Nx = iX2 - iX1;

    for (int jKF = _iKF; jKF <= iKF; ++jKF)
    {
      m_iKF2d[jKF + 1] += Nx;//加上这次观测到的新地图点数量
    }
    m_ds.insert(m_ds.begin() + id, Nx, Depth::InverseGaussian());//扩容逆深度以及相关flags
    m_uds.insert(m_uds.begin() + id, Nx, udFlag1);
    m_ucs[_iKF] |= GBA_FLAG_FRAME_UPDATE_DEPTH;
#if defined CFG_GROUND_TRUTH && defined CFG_HISTORY
    if (m_history >= 3) {
      m_ucsGT[_iKF] |= GBA_FLAG_FRAME_UPDATE_DEPTH;
      m_udsGT.insert(m_udsGT.begin() + id, Nx, GBA_FLAG_TRACK_UPDATE_DEPTH);
    }
#endif
    const GlobalMap::Point *Xs = IKF.m_Xs.data() + iX1;//当前关键帧首个地图点对应的指针
    Depth::InverseGaussian *ds = m_ds.data() + id;//当前地图点需要构造的对应的逆深度
    ubyte *uds = m_uds.data() + id;
    m_xsTmp.resize(Nx);
    std::vector<ubyte> &mcs = m_marksTmp1;
    mcs.assign(nKFs2, 0);//清空上一次的标识,赋值
    for (int i = 0; i < Nx; ++i)
    {//遍历所有地图点将所有地图点的逆深度存到m_ds里,以及m_xsTmp里新地图点观测信息的存储
      const GlobalMap::Point &X = Xs[i];//当前地图点
      ds[i] = X.m_d;//深度信息赋值
      m_xsTmp[i] = X.m_x;//新地图点的观测信息
      const int Nz = static_cast<int>(X.m_zs.size());

      for (int j = 0; j < Nz; ++j)
      {
        mcs[X.m_zs[j].m_iKF] = 1;
      }
    }

    std::vector<int> &ik2KF = m_idxsTmp1, &iKF2k = m_idxsTmp2;
    ik2KF.resize(0);
    iKF2k.assign(nKFs2, -1);
    for (int jKF = 0; jKF < nKFs2; ++jKF)//遍历当前所有的关键帧,这里干啥不清楚
    {
      if (!mcs[jKF])
      {
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
    const int Nx1 = static_cast<int>(m_KFs[_iKF].m_xs.size());
    for (int i = 0, ix = Nx1; i < Nx; ++i, ++ix)
    {//遍历所有新地图点
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
    std::vector<int> &izs = m_idxsTmp2, &ix2z = m_idxsTmp3;
    izs.resize(Nk);
    KeyFrame &_KF = m_KFs[_iKF];
    for (int ik2 = 0; ik2 < Nk; ++ik2)//这里好像用不到啊
    {
      const int iKF2 = ik2KF[ik2];
      KeyFrame &KF2 = m_KFs[iKF2];
      int jk2, &iz2 = izs[ik2];
      const std::vector<FTR::Measurement> &zs2 = m_zsListTmp[ik2];
      const int Nk21 = static_cast<int>(KF2.m_iKFsMatch.size());
      KF2.PushFeatureMeasurements(_iKF, zs2, &jk2, &iz2, &m_work);
      const int Nk22 = static_cast<int>(KF2.m_iKFsMatch.size());
      if (Nk22 > Nk21) {
        const std::vector<int>::iterator _ik = std::lower_bound(_KF.m_iKFsMatch.begin() +
                                                                _KF.m_Zm.m_SMczms.Size(),
                                                                _KF.m_iKFsMatch.end(), iKF2);

        _KF.m_iKFsMatch.insert(_ik, iKF2);
      }
      const int Nz2 = static_cast<int>(zs2.size());
      KF2.m_Zm.InsertFeatureMeasurement2(jk2, iz2, Nz2);
      for (int jk2 = KF2.m_Zm.m_SMczms.Size(); jk2 < Nk22; ++jk2)
      {
        KeyFrame &KF3 = m_KFs[KF2.m_iKFsMatch[jk2]];
        const int jk3 = KF3.SearchMatchKeyFrame(iKF2);

        KF3.m_Zm.InsertFeatureMeasurement1(jk3, iz2, Nz2);
      }
      ix2z.assign(Nx, -1);
      int *_ix2z = ix2z.data() - Nx1;
      for (int i = 0; i < Nz2; ++i)
      {
        const int ix = zs2[i].m_ix;

        _ix2z[ix] = iz2 + i;
      }
      for (int ik1 = 0; ik1 < ik2; ++ik1)
      {
        const int iKF1 = ik2KF[ik1];
        m_izmsTmp.resize(0);
        const int iz1 = izs[ik1];
        const std::vector<FTR::Measurement> &zs1 = m_zsListTmp[ik1];
        const int Nz1 = static_cast<int>(zs1.size());
        for (int i = 0; i < Nz1; ++i)
        {
          const int ix = zs1[i].m_ix, _iz2 = _ix2z[ix];
          if (_iz2 == -1)
          {
            continue;
          }

          m_izmsTmp.push_back(FTR::Measurement::Match(iz1 + i, _iz2));
        }
        if (m_izmsTmp.empty())
        {
          continue;
        }
        const std::vector<int>::iterator _jk2 = std::lower_bound(KF2.m_iKFsMatch.begin() + jk2,
                                                                 KF2.m_iKFsMatch.begin() +
                                                                 KF2.m_Zm.m_SMczms.Size(), iKF1);
        jk2 = static_cast<int>(_jk2 - KF2.m_iKFsMatch.begin());
        if (_jk2 == KF2.m_iKFsMatch.end() || *_jk2 != iKF1)
        {
          KF2.InsertMatchKeyFrame(iKF1, &_jk2);
          KeyFrame &KF1 = m_KFs[iKF1];
          const std::vector<int>::iterator jk1 = std::lower_bound(KF1.m_iKFsMatch.begin() +
                                                                  KF1.m_Zm.m_SMczms.Size(),
                                                                  KF1.m_iKFsMatch.end(), iKF2);

          KF1.m_iKFsMatch.insert(jk1, iKF2);
        }
        KF2.m_Zm.InsertFeatureMeasurementMatches(jk2, m_izmsTmp, &m_work);

      }
      const int Nk23 = static_cast<int>(KF2.m_iKFsMatch.size());

      if (Nk23 == Nk21)
      {
        continue;
      }
      const int Ncb = Nk23 - Nk21;
      for (int jKF = iKF2; jKF <= iKF; ++jKF)
      {
        m_iKF2cb[jKF + 1] += Ncb;
      }
    }
    m_KFs[_iKF].PushFeatures(m_xsTmp/*所有新地图点（由当前关键帧产生）的观测*/);
  }
  m_AdsLM.InsertZero(Nm1);//初始化一下
  m_Afps.InsertZero(nKFs1);
  m_AfmsLM.InsertZero(Nm1);
  m_SAcus.InsertZero(nKFs1);
  m_SMcus.InsertZero(nKFs1);
  m_SAcmsLM.InsertZero(Nm1);
}

void GlobalBundleAdjustor::GetMpInfo(std::vector<bool> &kfs_stereoz)
{
    int N_total = 0;
    for (int i = 0; i < m_KFs.size(); ++i)
    {
        int Nx = int(m_KFs[i].m_xs.size());
        kfs_stereoz.resize(N_total + Nx,false);
        for (int ix = 0; ix < Nx; ++ix)
        {
            if(m_KFs[i].m_xs[ix].m_xr.Valid())
                kfs_stereoz[N_total + ix] = true;
        }
        N_total += Nx;
    }
}

void GlobalBundleAdjustor::GetGbaInfo(std::vector<int> & iFrms, std::vector<Rigid3D> & Cs ,
        std::vector<ubyte> & ucs,std::vector<std::vector<int>> & CovisibleKFs,std::vector<bool> &lastkf_stereoz)
{

    iFrms  = m_iFrms;
    for (int i = 0; i < m_Cs.Size(); ++i)
    {
        Cs.push_back(m_Cs[i]);
        std::vector<int> cur_CovisibleKFs;
        for (int j = 0; j < m_KFs[i].m_iKFsMatch.size(); ++j) //这个关键帧的共视帧,我只要iFrm的id
            cur_CovisibleKFs.push_back(m_iFrms[m_KFs[i].m_iKFsMatch[j]]);
        CovisibleKFs.push_back(cur_CovisibleKFs);
    }

    if(!(m_KFs.size() == 1))
    {
        const int Nx = int(m_KFs[m_KFs.size()-2].m_xs.size());
        lastkf_stereoz.resize(Nx,false);
        for (int ix = 0; ix < Nx; ++ix)
        {
            if(m_KFs[m_KFs.size()-2].m_xs[ix].m_xr.Valid())
                lastkf_stereoz[ix] = true;
        }
    }

    ucs = m_ucs;
}

void GlobalBundleAdjustor::DeleteKeyFrame(const int iKF) {
  //Timer timer;
  //timer.Start();
  const int iFrm = m_iFrms[iKF];
  const int nKFs = static_cast<int>(m_KFs.size());
#ifdef CFG_DEBUG
  UT_ASSERT(iKF != nKFs - 1);
#endif
  m_KFs[iKF + 1].m_us.Insert(0, m_KFs[iKF].m_us, &m_work);
  if (iKF > 0 && !m_KFs[iKF + 1].m_us.Empty() &&
      m_KFs[iKF + 1].SearchMatchKeyFrame(iKF - 1) == -1) {
    m_KFs[iKF + 1].InsertMatchKeyFrame(iKF - 1);
    KeyFrame &KF = m_KFs[iKF - 1];
    const std::vector<int>::iterator ik = std::lower_bound(KF.m_iKFsMatch.begin() +
                                                           KF.m_Zm.m_SMczms.Size(),
                                                           KF.m_iKFsMatch.end(), iKF + 1);
#ifdef CFG_DEBUG
    UT_ASSERT(ik == KF.m_iKFsMatch.end() || *ik != iKF + 1);
#endif
    KF.m_iKFsMatch.insert(ik, iKF + 1);
  }
  const int Nd = static_cast<int>(m_KFs[iKF].m_xs.size());
  const int id1 = m_iKF2d[iKF], id2 = id1 + Nd;
  const ubyte *uds = m_uds.data() + id1;
  m_iKF2Z.resize(nKFs);
  for (int jKF2 = iKF + 1; jKF2 < nKFs; ++jKF2) {
    KeyFrame &KF2 = m_KFs[jKF2];
    const std::vector<FRM::Measurement>::iterator iZ2 = std::lower_bound(KF2.m_Zs.begin(),
                                                                         KF2.m_Zs.end(), iKF);
    m_iKF2Z[jKF2] = iZ2;
    const bool z2 = iZ2 != KF2.m_Zs.end() && iZ2->m_iKF == iKF;
    //const int Nk = static_cast<int>(KF2.m_iKFsMatch.size());
    const int Nk = KF2.m_Zm.m_SMczms.Size();
    for (int ik = 0; ik < Nk; ++ik) {
      const int jKF1 = KF2.m_iKFsMatch[ik];
      const std::vector<FRM::Measurement>::iterator iZ1 = m_iKF2Z[jKF1];
      const KeyFrame &KF1 = m_KFs[jKF1];
      const bool z1 = jKF1 > iKF && iZ1 != KF1.m_Zs.end() && iZ1->m_iKF == iKF;
      if (!z1 && !z2) {
        continue;
      }
      std::vector<FTR::Measurement::Match> &izms = KF2.m_Zm.m_izms;
      const int i1 = KF2.m_Zm.m_ik2zm[ik], i2 = KF2.m_Zm.m_ik2zm[ik + 1];
      if (z1) {
        const int Nz1 = iZ1->CountFeatureMeasurements();
        for (int i = i2 - 1; i >= i1 && izms[i].m_iz1 >= iZ1->m_iz2; --i) {
          izms[i].m_iz1 -= Nz1;
        }
      }
      if (z2) {
        const int Nz2 = iZ2->CountFeatureMeasurements();
        int i;
        for (i = i2 - 1; i >= i1 && izms[i].m_iz2 >= iZ2->m_iz2; --i) {
          izms[i].m_iz2 -= Nz2;
        }
        if (!z1) {
          continue;
        }
        const int j2 = i + 1;
        const int j1 = static_cast<int>(std::lower_bound(izms.begin() + i1,
                                                         izms.begin() + j2, iZ2->m_iz1) -
                                                         izms.begin());
        const int Nzm = j2 - j1;
        m_marksTmp1.assign(Nzm, 0);
        ubyte *ms = m_marksTmp1.data() - j1;
        for (int j = j1; j < j2; ++j) {
          const int ix = KF2.m_zs[izms[j].m_iz2].m_ix;
          ms[j] = (uds[ix] & GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO) ? 0 : 1;
        }
        KF2.m_Zm.DeleteFeatureMeasurementMatches(ik, j1, j2, ms);
      }
    }
  }
  {
    KeyFrame &KF = m_KFs[iKF];
    const int NZ = static_cast<int>(KF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ) {
      const FRM::Measurement &Z = KF.m_Zs[iZ];
      const int _iKF = Z.m_iKF;
      Camera::Factor::Unitary::CC &SAcxx = m_SAcus[_iKF];
      ubyte *_uds = m_uds.data() + m_iKF2d[_iKF];
      KeyFrame &_KF = m_KFs[_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
        const int ix = KF.m_zs[iz].m_ix;
        FTR::Factor::Full::A2 &A = KF.m_Azs2[iz];
        A.m_Acxx.MakeMinus();
        SAcxx += A.m_Acxx;
        A.m_adx.MakeMinus();
        _KF.m_Axs[ix].m_Sadx += A.m_adx;
        _uds[ix] |= GBA_FLAG_TRACK_UPDATE_INFORMATION;
      }
      if (Z.m_iz1 < Z.m_iz2) {
        m_ucs[_iKF] |= GBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION;
      }
    }
  }
  int icb = m_iKF2cb[iKF];
  for (int jKF = 0; jKF < iKF; ++jKF) {
    m_KFs[jKF].DeleteMatchKeyFrame(iKF);
  }
  for (int jKF = iKF + 1; jKF < nKFs; ++jKF) {
    const std::vector<FRM::Measurement>::iterator iZ = m_iKF2Z[jKF];
    KeyFrame &KF = m_KFs[jKF];
    const int _iZ = iZ != KF.m_Zs.end() && iZ->m_iKF == iKF ?
                      static_cast<int>(iZ - KF.m_Zs.begin()) : -1;
    if (_iZ != -1) {
      Camera::Factor::Unitary::CC &SAczz = m_SAcus[jKF], &SMczz = m_SMcus[jKF];
      for (int iz = iZ->m_iz1; iz < iZ->m_iz2; ++iz) {
        Camera::Factor::Unitary::CC &Aczz = KF.m_Azs2[iz].m_Aczz;
        Aczz.MakeMinus();
        SAczz += Aczz;
        if (uds[KF.m_zs[iz].m_ix] & GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO) {
          continue;
        }
        Camera::Factor::Unitary::CC &Mczz = KF.m_Mzs2[iz].m_Mczz;
        Mczz.MakeMinus();
        SMczz += Mczz;
      }
    }
    KF.DeleteKeyFrame(iKF, &iZ);
    m_iKF2d[jKF] -= Nd;
    m_iKF2cb[jKF] = icb;
    icb += static_cast<int>(KF.m_ikp2KF.size());
  }
  m_CsDel.push_back(HistoryCamera(m_KFs[iKF].m_T, m_Cs[iKF]));
  m_KFs.erase(m_KFs.begin() + iKF);
  m_iFrms.erase(m_iFrms.begin() + iKF);
  m_iKF2d.back() -= Nd;
  m_iKF2d.erase(m_iKF2d.begin() + iKF);
  m_iKF2cb.back() = icb;
  m_iKF2cb.erase(m_iKF2cb.begin() + iKF);
  m_Cs.Erase(iKF);
  const int Nm = m_CsLM.Size(), im = iKF - nKFs + Nm;
  if (im >= 0) {
    m_CsLM.Erase(im);
  }
#ifdef CFG_GROUND_TRUTH
  if (m_CsGT) {
    m_CsKFGT.Erase(iKF);
    if (im >= 0) {
      m_CsLMGT.Erase(im);
    }
  }
#endif
#ifdef CFG_HANDLE_SCALE_JUMP
  m_dsKF.erase(m_dsKF.begin() + iKF);
#endif
#ifdef CFG_INCREMENTAL_PCG
  m_xcs.Erase(iKF);
  if (im >= 0) {
    m_xmsLM.Erase(im);
  }
#endif
  m_ucs.erase(m_ucs.begin() + iKF);
  if (im >= 0) {
    m_ucmsLM.erase(m_ucmsLM.begin() + im);
    const int im1 = im - 1, im2 = im + 1;
    if (im1 >= 0 && im2 < Nm) {
      KeyFrame &KF1 = m_KFs[iKF - 1], &KF2 = m_KFs[iKF];
      IMU::Delta &D = m_DsLM[im2];
      if ((KF2.m_T.m_t - KF1.m_T.m_t) > GBA_MAX_IMU_TIME_INTERVAL) {
        KF2.m_us.Clear();
        D.Invalidate();
        m_ucmsLM[im] |= GBA_FLAG_CAMERA_MOTION_INVALID;
      } else {
        IMU::PreIntegrate(KF2.m_us, KF1.m_T.m_t, KF2.m_T.m_t, m_CsLM[im1], &D, &m_work, false,
                          KF1.m_us.Empty() ? NULL : &KF1.m_us.Back(), NULL, BA_ANGLE_EPSILON);
      }
    }
    m_DsLM.Erase(im);
  }
  m_Ucs.erase(m_Ucs.begin() + iKF);
  m_ds.erase(m_ds.begin() + id1, m_ds.begin() + id2);
  m_uds.erase(m_uds.begin() + id1, m_uds.begin() + id2);
  //m_Uds.erase(m_Uds.begin() + id1, m_Uds.begin() + id2);
#ifdef CFG_GROUND_TRUTH
  if (m_CsGT && im >= 0) {
    const int im1 = im - 1, im2 = im + 1;
    if (im1 >= 0 && im2 < Nm) {
      KeyFrame &KF1 = m_KFs[iKF - 1], &KF2 = m_KFs[iKF];
      IMU::Delta &D = m_DsLMGT[im2];
      if (KF2.m_us.Empty()) {
        D.Invalidate();
      } else {
        IMU::PreIntegrate(KF2.m_us, KF1.m_T.m_t, KF2.m_T.m_t, m_CsLMGT[im1], &D, &m_work, false,
                          KF1.m_us.Empty() ? NULL : &KF1.m_us.Back(), NULL, BA_ANGLE_EPSILON);
#ifdef GBA_DEBUG_GROUND_TRUTH_MEASUREMENT
        D.DebugSetMeasurement(m_CsLMGT[im1], m_CsLMGT[im], m_K.m_pu, BA_ANGLE_EPSILON);
        m_DsLM[im].DebugSetMeasurement(m_CsLMGT[im1], m_CsLMGT[im], m_K.m_pu, BA_ANGLE_EPSILON);
#endif
      }
    }
    m_DsLMGT.Erase(im);
  }
#ifdef CFG_HISTORY
  if (m_history >= 3) {
    m_ucsGT.erase(m_ucsGT.begin() + iKF);
    m_udsGT.erase(m_udsGT.begin() + id1, m_udsGT.begin() + id2);
  }
#endif
#endif
  m_Afps.Erase(iKF);
  m_SAcus.Erase(iKF);
  m_SMcus.Erase(iKF);
  if (im >= 0) {
    const int im1 = im - 1, im2 = im + 1;
    //Camera::Factor &SAcm = m_SAcmsLM[im];
    if (im1 >= 0) {
      IMU::Delta::Factor &A = m_AdsLM[im];
      A.m_A11.MakeMinus();
      m_SAcus[iKF - 1] += A.m_A11.m_Acc;
      Camera::Factor::Unitary &SAcm1 = m_SAcmsLM[im1].m_Au;
      SAcm1.m_Acm += A.m_A11.m_Acm;
      SAcm1.m_Amm += A.m_A11.m_Amm;
    }
    if (im2 < Nm) {
      IMU::Delta::Factor &A = m_AdsLM[im2];
      A.m_A22.MakeMinus();
      m_SAcus[iKF] += A.m_A22.m_Acc;
      Camera::Factor::Unitary &SAcm2 = m_SAcmsLM[im2].m_Au;
      SAcm2.m_Acm += A.m_A22.m_Acm;
      SAcm2.m_Amm += A.m_A22.m_Amm;
    }
    if (im1 >= 0 && im2 < Nm && m_DsLM[im].Valid()) {
      IMU::Delta::Factor::Auxiliary::Global U;
      Camera::Factor::Unitary &SAcm1 = m_SAcmsLM[im1].m_Au;
      Camera::Factor &SAcm2 = m_SAcmsLM[im2];
      IMU::Delta::Factor &A = m_AdsLM[im2];
      m_DsLM[im].GetFactor(BA_WEIGHT_IMU, m_CsLM[im1], m_CsLM[im], m_K.m_pu,
                           &A, &SAcm2.m_Ab, &U, BA_ANGLE_EPSILON);
      m_SAcus[iKF - 1] += A.m_A11.m_Acc;
      SAcm1.m_Acm += A.m_A11.m_Acm;
      SAcm1.m_Amm += A.m_A11.m_Amm;
      m_SAcus[iKF] += A.m_A22.m_Acc;
      SAcm2.m_Au.m_Acm += A.m_A22.m_Acm;
      SAcm2.m_Au.m_Amm += A.m_A22.m_Amm;
    } else if (im1 == -1) {
      m_SAcmsLM[im2].m_Ab.MakeZero();
    }
    m_AdsLM.Erase(im);
    m_AfmsLM.Erase(im);
    m_SAcmsLM.Erase(im);
  }
  int iZp, jZp;
  Camera::Factor::Unitary::CC Au, dAu;
  Camera::Factor::Binary::CC Ab, dAb;
  CameraPrior::Pose::Factor::Auxiliary Up;
  const int NZp1 = static_cast<int>(m_Zps.size());
  for (iZp = 0; iZp < NZp1; ++iZp) {
    CameraPrior::Pose &Zp = m_Zps[iZp];
    const std::vector<int>::iterator i = std::lower_bound(Zp.m_iKFs.begin(),
                                                          Zp.m_iKFs.end(), iKF);
    const int _i = static_cast<int>(i - Zp.m_iKFs.begin()) + 1;
    const bool del = Zp.m_iKFr == iKF || (i != Zp.m_iKFs.end() && *i == iKF);
    Zp.DeleteKeyFrame(iKF, &i);
    const int N = static_cast<int>(Zp.m_iKFs.size());
    if (!del && N != 0) {
      continue;
    }
    CameraPrior::Pose::Factor &Ap = m_Aps[iZp];
    Ap.Swap(m_ApTmp, m_bpTmp);
    const bool nonZero = Zp.Valid() && N != 0;
    if (nonZero) {
      m_ApTmp.Erase(_i);
      m_bpTmp.Erase(_i);
      m_work.Resize(Up.BindSize(N) / sizeof(float));
      Up.Bind(m_work.Data(), N);
      Zp.GetFactor(BA_WEIGHT_PRIOR_CAMERA_POSE, m_Cs, &Ap, &Up, BA_ANGLE_EPSILON);
    }
    for (int j = -1, _j = 0; j < N; ++j, ++_j) {
      const int jKF = j == -1 ? Zp.m_iKFr : Zp.m_iKFs[j];
      if (jKF != -1) {
        dAu.Set(m_ApTmp[_j][_j], m_bpTmp[_j]);
        if (nonZero) {
          Au.Set(Ap.m_A[_j][_j], Ap.m_b[_j]);
          Camera::Factor::Unitary::CC::AmB(Au, dAu, dAu);
        } else {
          dAu.MakeMinus();
        }
        m_SAcus[jKF] += dAu;
      }
      if (j == -1) {
        continue;
      }
      if (Zp.Valid()) {
        const int iKF1 = std::min(Zp.m_iKFr, jKF), iKF2 = std::max(Zp.m_iKFr, jKF);
        KeyFrame &KF2 = m_KFs[iKF2];
        m_ApTmp[0][_j].Get(dAb, Zp.m_iKFr > jKF);
        if (nonZero) {
          Ap.m_A[0][_j].Get(Ab, Zp.m_iKFr > jKF);
          Camera::Factor::Binary::CC::AmB(Ab, dAb, dAb);
        } else {
          dAb.MakeMinus();
        }
        const int ip = KF2.SearchPriorKeyFrame(iKF1);
        KF2.m_SAps[ip] += dAb;
      }
    }
    for (int i1 = 0, _i1 = 1; i1 < N; ++i1, ++_i1) {
      const int iKF1 = Zp.m_iKFs[i1];
      for (int i2 = i1 + 1, _i2 = i2 + 1; i2 < N; ++i2, ++_i2) {
        const int iKF2 = Zp.m_iKFs[i2];
        KeyFrame &KF = m_KFs[iKF2];
        const int ip = KF.SearchPriorKeyFrame(iKF1);
        if (nonZero) {
          Camera::Factor::Binary::CC::AmB(Ap.m_A[_i1][_i2], m_ApTmp[_i1][_i2], dAb);
        } else {
          m_ApTmp[_i1][_i2].GetMinus(dAb);
        }
        KF.m_SAps[ip] += dAb;
      }
    }
  }
  for (iZp = jZp = 0; iZp < NZp1; ++iZp) {
    const CameraPrior::Pose &Zp = m_Zps[iZp];
    if (Zp.Invalid() || Zp.m_iKFs.empty()) {
      continue;
    } else if (jZp < iZp) {
      m_Zps[jZp] = Zp;
      m_Aps[jZp] = m_Aps[iZp];
    }
    ++jZp;
  }
  const int NZp2 = jZp;
  m_Zps.resize(NZp2);
  m_Aps.resize(NZp2);
  if (m_ZpLM.DeleteKeyFrame(iKF)) {
    m_ApLM.MakeZero();
  }
//#ifdef CFG_DEBUG
#if 0
  if (m_debug) {
    m_xsGN.Resize(0);
    AssertConsistency(true, false);
  }
#endif
  //timer.Stop(true);
  //UT::Print("[%d] GBA::DeleteKeyFrame = %f\n", iFrm, timer.GetAverageMilliseconds());
}

void GlobalBundleAdjustor::DeleteMapPoints(const std::vector<int> &ids) {
  //Timer timer;
  //timer.Start();
//#ifdef CFG_DEBUG
#if 0
  if (m_debug) {
    m_xsGN.Resize(0);
    AssertConsistency();
    //AssertConsistency(true, false);
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
      UT_ASSERT((m_uds[id] & GBA_FLAG_TRACK_INVALID) == 0);
#endif
      if (m_uds[id] & GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO) {
        m_uds[id] = GBA_FLAG_TRACK_INVALID | GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;
        mds[id] = 2;
      } else {
        m_uds[id] = GBA_FLAG_TRACK_INVALID;
        mds[id] = 1;
      }
    }
  }
  std::vector<int> &izsDel = m_idxsTmp1;
  std::vector<ubyte> &mzs = m_marksTmp3;
  std::vector<std::vector<int> > &izsList = m_idxsListTmp;
  izsList.resize(nKFs);
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    KeyFrame &KF = m_KFs[iKF];
    const int id = m_iKF2d[iKF], Nx = static_cast<int>(KF.m_xs.size());
#ifdef CFG_DEBUG
    UT_ASSERT((mcs[iKF] != 0) == UT::VectorExistFlag<ubyte>(mds.data() + id, Nx, 3));
#endif
    if (mcs[iKF]) {
      Camera::Factor::Unitary::CC &SMcxx = m_SMcus[iKF];
      const ubyte *mxs = mds.data() + id;
      for (int ix = 0; ix < Nx; ++ix) {
        if (mxs[ix] != 1) {
          continue;
        }
        Camera::Factor::Unitary::CC &Mcxx = KF.m_Mxs2[ix].m_Mcxx;
        Mcxx.MakeMinus();
        SMcxx += Mcxx;
      }
      KF.InvalidateFeatures(mxs);
    }
    izsDel.resize(0);
    const int Nz = static_cast<int>(KF.m_zs.size());
    mzs.resize(Nz, 0);
    Camera::Factor::Unitary::CC &SAczz = m_SAcus[iKF], &SMczz = m_SMcus[iKF];
    const int NZ = static_cast<int>(KF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ) {
      const FRM::Measurement &Z = KF.m_Zs[iZ];
      if (!mcs[Z.m_iKF]) {
        continue;
      }
      Camera::Factor::Unitary::CC &SAcxx = m_SAcus[Z.m_iKF];
      Camera::Factor::Binary::CC &SAcxz = KF.m_SAcxzs[iZ], &SMcxz = KF.m_Zm.m_SMczms[Z.m_ik];
      const ubyte *mxs = mds.data() + m_iKF2d[Z.m_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
        const int ix = KF.m_zs[iz].m_ix;
        if (!mxs[ix]) {
          continue;
        }
        izsDel.push_back(iz);
        mzs[iz] = mxs[ix] == 2;
        FTR::Factor::Full::A2 &Az = KF.m_Azs2[iz];
        Az.m_Acxx.MakeMinus();
        SAcxx += Az.m_Acxx;
        Az.m_Acxz.MakeMinus();
        SAcxz += Az.m_Acxz;
        Az.m_Aczz.MakeMinus();
        SAczz += Az.m_Aczz;
        if (mzs[iz]) {
          continue;
        }
        FTR::Factor::Full::M2 &Mz = KF.m_Mzs2[iz];
        Mz.m_Mcxz.MakeMinus();
        SMcxz += Mz.m_Mcxz;
        Mz.m_Mczz.MakeMinus();
        SMczz += Mz.m_Mczz;
      }
    }
    std::vector<int> &izs = izsList[iKF];
    if (izsDel.empty()) {
      izs.resize(0);
    } else {
      KF.DeleteFeatureMeasurementsPrepare(izsDel, &izs);
      KF.DeleteFeatureMeasurements(izs);
      for (int iz = 0; iz < Nz; ++iz) {
        if (izs[iz] == -1 && mzs[iz]) {
          izs[iz] = -2;
        }
      }
    }
    //const int Nk = static_cast<int>(KF.m_iKFsMatch.size());
    const int Nk = KF.m_Zm.m_SMczms.Size();
    for (int ik = 0; ik < Nk; ++ik) {
      KF.m_Zm.DeleteFeatureMeasurementMatches(ik, KF.m_Zm.m_ik2zm[ik], KF.m_Zm.m_ik2zm[ik + 1],
                                              izsList[KF.m_iKFsMatch[ik]], izs);
    }
  }
//#ifdef CFG_DEBUG
#if 0
  if (m_debug) {
    m_xsGN.Resize(0);
    AssertConsistency();
    //AssertConsistency(true, false);
  }
#endif
  //timer.Stop(true);
  //UT::Print("GBA::DeleteMapPoints = %f\n", timer.GetAverageMilliseconds());
}

void GlobalBundleAdjustor::UpdateCameras(const std::vector<GlobalMap::InputCamera> &Cs) {
  std::vector<int>::iterator i = m_iFrms.begin();
  const int N = static_cast<int>(Cs.size()), nKFs = static_cast<int>(m_KFs.size());
  for (int j = 0; j < N; ++j) {
    const GlobalMap::InputCamera &C = Cs[j];
    i = std::lower_bound(i, m_iFrms.end(), C.m_iFrm);
    if (i == m_iFrms.end()) {
      break;
    } else if (*i != C.m_iFrm) {
      continue;
    }
    const int iKF = static_cast<int>(i - m_iFrms.begin());
    m_Cs[iKF] = C.m_Cam_pose;
    m_ucs[iKF] |= GBA_FLAG_FRAME_UPDATE_CAMERA;
    m_Ucs[iKF] |= GM_FLAG_FRAME_UPDATE_CAMERA;
    const int im = iKF - (nKFs - m_CsLM.Size());
    if (im < 0) {
      continue;
    }
    Camera &_C = m_CsLM[im];
    _C.m_Cam_pose = C.m_Cam_pose;
    _C.m_Cam_pose.GetPosition(_C.m_p);
    m_ucmsLM[im] |= GBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION |
                    GBA_FLAG_CAMERA_MOTION_UPDATE_POSITION;
  }
}
//m_Zps储存所有的运动先验,每个运动先验里是g和观测关键帧以及参考关键帧的先验约束,当Zp.m_iKFs.size()为时则只有g和参考关键帧的先验,m_Aps里存储所有先验的所有关键帧
//step1:将当前新增的运动约束加进m_Zps中,同时m_Aps对应的位置也要扩容成关键帧的数量,将上一次滑窗的参考关键帧设成需要更新旋转,剩下的观测关键帧都设成需要更新pose
//step2:上次滑窗的参考关键帧,观测关键帧,最老帧所对应的关键帧,这之中的每个关键帧都将他之前的这里面的关键帧push进自己的先验约束中,并构建索引
void GlobalBundleAdjustor::PushCameraPriorPose(const CameraPrior::Pose &Zp) {
  const int iZp = static_cast<int>(m_Zps.size());


  m_Zps.push_back(Zp);
  const int N = static_cast<int>(Zp.m_iKFs.size());//这次运动先验所涉及的观测关键帧(如果最老帧是关键帧时merge给的位姿约束的话,就还有最老帧的KF的id)的个数
  m_Aps.resize(iZp + 1);
  //m_Aps.push_back(CameraPrior::Pose::Factor());
  m_Aps[iZp].MakeZero(N);//
  m_ucs[Zp.m_iKFr] |= GBA_FLAG_FRAME_UPDATE_CAMERA;//将参考关键帧设成需要更新
  const int nKFs = static_cast<int>(m_KFs.size()), imr = Zp.m_iKFr - (nKFs - m_CsLM.Size());
  if (imr >= 0) {
    m_ucmsLM[imr] |= GBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION;//参考关键帧只更新旋转(因为在滑窗存储参考关键帧pose时,只考虑前后两次滑窗的so3误差)
  }
  const ubyte ucmFlag = GBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION |
                        GBA_FLAG_CAMERA_MOTION_UPDATE_POSITION;
  for (int i = 0; i < N; ++i) {//遍历所有具有位姿先验的关键帧id,均设成需要更新位姿
    const int _iKF = Zp.m_iKFs[i];
    m_ucs[_iKF] |= GBA_FLAG_FRAME_UPDATE_CAMERA;
    const int im = _iKF - (nKFs - m_CsLM.Size());
    if (im >= 0) {
      m_ucmsLM[im] |= ucmFlag;
    }
  }

  for (int i1 = -1; i1 < N; ++i1) {
    const int iKF1 = i1 == -1 ? Zp.m_iKFr : Zp.m_iKFs[i1];//iKF1初始是前一次滑窗的参考关键帧id
    for (int i2 = i1 + 1; i2 < N; ++i2) {//遍历所有具有位姿先验的关键帧
      const int iKF2 = Zp.m_iKFs[i2];//观测参考帧,最后一个是最老帧滑窗对应的关键帧id
      const int _iKF1 = std::min(iKF1, iKF2), _iKF2 = std::max(iKF1, iKF2);
      KeyFrame &KF = m_KFs[_iKF2];//靠后的这个关键帧
      const int Nkp = static_cast<int>(KF.m_ikp2KF.size());
      const int ip = KF.PushCameraPrior(_iKF1, &m_work);//向靠后的这个关键帧放进靠前的这个关键帧的先验
      if (ip == -1 || static_cast<int>(KF.m_ikp2KF.size()) == Nkp) {//static_cast<int>(KF.m_ikp2KF.size()) == Nkp应该是恒定满足的吧
        continue;
      }
      for (int jKF = _iKF2; jKF < nKFs; ++jKF) {
        ++m_iKF2cb[jKF + 1];
      }
    }
  }
}
//设置一下先验运动约束
void GlobalBundleAdjustor::SetCameraPriorMotion(const CameraPriorMotion &Zp) {
  const int nKFs = static_cast<int>(m_KFs.size());
  if (m_ZpLM.Valid()) {//将老的先验运动约束减去
    m_ApLM.MakeMinus();//上一次的motion的先验约束减去
    m_SAcus[m_ZpLM.m_iKF].Increase3(m_ApLM.m_Arr.m_A, m_ApLM.m_Arr.m_b);//R x R 对应的H|-b
    const int im = m_ZpLM.m_iKF - (nKFs - m_CsLM.Size());
    Camera::Factor::Unitary &SAcm = m_SAcmsLM[im].m_Au;//当前这个关键帧的Hcm以及Hmm(对角线上跟运动相关)和-mb部分
    SAcm.m_Acm.Increase3(m_ApLM.m_Arm);
    SAcm.m_Amm += m_ApLM.m_Amm;
  }
  m_ZpLM = Zp;//替换成新的运动约束,里面有LBA对于这个关键帧的观测
  const int im = m_ZpLM.m_iKF - (nKFs - m_CsLM.Size());
  m_ucmsLM[im] |= GBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY |//设置成需要更新motion
                  GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_ACCELERATION |
                  GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_GYROSCOPE;
  m_ApLM.MakeZero();
}
//处理一下共视之间的关系
void GlobalBundleAdjustor::PushFeatureMeasurementMatchesFirst(const FRM::Frame &F,/*滑窗中的当前帧*/
                                                              std::vector<int> &iKF2X,/*存储的是当前帧和关键帧的共视关系*/
                                                              std::vector<int> &iX2z) {
    int SNx = 0;
    const int NZ = static_cast<int>(F.m_Zs.size());//共视关键帧的个数
    iKF2X.assign(m_KFs.size(), -1);
    for (int iZ = 0; iZ < NZ; ++iZ) {
        const int iKF = F.m_Zs[iZ].m_iKF;//当前的共视关键帧id
        iKF2X[iKF] = SNx;
        SNx += static_cast<int>(m_KFs[iKF].m_xs.size());//第i个关键帧之前的共视帧有多少个地图点
    }
    iX2z.assign(SNx, -1);
    for (int iZ = 0; iZ < NZ; ++iZ) {
        const FRM::Measurement &Z = F.m_Zs[iZ];
        int *ix2z = iX2z.data() + iKF2X[Z.m_iKF];//跳转到这个关键帧的首个地图点
        for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {//如果这个地图点被观测到,就记录这个地图点在m_zs存储的起始位置
            ix2z[F.m_zs[iz].m_ix] = iz;
        }
    }
}

//将F1和F2之间的共视信息进行关联,都观测到了同一个点就存储下来,m_iz1是F1对这个地图点的观测索引,m_iz2是F1对这个地图点的观测索引
//将观测的匹配信息加入Zm中,并进行相关H矩阵的扩容
void GlobalBundleAdjustor::PushFeatureMeasurementMatchesNext(const FRM::Frame &F1,/*共视关键帧*/
                                                            const FRM::Frame &F2,/*当前关键帧*/
                                                            const std::vector<int> &iKF2X,/*[i]表示第i个关键帧之前的F2帧观测了多少个地图点*/
                                                            const std::vector<int> &iX2z2,/*共视帧的地图点被观测到,就记录这个地图点在m_zs存储的起始位置*/
                                                             FRM::MeasurementMatch &Zm) {
  m_izmsTmp.resize(0);
  const int NZ1 = static_cast<int>(F1.m_Zs.size());//共视关键帧的共视关键帧数量
  for (int iZ1 = 0; iZ1 < NZ1; ++iZ1) {//遍历共视关键帧对于关键帧M的观测
    const FRM::Measurement &Z1 = F1.m_Zs[iZ1];//共视关键帧对某一个关键帧的观测
    const int iX = iKF2X[Z1.m_iKF];
    if (iX == -1) {//如果F2没有观测到这个关键帧M就跳过
      continue;
    }
    const int *ix2z2 = iX2z2.data() + iX;
    const int iz11 = Z1.m_iz1, iz12 = Z1.m_iz2;//F1对于这个M中地图点观测的索引范围
    for (int iz1 = iz11; iz1 < iz12; ++iz1) {
      const int iz2 = ix2z2[F1.m_zs[iz1].m_ix];
      if (iz2 != -1) {//如果F2没有观测到这个点就跳过
        m_izmsTmp.push_back(FTR::Measurement::Match(iz1, iz2));//都观测到了同一个点就存储下来,m_iz1是F1对这个地图
        // 点的观测索引,m_iz2是F1对这个地图点的观测索引
      }
    }
  }
  Zm.PushFeatureMeasurementMatches(m_izmsTmp);//将观测信息加入Zm中,并进行相关H矩阵的扩容
}

int GlobalBundleAdjustor::CountMeasurementsFrame() {
  int SN = 0;
  const int nKFs = int(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    SN += static_cast<int>(m_KFs[iKF].m_Zs.size());
  }
  return SN;
}

int GlobalBundleAdjustor::CountMeasurementsFeature() {
  int SN = 0;
  const int nKFs = int(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    SN += static_cast<int>(m_KFs[iKF].m_zs.size());
  }
  return SN;
}

int GlobalBundleAdjustor::CountMeasurementsPriorCameraPose() {
  int SN = 0;
#if 0
  const int nKFs = int(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF)
    SN += m_KFs[iKF].m_Zps.Size();
#endif
  return SN;
}

int GlobalBundleAdjustor::CountSchurComplements() {
  return m_SAcus.Size() + CountSchurComplementsOffDiagonal();
}

int GlobalBundleAdjustor::CountSchurComplementsOffDiagonal() {
  return m_iKF2cb.back();
}
