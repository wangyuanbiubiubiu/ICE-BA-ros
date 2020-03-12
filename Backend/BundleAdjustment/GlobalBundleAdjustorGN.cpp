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
#ifdef CFG_DEBUG_EIGEN
//#define GBA_DEBUG_EIGEN_PCG
#endif
#include "GlobalBundleAdjustor.h"
#include <math.h>  // isfinite

#if defined WIN32 && defined CFG_DEBUG && defined CFG_GROUND_TRUTH
//#define GBA_DEBUG_GROUND_TRUTH_STATE
//#ifdef GBA_DEBUG_GROUND_TRUTH_STATE
//#define GBA_DEBUG_GROUND_TRUTH_STATE_ERROR
//#endif
#endif

#if defined CFG_DEBUG && defined CFG_VERBOSE
//#define GBA_DEBUG_PCG_SAVE_RESIDUAL
//#define GBA_DEBUG_PCG_SAVE_RESULT
//#define GBA_DEBUG_PCG_LOAD_RESULT
#endif

int GBAdebug_count = -1;
std::string gba_debug_file = "/home/wya/ICE-BA-Debug/ba/gba.txt";
void GlobalBundleAdjustor::UpdateFactors()
{

    GBAdebug_count++;
#ifdef CFG_VERBOSE
  if (m_verbose >= 3) {
    UT::PrintSeparator();
    UT::Print("*%2d: [GlobalBundleAdjustor::UpdateFactors]\n", m_iIter);
  }
#endif
  const float add = UT::Inverse(BA_VARIANCE_REGULARIZATION_DEPTH, BA_WEIGHT_FEATURE);
  const int nKFs = static_cast<int>(m_KFs.size());//GBA中的关键帧数量
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    KeyFrame &KF = m_KFs[iKF];//当前关键帧


    if (m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_CAMERA) {//是否需要更新位姿?
      KF.m_SAcxzs.MakeZero();
      KF.m_SAps.MakeZero();
      m_SAcus[iKF].MakeZero();
    } else {//如果不需要更新相机的话
      const int NZ = int(KF.m_Zs.size());
      for (int iZ = 0; iZ < NZ; ++iZ) {
        if (m_ucs[KF.m_Zs[iZ].m_iKF] & GBA_FLAG_FRAME_UPDATE_CAMERA)//如果观测到的关键帧需要更新
          KF.m_SAcxzs[iZ].MakeZero();
      }
      const int Np = KF.m_SAps.Size();
      for (int ip = 0; ip < Np; ++ip) {
        if (m_ucs[KF.m_iKFsPrior[ip]] & GBA_FLAG_FRAME_UPDATE_CAMERA) {
          KF.m_SAps[ip].MakeZero();
        }
      }
    }
    if (m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_DEPTH) {//是否需要更新深度,只对新的观测进行矩阵的操作
      const ubyte *uds = m_uds.data() + m_iKF2d[iKF];//从新的地图点
      const int Nx = static_cast<int>(KF.m_xs.size());//新地图点的size()
      for (int ix = 0; ix < Nx; ++ix) {//
        if (uds[ix] & GBA_FLAG_TRACK_UPDATE_DEPTH) {//如果需要更新深度
          KF.m_Axs[ix].MakeZero();
          KF.m_Axs[ix].m_Sadx.m_add.m_a = add;
        }
      }
    }
  }

  const ubyte ucmFlag = GBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION |//设定需要需要更新什么
                        GBA_FLAG_CAMERA_MOTION_UPDATE_POSITION |
                        GBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY |
                        GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_ACCELERATION |
                        GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_GYROSCOPE;
  const int Nm = m_CsLM.Size();
  for (int im = 0; im < Nm; ++im) {
    if (m_ucmsLM[im] & ucmFlag) {//如果需要更新状态
      m_SAcmsLM[im].MakeZero();
    }
  }
//#ifdef CFG_DEBUG
#if 0
  //const int iKF = nKFs - 1;
  const int iKF = 45;
  //m_Cs[iKF].Print(true);
  const Camera::Factor::Unitary::CC &A = m_SAcus[iKF];
#endif
  UpdateFactorsFeature();//更新视觉约束
//#ifdef CFG_DEBUG
#if 0
  //UT::Print("%.10e\n", A.m_A.m00());
  UT::Print("%.10e\n", A.m_b.v0());
#endif
  //用来对所有路标点的逆深度进行约束,对于只有左目观测令每个路标点的逆深度在该 source KF 所管辖的所有路标点的平均逆深度附近
  //双目都观测到的话,就用外参来约束
  UpdateFactorsPriorDepth();//地图点逆深度的先验
  //m_ts[TM_TEST_1].Start();
//更新LBA给GBA中各个关键帧之间的运动先验约束
  UpdateFactorsPriorCameraPose();

if(!IMU_GRAVITY_EXCLUDED)
{
//m_ts[TM_TEST_1].Stop();
//更新运动先验,和滑窗中运动之间的残差,也就是GBA和LBA要保持一致,LBA边缘化最老帧是关键帧时会给GBA这帧的先验约束,同时m_ZpLM存储的是LBA中的
//这个最老帧的(Rc0w*vw，ba,bj)作为测量量
    UpdateFactorsPriorCameraMotion();
    //更新imu约束所带来的因子
//每个imu约束的因子,都连接前后两个关键帧的pose和motion
//// 残差:
// e_r = -ln{预积分的Rij * exp[Jrbw *(bwi - z_bw)]x * Rcjw * Rciw.t}v
//  e_v = Rciw*(v_wj - v_wi + 0.5gt^2) - (m_v + m_Jvba * (bai - m_ba) + m_Jvbw * (bwi - m_bw))
//  e_p = Rciw*(p_wcj - p_wci - v_wci*dt + 0.5*g*dt^2) +
//  Rciw * Rcjw.t*tc0i - tc0i - (m_p + m_Jpba * (bai - m_ba) + m_Jpbw * (bwi - m_bw))
// e_ba = bai - baj
// e_bw = bwi - bwj
    UpdateFactorsIMU();//imu约束
}

  UpdateFactorsFixOrigin();//固定第一帧pose
  UpdateFactorsFixPositionZ();//这个没用上
  UpdateFactorsFixMotion();//这个也没用上啊,不过如果这个因子没有别的观测,那么倒是能用上
//#ifdef CFG_DEBUG
#if 0
  //UT::Print("%.10e\n", A.m_A.m00());
  UT::Print("%.10e\n", A.m_b.v0());
#endif
}

//更新新的观测带来的视觉约束,以及更新老的视觉约束(针对地图点的重投影误差),具体看FTR::GetFactor里的注释
//每个视觉约束,即重投影残差的因子,都连接投影前的观测关键帧pose,投影后的关键帧pose还有地图点的逆深度(就是在首次观测到它的这帧中的逆深度),
//对于H的上三角,有6处影响(投影前poseX投影前pose，投影前poseX投影后pose,逆深度X逆深度,
// 投影前poseX投影后pose,投影前poseX逆深度,投影后poseX逆深度)
// b有3处(投影前pose,投影后pose,逆深度)
// GBA中全部计算相关的雅克比
//如果是一个新的因子,那么在矩阵块里+=既可以了,如果是一个已有的因子需要更新,那么就+=(新因子 - 老因子)就可以了
//逆深度x逆深度的H|-b,投影前poseX逆深度的H更新在_KF.m_Axs[ix].m_Sadx _KF是观测关键帧即投影前的,ix就是对应的地图点的局部id
//投影后pose x 逆深度的H存储在KF.m_Azs1[iz]
//投影前pose x 投影前pose的H|-b更新在m_SAcus[_iKF].m_Acxx (m_Acxx是专门维护这个关键帧作为观测关键帧时的约束,m_Aczz是专门维护这个关键帧作为投影后关键帧时的约束)
//投影后pose x 投影后pose的H|-b更新在m_SAcus[iKF].m_Aczz
//投影前pose x 投影后pose的H更新在KF.m_SAcxzs[iZ] 即存储于投影后关键帧自己的数据结构里
void GlobalBundleAdjustor::UpdateFactorsFeature()
{

    // write result to file
//    std::ofstream foutC(gba_debug_file, std::ios::app);
//    foutC.setf(std::ios::fixed, std::ios::floatfield);
//    foutC << "UpdateFactorsFeature 第几次优化:"<<GBAdebug_count << "\n";

    //#ifdef CFG_DEBUG
#if 0
  UT::DebugStart();
  UT::DebugStop();
  return;
#endif
#ifdef CFG_VERBOSE
  int SNz = 0, SNZ = 0;
#endif
  Rigid3D Tr[2];
  FTR::Factor::DDC dadx;
  Camera::Factor::Unitary::CC dAcxx, dAczz;
  Camera::Factor::Binary::CC dAcxz;
  FTR::Factor::Full::U U;/*这个地图点从观测关键帧投影到投影后关键帧上的重投影误差的因子,存了H|-b,信息矩阵*/
  //float dF;
  const int nKFs = int(m_KFs.size());//关键帧的帧数
  for (int iKF = 0; iKF < nKFs; ++iKF) {//遍历所有关键帧
    KeyFrame &KF = m_KFs[iKF];//当前关键帧
    const Rigid3D &C = m_Cs[iKF];//当前关键帧对应的pose
    const bool ucz = (m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_CAMERA) != 0;//是否要各更新相机
    Camera::Factor::Unitary::CC &SAczz = m_SAcus[iKF];//这个关键帧自己的pose x pose对应的H|-b
    const int NZ = int(KF.m_Zs.size());//这个关键帧所有的观测
    for (int iZ = 0; iZ < NZ; ++iZ) {//遍历这个关键帧所有的观测关键帧
      const FRM::Measurement &Z = KF.m_Zs[iZ];
      const int _iKF = Z.m_iKF;//关键关键帧id
#ifdef CFG_DEBUG
      UT_ASSERT(_iKF < iKF);
#endif
      const bool ucx = (m_ucs[_iKF] & GBA_FLAG_FRAME_UPDATE_CAMERA) != 0, ucr = ucx || ucz;//投影帧和投影后的帧有需要更新的就为true
      if (!ucr && !(m_ucs[_iKF] & GBA_FLAG_FRAME_UPDATE_DEPTH)) {//这个关键帧里地图点是否有要更新的
        continue;
      }
      *Tr = C / m_Cs[_iKF];//Tc0(投影关键帧)_c0(观测关键帧)
#ifdef CFG_STEREO
      Tr[1] = Tr[0];
      Tr[1].SetTranslation(m_K.m_br + Tr[0].GetTranslation());//Tc1(投影关键帧)_c0(观测关键帧)
#endif
      const int id = m_iKF2d[_iKF];
      const Depth::InverseGaussian *_ds = m_ds.data() + id;
      ubyte *_uds = m_uds.data() + id;
      Camera::Factor::Unitary::CC &SAcxx = m_SAcus[_iKF];/*投影关键帧自己的pose x pose对应的H|-b*/
      Camera::Factor::Binary::CC &SAcxz = KF.m_SAcxzs[iZ];//这个投影前(观测关键帧)pose x 投影后pose的H
      KeyFrame &_KF = m_KFs[_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {//遍历每一个观测到的地图点,算一下视觉因子的H|b
        const FTR::Measurement &z = KF.m_zs[iz];
        const int ix = z.m_ix;
        const bool ud = (_uds[ix] & GBA_FLAG_TRACK_UPDATE_DEPTH) != 0;
        if (!ucr && !ud) {
          continue;
        }
        FTR::Factor::Full::A2 &A = KF.m_Azs2[iz];//m_adx.m_add是逆深度x逆深度的H|-b,m_adx.m_adc是投影前posex逆深度的H,m_Acxx是投影前pose x 投影前pose的H|-b,
          //m_Aczz是投影后pose x 投影后pose的H|-b,m_Acxz是投影前pose x 投影后pose的H
        //因子已经存在的话,就只需要更新,所以需要保存一下老的因子
          if (!ud) {
          dadx = A.m_adx;
        }
        if (!ucx) {
          dAcxx = A.m_Acxx;
        }
        if (!ucr) {
          dAcxz = A.m_Acxz;
        }
        if (!ucz) {
          dAczz = A.m_Aczz;
        }
          // Pczk = Tc0w(投影后关键帧) * Tcw.inv(投影前观测关键帧) *Pcxk (z是投影后关键帧,x是投影前关键帧)
        FTR::GetFactor<GBA_ME_FUNCTION>(BA_WEIGHT_FEATURE, Tr/*Tc(投影后关键帧)_c0(观测关键帧)*/, _KF.m_xs[ix]/*投影关键帧对这个地图点的观测*/,
                _ds[ix]/*逆深度*/, C/*投影后关键帧Tc0w*/, z,
                                        &KF.m_Lzs[iz],/*投影后关键帧中对这个地图点观测的重投影误差e,J(对投影前后的pose,对关键帧点的逆深度),cost*/
                                        &KF.m_Azs1[iz], //m_adcz存储H的投影后posex逆深度部分
                                        &A,//m_adx.m_add是逆深度x逆深度的H|-b,m_adx.m_adc是投影前pose x 逆深度的H,m_Acxx是投影前pose x 投影前pose的H|-b,
                                       //m_Aczz是投影后pose x 投影后pose的H|-b,m_Acxz是投影前pose x 投影后pose的H
                                        &U/*这个地图点从观测关键帧投影到投影后关键帧上的重投影误差的因子,存了H|-b,信息矩阵*/
#ifdef CFG_STEREO
                                      , m_K.m_br
#endif
                                      );
//#ifdef CFG_DEBUG
#if 0
        if (iKF == 87 && iz == 2) {
          UT::Print("%f\n", A.m_Aczz.m_A.m00());
        }
#endif
//
//          foutC.precision(0);
//          foutC <<"遍历kf:"<<iKF<<",观测到kf:"<<iZ<<",点id:"<< ix<<",";
//          foutC.precision(5);
//          foutC << A.m_adx.m_add.m_a << ","<< A.m_adx.m_add.m_b << std::endl;


        if (ud) {//如果是新的因子.直接+= ,否则更新增量
          _KF.m_Axs[ix].m_Sadx += A.m_adx;//逆深度x逆深度的H|-b,投影前pose x 逆深度的H
        } else {
          FTR::Factor::DDC::amb(A.m_adx, dadx, dadx);
          _KF.m_Axs[ix].m_Sadx += dadx;
        }
        if (ucx) {//投影前pose x 投影前pose的H|-b
          SAcxx += A.m_Acxx;
        } else {
          Camera::Factor::Unitary::CC::AmB(A.m_Acxx, dAcxx, dAcxx);
          SAcxx += dAcxx;
        }
        if (ucr) {
          SAcxz += A.m_Acxz;//投影前pose x 投影后pose的H
        } else {
          Camera::Factor::Binary::CC::AmB(A.m_Acxz, dAcxz, dAcxz);
          SAcxz += dAcxz;
        }
        if (ucz) {
          SAczz += A.m_Aczz;//投影后pose x 投影后pose的H|-b
        } else {
          Camera::Factor::Unitary::CC::AmB(A.m_Aczz, dAczz, dAczz);
          SAczz += dAczz;
        }
        //dF = A.m_F - dF;
        //m_F = dF + m_F;
        m_ucs[_iKF] |= GBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION;//将对应的因子设为已更新(已存在的flags)
        _uds[ix] |= GBA_FLAG_TRACK_UPDATE_INFORMATION;
#ifdef CFG_VERBOSE
        if (m_verbose >= 3) {
          ++SNz;
        }
#endif
      }
#ifdef CFG_VERBOSE
      if (m_verbose >= 3) {
        ++SNZ;
      }
#endif
    }
  }
#ifdef CFG_VERBOSE
  if (m_verbose >= 3) {
    const int Nz = CountMeasurementsFeature(), NZ = CountMeasurementsFrame();
    UT::Print("  Feature = %d / %d = %.2f%% (%d / %d = %.2f%%)\n", SNz, Nz, UT::Percentage(SNz, Nz),
              SNZ, NZ, UT::Percentage(SNZ, NZ));
  }
#endif

//    foutC.close();
}
//更新LBA给GBA中各个关键帧之间的运动先验约束
void GlobalBundleAdjustor::UpdateFactorsPriorCameraPose() {
#ifdef CFG_VERBOSE
  int SN = 0;
#endif
  Camera::Factor::Unitary::CC Au, dAu;
  Camera::Factor::Binary::CC Ab, dAb;
  CameraPrior::Pose::Factor::Auxiliary U;
  //float dF;
  const int NZ = static_cast<int>(m_Zps.size());//所有的位姿先验
//#ifdef CFG_DEBUG
#if 0
//#if 1
  FILE *fp = fopen("D:/tmp/Zp.txt", "w");
  for (int iZ = 0; iZ < NZ; ++iZ) {
    const CameraPrior::Pose &Z = m_Zps[iZ];
    fprintf(fp, "%d: ", Z.m_iKFr);
    const int N = static_cast<int>(Z.m_iKFs.size());
    for (int i = 0; i < N; ++i) {
      fprintf(fp, " %d", Z.m_iKFs[i]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  exit(0);
#endif

    //遍历所有的先验
    //计算先验约束对应的H|-b
    //step1:处理位姿先验带给参考关键帧,观测关键帧以及最老帧对应的关键帧自己posex自己pose的影响H|-b,存在每一帧自己的m_SAcus[iKF]中
    //step2:处理位姿先验带给H参考关键帧pose x 观测关键帧、最老帧之间的影响,存在两帧之中较新帧的m_SAps[ip],ip是较老帧在较新帧中的m_ikp2KF里的索引
    //step3:处理位姿先验带给H观测关键帧、最老帧之间 pose x 观测关键帧、最老帧之间的影响,存在两帧之中较新帧的m_SAps[ip],ip是较老帧在较新帧中的m_ikp2KF里的索引
  for (int iZ = 0; iZ < NZ; ++iZ) {//遍历所有的位姿先验
    const CameraPrior::Pose &Z = m_Zps[iZ];
    const bool ucr = (m_ucs[Z.m_iKFr] & GBA_FLAG_FRAME_UPDATE_CAMERA) != 0;
    bool uc = ucr;
    const int N = static_cast<int>(Z.m_iKFs.size());//这个位姿先验里的观测关键帧
    for (int i = 0; i < N && !uc; ++i) {
      uc = (m_ucs[Z.m_iKFs[i]] & GBA_FLAG_FRAME_UPDATE_CAMERA) != 0;//是否需要更新pose
    }
    if (!uc) {
      continue;
    }
#ifdef CFG_VERBOSE
    if (m_verbose >= 3) {
      ++SN;
    }
#endif
    CameraPrior::Pose::Factor &A = m_Aps[iZ];//对应的这个先验约束
    A.Swap(m_ApTmp, m_bpTmp);//暂时保存一下A中的约束
    m_work.Resize(U.BindSize(N) / sizeof(float));
    U.Bind(m_work.Data(), N);
    if (GBA_PRIOR_CAMERA_POSE_ROBUST)
    {
        //er为参考关键帧自身约束,m_ep，m_er为参考关键帧和观测到的关键帧之间的相对约束
        //er = -ln(eR) = -ln(Rwc0(观测) * Rc0w).xy  eR是wc0(观测) * Rc0w
        //m_ep = Rc0w(参考关键帧) * (twc0(观测到的关键帧) - twc0(参考关键帧)) - tc0(参考关键帧)c0(观测到的关键帧)(测量)
        //m_er = -ln(Rc0(参考关键帧)c0(观测到的关键帧)(测量) * Rc0w(观测到的关键帧) * Rc0w.t(参考关键帧))v
        Z.GetError(m_Cs/*GBA中所有关键帧的位姿*/, &A.m_Je.m_e, BA_ANGLE_EPSILON);
        float F;
        float w;
        if(!Z.isloop)
        {
            F = (Z.GetCost(1.0f, A.m_Je.m_e) - Z.m_xTb) / BA_WEIGHT_FEATURE;
            w = BA_WEIGHT_PRIOR_CAMERA_POSE * ME::Weight<GBA_ME_FUNCTION>(F);
        }
        else
        {
//            std::cout<<A.m_Je.m_e.m_ec[0].m_ep.x()<<" "<<A.m_Je.m_e.m_ec[0].m_ep.y()<<" "<<A.m_Je.m_e.m_ec[0].m_ep.z()<<std::endl;
            w = LOOP_BA_WEIGHT_PRIOR_CAMERA_POSE;//针对闭环的情况
        }

        Z.GetFactor(w, m_Cs/*GBA中所有关键帧的位姿*/, &A, &U, BA_ANGLE_EPSILON);//不过这里H|-b的位置我不太清楚,没对上,就是这个g没懂怎么转换的
    } else {
      Z.GetFactor(BA_WEIGHT_PRIOR_CAMERA_POSE, m_Cs, &A, &U, BA_ANGLE_EPSILON);
    }
#if 0
    m_ts[TM_TEST_2].Start();
    Z.GetErrorJacobian(m_Cs, &A.m_Je);
    m_ts[TM_TEST_2].Stop();
    m_ts[TM_TEST_3].Start();
    const xp128f _w = xp128f::get(BA_WEIGHT_PRIOR_CAMERA_POSE);
    U.Set(A.m_Je, _w, Z.m_Arr, Z.m_Arc, Z.m_Acc);
    m_ts[TM_TEST_3].Stop();
    m_ts[TM_TEST_4].Start();
    A.m_F = Z.GetCost(BA_WEIGHT_PRIOR_CAMERA_POSE, A.m_Je.m_e, U.m_Aer, U.m_Aec);
    m_ts[TM_TEST_4].Stop();
    m_ts[TM_TEST_5].Start();
    U.Get(A.m_Je.m_J.m_Jr, _w, Z.m_br, Z.m_bc, &A.m_A, &A.m_b);
    m_ts[TM_TEST_5].Stop();
#endif


    for (int i = -1, _i = 0; i < N; ++i, ++_i)
    {
        //step1:处理位姿先验带给参考关键帧,观测关键帧以及最老帧对应的关键帧自己posex自己pose的影响H|-b,存在每一帧自己的m_SAcus[iKF]中
        const int iKF = i == -1 ? Z.m_iKFr : Z.m_iKFs[i];//第一次是参考关键帧,后面都是观测关键帧
      Au.Set(A.m_A[_i][_i], A.m_b[_i]);
      uc = (m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_CAMERA) != 0;
      if (uc)//参考关键帧X参考关键帧或者观测关键帧x观测关键帧
      {
        m_SAcus[iKF] += Au;
      } else
      {
        dAu.Set(m_ApTmp[_i][_i], m_bpTmp[_i]);
        Camera::Factor::Unitary::CC::AmB(Au, dAu, dAu);
        m_SAcus[iKF] += dAu;
      }
      if (i == -1)
      {
        continue;
      }
        //step2:处理位姿先验带给H参考关键帧pose x 观测关键帧、最老帧之间的影响,存在两帧之中较新帧的m_SAps[ip],ip是较老帧在较新帧中的m_ikp2KF里的索引
      A.m_A[0][_i].Get(Ab, Z.m_iKFr > iKF);
      const int iKF1 = std::min(Z.m_iKFr, iKF), iKF2 = std::max(Z.m_iKFr, iKF);//iKF1为老关键帧,iKF2为新关键帧
      KeyFrame &KF2 = m_KFs[iKF2];
      const int ip = KF2.SearchPriorKeyFrame(iKF1);//在新的关键帧中找到对应的旧的关键帧先验的索引
      if (ucr || uc)
      {
        KF2.m_SAps[ip] += Ab;//H老帧posex新帧pose
      } else
      {
        m_ApTmp[0][_i].Get(dAb, Z.m_iKFr > iKF);
        Camera::Factor::Binary::CC::AmB(Ab, dAb, dAb);
        KF2.m_SAps[ip] += dAb;
      }
    }
      //step3:处理位姿先验带给H观测关键帧、最老帧之间 pose x 观测关键帧、最老帧之间的影响,存在两帧之中较新帧的m_SAps[ip],ip是较老帧在较新帧中的m_ikp2KF里的索引
    for (int i1 = 0, _i1 = 1; i1 < N; ++i1, ++_i1)//这里跳过了参考关键帧对应的位置
    {
      const int iKF1 = Z.m_iKFs[i1];//关键关键帧+最老帧对应的关键帧
      const bool uc1 = (m_ucs[iKF1] & GBA_FLAG_FRAME_UPDATE_CAMERA) != 0;
      for (int i2 = i1 + 1, _i2 = i2 + 1; i2 < N; ++i2, ++_i2)
      {
        const int iKF2 = Z.m_iKFs[i2];
#ifdef CFG_DEBUG
        UT_ASSERT(iKF2 > iKF1);
#endif
        const bool uc2 = (m_ucs[iKF2] & GBA_FLAG_FRAME_UPDATE_CAMERA) != 0;
        const CameraPrior::Element::CC &_Ab = A.m_A[_i1][_i2];
        KeyFrame &KF2 = m_KFs[iKF2];
        const int ip = KF2.SearchPriorKeyFrame(iKF1);
        if (uc1 || uc2)
        {
          KF2.m_SAps[ip] += _Ab;
        } else
        {
          Camera::Factor::Binary::CC::AmB(_Ab, m_ApTmp[_i1][_i2], dAb);
          KF2.m_SAps[ip] += dAb;
        }
      }
    }
  }
#ifdef CFG_VERBOSE
  if (m_verbose >= 3) {
    const int N = CountMeasurementsPriorCameraPose();
    UT::Print("  Prior Camera = %d / %d = %.2f%%\n", SN, N, UT::Percentage(SN, N));
  }
#endif
}
//更新运动先验,和滑窗中运动之间的残差,也就是GBA和LBA要保持一致,LBA边缘化最老帧是关键帧时会给GBA这帧的先验约束,同时m_ZpLM存储的是LBA中的
//这个最老帧的(Rc0w*vw，ba,bj)作为测量量
//R x R 对应的H|-b 更新在m_SAcus[m_ZpLM.m_iKF]即在滑窗中被边缘化了的滑窗帧对应的关键帧
//R x motion H 以及 motion x motion的H|-b更新在m_SAcmsLM[im].m_Au,im == m_ZpLM.m_iKF
void GlobalBundleAdjustor::UpdateFactorsPriorCameraMotion() {
  if (m_ZpLM.Invalid()) {//在solver.Start()里已经初始化过了
    return;
  }
  const int im = m_ZpLM.m_iKF - m_Cs.Size() + m_CsLM.Size();//这个应该就是m_ZpLM.m_iKF，即滑窗现在最新的给的运动先验约束是哪个关键帧
  const ubyte ucm = m_ucmsLM[im];//是否更新,一定会更新,因为滑窗给GBA时设置了
  const bool uc = (ucm & (GBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION |
                          GBA_FLAG_CAMERA_MOTION_UPDATE_POSITION)) != 0;
  const bool _ucm = uc || (ucm & (GBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY |//motion部分也是全部要更新的
                                  GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_ACCELERATION |
                                  GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_GYROSCOPE)) != 0;
  if (!_ucm) {
    return;
  }
  CameraPrior::Motion::Factor::RR dArr;
  CameraPrior::Motion::Factor::RM dArm;
  CameraPrior::Motion::Factor::MM dAmm;
  if (!uc) {
    dArr = m_ApLM.m_Arr;
  }
  if (!ucm) {
    dArm = m_ApLM.m_Arm;
    dAmm = m_ApLM.m_Amm;
  }
  CameraPrior::Motion::Factor::Auxiliary U;//设置一下motion先验部分因子的一个H和b部分
//和滑窗中运动之间的残差,也就是GBA和LBA要保持一致,LBA边缘化最老帧是关键帧时会给GBA这帧的先验约束,同时m_ZpLM存储的是LBA中的
// 这个最老帧的(Rc0w*vw，ba,bj)作为测量量
////ev =  Rc0w*vw - Rc0w*vw(测量)
//Rc0w右扰动 Rc0w * exp[-th]x*vw
//div(ev)/div(Rc0w) = Rc0w * [vw]x
//div(ev)/div(vw) = Rc0w;
////eba = ba - ba(测量) 1
////ebw = bw - bw(测量) 1
  m_ZpLM.GetFactor(BA_WEIGHT_PRIOR_CAMERA_MOTION/*运动先验的权重*/, m_CsLM[im]/*当前GBA中这个关键帧的状态*/,
          &m_ApLM/*雅克比，H，b对应的部分都保存在这里*/, &U/*里面是保存一些J.t*W 和 W*b的东西,一些中间变量的东西*/);
  Camera::Factor::Unitary::CC &SAcc = m_SAcus[m_ZpLM.m_iKF];//先验关键帧自己的pose x pose对应的H|-b
  Camera::Factor::Unitary &SAcm = m_SAcmsLM[im].m_Au;//m_Au存储了每一个关键帧的Hcm以及Hmm(对角线上跟运动相关)和-mb部分
  if (uc) {//R x R 对应的H|-b
    SAcc.Increase3(m_ApLM.m_Arr.m_A/*H中rr*/, m_ApLM.m_Arr.m_b/*b中rb*/);// 在原来的总的H中对应帧的部分进行增量,这个是rr的3×3的上三角和rb部分
  } else {//如果已经存在了,就
    CameraPrior::Motion::Factor::RR::AmB(m_ApLM.m_Arr, dArr, dArr);
    SAcc.Increase3(dArr.m_A, dArr.m_b);
  }
  if (ucm) {//R x motion H 以及 motion x motion的H|-b
    SAcm.m_Acm.Increase3(m_ApLM.m_Arm/*H的rm这部分*/);
    SAcm.m_Amm += m_ApLM.m_Amm;/*H的mm这部分,b中mb这部分*/
  } else {
    CameraPrior::Motion::Factor::RM::AmB(m_ApLM.m_Arm, dArm, dArm);
    CameraPrior::Motion::Factor::MM::AmB(m_ApLM.m_Amm, dAmm, dAmm);
    SAcm.m_Acm.Increase3(dArm);
    SAcm.m_Amm += dAmm;
  }
}
//用来对所有路标点的逆深度进行约束,对于只有左目观测令每个路标点的逆深度在该 source KF 所管辖的所有路标点的平均逆深度附近
//双目都观测到的话,就用外参来约束
void GlobalBundleAdjustor::UpdateFactorsPriorDepth() {

//    std::ofstream foutC(gba_debug_file, std::ios::app);
//    foutC.setf(std::ios::fixed, std::ios::floatfield);
//    foutC << "UpdateFactorsPriorDepth 第几次优化:"<<GBAdebug_count << "\n";

  FTR::Factor::DD dadd;//逆深度的先验因子
  //float dF;
#ifdef CFG_STEREO
  FTR::Factor::Stereo::U U;
#endif
#ifdef CFG_VERBOSE
  int SN = 0;
#endif
  const int nKFs = int(m_KFs.size());//关键帧数量
  for (int iKF = 0; iKF < nKFs; ++iKF)
  {//遍历所有的关键帧
    if (!(m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_DEPTH))
    {//是否需要更新深度?
      continue;
    }

    int totoal_count = 0;
    int stereo_count = 0;

    const int id = m_iKF2d[iKF];//新地图点的开始id
    const Depth::InverseGaussian *ds = m_ds.data() + id;//对应的第一个新地图点的逆深度
    ubyte *uds = m_uds.data() + id;//地图点操作的flags
    KeyFrame &KF = m_KFs[iKF];//当前关键帧

//      foutC.precision(0);
//      foutC << "kf_id:"<<iKF;
//      foutC.precision(5);
//      foutC <<",平均深度:"<<KF.m_d.u()<<",不确定度:"<<KF.m_d.s2() << std::endl;

    const Depth::Prior zp(KF.m_d.u()/*平均深度*/, 1.0f / (BA_VARIANCE_PRIOR_FRAME_DEPTH + KF.m_d.s2())/*信息矩阵*/);
    const int Nx = int(KF.m_xs.size());
    for (int ix = 0; ix < Nx; ++ix) {//遍历所有地图点
      if (!(uds[ix] & GBA_FLAG_TRACK_UPDATE_DEPTH)) {
        continue;
      }
#ifdef CFG_STEREO
      if (KF.m_xs[ix].m_xr.Valid()) {//如果是双目都观测到这个地图点的话
        FTR::Factor::Stereo &A = KF.m_Ards[ix];
        //dF = acc.m_F;
        FTR::GetFactor<GBA_ME_FUNCTION>(BA_WEIGHT_FEATURE/*权重*/, m_K.m_br/*-tc0_c1*/, ds[ix]/*地图点的观测*/, KF.m_xs[ix]/*地图点的观测*/, &A, &U);
        dadd = A.m_add;

//          foutC.precision(0);
//          foutC <<"双目约束"<<"遍历kf:"<<iKF<< ",点id:"<< ix<<",";
//          foutC.precision(5);
//          foutC <<A.m_F<<","<< A.m_add.m_a << ","<< A.m_add.m_b << std::endl;
          totoal_count++;
          stereo_count++;
      } else
#endif
      {
        Depth::Prior::Factor &A = KF.m_Apds[ix];
        //dF = acc.m_F;
        zp.GetFactor<GBA_ME_FUNCTION>(BA_WEIGHT_PRIOR_DEPTH/*先验深度的权重*/, ds[ix].u()/*这个点的深度,无双目观测的情况下是用场景平均深度给的先验*/, A/*关于地图点深度先验的因子*/);
        dadd = A;
//          foutC.precision(0);
//          foutC <<"单目约束"<<"遍历kf:"<<iKF<< ",点id:"<< ix<<",";
//          foutC.precision(5);
//          foutC << A.m_F<< ","<<A.m_a<< ","<< A.m_b << std::endl;
          totoal_count++;
      }
      KF.m_Axs[ix].m_Sadx.m_add += dadd;//这个因子是存在关键帧的因子中
      //dF = acc.m_F - dF;
      //m_F = dF + m_F;
      m_ucs[iKF] |= GBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION;//m_ucs[iKF]增加flag
      uds[ix] |= GBA_FLAG_TRACK_UPDATE_INFORMATION;//uds[ix]增加flag
#ifdef CFG_VERBOSE
      if (m_verbose >= 3) {
        ++SN;
      }
#endif
    }

//      foutC<<"KF:"<<iKF<<",单:"<<totoal_count - stereo_count<<",双:"<<stereo_count<<",双的比例:"<<double(stereo_count)/totoal_count<<"\n";
  }
#ifdef CFG_VERBOSE
  if (m_verbose >= 3) {
    const int N = int(m_ds.size());
    UT::Print("  Prior Depth  = %d / %d = %.2f%%\n", SN, N, UT::Percentage(SN, N));
  }
#endif
//  foutC.close();
}

//更新imu约束所带来的因子
//每个imu约束的因子,都连接前后两帧的pose和motion
//// 残差:
// e_r = -ln{预积分的Rij * exp[Jrbw *(bwi - z_bw)]x * Rcjw * Rciw.t}v
//  e_v = Rciw*(v_wj - v_wi + 0.5gt^2) - (m_v + m_Jvba * (bai - m_ba) + m_Jvbw * (bwi - m_bw))
//  e_p = Rciw*(p_wcj - p_wci - v_wci*dt + 0.5*g*dt^2) +
//  Rciw * Rcjw.t*tc0i - tc0i - (m_p + m_Jpba * (bai - m_ba) + m_Jpbw * (bwi - m_bw))
// e_ba = bai - baj
// e_bw = bwi - bwj

//也就是对于H的上三角,有10处影响(前帧poseX前帧pose，前帧motionX前帧motion，后帧poseX后帧pose,后帧motionX后帧motion,前帧poseX前帧motion
// ,前帧poseX后帧pose,前帧poseX后帧motion,前帧motionX后帧pose，前帧motionX后帧motion，后帧poseX后帧motion)
// b有4处(前帧pose,前帧motion,后帧pose，后帧motion),这里说的都是-b,我的习惯的是Hx=b,它定义的是Hx=-b,所以它最后求的增量是加了负号的
//遍历所有的imu约束(即m_DsLM预积分),其中前一帧关键帧id:ic1==im1,后一帧关键帧id:ic2==im2
// H部分的前帧poseX前帧pose,b部分的前帧pose存储在m_SAcus[ic1] 2处
// H部分的后帧poseX后帧pose,b部分的后帧pose存储在m_SAcus[ic2] 2处
// H部分的前帧poseX前帧motion,前帧motionX前帧motion,b部分的前帧motion存储在m_SAcmsLM[im1] 3处
// H部分的后帧poseX后帧motion,后帧motionX后帧motion,b部分的后帧motion, 7处
// 前帧poseX后帧pose,前帧poseX后帧motion,前帧motionX后帧pose，前帧motionX后帧motion都存储在m_SAcmsLM[im2]
// m_AdsLM[im2]也会作为中间变量存储H中前一帧的c,m x 前一帧的c,m,以及对应的-b,H中后一帧的c,m x 后一帧的c,m,以及对应的-b
void GlobalBundleAdjustor::UpdateFactorsIMU() {
#ifdef CFG_VERBOSE
  int SN = 0, SNu = 0;
#endif
  Camera::Factor::Unitary::CC dAcc1, dAcc2;
  Camera::Factor::Unitary dAcm1, dAcm2;
  IMU::Delta::Factor::Auxiliary::Global U;
  //float dF;
  const ubyte ucFlag = GBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION |
                       GBA_FLAG_CAMERA_MOTION_UPDATE_POSITION;//pose部分更新的flag
  const ubyte ucmFlag = ucFlag |//motion部分更新的flags
                        GBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY |
                        GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_ACCELERATION |
                        GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_GYROSCOPE;
  const int Nc = int(m_KFs.size());
  for (int ic1 = Nc - m_CsLM.Size(), ic2 = ic1 + 1, im1 = 0, im2 = 1; ic2 < Nc;
       ic1 = ic2++, im1 = im2++)
  {
    if (m_KFs[ic2].m_us.Empty()) //检查下imu数据
    {
      continue;
    }
#ifdef CFG_VERBOSE
    if (m_verbose >= 3)
    {
      ++SN;
    }
#endif
    const ubyte ucm1 = m_ucmsLM[im1], ucm2 = m_ucmsLM[im2];//前后运动更新的flags
    const bool _ucm1 = (ucm1 & ucmFlag) != 0, _ucm2 = (ucm2 & ucmFlag) != 0;
    if (!_ucm1 && !_ucm2) //如果前后运动状态都不需要更新就跳过
    {
      continue;
    }
    IMU::Delta::Factor &A = m_AdsLM[im2];/*当前imu因子,m_A11存储着H中前一帧的c,m x 前一帧的c,m,以及对应的-b,m_A22存储着H中后一帧的c,m x 后一帧的c,m,以及对应的-b*/
    Camera::Factor &SAcm1 = m_SAcmsLM[im1], &SAcm2 = m_SAcmsLM[im2];
    const bool uc1 = (ucm1 & ucFlag) != 0, uc2 = (ucm2 & ucFlag) != 0;
    //需要这个约束已经存在,只是需要更新,那么就将老的约束存储下来
    if (!uc1)
    {
      dAcc1 = A.m_A11.m_Acc;
    }
    if (!_ucm1)
    {
      dAcm1.m_Acm = A.m_A11.m_Acm;
      dAcm1.m_Amm = A.m_A11.m_Amm;
    }
    if (!uc2)
    {
      dAcc2 = A.m_A22.m_Acc;
    }
    if (!_ucm2)
    {
      dAcm2.m_Acm = A.m_A22.m_Acm;
      dAcm2.m_Amm = A.m_A22.m_Amm;
    }
    //dF = A.m_F;

    m_DsLM[im2]/*前后两帧之间的imu约束*/.GetFactor(BA_WEIGHT_IMU, m_CsLM[im1]/*前一帧状态*/, m_CsLM[im2]/*后一帧状态*/, m_K.m_pu,/*tc0_i*/
                          &A/*当前imu因子,m_A11存储着H中前一帧的c,m x 前一帧的c,m,以及对应的-b,m_A22存储着H中后一帧的c,m x 后一帧的c,m,以及对应的-b*/
                          , &SAcm2.m_Ab //存储了前一帧Pose x 后一帧Pose,前一帧Pose x 后一帧M,前一帧M x 后一帧Pose,前一帧M x 后一帧M
                          , &U/*这个imu约束总的H|-b*/, BA_ANGLE_EPSILON);
    if (uc1) //前一帧自己和自己的pose的H,-b部分关于这个因子的更新
    {
      m_SAcus[ic1] += A.m_A11.m_Acc;
    } else
    {
      Camera::Factor::Unitary::CC::AmB(A.m_A11.m_Acc, dAcc1, dAcc1);
      m_SAcus[ic1] += dAcc1;
    }
      //前一帧自己的pose和自己的M的H,自己的M和自己的M的H,-b部分关于这个因子的更新
    if (_ucm1)
    {
        SAcm1.m_Au.m_Acm += A.m_A11.m_Acm;//这一帧的pose和这一帧的motion的H
        SAcm1.m_Au.m_Amm += A.m_A11.m_Amm;//这一帧的motion自己和自己的H以及自己对应的-b
    } else
    {
      Camera::Factor::Unitary::CM::AmB(A.m_A11.m_Acm, dAcm1.m_Acm, dAcm1.m_Acm);
      Camera::Factor::Unitary::MM::AmB(A.m_A11.m_Amm, dAcm1.m_Amm, dAcm1.m_Amm);
      SAcm1.m_Au += dAcm1;
    }
    //后一帧自己和自己的pose的H,-b部分关于这个因子的更新
    if (uc2)
    {
      m_SAcus[ic2] += A.m_A22.m_Acc;
    } else
    {
      Camera::Factor::Unitary::CC::AmB(A.m_A22.m_Acc, dAcc2, dAcc2);
      m_SAcus[ic2] += dAcc2;
    }
      //后一帧自己的pose和自己的M的H,自己的M和自己的M的H,-b部分关于这个因子的更新
    if (_ucm2)
    {
      SAcm2.m_Au.m_Acm += A.m_A22.m_Acm;
      SAcm2.m_Au.m_Amm += A.m_A22.m_Amm;
    } else
    {
      Camera::Factor::Unitary::CM::AmB(A.m_A22.m_Acm, dAcm2.m_Acm, dAcm2.m_Acm);
      Camera::Factor::Unitary::MM::AmB(A.m_A22.m_Amm, dAcm2.m_Amm, dAcm2.m_Amm);
      SAcm2.m_Au += dAcm2;
    }
    //dF = A.m_F - dF;
    //m_F = dF + m_F;
#ifdef CFG_VERBOSE
    if (m_verbose >= 3) {
      ++SNu;
    }
#endif
  }
#ifdef CFG_VERBOSE
  if (m_verbose >= 3) {
    UT::Print("  Delta = %d / %d = %.2f%%\n", SNu, SN, UT::Percentage(SNu, SN));
  }
#endif
//#ifdef CFG_DEBUG
#if 0
  UT::DebugStart();
  SolveSchurComplementGT(&m_xs);
  const std::string dir = "D:/tmp/";
  const std::string fileName1 = dir + UT::String("imu_%d_before.txt", m_iIter);
  const std::string fileName2 = dir + UT::String("imu_%d_after.txt", m_iIter);
  FILE *fp1 = fopen(fileName1.c_str(), "w");
  FILE *fp2 = fopen(fileName2.c_str(), "w");
  IMU::Delta::Error e1, e2;
  const int Nm = m_CsLM.Size();
  const LA::Vector6f *xcs = (LA::Vector6f *) m_xs.Data();
  const LA::Vector9f *xms = (LA::Vector9f *) (xcs + Nc);
  for (int ic1 = Nc - m_CsLM.Size(), ic2 = ic1 + 1, im1 = 0, im2 = 1; ic2 < Nc;
       ic1 = ic2++, im1 = im2++) {
    if (m_KFs[ic2].m_imu_measures.Empty()) {
      continue;
    }
    const LA::Vector6f &xc1 = xcs[ic1], &xc2 = xcs[ic2];
    const LA::Vector9f &xm1 = xms[im1], &xm2 = xms[im2];
    const LA::AlignedVector3f xp1(&xc1.v0()), xr1(&xc1.v3());
    const LA::AlignedVector3f xp2(&xc2.v0()), xr2(&xc2.v3());
    const LA::AlignedVector3f xv1(&xm1.v0()), xba1(&xm1.v3()), xbw1(&xm1.v6());
    const LA::AlignedVector3f xv2(&xm2.v0()), xba2(&xm2.v3()), xbw2(&xm2.v6());
    const IMU::Delta &D = m_DsLM[im2];
    D.GetError(m_CsLM[im1], m_CsLM[im2], m_K.m_pu, e1);
    D.GetError(m_AdsLM[im2].m_Je, &xp1, &xr1, &xv1, &xba1, &xbw1,
                                  &xp2, &xr2, &xv2, &xba2, &xbw2, e2);
    fprintf(fp1, "%f %f %f %f\n", D.GetCost(BA_WEIGHT_IMU, e1),
                                  sqrtf(e1.m_er.SquaredLength()) * UT_FACTOR_RAD_TO_DEG,
                                  sqrtf(e1.m_ev.SquaredLength()), sqrtf(e1.m_ep.SquaredLength()));
    fprintf(fp2, "%f %f %f %f\n", D.GetCost(BA_WEIGHT_IMU, e2),
                                  sqrtf(e2.m_er.SquaredLength()) * UT_FACTOR_RAD_TO_DEG,
                                  sqrtf(e2.m_ev.SquaredLength()), sqrtf(e2.m_ep.SquaredLength()));
  }
  fclose(fp1);  UT::PrintSaved(fileName1);
  fclose(fp2);  UT::PrintSaved(fileName2);
  UT::DebugStop();
#endif
}
//固定原点
//m_ep = twc0 - 0
//er = -ln{Rwc0(初始的第一帧pose) * Rc0w(要优化的)}v
void GlobalBundleAdjustor::UpdateFactorsFixOrigin() {
  const int iKF = 0;
  if (m_KFs[iKF].m_T.m_iFrm != 0 || !(m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_CAMERA)) {//如果不是第一帧或者不需要更新相机
    return;
  }
  //float dF = m_Af->m_F;
  m_Zo.GetFactor(m_Cs[iKF]/*首帧的Tc0w*/, m_Ao/*原点固定因子*/, BA_ANGLE_EPSILON);//构造固定原点因子的H,b,存在m_Ao.m_A中,costfun存在m_Ao.m_F,m_Je里是J和残差
  m_SAcus[iKF] += m_Ao.m_A;//在第一帧的H|-b中加上这个因子
  //dF = m_Af->m_F - dF;
  //m_F = dF + m_F;
}

void GlobalBundleAdjustor::UpdateFactorsFixPositionZ() {
#ifdef CFG_VERBOSE
  int SN = 0;
#endif
  //float dF;
  const Camera::Fix::PositionZ z(BA_WEIGHT_FIX_POSITION_Z, BA_VARIANCE_FIX_POSITION_Z/*0*/);//这里BA_VARIANCE_FIX_POSITION_Z
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    const ubyte uc = m_ucs[iKF];
    if (!(uc & GBA_FLAG_FRAME_UPDATE_CAMERA))
    {
      continue;
    }
    Camera::Fix::PositionZ::Factor &A = m_Afps[iKF];
    Camera::Factor::Unitary::CC &SA = m_SAcus[iKF];//这个关键帧中自己的pose x pose对应的H|-b
    //dF = A.m_F;
    z.GetFactor(m_Cs[iKF].GetPositionZ()/*twc0[2]*/, A);//这里固定z的意义是？
    SA.m_A.m22() += z.m_w;//在对应的pose对角线矩阵上加上约束
    SA.m_b.v2() += A.m_b;
    //dF = A.m_F - dF;
    //m_F = dF + m_F;
#ifdef CFG_VERBOSE
    if (m_verbose >= 3)
      ++SN;
#endif
  }
#ifdef CFG_VERBOSE
  if (m_verbose >= 3) {
    UT::Print("  Fix Position Z = %d / %d = %.2f%%\n", SN, nKFs, UT::Percentage(SN, nKFs));
  }
#endif
}

void GlobalBundleAdjustor::UpdateFactorsFixMotion() {
#ifdef CFG_VERBOSE
  int SNv = 0, SNba = 0, SNbw = 0;
#endif
  //float dF;
  const Camera::Fix::Zero zv[3] = {
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_VELOCITY),
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_VELOCITY_INITIAL),
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_VELOCITY_INVALID)};
  const Camera::Fix::Zero zba[3] = {
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_BIAS_ACCELERATION),
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_BIAS_ACCELERATION_INITIAL),
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_BIAS_ACCELERATION_INVALID)};
  const Camera::Fix::Zero zbw[3] = {
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_BIAS_GYROSCOPE),
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_BIAS_GYROSCOPE_INITIAL),
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_BIAS_GYROSCOPE_INVALID)};
  const ubyte ucmFlag = GBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION |
                        GBA_FLAG_CAMERA_MOTION_UPDATE_POSITION |
                        GBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY |
                        GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_ACCELERATION |
                        GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_GYROSCOPE;
  const int Nm = m_CsLM.Size();
  for (int im = 0; im < Nm; ++im) {
    const ubyte ucm = m_ucmsLM[im];//当前这帧的flag
    if (!(ucm & ucmFlag)) {//如果没有要更新的就跳过
      continue;
    }
    Camera::Fix::Motion::Factor &A = m_AfmsLM[im];//fix motion部分的因子
    Camera::Factor::Unitary::MM &SA = m_SAcmsLM[im].m_Au.m_Amm;
    const Camera &C = m_CsLM[im];
    const int i = ucm >> 5;//这里都用的0,应该都没固定吧
    if (ucm & GBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY) {//如果要更新速度
      //dF = A.m_Av.m_F;
      zv[i].GetFactor(C.m_v, A.m_Av);//固定速度
      //dF = A.m_Av.m_F - dF;
      //m_F = dF + m_F;
#ifdef CFG_VERBOSE
      if (m_verbose >= 3)
        ++SNv;
#endif
    }
    if (ucm & GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_ACCELERATION) {
      //dF = A.m_Aba.m_F;
      zba[i].GetFactor(C.m_ba, A.m_Aba);//固定加速度bias
      //dF = A.m_Aba.m_F - dF;
      //m_F = dF + m_F;
#ifdef CFG_VERBOSE
      if (m_verbose >= 3)
        ++SNba;
#endif
    }
    if (ucm & GBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_GYROSCOPE) {
      //dF = A.m_Abw.m_F;
      zbw[i].GetFactor(C.m_bw, A.m_Abw);//固定陀螺仪bias
      //dF = A.m_Abw.m_F - dF;
      //m_F = dF + m_F;
#ifdef CFG_VERBOSE
      if (m_verbose >= 3)
        ++SNbw;
#endif
    }
    SA.m_A.IncreaseDiagonal012(zv[i].w());   SA.m_b.Increase(0, A.m_Av.m_b);//motion部分添加fix的这部分的约束
    SA.m_A.IncreaseDiagonal345(zba[i].w());  SA.m_b.Increase(3, A.m_Aba.m_b);
    SA.m_A.IncreaseDiagonal678(zbw[i].w());  SA.m_b.Increase(6, A.m_Abw.m_b);
  }
#ifdef CFG_VERBOSE
  if (m_verbose >= 3) {
    const int Nm = m_CsLM.Size();
    UT::Print("  Fix Motion = (%d %d %d) / %d = (%.2f%% %.2f%% %.2f%%)\n", SNv, SNba, SNbw, Nm,
              UT::Percentage(SNv, Nm), UT::Percentage(SNba, Nm), UT::Percentage(SNbw, Nm));
  }
#endif
}


//merge地图点以后对H|-b pose矩阵处造成的影响
//因子图假设三个帧 K1,K2共同观测到了点A,点A属于K0,那么就有两个视觉约束,一个连接A,K0,K1 一个连接A,K0,K2,将这两个视觉因子,即现在的影响是A对K0,1,2均有影响
//现在要把A边缘化掉,那么K0,K1,K2自身x自身,K0 x K1,K0 x K2,K1 x K2 这6处都会有影响

//step1:遍历所有关键帧的所有的地图点,计算舒尔补矩阵对角线部分要减去的东西 这里就在处理k0的自身x自身
//在KF.m_Mxs1[ix].m_mdx.m_add.m_a中保存Huu^-1,m_mdx.m_adc保存Hpose_u * Huu^-1|Huu^-1*-bu
// KF.m_Mxs2[ix].m_Mcxx = Hpose_u * Huu^-1 * Hpose_u.t|Hpose_u * Huu^-1 * -bu 这里的pose都是投影前的帧,即关键帧自己
//m_SMcus[iKF]里维护了每个关键帧舒尔补矩阵要减去的∑Hpose_u * Huu^-1 * Hpose_u.t|∑Hpose_u * Huu^-1 * -bu(∑为所有地图点)

//step2:遍历所有关键帧的观测,计算边缘化地图点以后会对舒而补中投影前pose x 投影后pose以及H的投影后posex投影后pose|投影后-b造成的影响,均存在投影后的关键帧中
//这个在处理K1,K2自身x自身,K0 x K1,K0 x K2的影响
//如果是有效观测iz2adcz记录是第几个有效观测,m_marksTmp1这个地图点索引处也改成1
//KF.m_Mzs2[iz]->m_Mcxz = H投影前pose_u * Huu^-1 * H投影后pose_u.t
//KF.m_Mzs2[iz]->m_Mczz = H投影后pose_u *Huu^-1 * H投影后pose_u.t |H投影后pose_u *Huu^-1 *-bu
//KF.m_Mzs1[iz]->m_adczA = H投影后pose_u *Huu^-1
//m_adczsTmp = H投影后pose_u
//KF.m_Zm.m_SMczms[Z.m_ik]中维护两个共视帧之间的所有地图点造成的投影前posex投影后pose的影响 ∑H投影前pose_u * Huu^-1 * H投影后pose_u.t
//m_SMcus[iKF]存储所有观测地图地图点边缘化以后造成的投影后posex投影后pose|投影后-b ∑H投影后pose_u *Huu^-1 * H投影后pose_u.t |∑H投影后pose_u *Huu^-1 *-bu
//两帧共视地图点,地图点merge以后对于两帧的影响
//KF.m_Zm.m_Mczms[i]存储地图点边缘化以后对这帧和它之前帧的共视的影响H_KFpose_u *Huu^-1*HKFpose_u.t

//step3:两帧共视地图点,地图点merge以后对于两帧的影响 这个在处理K1 x K2的影响
//KF.m_Zm.m_Mczms[i]存储地图点边缘化以后对这帧和它之前帧的共视的影响H_KFpose_u *Huu^-1*HKFpose_u.t
void GlobalBundleAdjustor::UpdateSchurComplement()
{
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    KeyFrame &KF = m_KFs[iKF];//当前这个关键帧
    if (m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_CAMERA)//
    {
      m_SMcus[iKF].MakeZero();
      KF.m_Zm.m_SMczms.MakeZero();
    } else
    {
      const int Nk = KF.m_Zm.m_SMczms.Size();
      for (int ik = 0; ik < Nk; ++ik)
      {
        if (m_ucs[KF.m_iKFsMatch[ik]] & GBA_FLAG_FRAME_UPDATE_CAMERA)//如果共视这帧需要更新,那么非对角线的元素需要设为0
        {
          KF.m_Zm.m_SMczms[ik].MakeZero();
        }
      }
    }
  }
//#ifdef CFG_DEBUG
#if 0
  UT::DebugStart();
  UT::DebugStop();
  return;
#endif
  //遍历所有关键帧中的新地图点,记录下有效的地图点(iX2d(m_idxsTmp2)中的value为有效点新编号后对应的id,-1代表无效观测,-2代表)
  int Nd = 0;//有效观测的地图点数量
  m_idxsTmp1.assign(nKFs, -1);
  int *iKF2X = m_idxsTmp1.data();//关键帧之前有多少个地图点
  std::vector<int> &iX2d = m_idxsTmp2;
  iX2d.resize(0);
  const float eps = FLT_EPSILON;
  const float epsd = UT::Inverse(BA_VARIANCE_MAX_DEPTH, BA_WEIGHT_FEATURE, eps);
  for (int iKF = 0; iKF < nKFs; ++iKF) {//遍历所有关键帧的所有当时关键帧所新观测到的地图点,如果H矩阵大于了阈值,就认为有效,加进有效索引里
    if (!(m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION))
    {
      continue;
    }
    ubyte *uds = m_uds.data() + m_iKF2d[iKF];//当前首个地图点对应的flags
    const KeyFrame &KF = m_KFs[iKF];//当前关键帧
    const int iX = static_cast<int>(iX2d.size()), Nx = static_cast<int>(KF.m_xs.size());//NX:新地图点的数量
    iKF2X[iKF] = iX;//保存一下这个关键帧之前有多少个地图点
    iX2d.resize(iX + Nx, -1);
    int *ix2d = iX2d.data() + iX;
    for (int ix = 0; ix < Nx; ++ix) {
      if (!(uds[ix] & GBA_FLAG_TRACK_UPDATE_INFORMATION))
      {//如果这个地图点不用更新
        continue;
      } else if (KF.m_Axs[ix].m_Sadx.m_add.m_a > epsd)//Huu大于0才加入(u代表这个点的逆深度对应的)
      {
        ix2d[ix] = Nd++;//有效的地图点么,双目观测时才有
      } else if (!(uds[ix] & GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO))//如果不是第一次观测到这个地图点,而且还不满足H>0时才会设成-2
      {
        ix2d[ix] = -2;
      }
    }
  }
#ifdef CFG_VERBOSE
  int SNX = 0, SNs1 = 0, SNs2 = 0, SNS = 0;
#endif
  const int NmddC = SIMD_FLOAT_CEIL(Nd);
  m_work.Resize(NmddC + Nd * sizeof(xp128f) / sizeof(float));
  float *mdds = m_work.Data();//存的是H地图点部分(V)的逆,V^-1
  xp128f *_mdds = (xp128f *) (mdds + NmddC);
  for (int iKF = 0; iKF < nKFs; ++iKF)//
  {//遍历所有关键帧上的新观测地图点,有效的逆深度先验因子就加到m_work里,然后再取逆
    const int iX = iKF2X[iKF];
    if (iX == -1)
    {
      continue;
    }
#ifdef CFG_VERBOSE
    if (m_verbose >= 3)
    {
      ++SNX;
    }
#endif
    const int *ix2d = iX2d.data() + iX;
    const KeyFrame &KF = m_KFs[iKF];
    const int Nx = static_cast<int>(KF.m_xs.size());
    for (int ix = 0; ix < Nx; ++ix)
    {
      const int id = ix2d[ix];
      if (id >= 0)
      {//有效的逆深度先验因子就加到m_work里
        mdds[id] = KF.m_Axs[ix].m_Sadx.m_add.m_a;
      }
    }
  }
  //SIMD::Add(Nd, UT::Inverse(BA_VARIANCE_REGULARIZATION_DEPTH, BA_WEIGHT_FEATURE), mdds);
  SIMD::Inverse(Nd, mdds);//地图点的H求逆得到H^-1,mdds已经变成Huu^-1了 u代表逆深度
  for (int id = 0; id < Nd; ++id)
  {
    _mdds[id].vdup_all_lane(mdds[id]);//数据类型转换
  }
#if defined CFG_VERBOSE && defined CFG_DEBUG
  if (m_verbose >= 3) {
    for (int iKF = 0; iKF < nKFs; ++iKF) {
      UT_ASSERT(!(m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_SCHUR_COMPLEMENT));
    }
  }
#endif

    //遍历所有关键帧的所有的地图点,计算舒尔补矩阵对角线部分要减去的东西
    //在KF.m_Mxs1[ix].m_mdx.m_add.m_a中保存Huu^-1,m_mdx.m_adc保存Hpose_u * Huu^-1|Huu^-1*-bu
    // KF.m_Mxs2[ix].m_Mcxx = Hpose_u * Huu^-1 * Hpose_u.t|Hpose_u * Huu^-1 * -bu 这里的pose都是投影前的帧,即关键帧自己
    //m_SMcus[iKF]里维护了每个关键帧舒尔补矩阵要减去的∑Hpose_u * Huu^-1 * Hpose_u.t|∑Hpose_u * Huu^-1 * -bu(∑为所有地图点)
  Camera::Factor::Unitary::CC dMcu;
  for (int iKF = 0; iKF < nKFs; ++iKF)
  {
    const int iX = iKF2X[iKF];
    if (iX == -1)
    {
      continue;
    }
    const int *ix2d = iX2d.data() + iX;//用来判断地图点是否是有效的观测
    ubyte *uds = m_uds.data() + m_iKF2d[iKF];//对应的逆深度
    Camera::Factor::Unitary::CC &SMcxx = m_SMcus[iKF];//
#ifdef CFG_VERBOSE
    ubyte Scxx = 0;
#endif
    const bool uc = (m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_CAMERA) != 0;//是否要更新pose
    KeyFrame &KF = m_KFs[iKF];
    const int Nx = static_cast<int>(KF.m_xs.size());


    for (int ix = 0; ix < Nx; ++ix)
    {//遍历所有新观测到的地图点
      const int id = ix2d[ix];
      if (id == -1)
      {//只取有效观测
        continue;
      }
      FTR::Factor::Full::Source::M2 &M = KF.m_Mxs2[ix];
      if (id >= 0)
      {
        if (uc || (uds[ix] & GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO))//如果这个地图点是一个新地图点
        {//KF.m_Mxs1[ix].m_mdx.m_add.m_a = Huu^-1* ,M.m_Mcxx = Hpose_u * Huu^-1 * Hpose_u.t|Hpose_u * Huu^-1 * -bu
          FTR::Factor::Full::Source::Marginalize(_mdds[id]/*这个点对应的Huu^-1*/, KF.m_Axs[ix].m_Sadx/*用到Hpose_u*/, &KF.m_Mxs1[ix], &M);
          SMcxx += M.m_Mcxx;
        } else
        {
          dMcu = M.m_Mcxx;//如果已经有了这个因子,需要更新,则+新-旧
          FTR::Factor::Full::Source::Marginalize(_mdds[id], KF.m_Axs[ix].m_Sadx, &KF.m_Mxs1[ix], &M);
          Camera::Factor::Unitary::CC::AmB(M.m_Mcxx, dMcu, dMcu);
          SMcxx += dMcu;
        }
#ifdef CFG_VERBOSE
        if (m_verbose >= 3)
        {
          ++SNs1;
        }
#endif
      } else
      {
#ifdef CFG_DEBUG
        UT_ASSERT((uds[ix] & GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO) == 0);
#endif
        if (!uc)
        {
          M.m_Mcxx.GetMinus(dMcu);
          SMcxx += dMcu;
        }
#ifdef CFG_DEBUG
        KF.m_Mxs1[ix].m_mdx.Invalidate();
        M.m_Mcxx.Invalidate();
#endif
      }
#ifdef CFG_VERBOSE
      if (m_verbose >= 3)
      {
        Scxx = 1;
      }
#endif
    }
#ifdef CFG_VERBOSE
    if (m_verbose >= 3 && Scxx)
    {
      m_ucs[iKF] |= GBA_FLAG_FRAME_UPDATE_SCHUR_COMPLEMENT;
    }
#endif
  }
#ifdef CFG_VERBOSE
  if (m_verbose >= 3)
  {
    SNs2 += static_cast<int>(m_ds.size());
  }
#endif
  //遍历所有关键帧的观测,计算边缘化地图点以后会对舒而补中投影前pose x 投影后pose以及H的投影后posex投影后pose|投影后-b造成的影响,均存在投影后的关键帧中
  //如果是有效观测iz2adcz记录是第几个有效观测,m_marksTmp1这个地图点索引处也改成1
    //KF.m_Mzs2[iz]->m_Mcxz = H投影前pose_u * Huu^-1 * H投影后pose_u.t
    //KF.m_Mzs2[iz]->m_Mczz = H投影后pose_u *Huu^-1 * H投影后pose_u.t |H投影后pose_u *Huu^-1 *-bu
    //KF.m_Mzs1[iz]->m_adczA = H投影后pose_u *Huu^-1
    //m_adczsTmp = H投影后pose_u
    //KF.m_Zm.m_SMczms[Z.m_ik]中维护两个共视帧之间的所有地图点造成的投影前posex投影后pose的影响 ∑H投影前pose_u * Huu^-1 * H投影后pose_u.t
    //m_SMcus[iKF]存储所有观测地图地图点边缘化以后造成的投影后posex投影后pose|投影后-b ∑H投影后pose_u *Huu^-1 * H投影后pose_u.t |∑H投影后pose_u *Huu^-1 *-bu
    //两帧共视地图点,地图点merge以后对于两帧的影响
//KF.m_Zm.m_Mczms[i]存储地图点边缘化以后对这帧和它之前帧的共视的影响H_KFpose_u *Huu^-1*HKFpose_u.t
  Camera::Factor::Binary::CC dMcb;
  std::vector<int> &iz2adcz = m_idxsTmp3;
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    KeyFrame &KF = m_KFs[iKF];//当前关键帧
    Camera::Factor::Unitary::CC &SMczz = m_SMcus[iKF];//当前关键帧舒尔补矩阵要减去的∑Hpose_u * Huu^-1 * Hpose_u.t|∑Hpose_u * Huu^-1 * -bu(∑为所有地图点)
#ifdef CFG_VERBOSE
    ubyte Sczz = 0;
#endif
    const int Nz = static_cast<int>(KF.m_zs.size());
    m_marksTmp1.assign(Nz, 0);
    iz2adcz.assign(Nz, -1);//扩容成观测的
    m_adczsTmp.Resize(0);
#ifdef CFG_VERBOSE
    m_marksTmp1.resize(Nz + KF.m_Zm.m_SMczms.Size(), 0);//观测+与哪个关键帧之间有共视的size
    ubyte *Scbs = m_marksTmp1.data() + Nz;
#endif
    const bool ucz = (m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_CAMERA) != 0;
    const int NZ = static_cast<int>(KF.m_Zs.size());
      //遍历所有关键帧的观测,计算边缘化地图点以后会对舒而补中投影前pose x 投影后pose以及H的投影后posex投影后pose|投影后-b造成的影响,均存在投影后的关键帧中
    for (int iZ = 0; iZ < NZ; ++iZ)//遍历所有观测关键帧
    {
      const FRM::Measurement &Z = KF.m_Zs[iZ];
      const int _iKF = Z.m_iKF/*观测到的关键帧*/, _iX = iKF2X[_iKF];/*首个地图点id*/
      if (_iX == -1)
      {
        continue;
      }
      const ubyte *_uds = m_uds.data() + m_iKF2d[_iKF];
      const int *_ix2mdd = iX2d.data() + _iX;
      Camera::Factor::Binary::CC &SMcxz = KF.m_Zm.m_SMczms[Z.m_ik];
#ifdef CFG_VERBOSE
      ubyte &Scxz = Scbs[Z.m_ik];
#endif
      const bool ucx = (m_ucs[_iKF] & GBA_FLAG_FRAME_UPDATE_CAMERA) != 0/*投影前这帧是否更新*/, ucr = ucx || ucz;//
      const KeyFrame &_KF = m_KFs[_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz)
      {
        const int ix = KF.m_zs[iz].m_ix, id = _ix2mdd[ix];
        if (id == -1)//==-1代表是无效的视觉观测
        {
          continue;
        }
        FTR::Factor::Full::M2 &M = KF.m_Mzs2[iz];
        if (id >= 0)
        {
          iz2adcz[iz] = m_adczsTmp.Size();//有效的共视,记录下来
          LA::ProductVector6f &adcz = m_adczsTmp.Push();
          if (_uds[ix] & GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO)//这个点如果是一个新的因子
          {//注释看下面的
            FTR::Factor::Full::Marginalize(_mdds[id], _KF.m_Mxs1[ix], KF.m_Azs1[iz], &KF.m_Mzs1[iz], &M, &adcz);
            SMcxz += M.m_Mcxz;
            SMczz += M.m_Mczz;
            m_marksTmp1[iz] = 1;//观测有效
          } else
          {
            if
            (!ucr) {
              dMcb = M.m_Mcxz;
            }
            if (!ucz) {
              dMcu = M.m_Mczz;
            }
            //KF.m_Mzs2[iz]->m_Mcxz = H投影前pose_u * Huu^-1 * H投影后pose_u
            //KF.m_Mzs2[iz]->m_Mczz = H投影后pose_u *Huu^-1 * H投影后pose_u |H投影后pose_u *Huu^-1 *-bu
            //KF.m_Mzs1[iz]->m_adczA = H投影后pose_u *Huu^-1
            //adcz = H投影后pose_u
            FTR::Factor::Full::Marginalize(_mdds[id]/*Huu^-1*/, _KF.m_Mxs1[ix],/*m_mdx.m_adc保存H投影前pose_u * Huu^-1*/
                    KF.m_Azs1[iz]/*m_adcz存储H的投影后pose x 逆深度部分*/, &KF.m_Mzs1[iz], &M, &adcz);
            if (ucr) {//更新投影前posex投影后pose这部分要减去的这部分
              SMcxz += M.m_Mcxz;
            } else {
              Camera::Factor::Binary::CC::AmB(M.m_Mcxz, dMcb, dMcb);
              SMcxz += dMcb;
            }
            if (ucz) {//更新投影后posex投影后pose以及-b投影后要减去的部分
              SMczz += M.m_Mczz;
            } else {
              Camera::Factor::Unitary::CC::AmB(M.m_Mczz, dMcu, dMcu);
              SMczz += dMcu;
            }
          }
#ifdef CFG_VERBOSE
          if (m_verbose >= 3) {
            SNs1 += 2;
          }
#endif
        } else
        {
#ifdef CFG_DEBUG
          UT_ASSERT((_uds[ix] & GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO) == 0);
#endif
          if (!ucr) {//否则就减去无效的因子
            M.m_Mcxz.GetMinus(dMcb);
            SMcxz += dMcb;
          }
          if (!ucz) {
            M.m_Mczz.GetMinus(dMcu);
            SMczz += dMcu;
          }
          iz2adcz[iz] = -2;
#ifdef CFG_DEBUG
          KF.m_Mzs1[iz].m_adcz.Invalidate();
          M.m_Mcxz.Invalidate();
          M.m_Mczz.Invalidate();
#endif
        }
#ifdef CFG_VERBOSE
        if (m_verbose >= 3)
          Scxz = Sczz = 1;
#endif
//#ifdef CFG_DEBUG
#if 0
        UT::Print("%d %10e\n", iz, SMczz.m_A.m02());
#endif
      }
    }


#ifdef CFG_VERBOSE
    if (m_verbose >= 3) {
      SNs2 += Nz * 2;
      if ((m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_SCHUR_COMPLEMENT) || Sczz) {
        ++SNS;
      }
      m_ucs[iKF] &= ~GBA_FLAG_FRAME_UPDATE_SCHUR_COMPLEMENT;
    }
#endif

//两帧共视地图点,地图点merge以后对于两帧的影响
//KF.m_Zm.m_Mczms[i]存储地图点边缘化以后对这帧和它之前帧的共视的影响H_KFpose_u *Huu^-1*HKFpose_u.t
    const int Nk = KF.m_Zm.m_SMczms.Size();//这个关键帧的所有观测到的关键帧的个数
    for (int ik = 0; ik < Nk; ++ik)
    {
      const int _iKF = KF.m_iKFsMatch[ik];
#ifdef CFG_DEBUG
      UT_ASSERT(_iKF < iKF);
#endif
      const KeyFrame &_KF = m_KFs[_iKF];//共视的这个关键帧
      Camera::Factor::Binary::CC &SMczm = KF.m_Zm.m_SMczms[ik];//两个共视帧之间的所有地图点造成的投影前posex投影后pose的影响 ∑H投影前pose_u * Huu^-1 * H投影后pose_u.t
#ifdef CFG_VERBOSE
      ubyte &Sczm = Scbs[ik];
#endif
      const bool _ucz = (m_ucs[_iKF] & GBA_FLAG_FRAME_UPDATE_CAMERA) != 0, uczm = _ucz || ucz;//这两帧中是否有需要更新pose的帧
      const int i1 = KF.m_Zm.m_ik2zm[ik], i2 = KF.m_Zm.m_ik2zm[ik + 1];
      for (int i = i1; i < i2; ++i)//遍历共视地图点
      {
        const FTR::Measurement::Match &izm = KF.m_Zm.m_izms[i];
        const int iadcz = iz2adcz[izm.m_iz2];
        if (iadcz == -1)//说明是有效的观测
        {
          continue;
        }
        Camera::Factor::Binary::CC &Mczm = KF.m_Zm.m_Mczms[i];
        if (iadcz >= 0)
        {
          if (uczm || m_marksTmp1[izm.m_iz2])//需要更新或者这个是有效的观测
          {//Mczm = H_KFpose_u *Huu^-1*HKFpose_u.t (因为_KF肯定是在KF前面的,只构建上三角处的)
            FTR::Factor::Full::Marginalize(_KF.m_Mzs1[izm.m_iz1]/*m_adczA = H_KFpose_u *Huu^-1*/, m_adczsTmp[iadcz]/*HKFpose_u*/, Mczm);
            SMczm += Mczm;
          } else
          {
            dMcb = Mczm;
            FTR::Factor::Full::Marginalize(_KF.m_Mzs1[izm.m_iz1], m_adczsTmp[iadcz], Mczm);
            Camera::Factor::Binary::CC::AmB(Mczm, dMcb, dMcb);
            SMczm += dMcb;
          }
#ifdef CFG_VERBOSE
          if (m_verbose >= 3)
            ++SNs1;
#endif
        } else
        {
          if (!uczm)
          {
            Mczm.GetMinus(dMcb);
            SMczm += dMcb;
          }
#ifdef CFG_DEBUG
          Mczm.Invalidate();
#endif
        }
#ifdef CFG_VERBOSE
        if (m_verbose >= 3)
        {
          Sczm = 1;
        }
#endif
      }
    }
#ifdef CFG_VERBOSE
    if (m_verbose >= 3)
    {
      SNs2 += static_cast<int>(KF.m_Zm.m_izms.size());
      for (int ik = 0; ik < Nk; ++ik)
      {
        if (Scbs[ik])
        {
          ++SNS;
        }
      }
    }
#endif
  }
  for (int iKF = 0; iKF < nKFs; ++iKF)
  {//
    const int iX = iKF2X[iKF];
    if (iX == -1)
    {
      continue;
    }
    ubyte *uds = m_uds.data() + m_iKF2d[iKF];//对应的新地图点的逆深度
    const int *ix2d = iX2d.data() + iX;
    const int Nx = static_cast<int>(m_KFs[iKF].m_xs.size());
    for (int ix = 0; ix < Nx; ++ix)
    {
      if (ix2d[ix] >= 0)//如果这个点这次有效,那么就将GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO设为0,否则为1,即上次没有有效信息
      {
        uds[ix] &= ~GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;
      } else if (ix2d[ix] == -2)
      {
        uds[ix] |= GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;
      }
    }
  }
#ifdef CFG_VERBOSE
  if (m_verbose >= 3) {
    const int _Nd = static_cast<int>(m_ds.size()), NS = CountSchurComplements();
    UT::PrintSeparator();
    UT::Print("*%2d: [GlobalBundleAdjustor::UpdateSchurComplement]\n", m_iIter);
    UT::Print("  Track = %d / %d = %.2f%% (%d / %d = %.2f%%)\n", Nd, _Nd, UT::Percentage(Nd, _Nd),
              SNX, nKFs, UT::Percentage(SNX, nKFs));
    UT::Print("  Schur = %d / %d = %.2f%% (%d / %d = %.2f%%)\n", SNs1, SNs2, UT::Percentage(SNs1, SNs2),
              SNS, NS, UT::Percentage(SNS, NS));
  }
#endif
}

//////////////////////////////////////////////////////////////////////////
//static int g_iIter = 0;
//////////////////////////////////////////////////////////////////////////

bool GlobalBundleAdjustor::SolveSchurComplement() {
//#ifdef CFG_INCREMENTAL_PCG
#if 0
#ifdef CFG_INCREMENTAL_PCG_1
  m_xcs.MakeZero();
  m_xmsLM.MakeZero();
#endif
#endif
  const bool scc = SolveSchurComplementPCG();
  //////////////////////////////////////////////////////////////////////////
  //const int nIters = 20;
  //for (g_iIter = 0; g_iIter < nIters; ++g_iIter) {
  //  const int Nc = m_Cs.Size(), Nm = m_CsLM.Size();
  //  const LA::Vector6f *xcsGN = (LA::Vector6f *) m_xsGN.Data();
  //  const LA::Vector9f *xmsGN = (LA::Vector9f *) (xcsGN + Nc);
  //  for (int ic = 0, im = Nm - Nc; ic < Nc; ++ic, ++im) {
  //    m_xcs[ic] = xcsGN[ic];
  //    if (im >= 0) {
  //      m_xmsLM[im] = xmsGN[im];
  //    }
  //  }
  //  UT::PrintSeparator('*');
  //  UT::Print("%d\n", g_iIter);
  //  SolveSchurComplementPCG();
  //}
  //////////////////////////////////////////////////////////////////////////
#ifdef GBA_DEBUG_GROUND_TRUTH_STATE
  SolveSchurComplementGT(m_Cs, m_CsLM, &m_xsGN);
#endif
  if (GBA_EMBEDDED_MOTION_ITERATION) {
//#ifdef CFG_DEBUG
#if 0
    const LA::Vector6f *xcs = (LA::Vector6f *) m_xsGN.Data();
    const LA::Vector9f *xms = (LA::Vector9f *) (xcs + m_KFs.size());
    ConvertCameraUpdates(xcs, &m_xcsP);
    const IMU::Delta::ES ESd1 = ComputeErrorStatisticIMU(m_xcsP.Data(), xms, false);
    CameraPrior::Motion::ES ESm1 = ComputeErrorStatisticPriorCameraMotion(m_xcsP.Data(), xms);
    const float F1 = ESd1.Total() + ESm1.Total();
#endif
    EmbeddedMotionIteration();
//#ifdef CFG_DEBUG
#if 0
    //PrintSchurComplementResidual();
    const IMU::Delta::ES ESd2 = ComputeErrorStatisticIMU(m_xcsP.Data(), xms, false);
    CameraPrior::Motion::ES ESm2 = ComputeErrorStatisticPriorCameraMotion(m_xcsP.Data(), xms);
    const float F2 = ESd2.Total() + ESm2.Total();
    //UT_ASSERT(F2 <= F1);
    UT::AssertReduction(F1, F2, 1, UT::String("[%d] %d", m_KFs.back().m_Cam_pose.m_iFrm));
    //UT::Print("%e --> %e\n", F1, F2);
#endif
  }
//#ifdef CFG_INCREMENTAL_PCG
#if 0
  FILE *fp;
  std::string fileName;
  const int iFrm = m_KFs.back().m_Cam_pose.m_iFrm;
#ifdef CFG_INCREMENTAL_PCG_1
  fileName = "D:/tmp/pcg/count_gba.txt";
#else
  fileName = "D:/tmp/pcg/count_gba_incr.txt";
#endif
  static bool g_first = true;
  fp = fopen(fileName.c_str(), g_first ? "w" : "a");
  g_first = false;
  fprintf(fp, "%d %d %d\n", iFrm, m_iIter, m_iIterPCG);
  fclose(fp);
  fileName = UT::String("D:/tmp/pcg/state_gba_%04d_%02d.txt", iFrm, m_iIter);
#ifdef CFG_INCREMENTAL_PCG_1
  fp = fopen(fileName.c_str(), "wb");
  m_xsGN.SaveB(fp);
  m_xp2s.SaveB(fp);
  m_xr2s.SaveB(fp);
  m_xv2s.SaveB(fp);
  m_xba2s.SaveB(fp);
  m_xbw2s.SaveB(fp);
  UT::SaveB(scc, fp);
  fclose(fp);
#else
  fp = fopen(fileName.c_str(), "rb");
  m_xsGN.LoadB(fp);
  m_xp2s.LoadB(fp);
  m_xr2s.LoadB(fp);
  m_xv2s.LoadB(fp);
  m_xba2s.LoadB(fp);
  m_xbw2s.LoadB(fp);
  const bool _scc = UT::LoadB<bool>(fp);
  fclose(fp);
  return _scc;
#endif
#endif
  if (!scc) {
    return false;
  }
  return true;
}
//Hx = b ==》 舒尔补以后就变成了S*δxc = g ==》 A*x = b 用PCG迭代法求解
bool GlobalBundleAdjustor::SolveSchurComplementPCG() {
  Camera::Factor::Unitary::CC Acc;
  const int pc = 6/*pose部分的size*/, pm/*motion部分的size*/ = 9;
  const int Nc = m_Cs.Size(), Nm = m_CsLM.Size()/*c和m应该数量相等的吧,有一帧的c就会有两帧之间的motion部分*/, Ncp = Nc * pc/*相机部分的长度*/, Nmp = Nm * pm/*motion部分的长度*/, N = Ncp + Nmp/*S的长度*/;
  m_Acus.Resize(Nc);
  m_bs.Resize(N);
  const float ar = UT::Inverse(BA_VARIANCE_REGULARIZATION_ROTATION, BA_WEIGHT_FEATURE);//0
  const float ap = UT::Inverse(BA_VARIANCE_REGULARIZATION_POSITION, BA_WEIGHT_FEATURE);//0
  const float av = UT::Inverse(BA_VARIANCE_REGULARIZATION_VELOCITY, BA_WEIGHT_FEATURE);//0
  const float aba = UT::Inverse(BA_VARIANCE_REGULARIZATION_BIAS_ACCELERATION, BA_WEIGHT_FEATURE);//0
  const float abw = UT::Inverse(BA_VARIANCE_REGULARIZATION_BIAS_GYROSCOPE, BA_WEIGHT_FEATURE);//0
  float *b = m_bs.Data();
  ////按帧的先后顺序进行遍历,构建S和-g
    //    //S的对角线的pose x pose部分 存储在m_Acus[ic]里,S的对角线的motionxmotion存在m_Amus[ic]里
    //    //S的非对角线的pose x 别的共视帧的pose 存在Acbs里,利用m_ic2b进行索引
    //    // -g存在m_ic2b里
  for (int ic = 0; ic < Nc; ++ic, b += pc) {
    Camera::Factor::Unitary::CC::AmB(m_SAcus[ic], m_SMcus[ic], Acc);//对角线Sij = U − W*V*W.T
    Acc.m_A.IncreaseDiagonal(ap, ar);
    Acc.m_A.GetAlignedMatrix6x6f(m_Acus[ic]);//p,q部分将Acc.m_A上三角 存到m_Acus[ic]里,然后扩展成对称矩阵
    Acc.m_b.Get(b);//-g中pq部分
  }
  m_AmusLM.Resize(Nm);//H的mm部分,-b的m部分。舒尔补对M的地方没有影响
  for (int im = 0; im < Nm; ++im, b += pm) {//遍历所有的运动,设置舒尔补以后的S中属于运动对角部分的元素，以及g中属于运动的部分
    const Camera::Factor::Unitary::MM &Amm = m_SAcmsLM[im].m_Au.m_Amm;//对角线部分mm的部分
    m_AmusLM[im].Set(Amm.m_A);//S中的v,ba,bg部分设置
    m_AmusLM[im].IncreaseDiagonal(av, aba, abw);
    Amm.m_b.Get(b);//设置-g中v,ba,bg部分
  }
  const int NKp = CountSchurComplementsOffDiagonal();//返回的是所有的共视帧个数,就是第一个关键帧的共视+ ..... +当前关键帧的共视
  m_Acbs.Resize(NKp);//扩容成所有共视的size
  m_Acbs.MakeZero();
  for (int ic = 0, im = Nm - Nc, iKp = 0; ic < Nc; ++ic, ++im) {
    LA::AlignedMatrix6x6f *Acbs = m_Acbs.Data() + m_iKF2cb[ic];//当前这个关键帧的共视起点
    const KeyFrame &KF = m_KFs[ic];
    const int NZ = static_cast<int>(KF.m_Zs.size());//添加所有的视觉约束给的H投影前(观测关键帧)pose x 投影后pose的影响
    for (int iZ = 0; iZ < NZ; ++iZ) {
      Acbs[KF.m_Zs[iZ].m_ik] = KF.m_SAcxzs[iZ];//维护所有观测投影帧的投影前(观测关键帧)pose x 投影后pose的H
    }
    const int Np = static_cast<int>(KF.m_iKFsPrior.size());//再添加这个关键帧和之前关键之间的位姿约束
    for (int ip = 0; ip < Np; ++ip) {
      Acbs[KF.m_iAp2kp[ip]] += KF.m_SAps[ip];
    }
    const int Nk = KF.m_Zm.m_SMczms.Size();//这个关键帧共视的数量
    for (int ik = 0; ik < Nk; ++ik) {
      Acbs[ik] -= KF.m_Zm.m_SMczms[ik];//地图点造成的投影前posex投影后pose的影响 ∑H投影前pose_u * Huu^-1 * H投影后pose_u.t
    }
    //if (im >= 1 && !(m_ucmsLM[im] & GBA_FLAG_CAMERA_MOTION_INVALID) &&
    //               !(m_ucmsLM[im - 1] & GBA_FLAG_CAMERA_MOTION_INVALID)) {
    if (im >= 1 && !KF.m_us.Empty()) {
#ifdef CFG_DEBUG
      //UT_ASSERT(KF.m_iKFsMatch.back() == ic - 1);
      UT_ASSERT(KF.m_iKFsMatch[KF.m_Zm.m_SMczms.Size() - 1] == ic - 1);
      UT_ASSERT(!(m_ucmsLM[im] & GBA_FLAG_CAMERA_MOTION_INVALID) &&
                !(m_ucmsLM[im - 1] & GBA_FLAG_CAMERA_MOTION_INVALID));
#endif
      Acbs[Nk - 1] += m_SAcmsLM[im].m_Ab.m_Acc;//imu约束带来的前一帧的pose x 后一帧的pose
    }
  }
  m_AcbTs.Resize(NKp);
  for (int ikp = 0; ikp < NKp; ++ikp) {
    m_Acbs[ikp].GetTranspose(m_AcbTs[ikp]);//m_AcbTs[] = m_Acbs[].t
  }
  PrepareConditioner();//利用S的对角线构造预优矩阵M,并且直接求逆(取倒数),算出M^-1

  m_xs.Resize(N);
  m_rs.Resize(N);
  m_ps.Resize(N);
  m_zs.Resize(N);
  m_drs.Resize(N);
  m_dxs.Resize(N);
//#ifdef CFG_DEBUG_EIGEN
#if 0
  EigenMatrixXd e_A;
  e_A.resize(N, N);
  e_A.setZero();
  for (int ic = 0, icp = 0; ic < Nc; ++ic, icp += pc) {
    e_A.block<pc, pc>(icp, icp) = EigenMatrix6x6f(m_Acus[ic]).cast<double>();
    const KeyFrame &KF = m_KFs[ic];
    const LA::AlignedMatrix6x6f *Acbs = m_Acbs.Data() + KF.m_iKp;
    const int Nkp = static_cast<int>(KF.m_ikp2KF.size());
    for (int ikp = 0; ikp < Nkp; ++ikp) {
      const int _icp = KF.m_ikp2KF[ikp] * pc;
      e_A.block<pc, pc>(_icp, icp) = EigenMatrix6x6f(Acbs[ikp]).cast<double>();
    }
  }
  for (int ic = Nc - Nm, icp = ic * pc, im = 0, imp = Ncp; ic < Nc;
       ++ic, icp += pc, ++im, imp += pm) {
    const Camera::EigenFactor e_Acm = m_SAcmsLM[im];
    //e_A.block<pm, pm>(imp, imp) = e_Acm.m_Au.m_Amm.cast<double>();
    e_A.block<pm, pm>(imp, imp) = EigenMatrix9x9f(m_AmusLM[im]).cast<double>();
    e_A.block<pc, pm>(icp, imp) = e_Acm.m_Au.m_Acm.cast<double>();
    if (im > 0) {
      const int _icp = icp - pc, _imp = imp - pm;
      e_A.block<pc, pm>(_icp, imp) = e_Acm.m_Ab.m_Acm.cast<double>();
      e_A.block<pm, pc>(_imp, icp) = e_Acm.m_Ab.m_Amc.cast<double>();
      e_A.block<pc, pm>(icp, _imp) = e_Acm.m_Ab.m_Amc.cast<double>().transpose();
      e_A.block<pm, pm>(_imp, imp) = e_Acm.m_Ab.m_Amm.cast<double>();
    }
  }
  e_A.SetLowerFromUpper();
  EigenVectorXd e_s;
  const int e_rankLU = EigenRankLU(e_A), e_rankQR = EigenRankQR(e_A);
  UT::Print("rank = %d (%d) / %d\n", e_rankLU, e_rankQR, N);
  const double e_cond = EigenConditionNumber(e_A, &e_s);
  UT::Print("cond = %e\n", e_cond);

  const EigenVectorXd e_b = EigenVectorXd(m_bs);
  const EigenVectorXd e_x = EigenVectorXd(e_A.ldlt().solve(e_b));
  const EigenVectorXd e_Ax = EigenVectorXd(e_A * e_x);
  const EigenVectorXd e_r = EigenVectorXd(e_Ax - e_b);
  m_xsGN = e_x.GetAlignedVectorXf();
  m_xsGN.MakeMinus();
  SolveSchurComplementGT(m_Cs, m_CsLM, &m_xsGT);
  m_xsGT.MakeMinus();
  const EigenVectorXd e_xGT = EigenVectorXd(m_xsGT);
  const EigenVectorXd e_AxGT = EigenVectorXd(e_A * e_xGT);
  const EigenVectorXd e_rGT = EigenVectorXd(e_AxGT - e_b);
  //const Residual RGT = ComputeResidual(m_xsGT, true);
  UT::Print("%e vs %e\n", e_r.norm(), e_rGT.norm());
  ConvertCameraUpdates(m_xsGN.Data(), &m_xp2s, &m_xr2s);
  ConvertMotionUpdates(m_xsGN.Data() + Ncp, &m_xv2s, &m_xba2s, &m_xbw2s);
  UT::Print("%f %f %f %f %f\n", sqrtf(m_xp2s.Mean()), sqrtf(m_xr2s.Mean()) * UT_FACTOR_RAD_TO_DEG,
            sqrtf(m_xv2s.Mean()), sqrtf(m_xba2s.Mean()), sqrtf(m_xbw2s.Mean()) * UT_FACTOR_RAD_TO_DEG);
  return true;
#endif

  bool scc = true;
  PCG_TYPE Se2, Se2Pre, e2Max, alpha, beta, F, FMin;
  PCG_TYPE Se2Min/*最小残差*/, e2MaxMin;
  m_rs = m_bs;//rs这里先存一下b的值
  //m_rs.Swap(m_bs);
#ifdef CFG_INCREMENTAL_PCG
  m_xsGN.Resize(0);
  m_xsGN.Push((float *) m_xcs.Data(), Ncp);//pose部分数据
  m_xsGN.Push((float *) m_xmsLM.Data(), Nmp);//motion部分数据
  m_xsGN.MakeMinus();//这里求的是A*-xcm = b m_xcs,m_xmsLM,所以这里要负,下面PCG的部分我就把-xcm称为x
  //UT::DebugStart();
  //m_xsGN.MakeZero();
  //UT::DebugStop();
  m_xs = m_xsGN;//之前的优化变量作为初值x0
  ApplyA(m_xs, &m_drs);//计算 m_drs  = A*x0 S
  m_rs -= m_drs;// r0 = b - A*X0
#else
  m_xsGN.MakeZero();
  m_xs.MakeZero();
#endif
  ApplyM(m_rs/*残差*/, &m_ps/*p0*/);//z0 = M.inv * r0,初始的梯度方向z0同时第1步时梯度方向就是下降方向
#if 0
//#if 1
  UT::DebugStart();
  LA::AlignedVectorXf xs, rs, ps;
  SolveSchurComplementGT(m_Cs, m_CsLM, &m_xsGT);
  m_xsGT.MakeMinus();
  LA::AlignedVectorXf::AmB(m_xsGT, m_xs, ps);
  ApplyA(m_xsGT, &m_drs);
  rs = m_bs;
  rs -= m_drs;
  const PCG_TYPE r0 = rs.SquaredLength();
  const PCG_TYPE F0 = m_xsGT.Dot(m_drs) * 0.5 - m_xsGT.Dot(m_bs);
  ApplyA(ps, &m_drs);
  const PCG_TYPE a0 = m_rs.Dot(ps) / ps.Dot(m_drs);
  const PCG_TYPE a1 = 0.9;
  const PCG_TYPE a2 = 1.1;
  const PCG_TYPE da = 1.0e-3;
  for (PCG_TYPE acc = a0; acc < a2; acc = acc == a0 ? a1 : acc + da) {
    ps.GetScaled(acc, xs);
    xs += m_xs;
    ApplyA(xs, &m_drs);
    rs = m_bs;
    rs -= m_drs;
    const PCG_TYPE r = rs.SquaredLength();
    const PCG_TYPE F = xs.Dot(m_drs) * 0.5 - xs.Dot(m_bs);
    UT::Print("acc = %f, r = %e, F = %e\n", acc, r - r0, F - F0);
  }
  UT::DebugStop();
#endif
  ConvertCameraMotionResiduals(m_rs/*残差*/, m_ps/*p0(z0)*/, &Se2, &e2Max);//计算Se2 = r0.t * z0
#if 0
  const PCG_TYPE s = 1;
  //const PCG_TYPE s = 1 / Se2;
  const xp128f _s = xp128f::get(float(s));
  for (int ic = 0; ic < Nc; ++ic) {
    m_Mcs[ic] *= _s;
  }
  for (int im = 0; im < Nm; ++im) {
    m_MmsLM[im] *= _s;
  }
  m_ps *= s;
  Se2 *= s;
  e2Max *= s;
#endif
#ifdef CFG_DEBUG
  UT_ASSERT(Se2 >= 0 && e2Max >= 0);
#endif
//#ifdef CFG_DEBUG
#if 0
  if (m_iIter == 0) {
    UT::DebugStart();
  }
#endif
  ApplyA(m_ps, &m_drs);//m_drs = A*p0
  alpha = Se2 / m_ps.Dot(m_drs);//r0z0 / p0.dot(A*p0) 这个是第1步的步长
//#ifdef CFG_DEBUG
#if 0
  //const std::string dir = m_dir + "pcg/";
  const std::string dir = "D:/tmp/pcg/";
#if 0
#ifdef WIN32
  m_bs.AssertEqual(UT::String("%sb_%02d.txt", dir.c_str(), m_iIter), 2, "", -1.0f, -1.0f);
  m_ps.AssertEqual(UT::String("%sp_%02d.txt", dir.c_str(), m_iIter), 2, "", -1.0f, -1.0f);
  m_drs.AssertEqual(UT::String("%sAp_%02d.txt", dir.c_str(), m_iIter), 2, "", -1.0f, -1.0f);
#else
  m_bs.SaveB(UT::String("%sb_%02d.txt", dir.c_str(), m_iIter));
  m_ps.SaveB(UT::String("%sp_%02d.txt", dir.c_str(), m_iIter));
  m_drs.SaveB(UT::String("%sAp_%02d.txt", dir.c_str(), m_iIter));
#endif
#endif
#endif
#ifdef _MSC_VER
  if (_finite(alpha)) {
#else
  if (std::isfinite(alpha)) {
#endif  // _MSC_VER
    const PCG_TYPE e2MaxConv[2] = {ME::ChiSquareDistance<3>(BA_PCG_MIN_CONVERGE_PROBABILITY,
                                                            BA_WEIGHT_FEATURE)/* * s*/,
                                   ME::ChiSquareDistance<3>(BA_PCG_MAX_CONVERGE_PROBABILITY,
                                                            BA_WEIGHT_FEATURE)/* * s*/};
    //////////////////////////////////////////////////////////////////////////
    Se2Min = Se2;
    e2MaxMin = e2Max;
    //////////////////////////////////////////////////////////////////////////
#ifdef CFG_VERBOSE
#ifdef GBA_DEBUG_PCG_SAVE_RESIDUAL
    const std::string fileName = "D:/tmp/pcg.txt";
    FILE *fp = fopen(fileName.c_str(), "w");
    SolveSchurComplementGT(m_Cs, m_CsLM, &m_xsGT);
#endif
    if (m_verbose >= 3) {
      m_dxs = m_bs;
      m_dxs += m_rs;
      F = FMin = m_dxs.Dot(m_xs) * -0.5f;
      UT::PrintSeparator();
      UT::Print("*%2d: [GlobalBundleAdjustor::SolveSchurComplement]\n", m_iIter);
      UT::Print("  *%2d: F = %e, rTz = %e, acc = %f\n", 0, F, Se2/* / s*/, alpha/* * s*/);
#ifdef GBA_DEBUG_PCG_SAVE_RESIDUAL
      const Residual R = ComputeResidual(m_xs, true);
      LA::AlignedVectorXf::ApB(m_xsGT, m_xs, m_dxs);
      ConvertCameraUpdates(m_dxs.Data(), &m_xp2s, &m_xr2s);
      ConvertMotionUpdates(m_dxs.Data() + Ncp, &m_xv2s, &m_xba2s, &m_xbw2s);
      const PCG_TYPE ep = sqrtf(m_xp2s.Mean());
      const PCG_TYPE er = sqrtf(m_xr2s.Mean()) * UT_FACTOR_RAD_TO_DEG;
      const PCG_TYPE ev = sqrtf(m_xv2s.Mean());
      const PCG_TYPE eba = sqrtf(m_xba2s.Mean());
      const PCG_TYPE ebw = sqrtf(m_xbw2s.Mean()) * UT_FACTOR_RAD_TO_DEG;
      fprintf(fp, "%e %e %e %e %e %e %e %e\n", R.m_r2, R.m_F, Se2 / s, ep, er, ev, eba, ebw);
#endif
    }
#endif
    m_drs *= alpha;//alpha0 * A*pk-1
    m_rs -= m_drs;//r1 = r0 - alpha0 * A*p0 //更新第二步的残差
#ifdef CFG_INCREMENTAL_PCG
    m_ps.GetScaled(alpha, m_dxs);//m_dxs = p0 * alpha0
    m_xs += m_dxs;//x1 = x0 + p0 * alpha0  //更新第二步的x
#if defined GBA_DEBUG_PCG_SAVE_RESULT || defined GBA_DEBUG_PCG_LOAD_RESULT
    const std::string xFileName = "D:/tmp/pcg/x0000.txt";
#endif
#ifdef GBA_DEBUG_PCG_SAVE_RESULT
    m_xs.SaveB(xFileName);
#endif
#ifdef GBA_DEBUG_PCG_LOAD_RESULT
    m_xs.LoadB(xFileName);
#endif
#else
    m_ps.GetScaled(alpha, m_xs);
#endif
#ifdef CFG_INCREMENTAL_PCG
    Se2Pre = Se2;
    const PCG_TYPE Se2ConvMin = Se2Pre * BA_PCG_MIN_CONVERGE_RESIDUAL_RATIO;
    const PCG_TYPE Se2ConvMax = Se2Pre * BA_PCG_MAX_CONVERGE_RESIDUAL_RATIO;
#else
    const PCG_TYPE Se2ConvMin = Se2 * BA_PCG_MIN_CONVERGE_RESIDUAL_RATIO;
    const PCG_TYPE Se2ConvMax = Se2 * BA_PCG_MAX_CONVERGE_RESIDUAL_RATIO;
#endif
    const int nIters = std::min(N, BA_PCG_MAX_ITERATIONS);//最大迭代次数
    for (m_iIterPCG = 0; m_iIterPCG < nIters; ++m_iIterPCG) {
      ApplyM(m_rs, &m_zs);//zk = M^-1*rk 更新梯度方向
      Se2Pre = Se2;
      ConvertCameraMotionResiduals(m_rs, m_zs, &Se2, &e2Max);//计算Se2=e2Max = rk.t * zk
#ifdef CFG_DEBUG
      UT_ASSERT(Se2 >= 0 && e2Max >= 0);
#endif
      //////////////////////////////////////////////////////////////////////////
      if (Se2 < Se2Min) {//如果小于最小残差^2,就更新一下
        Se2Min = Se2;
        e2MaxMin = e2Max;//e2Max 好像是等于Se2的
        m_xsGN = m_xs;//更新x
      }
      //////////////////////////////////////////////////////////////////////////
//#if 0
#if 1
      beta = Se2 / Se2Pre;//用来算下一次下降方向的比例系数
#else
      beta = -m_zs.Dot(m_drs) / Se2Pre;
#endif
#ifdef CFG_VERBOSE
      if (m_verbose >= 3) {
        m_dxs = m_bs;
        m_dxs += m_rs;
        F = m_dxs.Dot(m_xs) * -0.5f;
        UT::Print("  *%2d: F = %e, rTz = %e, acc = %f, b = %f", m_iIterPCG + 1, F, Se2/* / s*/, alpha/* * s*/, beta);
        if (F < FMin) {
          UT::Print("*");
          FMin = F;
        }
        UT::Print("\n");
#ifdef GBA_DEBUG_PCG_SAVE_RESIDUAL
        const Residual R = ComputeResidual(m_xs, true);
        LA::AlignedVectorXf::ApB(m_xsGT, m_xs, m_dxs);
        ConvertCameraUpdates(m_dxs.Data(), &m_xp2s, &m_xr2s);
        ConvertMotionUpdates(m_dxs.Data() + Ncp, &m_xv2s, &m_xba2s, &m_xbw2s);
        const PCG_TYPE ep = sqrtf(m_xp2s.Mean());
        const PCG_TYPE er = sqrtf(m_xr2s.Mean()) * UT_FACTOR_RAD_TO_DEG;
        const PCG_TYPE ev = sqrtf(m_xv2s.Mean());
        const PCG_TYPE eba = sqrtf(m_xba2s.Mean());
        const PCG_TYPE ebw = sqrtf(m_xbw2s.Mean()) * UT_FACTOR_RAD_TO_DEG;
        fprintf(fp, "%e %e %e %e %e %e %e %e\n", R.m_r2, R.m_F, Se2/* / s*/, ep, er, ev, eba, ebw);
#endif
      }
#endif
      //////////////////////////////////////////////////////////////////////////
      //if (cnt == BA_PCG_MIN_ITERATIONS) {
      //  scc = true;
      //  break;
      //}
      //if (Se2Min <= Se2ConvMin && m_iIterPCG >= BA_PCG_MIN_ITERATIONS) {
      //  scc = true;
      //  break;
      //}
      //////////////////////////////////////////////////////////////////////////
//#ifdef CFG_DEBUG
#if 0
      ApplyA(m_xs, &m_drs);
      m_drs -= m_bs;
      m_drs.MakeMinus();
      m_drs -= m_rs;
#endif
      const int i = (Se2Min <= Se2ConvMin && m_iIterPCG >= BA_PCG_MIN_ITERATIONS) ? 0 : 1;//如果残差小于阈值或者迭代次数超过最小迭代次数,那么i就设成1
      if (Se2 == 0.0f || e2MaxMin < e2MaxConv[i]) {//残差^2==0或者小于残差^2可接受的最大残差^2就认为迭代收敛
        scc = true;
        break;
      } else if (Se2 > Se2ConvMax) {//如果超过最大最大残差,迭代失败
        scc = false;
        break;
      }
      //////////////////////////////////////////////////////////////////////////
      //if (Se2 < Se2Pre) {
      //  m_ps *= beta;
      //  m_ps += m_zs;
      //} else {
      //  //m_ps = m_zs;
      //  break;
      //}
      //////////////////////////////////////////////////////////////////////////
      m_ps *= beta;// pk-1*beta
      m_ps += m_zs;// pk = zk + pk-1*beta //更新当前步骤下降方向
      ApplyA(m_ps, &m_drs);//m_drs = A*pk
      alpha = Se2 / m_ps.Dot(m_drs);//alphak = (rk.t * zk)/(pk.t * A*pk) 更新当前步骤
#ifdef _MSC_VER
      if (!_finite(alpha)) {
#else
      if (!std::isfinite(alpha)) {//步长出了问题也退出
#endif  // _MSC_VER
        scc = false;
        break;
      }
      m_drs *= alpha;//m_drs = alphak * A*pk
      m_rs -= m_drs;//rk+1 = rk - alphak * A*pk //更新下一步的残差
      m_ps.GetScaled(alpha, m_dxs);//m_dxs = pk * alphak
      m_xs += m_dxs;//xk+1 = xk + pk * alphak  //更新下一步的x
//#ifdef CFG_DEBUG
#if 0
      if (m_iIter == 3) {
        ConvertCameraUpdates(m_xs.Data(), &m_xp2s, &m_xr2s);
        UT::Print("%d %f\n", m_iIterPCG, sqrtf(m_xr2s.Mean()) * UT_FACTOR_RAD_TO_DEG);
      }
#endif
#ifdef GBA_DEBUG_PCG_SAVE_RESULT
      m_xs.SaveB(UT::FileNameIncreaseSuffix(xFileName, m_iIterPCG + 1));
#endif
#ifdef GBA_DEBUG_PCG_LOAD_RESULT
      m_xs.LoadB(UT::FileNameIncreaseSuffix(xFileName, m_iIterPCG + 1));
#endif
      //////////////////////////////////////////////////////////////////////////
      //ApplyA(m_xs, &m_drs);
      //m_rs = m_bs;
      //m_rs -= m_drs;
      //////////////////////////////////////////////////////////////////////////
//#ifdef CFG_DEBUG
#if 0
      if (m_iIterPCG == 0) {
        m_bs.AssertEqual(UT::String("%sb_%02d.txt", dir.c_str(), m_iIter), 2, "", -1.0f, -1.0f);
      }
      m_xs.AssertEqual(UT::String("%sx_%02d_%02d.txt", dir.c_str(), m_iIter, m_iIterPCG), 2, "", -1.0f, -1.0f);
      m_rs.AssertEqual(UT::String("%sr_%02d_%02d.txt", dir.c_str(), m_iIter, m_iIterPCG), 2, "", -1.0f, -1.0f);
      m_ps.AssertEqual(UT::String("%sp_%02d_%02d.txt", dir.c_str(), m_iIter, m_iIterPCG), 2, "", -1.0f, -1.0f);
      m_zs.AssertEqual(UT::String("%sz_%02d_%02d.txt", dir.c_str(), m_iIter, m_iIterPCG), 2, "", -1.0f, -1.0f);
#endif
    }
#ifdef GBA_DEBUG_PCG_SAVE_RESIDUAL
    fclose(fp);
    UT::PrintSaved(fileName);
#endif
  } else {
    m_iIterPCG = 0;
  }
//#ifdef CFG_DEBUG
#if 0
  const Residual R = ComputeResidual(m_xsGN, true);
#endif
  //////////////////////////////////////////////////////////////////////////
  //m_xsGN = m_xs;
  //////////////////////////////////////////////////////////////////////////
  m_xsGN.MakeMinus();//因为前面求得都是A*x=b的x但是实际上对应的是-x,所以这里变回实际增量
  ConvertCameraUpdates(m_xsGN.Data(), &m_xp2s/*位置部分优化变量增量的^2*/, &m_xr2s/*朝向部分优化变量增量的^2*/);//位置和朝向部分优化变量增量的^2
  ConvertMotionUpdates(m_xsGN.Data() + Ncp, &m_xv2s, &m_xba2s, &m_xbw2s);//同上
//#ifdef CFG_DEBUG
#if 0
#ifdef WIN32
  m_xsGN.AssertEqual(UT::String("%sx_%02d.txt", dir.c_str(), m_iIter), 2, "", -1.0f, -1.0f);
#else
  m_xsGN.SaveB(UT::String("%sx_%02d.txt", dir.c_str(), m_iIter));
#endif
#endif
  return scc;
}

void GlobalBundleAdjustor::SolveSchurComplementGT(const AlignedVector<Rigid3D> &Cs,
                                                  const AlignedVector<Camera> &CsLM,
                                                  LA::AlignedVectorXf *xs, const bool motion) {
  if (!m_CsGT) {
    return;
  }
#ifdef GBA_DEBUG_GROUND_TRUTH_STATE_ERROR
  const float dpMax = 0.01f;
  const float drMax = 0.1f;
  const float dvMax = 0.1f;
  const float dbaMax = 0.1f;
  const float dbwMax = 0.1f;
#endif
  const int pc = 6, pm = 9;
  const int Nc = m_Cs.Size(), Nm = motion ? m_CsLM.Size() : 0;
  xs->Resize(Nc * pc + Nm * pm);
  Rotation3D dR;
  Point3D p, pGT;
  LA::AlignedVector3f dr, dp;
  LA::Vector6f *xcs = (LA::Vector6f *) xs->Data();
  for (int ic = 0; ic < Nc; ++ic) {
    const Rigid3D &C = Cs[ic], &CGT = m_CsKFGT[ic];
    Rotation3D::ATB(C, CGT, dR);
    dR.GetRodrigues(dr, BA_ANGLE_EPSILON);
    C.GetPosition(p);
    CGT.GetPosition(pGT);
    LA::AlignedVector3f::amb(pGT, p, dp);
#ifdef GBA_DEBUG_GROUND_TRUTH_STATE_ERROR
    dp += LA::AlignedVector3f::GetRandom(dpMax);
    dr += LA::AlignedVector3f::GetRandom(drMax * UT_FACTOR_DEG_TO_RAD);
#ifdef CFG_DEBUG
    dr.z() = 0.0f;
#endif
#endif
    xcs[ic].Set(dp, dr);
  }
  ConvertCameraUpdates(xs->Data(), &m_xp2s, &m_xr2s);
  if (motion) {
    LA::AlignedVector3f dv, dba, dbw;
    LA::Vector9f *xms = (LA::Vector9f *) (xcs + Nc);
    for (int im = 0; im < Nm; ++im) {
      const Camera &C = CsLM[im], &CGT = m_CsLMGT[im];
      LA::Vector9f &xm = xms[im];
      if (CGT.m_v.Valid()) {
        LA::AlignedVector3f::amb(CGT.m_v, C.m_v, dv);
#ifdef GBA_DEBUG_GROUND_TRUTH_STATE_ERROR
        dv += LA::AlignedVector3f::GetRandom(dvMax);
#endif
        xm.Set012(dv);
      }
      if (CGT.m_ba.Valid()) {
        LA::AlignedVector3f::amb(CGT.m_ba, C.m_ba, dba);
#ifdef GBA_DEBUG_GROUND_TRUTH_STATE_ERROR
        dba += LA::AlignedVector3f::GetRandom(dbaMax);
#endif
        xm.Set345(dba);
      }
      if (CGT.m_bw.Valid()) {
        LA::AlignedVector3f::amb(CGT.m_bw, C.m_bw, dbw);
#ifdef GBA_DEBUG_GROUND_TRUTH_STATE_ERROR
        dbw += LA::AlignedVector3f::GetRandom(dbwMax * UT_FACTOR_DEG_TO_RAD);
#endif
        xm.Set678(dbw);
      }
    }
  }
}

#ifdef GBA_DEBUG_EIGEN_PCG
static EigenMatrixXd g_A/*, g_S, g_SI*/, g_M, g_L, g_D, g_LT;
static EigenVectorXd g_z1, g_z2, g_r;
static std::vector<EigenMatrixXd> g_Ms;
#endif

void GlobalBundleAdjustor::PrepareConditioner() {
  const int pc = 6, pm = 9;
  //const float eps = 0.0f;
  const float eps = FLT_EPSILON;
  const float epsr = UT::Inverse(BA_VARIANCE_MAX_ROTATION, BA_WEIGHT_FEATURE, eps);
  const float epsp = UT::Inverse(BA_VARIANCE_MAX_POSITION, BA_WEIGHT_FEATURE, eps);
  const float epsv = UT::Inverse(BA_VARIANCE_MAX_VELOCITY, BA_WEIGHT_FEATURE, eps);
  const float epsba = UT::Inverse(BA_VARIANCE_MAX_BIAS_ACCELERATION, BA_WEIGHT_FEATURE, eps);
  const float epsbw = UT::Inverse(BA_VARIANCE_MAX_BIAS_GYROSCOPE, BA_WEIGHT_FEATURE, eps);
  const float epsc[pc] = {epsp, epsp, epsp, epsr, epsr, epsr};
  const float epsm[pm] = {epsv, epsv, epsv, epsba, epsba, epsba, epsbw, epsbw, epsbw};
  const int Nb = GBA_PCG_CONDITIONER_BAND, Nc = m_Cs.Size(), Nm = m_CsLM.Size();
  if (Nb <= 1) {
    m_Mcs.Resize(Nc);
    for (int ic = 0; ic < Nc; ++ic) {
//#ifdef CFG_DEBUG
#if 0
      if (m_iIter == 3 && ic == 136) {
        UT::DebugStart();
        LA::AlignedMatrix3x3f A, B, M1, M2, I1, I2;
        LA::SymmetricMatrix3x3d Cam_state, M3;
        m_SAcus[ic].m_A.Get00(&A);
        //const float s = 1.0e-7f;
        const float s = 1.0e-5f;
        //const float s = 1.0e-2f;
        //const float s = 1.0f;
        A.GetScaled(s, B);
        B.GetInverse(M1);
        B.GetInverseLDL(M2);
        M1 *= s;
        M2 *= s;
        Cam_state.Set(B);
        Cam_state.GetInverse(M3);
        M3 *= s;
        LA::AlignedMatrix3x3f::AB(A, M1, I1);
        LA::AlignedMatrix3x3f::AB(A, M2, I2);
        A.Print(" A = ", false);
        B.Print(" B = ", false);
        M1.Print("M1 = ", false);
        M2.Print("M2 = ", false);
        M3.Print("M3 = ", false);
        I1.Print("I1 = ", false);
        I2.Print("I2 = ", false);
        UT::DebugStop();
      }
#endif
      m_Mcs[ic].Set(m_SAcus[ic]/*关键帧pose自己和自己的H以及自己对应的-b*/, BA_PCG_CONDITIONER_MAX, BA_PCG_CONDITIONER_EPSILON, epsc);
      //m_Mcs[ic].Set(m_Acus[ic], BA_PCG_CONDITIONER_MAX, BA_PCG_CONDITIONER_EPSILON, epsc);
    }
    m_MmsLM.Resize(Nm);
    for (int im = 0; im < Nm; ++im) {
      m_MmsLM[im].Set(m_AmusLM[im], BA_PCG_CONDITIONER_MAX, BA_PCG_CONDITIONER_EPSILON, epsm);
    }
    return;
  }
  //m_ss.Resize(Nc * pc + Nm * pm);
  //LA::Vector6f *scs = (LA::Vector6f *) m_ss.Data();
  //LA::Vector9f *sms = (LA::Vector9f *) (scs + Nc);
  //for (int ic = 0; ic < Nc; ++ic) {
  //  m_Acus[ic].GetDiagonal(scs[ic]);
  //}
  //for (int im = 0; im < Nm; ++im) {
  //  m_AmusLM[im].GetDiagonal(sms[im]);
  //}
  //m_ss.MakeSquareRoot();
  //m_ss.MakeInverse();
#if 0
//#if 1
  UT::DebugStart();
  m_ss.Set(1.0f);
  UT::DebugStop();
#endif
  //LA::ProductVector6f sc;
  //LA::AlignedVector9f sm;
  m_Mcc.Resize(Nc, Nb);       m_MccT.Resize(Nc, Nb);
  m_McmLM.Resize(Nm, 2);      m_McmTLM.Resize(Nm, 2);
  m_MmcLM.Resize(Nm, Nb - 1); m_MmcTLM.Resize(Nm, Nb - 1);
  m_MmmLM.Resize(Nm, 2);      m_MmmTLM.Resize(Nm, 2);
  m_Mcc.MakeZero();
  for (int ic = 0; ic < Nc; ++ic) {//相机部分
    m_Mcc[ic][0] = m_Acus[ic];//舒尔补以后S矩阵pq对应的对角线部分
    //sc.Set(scs[ic]);
    //m_Acus[ic].GetScaledColumn(sc, m_Mcc[ic][0]);
    //m_Mcc[ic][0].ScaleRow(scs[ic]);
    const LA::AlignedMatrix6x6f *Acbs = m_Acbs.Data() + m_iKF2cb[ic];
    const KeyFrame &KF = m_KFs[ic];
    const int Nkp = static_cast<int>(KF.m_ikp2KF.size());
    for (int ikp = 0; ikp < Nkp; ++ikp) {
      const int _ic = KF.m_ikp2KF[ikp], ib = ic - _ic;
      if (ib >= Nb) {
        continue;
      }
      m_Mcc[_ic][ib] = Acbs[ikp];
      //Acbs[ikp].GetScaledColumn(sc, m_Mcc[_ic][ib]);
      //m_Mcc[_ic][ib].ScaleRow(scs[_ic]);
    }
  }
  for (int ic = Nc - Nm, im = 0; ic < Nc; ++ic, ++im) {//遍历非对角线部分
    m_McmLM[im][0] = m_SAcmsLM[im].m_Au.m_Acm;//舒尔补以后S矩阵pose和motion对应的非对角部分
    m_MmmLM[im][0] = m_AmusLM[im];//舒尔补以后S矩阵motion对应的对角线部分
    //if (im == 0) {
    //  sm.Set(sms[im]);
    //}
    //m_SAcmsLM[im].m_Au.m_Acm.GetScaledColumn(sm, m_McmLM[im][0]);
    //m_McmLM[im][0].ScaleRow(scs[ic]);
    //m_AmusLM[im].GetScaledColumn(sm, m_MmmLM[im][0]);
    //m_MmmLM[im][0].ScaleRow(sms[im]);
    const int _im = im + 1;
    if (_im == Nm) {
      continue;
    }
    const Camera::Factor::Binary &Ab = m_SAcmsLM[_im].m_Ab;
    m_McmLM[im][1] = Ab.m_Acm;
    m_MmmLM[im][1] = Ab.m_Amm;
    //sm.Set(sms[_im]);
    //Ab.m_Acm.GetScaledColumn(sm, m_McmLM[im][1]);
    //m_McmLM[im][1].ScaleRow(scs[ic]);
    //Ab.m_Amm.GetScaledColumn(sm, m_MmmLM[im][1]);
    //m_MmmLM[im][1].ScaleRow(sms[im]);
    
    LA::AlignedMatrix9x6f *Amcs = m_MmcLM[im];
    Amcs[0] = Ab.m_Amc;
    //Ab.m_Amc.GetScaledColumn(scs[ic + 1], Amcs[0]);
    //Amcs[0].ScaleRow(sms[im]);
    const int Nbm = (im + Nb > Nm ? Nm - im : Nb) - 1;
    for (int ib = 1; ib < Nbm; ++ib) {
      Amcs[ib].MakeZero();
    }
  }
#ifdef GBA_DEBUG_EIGEN_PCG
  const double e_epsc[pc] = {epsp, epsp, epsp, epsr, epsr, epsr};
  const double e_epsm[pm] = {epsv, epsv, epsv, epsba, epsba, epsba, epsbw, epsbw, epsbw};
  const int pcm = pc + pm, N = Nc * (pc + pm);
  g_A.resize(N, N);
  g_A.setZero();
  for (int ic = 0, icp = 0; ic < Nc; ++ic, icp += pcm) {
    g_A.block<pc, pc>(icp, icp) = EigenMatrix6x6f(m_Acus[ic]).cast<double>();
    const KeyFrame &KF = m_KFs[ic];
    const LA::AlignedMatrix6x6f *Acbs = m_Acbs.Data() + KF.m_iKp;
    const int Nkp = int(KF.m_ikp2KF.size());
    for (int ikp = 0; ikp < Nkp; ++ikp) {
      const int _ic = KF.m_ikp2KF[ikp];
      if (ic < _ic + Nb) {
        g_A.block<pc, pc>(_ic * pcm, icp) = EigenMatrix6x6f(Acbs[ikp]).cast<double>();
      }
    }
  }
  for (int ic = Nc - Nm, icp = ic * pcm, im = 0, imp = icp + pc; ic < Nc;
       ++ic, icp += pcm, ++im, imp += pcm) {
    const Camera::EigenFactor e_Acm = m_SAcmsLM[im];
    g_A.block<pm, pm>(imp, imp) = EigenMatrix9x9f(m_AmusLM[im]).cast<double>();
    g_A.block<pc, pm>(icp, imp) = e_Acm.m_Au.m_Acm.cast<double>();
    if (im > 0) {
      const int _icp = icp - pcm, _imp = imp - pcm;
      g_A.block<pc, pm>(_icp, imp) = e_Acm.m_Ab.m_Acm.cast<double>();
      g_A.block<pm, pc>(_imp, icp) = e_Acm.m_Ab.m_Amc.cast<double>();
      g_A.block<pm, pm>(_imp, imp) = e_Acm.m_Ab.m_Amm.cast<double>();
    }
  }
  g_A.SetLowerFromUpper();
  g_M = g_A;
  //g_S.Resize(N, N);   g_S.MakeZero();
  //g_SI.Resize(N, N);  g_SI.MakeZero();
  //for (int ic = 0, im = Nm - Nc, icp = 0, imp = pc; ic < Nc; ++ic, ++im, icp += pcm, imp += pcm) {
  //  const float *sc = scs[ic];
  //  for (int ip = 0, jcp = icp; ip < pc; ++ip, ++jcp) {
  //    g_S(jcp, jcp) = sc[ip];
  //    g_SI(jcp, jcp) = 1 / sc[ip];
  //  }
  //  if (im < 0) {
  //    continue;
  //  }
  //  const float *sm = sms[im];
  //  for (int ip = 0, jmp = imp; ip < pm; ++ip, ++jmp) {
  //    g_S(jmp, jmp) = sm[ip];
  //    g_SI(jmp, jmp) = 1 / sm[ip];
  //  }
  //}
  //g_M = EigenMatrixXd(g_S * g_A * g_S);
  g_Ms.resize(0);
#endif

  AlignedVector<LA::AlignedMatrix6x6f> AccsT;
  AlignedVector<LA::AlignedMatrix9x6f> AcmsTLM;
  AlignedVector<LA::AlignedMatrix6x9f> AmcsTLM;
  AlignedVector<LA::AlignedMatrix9x9f> AmmsTLM;
  m_work.Resize((AccsT.BindSize(Nb) + AcmsTLM.BindSize(2) +
                 AmcsTLM.BindSize(Nb) + AmmsTLM.BindSize(2)) / sizeof(float));
  AccsT.Bind(m_work.Data(), Nb);
  AcmsTLM.Bind(AccsT.BindNext(), 2);
  AmcsTLM.Bind(AcmsTLM.BindNext(), Nb);
  AmmsTLM.Bind(AmcsTLM.BindNext(), Nb);
  for (int ic = 0, im = Nm - Nc; ic < Nc; ++ic, ++im) {
    LA::AlignedMatrix6x6f *Mccs/*对角线pose部分*/ = m_Mcc[ic], *MccsT = m_MccT[ic];
    LA::AlignedMatrix6x9f *McmsLM/*非对角线pose，motion部分*/ = im >= 0 ? m_McmLM[im] : NULL;
    LA::AlignedMatrix9x6f *McmsTLM = im >= 0 ? m_McmTLM[im] : NULL;
    LA::AlignedMatrix9x6f *MmcsLM = im >= 0 ? m_MmcLM[im] : NULL;
    LA::AlignedMatrix6x9f *MmcsTLM = im >= 0 ? m_MmcTLM[im] : NULL;
    LA::AlignedMatrix9x9f *MmmsLM/*对角线motion部分*/ = im >= 0 ? m_MmmLM[im] : NULL;
    LA::AlignedMatrix9x9f *MmmsTLM = im >= 0 ? m_MmmTLM[im] : NULL;
    const int Nbcc = ic + Nb > Nc ? Nc - ic : Nb;
    const int Nbcm = im >= 0 ? (ic + 1 == Nc ? 1 : 2) : 0;
    const int Nbmc = im >= 0 ? Nbcc - 1 : 0;
    const int Nbmm = Nbcm;
#ifdef GBA_DEBUG_EIGEN_PCG
//#if 0
    const int icp = ic * pcm, imp = icp + pc;
    g_M.Marginalize(icp, pc, e_epsc, false, false);
//#ifdef CFG_DEBUG
#if 0
    const EigenMatrixXd e_Aii = EigenMatrixXd(g_M.block<pm, pm>(imp, imp));
    const EigenMatrixXd e_Mii = e_Aii.GetInverseLDL(e_epsm);
    const int imChk = 4;
    //if (im == imChk) {
    //  UT::PrintSeparator();
    //  EigenMatrix9x9f(e_Aii.cast<float>()).Print(true);
    //}
#endif
    g_M.Marginalize(imp, pm, e_epsm, false, false);
    g_Ms.push_back(g_M);
//#ifdef CFG_DEBUG
#if 0
    if (UT::Debugging()) {
      UT::PrintSeparator();
      EigenMatrix9x9f(g_M.block<pm, pm>(imp, imp).cast<float>()).Print(true);
    }
#endif
#endif
    LA::AlignedMatrix6x6f &Mcc = Mccs[0];


    if (Mcc.InverseLDL(epsc)) {//Mcc,ldl求逆
      MccsT[0] = Mcc;
      Mcc.MakeMinus();
      for (int ib = 1; ib < Nbcc; ++ib) {
        Mccs[ib].GetTranspose(AccsT[ib]);
        LA::AlignedMatrix6x6f::ABT(Mcc, AccsT[ib], Mccs[ib]);
        Mccs[ib].GetTranspose(MccsT[ib]);
      }
      for (int ib = 0; ib < Nbcm; ++ib) {
        McmsLM[ib].GetTranspose(AcmsTLM[ib]);
        LA::AlignedMatrix9x6f::ABT(Mcc, AcmsTLM[ib], McmsLM[ib]);
        McmsLM[ib].GetTranspose(McmsTLM[ib]);
      }
      for (int ib = 1; ib < Nbcc; ++ib) {
        const LA::AlignedMatrix6x6f &MccT = MccsT[ib];
        const int _ic = ic + ib;
        LA::AlignedMatrix6x6f *_Mccs = m_Mcc[_ic] - ib;
        LA::AlignedMatrix6x6f::AddABTToUpper(MccT, AccsT[ib], _Mccs[ib]);
#ifdef GBA_DEBUG_EIGEN_PCG
        _Mccs[ib].SetLowerFromUpper();
#endif
        for (int jb = ib + 1; jb < Nbcc; ++jb) {
          LA::AlignedMatrix6x6f::AddABTTo(MccT, AccsT[jb], _Mccs[jb]);
        }
        if (ib == 1 && im >= 0) {
          LA::AlignedMatrix9x6f::AddABTTo(MccT, AcmsTLM[ib], m_McmLM[im + ib][0]);
        }
      }
      for (int ib = 0; ib < Nbcm; ++ib) {
        const LA::AlignedMatrix9x6f &McmT = McmsTLM[ib];
        const int _im = im + ib;
        const LA::AlignedMatrix6x6f *_AccsT = AccsT.Data() + 1;
        LA::AlignedMatrix9x6f *_MmcsLM = m_MmcLM[_im] - ib;
        for (int jb = ib; jb < Nbmc; ++jb) {
          LA::AlignedMatrix9x6f::AddABTTo(McmT, _AccsT[jb], _MmcsLM[jb]);
        }
        LA::AlignedMatrix9x9f *_MmmsLM = m_MmmLM[_im];
//#ifdef CFG_DEBUG
#if 0
        if (im == imChk && _im == imChk) {
          LA::AlignedMatrix9x9f T;
          LA::AlignedMatrix9x9f::ABT(McmT, AcmsTLM[ib], T);
          UT::Print("%e + %e = %f\n", _MmmsLM[0][0][3], T[0][3], _MmmsLM[0][0][3] + T[0][3]);
        }
#endif
        LA::AlignedMatrix9x9f::AddABTToUpper(McmT, AcmsTLM[ib], _MmmsLM[0]);
#ifdef GBA_DEBUG_EIGEN_PCG
        _MmmsLM[0].SetLowerFromUpper();
#endif
//#ifdef CFG_DEBUG
#if 0
        if (im == imChk && _im == imChk) {
          UT::PrintSeparator();
          _MmmsLM[0].Print(true);
        }
#endif
        if (ib == 0 && Nbcm == 2) {
          LA::AlignedMatrix9x9f::AddABTTo(McmT, AcmsTLM[1], _MmmsLM[1]);
        }
      }
    } else {//如果不能分解,就设置为0
      for (int ib = 0; ib < Nbcc; ++ib) {
        Mccs[ib].MakeZero();
        MccsT[ib].MakeZero();
      }
      for (int ib = 0; ib < Nbcm; ++ib) {
        McmsLM[ib].MakeZero();
        McmsTLM[ib].MakeZero();
      }
    }
    if (im < 0) {
      continue;
    }
    LA::AlignedMatrix9x9f &Mmm = MmmsLM[0];
//#ifdef CFG_DEBUG
#if 0
    if (im == imChk) {
      /*const */LA::AlignedMatrix9x9f Aii = Mmm;
      //EigenMatrix9x9f(e_Aii.cast<float>()).AssertEqual(Aii, 2, "", -1.0f, -1.0f);
      EigenMatrix9x9f(e_Aii.cast<float>()).Get(Aii);
      const LA::AlignedMatrix9x9f Mii = Aii.GetInverseLDL(epsm);
      //EigenMatrix9x9f(e_Mii.cast<float>()).AssertEqual(Mii, 2, "", -1.0f, -1.0f);
      UT::PrintSeparator();
      EigenMatrix9x9f((e_Aii * e_Mii).cast<float>()).Print(true);
      UT::PrintSeparator();
      EigenMatrix9x9f(EigenMatrix9x9f(Aii) * EigenMatrix9x9f(Mii)).Print(true);
    }
#endif
    if (Mmm.InverseLDL(epsm)) {
      MmmsTLM[0] = Mmm;
      Mmm.MakeMinus();
//#ifdef CFG_DEBUG
#if 0
      if (UT::Debugging()) {
        UT::PrintSeparator();
        Mmm.Print(true);
        UT::DebugStop();
      }
#endif
      for (int ib = 0; ib < Nbmc; ++ib) {
        MmcsLM[ib].GetTranspose(AmcsTLM[ib]);
        LA::AlignedMatrix9x9f::ABT(Mmm, AmcsTLM[ib], MmcsLM[ib]);
        MmcsLM[ib].GetTranspose(MmcsTLM[ib]);
      }
      if (Nbmm == 2) {
        MmmsLM[1].GetTranspose(AmmsTLM[1]);
        LA::AlignedMatrix9x9f::ABT(Mmm, AmmsTLM[1], MmmsLM[1]);
        MmmsLM[1].GetTranspose(MmmsTLM[1]);
      }
      for (int ib = 0; ib < Nbmc; ++ib) {
        const LA::AlignedMatrix6x9f &MmcT = MmcsTLM[ib];
        const int _ic = ic + ib + 1;
        LA::AlignedMatrix6x6f *_Mccs = m_Mcc[_ic] - ib;
        LA::AlignedMatrix6x9f::AddABTToUpper(MmcT, AmcsTLM[ib], _Mccs[ib]);
#ifdef GBA_DEBUG_EIGEN_PCG
        _Mccs[ib].SetLowerFromUpper();
#endif
        for (int jb = ib + 1; jb < Nbmc; ++jb) {
          LA::AlignedMatrix6x9f::AddABTTo(MmcT, AmcsTLM[jb], _Mccs[jb]);
        }
        if (ib == 0 && Nbmm == 2) {
          LA::AlignedMatrix9x9f::AddABTTo(MmcT, AmmsTLM[1], m_McmLM[im + ib + 1][0]);
        }
      }
      if (Nbmm == 2) {
        const LA::AlignedMatrix9x9f &MmmT = MmmsTLM[1];
        const int _im = im + 1;
        LA::AlignedMatrix9x6f *_Mmcs = m_MmcLM[_im] - 1;
        for (int jb = 1; jb < Nbmc; ++jb) {
          LA::AlignedMatrix9x9f::AddABTTo(MmmT, AmcsTLM[jb], _Mmcs[jb]);
        }
#if 0
        if (UT::Debugging()) {
          LA::AlignedMatrix9x9f T;
          LA::AlignedMatrix9x9f::ABT(MmmT, AmmsTLM[1], T);
          UT::Print("%f + %f = %f\n", m_MmmLM[_im][0][0][3], T[0][3], m_MmmLM[_im][0][0][3] + T[0][3]);
          UT::DebugStop();
        }
#endif
        LA::AlignedMatrix9x9f::AddABTToUpper(MmmT, AmmsTLM[1], m_MmmLM[_im][0]);
#ifdef GBA_DEBUG_EIGEN_PCG
        m_MmmLM[_im][0].SetLowerFromUpper();
#endif
      }
    } else {
      for (int ib = 0; ib < Nbmc; ++ib) {
        MmcsLM[ib].MakeZero();
        MmcsTLM[ib].MakeZero();
      }
      for (int ib = 0; ib < Nbmm; ++ib) {
        MmmsLM[ib].MakeZero();
        MmmsTLM[ib].MakeZero();
      }
    }
#ifdef GBA_DEBUG_EIGEN_PCG
//#if 0
    for (int ic1 = ic, im1 = im, icp1 = icp, imp1 = imp; ic1 < Nc;
         ++ic1, ++im1, icp1 += pcm, imp1 += pcm) {
      const int Nbcc1 = ic1 + Nb > Nc ? Nc - ic1 : Nb;
      const int Nbcm1 = im1 >= 0 ? (ic1 + 1 == Nc ? 1 : 2) : 0;
      const int Nbmc1 = im1 >= 0 ? Nbcc1 - 1 : 0;
      const int Nbmm1 = Nbcm1;
      for (int ib = 0, ic2 = ic1, icp2 = icp1; ib < Nbcc1; ++ib, ++ic2, icp2 += pcm) {
        const EigenMatrix6x6f e_Mcc = g_M.block<pc, pc>(icp1, icp2).cast<float>();
        //e_Mcc.AssertEqual(m_Mcc[ic1][ib], 1, UT::String("Mcc[%d][%d]", ic1, ic2));
        //g_M.block<pc, pc>(icp1, icp2) = EigenMatrix6x6f(m_Mcc[ic1][ib]).cast<double>();
        e_Mcc.Get(m_Mcc[ic1][ib]);
      }
      if (im1 >= 0) {
        for (int ib = 0, im2 = im1, imp2 = imp1; ib < Nbcm1; ++ib, ++im2, imp2 += pcm) {
          const EigenMatrix6x9f e_Mcm = g_M.block<pc, pm>(icp1, imp2).cast<float>();
          //e_Mcm.AssertEqual(m_McmLM[ic1][ib], 1, UT::String("Mcm[%d][%d]", ic1, im2));
          //g_M.block<pc, pm>(icp1, imp2) = EigenMatrix6x9f(m_McmLM[ic1][ib]).cast<double>();
          e_Mcm.Get(m_McmLM[ic1][ib]);
        }
        for (int ib = 0, ic2 = ic1 + 1, icp2 = icp1 + pcm; ib < Nbmc1; ++ib, ++ic2, icp2 += pcm) {
          const EigenMatrix9x6f e_Mmc = g_M.block<pm, pc>(imp1, icp2).cast<float>();
          //e_Mmc.AssertEqual(m_MmcLM[im1][ib], 1, UT::String("Mmc[%d][%d]", im1, ic2));
          //g_M.block<pm, pc>(imp1, icp2) = EigenMatrix9x6f(m_MmcLM[im1][ib]).cast<double>();
          e_Mmc.Get(m_MmcLM[im1][ib]);
        }
        for (int ib = 0, im2 = im1, imp2 = imp1; ib < Nbmm1; ++ib, ++im2, imp2 += pcm) {
          const EigenMatrix9x9f e_Mmm = g_M.block<pm, pm>(imp1, imp2).cast<float>();
          //e_Mmm.AssertEqual(m_MmmLM[im1][ib], 1, UT::String("Mmm[%d][%d]", im1, im2));
          //g_M.block<pm, pm>(imp1, imp2) = EigenMatrix9x9f(m_MmmLM[im1][ib]).cast<double>();
          e_Mmm.Get(m_MmmLM[im1][ib]);
        }
      }
    }
#endif
  }
#ifdef GBA_DEBUG_EIGEN_PCG
  g_M.SetLowerFromUpper();
  g_D.Resize(N, N);
  g_D.MakeZero();
  g_LT.Resize(N, N);
  g_LT.MakeZero();
  for (int ic = 0, icp = 0, imp = pc; ic < Nc; ++ic, icp += pcm, imp += pcm) {
    g_D.block<pc, pc>(icp, icp) = -g_M.block<pc, pc>(icp, icp).inverse();
    g_D.block<pm, pm>(imp, imp) = -g_M.block<pm, pm>(imp, imp).inverse();
    for (int ip = 0; ip < pc; ++ip) {
      g_LT(icp + ip, icp + ip) = -1.0f;
    }
    const int jcp = icp + pc;
    g_LT.block(icp, jcp, pc, N - jcp) = g_M.block(icp, jcp, pc, N - jcp);
    for (int ip = 0; ip < pm; ++ip) {
      g_LT(imp + ip, imp + ip) = -1.0f;
    }
    const int jmp = imp + pm;
    g_LT.block(imp, jmp, pm, N - jmp) = g_M.block(imp, jmp, pm, N - jmp);
  }
  g_L = EigenMatrixXd(g_LT.transpose());
  //const EigenMatrixXd e_A1 = EigenMatrixXd(g_S * g_A * g_S);
  const EigenMatrixXd &e_A1 = g_A;
  const EigenMatrixXd e_A2 = EigenMatrixXd(g_L * g_D * g_LT);
  for (int ic = 0, icp = 0; ic < Nc; ++ic, icp += pcm) {
    for (int jc = 0, jcp = 0; jc < Nc; ++jc, jcp += pcm) {
      const std::string str = UT::String("A[%d][%d]", ic, jc);
      const EigenMatrix15x15f e_A1ij(e_A1.block<pcm, pcm>(icp, jcp).cast<float>());
      const EigenMatrix15x15f e_A2ij(e_A2.block<pcm, pcm>(icp, jcp).cast<float>());
      e_A1ij.AssertEqual(e_A2ij, 1, str);
    }
  }

  EigenMatrix15x15f e_I;
  e_I.setIdentity();
  const EigenMatrixXd e_AI = EigenMatrixXd(g_A.inverse());
  const EigenMatrixXd e_I1 = EigenMatrixXd(g_A * e_AI), e_I2 = EigenMatrixXd(e_AI * g_A);
  for (int ic = 0, icp = 0; ic < Nc; ++ic, icp += pcm) {
    for (int jc = 0, jcp = 0; jc < Nc; ++jc, jcp += pcm) {
      const std::string str1 = UT::String("I1[%d][%d]", ic, jc);
      const std::string str2 = UT::String("I2[%d][%d]", ic, jc);
      if (ic == jc) {
        EigenMatrix15x15f(e_I1.block<pcm, pcm>(icp, jcp).cast<float>()).AssertEqual(e_I, 1, str1);
        EigenMatrix15x15f(e_I2.block<pcm, pcm>(icp, jcp).cast<float>()).AssertEqual(e_I, 1, str2);
      } else {
        EigenMatrix15x15f(e_I1.block<pcm, pcm>(icp, jcp).cast<float>()).AssertZero(1, str1);
        EigenMatrix15x15f(e_I2.block<pcm, pcm>(icp, jcp).cast<float>()).AssertZero(1, str2);
      }
    }
  }
  const float rMax = 1.0f;
  m_rs.Resize(N);
  m_rs.Random(rMax);
  EigenVectorXd e_r;
  e_r.Resize(N);
  const LA::Vector6f *rcs = (LA::Vector6f *) m_rs.Data();
  const LA::Vector9f *rms = (LA::Vector9f *) (rcs + Nc);
  for (int ic = 0, im = Nm - Nc, icp = 0, imp = pc; ic < Nc; ++ic, ++im, icp += pcm, imp += pcm) {
    e_r.block<pc, 1>(icp, 0) = EigenVector6f(rcs[ic]).cast<double>();
    if (im >= 0) {
      e_r.block<pm, 1>(imp, 0) = EigenVector9f(rms[im]).cast<double>();
    } else {
      e_r.block<pm, 1>(imp, 0).setZero();
    }
  }
  //g_z1 = EigenVectorXd(e_AI * e_r);
  g_z1 = EigenVectorXd(g_A.ldlt().solve(e_r));
  g_z2.Resize(N);
  ApplyM(m_rs, &m_zs);
  const LA::Vector6f *zcs = (LA::Vector6f *) m_zs.Data();
  const LA::Vector9f *zms = (LA::Vector9f *) (zcs + Nc);
  for (int ic = 0, im = Nm - Nc, icp = 0, imp = pc; ic < Nc; ++ic, ++im, icp += pcm, imp += pcm) {
    const EigenVector6f e_zc1 = EigenVector6f(g_z1.block<pc, 1>(icp, 0).cast<float>());
    const EigenVector6f e_zc2 = EigenVector6f(zcs[ic]);
    e_zc1.AssertEqual(e_zc2, 1, UT::String("zc[%d]", ic));
    const EigenVector9f e_zm1 = EigenVector9f(g_z1.block<pm, 1>(imp, 0).cast<float>());
    EigenVector9f e_zm2;
    if (im >= 0) {
      e_zm2 = EigenVector9f(zms[im]);
      e_zm1.AssertEqual(e_zm2, 1, UT::String("zm[%d]", im));
    } else {
      e_zm2.setZero();
    }
    g_z2.block<pc, 1>(icp, 0) = e_zc2.cast<double>();
    g_z2.block<pm, 1>(imp, 0) = e_zm2.cast<double>();
  }
  const EigenVectorXd e_Az1 = EigenVectorXd(g_A * g_z1), e_e1 = EigenVectorXd(e_Az1 - e_r);
  const EigenVectorXd e_Az2 = EigenVectorXd(g_A * g_z2), e_e2 = EigenVectorXd(e_Az2 - e_r);
  UT::Print("%e vs %e\n", e_e1.norm(), e_e2.norm());
#endif
}
//z0 = M.inv * r0 m_Mcs,m_Mms是M^-1，即S的对角线的逆
void GlobalBundleAdjustor::ApplyM(const LA::AlignedVectorX<PCG_TYPE> &xs,/*残差*/
                                  LA::AlignedVectorX<PCG_TYPE> *Mxs) {
  const int Nb = GBA_PCG_CONDITIONER_BAND, Nc = m_Cs.Size(), Nm = m_CsLM.Size();//pose和motion的块数量,这里应该是相等的吧
  if (Nb <= 1) {
#ifdef CFG_PCG_FULL_BLOCK
    LA::ProductVector6f xc;
    LA::AlignedVector9f xm;
#else
    LA::AlignedVector3f xp, xr, xv, xba, xbw;
#endif
    Mxs->Resize(xs.Size());
    const LA::Vector6<PCG_TYPE> *xcs = (LA::Vector6<PCG_TYPE> *) xs.Data();
    LA::Vector6<PCG_TYPE> *Mxcs = (LA::Vector6<PCG_TYPE> *) Mxs->Data();
    for (int ic = 0; ic < Nc; ++ic) {
#ifdef CFG_PCG_FULL_BLOCK
      xc.Set(xcs[ic]);
      m_Mcs[ic].Apply(xc, (PCG_TYPE *) &Mxcs[ic]);
#else
      xcs[ic].Get(xp, xr);//取出位置和朝向的
      m_Mcs[ic].Apply(xp, xr, (PCG_TYPE *) &Mxcs[ic]);
#endif
    }
    const LA::Vector9<PCG_TYPE> *xms = (LA::Vector9<PCG_TYPE> *) (xcs + Nc);
    LA::Vector9<PCG_TYPE> *Mxms = (LA::Vector9<PCG_TYPE> *) (Mxcs + Nc);
    const int Nm = m_CsLM.Size();
    for (int im = 0; im < Nm; ++im) {
#ifdef CFG_PCG_FULL_BLOCK
      xm.Set(xms[im]);
      m_MmsLM[im].Apply(xm, (PCG_TYPE *) &Mxms[im]);
#else
      xms[im].Get(xv, xba, xbw);
      m_MmsLM[im].Apply(xv, xba, xbw, (PCG_TYPE *) &Mxms[im]);
#endif
    }
    return;
  }
  
  LA::ProductVector6f bc;//pose部分的r
  LA::AlignedVector9f bm;//motion部分的r
  Mxs->Set(xs);//Mxs先保存一下残差
  //xs.GetScaled(m_ss, *Mxs);
  LA::Vector6f *bcs = (LA::Vector6f *) Mxs->Data();//相机pose部分的r
  LA::Vector9f *bmsLM = (LA::Vector9f *) (bcs + Nc);//相机motion部分的r
#ifdef GBA_DEBUG_EIGEN_PCG
  const int pc = 6, pm = 9;
  const int pcm = pc + pm;
  const int N = Nc * pcm;
  EigenVectorXd e_b;
  e_b.Resize(N);
  e_b.MakeZero();
  for (int ic = 0, im = Nm - Nc, icp = 0, imp = pc; ic < Nc; ++ic, ++im, icp += pcm, imp += pcm) {
    e_b.block<pc, 1>(icp, 0) = EigenVector6f(bcs[ic]).cast<double>();
    if (im >= 0) {
      e_b.block<pm, 1>(imp, 0) = EigenVector9f(bmsLM[im]).cast<double>();
    }
  }
  //const EigenVectorXd e_z = EigenVectorXd(g_SI * g_z1);
  const EigenVectorXd &e_z = g_z1;
#endif
  for (int ic = 0, im = Nm - Nc; ic < Nc; ++ic, ++im) {//遍历下相机
    const int Nbcc = ic + Nb > Nc ? Nc - ic : Nb;
    const int Nbcm = im >= 0 ? (ic + 1 == Nc ? 1 : 2) : 0;
    const int Nbmc = im >= 0 ? Nbcc - 1 : 0;
    const int Nbmm = Nbcm;
    
    bc.Set(bcs[ic]);
    const LA::AlignedMatrix6x6f *MccsT = m_MccT[ic];//*舒尔补以后S矩阵pq对应的对角线部分的转置
    for (int ib = 1; ib < Nbcc; ++ib) {
      LA::AlignedMatrix6x6f::AddAbTo<float>(MccsT[ib], bc, bcs[ic + ib]);
    }
    if (im >= 0) {
      const LA::AlignedMatrix9x6f *McmsTLM = m_McmTLM[im];/*舒尔补以后S矩阵pose和motion对应的非对角部分的转置*/
      for (int ib = 0; ib < Nbcm; ++ib) {
        LA::AlignedMatrix9x6f::AddAbTo<float>(McmsTLM[ib], bc, bmsLM[im + ib]);//对应的motion部分的b
      }
    }
    LA::AlignedMatrix6x6f::Ab<float>(MccsT[0], bc, bcs[ic]);

    if (im >= 0) {
      bm.Set(bmsLM[im]);//bm = bmsLM[im],im是哈
      const LA::AlignedMatrix6x9f *MmcsTLM = m_MmcTLM[im];
      for (int ib = 0; ib < Nbmc; ++ib) {
        LA::AlignedMatrix6x9f::AddAbTo<float>(MmcsTLM[ib], bm, bcs[im + ib + 1]);
      }
      const LA::AlignedMatrix9x9f *MmmsTLM = m_MmmTLM[im];//motion部分的对角的.t
      if (Nbmm == 2) {
        LA::AlignedMatrix9x9f::AddAbTo<float>(MmmsTLM[1], bm, bmsLM[im + 1]);
      }
      LA::AlignedMatrix9x9f::Ab<float>(MmmsTLM[0], bm, bmsLM[im]);
    }

#ifdef GBA_DEBUG_EIGEN_PCG
    const int icp = ic * pcm, imp = icp + pc;
    const EigenVector6f e_bci = EigenVector6f(e_b.block<pc, 1>(icp, 0).cast<float>());
    e_b.block<pm, 1>(imp, 0) += g_M.block<pm, pc>(imp, icp) * e_bci.cast<double>();
    for (int _ic = ic + 1, _icp = icp + pcm; _ic < Nc; ++_ic, _icp += pcm) {
      e_b.block<pcm, 1>(_icp, 0) += g_M.block<pcm, pc>(_icp, icp) * e_bci.cast<double>();
    }
    e_b.block<pc, 1>(icp, 0) = -g_M.block<pc, pc>(icp, icp) * e_bci.cast<double>();
    const EigenVector9f e_bmi = EigenVector9f(e_b.block<pm, 1>(imp, 0).cast<float>());
    for (int _ic = ic + 1, _icp = icp + pcm; _ic < Nc; ++_ic, _icp += pcm) {
      e_b.block<pcm, 1>(_icp, 0) += g_M.block<pcm, pm>(_icp, imp) * e_bmi.cast<double>();
    }
    e_b.block<pm, 1>(imp, 0) = -g_M.block<pm, pm>(imp, imp) * e_bmi.cast<double>();
    for (int _ic = ic, _im = im, _icp = ic * pcm, _imp = _icp + pc; _ic < Nc;
         ++_ic, ++_im, _icp += pcm, _imp += pcm) {
      const EigenVector6f e_bci = EigenVector6f(e_b.block<pc, 1>(_icp, 0).cast<float>());
      e_bci.AssertEqual(bcs[_ic], 1, UT::String("bc[%d][%d]", ic, _ic));
      e_b.block<pc, 1>(_icp, 0) = EigenVector6f(bcs[_ic]).cast<double>();
      //bcs[_ic] = e_bci.GetVector6f();
      if (_im >= 0) {
        const EigenVector9f e_bmi = EigenVector9f(e_b.block<pm, 1>(_imp, 0).cast<float>());
        e_bmi.AssertEqual(bmsLM[_im], 1, UT::String("bm[%d][%d]", im, _im));
        e_b.block<pm, 1>(_imp, 0) = EigenVector9f(bmsLM[_im]).cast<double>();
        //bmsLM[_im] = e_bmi.GetVector9f();
      }
    }
    const int i = icp + pcm, _N = N - i;
    const EigenMatrixXd e_Ai = EigenMatrixXd(g_Ms[ic].block(i, i, _N, _N));
    const EigenVectorXd e_bi = EigenVectorXd(e_b.block(i, 0, _N, 1));
    const EigenVectorXd e_xi = EigenVectorXd(e_z.block(i, 0, _N, 1));
    const EigenVectorXd e_Axi = EigenVectorXd(e_Ai * e_xi);
    const EigenVectorXd e_ei = EigenVectorXd(e_Axi - e_bi);
    const double e2 = e_ei.norm();
    UT::Print("%d %e\n", ic, e2);
#endif
  }
  m_bcs.Resize(Nc);
  m_bmsLM.Resize(Nm);
  for (int ic = Nc - 1, im = Nm - 1; ic >= 0; --ic, --im) {
    const int Nbcc = ic + Nb > Nc ? Nc - ic : Nb;
    const int Nbcm = im >= 0 ? (ic + 1 == Nc ? 1 : 2) : 0;
    const int Nbmc = im >= 0 ? Nbcc - 1 : 0;
    const int Nbmm = Nbcm;

    if (im >= 0) {
      float *_bm = bmsLM[im];
      const LA::AlignedMatrix9x6f *MmcsLM = m_MmcLM[im];
      for (int ib = 0; ib < Nbmc; ++ib) {
        LA::AlignedMatrix9x6f::AddAbTo(MmcsLM[ib], m_bcs[ic + ib + 1], _bm);
      }
      if (Nbmm == 2) {
        LA::AlignedMatrix9x9f::AddAbTo(m_MmmLM[im][1], m_bmsLM[im + 1], _bm);
      }
      m_bmsLM[im].Set(_bm);
    }

    float *_bc = bcs[ic];
    const LA::AlignedMatrix6x6f *Mccs = m_Mcc[ic];
    for (int ib = 1; ib < Nbcc; ++ib) {
      LA::AlignedMatrix6x6f::AddAbTo(Mccs[ib], m_bcs[ic + ib], _bc);
    }
    const LA::AlignedMatrix6x9f *McmsLM = m_McmLM[im];
    for (int ib = 0; ib < Nbcm; ++ib) {
      LA::AlignedMatrix6x9f::AddAbTo(McmsLM[ib], m_bmsLM[im + ib], _bc);
    }
    m_bcs[ic].Set(_bc);
    
#ifdef GBA_DEBUG_EIGEN_PCG
    const int icp = ic * pcm, imp = icp + pc;
    for (int _ic = ic + 1, _icp = icp + pcm; _ic < Nc; ++_ic, _icp += pcm) {
      e_b.block<pm, 1>(imp, 0) += g_M.block<pm, pcm>(imp, _icp) * e_b.block<pcm, 1>(_icp, 0);
    }
    e_b.block<pc, 1>(icp, 0) += g_M.block<pc, pm>(icp, imp) * e_b.block<pm, 1>(imp, 0);
    for (int _ic = ic + 1, _icp = icp + pcm; _ic < Nc; ++_ic, _icp += pcm) {
      e_b.block<pc, 1>(icp, 0) += g_M.block<pc, pcm>(icp, _icp) * e_b.block<pcm, 1>(_icp, 0);
    }
    const EigenVector6f e_bci = EigenVector6f(e_b.block<pc, 1>(icp, 0).cast<float>());
    e_bci.AssertEqual(bcs[ic], 1, UT::String("bc[%d]", ic));
    e_b.block<pc, 1>(icp, 0) = EigenVector6f(bcs[ic]).cast<double>();
    //bcs[ic] = e_bci.GetVector6f();
    if (im >= 0) {
      const EigenVector9f e_bmi = EigenVector9f(e_b.block<pm, 1>(imp, 0).cast<float>());
      e_bmi.AssertEqual(bmsLM[im], 1, UT::String("bm[%d]", im));
      e_b.block<pm, 1>(imp, 0) = EigenVector9f(bmsLM[im]).cast<double>();
      //bmsLM[im] = e_bmi.GetVector9f();
    }
#endif
  }
  //Mxs->Scale(m_ss);
}
//计算 m_drs  = A*x0 A是舒尔补以后的S
//计算顺序：对角线App部分的*x0,非对角线共视帧之间App部分*x0,ApplyAcm算的就是前后motion约束带来的A里的部分*x0
// （不用计算App了是因为非对角线共视帧之间App部分已经算过了）
void GlobalBundleAdjustor::ApplyA(const LA::AlignedVectorX<PCG_TYPE> &xs,/*优化变量 pose+motion*/
                                  LA::AlignedVectorX<PCG_TYPE> *Axs) {
  const LA::Vector6<PCG_TYPE> *xcs = (LA::Vector6<PCG_TYPE> *) xs.Data();//pose部分的增量
  ConvertCameraUpdates(xcs, &m_xcsP);//m_xcsP保存pose的增量
  Axs->Resize(xs.Size());//campose_num * 6 + motion_num * 9
  LA::Vector6<PCG_TYPE> *Axcs = (LA::Vector6<PCG_TYPE> *) Axs->Data();//就是Ax的结果
  const int Nc = int(m_KFs.size());//关键帧数量
  //对角线部分的A*x0
  for (int ic = 0; ic < Nc; ++ic) {
    LA::AlignedMatrix6x6f::Ab(m_Acus[ic]/*舒尔补以后每帧自己和自己pose的对角线部分*/, m_xcsP[ic], (PCG_TYPE *) &Axcs[ic]);//Axcs[ic] = S*x
  }
  for (int ic = 0; ic < Nc; ++ic) {//遍历所有的关键帧
    const int icb = m_iKF2cb[ic];//这个
    const LA::AlignedMatrix6x6f *Acbs = m_Acbs.Data() + icb;//右上非对角线的posexpose(即不同帧因为共视和motion导致有影响的地方)
    const LA::AlignedMatrix6x6f *AcbTs = m_AcbTs.Data() + icb;//右下非对角线的posexpose(即不同帧因为共视和motion导致有影响的地方)
    const LA::ProductVector6f &xc = m_xcsP[ic];//当前帧pose
    PCG_TYPE *Axc = Axcs[ic];
    const KeyFrame &KF = m_KFs[ic];//当前遍历的关键帧
    const int Nkp = static_cast<int>(KF.m_ikp2KF.size());//共视的关键帧数量
    for (int ikp = 0; ikp < Nkp; ++ikp) {//遍历所有共视的关键帧
      const int _ic = KF.m_ikp2KF[ikp];//这个关键帧的id
      LA::AlignedMatrix6x6f::AddAbTo<PCG_TYPE>(Acbs[ikp], xc, (PCG_TYPE *) &Axcs[_ic]);//A_icic*pic = Axcs[_ic] 坐下的的这个矩阵块*xic
      LA::AlignedMatrix6x6f::AddAbTo<PCG_TYPE>(AcbTs[ikp], m_xcsP[_ic], Axc);//Aic_ic*p_ic = Axcs[ic] 右上的这个矩阵块*x_ic
    }
  }
  //ConvertCameraUpdates(m_Axcs, Axcs);
  const int Nm = m_CsLM.Size();//Amm部分和cm部分 *X的计算
  const int ic0 = Nc - Nm;
  ApplyAcm(m_xcsP.Data() + ic0/*Xpose*/, (LA::Vector9<PCG_TYPE> *) (xcs + Nc)/*Xmotion*/, Axcs + ic0/*A*X pose*/,
           (LA::Vector9<PCG_TYPE> *) (Axcs + Nc)/*A*X motion的部分*/, false,/*需不需要算pose部分的*/
           m_AmusLM.Size() == Nm ? m_AmusLM.Data() : NULL);
}

void GlobalBundleAdjustor::ApplyAcm(const LA::ProductVector6f *xcs/*Xpose*/, const LA::Vector9f *xms/*Xmotion*/,
                                    LA::Vector6f *Axcs/*A*X pose的部分*/, LA::Vector9f *Axms/*A*X motion的部分*/, const bool Acc,
                                    const LA::AlignedMatrix9x9f *Amus) {
  LA::AlignedMatrix6x6f A66;
  LA::AlignedMatrix6x9f A69;
  LA::AlignedMatrix9x6f A96;
  LA::AlignedMatrix9x9f A99;
  LA::AlignedVector9f v9[2];
#ifndef CFG_IMU_FULL_COVARIANCE
  LA::AlignedMatrix3x3f A33;
  LA::AlignedMatrix3x9f A39;
  LA::AlignedVector3f v3;
#endif
  const int Nm = m_CsLM.Size();
  for (int im = 0, r = 0; im < Nm; ++im, r = 1 - r) {//遍历所有关键帧
    const LA::ProductVector6f &xc = xcs[im];//这帧的pose
    LA::AlignedVector9f &xm = v9[r];//这帧对应的motion
    xm.Set(xms[im]);//xm = xms[im] //为sm设置这帧对应的motion
    float *Axc = Axcs[im]/*A*X pose的部分*/, *Axm = Axms[im]/*A*X motion的部分*/;
    const Camera::Factor &SAcm = m_SAcmsLM[im];//滑窗内每一帧的Hrm以及Hmm(对角线上跟运动相关)和mb部分
    if (Amus) {
      LA::AlignedMatrix9x9f::Ab(Amus[im], xm, Axm);//Axm = S[im]*xm 也就是对角线上mm部分的A*x
    } else {
      A99.Set(SAcm.m_Au.m_Amm.m_A);
      LA::AlignedMatrix9x9f::Ab(A99, xm, Axm);
    }
    LA::AlignedMatrix6x9f::AddAbTo(SAcm.m_Au.m_Acm, xm, Axc);//A[pose,motion] * Xmotion 存在A*Xpose对应的位置
    SAcm.m_Au.m_Acm.GetTranspose(A96);//A96 = S[pose,motion].t
    LA::AlignedMatrix9x6f::AddAbTo(A96, xc, Axm);//A[motion,pose] * Xpose 存在A*Xmotion对应的位置
    if (im == 0) {
      continue;
    }
    const int _im = im - 1;//前一帧的运动
    const LA::ProductVector6f &_xc = xcs[_im];//前一帧的poe
    const LA::AlignedVector9f &_xm = v9[1 - r];//前一帧的motion
    float *_Axc = Axcs[_im]/*前一帧的A*X pose部分*/, *_Axm = Axms[_im];//前一帧A*Xmotion的部分
    if (Acc) {//这个之前已经算过了,所以不用再算了
      LA::AlignedMatrix6x6f::AddAbTo(SAcm.m_Ab.m_Acc, xc, _Axc);//A[T0,T1] *XT1 存在A*Xpose T0的位置
      SAcm.m_Ab.m_Acc.GetTranspose(A66);
      LA::AlignedMatrix6x6f::AddAbTo(A66, _xc, Axc);//A[T1,T0] *XT0 存在A*Xpose T1的位置
    }
#ifdef CFG_IMU_FULL_COVARIANCE//比如T0(pose),M0(motion),T1,M1
    LA::AlignedMatrix6x9f::AddAbTo(SAcm.m_Ab.m_Acm, xm, _Axc);//A[T0,M1] * XM1 存在A*Xpose T0的位置
    SAcm.m_Ab.m_Acm.GetTranspose(A96);
    LA::AlignedMatrix9x6f::AddAbTo(A96, _xc, Axm);//A[M1,T0] * XT0 存在A*Xmotion T0的位置
    LA::AlignedMatrix9x6f::AddAbTo(SAcm.m_Ab.m_Amc, xc, _Axm);//A[M0,T1] * XT1 存在A*Xmotion M0的位置
    SAcm.m_Ab.m_Amc.GetTranspose(A69);
    LA::AlignedMatrix6x9f::AddAbTo(A69, _xm, Axc);//A[T1,M0] * XM0 存在A*Xpose T1的位置
    LA::AlignedMatrix9x9f::AddAbTo(SAcm.m_Ab.m_Amm, xm, _Axm);//A[M0,M1] *XM1 存在A*Xmotion M0的位置
    SAcm.m_Ab.m_Amm.GetTranspose(A99);
    LA::AlignedMatrix9x9f::AddAbTo(A99, _xm, Axm);//A[M1,M0] *XM0 存在A*Xmotion M1的位置
#else
    xm.Get012(v3);
    LA::AlignedMatrix3x3f::AddAbTo(SAcm.m_Ab.m_Acm.m_Arv, v3, &_Axc.v3());
    LA::AlignedMatrix9x3f::AddAbTo(SAcm.m_Ab.m_Amm.m_Amv, v3, _Axm);
    SAcm.m_Ab.m_Acm.m_Arv.GetTranspose(A33);
    _xc.Get345(v3);
    LA::AlignedMatrix3x3f::AddAbTo(A33, v3, &Axm.v0());
    LA::AlignedMatrix9x6f::AddAbTo(SAcm.m_Ab.m_Amc, xc, _Axm);
    SAcm.m_Ab.m_Amc.GetTranspose(A69);
    LA::AlignedMatrix6x9f::AddAbTo(A69, _xm, Axc);
    SAcm.m_Ab.m_Amm.m_Amv.GetTranspose(A39);
    LA::AlignedMatrix3x9f::AddAbTo(A39, _xm, &Axm.v0());
    for (int i = 3; i < 9; ++i) {
      const float acc = i < 6 ? SAcm.m_Ab.m_Amm.m_Ababa : SAcm.m_Ab.m_Amm.m_Abwbw;
      _Axm[i] = acc * xm[i] + _Axm[i];
      Axm[i] = acc * _xm[i] + Axm[i];
    }
#endif
  }
}

void GlobalBundleAdjustor::ApplyAcm(const LA::ProductVector6f *xcs, const LA::Vector9d *xms,
                                    LA::Vector6d *Axcs, LA::Vector9d *Axms, const bool Acc,
                                    const LA::AlignedMatrix9x9f *Amus) {
  LA::AlignedMatrix6x6f A66;
  LA::AlignedMatrix6x9f A69;
  LA::AlignedMatrix9x6f A96;
  LA::AlignedMatrix9x9f A99;
  LA::AlignedVector9f v9[2];
#ifndef CFG_IMU_FULL_COVARIANCE
  LA::AlignedMatrix3x3f A33;
  LA::AlignedMatrix3x9f A39;
  LA::AlignedVector3f v3;
#endif
  const int Nm = m_CsLM.Size();
  for (int im = 0, r = 0; im < Nm; ++im, r = 1 - r) {
    const LA::ProductVector6f &xc = xcs[im];
    LA::AlignedVector9f &xm = v9[r];
    xm.Set(xms[im]);
    double *Axc = Axcs[im], *Axm = Axms[im];
    const Camera::Factor &SAcm = m_SAcmsLM[im];
    if (Amus) {
      LA::AlignedMatrix9x9f::Ab(Amus[im], xm, Axm);
    } else {
      A99.Set(SAcm.m_Au.m_Amm.m_A);
      LA::AlignedMatrix9x9f::Ab(A99, xm, Axm);
    }
    LA::AlignedMatrix6x9f::AddAbTo(SAcm.m_Au.m_Acm, xm, Axc);
    SAcm.m_Au.m_Acm.GetTranspose(A96);
    LA::AlignedMatrix9x6f::AddAbTo(A96, xc, Axm);
    if (im == 0) {
      continue;
    }
    const int _im = im - 1;
    const LA::ProductVector6f &_xc = xcs[_im];
    const LA::AlignedVector9f &_xm = v9[1 - r];
    double *_Axc = Axcs[_im], *_Axm = Axms[_im];
    if (Acc) {
      LA::AlignedMatrix6x6f::AddAbTo(SAcm.m_Ab.m_Acc, xc, _Axc);
      SAcm.m_Ab.m_Acc.GetTranspose(A66);
      LA::AlignedMatrix6x6f::AddAbTo(A66, _xc, Axc);
    }
#ifdef CFG_IMU_FULL_COVARIANCE
    LA::AlignedMatrix6x9f::AddAbTo(SAcm.m_Ab.m_Acm, xm, _Axc);
    SAcm.m_Ab.m_Acm.GetTranspose(A96);
    LA::AlignedMatrix9x6f::AddAbTo(A96, _xc, Axm);
    LA::AlignedMatrix9x6f::AddAbTo(SAcm.m_Ab.m_Amc, xc, _Axm);
    SAcm.m_Ab.m_Amc.GetTranspose(A69);
    LA::AlignedMatrix6x9f::AddAbTo(A69, _xm, Axc);
    LA::AlignedMatrix9x9f::AddAbTo(SAcm.m_Ab.m_Amm, xm, _Axm);
    SAcm.m_Ab.m_Amm.GetTranspose(A99);
    LA::AlignedMatrix9x9f::AddAbTo(A99, _xm, Axm);
#else
    xm.Get012(v3);
    LA::AlignedMatrix3x3f::AddAbTo(SAcm.m_Ab.m_Acm.m_Arv, v3, &_Axc.v3());
    LA::AlignedMatrix9x3f::AddAbTo(SAcm.m_Ab.m_Amm.m_Amv, v3, _Axm);
    SAcm.m_Ab.m_Acm.m_Arv.GetTranspose(A33);
    _xc.Get345(v3);
    LA::AlignedMatrix3x3f::AddAbTo(A33, v3, &Axm.v0());
    LA::AlignedMatrix9x6f::AddAbTo(SAcm.m_Ab.m_Amc, xc, _Axm);
    SAcm.m_Ab.m_Amc.GetTranspose(A69);
    LA::AlignedMatrix6x9f::AddAbTo(A69, _xm, Axc);
    SAcm.m_Ab.m_Amm.m_Amv.GetTranspose(A39);
    LA::AlignedMatrix3x9f::AddAbTo(A39, _xm, &Axm.v0());
    for (int i = 3; i < 9; ++i) {
      const float acc = i < 6 ? SAcm.m_Ab.m_Amm.m_Ababa : SAcm.m_Ab.m_Amm.m_Abwbw;
      _Axm[i] = acc * xm[i] + _Axm[i];
      Axm[i] = acc * _xm[i] + Axm[i];
    }
#endif
  }
}

GlobalBundleAdjustor::Residual GlobalBundleAdjustor::ComputeResidual(const
                                                                     LA::AlignedVectorX<PCG_TYPE> &xs,
                                                                     const bool minus) {
  Residual R;
  const int N = xs.Size();
  m_work.Resize(N);
  LA::AlignedVectorXf rs(m_work.Data(), N, false);
  ApplyA(xs, &rs);
  if (minus) {
    R.m_F = xs.Dot(rs) / 2 - xs.Dot(m_bs);
    rs -= m_bs;
  } else {
    R.m_F = xs.Dot(rs) / 2 + xs.Dot(m_bs);
    rs += m_bs;
  }
  R.m_r2 = rs.SquaredLength();
  return R;
}
//反求逆深度的增量
//step1:先遍历所有关键帧中的新地图点,以及每个关键帧对于地图点的观测,如果这个关键的pose增量有变动,将自己的地图点以及观测地图点设置成需要反求的flags
//step2:遍历关键帧自己的地图点以及它观测到的地图点,需要更新的话需要求出d_u，d_u =Wcu.t * Huu^-1 * dxc +  Huu^-1*-bu
//step3: 遍历所有的关键帧中的新地图点,对需要反求逆深度增量的du取逆,即du =-Wcu.t * Huu^-1 * dxc +  Huu^-1*bu (Huu就是V,bu就是v)
// = Huu^-1*(bu -Wcu.t * dxc )也就求出了地图点逆深度的增量
//step4:更新状态将所有所有需要反求增量的du push进m_xsGN
void GlobalBundleAdjustor::SolveBackSubstitution() {
  const ubyte ucFlag = GBA_FLAG_FRAME_UPDATE_DELTA | GBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION;
  const int nKFs = static_cast<int>(m_KFs.size());//
  for (int iKF = 0; iKF < nKFs; ++iKF)
  {//遍历所有的关键帧
    const KeyFrame &KF = m_KFs[iKF];
    //const int Nx = static_cast<int>(KF.m_xs.size());
    //if (Nx == 0) {
    //  continue;
    //}
    const bool dc = m_xr2s[iKF] > BA_BACK_SUBSTITUTE_ROTATION ||//这里的条件基本就是pose增量只要有了变动就会更新吧
                    m_xp2s[iKF] > BA_BACK_SUBSTITUTE_POSITION;
    if (dc) {
      m_ucs[iKF] |= GBA_FLAG_FRAME_UPDATE_DELTA;//m_ucs更新这个关键帧:需要更新增量
    } else {
      m_ucs[iKF] &= ~GBA_FLAG_FRAME_UPDATE_DELTA;//m_ucs更新这个关键帧:不需要更新增量
    }
    if (!(m_ucs[iKF] & ucFlag))
    {//不更新增量并且不更新轨迹信息,那么直接跳过
      continue;
    }
    const int Nx = static_cast<int>(KF.m_xs.size());//这个关键帧的新的地图点
    if (Nx > 0)
    {
      ubyte *uds = m_uds.data() + m_iKF2d[iKF];//当前第一个新地图点的逆深度指针位置
      for (int ix = 0; ix < Nx; ++ix) {//遍历所有地图点
        if (!dc && !(uds[ix] & GBA_FLAG_TRACK_UPDATE_INFORMATION)) {//如果增量和轨迹信息都不需要更新就跳过
          continue;
        }
        uds[ix] |= GBA_FLAG_TRACK_UPDATE_BACK_SUBSTITUTION;//设置成需要反求地图点逆深度增量
        m_ucs[iKF] |= GBA_FLAG_FRAME_UPDATE_BACK_SUBSTITUTION;//这个地图点所属的关进帧也设置一下
      }
    }
    if (!dc) {//如果增量不需要更新就跳过
      continue;
    }
    const int NZ = static_cast<int>(KF.m_Zs.size());//关键帧的观测
    for (int iZ = 0; iZ < NZ; ++iZ) {
      const FRM::Measurement &Z = KF.m_Zs[iZ];
#ifdef CFG_DEBUG
      UT_ASSERT(Z.m_iKF < iKF);
#endif
      if (Z.m_iz1 == Z.m_iz2) {
        continue;
      }
      m_ucs[Z.m_iKF] |= GBA_FLAG_FRAME_UPDATE_BACK_SUBSTITUTION;
      ubyte *_uds = m_uds.data() + m_iKF2d[Z.m_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
        _uds[KF.m_zs[iz].m_ix] |= GBA_FLAG_TRACK_UPDATE_BACK_SUBSTITUTION;
      }
    }
  }

  int iX = 0;
  m_iKF2X.assign(nKFs, -1);//就建立一下需要反求增量的点的索引
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    if (!(m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_BACK_SUBSTITUTION)) {//如果当前关键帧不需要更新这个东西就跳过
      continue;
    }
    m_iKF2X[iKF] = iX;
    iX += int(m_KFs[iKF].m_xs.size());
  }
  m_xds.Resize(iX);//扩容成当前新地图点个数的size


    //遍历关键帧自己的地图点以及它观测到的地图点,需要更新的话需要求出d_u，d_u =Wcu.t * Huu^-1 * dxc +  Huu^-1*-bu
  LA::AlignedVector6f xc;//当前这帧的pose增量
  const LA::Vector6f *xcs = (LA::Vector6f *) m_xsGN.Data();
  for (int iKF = 0; iKF < nKFs; ++iKF)
  {
    //if (!(m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_BACK_SUBSTITUTION)) {
    //  continue;
    //}
    const ubyte dx = m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_DELTA;//当前帧是否要更新pose增量
    if (dx) {
      xc.Set(xcs[iKF]);
    }
    const KeyFrame &KF = m_KFs[iKF];//当前关键帧
    if (m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_BACK_SUBSTITUTION) {//如果要更新
      const LA::AlignedVector6f *_xc = dx ? &xc : NULL;
      const ubyte *uds = m_uds.data() + m_iKF2d[iKF];//这个关键帧第一个新地图点的flags地址
      float *xds = m_xds.Data() + m_iKF2X[iKF];//这个关键帧第一个新地图点的逆深度
      const int Nx = static_cast<int>(KF.m_xs.size());//新地图点数量
      for (int ix = 0; ix < Nx; ++ix)
      {
        if (!(uds[ix] & GBA_FLAG_TRACK_UPDATE_BACK_SUBSTITUTION)) {//不需要反求就跳过
          continue;
        } else if (uds[ix] & GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO) {//之前的观测没有有效
          xds[ix] = 0.0f;
        } else {
          xds[ix] = KF.m_Mxs1[ix].BackSubstitute(_xc);//Wcu.t * Huu^-1 * dxc +  Huu^-1*-bu
        }
      }
    }
    if (!dx) {//不需要更新增量的话就跳过
      continue;
    }
    const int NZ = static_cast<int>(KF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ)
    {
      const FRM::Measurement &Z = KF.m_Zs[iZ];//这个关键帧对于关键帧的观测
#ifdef CFG_DEBUG
      UT_ASSERT(Z.m_iKF < iKF);
#endif
      const ubyte *_uds = m_uds.data() + m_iKF2d[Z.m_iKF];
      float *_xds = m_xds.Data() + m_iKF2X[Z.m_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
        const int ix = KF.m_zs[iz].m_ix;
        if (_uds[ix] & GBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO) {//之前没有过观测信息
          continue;
        }
        _xds[ix] = KF.m_Mzs1[iz].BackSubstitute(xc) + _xds[ix];//原先基础上加上Wcu.t * Huu^-1 * dxc +  Huu^-1*-bu
      }
    }
  }

    //遍历所有的关键帧中的新地图点,对需要反求逆深度增量的du取逆
    // 即du =-Wcu.t * Huu^-1 * dxc +  Huu^-1*bu (Huu就是V,bu就是v) = Huu^-1*(bu -Wcu.t * dxc )也就求出了地图点逆深度的增量
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    const int iX = m_iKF2X[iKF];
    if (iX == -1) {
      continue;
    }
    const int id = m_iKF2d[iKF];
    const Depth::InverseGaussian *ds = m_ds.data() + id;//新地图点对应的逆深度
    ubyte *uds = m_uds.data() + id;//地图点对应的flags
    float *xds = m_xds.Data() + iX;//
    const int Nx = static_cast<int>(m_KFs[iKF].m_xs.size());
    for (int ix = 0; ix < Nx; ++ix) {
      if (!(uds[ix] & GBA_FLAG_TRACK_UPDATE_BACK_SUBSTITUTION)) {
        continue;
      }
      xds[ix] = -xds[ix];//取逆
      //if (Depth::InverseGaussian::Valid(xds[ix] + ds[ix].u())) {
      //  continue;
      //}
      //xds[ix] = 0.0f;
      //uds[ix] &= ~GBA_FLAG_TRACK_UPDATE_BACK_SUBSTITUTION;
    }
  }//遍历所有的关键帧和它们观测到的新地图点,更更新flag信息:不再需要更新轨迹信息
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    if (!(m_ucs[iKF] & GBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION)) {//这个关键帧是否要更新轨迹信息
      continue;
    }
    m_ucs[iKF] &= ~GBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION;//m_ucs更新这个关键帧:不再需要更新轨迹信息
    ubyte *uds = m_uds.data() + m_iKF2d[iKF];
    const int Nx = static_cast<int>(m_KFs[iKF].m_xs.size());
    for (int ix = 0; ix < Nx; ++ix) {
      uds[ix] &= ~GBA_FLAG_TRACK_UPDATE_INFORMATION;//m_uds更新这个地图点:不再需要更新轨迹信息
    }
  }
#ifdef GBA_DEBUG_GROUND_TRUTH_STATE
  if (m_dsGT) {
    for (int iKF = 0; iKF < nKFs; ++iKF) {
      const int iX = m_iKF2X[iKF];
      if (iX == -1) {
        continue;
      }
      float *xds = m_xds.Data() + iX;
      const int id = m_iKF2d[iKF];
      const Depth::InverseGaussian *ds = m_ds.data() + id, *dsGT = m_dsGT->data() + id;
      const int Nx = static_cast<int>(m_KFs[iKF].m_xs.size());
      for (int ix = 0; ix < Nx; ++ix) {
        xds[ix] = dsGT[ix].u() - ds[ix].u();
      }
    }
  }
#endif
  PushDepthUpdates(m_xds, &m_xsGN);//保存需要反求的增量
  m_x2GN = m_xsGN.SquaredLength();
//#ifdef CFG_DEBUG
#if 0
  const std::string dir = m_dir + "pcg/";
#ifdef WIN32
  m_xsGN.AssertEqual(UT::String("%sx%02d.txt", dir.c_str(), m_iIter), 2, "", -1.0f, -1.0f);
#else
  m_xsGN.SaveB(UT::String("%sx%02d.txt", dir.c_str(), m_iIter));
#endif
#endif
}

void GlobalBundleAdjustor::SolveBackSubstitutionGT(const std::vector<Depth::InverseGaussian> &ds,
                                                   LA::AlignedVectorXf *xs) {
  if (!m_dsGT) {
    return;
  }
  const int Nd = static_cast<int>(m_ds.size());
  LA::AlignedVectorXf dus, dusGT;
  m_work.Resize(dus.BindSize(Nd) + dusGT.BindSize(Nd));
  dus.Bind(m_work.Data(), Nd);
  dusGT.Bind(dus.BindNext(), Nd);
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    const int id = m_iKF2d[iKF], iX = m_iKF2X[iKF];
    const Depth::InverseGaussian *_ds = iX == -1 ? m_ds.data() + id : ds.data() + iX;
    float *_dus = dus.Data() + id;
    const int Nx = static_cast<int>(m_KFs[iKF].m_xs.size());
    for (int ix = 0; ix < Nx; ++ix) {
      _dus[ix] = _ds[ix].u();
    }
  }
  for (int id = 0; id < Nd; ++id) {
    dusGT[id] = m_dsGT->at(id).u();
  }
  dusGT -= dus;
  xs->Push(dusGT);
}

bool GlobalBundleAdjustor::EmbeddedMotionIteration() {
  const int pc = 6, pm = 9;
  const int Nc = m_Cs.Size(), Nm = m_CsLM.Size();
  const LA::Vector6f *xcs = ((LA::Vector6f *) m_xsGN.Data()) + Nc - Nm;
  LA::Vector9f *xms = (LA::Vector9f *) (xcs + Nm);
  //const float eps = 0.0f;
  const float eps = FLT_EPSILON;
#if 0
//#if 1
  AlignedVector<LA::AlignedMatrix9x9f> Amus, Ambs;
  AlignedVector<LA::AlignedVector9f> bms;
  m_work.Resize((Amus.BindSize(Nm) + Ambs.BindSize(Nm - 1) + bms.BindSize(Nm)) / sizeof(float));
  Amus.Bind(m_work.Data(), Nm);
  Ambs.Bind(Amus.End(), Nm - 1);
  bms.Bind(Ambs.End(), Nm);

  LA::AlignedVector6f xc[2];
  LA::AlignedMatrix9x6f Amc;
  for (int im = 0, r = 0; im < Nm; ++im, r = 1 - r) {
    const Camera::Factor &A = m_SAcmsLM[im];
    LA::AlignedVector9f &bm = bms[im];
    A.m_Au.m_Amm.Get(&Amus[im], &bm);
    xc[r].Set(xcs[im]);
    A.m_Au.m_Acm.GetTranspose(Amc);
    LA::AlignedMatrix9x6f::AddAbTo(Amc, xc[r], bm);
    if (im == 0) {
      continue;
    }
    const int _im = im - 1;
    Ambs[_im] = A.m_Ab.m_Amm;
    A.m_Ab.m_Acm.GetTranspose(Amc);
    LA::AlignedMatrix9x6f::AddAbTo(Amc, xc[1 - r], bm);
    LA::AlignedMatrix9x6f::AddAbTo(A.m_Ab.m_Amc, xc[r], bms[_im]);
  }
#if 0
  if (Nm > 0) {
    UT::DebugStart();
    LA::AlignedMatrix9x9f &Am = Amus[Nm - 1];
    Am[0][0] = Am[1][1] = Am[2][2] = FLT_EPSILON;
    UT::DebugStop();
  }
#endif
  LA::AlignedMatrix9x9f Am21, Mm21;
  LA::AlignedVector9f bm1;
  for (int im1 = 0, im2 = 1; im1 < Nm; im1 = im2++) {
    LA::AlignedMatrix9x9f &Mm11 = Amus[im1];
    //if (im1 == 1) {
    //  Mm11.Print(true);
    //  UT::PrintSeparator();
    //}
    if (!Mm11.InverseLDL(eps)) {
      //return false;
      Mm11.MakeZero();
      if (im2 < Nm) {
        Ambs[im1].MakeZero();
      }
      bms[im1].MakeZero();
      continue;
    }
    bm1 = bms[im1];
    LA::AlignedMatrix9x9f::Ab(Mm11, bm1, bms[im1]);
    if (im2 == Nm) {
      break;
    }
    Ambs[im1].GetTranspose(Am21);
    LA::AlignedMatrix9x9f::ABT(Am21, Mm11, Mm21);
    Mm21.GetTranspose(Ambs[im1]);
    //UT::PrintSeparator();
    //Amus[im2].Print(true);
    LA::AlignedMatrix9x9f::SubtractABTFromUpper(Mm21, Am21, Amus[im2]);
    //UT::PrintSeparator();
    //Amus[im2].Print(true);
    LA::AlignedMatrix9x9f::SubtractAbFrom(Mm21, bm1, bms[im2]);
  }
  for (int im1 = Nm - 2, im2 = im1 + 1; im1 >= 0; im2 = im1--) {
    LA::AlignedMatrix9x9f::SubtractAbFrom(Ambs[im1], bms[im2], bms[im1]);
  }
  for (int im = 0; im < Nm; ++im) {
    xms[im].Set(bms[im]);
  }
#else
  const int Nmr = Nc * pm, Nmc = pm + pm;
  LA::AlignedMatrixXf A;
  LA::AlignedVectorXf b, ai;
  AlignedVector<LA::AlignedVector18f> x;
  m_work.Resize(A.BindSize(Nmr, Nmc) + b.BindSize(Nmr) + x.BindSize(Nc) + ai.BindSize(Nmc));
  A.Bind(m_work.Data(), Nmr, Nmc);
  b.Bind(A.BindNext(), Nmr);
  x.Bind(b.BindNext(), Nc);
  ai.Bind(x.BindNext(), Nmc);

  LA::AlignedVector6f xc[2];
  LA::AlignedMatrix9x9f Amm;
  LA::AlignedMatrix9x6f Amc;
  for (int im1 = -1, im2 = 0, imp1 = -pm, imp2 = 0, r = 0; im2 < Nc;
       im1 = im2++, imp1 = imp2, imp2 += pm, r = 1 - r) {
    const Camera::Factor &_A = m_SAcmsLM[im2];
    Amm.Set(_A.m_Au.m_Amm.m_A);
    A.SetBlock(imp2, 0, Amm);
    float *_b = b.Data() + imp2;
    _A.m_Au.m_Amm.m_b.Get(_b);
    xc[r].Set(xcs[im2]);
    _A.m_Au.m_Acm.GetTranspose(Amc);
    LA::AlignedMatrix9x6f::AddAbTo(Amc, xc[r], _b);
    if (im2 == 0) {
      continue;
    }
    A.SetBlock(imp1, pm, _A.m_Ab.m_Amm);
    _A.m_Ab.m_Acm.GetTranspose(Amc);
    LA::AlignedMatrix9x6f::AddAbTo(Amc, xc[1 - r], _b);
    LA::AlignedMatrix9x6f::AddAbTo(_A.m_Ab.m_Amc, xc[r], b.Data() + imp1);
  }

  LA::AlignedVector9f _mi;
  const float epsv = UT::Inverse(BA_VARIANCE_MAX_VELOCITY, BA_WEIGHT_FEATURE, eps);
  const float epsba = UT::Inverse(BA_VARIANCE_MAX_BIAS_ACCELERATION, BA_WEIGHT_FEATURE, eps);
  const float epsbw = UT::Inverse(BA_VARIANCE_MAX_BIAS_GYROSCOPE, BA_WEIGHT_FEATURE, eps);
  const float _eps[] = {epsv, epsv, epsv, epsba, epsba, epsba, epsbw, epsbw, epsbw};
  for (int im = 0, imp = 0; im < Nc; ++im) {
    for (int ip = 0; ip < pm; ++ip, ++imp) {
      float *mi = A[imp];
      float &ni = b[imp];
      const float aii = mi[ip];
      if (aii <= _eps[ip]) {
        memset(mi, 0, sizeof(float) * Nmc);
        ni = 0.0f;
        continue;
      }
      const float mii = 1.0f / aii;
      mi[ip] = mii;
      ai.Set(mi, Nmc);
      ai.MakeMinus(ip + 1);
      SIMD::Multiply(ip + 1, Nmc, mii, mi);
      ni *= mii;

      int jmp = imp + 1;
      for (int jp = ip + 1; jp < pm; ++jp, ++jmp) {
        const float aij = ai[jp];
        SIMD::MultiplyAddTo(jp, Nmc, aij, mi, A[jmp]);
        b[jmp] += aij * ni;
      }
      if (im == Nc - 1) {
        continue;
      }
      const float *_ai = ai.Data() + pm;
      _mi.Set(mi + pm);
      for (int jp = 0; jp < pm; ++jp, ++jmp) {
        const float aij = _ai[jp];
        SIMD::MultiplyAddTo(jp, pm, aij, _mi, A[jmp]);
        b[jmp] += aij * ni;
      }
    }
  }

  for (int im = 0, imp = 0; im < Nc; ++im, imp += pm) {
    memcpy(x[im], b.Data() + imp, 36);
  }
  for (int im = Nc - 1, imp = Nmr - 1, r = im & 1; im >= 0; --im, r = 1 - r) {
    const int _im = im + 1;
    const int _Nmc = _im == Nc ? pm : Nmc;
    float *xi = x[im];
    if (_im < Nc) {
      memcpy(xi + pm, x[_im], 36);
    }
    for (int ip = pm - 1; ip >= 0; --ip, --imp) {
      xi[ip] -= SIMD::Dot(ip + 1, _Nmc, A[imp], xi);
    }
  }
  for (int im = 0; im < Nc; ++im) {
    xms[im].Set(x[im]);
  }
#endif
  m_xsGN.MakeMinus(pc * Nc);
  ConvertMotionUpdates((float *) xms, &m_xv2s, &m_xba2s, &m_xbw2s);
  return true;
}

void GlobalBundleAdjustor::EmbeddedPointIteration(const AlignedVector<Rigid3D> &Cs,
                                                  const std::vector<ubyte> &ucs,
                                                  const std::vector<ubyte> &uds,
                                                  std::vector<Depth::InverseGaussian> *ds) {
  std::vector<int> &iKF2X = m_idxsTmp1, &iX2d = m_idxsTmp2;
  const int nKFs = static_cast<int>(m_KFs.size());
  iKF2X.assign(nKFs, -1);
  iX2d.resize(0);

  int Nd = 0;
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    if (!(ucs[iKF] & GBA_FLAG_FRAME_UPDATE_DEPTH)) {
      continue;
    }
    const ubyte *_uds = uds.data() + m_iKF2d[iKF];
    const int iX = static_cast<int>(iX2d.size()), Nx = static_cast<int>(m_KFs[iKF].m_xs.size());
    iKF2X[iKF] = iX;
    iX2d.resize(iX + Nx, -1);
    int *ix2d = iX2d.data() + iX;
    for (int ix = 0; ix < Nx; ++ix) {
      if (_uds[ix] & GBA_FLAG_TRACK_UPDATE_DEPTH) {
        ix2d[ix] = Nd++;
      }
    }
  }

  int Nt = 0;
  m_idxsTmp3.resize(Nd + Nd + 1);
  int *Nzs = m_idxsTmp3.data(), *id2z = Nzs + Nd;
  memset(Nzs, 0, sizeof(int) * Nd);
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    const KeyFrame &KF = m_KFs[iKF];
#ifdef CFG_STEREO
    const int iX = iKF2X[iKF];
    if (iX != -1) {
      const int *ix2d = iX2d.data() + iX;
      const int Nx = static_cast<int>(KF.m_xs.size());
      for (int ix = 0; ix < Nx; ++ix) {
        const int id = ix2d[ix];
        if (id != -1 && KF.m_xs[ix].m_xr.Valid()) {
          ++Nzs[id];
        }
      }
    }
#endif
    const int NZ = static_cast<int>(KF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ) {
      const FRM::Measurement &Z = KF.m_Zs[iZ];
      const int _iX = iKF2X[Z.m_iKF];
      if (_iX == -1) {
        continue;
      }
      bool t = false;
      const int *_ix2d = iX2d.data() + _iX;
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
        const FTR::Measurement &z = KF.m_zs[iz];
        const int id = _ix2d[z.m_ix];
        if (id == -1) {
          continue;
        }
#ifdef CFG_STEREO
        if (z.m_z.Valid()) {
          ++Nzs[id];
        }
        if (z.m_zr.Valid()) {
          ++Nzs[id];
        }
#else
        ++Nzs[id];
#endif
        t = true;
      }
      if (t) {
        ++Nt;
      }
    }
  }
#ifdef CFG_STEREO
  m_t12s.Resize(Nt + Nt);
#else
  m_t12s.Resize(Nt);
#endif
  id2z[0] = 0;
  for (int id = 0, iz = 0; id < Nd; ++id) {
//#ifdef CFG_DEBUG
#if 0
    UT_ASSERT(Nzs[id] > 0);
#endif
    id2z[id + 1] = id2z[id] + Nzs[id];
  }
  m_zds.resize(id2z[Nd]);

  LA::Vector3f Rx;
  //LA::SymmetricMatrix2x2f W;
  Nt = 0;
  memset(Nzs, 0, sizeof(int) * Nd);
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    const KeyFrame &KF = m_KFs[iKF];
#ifdef CFG_STEREO
    const int iX = iKF2X[iKF];
    if (iX != -1) {
      const int *ix2d = iX2d.data() + iX;
      const int Nx = static_cast<int>(KF.m_xs.size());
      for (int ix = 0; ix < Nx; ++ix) {
        const int id = ix2d[ix];
        const FTR::Source &x = KF.m_xs[ix];
        if (id == -1 || x.m_xr.Invalid()) {
          continue;
        }
        Rx.Set(x.m_x.x(), x.m_x.y(), 1.0f);
        //x.m_Wr.GetScaled(KF.m_Ards[ix].m_wx, W);
        const LA::SymmetricMatrix2x2f &W = x.m_Wr;
        const int iz = id2z[id] + Nzs[id]++;
        m_zds[iz].Set(m_K.m_br, Rx, x.m_xr, W);
      }
    }
#endif
    const Rigid3D &C = Cs[iKF];
    const int NZ = static_cast<int>(KF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ) {
      const FRM::Measurement &Z = KF.m_Zs[iZ];
      const int _iX = iKF2X[Z.m_iKF];
      if (_iX == -1) {
        continue;
      }
      const int *_ix2d = iX2d.data() + _iX;
      bool found = false;
      for (int iz = Z.m_iz1; iz < Z.m_iz2 && !found; ++iz) {
        found = _ix2d[KF.m_zs[iz].m_ix] != -1;
      }
      if (!found) {
        continue;
      }
      const Rigid3D T = C / Cs[Z.m_iKF];
      LA::AlignedVector3f *t = m_t12s.Data() + Nt++;
      T.GetTranslation(*t);
#ifdef CFG_STEREO
      LA::AlignedVector3f::apb(t[0], m_K.m_br, t[1]);
      ++Nt;
#endif
      const KeyFrame &_KF = m_KFs[Z.m_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
        const FTR::Measurement &z = KF.m_zs[iz];
        const int id = _ix2d[z.m_ix];
        if (id == -1) {
          continue;
        }
        T.ApplyRotation(_KF.m_xs[z.m_ix].m_x, Rx);
#ifdef CFG_STEREO
        if (z.m_z.Valid()) {
          //z.m_W.GetScaled(KF.m_Lzs[iz].m_wx, W);
          const LA::SymmetricMatrix2x2f &W = z.m_W;
          const int i = id2z[id] + Nzs[id]++;
          m_zds[i].Set(t[0], Rx, z.m_z, W);
        }
        if (z.m_zr.Valid()) {
          //z.m_Wr.GetScaled(KF.m_Lzs[iz].m_wxr, W);
          const LA::SymmetricMatrix2x2f &W = z.m_Wr;
          const int i = id2z[id] + Nzs[id]++;
          m_zds[i].Set(t[1], Rx, z.m_zr, W);
        }
#else
        //z.m_W.GetScaled(KF.m_Lzs[iz].m_wx, W);
        const LA::SymmetricMatrix2x2f &W = z.m_W;
        const int i = ++Nzs[id];
        m_zds[i].Set(*t, Rx, z.m_z, z.m_W);
#endif
      }
    }
  }

  for (int iKF = 0; iKF < nKFs; ++iKF) {
    const int iX = iKF2X[iKF];
    if (iX == -1) {
      continue;
    }
    Depth::InverseGaussian *_ds = ds->data() + m_iKF2d[iKF];
    const int *ix2d = iX2d.data() + iX;
    const int Nx = static_cast<int>(m_KFs[iKF].m_xs.size());
    for (int ix = 0; ix < Nx; ++ix) {
      const int id = ix2d[ix];
      if (id == -1) {
        continue;
      }
      Depth::InverseGaussian &d = _ds[ix];
      const Depth::InverseGaussian dBkp = d;
      if (!Depth::Triangulate(BA_WEIGHT_FEATURE, Nzs[id], m_zds.data() + id2z[id], &d, &m_work, true)) {
        d = dBkp;
      }
    }
  }
}
