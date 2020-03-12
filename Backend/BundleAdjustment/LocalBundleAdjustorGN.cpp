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
//#define LBA_DEBUG_EIGEN_PCG
#endif
#include "LocalBundleAdjustor.h"
#include "Vector12.h"

#if defined WIN32 && defined CFG_DEBUG && defined CFG_GROUND_TRUTH
//#define LBA_DEBUG_GROUND_TRUTH_STATE
//#ifdef LBA_DEBUG_GROUND_TRUTH_STATE
//#define LBA_DEBUG_GROUND_TRUTH_STATE_ERROR
//#endif
#endif

//#ifdef CFG_DEBUG
#if 0
#ifdef LBA_ME_FUNCTION
#undef LBA_ME_FUNCTION
#define LBA_ME_FUNCTION ME::FUNCTION_NONE
#endif
#endif
int LBAdebug_count = -1;
std::string lba_debug_file = "/home/wya/ICE-BA-Debug/ba/lba.txt";
void LocalBundleAdjustor::UpdateFactors()
{
    LBAdebug_count++;
#ifdef CFG_VERBOSE
  if (m_verbose >= 3) {
    UT::PrintSeparator();
    UT::Print("*%2d: [LocalBundleAdjustor::UpdateFactors]\n", m_iIter);
  }
#endif
  const int nLFs = static_cast<int>(m_LFs.size());//滑窗中的普通帧数量
  for (int iLF = 0; iLF < nLFs; ++iLF) {//遍历每一帧,如果要更新对应的矩阵块,就置0
    if (m_ucsLF[iLF] & LBA_FLAG_FRAME_UPDATE_CAMERA) {
      m_SAcusLF[iLF].MakeZero();//pose是否要更新
    }
    if (m_ucmsLF[iLF]) {//motion部分是否要更新
      m_SAcmsLF[iLF].MakeZero();
    }
  }
  const ubyte ucFlag = LBA_FLAG_FRAME_PUSH_TRACK | LBA_FLAG_FRAME_UPDATE_DEPTH;//push进轨迹或者更新深度
  const ubyte udFlag = LBA_FLAG_TRACK_PUSH | LBA_FLAG_TRACK_UPDATE_DEPTH;//轨迹push或者更新深度
  const float add = UT::Inverse(BA_VARIANCE_REGULARIZATION_DEPTH, BA_WEIGHT_FEATURE);
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF)
  {//判断一下这个关键帧需不需要更新
    if (!(m_ucsKF[iKF] & ucFlag))
    {
      continue;
    }
    const ubyte *uds = m_uds.data() + m_iKF2d[iKF];//
    KeyFrame &KF = m_KFs[iKF];
    const int Nx = static_cast<int>(KF.m_xs.size());
    for (int ix = 0; ix < Nx; ++ix) {//遍历所有新地图点
      if (!(uds[ix] & udFlag))
      {//如果不需要更新轨迹或者更新深度
        continue;
      } else if (uds[ix] & LBA_FLAG_TRACK_UPDATE_DEPTH)
      {//如果需要更新深度,就把这个地图点相关的因子都清0,反正一些也要把LFs全部遍历一遍
        KF.m_Axps[ix].MakeZero();
        KF.m_AxpsST[ix].MakeZero();
        KF.m_Axs[ix].MakeZero();
        const int iST1 = KF.m_ix2ST[ix], iST2 = KF.m_ix2ST[ix + 1];
        KF.m_AxsST.MakeZero(iST1, iST2 - iST1);//iST2 - iST1表示这个ix点的子轨迹的数量
        KF.m_Axs[ix].m_Sadd.m_a = add;
        for (int iST = iST1; iST < iST2; ++iST) {
          KF.m_AxsST[iST].m_Sadd.m_a = add;
        }
      } else {//有新的子轨迹生成
        const int iST1 = KF.m_ix2ST[ix], iST2 = KF.m_ix2ST[ix + 1];
        for (int iST = iST2 - 1; iST >= iST1 && (KF.m_usST[iST] & LBA_FLAG_TRACK_PUSH); --iST) {
          KF.m_AxsST[iST].m_Sadd.m_a = add;
        }
      }
    }
  }
  UpdateFactorsFeaturePriorDepth();//视觉约束的H|-b的更新
//#ifdef CFG_DEBUG
#if 0
  const int iLF = m_ic2LF.back();
  m_SAcusLF[iLF].m_b.Print(true);
#endif
  if(!IMU_GRAVITY_EXCLUDED)
  {
      //运动先验部分就是对滑窗内的最老帧进行约束,m_ZpLF里存储着滑窗后对于最老帧的motion先验约束,残差rv,rba,rbg，其中rv =Rc0w * Vw所以优化变量除了v,ba,bg还有rv
      //要更新的H右上角的部分就是R0 X motion0,R0 X R0, motion0 X motion0 b 就是 R0,motion0
      // m_SAcusLF[iLF] 里更新普通帧的pose自己和自己的H以及自己对应的-b
      //m_SAcmsLF[iLF].m_Au里更新 motion0 X motion0,R0 X motion0, -bmotion0
      UpdateFactorsPriorCameraMotion();
//更新imu约束所带来的因子
//每个imu约束的因子,都连接前后两帧的pose和motion
//// 残差:
// e_r = -ln{预积分的Rij * exp[Jrbw *(bwi - z_bw)]x * Rcjw * Rciw.t}v
//  e_v = Rciw*(v_wj - v_wi + 0.5gt^2) - (m_v + m_Jvba * (bai - m_ba) + m_Jvbw * (bwi - m_bw))
//  e_p = Rciw*(p_wcj - p_wci - v_wci*dt + 0.5*g*dt^2) +
//  Rciw * Rcjw.t*tc0i - tc0i - (m_p + m_Jpba * (bai - m_ba) + m_Jpbw * (bwi - m_bw))
// e_ba = bai - baj
// e_bw = bwi - bwj
      UpdateFactorsIMU();
  }

  //UpdateFactorsFixOrigin();
  UpdateFactorsFixPositionZ();//这个是固定滑床内的twc0.z,但是H,b都是0,加上也没有用啊
  UpdateFactorsFixMotion();//同上,在LBA里并没有贡献,我给注释了
//#ifdef CFG_DEBUG
#if 0
  m_SAcusLF[iLF].m_b.Print(true);
#endif
}

#ifdef CFG_VERBOSE
static int g_SNzLF, g_SNZLF, g_SNzLFST, g_SNZLFST;
static int g_SNzKF, g_SNZKF;
static int g_SNd, g_SNdST;
#endif

void LocalBundleAdjustor::UpdateFactorsFeaturePriorDepth() {
#ifdef CFG_VERBOSE
  if (m_verbose >= 3) {
    g_SNzLF = g_SNZLF = g_SNzLFST = g_SNZLFST = 0;
    g_SNzKF = g_SNZKF = 0;
    g_SNd = g_SNdST = 0;
  }
#endif

// 在LBA中是不会优化KF的pose的,所以所有关于KF的pose的雅克比是不求的,置0


// Pcl = Tclw * Tckw.inv *Pck
//这里进行计算的就是H的普通帧poseX普通帧pose,逆深度X逆深度,普通帧poseX逆深度,b的普通帧pose,逆深度对应的矩阵块 5个块
//遍历所有滑窗中普通帧,然后遍历它们对于地图点的观测,如果关键帧pose和普通帧pose还有逆深度都不要更新的话,就跳过不进行因子的更新
//如果是一个新的因子,那么在矩阵块里+=既可以了,如果是一个已有的因子需要更新,那么就+=(新因子 - 老因子)就可以了
//一个地图点,在它所在的KF中的索引是ix,KF.m_Axs[ix].m_Sadd里是总的逆深度X逆深度,逆深度的-b,进行这个因子的更新
//同时一个地图点可能有多条子轨迹,KF.m_AxsST[iST].m_Sadd里就是每个点的每条轨迹的总的逆深度X逆深度,逆深度的-b,这里是实际的1/轨迹条数
//m_SAcusLF[iLF]就是滑窗中每一个普通帧poseX普通帧pose的H以及普通帧pose对应的-b,也进行更新这个因子的更新
  UpdateFactorsFeatureLF();
// Pck2 = Tck2w * Tck1w.inv *Pck1 这里就是一个关键帧观测到了这个地图点,K1就是首次观测到地图点的关键帧
//因为固定KF的pose,所以这里就只求逆深度的扰动
//KF.m_Axs[ix].m_Sadd里是总的逆深度X逆深度,逆深度的-b,进行这个因子的更新
//KF.m_Axps[ix].m_Sadd里是只由关键帧之间的观测引起的逆深度X逆深度,逆深度的-b,进行这个因子的更新
  UpdateFactorsFeatureKF();
  //遍历每个关键帧的每一个地图点,如果需要更新的话,如果只有左目观测,那么就用r = 平均地图点逆深度- 地图点逆深度 做约束
  //如果有双目观测,那么就用r = 归一化(Pnc0 - Uc0 * tc0c1) - 归一化(Rc0c1 *Pnc1) 做约束
  //KF.m_Axs[ix].m_Sadd里是总的逆深度X逆深度,逆深度的-b,进行这个因子的更新
  //同时一个地图点可能有多条子轨迹,KF.m_AxsST[iST].m_Sadd里就是每个点的每条轨迹的总的逆深度X逆深度,逆深度的-b,这里是实际的1/轨迹条数
  UpdateFactorsPriorDepth();
#ifdef CFG_VERBOSE
  if (m_verbose >= 3) {
    const int NzLF = CountMeasurementsFeatureLF(), NZLF = CountMeasurementsFrameLF();
    const int NzKF = CountMeasurementsFeatureKF(), NZKF = CountMeasurementsFrameKF();
    const int NzLFST = NzLF - g_SNzLF, NZLFST = NZLF - g_SNZLF;
    const int NzKFST = NzKF - g_SNzKF, NZKFST = NZKF - g_SNZKF;
    UT::Print("  FeatureLF = %4d / %4d = %.2f%% (%d / %2d = %.2f%%)\n", g_SNzLF, NzLF,
              UT::Percentage(g_SNzLF, NzLF), g_SNZLF, NZLF, UT::Percentage(g_SNZLF, NZLF));
    if (g_SNZLFST > 0 || g_SNzLFST > 0) {
      UT::Print("            + %4d / %4d = %.2f%% (%d / %2d = %.2f%%)\n", g_SNzLFST, NzLFST,
                UT::Percentage(g_SNzLFST, NzLFST), g_SNZLFST, NZLFST, UT::Percentage(g_SNZLFST, NZLFST));
    }
    UT::Print("  FeatureKF = %4d / %4d = %.2f%% (%d / %2d = %.2f%%)\n", g_SNzKF, NzKF,
              UT::Percentage(g_SNzKF, NzKF), g_SNZKF, NZKF, UT::Percentage(g_SNZKF, NZKF));
    const int Nd = int(m_ds.size()), NdST = Nd - g_SNd;
    UT::Print("  Prior Depth  = %d / %d = %.2f%% + %d / %d = %.2f%%\n", g_SNd, Nd,
              UT::Percentage(g_SNd, Nd), g_SNdST, NdST, UT::Percentage(g_SNdST, NdST));
  }
#endif
}

//更新滑窗中帧和它们的观测的因子(针对地图点的重投影误差),具体看FTR::GetFactor里的注释
//每个视觉约束,即重投影残差的因子,都连接投影前的那帧pose,投影后的那帧pose还有地图点的逆深度(就是在首次观测到它的这帧中的逆深度),
// 针对UpdateFactorsFeatureLF 投影前是关键帧,投影后是普通帧
//也就是对于H的上三角,有6处影响(关键帧poseX关键帧pose，普通帧poseX普通帧pose,逆深度X逆深度,关键帧poseX普通帧pose,关键帧poseX逆深度,普通帧poseX逆深度)
// b有3处(关键帧pose,普通帧pose,逆深度)
//这里进行计算的就是H的普通帧poseX普通帧pose,逆深度X逆深度,普通帧poseX逆深度,b的普通帧pose,逆深度对应的矩阵块 5个块
//遍历所有滑窗中普通帧,然后遍历它们对于地图点的观测,如果关键帧pose和普通帧pose还有逆深度都不要更新的话,就跳过不进行因子的更新
//如果是一个新的因子,那么在矩阵块里+=既可以了,如果是一个已有的因子需要更新,那么就+=(新因子 - 老因子)就可以了
//一个地图点,在它所在的KF中的索引是ix,KF.m_Axs[ix].m_Sadd里是总的逆深度X逆深度,逆深度的-b,进行这个因子的更新
//同时一个地图点可能有多条子轨迹,KF.m_AxsST[iST].m_Sadd里就是每个点的每条轨迹的总的逆深度X逆深度,逆深度的-b,这里是实际的1/轨迹条数
//m_SAcusLF[iLF]就是滑窗中每一个普通帧poseX普通帧pose的H以及普通帧pose对应的-b,也进行更新这个因子的更新
void LocalBundleAdjustor::UpdateFactorsFeatureLF() {

//    std::ofstream foutC(lba_debug_file, std::ios::app);
//    foutC.setf(std::ios::fixed, std::ios::floatfield);
//    foutC << "UpdateFactorsFeatureLF 第几次优化:"<<LBAdebug_count << "\n";
  Rigid3D Tr[2];
  FTR::Factor::DD dadd, daddST;//就是一些中间状态，因子已存在但需要更新,所以要保存一些这个老的因子
  Camera::Factor::Unitary::CC dAczz;
  FTR::Factor::FixSource::U U;
  //float dF;
  const ubyte ucFlag = LBA_FLAG_FRAME_PUSH_TRACK | LBA_FLAG_FRAME_POP_TRACK |
                       LBA_FLAG_FRAME_UPDATE_DEPTH;
  const ubyte udFlag = LBA_FLAG_TRACK_PUSH | LBA_FLAG_TRACK_POP |
                       LBA_FLAG_TRACK_UPDATE_DEPTH;
  const int nLFs = static_cast<int>(m_LFs.size());
  for (int iLF = 0; iLF < nLFs; ++iLF)
  {//遍历所有滑窗中的帧
    LocalFrame &LF = m_LFs[iLF];//当前普通帧
    const Rigid3D &C = m_CsLF[iLF].m_Cam_pose;//当前这个普通帧的pose
    const bool ucz = (m_ucsLF[iLF] & LBA_FLAG_FRAME_UPDATE_CAMERA) != 0;//是否更新这个滑窗内普通帧的pose
    Camera::Factor::Unitary::CC &SAczz = m_SAcusLF[iLF];//m_A是这帧pose和这帧pose的H,m_b存储这帧pose的-b
    const int NZ = static_cast<int>(LF.m_Zs.size());//这个普通帧观测到的关键帧数量
    for (int iZ = 0; iZ < NZ; ++iZ)//遍历所有对于关键帧的观测
    {
      const FRM::Measurement &Z = LF.m_Zs[iZ];
      const int iKF = Z.m_iKF;//观测到的kf_id
      const bool ucx = (m_ucsKF[iKF] & LBA_FLAG_FRAME_UPDATE_CAMERA) != 0/*是否要更新关键帧*/, ucr = ucx || ucz;//只要更新关键帧或者普通帧就是true
      if (!ucr && !(m_ucsKF[iKF] & ucFlag))
      {
        continue;
      }
      const bool pushFrm = (m_ucsKF[iKF] & LBA_FLAG_FRAME_PUSH_TRACK) != 0;//是否要push这帧
      *Tr = C / m_CsKF[iKF];//Tc0(LF)_c0(KF) = Tc0w(LF) * Tc0w.t(KF)
#ifdef CFG_STEREO
      Tr[1] = Tr[0];
      Tr[1].SetTranslation(m_K.m_br + Tr[0].GetTranslation());//Tc1(LF)_c0(KF) //这里是考虑成外参的R是I了
#endif
      const int id = m_iKF2d[iKF];
      ubyte *uds = m_uds.data() + id;
      const Depth::InverseGaussian *ds = m_ds.data() + id;
      KeyFrame &KF = m_KFs[iKF];//当前这个观测的关键帧
      for (int iz/*起始id*/ = Z.m_iz1; iz < Z.m_iz2; ++iz)//遍历所有对于这个关键帧内地图点的观测
      {
//#ifdef CFG_DEBUG
#if 0
        if (iLF == m_ic2LF[12]) {
          UT::Print("%d: %.10e\n", iz, SAczz.m_b.v4());
        }
#endif
        const FTR::Measurement &z = LF.m_zs[iz];//当前帧对这个地图点的观测
        const int ix = z.m_ix;//这个地图点的局部id
        if (!ucr && !(uds[ix] & udFlag))
        {
          continue;
        }
        //举个例子
        //有一个点它在0-6帧中都出现的话:01234,12345,23456这个是目前这个点的子轨迹,而帧对5号帧来说,那么12345,23456就是它所属于的子轨迹
        //假设01234在KF中m_STs的id=5,那么iST0 = 5,iST1 = 5,iST2 =7,ST.m_ist1 =0,ST.m_ist2 =2
        const LocalFrame::SlidingTrack &ST = LF.m_STs[iz];//滑窗中对这个地图点观测的轨迹位置
        const int iST0 = KF.m_ix2ST[ix]/*KF中这个地图点的第一条子轨迹索引*/, iST1 = iST0 + ST.m_ist1/*LF的起始子轨迹*/, iST2 = iST0 + ST.m_ist2;//LF的终止子轨迹
        const int Nst = iST2 - iST1;//轨迹的数量,就比如我现在id为4的帧,它在01234，12345，23456都出现的话,那么这里就是在3条子轨迹都出现了,就是3了
        FTR::Factor::FixSource::A2 &A = LF.m_Azs2[iz];//m_add里的m_a就存的是逆深度和逆深度的H,逆深度的-b，m_Aczz里的m_A存储普通帧pose和普通帧pose的H,m_b存储普通帧pose的-b
        FTR::Factor::FixSource::A3 &AST = LF.m_AzsST[iz];//m_adcA:存的是这个地图点的逆深度和普通帧pose的H m_add存的是逆深度和逆深度的H,逆深度的-b
        const bool ud = (uds[ix] & LBA_FLAG_TRACK_UPDATE_DEPTH) != 0;//是否要更新深度
        const bool pushST = pushFrm && (KF.m_usST[iST2 - 1] & LBA_FLAG_TRACK_PUSH/*这个点的最后一条子轨迹是否需要push,一般刚生成一条新轨迹需要push*/);
        if (ucr || ud)//如果更新关键帧或者滑窗中的普通帧或者点的深度
        {
          if (!ud)//如果不需要更新深度的话
          {//保存一下老的这个特征点的因子
            dadd = A.m_add;//逆深度和逆深度的H,逆深度的-b
            daddST = AST.m_add;//子轨迹的逆深度和逆深度的H,逆深度的-b
          }
          if (!ucz)//这个滑窗内普通帧状态不用更新的话,也就是老帧
          {
            dAczz = A.m_Aczz;//存储一下旧的这个因子m_A存储普通帧pose和普通帧pose的H,m_b存储普通帧pose的-b
          }
          //dF = A.m_F;//以后说逆深度就是指这个点在它首次观测的关键帧中的逆深度
          FTR::GetFactor<LBA_ME_FUNCTION>(BA_WEIGHT_FEATURE, Tr/*Tc(LF)_c0(KF)*/, KF.m_xs[ix]/*关键帧对这个地图点的观测*/, ds[ix]/*逆深度*/
                  , C/*当前帧Tc0w*/, z,/*当前帧对这个地图点的观测*/
                                          &LF.m_Lzs[iz]/*当前帧中对这个地图点观测的重投影误差e,J(对当前帧的pose,对关键帧点的逆深度),cost*/,
                                          &LF.m_Azs1[iz],/*m_adcz存的是当前这个普通帧pose和观测到的地图点在kf中逆深度的H*/
                                          &A,//A->m_add里的m_a就存的是逆深度和逆深度的H,逆深度的-b，A->m_Aczz里的m_A存储普通帧pose和普通帧pose的H,m_b存储普通帧pose的-b
                                          &U/*这个地图点从关键帧投影到当前帧上的重投影误差的因子,存了H|-b,信息矩阵*/
#ifdef CFG_STEREO
                                        , m_K.m_br/*-tc0_c1*/
#endif
                                        );
//            foutC.precision(0);
//            foutC <<"遍历lf:"<<iLF<<",观测到kf:"<<iZ<<",点id:"<< ix<<",";
//            foutC.precision(5);
//            foutC << A.m_add.m_a << ","<< A.m_add.m_b << std::endl;
          AST.Set(A.m_add/*逆深度和逆深度的H,逆深度的-b*/, LF.m_Azs1[iz].m_adczA/*这个地图点的逆深度和普通帧pose的H*/);//拷贝一下副本
          if (Nst >= 1)
          {
            AST *= 1.0f / Nst;//这里为啥要除个子轨迹数量还不是太清楚
          }
          LF.m_Nsts[iz] = Nst;
          if (ud)//如果需要更新深度的话,也就是新来了一个地图点的观测
          {
//#ifdef CFG_DEBUG
#if 0
            if (iKF == 21 && ix == 316) {
              UT::Print(" [%d] %d: [%d] %e + %e = %e\n", m_LFs[m_ic2LF.back()].m_Cam_pose.m_iFrm, m_iIter, LF.m_Cam_pose.m_iFrm,
                        KF.m_Axs[ix].m_Sadd.m_a, A.m_add.m_a, KF.m_Axs[ix].m_Sadd.m_a + A.m_add.m_a);
            }
#endif
            KF.m_Axs[ix].m_Sadd += A.m_add;//如果要更新的话就在这个地图点对应的位置加上逆深度和逆深度的H,逆深度的-b
            for (int iST = iST1; iST < iST2; ++iST)//遍历这个地图点所有的子轨迹
            {
              KF.m_AxsST[iST].m_Sadd += AST.m_add;//如果要更新的话就在这个地图点所有子轨迹对应的位置加上逆深度和逆深度的H,逆深度的-b,不过这里都除了一个轨迹长度
              KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
            }
          } else
          {//说明是老的观测
              FTR::Factor::DD::amb(A.m_add, dadd, dadd);//新的这个因子减旧的这个因子
//#ifdef CFG_DEBUG
#if 0
              if (iKF == 21 && ix == 316) {
              UT::Print(" [%d] %d: [%d] %e + %e = %e\n", m_LFs[m_ic2LF.back()].m_Cam_pose.m_iFrm, m_iIter, LF.m_Cam_pose.m_iFrm,
                        KF.m_Axs[ix].m_Sadd.m_a, dadd.m_a, KF.m_Axs[ix].m_Sadd.m_a + dadd.m_a);
            }
#endif
              KF.m_Axs[ix].m_Sadd += dadd;//加上新的因子,减去旧的因子
              if (pushST)//如果这个特征点新增了一条子轨迹的话
              {
                  int iST;//从最新的轨迹开始往前找,如果这条是个新轨迹的话（根据KF.m_usST[iST]）,那么这个就是一个新的因子,就直接加到对应的矩阵块里就可以
                  for (iST = iST2 - 1; iST >= iST1 && (KF.m_usST[iST] & LBA_FLAG_TRACK_PUSH); --iST)
                  {
                      KF.m_AxsST[iST].m_Sadd += AST.m_add;
                      KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
                  }
                  FTR::Factor::DD::amb(AST.m_add, daddST, daddST);//剩下的就是老轨迹,因子已经存在了,还是加上新的减去旧的
                  for (; iST >= iST1; --iST)
                  {
                      KF.m_AxsST[iST].m_Sadd += daddST;
                      KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
                  }
              }
              else
              {//如果没有新增子轨迹那么也是加上新的因子,减去旧的因子然后更新状态
                  FTR::Factor::DD::amb(AST.m_add, daddST, daddST);
                  for (int iST = iST1; iST < iST2; ++iST)
                  {
                      KF.m_AxsST[iST].m_Sadd += daddST;
                      KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
                  }
              }
          }
          if (ucz)//需要更新位姿一般都是新来了一帧也就是来了一个新的因子,所以直接+=
          {
            SAczz += A.m_Aczz;//m_SAcusLF[iLF]所维护的这个普通帧自己和自己的H以及自己对应的-b就需要加上这次的约束
//#ifdef CFG_DEBUG
#if 0
            UT::Print("%d: %e + %e = %e\n", iz, A.m_Aczz.m_A.m00(), SAczz.m_A.m00(), A.m_Aczz.m_A.m00() + SAczz.m_A.m00());
#endif
          } else
          {//这种情况就是老帧,就把旧的因子减去加上新的
              Camera::Factor::Unitary::CC::AmB(A.m_Aczz, dAczz, dAczz);
              SAczz += dAczz;//SAczz += A.m_Aczz - dAczz 加上新的因子,减去旧的因子
          }
          //dF = A.m_F - dF;
          //m_F = dF + m_F;
          m_ucsKF[iKF] |= LBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION;//更新这个关键帧和地图点的状态
          uds[ix] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
#ifdef CFG_VERBOSE
          if (m_verbose >= 3)
          {
            ++g_SNzLF;
          }
#endif
        } else if (pushST || LF.m_Nsts[iz] != Nst)//如果不需要更新关键帧pose||地图点逆深度||普通帧pose但是有新的子轨迹生成或者说
        {
          if (LF.m_Nsts[iz] == Nst)//这种情况不可能发生Nst肯定比LF.m_Nsts[iz]大1
          {
#ifdef CFG_DEBUG
            UT_ASSERT(pushST);
#endif
            for (int iST = iST2 - 1; iST >= iST1 && (KF.m_usST[iST] & LBA_FLAG_TRACK_PUSH); --iST)
            {
              KF.m_AxsST[iST].m_Sadd += AST.m_add;
              KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
            }
          }
          else
          {
              daddST = AST.m_add;//这个应该是这个轨迹之前哪个轨迹的因子
              AST.Set(A.m_add, LF.m_Azs1[iz].m_adczA);//老的因子,因为轨迹加1了需要重新的除一下
              if (Nst >= 1)
              {
                  AST *= 1.0f / Nst;//设置成新的因子
              }
              LF.m_Nsts[iz] = Nst;//更新一下轨迹数量
              if (pushST)
              {//从最新的轨迹开始往前找,如果这条是个新轨迹的话（根据KF.m_usST[iST]）,那么这个就是一个新的因子,就直接加到对应的矩阵块里就可以
                  int iST;
                  for (iST = iST2 - 1; iST >= iST1 && (KF.m_usST[iST] & LBA_FLAG_TRACK_PUSH); --iST)
                  {
                      KF.m_AxsST[iST].m_Sadd += AST.m_add;
                      KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
                  }
                  FTR::Factor::DD::amb(AST.m_add, daddST, daddST);
                  for (; iST >= iST1; --iST)//剩下的就是老轨迹,因子已经存在了,还是加上新的减去旧的
                  {
                      KF.m_AxsST[iST].m_Sadd += daddST;
                      KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
                  }
              }
              else//这种就是虽然没有生成新的子轨迹,但是子轨迹需要更新
              {
                  FTR::Factor::DD::amb(AST.m_add, daddST, daddST);
                  for (int iST = iST1; iST < iST2; ++iST)//都是老轨迹,因子已经存在了,还是加上新的减去旧的
                  {
                      KF.m_AxsST[iST].m_Sadd += daddST;
                      KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
                  }
              }
          }
          m_ucsKF[iKF] |= LBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION;
#ifdef CFG_VERBOSE
          if (m_verbose >= 3)
            ++g_SNzLFST;
#endif
        }
//#ifdef CFG_DEBUG
#if 0
        if (iLF == m_ic2LF.back()) {
        //if (iLF == m_ic2LF[12]) {
          UT::Print("%d: %.10e %.10e\n", iz, A.m_Aczz.m_b.v0(), SAczz.m_b.v0());
        }
#endif
      }
#ifdef CFG_VERBOSE
      if (m_verbose >= 3)
      {
        if (ucr || (m_ucsKF[iKF] & LBA_FLAG_FRAME_UPDATE_DEPTH)) {
          ++g_SNZLF;
        } else {
          ++g_SNZLFST;
        }
      }
#endif
    }
  }
//  foutC.close();
}

void LocalBundleAdjustor::UpdateFactorsFeatureKF()
{
//    std::ofstream foutC(lba_debug_file, std::ios::app);
//    foutC.setf(std::ios::fixed, std::ios::floatfield);
//    foutC << "UpdateFactorsFeatureKF 第几次优化:"<<LBAdebug_count << "\n";
  Rigid3D Tr[2];
  FTR::Factor::DD dadd;
  FTR::Factor::Depth::U U;
  //float dF;
  const ubyte ucFlag = LBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION |
                       LBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION_KF;
  const ubyte udFlag = LBA_FLAG_TRACK_UPDATE_INFORMATION | LBA_FLAG_TRACK_UPDATE_INFORMATION_KF;
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF) //遍历所有的关键帧
  {
    KeyFrame &KF = m_KFs[iKF];//当前关键帧
    const Rigid3D &C = m_CsKF[iKF];//对应关键帧的左相机位姿
    const bool ucz = (m_ucsKF[iKF] & LBA_FLAG_FRAME_UPDATE_CAMERA) != 0;//z代表的是投影后那帧的意思,x是代表投影前那帧的意思
    const int NZ = int(KF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ)//第二个关键帧才会有这个
    {
      const FRM::Measurement &Z = KF.m_Zs[iZ];//关键帧的观测
      const int _iKF = Z.m_iKF;//所观测到的关键帧的id
      const bool ucx = (m_ucsKF[_iKF] & LBA_FLAG_FRAME_UPDATE_CAMERA) != 0, ucr = ucx || ucz;//只要投影的这两帧有一帧需要更新
      if (!ucr && !(m_ucsKF[_iKF] & LBA_FLAG_FRAME_UPDATE_DEPTH))
      {
        continue;
      }
      *Tr = C / m_CsKF[_iKF];//Tc0(投影后kf)_c0(投影前kf))
#ifdef CFG_STEREO
      Tr[1] = Tr[0];
      Tr[1].SetTranslation(m_K.m_br + Tr[0].GetTranslation());
#endif
      const int id = m_iKF2d[_iKF];
      ubyte *_uds = m_uds.data() + id;
      const Depth::InverseGaussian *_ds = m_ds.data() + id;
      KeyFrame &_KF = m_KFs[_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz)//遍历所有关键帧的观测
      {
        const FTR::Measurement &z = KF.m_zs[iz];
        const int ix = z.m_ix;//在首次观测关键帧中的局部id
        const bool ud = (_uds[ix] & LBA_FLAG_TRACK_UPDATE_DEPTH) != 0;
        if (!ucr && !ud)
        {
          continue;
        }
        FTR::Factor::Depth &A = KF.m_Azs[iz];//这个关键帧中维护了这个视觉约束的因子,A->m_add里的m_a就存的是逆深度和逆深度的H,逆深度的-b
        if (!ud)
        {
          dadd = A.m_add;//老的
        }
        //dF = A.m_F;
        //因为固定KF的pose,所以只求了逆深度和逆深度的H|-b
        FTR::GetFactor<LBA_ME_FUNCTION>(BA_WEIGHT_FEATURE_KEY_FRAME, Tr/*Tc0(投影后)_c0(投影前))*/, _KF.m_xs[ix]/*投影前对这个地图点的观测*/, _ds[ix],/*逆深度*/
                                        C/*投影后Tc0w*/, z/*投影后对这个地图点的观测*/,
                                        &A, //A->m_add里的m_a就存的是逆深度和逆深度的H,逆深度的-b
                                        &U/*这个地图点从关键帧投影到当前帧上的重投影误差的因子,存了H|-b,信息矩阵*/
#ifdef CFG_STEREO
                                      , m_K.m_br/*-tc0_c1*/
#endif
                                      );
//          foutC.precision(0);
//          foutC <<"遍历kf:"<<iKF<<",观测到kf:"<<iZ<<",点id:"<< ix<<",";
//          foutC.precision(5);
//          foutC << A.m_add.m_a << ","<< A.m_add.m_b << std::endl;

        if (ud)//是一个新的观测,所以直接+=就可以
        {
//#ifdef CFG_DEBUG
#if 0
          if (_iKF == 21 && ix == 316) {
            UT::Print(" [%d] %d: [%d] %e + %e = %e\n", m_LFs[m_ic2LF.back()].m_Cam_pose.m_iFrm, m_iIter, KF.m_Cam_pose.m_iFrm,
                      _KF.m_Axs[ix].m_Sadd.m_a, A.m_add.m_a, _KF.m_Axs[ix].m_Sadd.m_a + A.m_add.m_a);
          }
#endif
          _KF.m_Axps[ix].m_Sadd += A.m_add;//这个好像是存关键帧之间引进的因子
          _KF.m_Axs[ix].m_Sadd += A.m_add;
        } else
        {//以前观测过,加上新的因子减去旧的因子
//#ifdef CFG_DEBUG
#if 0
            if (_iKF == 21 && ix == 316) {
            UT::Print(" [%d] %d: [%d] %e + %e = %e\n", m_LFs[m_ic2LF.back()].m_Cam_pose.m_iFrm, m_iIter, KF.m_Cam_pose.m_iFrm,
                      _KF.m_Axs[ix].m_Sadd.m_a, dadd.m_a, _KF.m_Axs[ix].m_Sadd.m_a + dadd.m_a);
          }
#endif
            FTR::Factor::DD::amb(A.m_add, dadd, dadd);
            _KF.m_Axps[ix].m_Sadd += dadd;
            _KF.m_Axs[ix].m_Sadd += dadd;
        }
        //dF = A.m_F - dF;
        //m_F = dF + m_F;
        m_ucsKF[_iKF] |= ucFlag;
        _uds[ix] |= udFlag;
#ifdef CFG_VERBOSE
        if (m_verbose >= 3)
        {
          ++g_SNzKF;
        }
#endif
      }
#ifdef CFG_VERBOSE
      if (m_verbose >= 3)
      {
        ++g_SNZKF;
      }
#endif
    }
  }
//  foutC.close();
}

void LocalBundleAdjustor::UpdateFactorsPriorDepth()
{

//    std::ofstream foutC(lba_debug_file, std::ios::app);
//    foutC.setf(std::ios::fixed, std::ios::floatfield);
//    foutC << "UpdateFactorsPriorDepth 第几次优化:"<<LBAdebug_count << "\n";
  FTR::Factor::DD dadd, daddST;
  //float dF;
#ifdef CFG_STEREO
  FTR::Factor::Stereo::U U;
#endif
  const ubyte ucFlag1 = LBA_FLAG_FRAME_PUSH_TRACK | LBA_FLAG_FRAME_POP_TRACK |
                        LBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION_KF;
  const ubyte ucFlag2 = ucFlag1 | LBA_FLAG_FRAME_UPDATE_DEPTH;
  const ubyte udFlag1 = LBA_FLAG_TRACK_PUSH | LBA_FLAG_TRACK_POP |
                        LBA_FLAG_TRACK_UPDATE_INFORMATION_KF;
  const ubyte udFlag2 = udFlag1 | LBA_FLAG_TRACK_UPDATE_DEPTH;
  const int nKFs = static_cast<int>(m_KFs.size());//关键帧的数量
  for (int iKF = 0; iKF < nKFs; ++iKF)
  {
    if (!(m_ucsKF[iKF] & ucFlag2))//如果这个关键帧不需要更新
    {
      continue;
    }
    const bool pushFrm = (m_ucsKF[iKF] & LBA_FLAG_FRAME_PUSH_TRACK) != 0;//是否要push这帧

    m_ucsKF[iKF] &= ~ucFlag1;//将对应ucFlag1所有flags置0
    const int id = m_iKF2d[iKF];
    const Depth::InverseGaussian *ds = m_ds.data() + id;
    ubyte *uds = m_uds.data() + id;
    KeyFrame &KF = m_KFs[iKF];//当前这个关键帧

//      foutC.precision(0);
//      foutC << "kf_id:"<<iKF;
//      foutC.precision(5);
//      foutC <<",平均深度:"<<KF.m_d.u()<<",不确定度:"<<KF.m_d.s2() << std::endl;

    const Depth::Prior zp(KF.m_d.u(), 1.0f / (BA_VARIANCE_PRIOR_FRAME_DEPTH + KF.m_d.s2()));//地图点平均深度初始化一下先验因子
    const int Nx = static_cast<int>(KF.m_xs.size());//这个关键帧的新地图点
    for (int ix = 0; ix < Nx; ++ix)
    {
      if (!(uds[ix] & udFlag2))//如果需要更新,不过这里每个flags啥时候置0啥时候置1我也没有管了
      {
        continue;
      }
      const ubyte ud = uds[ix];//当前这个地图点对应的flags
      uds[ix] &= ~udFlag1;//将udFlag1相关flags置0
      const int iST1 = KF.m_ix2ST[ix]/*当前地图点的子轨迹*/, iST2 = KF.m_ix2ST[ix + 1]/*下一个点的第一条子轨迹*/, Nst = iST2 - iST1;//当前地图点的子轨迹数量
      const bool _ud = (ud & LBA_FLAG_TRACK_UPDATE_DEPTH) != 0;//是否需要更新深度
      if (_ud)//如果要更新深度
      {
#ifdef CFG_STEREO
        if (KF.m_xs[ix].m_xr.Valid())//如果是有双目观测的情况
        {
          FTR::Factor::Stereo &A = KF.m_Ards[ix];
          //dF = acc.m_F;
          FTR::GetFactor<LBA_ME_FUNCTION>(BA_WEIGHT_FEATURE_KEY_FRAME, m_K.m_br/*-tc0_c1*/, ds[ix], KF.m_xs[ix], &A, &U);
          dadd = A.m_add;

//            foutC.precision(0);
//            foutC <<"双目约束"<<"遍历kf:"<<iKF<< ",点id:"<< ix<<",";
//            foutC.precision(5);
//            foutC <<A.m_F<<","<< A.m_add.m_a << ","<< A.m_add.m_b << std::endl;

        } else
#endif
        {
          Depth::Prior::Factor &A = KF.m_Apds[ix];//只有左目观测时候的先验
          //dF = acc.m_F;
          zp.GetFactor<LBA_ME_FUNCTION>(BA_WEIGHT_PRIOR_DEPTH, ds[ix].u()/*逆深度*/, A);
          dadd = A;

//            foutC.precision(0);
//            foutC <<"单目约束"<<"遍历kf:"<<iKF<< ",点id:"<< ix<<",";
//            foutC.precision(5);
//            foutC << A.m_F<< ","<<A.m_a<< ","<< A.m_b << std::endl;
        }
        KF.m_Axps[ix].m_Sadd += dadd;//将左右目约束以及左目和平均场景的约束加入H|-b
        KF.m_Axs[ix].m_Sadd += dadd;
        //dF = acc.m_F - dF;
        //m_F = dF + m_F;
        m_ucsKF[iKF] |= LBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION;
        uds[ix] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
      }
      FTR::Factor::FixSource::Source::A &AST = KF.m_AxpsST[ix];
      if (Nst == 0)
      {
        AST = KF.m_Axps[ix];
        KF.m_Nsts[ix] = Nst;
        continue;
      }
      const bool pushST = pushFrm && (KF.m_usST[iST2 - 1] & LBA_FLAG_TRACK_PUSH);//
      if (_ud || (ud & LBA_FLAG_TRACK_UPDATE_INFORMATION_KF))//后面和UpdateFactorsFeatureLF是一样的就是对子轨迹要分开操作,就不注释了
      {
        if (!_ud)
        {
          daddST = AST.m_Sadd;
        }
        AST = KF.m_Axps[ix];
        if (Nst > 1)
        {
          AST *= 1.0f / Nst;
        }
        KF.m_Nsts[ix] = Nst;
        if (_ud)
        {
          for (int iST = iST1; iST < iST2; ++iST)
          {
            KF.m_AxsST[iST].m_Sadd += AST.m_Sadd;
            KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
          }
          if (pushST)
          {
            for (int iST = iST2 - 1; iST >= iST1 && (KF.m_usST[iST] & LBA_FLAG_TRACK_PUSH); --iST)
            {
              KF.m_usST[iST] &= ~LBA_FLAG_TRACK_PUSH;
            }
          }
        } else
        {
          if (pushST)
          {
            int iST;
            for (iST = iST2 - 1; iST >= iST1 && (KF.m_usST[iST] & LBA_FLAG_TRACK_PUSH); --iST)
            {
              KF.m_AxsST[iST].m_Sadd += AST.m_Sadd;
              KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
              KF.m_usST[iST] &= ~LBA_FLAG_TRACK_PUSH;//取消要Push的flags
            }
            FTR::Factor::DD::amb(AST.m_Sadd, daddST, daddST);
            for (; iST >= iST1; --iST)
            {
              KF.m_AxsST[iST].m_Sadd += daddST;
              KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
            }
          } else
          {
            FTR::Factor::DD::amb(AST.m_Sadd, daddST, daddST);
            for (int iST = iST1; iST < iST2; ++iST)
            {
              KF.m_AxsST[iST].m_Sadd += daddST;
              KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
            }
          }
        }
#ifdef CFG_VERBOSE
        if (m_verbose >= 3)
        {
          ++g_SNd;
        }
#endif
      } else if (pushST || KF.m_Nsts[ix] != Nst)
      {
        if (KF.m_Nsts[ix] == Nst)
        {
#ifdef CFG_DEBUG
          UT_ASSERT(pushST);
#endif
          for (int iST = iST2 - 1; iST >= iST1 && (KF.m_usST[iST] & LBA_FLAG_TRACK_PUSH); --iST)
          {
            KF.m_AxsST[iST].m_Sadd += AST.m_Sadd;
            KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
            KF.m_usST[iST] &= ~LBA_FLAG_TRACK_PUSH;
          }
        } else
        {
          daddST = AST.m_Sadd;
          AST = KF.m_Axps[ix];
          if (Nst > 1)
          {
            AST *= 1.0f / Nst;
          }
          KF.m_Nsts[ix] = Nst;
          if (pushST)
          {
            int iST;
            for (iST = iST2 - 1; iST >= iST1 && (KF.m_usST[iST] & LBA_FLAG_TRACK_PUSH); --iST)
            {
              KF.m_AxsST[iST].m_Sadd += AST.m_Sadd;
              KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
              KF.m_usST[iST] &= ~LBA_FLAG_TRACK_PUSH;
            }
            FTR::Factor::DD::amb(AST.m_Sadd, daddST, daddST);
            for (; iST >= iST1; --iST)
            {
              KF.m_AxsST[iST].m_Sadd += daddST;
              KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
            }
          } else
          {
            FTR::Factor::DD::amb(AST.m_Sadd, daddST, daddST);
            for (int iST = iST1; iST < iST2; ++iST)
            {
              KF.m_AxsST[iST].m_Sadd += daddST;
              KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION;
            }
          }
        }
        m_ucsKF[iKF] |= LBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION;
#ifdef CFG_VERBOSE
        if (m_verbose >= 3)
        {
          ++g_SNdST;
        }
#endif
      }
    }
  }
//  foutC.close();
}

void LocalBundleAdjustor::UpdateFactorsPriorCameraMotion()
{
  const int iLF = m_ic2LF.front();//滑窗中目前最老的帧
  const ubyte ucm = m_ucmsLF[iLF];//这帧的motion是否要更新
  if (!ucm)
  {
    return;
  }
  const bool uc = (ucm & (LBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION |//需要更新rt
                          LBA_FLAG_CAMERA_MOTION_UPDATE_POSITION)) != 0;
  CameraPrior::Motion::Factor::RR dArr;
  CameraPrior::Motion::Factor::RM dArm;
  CameraPrior::Motion::Factor::MM dAmm;
  if (!uc)
  {
    dArr = m_ApLF.m_Arr;
  }
  if (!ucm)
  {
    dArm = m_ApLF.m_Arm;
    dAmm = m_ApLF.m_Amm;
  }
  CameraPrior::Motion::Factor::Auxiliary U;//残差就是速度,ba,bg 优化变量是r,v,ba,bg
  //当滑窗开始的时候,m_ZpLF会在滑窗中被计算当前最老帧的motion先验约束
  m_ZpLF.GetFactor(BA_WEIGHT_PRIOR_CAMERA_MOTION, m_CsLF[iLF]/*最老的帧对应的pose*/, &m_ApLF/*运动先验部分的H|-b*/, &U);
//  m_ApLF.Print();
  Camera::Factor::Unitary::CC &SAcc = m_SAcusLF[iLF];//最早的滑窗的这个普通帧自己和自己的H以及自己对应的-b*/
  Camera::Factor::Unitary &SAcm = m_SAcmsLF[iLF].m_Au;
  if (uc)//如果是一个新的先验因子
  {
    SAcc.Increase3(m_ApLF.m_Arr.m_A, m_ApLF.m_Arr.m_b);//在Hrr的部分加上对应的约束
  } else//这个先验因子已经有了,就加新的减旧的
  {
    CameraPrior::Motion::Factor::RR::AmB(m_ApLF.m_Arr, dArr, dArr);
    SAcc.Increase3(dArr.m_A, dArr.m_b);
  }
  if (ucm)//这里是肯定要更新的 就是这帧的位姿 X 运动的部分,因为没有P,所以只有r需要加
  {
    SAcm.m_Acm.Increase3(m_ApLF.m_Arm);
    SAcm.m_Amm += m_ApLF.m_Amm;
  } else
  {
    CameraPrior::Motion::Factor::RM::AmB(m_ApLF.m_Arm, dArm, dArm);
    CameraPrior::Motion::Factor::MM::AmB(m_ApLF.m_Amm, dAmm, dAmm);
    SAcm.m_Acm.Increase3(dArm);
    SAcm.m_Amm += dAmm;
  }
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
//遍历所有的imu约束(即m_DsLF预积分),其中前一帧滑窗中id:iLF1,后一帧滑窗中id:iLF2
// H部分的前帧poseX前帧pose,b部分的前帧pose存储在m_SAcusLF[iLF1] 2处
// H部分的后帧poseX后帧pose,b部分的后帧pose存储在m_SAcusLF[iLF2] 2处
// H部分的前帧poseX前帧motion,前帧motionX前帧motion,b部分的前帧motion存储在m_SAcmsLF[iLF1] 3处
// H部分的后帧poseX后帧motion,后帧motionX后帧motion,b部分的后帧motion, 7处
// 前帧poseX后帧pose,前帧poseX后帧motion,前帧motionX后帧pose，前帧motionX后帧motion都存储在m_SAcmsLF[iLF2]
// m_AdsLF[ic2]也会作为中间变量存储H中前一帧的c,m x 前一帧的c,m,以及对应的-b,H中后一帧的c,m x 后一帧的c,m,以及对应的-b
void LocalBundleAdjustor::UpdateFactorsIMU() {
#ifdef CFG_VERBOSE
  int SN = 0;
#endif
  Camera::Factor::Unitary::CC dAcc1, dAcc2;
  Camera::Factor::Unitary dAcm1, dAcm2;
  IMU::Delta::Factor::Auxiliary::Global U;
  //float dF;
  const ubyte ucFlag = LBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION |
                       LBA_FLAG_CAMERA_MOTION_UPDATE_POSITION;
  const int nLFs = static_cast<int>(m_LFs.size());
  for (int ic1 = 0, ic2 = 1; ic2 < nLFs; ic1 = ic2++)//从滑动窗口最老帧开始
  {
    const int iLF1 = m_ic2LF[ic1], iLF2 = m_ic2LF[ic2];//从最老的那一个imu约束开始,也就是它连接着最老的一帧和次老的一帧
    const ubyte ucm1 = m_ucmsLF[iLF1], ucm2 = m_ucmsLF[iLF2];//pose和motion之间的约束是否要更新的flags
    if (!ucm1 && !ucm2)
    {
      continue;
    }
     // m_A11存储着H中前一帧的c,m x 前一帧的c,m,以及对应的-b,m_A22存储着H中后一帧的c,m x 后一帧的c,m,以及对应的-b
    IMU::Delta::Factor &A = m_AdsLF[iLF2];//后一帧的索引对应当前这个imu预积分对应的imu约束

    Camera::Factor &SAcm1 = m_SAcmsLF[iLF1], &SAcm2 = m_SAcmsLF[iLF2];//后一帧的索引对应当前这个imu预积分对应的imu约束,存储了
    // 前一帧Pose x 后一帧Pose,前一帧Pose x 后一帧M,前一帧M x 后一帧Pose,前一帧M x 后一帧M

    const bool uc1 = (ucm1 & ucFlag) != 0, uc2 = (ucm2 & ucFlag) != 0;
    if (!uc1)
    {
      dAcc1 = A.m_A11.m_Acc;
    }
    if (!ucm1)
    {
      dAcm1.m_Acm = A.m_A11.m_Acm;
      dAcm1.m_Amm = A.m_A11.m_Amm;
    }
    if (!uc2)
    {
      dAcc2 = A.m_A22.m_Acc;
    }
    if (!ucm2)
    {
      dAcm2.m_Acm = A.m_A22.m_Acm;
      dAcm2.m_Amm = A.m_A22.m_Amm;
    }
    //dF = A.m_F;//m_DsLF[iLF2]就是之前从m_CsLF[iLF1]预积分到m_CsLF[iLF2]的变量
    m_DsLF[iLF2].GetFactor(BA_WEIGHT_IMU, m_CsLF[iLF1]/*前一帧状态*/, m_CsLF[iLF2]/*后一帧状态*/, m_K.m_pu/*tc0_i*/,
                           &A/*当前imu因子,m_A11存储着H中前一帧的c,m x 前一帧的c,m,以及对应的-b,m_A22存储着H中后一帧的c,m x 后一帧的c,m,以及对应的-b*/,
                           &SAcm2.m_Ab,//存储了前一帧Pose x 后一帧Pose,前一帧Pose x 后一帧M,前一帧M x 后一帧Pose,前一帧M x 后一帧M
                           &U, BA_ANGLE_EPSILON);
    //前一帧自己和自己的pose的H,-b部分关于这个因子的更新
    if (uc1)//如果前一帧这个因子没有影响过,那么直接+=
    {
      m_SAcusLF[iLF1] += A.m_A11.m_Acc;
    } else//如果前一帧这个因子已经过过约束了,但是需要更新
    {
      Camera::Factor::Unitary::CC::AmB(A.m_A11.m_Acc, dAcc1, dAcc1);
      m_SAcusLF[iLF1] += dAcc1;
    }
    //前一帧自己的pose和自己的M的H,自己的M和自己的M的H,-b部分关于这个因子的更新
    if (ucm1)
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
      m_SAcusLF[iLF2] += A.m_A22.m_Acc;
    } else
    {
      Camera::Factor::Unitary::CC::AmB(A.m_A22.m_Acc, dAcc2, dAcc2);
      m_SAcusLF[iLF2] += dAcc2;
    }
      //后一帧自己的pose和自己的M的H,自己的M和自己的M的H,-b部分关于这个因子的更新
    if (ucm2)
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
      ++SN;
    }
#endif
  }
#ifdef CFG_VERBOSE
  if (m_verbose >= 3) {
    const int N = nLFs - 1;
    UT::Print("  Delta = %d / %d = %.2f%%\n", SN, N, UT::Percentage(SN, N));
  }
#endif
}

void LocalBundleAdjustor::UpdateFactorsFixOrigin() {
  const int iLF = m_ic2LF[0];
  if (m_LFs[iLF].m_T.m_iFrm != 0 || !(m_ucsLF[iLF] & LBA_FLAG_FRAME_UPDATE_CAMERA)) {
    return;
  }
  //float dF = m_Af->m_F;
  m_Zo.GetFactor(m_CsLF[iLF].m_Cam_pose, m_Ao, BA_ANGLE_EPSILON);
  m_SAcusLF[iLF] += m_Ao.m_A;
  //dF = m_Af->m_F - dF;
  //m_F = dF + m_F;
}

void LocalBundleAdjustor::UpdateFactorsFixPositionZ()
{
#ifdef CFG_VERBOSE
  int SN = 0;
#endif
  //float dF;
  const Camera::Fix::PositionZ z(BA_WEIGHT_FIX_POSITION_Z, BA_VARIANCE_FIX_POSITION_Z);
  const ubyte ucmFlag = LBA_FLAG_CAMERA_MOTION_UPDATE_ROTATION |
                        LBA_FLAG_CAMERA_MOTION_UPDATE_POSITION;
  const int nLFs = static_cast<int>(m_LFs.size());
  for (int iLF = 0; iLF < nLFs; ++iLF)
  {
    const ubyte ucm = m_ucmsLF[iLF];
    if (!(ucm & ucmFlag)) {
      continue;
    }
    Camera::Fix::PositionZ::Factor &A = m_AfpsLF[iLF];
    Camera::Factor::Unitary::CC &SA = m_SAcusLF[iLF];
    if (ucm & LBA_FLAG_CAMERA_MOTION_UPDATE_POSITION)
    {
      //dF = A.m_F;
      z.GetFactor(m_CsLF[iLF].m_p.z()/*左相机的z*/, A);//信息矩阵是0啊,这个完全没有固定的意义啊
      //dF = A.m_F - dF;
      //m_F = dF + m_F;
#ifdef CFG_VERBOSE
      if (m_verbose >= 3)
      {
        ++SN;
      }
#endif
    }
    SA.m_A.m22() += z.m_w;
    SA.m_b.v2() += A.m_b;
  }
#ifdef CFG_VERBOSE
  if (m_verbose >= 3) {
    UT::Print("  Fix Position Z = %d / %d = %.2f%%\n", SN, nLFs, UT::Percentage(SN, nLFs));
  }
#endif
}

void LocalBundleAdjustor::UpdateFactorsFixMotion() {
#ifdef CFG_VERBOSE
  int SNv = 0, SNba = 0, SNbw = 0;
#endif
  //float dF;
  const Camera::Fix::Zero zv[2] = {
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_VELOCITY),
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_VELOCITY_INITIAL)};
  const Camera::Fix::Zero zba[2] = {
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_BIAS_ACCELERATION),
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_BIAS_ACCELERATION_INITIAL)};
  const Camera::Fix::Zero zbw[2] = {
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_BIAS_GYROSCOPE),
        Camera::Fix::Zero(BA_WEIGHT_FIX_MOTION, BA_VARIANCE_FIX_BIAS_GYROSCOPE_INITIAL)};
  const int nLFs = int(m_LFs.size());
  for (int iLF = 0; iLF < nLFs; ++iLF) {
    const ubyte ucm = m_ucmsLF[iLF];
    if (!ucm) {
      continue;
    }
    Camera::Fix::Motion::Factor &A = m_AfmsLF[iLF];//这真motion部分的先验
    Camera::Factor::Unitary::MM &SA = m_SAcmsLF[iLF].m_Au.m_Amm;//这帧H中自己的mm
    const Camera &C = m_CsLF[iLF];
    const int i = m_LFs[iLF].m_T.m_iFrm == 0 ? 1 : 0;
    if (ucm & LBA_FLAG_CAMERA_MOTION_UPDATE_VELOCITY) {
      //dF = A.m_Av.m_F;
      zv[i].GetFactor(C.m_v, A.m_Av);
      //dF = A.m_Av.m_F - dF;
      //m_F = dF + m_F;
#ifdef CFG_VERBOSE
      if (m_verbose >= 3) {
        ++SNv;
      }
#endif
    }
    if (ucm & LBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_ACCELERATION) {
      //dF = A.m_Aba.m_F;
      zba[i].GetFactor(C.m_ba, A.m_Aba);
      //dF = A.m_Aba.m_F - dF;
      //m_F = dF + m_F;
#ifdef CFG_VERBOSE
      if (m_verbose >= 3) {
        ++SNba;
      }
#endif
    }
    if (ucm & LBA_FLAG_CAMERA_MOTION_UPDATE_BIAS_GYROSCOPE) {
      zbw[i].GetFactor(C.m_bw, A.m_Abw);
      //dF = A.m_Abw.m_F - dF;
      //m_F = dF + m_F;
#ifdef CFG_VERBOSE
      if (m_verbose >= 3) {
        ++SNbw;
      }
#endif
    }
    SA.m_A.IncreaseDiagonal012(zv[i].w());   SA.m_b.Increase(0, A.m_Av.m_b);//都是0不知道有啥意义
    SA.m_A.IncreaseDiagonal345(zba[i].w());  SA.m_b.Increase(3, A.m_Aba.m_b);
    SA.m_A.IncreaseDiagonal678(zbw[i].w());  SA.m_b.Increase(6, A.m_Abw.m_b);
  }
#ifdef CFG_VERBOSE
  if (m_verbose >= 3) {
    UT::Print("  Fix Motion = (%d %d %d) / %d = (%.2f%% %.2f%% %.2f%%)\n", SNv, SNba, SNbw, nLFs,
              UT::Percentage(SNv, nLFs), UT::Percentage(SNba, nLFs), UT::Percentage(SNbw, nLFs));
  }
#endif
}


// 之前UpdateFactors已经构建好了H* dx = b的H,-b 为了加速,所以先将地图点边缘化,求出pose和motion的增量
//H|b ==》 S|g  (为了不再打符号,我就写b了)
//S* -dx_pose+motion = g
//H.row(1) [U,W] H.row(2) [W.t,V] b [u,v].t ==》 S = U - W*V^-1*W.t g = u - W*V^-1*v
//这里在求的就是W*V^-1*W.t g 和 W*V^-1*v 不过有一点要记住,g是用的-b做的,所以求出的增量是要取负的
//如论文里所说,用了子轨迹加速舒尔补的构建（一个长的共视轨迹拆成多个短的,这样S不会稠密）,假设现在有i1,i2帧看到了j这个地图点
//针对每一个子块: j_S_i1i2 =  W_i1j * j_Q_i1i2 * W_i2j.t   j_g_i1 = W_i1j * j_q_i
//  j_Q_i1i2 = ∑ST_Vjj^-1 j_q_i = ∑ST_Vjj^-1*ST_vj 就是这两帧所共有的这个点的子轨迹部分的地图点的逆相加

// 考虑一个被merge,那么他会给S中两帧的自己和自己位置(即对角线处)加影响,以及两帧之间也会有影响,即非对角线,然后这里回答一下崔华坤的ice解析里
// 里提到三帧共视呢,13的约束在哪里给,在LF.m_iLFsMatch里记录了每一帧和它之后5帧内的共视匹配,这样遍历就可以处理13的约束了

//step1 遍历滑窗内所有的帧,老帧新来的帧后面要用相关的矩阵清零(注释在数据结构里)

//step2 //遍历所有关键帧中的新地图点以及它们的子轨迹,记录下有效的地图点(iX2d中的value为有效点新编号后对应的id)
// 和子轨迹数(iXST2dST对应于有效的子轨迹新id),以及更新对应的flags(在KF.m_ms[ix]里),以确定哪些是新因子,哪些是老的但是需要更新的

//step3 上一步已经筛选了符合条件的子轨迹的H和地图点的H (H指的是逆深度x逆深度对应的H,其中子轨迹的H = 对应地图点的H/这个地图点的子轨迹数量)
//遍历所有的地图点,将有效的地图点H加进mdds[id],-b加进nds[id],iKF2X和iX2d组合使用,iKF2X找到这个关键帧中第一个地图点位置,而iX2d提供了这个地图点局部id
//遍历所有的子轨迹,将有效的子轨迹H加进mddsST[idST],-b加进ndsST[idST],iKF2XST找到这个关键帧中第一个子轨迹位置,而ixST2dST提供了这个子轨迹局部id

//step4 对mdds,mddsST取逆,之后mdds,mddsST存的都是Huu^-1和ST_Huu^-1了，同时nds,ndsST改成存Huu^-1*-bu和ST_Huu^-1*-bu
// 遍历所有的地图点,将有效的地图点的 Huu^-1|Huu^-1*-bu保存到KF.m_Mxs[ix].m_mdd

//step5 遍历这个地图点的所有子轨迹,将有效的ST_Huu^-1|ST_Huu^-1*-ST_bu保存到KF.m_MxsST[iST].m_mdd
// 如果因子存在但要更新,m_MxsTmp[idST].m_mdd保存因子的变化量(新-旧),如果因子已经存在且要边缘化,
// 但是H不合法(应该是一直没有被观测到),那么就减去这个因子,KF.m_MxsST[iST].m_mdd设成-

//step6 开始计算 S要减的对角线的W*V^-1*W.t 以及g要减的的W*V^-1*v
// 遍历这个普通帧观测到的所有的地图点(先遍历关键帧,然后遍历关键对应的观测到的地图点)
//在LF.m_Mzs1[iz]中保存W_iLF_iz * H_iz_iz^-1   iz是这个点在LF中的观测id索引
//接下来遍历这个点所有的子轨迹,计算这个子轨迹的ST_Huu^-1,ST_Huu^-1*-ST_bu的和(对应于ice-ba论文里的公式13的Q|q)
//SmdczST存储这个点的Hpose_u *∑ST_Huu^-1|∑ST_Huu^-1 * ST_bu
//LF.m_Mzs2[iz]中的m_A里存储Hpose_u *∑ST_Huu^-1 * Hpose_u.t,m_b存储Hpose_u *∑ST_Huu^-1* ST_bu,即这帧pose对应的对角线的S,g
//将更新的后的W*V^-1*W.t,W*V^-1*v加进m_SMcusLF[iLF]

//step7 开始计算 S要减的非对角线的W*V^-1*W.t LF为正在遍历的普通帧,_LF为它后面几帧内和它共视的普通帧
//遍历每一个滑窗帧,找寻与它共视的滑窗帧(存储在LF.m_iLFsMatch里),因为这两帧如果有共视点的时候,边缘化这个点就会在这两帧中对应的S处有影响
// 找到共视的这个点的所有共视子轨迹,LF.m_Zm.m_SmddsST[i]里保存所有共视子轨迹的∑ST_Huu^-1(对应于ice-ba论文里的公式13的Q)
// LF.m_Zm.m_Mczms[i],i就是共视的索引,中存储HLFp_u*∑ST_Huu^-1*H_LFp_u.t,即这帧pose和共视的pose对应的非对角线的S
// 将更新的后的SLF_LF加进LF.m_Zm.m_SMczms[I.m_ik(m_ik就是这个共视帧是第几个进来的)]（LF肯定是比_LF早的,这里只存了上三角部分）

//step8 遍历所有的关键帧里的地图点,如果这个点这次是有效观测,就将关键帧里记录关于因子已存在的flags设成1,如果==-2,就置0,子轨迹也是一样的操作
//遍历所有的滑窗帧中的地图点观测,以及共视信息都设成不更新(这里不保证对,flags太多了,我也记不住)
void LocalBundleAdjustor::UpdateSchurComplement()
{
  const int nLFs = static_cast<int>(m_LFs.size());
  for (int iLF = 0; iLF < nLFs; ++iLF)//遍历所有的滑窗内的帧
  {
    LocalFrame &LF = m_LFs[iLF];
    if (m_ucsLF[iLF] & LBA_FLAG_FRAME_UPDATE_CAMERA)//如果这帧需要更新pose(就是这个是当前帧),就把所有的预留矩阵设成0
    {
      m_SMcusLF[iLF].MakeZero();
      LF.m_Zm.m_SMczms.MakeZero();
      LF.m_SmddsST.MakeZero();
      LF.m_Zm.m_SmddsST.MakeZero();
    } else//老帧的情况
    {
      const int Nk = static_cast<int>(LF.m_iLFsMatch.size());//多少个共视帧
#ifdef CFG_DEBUG
      UT_ASSERT(LF.m_Zm.m_SMczms.Size() == Nk);
#endif
      for (int ik = 0; ik < Nk; ++ik)//遍历这帧的所有共视帧
      {
        if (!(m_ucsLF[LF.m_iLFsMatch[ik]] & LBA_FLAG_FRAME_UPDATE_CAMERA))//如果共视的这帧不需要更新就跳过
        {
          continue;
        }
        LF.m_Zm.m_SMczms[ik].MakeZero();//看数据结构的注释吧,反正就是先把相关要用的矩阵清0
        const int izm = LF.m_Zm.m_ik2zm[ik], Nzm = LF.m_Zm.m_ik2zm[ik + 1] - izm;
        LF.m_Zm.m_SmddsST.MakeZero(izm, Nzm);
      }
      const int NZ = int(LF.m_Zs.size());//遍历这帧的观测
      for (int iZ = 0; iZ < NZ; ++iZ)
      {
        const FRM::Measurement &Z = LF.m_Zs[iZ];
        if (!(m_ucsKF[Z.m_iKF] & LBA_FLAG_FRAME_UPDATE_DEPTH))//这的观测到的地图点的所属关键帧不需要更新的话就跳过
        {
          continue;
        }
        const ubyte *uds = m_uds.data() + m_iKF2d[Z.m_iKF];
        for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz)
        {
          if (uds[LF.m_zs[iz].m_ix] & LBA_FLAG_FRAME_UPDATE_DEPTH)//需要更新的话也清0
          {
            LF.m_SmddsST[iz].MakeZero();
          }
        }
      }
      const int NI = int(LF.m_Zm.m_Is.size());//遍历所有的共视点,需要用到的矩阵清0
      for (int iI = 0; iI < NI; ++iI)
      {
        const MeasurementMatchLF::Index &I = LF.m_Zm.m_Is[iI];
        if (!(m_ucsKF[I.m_iKF] & LBA_FLAG_FRAME_UPDATE_DEPTH) ||
            (m_ucsLF[LF.m_iLFsMatch[I.m_ik]] & LBA_FLAG_FRAME_UPDATE_CAMERA))
        {
          continue;
        }
        const ubyte *uds = m_uds.data() + m_iKF2d[I.m_iKF];
        const int i1 = LF.m_Zm.m_iI2zm[iI], i2 = LF.m_Zm.m_iI2zm[iI + 1];
        for (int i = i1; i < i2; ++i)
        {
          if (uds[LF.m_zs[LF.m_Zm.m_izms[i].m_iz1].m_ix] & LBA_FLAG_FRAME_UPDATE_DEPTH)
          {
            LF.m_Zm.m_SmddsST[i] = 0.0f;
          }
        }
      }
    }
  }
//遍历所有关键帧中的新地图点以及它们的子轨迹,记录下有效的地图点(iX2d中的value为有效点新编号后对应的id)和子轨迹数(iXST2dST对应于有效的子轨迹新id),
// 以及更新对应的flags(在KF.m_ms[ix]里)
  int Nd = 0, NdST = 0;//有效的深度和子轨迹计数
  const int nKFs = int(m_KFs.size());
  m_idxsTmp1.assign(nKFs + nKFs, -1);
  int *iKF2X = m_idxsTmp1.data()/*前半部分记录了这个关键帧来之前的所有地图点数量*/, *iKF2XST = m_idxsTmp1.data() + nKFs;/*后半部分记录了这个关键帧来之前的所有地图点子轨迹数量,只是为了方便索引*/
  std::vector<int> &iX2d = m_idxsTmp2/*key是地图点局部id,值是第几个有效的地图点*/, &iXST2dST = m_idxsTmp3;//key是子轨迹id,值是第几个有效的子轨迹
  iX2d.resize(0);
  iXST2dST.resize(0);
  const float eps = FLT_EPSILON;
  const float epsd = UT::Inverse(BA_VARIANCE_MAX_DEPTH, BA_WEIGHT_FEATURE, eps);
  const float epsdST = UT::Inverse(BA_VARIANCE_MAX_DEPTH_SLIDING_TRACK, BA_WEIGHT_FEATURE, eps);
  for (int iKF = 0; iKF < nKFs; ++iKF)//遍历所有关键帧
  {
    if (!(m_ucsKF[iKF] & LBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION))
    {
      continue;
    }
    const ubyte *uds = m_uds.data() + m_iKF2d[iKF];//对应于这个关键帧的首个地图点flags指针
    const int iX = static_cast<int>(iX2d.size()), iXST = static_cast<int>(iXST2dST.size());
    KeyFrame &KF = m_KFs[iKF];
    const int Nx = static_cast<int>(KF.m_xs.size());//关键帧的新地图点数量
    iKF2X[iKF] = iX;          iKF2XST[iKF] = iXST;//这一步？
    iX2d.resize(iX + Nx, -1)/*扩容成这个关键帧里新地图点数量*/; iXST2dST.resize(iXST + static_cast<int>(KF.m_STs.size()), -1);//扩容成这个关键帧里地图点的所有子轨迹数量
    int *ix2d = iX2d.data() + iX, *ixST2dST = iXST2dST.data() + iXST;
    for (int ix = 0; ix < Nx; ++ix)//遍历这个关键帧中的所有新地图点
    {
      if (uds[ix] & LBA_FLAG_TRACK_UPDATE_INFORMATION)
      {
        if (KF.m_Axs[ix].m_Sadd.m_a > epsd)//如果H部分大于一个很小的数说明是有效的H部分,比如一个地图点刚被观测到,同时还没有右目观测,那么它的这里就是0
        {
          ix2d[ix] = Nd++;//记录下是第几个有效的深度
        } else if (!(uds[ix] & LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO))//如果并不是首次观测,标记成-2
        {
          ix2d[ix] = -2;
        }
      }
      const int iST1 = KF.m_ix2ST[ix]/*这个地图点的第一条轨迹*/, iST2 = KF.m_ix2ST[ix + 1];/*这个地图点的最后一条轨迹id+1*/
      if (iST2 == iST1)
      {
        continue;
      }
      ubyte update = LBA_FLAG_MARGINALIZATION_DEFAULT, nonZero = LBA_FLAG_MARGINALIZATION_DEFAULT;
      for (int iST = iST1; iST < iST2; ++iST)//遍历这个地图点的子轨迹
      {
        if (KF.m_usST[iST] & LBA_FLAG_TRACK_UPDATE_INFORMATION)//需要更新时
        {
          if (KF.m_AxsST[iST].m_Sadd.m_a > epsdST)//如果子轨迹的H大于阈值
          {
            ixST2dST[iST] = NdST++;//记录下是第几个有效的子轨迹
            update = LBA_FLAG_MARGINALIZATION_UPDATE;//需要边缘化
            nonZero = LBA_FLAG_MARGINALIZATION_NON_ZERO;//更新flags,非0
          } else if (!(KF.m_usST[iST] & LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO))
          {
            ixST2dST[iST] = -2;//已经不是刚生成这条子轨迹了
            update = LBA_FLAG_MARGINALIZATION_UPDATE;
          }
        } else
        {
          if (!(KF.m_usST[iST] & LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO))
          {
            nonZero = LBA_FLAG_MARGINALIZATION_NON_ZERO;
          }
        }
      }
      KF.m_ms[ix] |= update;
      if (nonZero)//标记一下是否是非0的
      {
        KF.m_ms[ix] |= LBA_FLAG_MARGINALIZATION_NON_ZERO;
      } else
      {
        KF.m_ms[ix] &= ~LBA_FLAG_MARGINALIZATION_NON_ZERO;
      }
    }
  }

#ifdef CFG_VERBOSE
  int SNX = 0;
  int SNs1 = 0, SNs2 = 0, SNS = 0;
#endif

  //上一步已经筛选了符合条件的子轨迹的H和地图点的H (H指的是逆深度x逆深度对应的H,其中子轨迹的H = 对应地图点的H/这个地图点的子轨迹数量)
  //遍历所有的地图点,将有效的地图点H加进mdds[id],-b加进nds[id],iKF2X和iX2d组合使用,iKF2X找到这个关键帧中第一个地图点位置,而iX2d提供了这个地图点局部id
  //遍历所有的子轨迹,将有效的子轨迹H加进mddsST[idST],-b加进ndsST[idST],iKF2XST找到这个关键帧中第一个子轨迹位置,而ixST2dST提供了这个子轨迹局部id
  const int N = Nd + NdST/*有效的地图点+子轨迹*/, NC = SIMD_FLOAT_CEIL(N);
  m_work.Resize(NC + NC + Nd * sizeof(xp128f) / sizeof(float));//id都重新编号了
  float *mdds = m_work.Data()/*记录有效深度逆深度x逆深度的H,SIMD::Inverse会被取逆*/, *mddsST = mdds + Nd;/*记录有效子轨迹的逆深度x逆深度的H但是SIMD::Inverse会被取逆*/
  float *nds = m_work.Data() + NC/*记录有效深度逆深度的-b*/, *ndsST = nds + Nd;/*记录有效子轨迹的逆深度的-b*/
  xp128f *_mdds = (xp128f *) (m_work.Data() + NC + NC);//这里存的是地图点的H^-1
  for (int iKF = 0; iKF < nKFs; ++iKF)//遍历所有的关键帧
  {
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
    const KeyFrame &KF = m_KFs[iKF];
    const int *ix2d = iX2d.data() + iX;//有效点的id
    const int Nx = int(KF.m_xs.size());
    for (int ix = 0; ix < Nx; ++ix)//遍历这帧关键帧中地图点,将有效的H|-b保存到mdds[id],nds[id]中
    {
      const int id = ix2d[ix];
      if (id < 0)//说明这次是无效观测,H不符合要求
      {
        continue;
      }
      const FTR::Factor::DD &Sadd = KF.m_Axs[ix].m_Sadd;//H中这个点的逆深度x逆深度的H|-b
      mdds[id] = Sadd.m_a;
      nds[id] = Sadd.m_b;
    }
    const int *ixST2dST = iXST2dST.data() + iKF2XST[iKF];//这个关键帧的有效子轨迹索引部分
    const int NST = int(KF.m_STs.size());//这个关键帧中所有子轨迹数量
    for (int iST = 0; iST < NST; ++iST)//遍历所有的子轨迹,如果这个子轨迹对应的H有效,那么也加进mddsST[idST],ndsST[idST]中
    {
      const int idST = ixST2dST[iST];
      if (idST < 0)
      {
        continue;
      }
      const FTR::Factor::DD &SaddST = KF.m_AxsST[iST].m_Sadd;
      mddsST[idST] = SaddST.m_a;//H
      ndsST[idST] = SaddST.m_b;//-b
    }
  }
  //SIMD::Add(N, UT::Inverse(BA_VARIANCE_REGULARIZATION_DEPTH, BA_WEIGHT_FEATURE), mdds);
  SIMD::Inverse(N, mdds);//地图点和子轨迹的H求逆得到H^-1,mdds和mddsST里的H已经变成Huu^-1了 u代表逆深度
  SIMD::Multiply(N, mdds, nds);//nds = mdds*nds (H^-1*b)
  for (int id = 0; id < Nd; ++id)//保存H^-1*b到_mdds
  {
    _mdds[id].vdup_all_lane(mdds[id]);
  }

  m_MxsTmp.Resize(NdST);
#ifdef CFG_DEBUG
  for (int idST = 0; idST < NdST; ++idST) {
    m_MxsTmp[idST].m_mdd.Invalidate();
  }
#endif
// 遍历所有的地图点,将有效的地图点的 Huu^-1|Huu^-1*-bu保存到KF.m_Mxs[ix].m_mdd
//遍历这个地图点的所有子轨迹,将有效的ST_Huu^-1|ST_Huu^-1*-ST_bu保存到KF.m_MxsST[iST].m_mdd
// 如果因子存在但要更新,m_MxsTmp[idST].m_mdd保存因子的变化量(新-旧),如果因子已经存在且要边缘化,
// 但是H不合法(应该是一直没有被观测到),那么就减去这个因子,KF.m_MxsST[iST].m_mdd设成-
  for (int iKF = 0; iKF < nKFs; ++iKF)//遍历所有关键帧
  {
    const int iX = iKF2X[iKF];
    if (iX == -1)
    {
      continue;
    }
    const int iXST = iKF2XST[iKF];//这个关键帧的第一个子轨迹索引
    const int *ix2d = iX2d.data() + iX, *ixST2dST = iXST2dST.data() + iXST;
    const ubyte *uds = m_uds.data() + m_iKF2d[iKF];//就是找到这个关键帧第一个地图点的变量地址
    KeyFrame &KF = m_KFs[iKF];
    const int Nx = static_cast<int>(KF.m_xs.size());
    for (int ix = 0; ix < Nx; ++ix)//遍历所有的地图点,将有效的地图点的 Huu^-1|Huu^-1*-bu保存到KF.m_Mxs[ix].m_mdd
    {
      const int id = ix2d[ix];
      FTR::Factor::FixSource::Source::M &Mx = KF.m_Mxs[ix];
      if (id >= 0)
      {
        Mx.m_mdd.Set(mdds[id], nds[id]);//Huu^-1|Huu^-1*-bu
      }
#ifdef CFG_DEBUG
      else if (id == -2) {
        Mx.m_mdd.Invalidate();
      }
#endif
      if (KF.m_Nsts[ix] == 0 || !(KF.m_ms[ix] & LBA_FLAG_MARGINALIZATION_UPDATE))//如果子轨迹长度为0或者不需要边缘化更新
      {
        continue;
      }
      const ubyte ud = uds[ix] & LBA_FLAG_TRACK_UPDATE_DEPTH;//这个地图点是否要更新深度
      const int iST1 = KF.m_ix2ST[ix], iST2 = KF.m_ix2ST[ix + 1];
      for (int iST = iST1; iST < iST2; ++iST)//遍历这个地图点的所有子轨迹,将有效的ST_Huu^-1|ST_Huu^-1*-ST_bu保存到KF.m_MxsST[iST].m_mdd
      {                            //这里舒尔补的更新也是用的增量形式 如果因子存在但要更新,m_MxsTmp[idST].m_mdd保存因子的变化量(新-旧),如果因子已经存在且要边缘化,
                                             // 但是H不合法(应该是一直没有被观测到),那么就减去这个因子,KF.m_MxsST[iST].m_mdd设成-
        const int idST = ixST2dST[iST];
        if (idST == -1)
        {
          continue;
        }
        FTR::Factor::FixSource::Source::M &MxST = KF.m_MxsST[iST];
        if (idST >= 0)
        {
          FTR::Factor::DD &dmddST = m_MxsTmp[idST].m_mdd;//存储的是逆深度视觉因子的增量
          if (ud || (KF.m_usST[iST] & LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO))//如果之前还没有这个因子
          {
            MxST.m_mdd.Set(mddsST[idST], ndsST[idST]);
            dmddST = MxST.m_mdd;
          } else//如果之前这个因子已经存在了
          {
            dmddST = MxST.m_mdd;//保存一下旧的这个因子
            MxST.m_mdd.Set(mddsST[idST], ndsST[idST]);//更新这个因子
            FTR::Factor::DD::amb(MxST.m_mdd, dmddST, dmddST);//新的因子 - 旧的因子
          }
        } else//如果这条子轨迹H接近0,而且不是第一次观测到了
        {
#ifdef CFG_DEBUG
          UT_ASSERT((KF.m_usST[iST] & LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO) == 0);
#endif
          if (!ud)
          {
            MxST.m_mdd.MakeMinus();//就将这个因子减去
          }
//#ifdef CFG_DEBUG
#if 0
          MxST.m_mdd.Invalidate();
#endif
        }
      }
    }
  }

  LA::ProductVector6f SmdczST;//就是一个中间变量,存储Hpose_u *∑ST_Huu^-1|∑ST_Huu^-1 * ST_bu或者Hpose_u *∑ST_Huu^-1
  Camera::Factor::Unitary::CC dMczz;
  Camera::Factor::Binary::CC dMczm;
  for (int iLF = 0; iLF < nLFs; ++iLF)//遍历滑窗中所有帧
  {
    LocalFrame &LF = m_LFs[iLF];
    Camera::Factor::Unitary::CC &SMczz = m_SMcusLF[iLF];//维护了滑窗中每一个普通帧Hpose_u *∑ST_Huu^-1 * Hpose_u.t和Hpose_u *∑ST_Huu^-1* ST_bu
#ifdef CFG_VERBOSE
    ubyte Sczz = 0;
#endif
    const bool ucz = (m_ucsLF[iLF] & LBA_FLAG_FRAME_UPDATE_CAMERA) != 0;
    const int NZ = int(LF.m_Zs.size());//这帧观测到了多少个关键帧的地图点
    //遍历这个普通帧观测到的所有的地图点(先遍历关键帧,然后遍历关键对应的观测到的地图点)
    //在LF.m_Mzs1[iz]中保存W_iLF_iz * H_iz_iz^-1   iz是这个点在LF中的观测id索引
    //接下来遍历这个点所有的子轨迹,计算这个子轨迹的ST_Huu^-1,ST_Huu^-1*-ST_bu的和(对应于ice-ba论文里的公式13的Q|q)
    //SmdczST存储这个点的Hpose_u *∑ST_Huu^-1|∑ST_Huu^-1 * ST_bu
    //LF.m_Mzs2[iz]中的m_A里存储Hpose_u *∑ST_Huu^-1 * Hpose_u.t,m_b存储Hpose_u *∑ST_Huu^-1* ST_bu,即这帧pose对应的对角线的S,g
    //将更新的后的Hpose_u *∑ST_Huu^-1 * Hpose_u.t,Hpose_u *∑ST_Huu^-1* ST_bu加进m_SMcusLF[iLF]
    for (int iZ = 0; iZ < NZ; ++iZ)//遍历所有的观测到的关键帧
    {
      const FRM::Measurement &Z = LF.m_Zs[iZ];
      const int iKF = Z.m_iKF,/*观测到的着这个KF的id*/ iX = iKF2X[iKF];
      if (iX == -1)
      {
        continue;
      }
      const int *ix2d = iX2d.data() + iKF2X[iKF];//定位到这个kf的首个地图点
      const int *ixST2dST = iXST2dST.data() + iKF2XST[iKF];//定位到这个kf的首个子轨迹
      const ubyte *uds = m_uds.data() + m_iKF2d[iKF];
      //const bool ucx = (m_ucsKF[iKF] & LBA_FLAG_FRAME_UPDATE_CAMERA) != 0, ucr = ucx || ucz;
      const KeyFrame &KF = m_KFs[iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz)//遍历当前滑窗这帧对这个关键帧的观测
      {
        const int ix = LF.m_zs[iz].m_ix, id = ix2d[ix];
        if (id >= 0)//如果这个观测是一个有效观测
        {//LF.m_Mzs1[iz] = _mdds[id]*LF.m_Azs1[iz] = W_iLF_iz * H_iz_iz^-1
          FTR::Factor::FixSource::Marginalize(_mdds[id]/*逆深度x逆深度的H^-1*/, LF.m_Azs1[iz]/*这个地图点逆深度和当前普通帧pose的H*/, &LF.m_Mzs1[iz]);
        }
#ifdef CFG_DEBUG
        else if (id == -2) {
          LF.m_Mzs1[iz].m_adcz.Invalidate();
        }
#endif
        if (!(KF.m_ms[ix] & LBA_FLAG_MARGINALIZATION_UPDATE))//H为0的也就是初始无双目观测会在这里跳过
        {
          continue;
        }
        FTR::Factor::DD &SmddST = LF.m_SmddsST[iz];//含有这个lf中这个地图点的观测的子轨迹的ST_Huu^-1,ST_Huu^-1*-ST_bu的和
        const ubyte nonZero1 = (LF.m_ms[iz] & LBA_FLAG_MARGINALIZATION_NON_ZERO);//用来判断以前有没有过这个因子
        if (KF.m_ms[ix] & LBA_FLAG_MARGINALIZATION_NON_ZERO)//如果KF中标记了这个点的H是大于0的
        {
          const LocalFrame::SlidingTrack &ST = LF.m_STs[iz];//lf中这个地图点的观测是包含在哪几条子轨迹中的
          const int iST0 = KF.m_ix2ST[ix], iST1 = iST0 + ST.m_ist1,/*起点*/ iST2 = iST0 + ST.m_ist2;/*终点*/
          ubyte update = LBA_FLAG_MARGINALIZATION_DEFAULT;
          ubyte nonZero2 = LBA_FLAG_MARGINALIZATION_DEFAULT;
          for (int iST = iST1; iST < iST2; ++iST)//遍历含有这个lf中这个地图点的观测的子轨迹,把ST_Huu^-1,ST_Huu^-1*-ST_bu更新以后都加起来储存到LF.m_SmddsST[iz]中,
          {// 因为子轨迹的H|-b在更新因子那步除了子轨迹的数量
            const int idST = ixST2dST[iST];//这个子轨迹是否有效
#if 0
            if (m_iIter == 6 && iLF == 19 && iz == 58) {
              if (iST == iST1) {
                UT::PrintSeparator();
                UT::Print("iKF = %d, ix = %d\n", iKF, ix);
              }
              UT::Print("  iST = %d, SmddST = %e", iST, SmddST.m_a);
              if (idST >= 0) {
                UT::Print(", mdd2 - mdd1 = %e\n", m_MxsTmp[idST].m_mdd.m_a);
              } else if (idST == -2) {
                UT::Print(", -mdd1 = %e\n", KF.m_MxsST[iST].m_mdd.m_a);
              }
            }
#endif
            if (idST >= 0)
            {
              if (ucz)//说明以前没有这个因子
              {
                SmddST += KF.m_MxsST[iST].m_mdd;//逆深度和逆深度的ST_Huu^-1,逆深度的ST_Huu^-1*-ST_bu
              } else//已经有了这个因子,更新增量就可以
              {
                SmddST += m_MxsTmp[idST].m_mdd;
              }
              update = LBA_FLAG_MARGINALIZATION_UPDATE;//更新状态
              nonZero2 = LBA_FLAG_MARGINALIZATION_NON_ZERO;
            } else if (idST == -1)
            {
              if (!(KF.m_usST[iST] & LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO))//如果这个子轨迹不是第一次被观测到
              {
                if (ucz)//但是要更新深度的话,这里应该是加上
                {
                  SmddST += KF.m_MxsST[iST].m_mdd;
                }
                nonZero2 = LBA_FLAG_MARGINALIZATION_NON_ZERO;
              }
            } else//哪就是idST == -2的情况了
            {
              if (!ucz && !(uds[ix] & LBA_FLAG_TRACK_UPDATE_DEPTH))
              {
#ifdef CFG_DEBUG
                UT_ASSERT(nonZero1 != 0);
#endif
                SmddST += KF.m_MxsST[iST].m_mdd;//这个子轨迹不是一条新的轨迹,而却不用更新深度了,那么就将这个因子减去
              }
              update = LBA_FLAG_MARGINALIZATION_UPDATE;
            }
          }
          LF.m_ms[iz] |= update;//也给LF里这个地图点的观测更新状态
          if (LF.m_ms[iz] & LBA_FLAG_MARGINALIZATION_UPDATE)//如果是update,说明H>0
          {
            if (nonZero2)
            {
              if (!nonZero1)
              {
                LF.m_ms[iz] |= LBA_FLAG_MARGINALIZATION_NON_ZERO;//也要给LF中这个点的观测更新状态
              }
              if (LF.m_Nsts[iz] == 1)//只有一条轨迹
              {
                SmddST = KF.m_MxsST[iST1].m_mdd;
              }
            } else
            {
              if (nonZero1)
              {
                LF.m_ms[iz] &= ~LBA_FLAG_MARGINALIZATION_NON_ZERO;
                SmddST.MakeZero();
              }
            }
          }
        } else if (nonZero1)//如果H不合格,并且也有过这个因子,就把LF.m_ms[iz]设置成需要更新,且是0
        {
          LF.m_ms[iz] |= LBA_FLAG_MARGINALIZATION_UPDATE;
          LF.m_ms[iz] &= ~LBA_FLAG_MARGINALIZATION_NON_ZERO;
          SmddST.MakeZero();
        }
        if (!(LF.m_ms[iz] & LBA_FLAG_MARGINALIZATION_UPDATE))
        {
          continue;
        }
        FTR::Factor::FixSource::M2 &M = LF.m_Mzs2[iz];//m_A里存储Hpose_u *∑ST_Huu^-1 * Hpose_u.t,m_b存储Hpose_u *∑ST_Huu^-1* ST_bu
        if (LF.m_ms[iz] & LBA_FLAG_MARGINALIZATION_NON_ZERO)//需要边缘化的
        {
          if (!ucz && nonZero1)//说明以前已经有过这个视觉因子,加新的因子减去旧的
          {
            dMczz = M.m_Mczz;//记录一下老的边缘化的Hpose_u *∑ST_Huu^-1 * Hpose_u.t,Hpose_u *∑ST_Huu^-1* ST_bu,
            FTR::Factor::FixSource::Marginalize(SmddST, LF.m_AzsST[iz].m_adcA, &SmdczST, &M);//计算新的
            Camera::Factor::Unitary::CC::AmB(M.m_Mczz, dMczz, dMczz);//增量更新
            SMczz += dMczz;
          } else {//说明是新的观测
              // //M->m_Mczz.m_A = adcz * Smdd.m_a * adcz.t(Hpose_u *∑ST_Huu^-1 * Hpose_u.t )
              //  //  M->m_Mczz.m_b = adcz * Smdd.m_b (Hpose_u *∑ST_Huu^-1* ST_bu )
            FTR::Factor::FixSource::Marginalize(SmddST/*含有这个lf中这个地图点的观测的子轨迹的H,-b的和*/,
                    LF.m_AzsST[iz].m_adcA/*这个地图点子轨迹的逆深度和这帧pose的H*/, &SmdczST, &M);
            SMczz += M.m_Mczz;
          }
#ifdef CFG_VERBOSE
          if (m_verbose >= 3)
          {
            ++SNs1;
          }
#endif
#if 0
          if (m_iIter == 6 && iLF == 19 && iz == 58) {
            UT::PrintSeparator();
            UT::Print("SmddST = \n");
            SmddST.Print(true);
            UT::Print("adczST = \n");
            LF.m_AzsST[iz].m_adcA.Print(true);
            UT::Print("Mczz = \n");
            M.m_Mczz.Print(true);
          }
#endif
        } else//不需要边缘化并且非0的,就减去这个约束
        {
          if (!ucz && nonZero1)
          {
            M.m_Mczz.GetMinus(dMczz);//
            SMczz += dMczz;
          }
#ifdef CFG_DEBUG
          //UT_ASSERT(nonZero1 != 0);
          M.m_Mczz.Invalidate();
#endif
        }
#ifdef CFG_VERBOSE
        if (m_verbose >= 3)
        {
          Sczz = 1;
        }
#endif
//#ifdef CFG_DEBUG
#if 0
//#if 1
        if (m_iIter == 6 && iLF == 19) {
          if (iz == 0) {
            UT::PrintSeparator();
          }
          UT::Print("%d %.10e\n", iz, SMczz.m_A.m00());
        }
#endif
      }
    }
#ifdef CFG_VERBOSE
    if (m_verbose >= 3)
    {
      SNs2 += int(LF.m_zs.size());
    }
#endif
#ifdef CFG_VERBOSE
    //处理非对角线部分,遍历每一个滑窗帧,找寻与它共视的滑窗帧,因为这两帧如果有共视点的时候,边缘化这个点就会在这两帧中对应的S处有影响
      // 找到共视的这个点的所有共视子轨迹,LF.m_Zm.m_SmddsST[i]里保存所有共视子轨迹的∑ST_Huu^-1(对应于ice-ba论文里的公式13的Q)
      // LF.m_Zm.m_Mczms[i],i就是共视的索引,中存储HLFp_u*∑ST_Huu^-1*H_LFp_u.t,即这帧pose和共视的pose对应的非对角线的S
      //将更新的后的SLF_LF加进LF.m_Zm.m_SMczms[I.m_ik]（LF肯定是比_LF早的,这里只存了上三角部分）
    m_marksTmp1.assign(LF.m_iLFsMatch.size(), 0);
    ubyte *Sczms = m_marksTmp1.data();
#endif
    const int NI = int(LF.m_Zm.m_Is.size());//这个普通帧
    for (int iI = 0; iI < NI; ++iI)//遍历当前这个点的共视信息(只在小的追踪轨迹内的,即它后面5帧内),
    {
      const MeasurementMatchLF::Index &I = LF.m_Zm.m_Is[iI];//当前共视组,就是当前帧和另外滑窗里的帧观测到了哪个关键帧
      const int iXST = iKF2XST[I.m_iKF];
      if (iXST == -1)
      {
        continue;
      }
      const int *ixST2dST = iXST2dST.data() + iXST;
      const ubyte *uds = m_uds.data() + m_iKF2d[I.m_iKF];
      Camera::Factor::Binary::CC &SMczm = LF.m_Zm.m_SMczms[I.m_ik];//这帧和与他的共视的那帧,在舒尔补矩阵后对应的位置的
#ifdef CFG_VERBOSE
      ubyte &Sczm = Sczms[I.m_ik];
#endif
      const int _iLF = LF.m_iLFsMatch[I.m_ik];//当前共视的这个普通帧id
      const LocalFrame &_LF = m_LFs[_iLF];//当前共视的这个普通帧
      const bool _ucz = (m_ucsLF[_iLF] & LBA_FLAG_FRAME_UPDATE_CAMERA) != 0, uczm = ucz || _ucz;//这两个帧只有有一帧需要改变pose,motion就需要改变
      const KeyFrame &KF = m_KFs[I.m_iKF];//这两个普通帧共视的这个关键帧
      const int i1 = LF.m_Zm.m_iI2zm[iI], i2 = LF.m_Zm.m_iI2zm[iI + 1];//
      for (int i = i1; i < i2; ++i)//遍历iLF和_iLF之间的所有共视
      {
        const FTR::Measurement::Match &izm = LF.m_Zm.m_izms[i];//iLF和_iLF对某个地图点的共视
        if (!(LF.m_ms[izm.m_iz1] & LBA_FLAG_MARGINALIZATION_UPDATE))//不需要边缘化更新的话就跳过
        {
          continue;
        }
        float &SmddST = LF.m_Zm.m_SmddsST[i];//LF,_LF这两帧所有共视的的子轨迹的∑ST_Huu^-1
        const ubyte nonZero1 = (LF.m_Zm.m_ms[i] & LBA_FLAG_MARGINALIZATION_NON_ZERO);//之前有没有这个因子
        if (LF.m_ms[izm.m_iz1] & LBA_FLAG_MARGINALIZATION_NON_ZERO)//之前有没有这个因子的话
        {//只找这两帧共视的子轨迹,因为_LF肯定是在LF之后的,_LF这点的第一条子随轨迹和LF这点的最后一条子轨迹的公共区域才有可能有共视
          const int ix = LF.m_zs[izm.m_iz1].m_ix, iST0 = KF.m_ix2ST[ix];
          const int iST1 = iST0 + _LF.m_STs[izm.m_iz2].m_ist1;
          const int iST2 = iST0 + LF.m_STs[izm.m_iz1].m_ist2;
          ubyte update = LBA_FLAG_MARGINALIZATION_DEFAULT;
          ubyte nonZero2 = LBA_FLAG_MARGINALIZATION_DEFAULT;
          for (int iST = iST1; iST < iST2; ++iST)//遍历所有的共视子轨迹,把共视的所有子轨迹的深度和逆深度的ST_Huu^-1求和
          {
            const int idST = ixST2dST[iST];//这条子轨迹是否有观测
            if (idST >= 0)
            {
              if (uczm)//如果是一个新的因子
              {
                SmddST += KF.m_MxsST[iST].m_mdd.m_a;//逆深度和逆深度的ST_Huu^-1
              } else//因子已经存在,但是需要更新
              {
                SmddST += m_MxsTmp[idST].m_mdd.m_a;//舒尔补的增量
              }
              update = LBA_FLAG_MARGINALIZATION_UPDATE;
              nonZero2 = LBA_FLAG_MARGINALIZATION_NON_ZERO;
            } else if (idST == -1)
            {
              if (!(KF.m_usST[iST] & LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO))
              {
                if (uczm)
                {
                  SmddST += KF.m_MxsST[iST].m_mdd.m_a;
                }
                nonZero2 = LBA_FLAG_MARGINALIZATION_NON_ZERO;
              }
            } else
            {
              if (!uczm && !(uds[ix] & LBA_FLAG_TRACK_UPDATE_DEPTH))
              {
#ifdef CFG_DEBUG
                UT_ASSERT(nonZero1 != 0);
#endif
                SmddST += KF.m_MxsST[iST].m_mdd.m_a;
              }
              update = LBA_FLAG_MARGINALIZATION_UPDATE;
            }
          }
          LF.m_Zm.m_ms[i] |= update;//更新flags,是否要边缘化
          if (LF.m_Zm.m_ms[i] & LBA_FLAG_MARGINALIZATION_UPDATE)//如果需要更新的话
          {
            if (nonZero2)
            {
              if (!nonZero1)
              {
                LF.m_Zm.m_ms[i] |= LBA_FLAG_MARGINALIZATION_NON_ZERO;//更新这个观测对应的flags,也就是这个因子已经存在过了
              }
              if (iST2 - iST1 == 1)//说明子轨迹只有一条
              {
                SmddST = KF.m_MxsST[iST1].m_mdd.m_a;//这个重复了吧,前面+=y已经是这个效果了
              }
            } else
            {
              if (nonZero1)
              {
                LF.m_Zm.m_ms[i] &= ~LBA_FLAG_MARGINALIZATION_NON_ZERO;
                SmddST = 0.0f;
              }
            }
          }
        } else if (nonZero1)//如果H不合格,并且也有过这个因子,就把LF.m_ms[iz]设置成需要更新,且是0
        {
          LF.m_Zm.m_ms[i] |= LBA_FLAG_MARGINALIZATION_UPDATE;
          LF.m_Zm.m_ms[i] &= ~LBA_FLAG_MARGINALIZATION_NON_ZERO;
          SmddST = 0.0f;
        }
        if (!(LF.m_Zm.m_ms[i] & LBA_FLAG_MARGINALIZATION_UPDATE))//如果不需要更新就跳过
        {
          continue;
        }
        Camera::Factor::Binary::CC &Mczm = LF.m_Zm.m_Mczms[i];//
        if (LF.m_Zm.m_ms[i] & LBA_FLAG_MARGINALIZATION_NON_ZERO)//如果这个观测的H是大于0的,说明是有效观测
        {
          if (!uczm && nonZero1)//说明是老的共视观测,已经算过这个因子了,只需要更新就可以
          {
            dMczm = Mczm;
            FTR::Factor::FixSource::Marginalize(SmddST, LF.m_AzsST[izm.m_iz1],//注释同下
                                                _LF.m_AzsST[izm.m_iz2], &SmdczST, &Mczm);
            Camera::Factor::Binary::CC::AmB(Mczm, dMczm, dMczm);
            SMczm += dMczm;
          } else
          {//算的就是舒尔补非对角部分的元素
            FTR::Factor::FixSource::Marginalize(SmddST/*LF,_LF这两帧所有共视的的子轨迹的∑ST_Huu^-1*/,
                    LF.m_AzsST[izm.m_iz1],/*H中当前遍历的这帧LFposex逆深度*/
                    _LF.m_AzsST[izm.m_iz2]/*H中与LF共识的_LF的posex逆深度*/, &SmdczST/*H_LFp_u *∑ST_Huu^-1*/,
                    &Mczm/*HLFp_u*∑ST_Huu^-1*H_LFp_u.t*/);
            SMczm += Mczm;
          }
#ifdef CFG_VERBOSE
          if (m_verbose >= 3)
          {
            ++SNs1;
          }
#endif
        } else
        {
          if (!uczm && nonZero1)//如果是不用更新的,减去
          {
            Mczm.GetMinus(dMczm);
            SMczm += dMczm;
          }
#ifdef CFG_DEBUG
          //UT_ASSERT(nonZero1 != 0);
          Mczm.Invalidate();
#endif
        }
#ifdef CFG_VERBOSE
        if (m_verbose >= 3)
        {
          Sczm = 1;
        }
#endif
//#ifdef CFG_DEBUG
#if 0
        if (m_iIter == 1 && iLF == 27 && _iLF == 28) {
          UT::Print("%d %.10e\n", i, SMczm[0][0]);
        }
#endif
      }
    }
#ifdef CFG_VERBOSE
    if (m_verbose >= 3)
    {
      SNs2 += int(LF.m_Zm.m_izms.size());
      const int Nk = int(LF.m_iLFsMatch.size());
      for (int ik = 0; ik < Nk; ++ik)
      {
        if (Sczms[ik])
        {
          ++SNS;
        }
      }
    }
#endif
  }

#ifdef CFG_DEBUG
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    if (iKF2X[iKF] == -1) {
      continue;
    }
    KeyFrame &KF = m_KFs[iKF];
    const int *ixST2dST = iXST2dST.data() + iKF2XST[iKF];
    const int NST = int(KF.m_STs.size());
    for (int iST = 0; iST < NST; ++iST) {
      if (ixST2dST[iST] == -2) {
        KF.m_MxsST[iST].m_mdd.Invalidate();
      }
    }
  }
#endif
  //遍历所有的关键帧里的地图点,如果这个点这次是有效观测,就将关键帧里记录关于因子已存在的flags设成1,如果==-2,就置0,子轨迹也是一样的操作
    //遍历所有的滑窗帧中的地图点观测,以及共视信息都设成不更新
  for (int iKF = 0; iKF < nKFs; ++iKF)//遍历所有关键帧
  {
    const int iX = iKF2X[iKF];
    if (iX == -1)
    {
      continue;
    }
    ubyte *uds = m_uds.data() + m_iKF2d[iKF];
    const int *ix2d = iX2d.data() + iX;
    KeyFrame &KF = m_KFs[iKF];
    const int Nx = static_cast<int>(KF.m_xs.size());
    for (int ix = 0; ix < Nx; ++ix)//遍历这个关键帧所产生的特征点
    {
      if (ix2d[ix] >= 0)//说明是有效点,将LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO这个flag置0
      {
        uds[ix] &= ~LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;
      } else if (ix2d[ix] == -2)//如果是-2就赋值LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO
      {
        uds[ix] |= LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;
      }
      KF.m_ms[ix] &= ~LBA_FLAG_MARGINALIZATION_UPDATE;//LBA_FLAG_MARGINALIZATION_UPDATE置0
    }
    const int *ixST2dST = iXST2dST.data() + iKF2XST[iKF];
    const int NST = static_cast<int>(KF.m_STs.size());
    for (int iST = 0; iST < NST; ++iST)//遍历这帧所有子轨迹,更新flags
    {
      if (ixST2dST[iST] >= 0)
      {
        KF.m_usST[iST] &= ~LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;
      } else if (ixST2dST[iST] == -2)
      {
        KF.m_usST[iST] |= LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO;
      }
    }
  }
  for (int iLF = 0; iLF < nLFs; ++iLF)
  {
    LocalFrame &LF = m_LFs[iLF];
    const int NZ = int(LF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ)
    {
      const FRM::Measurement &Z = LF.m_Zs[iZ];
      if (iKF2X[Z.m_iKF] == -1)
      {
        continue;
      }
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz)
      {
        LF.m_ms[iz] &= ~LBA_FLAG_MARGINALIZATION_UPDATE;
      }
    }
    const int Nzm = int(LF.m_Zm.m_ms.size());
    for (int i = 0; i < Nzm; ++i)
    {
      LF.m_Zm.m_ms[i] &= ~LBA_FLAG_MARGINALIZATION_UPDATE;
    }
  }
#ifdef CFG_VERBOSE
  if (m_verbose >= 3)
  {
    const int _Nd = int(m_ds.size()), NST = CountSlidingTracks();
    const int NSLLF = CountSchurComplements();
    UT::PrintSeparator();
    UT::Print("*%2d: [LocalBundleAdjustor::UpdateSchurComplement]\n", m_iIter);
    UT::Print("  Track    = %5d / %5d = %.2f%% (%d / %d = %.2f%%)\n", Nd, _Nd,
              UT::Percentage(Nd, _Nd), SNX, nKFs, UT::Percentage(SNX, nKFs));
    UT::Print("  TrackST  = %5d / %5d = %.2f%%\n", NdST, NST, UT::Percentage(NdST, NST));
    UT::Print("  Schur    = %5d / %5d = %.2f%% (%d / %d = %.2f%%)\n", SNs1, SNs2,
              UT::Percentage(SNs1, SNs2), SNS, NSLLF, UT::Percentage(SNS, NSLLF));
  }
#endif
}

#ifdef CFG_INCREMENTAL_PCG
//#define CFG_INCREMENTAL_PCG_1
#endif

bool LocalBundleAdjustor::SolveSchurComplement()
{
//#ifdef CFG_INCREMENTAL_PCG
#if 0
#ifdef CFG_INCREMENTAL_PCG_1
  m_xcsLF.MakeZero();
  m_xmsLF.MakeZero();
#endif
#endif
  if (LBA_PROPAGATE_CAMERA >= 2 && m_iIter == 0 && SolveSchurComplementLast()) {
    return true;
  }
  bool scc = SolveSchurComplementPCG();//PCG加速求解S*x=-g 其中预优矩阵M取S对角线元素
#ifdef LBA_DEBUG_GROUND_TRUTH_STATE
  SolveSchurComplementGT(m_CsLF, &m_xsGN);
#endif
  if (LBA_EMBEDDED_MOTION_ITERATION) {
#ifdef CFG_DEBUG
//#if 0
    const LA::Vector6f *xcs = (LA::Vector6f *) m_xsGN.Data();
    const LA::Vector9f *xms = (LA::Vector9f *) (xcs + m_LFs.size());
    ConvertCameraUpdates(xcs, &m_xcsP);
    const IMU::Delta::ES ESd1 = ComputeErrorStatisticIMU(m_xcsP.Data(), xms, false);
    CameraPrior::Motion::ES ESm1 = ComputeErrorStatisticPriorCameraMotion(m_xcsP.Data(), xms);
    const float F1 = ESd1.Total() + ESm1.Total();
#endif
    EmbeddedMotionIteration();
#ifdef CFG_DEBUG
//#if 0
    //if (m_iIter == 2) {
    //  PrintSchurComplementResidual();
    //}
    const IMU::Delta::ES ESd2 = ComputeErrorStatisticIMU(m_xcsP.Data(), xms, false);
    CameraPrior::Motion::ES ESm2 = ComputeErrorStatisticPriorCameraMotion(m_xcsP.Data(), xms);
    const float F2 = ESd2.Total() + ESm2.Total();
    //UT_ASSERT(F2 <= F1);
    UT::AssertReduction(F1, F2, 1, UT::String("[%d] %d", m_LFs[m_ic2LF.back()].m_Cam_pose.m_iFrm));
    //UT::Print("%e --> %e\n", F1, F2);
#endif
  }
//#ifdef CFG_INCREMENTAL_PCG
#if 0
  FILE *fp;
  std::string fileName;
  const int iFrm = m_LFs[m_ic2LF.back()].m_Cam_pose.m_iFrm;
#ifdef CFG_INCREMENTAL_PCG_1
  fileName = "D:/tmp/pcg/count_lba.txt";
#else
  fileName = "D:/tmp/pcg/count_lba_incr.txt";
#endif
  static bool g_first = true;
  fp = fopen(fileName.c_str(), g_first ? "w" : "a");
  g_first = false;
  fprintf(fp, "%d %d %d\n", iFrm, m_iIter, m_iIterPCG);
  fclose(fp);
  fileName = UT::String("D:/tmp/pcg/state_lba_%04d_%02d.txt", iFrm, m_iIter);
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
//Hxcm = -b ==》 舒尔补以后就变成了S*δxc = -g ==》 A*-xc = b 用PCG迭代法求解
bool LocalBundleAdjustor::SolveSchurComplementPCG()
{
  Camera::Factor::Unitary::CC Acc;
  const int pc = 6/*pose部分的size*/, pm/*motion部分的size*/ = 9;
  const int pcm = pc + pm;
  const int Nc = int(m_LFs.size())/*所有滑窗帧的size*/, Ncp = Nc * pc/*相机部分的长度*/, Nmp = Nc * pm/*motion部分的长度*/, N = Ncp + Nmp;
  m_Acus.Resize(Nc);
  m_Amus.Resize(Nc);
  m_bs.Resize(N);
  float *bc = m_bs.Data()/*舒尔补以后的-g*/, *bm = bc + Ncp;
  const int Nb = CountSchurComplementsOffDiagonal();//滑窗中所有共视帧的数量
  m_Acbs.Resize(Nb);
  m_ic2b.resize(Nc);
  const float ar = UT::Inverse(BA_VARIANCE_REGULARIZATION_ROTATION, BA_WEIGHT_FEATURE);
  const float ap = UT::Inverse(BA_VARIANCE_REGULARIZATION_POSITION, BA_WEIGHT_FEATURE);
  const float av = UT::Inverse(BA_VARIANCE_REGULARIZATION_VELOCITY, BA_WEIGHT_FEATURE);
  const float aba = UT::Inverse(BA_VARIANCE_REGULARIZATION_BIAS_ACCELERATION, BA_WEIGHT_FEATURE);
  const float abw = UT::Inverse(BA_VARIANCE_REGULARIZATION_BIAS_GYROSCOPE, BA_WEIGHT_FEATURE);
    //按帧的先后顺序进行遍历,构建S和-g
    //S的对角线的pose x pose部分 存储在m_Acus[ic]里,S的对角线的motionxmotion存在m_Amus[ic]里
    //S的非对角线的pose x 别的共视帧的pose 存在Acbs里,利用m_ic2b进行索引
    // -g存在m_ic2b里
  for (int ic = 0, ib = 0; ic < Nc; ++ic, bc += pc, bm += pm)
  {
    m_ic2b[ic] = ib;
    const int iLF = m_ic2LF[ic];//当前这帧的id
    const LocalFrame &LF = m_LFs[iLF];//当前遍历的这帧
    //计算舒尔补S中pose对应的对角线部分 Hpipi,-bpi - Hpose_u *∑ST_Huu^-1 * Hpose_u.t
    // 以及pose对应的-g : -bpi - Hpose_u *∑ST_Huu^-1* -ST_bu 都存在Acc里
    Camera::Factor::Unitary::CC::AmB(m_SAcusLF[iLF]/*Hpipi,-bpi*/,
            m_SMcusLF[iLF]/*普通帧Hpose_u *∑ST_Huu^-1 * Hpose_u.t和Hpose_u *∑ST_Huu^-1* -ST_bu*/, Acc);
    Acc.m_A.IncreaseDiagonal(ap, ar);//为了保证正定吧
    Acc.m_A.GetAlignedMatrix6x6f(m_Acus[ic]);//m_Acus[ic]保存舒尔补以后对角线部分
    Acc.m_b.Get(bc);//将舒尔补以后的-g存到m_bs里
    const Camera::Factor::Unitary::MM &Amm = m_SAcmsLF[iLF].m_Au.m_Amm;//H的mm部分,-b的m部分。舒尔补对M的地方没有影响
    m_Amus[ic].Set(Amm.m_A);//m_Amus保存S中motion对应的对角线部分
    m_Amus[ic].IncreaseDiagonal(av, aba, abw);//保证正定
    Amm.m_b.Get(bm);//m_bs后半部分保存motion的-g
    const int _ic = ic + 1;//后一帧
    if (_ic == Nc)
    {
      continue;
    }
    LA::AlignedMatrix6x6f *Acbs = m_Acbs.Data() + ib;//存储非对角线舒尔补以后的矩阵
    const int _iLF = m_ic2LF[_ic];
    //滑窗内_iLF这帧和它前一帧之间的约束
    const Camera::Factor::Binary &SAcmb = m_SAcmsLF[_iLF].m_Ab;
    //Acbs[0]先存一下S中前后帧对应位置(因为前后帧的motion约束会让H对应位置有约束) = H前一帧的pose x 后一帧的pose - H前p_u*∑ST_Huu^-1*H后p_u.t
    LA::AlignedMatrix6x6f::AmB(SAcmb.m_Acc, LF.m_Zm.m_SMczms[0]/*0就是它最近一帧*/, Acbs[0]);
    const int Nk = static_cast<int>(LF.m_iLFsMatch.size());//LF的共视帧
#ifdef CFG_DEBUG
    UT_ASSERT(Nk >= 1 && Nk == std::min(Nc - ic, LBA_MAX_SLIDING_TRACK_LENGTH) - 1);
#endif
    for (int ik = 1; ik < Nk; ++ik)//0 - H前p_u*∑ST_Huu^-1*H后p_u.t
    {
      LF.m_Zm.m_SMczms[ik].GetMinus(Acbs[ik]);
    }
    ib += Nk;//加上共视帧的
//#ifdef CFG_DEBUG
#if 0
    if (ic == 12) {
      UT::Print("%.10e %.10e %.10e\n", m_SAcusLF[iLF].m_b.v4(), m_SMcusLF[iLF].m_b.v4(), Acc.m_b.v4());
    }
#endif
  }
  m_AcbTs.Resize(Nb);
  for (int ib = 0; ib < Nb; ++ib)
  {
    m_Acbs[ib].GetTranspose(m_AcbTs[ib]);//下三角区域
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
  for (int ic = 0, icp = 0, imp = Ncp, ib = 0; ic < Nc; ++ic, icp += pc, imp += pm) {
    e_A.block<pc, pc>(icp, icp) = EigenMatrix6x6f(m_Acus[ic]).cast<double>();
    const int iLF = m_ic2LF[ic];
    const LocalFrame &LF = m_LFs[iLF];
    const LA::AlignedMatrix6x6f *Acbs = m_Acbs.Data() + ib;
    const int Nk = static_cast<int>(LF.m_iLFsMatch.size());
    for (int ik = 0, _icp = icp + pc; ik < Nk; ++ik, _icp += pc) {
      e_A.block<pc, pc>(icp, _icp) = EigenMatrix6x6f(Acbs[ik]).cast<double>();
    }
    ib += Nk;
    
    const Camera::EigenFactor e_Acm = m_SAcmsLF[iLF];
    e_A.block<pm, pm>(imp, imp) = EigenMatrix9x9f(m_Amus[ic]).cast<double>();
    e_A.block<pc, pm>(icp, imp) = e_Acm.m_Au.m_Acm.cast<double>();
    if (ic > 0) {
      const int _icp = icp - pc, _imp = imp - pm;
      e_A.block<pc, pm>(_icp, imp) = e_Acm.m_Ab.m_Acm.cast<double>();
      e_A.block<pm, pc>(_imp, icp) = e_Acm.m_Ab.m_Amc.cast<double>();
      e_A.block<pc, pm>(icp, _imp) = e_Acm.m_Ab.m_Amc.cast<double>().transpose();
      e_A.block<pm, pm>(_imp, imp) = e_Acm.m_Ab.m_Amm.cast<double>();
    }
  }
  //e_A = EigenMatrixXd(e_A.block(0, 0, Ncp, Ncp));
  e_A.SetLowerFromUpper();
  //UT::PrintSeparator();
  //e_A.Print(true);
  //UT::PrintSeparator();
  //e_b.Print(true);
  EigenVectorXd e_s;
  const int e_rankLU = EigenRankLU(e_A), e_rankQR = EigenRankQR(e_A);
  UT::Print("rank = %d (%d) / %d\n", e_rankLU, e_rankQR, N);
  const double e_cond = EigenConditionNumber(e_A, &e_s);
  UT::Print("cond = %e\n", e_cond);

  const EigenVectorXd e_b = EigenVectorXd(m_bs);
  const EigenVectorXd e_x = EigenVectorXd(e_A.ldlt().solve(e_b));
  m_xsGN = e_x.GetAlignedVectorXf();
  m_xsGN.MakeMinus();
  //const EigenVectorXd e_r = e_A * e_x - e_b;
  //const Residual R = ComputeResidual(m_xsGN, true);
  ConvertCameraUpdates(m_xsGN.Data(), &m_xp2s, &m_xr2s);
  ConvertMotionUpdates(m_xsGN.Data() + Ncp, &m_xv2s, &m_xba2s, &m_xbw2s);
  UT::Print("%f %f %f %f %f\n", sqrtf(m_xp2s.Mean()), sqrtf(m_xr2s.Mean()) * UT_FACTOR_RAD_TO_DEG,
            sqrtf(m_xv2s.Mean()), sqrtf(m_xba2s.Mean()), sqrtf(m_xbw2s.Mean()) * UT_FACTOR_RAD_TO_DEG);
  return true;
#endif
//#ifdef CFG_DEBUG
#if 0
  const int ic = 13;
  const int i = ic * pc + 3;
  //const int i = Ncp + ic * pm + 6;
  const LA::Vector3f *x = (LA::Vector3f *) (m_xs.Data() + i);
  const LA::Vector3f *r = (LA::Vector3f *) (m_rs.Data() + i);
  const LA::Vector3f *p = (LA::Vector3f *) (m_ps.Data() + i);
  const LA::Vector3f *z = (LA::Vector3f *) (m_zs.Data() + i);
#endif
  bool scc = true;
  float Se2, Se2Pre, Se2Min/*最小残差*/, e2Max, e2MaxMin, alpha, beta;
  m_rs = m_bs;//rs这里先存一下b的值
#ifdef CFG_INCREMENTAL_PCG
  LA::Vector6f *xcs = (LA::Vector6f *) m_xs.Data();
  LA::Vector9f *xms = (LA::Vector9f *) (xcs + Nc);
  for (int ic = 0; ic < Nc; ++ic)
  {
    const int iLF = m_ic2LF[ic];
    xcs[ic] = m_xcsLF[iLF];//上一次优化得到pose优化增量(如果成功更新的话,这里应该是0)
    xms[ic] = m_xmsLF[iLF];//上一次优化得到motion优化增量(如果成功更新的话,这里应该是0)
  }
  m_xs.MakeMinus();//这里求的是A*-xcm = b 而m_xcsLF,m_xmsLF储存的是x的增量,所以这里要负,下面PCG的部分我就把-xcm称为x
  m_xsGN = m_xs;//之前的优化结果作为初值x0
  //////////////////////////////////////////////////////////////////////////
  //SolveSchurComplementGT(true);
  //m_xsGN.MakeMinus();
  //m_xs = m_xsGN;
  //////////////////////////////////////////////////////////////////////////
  ApplyA(m_xs/*x*/, &m_drs);//m_drs  = A*x0
  m_rs -= m_drs;// r0 = b - A*X0
#else
  m_xsGN.Resize(N);
  m_xsGN.MakeZero();
#endif
  ApplyM(m_rs/*r0*/, &m_ps/*p0*/);//z0 = M.inv * r0,初始的梯度方向z0同时第1步时梯度方向就是下降方向
  ConvertCameraMotionResiduals(m_rs/*残差*/, m_ps/*p0(z0)*/, &Se2, &e2Max);//计算Se2 = r0.t * z0
#ifdef CFG_DEBUG
  UT_ASSERT(Se2 >= 0.0f && e2Max >= 0.0f);
#endif
//#ifdef CFG_DEBUG
#if 0
  UT::DebugStart();
#endif
  ApplyA(m_ps, &m_drs);//m_drs = A*p0
//#ifdef CFG_DEBUG
#if 0
  UT::DebugStop();
#endif
  alpha = Se2 / m_ps.Dot(m_drs);//r0z0 / p0.dot(A*p0) 这个是第1步的步长
//#ifdef CFG_DEBUG
#if 0
  const std::string dir = m_dir + "pcg/";
  m_ps.AssertEqual(UT::String("%sp.txt", dir.c_str()), 2, "", -1.0f, -1.0f);
  m_drs.AssertEqual(UT::String("%sAp.txt", dir.c_str()), 2, "", -1.0f, -1.0f);
#endif
#ifdef _MSC_VER
  if (_finite(alpha)) {
#else
  if (std::isfinite(alpha))
  {
#endif  // _MSC_VER
    const float e2MaxConv[2] = {ME::ChiSquareDistance<3>(BA_PCG_MIN_CONVERGE_PROBABILITY,
                                                         BA_WEIGHT_FEATURE),
                                ME::ChiSquareDistance<3>(BA_PCG_MAX_CONVERGE_PROBABILITY,
                                                         BA_WEIGHT_FEATURE)};
    Se2Min = Se2;
    e2MaxMin = e2Max;
#ifdef CFG_VERBOSE
    if (m_verbose >= 3)
    {
      UT::PrintSeparator();
      UT::Print("*%2d: [LocalBundleAdjustor::SolveSchurComplement]\n", m_iIter);
      UT::Print("  *%2d: |r| = (%e %e) >= (%e %e)*\n", 0, Se2, e2Max, Se2Min, e2MaxMin);
    }
#endif
    m_drs *= alpha;//alpha0 * A*pk-1
    m_rs -= m_drs;//r1 = r0 - alpha0 * A*p0 //更新第二步的残差
#ifdef CFG_INCREMENTAL_PCG
    m_ps.GetScaled(alpha, m_dxs);//m_dxs = p0 * alpha0
    m_xs += m_dxs;//x1 = x0 + p0 * alpha0  //更新第二步的x
#else
    m_ps.GetScaled(alpha, m_xs);
#endif
#ifdef CFG_INCREMENTAL_PCG
    ApplyM(m_bs, &m_drs);
    Se2Pre = m_bs.Dot(m_drs);
    const float Se2ConvMin = Se2Pre * BA_PCG_MIN_CONVERGE_RESIDUAL_RATIO;
    const float Se2ConvMax = Se2Pre * BA_PCG_MAX_CONVERGE_RESIDUAL_RATIO;
#else
    const float Se2ConvMin = Se2 * BA_PCG_MIN_CONVERGE_RESIDUAL_RATIO;
    const float Se2ConvMax = Se2 * BA_PCG_MAX_CONVERGE_RESIDUAL_RATIO;
#endif
    int cnt = 0;
    const int nIters = std::min(N, BA_PCG_MAX_ITERATIONS);//最大迭代次数
    for (m_iIterPCG = 0; m_iIterPCG < nIters; ++m_iIterPCG)
    {
      ApplyM(m_rs, &m_zs);//zk = M^-1*rk 更新梯度方向
//#ifdef CFG_DEBUG
#if 0
      if (m_iIterPCG == 13)
        UT::Print("%.10e %.10e\n", r->x(), z->x());
#endif
      Se2Pre = Se2;
//#ifdef CFG_DEBUG
#if 0
      if (m_iIterPCG == 12)
        UT::DebugStart();
#endif
//#ifdef CFG_DEBUG
#if 0
      float e2p, e2r;
      if (m_iIter == 4 && m_iIterPCG == 4) {
        const int ic = 11;
        LA::AlignedVector3f rp, rr, Mrp, Mrr;
        const LA::Vector6f *rcs = (LA::Vector6f *) m_rs.Data();
        rcs[ic].Get(rp, rr);
        const Camera::Conditioner::Cam_state &Mc = m_Mcs[ic];
        LA::AlignedMatrix3x3f::Ab(Mc.m_Mp, rp, Mrp);
        LA::AlignedMatrix3x3f::Ab(Mc.m_Mr, rr, Mrr);
        e2p = Mrp.Dot(rp);
        e2r = Mrr.Dot(rr);
      }
#endif
      ConvertCameraMotionResiduals(m_rs, m_zs, &Se2, &e2Max);//计算Se2=e2Max = rk.t * zk
//#ifdef CFG_DEBUG
#if 0
      //UT::Print("%d %.10e\n", m_iIterPCG, Se2);
      if (UT::Debugging()) {
        UT::DebugStop();
        //UT::PrintSeparator();
        //m_rs.Print(true);
        //UT::PrintSeparator();
        //m_zs.Print(true);
      }
#endif
#ifdef CFG_DEBUG
      UT_ASSERT(Se2 >= 0.0f && e2Max >= 0.0f);
#endif
      if (Se2 < Se2Min)//如果小于最小残差^2,就更新一下
      {
        Se2Min = Se2;
        e2MaxMin = e2Max;
        m_xsGN = m_xs;//更新x
        //cnt = 0;
      } else
      {
        //////////////////////////////////////////////////////////////////////////
        //Se2Min = Se2;
        //e2MaxMin = e2Max;
        //m_xsGN = m_xs;
        //////////////////////////////////////////////////////////////////////////
        ++cnt;
      }
#ifdef CFG_VERBOSE
      if (m_verbose >= 3)
      {
        UT::Print("  *%2d: |r| = (%e %e) >= (%e %e)", m_iIterPCG + 1, Se2, e2Max, Se2Min, e2MaxMin);
        if (Se2 == Se2Min)
        {
          UT::Print("*");
        }
        UT::Print("\n");
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
      if (m_iIterPCG == 4) {
        UT::DebugStart();
      }
      ApplyA(m_xs, &m_drs);
      m_drs -= m_bs;
      m_drs.MakeMinus();
      m_drs -= m_rs;
#endif
      const int i = (Se2Min <= Se2ConvMin && m_iIterPCG >= BA_PCG_MIN_ITERATIONS) ? 0 : 1;
      if (Se2 == 0.0f || e2MaxMin < e2MaxConv[i])
      {
        scc = true;
        break;
      } else if (Se2Min > Se2ConvMax)
      {
        scc = false;
        break;
      }
//#if 0
#if 1
      beta = Se2 / Se2Pre;//用来算下一次下降方向的比例系数
#else
      beta = -m_zs.Dot(m_drs) / Se2Pre;
#endif
      m_ps *= beta;// pk-1*beta
      m_ps += m_zs;// pk = zk + pk-1*beta //更新当前步骤下降方向
//#ifdef CFG_DEBUG
#if 0
      const std::string dir = m_dir + "pcg/";
      UT::Print("%d %.10e\n", m_iIterPCG, beta);
      m_rs.AssertEqual(UT::String("%sr%02d.txt", dir.c_str(), m_iIterPCG), 2, "", -1.0f, -1.0f);
      m_zs.AssertEqual(UT::String("%sz%02d.txt", dir.c_str(), m_iIterPCG), 2, "", -1.0f, -1.0f);
      m_ps.AssertEqual(UT::String("%sp%02d.txt", dir.c_str(), m_iIterPCG), 2, "", -1.0f, -1.0f);
#endif
//#ifdef CFG_DEBUG
#if 0
      if (m_iIterPCG == 13) {
        UT::Print("%.10e %.10e %.10e %.10e %.10e\n", Se2, Se2Pre, beta, z->x(), p->x());
      }
#endif
//#ifdef CFG_DEBUG
#if 0
      if (m_iIterPCG == 11) {
        UT::DebugStart();
      }
#endif
      ApplyA(m_ps, &m_drs);//m_drs = A*pk
//#ifdef CFG_DEBUG
#if 0
      const std::string dir = m_dir + "pcg/";
      m_ps.AssertEqual(UT::String("%sp%02d.txt", dir.c_str(), m_iIterPCG), 2, "", -1.0f, -1.0f);
      m_drs.AssertEqual(UT::String("%sAp%02d.txt", dir.c_str(), m_iIterPCG), 2, "", -1.0f, -1.0f);
#endif
//#ifdef CFG_DEBUG
#if 0
      if (UT::Debugging()) {
        UT::DebugStop();
      }
#endif
      alpha = Se2 / m_ps.Dot(m_drs);//alphak = (rk.t * zk)/(pk.t * A*pk) 更新当前步骤
#ifdef _MSC_VER
      if (!_finite(alpha)) {
#else
      if (!std::isfinite(alpha))//步长出了问题也退出
      {
#endif  // _MSC_VER
        scc = false;
        break;
      }
      m_drs *= alpha;//m_drs = alphak * A*pk
      m_rs -= m_drs;//rk+1 = rk - alphak * A*pk //更新下一步的残差
      m_ps.GetScaled(alpha, m_dxs);//m_dxs = pk * alphak
      m_xs += m_dxs;//xk+1 = xk + pk * alphak  //更新下一步的x
#if 0
//#if 1
      ApplyA(m_xs, &m_drs);
      m_rs = m_bs;
      m_rs -= m_drs;
#endif
//#ifdef CFG_DEBUG
#if 0
      UT::Print("%d %.10e %.10e %.10e %.10e\n", m_iIterPCG, x->z(), p->z(), r->z(), z->z());
#endif
//#ifdef CFG_DEBUG
#if 0
      const std::string dir = m_dir + "pcg/";
      if (m_iIterPCG == 0) {
        m_bs.AssertEqual(UT::String("%sb.txt", dir.c_str()), 2, "", -1.0f, -1.0f);
      }
      m_xs.AssertEqual(UT::String("%sx%02d.txt", dir.c_str(), m_iIterPCG), 2, "", -1.0f, -1.0f);
      m_rs.AssertEqual(UT::String("%sr%02d.txt", dir.c_str(), m_iIterPCG), 2, "", -1.0f, -1.0f);
      m_ps.AssertEqual(UT::String("%sp%02d.txt", dir.c_str(), m_iIterPCG), 2, "", -1.0f, -1.0f);
      m_zs.AssertEqual(UT::String("%sz%02d.txt", dir.c_str(), m_iIterPCG), 2, "", -1.0f, -1.0f);
#endif
    }
#if 0
    static bool g_first = true;
    FILE *fp = fopen("D:/tmp/pcg/scc_lba.txt", g_first ? "w" : "a");
    g_first = false;
    fprintf(fp, "%d %d %d %d %f\n", m_LFs[m_ic2LF.back()].m_Cam_pose.m_iFrm, m_iIter, cnt, m_iIterPCG, UT::Percentage(cnt, m_iIterPCG));
    fclose(fp);
#endif
  } else
  {
    m_iIterPCG = 0;
  }
  m_xsGN.MakeMinus();//这里求的是A*-xcm = b 所以求负才是xcm即实际增量
//#ifdef CFG_DEBUG
#if 0
//#if 1
  PrintSchurComplementResidual();
#endif
  ConvertCameraUpdates(m_xsGN.Data(), &m_xp2s, &m_xr2s);//位置和朝向部分优化变量增量的^2
  ConvertMotionUpdates(m_xsGN.Data() + Ncp, &m_xv2s, &m_xba2s, &m_xbw2s);//同上
  return scc;
}

#ifdef CFG_GROUND_TRUTH
void LocalBundleAdjustor::SolveSchurComplementGT(const AlignedVector<Camera> &CsLF,
                                                 LA::AlignedVectorXf *xs, const bool motion) {
  if (!m_CsGT) {
    return;
  }
#ifdef LBA_DEBUG_GROUND_TRUTH_STATE_ERROR
  const float dpMax = 0.01f;
  //const float drMax = 0.0f;
  const float drMax = 0.1f;
  const float dvMax = 0.1f;
  const float dbaMax = 0.1f;
  //const float dbwMax = 0.0f;
  const float dbwMax = 0.1f;
#endif
  const int pc = 6, pm = 9;
  const int Nc = static_cast<int>(m_LFs.size());
  xs->Resize(Nc * (pc + pm));

  Rotation3D dR;
  LA::AlignedVector3f dr, dp, dv, dba, dbw;
  LA::Vector6f *xcs = (LA::Vector6f *) xs->Data();
  LA::Vector9f *xms = (LA::Vector9f *) (xcs + Nc);
  for (int ic = 0; ic < Nc; ++ic) {
    const int iLF = m_ic2LF[ic];
    const Camera &C = CsLF[iLF], &CGT = m_CsLFGT[iLF];
    Rotation3D::ATB(C.m_Cam_pose, CGT.m_Cam_pose, dR);
    dR.GetRodrigues(dr, BA_ANGLE_EPSILON);
    LA::AlignedVector3f::amb(CGT.m_p, C.m_p, dp);
#ifdef LBA_DEBUG_GROUND_TRUTH_STATE_ERROR
    dp += LA::AlignedVector3f::GetRandom(dpMax);
    dr += LA::AlignedVector3f::GetRandom(drMax * UT_FACTOR_DEG_TO_RAD);
#endif
    xcs[ic].Set(dp, dr);
    if (motion) {
      LA::Vector9f &xm = xms[ic];
      if (CGT.m_v.Valid()) {
        LA::AlignedVector3f::amb(CGT.m_v, C.m_v, dv);
#ifdef LBA_DEBUG_GROUND_TRUTH_STATE_ERROR
        dv += LA::AlignedVector3f::GetRandom(dvMax);
#endif
        xm.Set012(dv);
      }
      if (CGT.m_ba.Valid()) {
        LA::AlignedVector3f::amb(CGT.m_ba, C.m_ba, dba);
#ifdef LBA_DEBUG_GROUND_TRUTH_STATE_ERROR
        dba += LA::AlignedVector3f::GetRandom(dbaMax);
#endif
        xm.Set345(dba);
      }
      if (CGT.m_bw.Valid()) {
        LA::AlignedVector3f::amb(CGT.m_bw, C.m_bw, dbw);
#ifdef LBA_DEBUG_GROUND_TRUTH_STATE_ERROR
        dbw += LA::AlignedVector3f::GetRandom(dbwMax * UT_FACTOR_DEG_TO_RAD);
#endif
        xm.Set678(dbw);
      }
    }
  }
  ConvertCameraUpdates(xs->Data(), &m_xp2s, &m_xr2s);
  if (motion) {
    ConvertMotionUpdates((float *) xms, &m_xv2s, &m_xba2s, &m_xbw2s);
  }
}
#endif

bool LocalBundleAdjustor::SolveSchurComplementLast() {
  const int Nc = static_cast<int>(m_LFs.size());
  if (Nc < 3) {
    return false;
  }
  const int pc = 6, pm = 9;
  const int pcm = pc + pm/*, pmcm = pm + pcm*/;
  const float eps = FLT_EPSILON;
  const float epsr = UT::Inverse(BA_VARIANCE_MAX_ROTATION, BA_WEIGHT_FEATURE, eps);
  const float epsp = UT::Inverse(BA_VARIANCE_MAX_POSITION, BA_WEIGHT_FEATURE, eps);
  const float epsv = UT::Inverse(BA_VARIANCE_MAX_VELOCITY, BA_WEIGHT_FEATURE, eps);
  const float epsba = UT::Inverse(BA_VARIANCE_MAX_BIAS_ACCELERATION, BA_WEIGHT_FEATURE, eps);
  const float epsbw = UT::Inverse(BA_VARIANCE_MAX_BIAS_GYROSCOPE, BA_WEIGHT_FEATURE, eps);
  //const float _eps[pmcm] = {epsv, epsv, epsv, epsba, epsba, epsba, epsbw, epsbw, epsbw,
  //                          epsp, epsp, epsp, epsr, epsr, epsr,
  //                          epsv, epsv, epsv, epsba, epsba, epsba, epsbw, epsbw, epsbw};
  const float _eps[pcm] = {epsp, epsp, epsp, epsr, epsr, epsr,
                           epsv, epsv, epsv, epsba, epsba, epsba, epsbw, epsbw, epsbw};
  const float ar = UT::Inverse(BA_VARIANCE_REGULARIZATION_ROTATION, BA_WEIGHT_FEATURE);
  const float ap = UT::Inverse(BA_VARIANCE_REGULARIZATION_POSITION, BA_WEIGHT_FEATURE);
  const float av = UT::Inverse(BA_VARIANCE_REGULARIZATION_VELOCITY, BA_WEIGHT_FEATURE);
  const float aba = UT::Inverse(BA_VARIANCE_REGULARIZATION_BIAS_ACCELERATION, BA_WEIGHT_FEATURE);
  const float abw = UT::Inverse(BA_VARIANCE_REGULARIZATION_BIAS_GYROSCOPE, BA_WEIGHT_FEATURE);
  
  LA::AlignedMatrixXf A;
  LA::AlignedVectorXf b;
  //m_work.Resize(A.BindSize(pmcm, pmcm) + b.BindSize(pmcm));
  //A.Bind(m_work.Data(), pmcm, pmcm);
  //b.Bind(A.BindNext(), pmcm);
  m_work.Resize(A.BindSize(pcm, pcm) + b.BindSize(pcm));
  A.Bind(m_work.Data(), pcm, pcm);
  b.Bind(A.BindNext(), pcm);

  Camera::Factor::Unitary::CC Acc;
  LA::AlignedMatrix9x9f Amm;
  //const int ic1 = Nc - 2, iLF1 = m_ic2LF[ic1];
  //const int ic2 = Nc - 1, iLF2 = m_ic2LF[ic2];
  //const Camera::Factor::Unitary::MM &Amm1 = m_SAcmsLF[iLF1].m_Au.m_Amm;
  //Amm.Set(Amm1.m_A);
  //Amm.IncreaseDiagonal(av, aba, abw);
  //A.SetBlock(0, 0, Amm);
  //const Camera::Factor &Acm2 = m_SAcmsLF[iLF2];
  //A.SetBlock(0, pm, Acm2.m_Ab.m_Amc);
  //A.SetBlock(0, pcm, Acm2.m_Ab.m_Amm);
  //b.SetBlock(0, Amm1.m_b);
  //Camera::Factor::Unitary::CC::AmB(m_SAcusLF[iLF2], m_SMcusLF[iLF2], Acc);
  //Acc.m_A.IncreaseDiagonal(ap, ar);
  //A.SetBlock(pm, pm, Acc.m_A);
  //A.SetBlock(pm, pcm, Acm2.m_Au.m_Acm);
  //b.SetBlock(pm, Acc.m_b);
  //Amm.Set(Acm2.m_Au.m_Amm.m_A);
  //Amm.IncreaseDiagonal(av, aba, abw);
  //A.SetBlock(pcm, pcm, Amm);
  //b.SetBlock(pcm, Acm2.m_Au.m_Amm.m_b);
  //A.SetLowerFromUpper();
  ////A.Print(true);
  ////A.PrintDiagonal(true);
  const int ic2 = Nc - 1, iLF2 = m_ic2LF[ic2];
  Camera::Factor::Unitary::CC::AmB(m_SAcusLF[iLF2], m_SMcusLF[iLF2], Acc);
  Acc.m_A.IncreaseDiagonal(ap, ar);
  A.SetBlockDiagonal(0, Acc.m_A);
  const Camera::Factor &Acm2 = m_SAcmsLF[iLF2];
  A.SetBlock(0, pc, Acm2.m_Au.m_Acm);
  b.SetBlock(0, Acc.m_b);
  Amm.Set(Acm2.m_Au.m_Amm.m_A);
  Amm.IncreaseDiagonal(av, aba, abw);
  A.SetBlock(pc, pc, Amm);
  b.SetBlock(pc, Acm2.m_Au.m_Amm.m_b);

  A.SetLowerFromUpper();
  //A.Print(true);
  //A.PrintDiagonal(true);
  if (!A.SolveLDL(b, _eps)) {
    return false;
  }
  b.MakeMinus();

  LA::AlignedVector3f x;
  const int N = Nc * pcm;
  m_xsGN.Resize(N);     m_xsGN.MakeZero();
  m_xp2s.Resize(Nc);    m_xp2s.MakeZero();
  m_xr2s.Resize(Nc);    m_xr2s.MakeZero();
  m_xv2s.Resize(Nc);    m_xv2s.MakeZero();
  m_xba2s.Resize(Nc);   m_xba2s.MakeZero();
  m_xbw2s.Resize(Nc);   m_xbw2s.MakeZero();
  float *xc2 = m_xsGN.Data() + ic2 * pc;
  float *xm2 = m_xsGN.End() - pm;
  //float *xm1 = xm2 - pm;
  //b.GetBlock(0, x);   x.Get(xm1);     m_xv2s[ic1] = x.SquaredLength();
  //b.GetBlock(3, x);   x.Get(xm1 + 3); m_xba2s[ic1] = x.SquaredLength();
  //b.GetBlock(6, x);   x.Get(xm1 + 6); m_xbw2s[ic1] = x.SquaredLength();
  //b.GetBlock(9, x);   x.Get(xc2);     m_xp2s[ic2] = x.SquaredLength();
  //b.GetBlock(12, x);  x.Get(xc2 + 3); m_xr2s[ic2] = x.SquaredLength();
  //b.GetBlock(15, x);  x.Get(xm2);     m_xv2s[ic2] = x.SquaredLength();
  //b.GetBlock(18, x);  x.Get(xm2 + 3); m_xba2s[ic2] = x.SquaredLength();
  //b.GetBlock(21, x);  x.Get(xm2 + 6); m_xbw2s[ic2] = x.SquaredLength();
  b.GetBlock(0, x);   x.Get(xc2);     m_xp2s[ic2] = x.SquaredLength();
  b.GetBlock(3, x);   x.Get(xc2 + 3); m_xr2s[ic2] = x.SquaredLength();
  b.GetBlock(6, x);   x.Get(xm2);     m_xv2s[ic2] = x.SquaredLength();
  b.GetBlock(9, x);   x.Get(xm2 + 3); m_xba2s[ic2] = x.SquaredLength();
  b.GetBlock(12, x);  x.Get(xm2 + 6); m_xbw2s[ic2] = x.SquaredLength();
  return true;
}

void LocalBundleAdjustor::PrepareConditioner()
{
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
  const int Nb = std::min(LBA_PCG_CONDITIONER_BAND, LBA_MAX_SLIDING_TRACK_LENGTH);
  const int Nc = static_cast<int>(m_LFs.size());
  if (Nb <= 1)
  {
    m_Mcs.Resize(Nc);
    m_Mms.Resize(Nc);
    for (int ic = 0; ic < Nc; ++ic) {//遍历所有普通帧,构造预优矩阵
      const int iLF = m_ic2LF[ic];
      m_Mcs[ic].Set(m_SAcusLF[iLF]/*普通帧pose自己和自己的H以及自己对应的-b*/, BA_PCG_CONDITIONER_MAX, BA_PCG_CONDITIONER_EPSILON, epsc);
      //m_Mcs[ic].Set(m_Acus[ic], BA_PCG_CONDITIONER_MAX, BA_PCG_CONDITIONER_EPSILON, epsc);
      m_Mms[ic].Set(m_Amus[ic], BA_PCG_CONDITIONER_MAX, BA_PCG_CONDITIONER_EPSILON, epsm);
    }
    return;
  }
  m_Mcc.Resize(Nc, Nb);     m_MccT.Resize(Nc, Nb);
  m_Mcm.Resize(Nc, 2);      m_McmT.Resize(Nc, 2);
  m_Mmc.Resize(Nc, Nb - 1); m_MmcT.Resize(Nc, Nb - 1);
  m_Mmm.Resize(Nc, 2);      m_MmmT.Resize(Nc, 2);
  for (int ic = 0; ic < Nc; ++ic)
  {
    LA::AlignedMatrix6x6f *Accs = m_Mcc[ic];
    Accs[0] = m_Acus[ic];
    const LA::AlignedMatrix6x6f *Acbs = m_Acbs.Data() + m_ic2b[ic] - 1;
    const int Nbc = ic + Nb > Nc ? Nc - ic : Nb;
    for (int ib = 1; ib < Nbc; ++ib)
    {
      Accs[ib] = Acbs[ib];
    }
    m_Mcm[ic][0] = m_SAcmsLF[m_ic2LF[ic]].m_Au.m_Acm;
    m_Mmm[ic][0] = m_Amus[ic];
    const int _ic = ic + 1;
    if (_ic == Nc)
    {
      continue;
    }
    const Camera::Factor::Binary &Ab = m_SAcmsLF[m_ic2LF[_ic]].m_Ab;
    m_Mcm[ic][1] = Ab.m_Acm;
    m_Mmm[ic][1] = Ab.m_Amm;
    LA::AlignedMatrix9x6f *Amcs = m_Mmc[ic];
    Amcs[0] = Ab.m_Amc;
    const int Nbm = Nbc - 1;
    for (int ib = 1; ib < Nbm; ++ib)
    {
      Amcs[ib].MakeZero();
    }
  }
#ifdef LBA_DEBUG_EIGEN_PCG
  EigenMatrixXd e_A;
  const double e_epsc[pc] = {epsp, epsp, epsp, epsr, epsr, epsr};
  const double e_epsm[pm] = {epsv, epsv, epsv, epsba, epsba, epsba, epsbw, epsbw, epsbw};
  const int pcm = pc + pm, N = Nc * (pc + pm);
  e_A.resize(N, N);
  e_A.setZero();
  for (int ic = 0, icp = 0, imp = pc; ic < Nc; ++ic, icp += pcm, imp += pcm) {
    e_A.block<pc, pc>(icp, icp) = EigenMatrix6x6f(m_Acus[ic]).cast<double>();
    const LA::AlignedMatrix6x6f *Acbs = m_Acbs.Data() + m_ic2b[ic] - 1;
    const int Nbc = ic + Nb > Nc ? Nc - ic : Nb;
    for (int ib = 1, _ic = ic + 1; ib < Nbc; ++ib, ++_ic) {
      e_A.block<pc, pc>(icp, _ic * pcm) = EigenMatrix6x6f(Acbs[ib]).cast<double>();
    }
    const int iLF = m_ic2LF[ic];
    const Camera::EigenFactor e_Acm = m_SAcmsLF[iLF];
    e_A.block<pm, pm>(imp, imp) = EigenMatrix9x9f(m_Amus[ic]).cast<double>();
    e_A.block<pc, pm>(icp, imp) = e_Acm.m_Au.m_Acm.cast<double>();
    if (ic > 0) {
      const int _icp = icp - pcm, _imp = imp - pcm;
      e_A.block<pc, pm>(_icp, imp) = e_Acm.m_Ab.m_Acm.cast<double>();
      e_A.block<pm, pc>(_imp, icp) = e_Acm.m_Ab.m_Amc.cast<double>();
      e_A.block<pm, pm>(_imp, imp) = e_Acm.m_Ab.m_Amm.cast<double>();
    }
  }
  e_A.SetLowerFromUpper();
  EigenMatrixXd e_M = e_A;
#endif

  AlignedVector<LA::AlignedMatrix6x6f> AccsT;
  AlignedVector<LA::AlignedMatrix9x6f> AcmsT;
  AlignedVector<LA::AlignedMatrix6x9f> AmcsT;
  AlignedVector<LA::AlignedMatrix9x9f> AmmsT;
  m_work.Resize((AccsT.BindSize(Nb) + AcmsT.BindSize(2) +
                 AmcsT.BindSize(Nb) + AmmsT.BindSize(2)) / sizeof(float));
  AccsT.Bind(m_work.Data(), Nb);
  AcmsT.Bind(AccsT.BindNext(), 2);
  AmcsT.Bind(AcmsT.BindNext(), Nb);
  AmmsT.Bind(AmcsT.BindNext(), Nb);
  for (int ic = 0; ic < Nc; ++ic)
  {
    LA::AlignedMatrix6x6f *Mccs = m_Mcc[ic], *MccsT = m_MccT[ic];
    LA::AlignedMatrix6x9f *Mcms = m_Mcm[ic], *MmcsT = m_MmcT[ic];
    LA::AlignedMatrix9x6f *McmsT = m_McmT[ic], *Mmcs = m_Mmc[ic];
    LA::AlignedMatrix9x9f *Mmms = m_Mmm[ic], *MmmsT = m_MmmT[ic];
    const int Nbcc = ic + Nb > Nc ? Nc - ic : Nb;
    const int Nbcm = ic + 1 == Nc ? 1 : 2;
    const int Nbmc = Nbcc - 1;
    const int Nbmm = Nbcm;
#ifdef LBA_DEBUG_EIGEN_PCG
    const int icp = ic * pcm, imp = icp + pc;
    e_M.Marginalize(icp, pc, e_epsc, false, false);
    e_M.Marginalize(imp, pm, e_epsm, false, false);
#endif
    LA::AlignedMatrix6x6f &Mcc = Mccs[0];
    if (Mcc.InverseLDL(epsc))
    {
      MccsT[0] = Mcc;
      Mcc.MakeMinus();
      for (int ib = 1; ib < Nbcc; ++ib)
      {
        Mccs[ib].GetTranspose(AccsT[ib]);
        LA::AlignedMatrix6x6f::ABT(Mcc, AccsT[ib], Mccs[ib]);
        Mccs[ib].GetTranspose(MccsT[ib]);
      }
      for (int ib = 0; ib < Nbcm; ++ib)
      {
        Mcms[ib].GetTranspose(AcmsT[ib]);
        LA::AlignedMatrix9x6f::ABT(Mcc, AcmsT[ib], Mcms[ib]);
        Mcms[ib].GetTranspose(McmsT[ib]);
      }
      for (int ib = 1; ib < Nbcc; ++ib)
      {
        const LA::AlignedMatrix6x6f &MccT = MccsT[ib];
        const int _ic = ic + ib;
        LA::AlignedMatrix6x6f *_Mccs = m_Mcc[_ic] - ib;
        LA::AlignedMatrix6x6f::AddABTToUpper(MccT, AccsT[ib], _Mccs[ib]);
#ifdef LBA_DEBUG_EIGEN_PCG
        _Mccs[ib].SetLowerFromUpper();
#endif
        for (int jb = ib + 1; jb < Nbcc; ++jb)
        {
          LA::AlignedMatrix6x6f::AddABTTo(MccT, AccsT[jb], _Mccs[jb]);
        }
        if (ib == 1)
        {
          LA::AlignedMatrix9x6f::AddABTTo(MccT, AcmsT[ib], m_Mcm[_ic][0]);
        }
      }
      for (int ib = 0; ib < Nbcm; ++ib)
      {
        const LA::AlignedMatrix9x6f &McmT = McmsT[ib];
        const int _ic = ic + ib;
        const LA::AlignedMatrix6x6f *_AccsT = AccsT.Data() + 1;
        LA::AlignedMatrix9x6f *_Mmcs = m_Mmc[_ic] - ib;
        for (int jb = ib; jb < Nbmc; ++jb)
        {
          LA::AlignedMatrix9x6f::AddABTTo(McmT, _AccsT[jb], _Mmcs[jb]);
        }
        LA::AlignedMatrix9x9f *_Mmms = m_Mmm[_ic];
        LA::AlignedMatrix9x9f::AddABTToUpper(McmT, AcmsT[ib], _Mmms[0]);
#ifdef LBA_DEBUG_EIGEN_PCG
        _Mmms[0].SetLowerFromUpper();
#endif
        if (ib == 0 && Nbcm == 2)
        {
          LA::AlignedMatrix9x9f::AddABTTo(McmT, AcmsT[1], _Mmms[1]);
        }
      }
    } else {
      for (int ib = 0; ib < Nbcc; ++ib)
      {
        Mccs[ib].MakeZero();
        MccsT[ib].MakeZero();
      }
      for (int ib = 0; ib < Nbcm; ++ib)
      {
        Mcms[ib].MakeZero();
        McmsT[ib].MakeZero();
      }
    }
    LA::AlignedMatrix9x9f &Mmm = Mmms[0];
    if (Mmm.InverseLDL(epsm))
    {
      MmmsT[0] = Mmm;
      Mmm.MakeMinus();
      for (int ib = 0; ib < Nbmc; ++ib)
      {
        Mmcs[ib].GetTranspose(AmcsT[ib]);
        LA::AlignedMatrix9x9f::ABT(Mmm, AmcsT[ib], Mmcs[ib]);
        Mmcs[ib].GetTranspose(MmcsT[ib]);
      }
      if (Nbmm == 2)
      {
        Mmms[1].GetTranspose(AmmsT[1]);
        LA::AlignedMatrix9x9f::ABT(Mmm, AmmsT[1], Mmms[1]);
        Mmms[1].GetTranspose(MmmsT[1]);
      }
      for (int ib = 0; ib < Nbmc; ++ib)
      {
        const LA::AlignedMatrix6x9f &MmcT = MmcsT[ib];
        const int _ic = ic + ib + 1;
        LA::AlignedMatrix6x6f *_Mccs = m_Mcc[_ic] - ib;
        LA::AlignedMatrix6x9f::AddABTToUpper(MmcT, AmcsT[ib], _Mccs[ib]);
#ifdef LBA_DEBUG_EIGEN_PCG
        _Mccs[ib].SetLowerFromUpper();
#endif
        for (int jb = ib + 1; jb < Nbmc; ++jb)
        {
          LA::AlignedMatrix6x9f::AddABTTo(MmcT, AmcsT[jb], _Mccs[jb]);
        }
        if (ib == 0 && Nbmm == 2)
        {
          LA::AlignedMatrix9x9f::AddABTTo(MmcT, AmmsT[1], m_Mcm[_ic][0]);
        }
      }
      if (Nbmm == 2)
      {
        const LA::AlignedMatrix9x9f &MmmT = MmmsT[1];
        const int _ic = ic + 1;
        LA::AlignedMatrix9x6f *_Mmcs = m_Mmc[_ic] - 1;
        for (int jb = 1; jb < Nbmc; ++jb)
        {
          LA::AlignedMatrix9x9f::AddABTTo(MmmT, AmcsT[jb], _Mmcs[jb]);
        }
        LA::AlignedMatrix9x9f::AddABTToUpper(MmmT, AmmsT[1], m_Mmm[_ic][0]);
#ifdef LBA_DEBUG_EIGEN_PCG
        m_Mmm[_ic][0].SetLowerFromUpper();
#endif
      }
    } else
    {
      for (int ib = 0; ib < Nbmc; ++ib)
      {
        Mmcs[ib].MakeZero();
        MmcsT[ib].MakeZero();
      }
      for (int ib = 0; ib < Nbmm; ++ib)
      {
        Mmms[ib].MakeZero();
        MmmsT[ib].MakeZero();
      }
    }
#ifdef LBA_DEBUG_EIGEN_PCG
    for (int ic1 = ic, icp1 = icp, imp1 = imp; ic1 < Nc; ++ic1, icp1 += pcm, imp1 += pcm) {
      const int Nbcc1 = ic1 + Nb > Nc ? Nc - ic1 : Nb;
      const int Nbcm1 = ic1 + 1 == Nc ? 1 : 2;
      const int Nbmc1 = Nbcc1 - 1;
      const int Nbmm1 = Nbcm1;
      for (int ib = 0, ic2 = ic1, icp2 = icp1; ib < Nbcc1; ++ib, ++ic2, icp2 += pcm) {
        const EigenMatrix6x6f e_Mcc = e_M.block<pc, pc>(icp1, icp2).cast<float>();
        e_Mcc.AssertEqual(m_Mcc[ic1][ib], 1, UT::String("Mcc[%d][%d]", ic1, ic2));
        e_M.block<pc, pc>(icp1, icp2) = EigenMatrix6x6f(m_Mcc[ic1][ib]).cast<double>();
      }
      for (int ib = 0, ic2 = ic1, imp2 = imp1; ib < Nbcm1; ++ib, ++ic2, imp2 += pcm) {
        const EigenMatrix6x9f e_Mcm = e_M.block<pc, pm>(icp1, imp2).cast<float>();
        e_Mcm.AssertEqual(m_Mcm[ic1][ib], 1, UT::String("Mcm[%d][%d]", ic1, ic2));
        e_M.block<pc, pm>(icp1, imp2) = EigenMatrix6x9f(m_Mcm[ic1][ib]).cast<double>();
      }
      for (int ib = 0, ic2 = ic1 + 1, icp2 = icp1 + pcm; ib < Nbmc1; ++ib, ++ic2, icp2 += pcm) {
        const EigenMatrix9x6f e_Mmc = e_M.block<pm, pc>(imp1, icp2).cast<float>();
        e_Mmc.AssertEqual(m_Mmc[ic1][ib], 1, UT::String("Mmc[%d][%d]", ic1, ic2));
        e_M.block<pm, pc>(imp1, icp2) = EigenMatrix9x6f(m_Mmc[ic1][ib]).cast<double>();
      }
      for (int ib = 0, ic2 = ic1, imp2 = imp1; ib < Nbmm1; ++ib, ++ic2, imp2 += pcm) {
        const EigenMatrix9x9f e_Mmm = e_M.block<pm, pm>(imp1, imp2).cast<float>();
        e_Mmm.AssertEqual(m_Mmm[ic1][ib], 1, UT::String("Mmm[%d][%d]", ic1, ic2));
        e_M.block<pm, pm>(imp1, imp2) = EigenMatrix9x9f(m_Mmm[ic1][ib]).cast<double>();
      }
    }
#endif
  }
#ifdef LBA_DEBUG_EIGEN_PCG
  const EigenMatrixXd e_AI = EigenMatrixXd(e_A.inverse());
  const EigenMatrixXd e_I1 = EigenMatrixXd(e_A * e_AI), e_I2 = EigenMatrixXd(e_AI * e_A);
  EigenMatrix15x15f e_I;
  e_I.setIdentity();
  for (int ic1 = 0, icp1 = 0; ic1 < Nc; ++ic1, icp1 += pcm) {
    for (int ic2 = 0, icp2 = 0; ic2 < Nc; ++ic2, icp2 += pcm) {
      const std::string str1 = UT::String("I1[%d][%d]", ic1, ic2);
      const std::string str2 = UT::String("I2[%d][%d]", ic1, ic2);
      if (ic1 == ic2) {
        EigenMatrix15x15f(e_I1.block<pcm, pcm>(icp1, icp2).cast<float>()).AssertEqual(e_I, 1, str1);
        EigenMatrix15x15f(e_I2.block<pcm, pcm>(icp1, icp2).cast<float>()).AssertEqual(e_I, 1, str2);
      } else {
        EigenMatrix15x15f(e_I1.block<pcm, pcm>(icp1, icp2).cast<float>()).AssertZero(1, str1);
        EigenMatrix15x15f(e_I2.block<pcm, pcm>(icp1, icp2).cast<float>()).AssertZero(1, str2);
      }
    }
  }
  const float rMax = 1.0f;
  m_rs.Resize(N);
  m_rs.Random(rMax);
  ApplyM(m_rs, &m_zs);
  EigenVectorXd e_r;
  e_r.Resize(N);
  const LA::Vector6f *rcs = (LA::Vector6f *) m_rs.Data(), *zcs = (LA::Vector6f *) m_zs.Data();
  const LA::Vector9f *rms = (LA::Vector9f *) (rcs + Nc), *zms = (LA::Vector9f *) (zcs + Nc);
  for (int ic = 0, icp = 0, imp = pc; ic < Nc; ++ic, icp += pcm, imp += pcm) {
    e_r.block<pc, 1>(icp, 0) = EigenVector6f(rcs[ic]).cast<double>();
    e_r.block<pm, 1>(imp, 0) = EigenVector9f(rms[ic]).cast<double>();
  }
  const EigenVectorXd e_z1 = e_AI * e_r;
  EigenVectorXd e_z2;
  e_z2.Resize(N);
  for (int ic = 0, icp = 0, imp = pc; ic < Nc; ++ic, icp += pcm, imp += pcm) {
    const EigenVector6f e_zc1 = EigenVector6f(e_z1.block<pc, 1>(icp, 0).cast<float>());
    const EigenVector6f e_zc2 = EigenVector6f(zcs[ic]);
    e_zc1.AssertEqual(e_zc2, 1, UT::String("zc[%d]", ic));
    const EigenVector9f e_zm1 = EigenVector9f(e_z1.block<pm, 1>(imp, 0).cast<float>());
    const EigenVector9f e_zm2 = EigenVector9f(zms[ic]);
    e_zm1.AssertEqual(e_zm2, 1, UT::String("zm[%d]", ic));
    e_z2.block<pc, 1>(icp, 0) = e_zc2.cast<double>();
    e_z2.block<pm, 1>(imp, 0) = e_zm2.cast<double>();
  }
  const EigenVectorXd e_e1 = EigenVectorXd(e_A * e_z1 - e_r);
  const EigenVectorXd e_e2 = EigenVectorXd(e_A * e_z2 - e_r);
  UT::Print("%e vs %e\n", e_e1.norm(), e_e2.norm());
#endif
}
//z0 = M.inv * r0 m_Mcs,m_Mms是M^-1，即S的对角线的逆
void LocalBundleAdjustor::ApplyM(const LA::AlignedVectorXf &xs/*r0*/, LA::AlignedVectorXf *Mxs) {
  const int Nb = std::min(LBA_PCG_CONDITIONER_BAND, LBA_MAX_SLIDING_TRACK_LENGTH);
  const int Nc = int(m_LFs.size());
  if (Nb <= 1) {
#ifdef CFG_PCG_FULL_BLOCK
    LA::ProductVector6f xc;
    LA::AlignedVector9f xm;
#else
    LA::AlignedVector3f xp, xr, xv, xba, xbw;
#endif
    Mxs->Resize(xs.Size());
    const LA::Vector6f *xcs = (LA::Vector6f *) xs.Data();
    LA::Vector6f *Mxcs = (LA::Vector6f *) Mxs->Data();
    for (int ic = 0; ic < Nc; ++ic) {
#ifdef CFG_PCG_FULL_BLOCK
      xc.Set(xcs[ic]);
      m_Mcs[ic].Apply(xc, (PCG_TYPE *) &Mxcs[ic]);
#else
      xcs[ic].Get(xp, xr);//取出位置和朝向的
      m_Mcs[ic].Apply(xp, xr, (float *) Mxcs[ic]);
#endif
    }
    const LA::Vector9f *xms = (LA::Vector9f *) (xcs + Nc);
    LA::Vector9f *Mxms = (LA::Vector9f *) (Mxcs + Nc);
    for (int ic = 0; ic < Nc; ++ic) {
#ifdef CFG_PCG_FULL_BLOCK
      xm.Set(xms[ic]);
      m_Mms[ic].Apply(xm, (PCG_TYPE *) &Mxms[ic]);
#else
      xms[ic].Get(xv, xba, xbw);
      m_Mms[ic].Apply(xv, xba, xbw, (float *) Mxms[ic]);
#endif
    }
    return;
  }
  LA::ProductVector6f bc;
  LA::AlignedVector9f bm;
  Mxs->Set(xs);
  LA::Vector6f *bcs = (LA::Vector6f *) Mxs->Data();
  LA::Vector9f *bms = (LA::Vector9f *) (bcs + Nc);
  for (int ic = 0; ic < Nc; ++ic) {
    const int Nbcc = ic + Nb > Nc ? Nc - ic : Nb;
    const int Nbcm = ic + 1 == Nc ? 1 : 2;
    const int Nbmc = Nbcc - 1;
    const int Nbmm = Nbcm;
    
    bc.Set(bcs[ic]);
    const LA::AlignedMatrix6x6f *MccsT = m_MccT[ic];
    for (int ib = 1; ib < Nbcc; ++ib) {
      LA::AlignedMatrix6x6f::AddAbTo<float>(MccsT[ib], bc, bcs[ic + ib]);
    }
    const LA::AlignedMatrix9x6f *McmsT = m_McmT[ic];
    for (int ib = 0; ib < Nbcm; ++ib) {
      LA::AlignedMatrix9x6f::AddAbTo<float>(McmsT[ib], bc, bms[ic + ib]);
    }
    LA::AlignedMatrix6x6f::Ab<float>(MccsT[0], bc, bcs[ic]);

    bm.Set(bms[ic]);
    const LA::AlignedMatrix6x9f *MmcsT = m_MmcT[ic];
    for (int ib = 0; ib < Nbmc; ++ib) {
      LA::AlignedMatrix6x9f::AddAbTo<float>(MmcsT[ib], bm, bcs[ic + ib + 1]);
    }
    const LA::AlignedMatrix9x9f *MmmsT = m_MmmT[ic];
    if (Nbmm == 2) {
      LA::AlignedMatrix9x9f::AddAbTo<float>(MmmsT[1], bm, bms[ic + 1]);
    }
    LA::AlignedMatrix9x9f::Ab<float>(MmmsT[0], bm, bms[ic]);
  }
  m_bcs.Resize(Nc);
  m_bms.Resize(Nc);
  for (int ic = Nc - 1; ic >= 0; --ic) {
    const int Nbcc = ic + Nb > Nc ? Nc - ic : Nb;
    const int Nbcm = ic + 1 == Nc ? 1 : 2;
    const int Nbmc = Nbcc - 1;
    const int Nbmm = Nbcm;

    float *_bm = bms[ic];
    const LA::AlignedMatrix9x6f *Mmcs = m_Mmc[ic];
    for (int ib = 0; ib < Nbmc; ++ib) {
      LA::AlignedMatrix9x6f::AddAbTo(Mmcs[ib], m_bcs[ic + ib + 1], _bm);
    }
    if (Nbmm == 2) {
      LA::AlignedMatrix9x9f::AddAbTo(m_Mmm[ic][1], m_bms[ic + 1], _bm);
    }
    m_bms[ic].Set(_bm);

    float *_bc = bcs[ic];
    const LA::AlignedMatrix6x6f *Mccs = m_Mcc[ic];
    for (int ib = 1; ib < Nbcc; ++ib) {
      LA::AlignedMatrix6x6f::AddAbTo(Mccs[ib], m_bcs[ic + ib], _bc);
    }
    const LA::AlignedMatrix6x9f *Mcms = m_Mcm[ic];
    for (int ib = 0; ib < Nbcm; ++ib) {
      LA::AlignedMatrix6x9f::AddAbTo(Mcms[ib], m_bms[ic + ib], _bc);
    }
    m_bcs[ic].Set(_bc);
  }
}
//计算 m_drs  = A*x0 A是舒尔补以后的S
//计算顺序：对角线App部分的*x0,非对角线共视帧之间App部分*x0,ApplyAcm算的就是前后motion约束带来的A里的部分*x0
// （不用计算App了是因为非对角线共视帧之间App部分已经算过了）
void LocalBundleAdjustor::ApplyA(const LA::AlignedVectorXf &xs, LA::AlignedVectorXf *Axs) {
  //LA::AlignedVector6f v6;
  const LA::Vector6f *xcs = (LA::Vector6f *) xs.Data();//pose部分的增量
  ConvertCameraUpdates(xcs, &m_xcsP);//m_xcsP保存pose的增量
  Axs->Resize(xs.Size());
  LA::Vector6f *Axcs = (LA::Vector6f *) Axs->Data();
  const int Nc = int(m_LFs.size());
  if (m_Acus.Size() < Nc) {
    Axs->MakeZero();
    return;
  }
  //m_Axcs.Resize(Nc);对角线部分的A*x0
  for (int ic = 0; ic < Nc; ++ic) {
    LA::AlignedMatrix6x6f::Ab(m_Acus[ic]/*舒尔补以后每帧自己和自己的对角线部分*/, m_xcsP[ic]/*对应的x*/, (float *) &Axcs[ic]);
  }
  for (int ic = 0; ic < Nc; ++ic) {//遍历所有的滑窗中的帧
    const int ib = m_ic2b[ic];
    const LA::AlignedMatrix6x6f *Acbs = m_Acbs.Data() + ib;//右上非对角线的posexpose(即不同帧因为共视和motion导致有影响的地方)
    const LA::AlignedMatrix6x6f *AcbTs = m_AcbTs.Data() + ib;//右下非对角线的posexpose(即不同帧因为共视和motion导致有影响的地方)
    const LA::ProductVector6f &xc = m_xcsP[ic];//当前帧pose
    float *Axc = Axcs[ic];
    const LocalFrame &LF = m_LFs[m_ic2LF[ic]];//当前遍历的普通帧
    const int Nk = int(LF.m_iLFsMatch.size());//共视的数量
    for (int ik = 0; ik < Nk; ++ik) {
      const int _ic = ic + ik + 1;//
      LA::AlignedMatrix6x6f::AddAbTo(Acbs[ik], m_xcsP[_ic], Axc);//Aic_ic*p_ic = Axcs[ic] 右上的这个矩阵块*x_ic
      LA::AlignedMatrix6x6f::AddAbTo(AcbTs[ik], xc, (float *) &Axcs[_ic]);//A_icic*pic = Axcs[_ic] 坐下的的这个矩阵块*xic
    }
  }
  //ConvertCameraUpdates(m_Axcs, Axcs);
  ApplyAcm(m_xcsP.Data()/*Xpose*/, (LA::Vector9f *) (xcs + Nc)/*Xmotion*/, Axcs/*A*X pose*/
          , (LA::Vector9f *) (Axcs + Nc/*motion部分*/), false/*需不需要算pose部分的*/,m_Amus.Size() == Nc ? m_Amus.Data() : NULL);
}

void LocalBundleAdjustor::ApplyAcm(const LA::ProductVector6f *xcs, const LA::Vector9f *xms,
                                   LA::Vector6f *Axcs, LA::Vector9f *Axms, const bool Acc,
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
  const int Nc = int(m_LFs.size());
  for (int ic = 0, r = 0; ic < Nc; ++ic, r = 1 - r) {//遍历所有滑窗帧
    const LA::ProductVector6f &xc = xcs[ic];//这帧的Xpose
    LA::AlignedVector9f &xm = v9[r];//这帧对应的Xmotion
    xm.Set(xms[ic]);//xm = xms[ic] //为sm设置这帧对应的motion
    float *Axc = Axcs[ic], *Axm = Axms[ic];
    const Camera::Factor &SAcm = m_SAcmsLF[m_ic2LF[ic]];
    if (Amus) {//Amm * xm
      LA::AlignedMatrix9x9f::Ab(Amus[ic], xm, Axm);
    } else {
      A99.Set(SAcm.m_Au.m_Amm.m_A);
      LA::AlignedMatrix9x9f::Ab(A99, xm, Axm);
    }//A中同一帧Pose x M * cm （右上角）
    LA::AlignedMatrix6x9f::AddAbTo(SAcm.m_Au.m_Acm, xm, Axc);
    SAcm.m_Au.m_Acm.GetTranspose(A96);//左下角
    LA::AlignedMatrix9x6f::AddAbTo(A96, xc, Axm);
    if (ic == 0) {
      continue;
    }
    const int _ic = ic - 1;
    const LA::ProductVector6f &_xc = xcs[_ic];
    const LA::AlignedVector9f &_xm = v9[1 - r];
    float *_Axc = Axcs[_ic], *_Axm = Axms[_ic];
    if (Acc) {//不需要再算前后帧的posexpose了,因为前面算过了
      LA::AlignedMatrix6x6f::AddAbTo(SAcm.m_Ab.m_Acc, xc, _Axc);
      SAcm.m_Ab.m_Acc.GetTranspose(A66);
      LA::AlignedMatrix6x6f::AddAbTo(A66, _xc, Axc);
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

LocalBundleAdjustor::Residual LocalBundleAdjustor::ComputeResidual(const LA::AlignedVectorXf &xs,
                                                                   const bool minus) {
  Residual R;
  ApplyA(xs, &m_rs);
  if (minus) {
    R.m_F = xs.Dot(m_rs) / 2 - xs.Dot(m_bs);
    m_rs -= m_bs;
  } else {
    R.m_F = xs.Dot(m_rs) / 2 + xs.Dot(m_bs);
    m_rs += m_bs;
  }
  R.m_r2 = m_rs.SquaredLength();
  return R;
}

//反求逆深度的增量
//step1:先遍历所有关键帧中的新地图点,将需要更新的关键帧以及地图点设置成需要反求的flags
//step2:遍历滑窗中所有的帧,如果他们对应的pose增量太小，那么就不进行更新,观测到的地图点也不会反求深度增量
//step3:遍历所有的地图点,需要更新增量的地图点的du = Huu^-1*-bu
//step4:遍历所有的滑窗中的帧（因为Wcu.t * Huu^-1存在滑窗中的观测中）,需要更新增量的地图点的du =Wcu.t * Huu^-1 * dxc +  Huu^-1*-bu (Huu就是V,bu就是v)
//step5:遍历所有的关键帧中的帧（因为Wcu.t * Huu^-1存在滑窗中的观测中）,需要更新增量的地图点的
// du =-Wcu.t * Huu^-1 * dxc +  Huu^-1*bu (Huu就是V,bu就是v) = Huu^-1*(bu -Wcu.t * dxc )也就求出了地图点逆深度的增量
//step6:将所有所有需要反求增量的du push进m_xsGN
void LocalBundleAdjustor::SolveBackSubstitution()
{
  const int nKFs = static_cast<int>(m_KFs.size());
  for (int iKF = 0; iKF < nKFs; ++iKF)//遍历所有关键帧中的新地图点,将需要更新的关键帧以及地图点设置成需要反求的flags
  {
    if (!(m_ucsKF[iKF] & LBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION))
    {
      continue;
    }
    const KeyFrame &KF = m_KFs[iKF];
    const int Nx = static_cast<int>(KF.m_xs.size());
    if (Nx == 0)
    {
      continue;
    }
    ubyte *uds = m_uds.data() + m_iKF2d[iKF];
    for (int ix = 0; ix < Nx; ++ix)
    {
      if (!(uds[ix] & LBA_FLAG_TRACK_UPDATE_INFORMATION))
      {
        continue;
      }
      uds[ix] |= LBA_FLAG_TRACK_UPDATE_BACK_SUBSTITUTION;
      m_ucsKF[iKF] |= LBA_FLAG_FRAME_UPDATE_BACK_SUBSTITUTION;
    }
  }
  const int Nc = static_cast<int>(m_LFs.size());
  for (int ic = 0; ic < Nc; ++ic)
  {
    const int iLF = m_ic2LF[ic];
    if (m_xr2s[ic] <= BA_BACK_SUBSTITUTE_ROTATION &&//增量太小就不进行更新
        m_xp2s[ic] <= BA_BACK_SUBSTITUTE_POSITION)
    {
      m_ucsLF[iLF] &= ~LBA_FLAG_FRAME_UPDATE_DELTA;//将这个pose更新的flags设成0
      continue;
    }
    m_ucsLF[iLF] |= LBA_FLAG_FRAME_UPDATE_DELTA;//如果增量变动符合要求对应位就设成1
    const LocalFrame &LF = m_LFs[iLF];
    const int NZ = int(LF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ)//遍历普通帧的观测,将对应的观测关键帧和观测的地图点设成LBA_FLAG_FRAME_UPDATE_BACK_SUBSTITUTION
    {
      const FRM::Measurement &Z = LF.m_Zs[iZ];
      if (Z.m_iz1 == Z.m_iz2)
      {
        continue;
      }
      m_ucsKF[Z.m_iKF] |= LBA_FLAG_FRAME_UPDATE_BACK_SUBSTITUTION;
      ubyte *uds = m_uds.data() + m_iKF2d[Z.m_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz)
      {
        uds[LF.m_zs[iz].m_ix] |= LBA_FLAG_TRACK_UPDATE_BACK_SUBSTITUTION;
      }
    }
  }

  int iX = 0;//构建关键帧和它第一个地图地图点之间的索引
  m_iKF2X.assign(nKFs, -1);
  for (int iKF = 0; iKF < nKFs; ++iKF)
  {
    if (!(m_ucsKF[iKF] & LBA_FLAG_FRAME_UPDATE_BACK_SUBSTITUTION))
    {
      continue;
    }
    m_iKF2X[iKF] = iX;
    iX += static_cast<int>(m_KFs[iKF].m_xs.size());
  }
  m_xds.Resize(iX);

  //遍历所有的地图点,需要更新的话需要求出d_u，d_u = V^-1* (v − W.T * dc )，这里的du = Huu^-1*-bu (Huu就是V,bu就是v)
  for (int iKF = 0; iKF < nKFs; ++iKF)//遍历所有关键帧,设置一下m_xds里的数据
  {
    if (!(m_ucsKF[iKF] & LBA_FLAG_FRAME_UPDATE_BACK_SUBSTITUTION))
    {
      continue;
    }
    const ubyte *uds = m_uds.data() + m_iKF2d[iKF];
    float *xds = m_xds.Data() + m_iKF2X[iKF];
    KeyFrame &KF = m_KFs[iKF];
    const int Nx = static_cast<int>(KF.m_xs.size());
    for (int ix = 0; ix < Nx; ++ix) {
      if (!(uds[ix] & LBA_FLAG_TRACK_UPDATE_BACK_SUBSTITUTION))
      {
        continue;
      } else if (uds[ix] & LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO)//之前没有过观测信息
      {
        xds[ix] = 0.0f;
      } else
      {
        xds[ix] = KF.m_Mxs[ix].BackSubstitute();//如果不是第一次观测的话就保存Huu^-1*-bu
      }
    }
  }
  LA::AlignedVector6f xc;
  const LA::Vector6f *xcs = (LA::Vector6f *) m_xsGN.Data();
  //遍历所有的滑窗中的帧（因为Wcu.t * Huu^-1存在滑窗中的观测中）,需要更新的话需要求出d_u，d_u = V^-1* (v − W.T * dxc )，
  // 这里的du =Wcu.t * Huu^-1 * dxc +  Huu^-1*-bu (Huu就是V,bu就是v)
  for (int ic = 0; ic < Nc; ++ic)//遍历滑窗中所有的帧
  {
    const int iLF = m_ic2LF[ic];
    if (!(m_ucsLF[iLF] & LBA_FLAG_FRAME_UPDATE_DELTA))
    {
      continue;
    }
    xc.Set(xcs[ic]);//如果需要更新增量的话,就遍历所有观测的地图点,
    const LocalFrame &LF = m_LFs[iLF];
    const int NZ = int(LF.m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ)
    {
      const FRM::Measurement &Z = LF.m_Zs[iZ];
      const ubyte *uds = m_uds.data() + m_iKF2d[Z.m_iKF];
      float *xds = m_xds.Data() + m_iKF2X[Z.m_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz)
      {
        const int ix = LF.m_zs[iz].m_ix;
        if (!(uds[ix] & LBA_FLAG_TRACK_UPDATE_INFORMATION_ZERO))
        {
          xds[ix] = LF.m_Mzs1[iz].BackSubstitute(xc) + xds[ix];//W_iLF_iz.t * H_iz_iz^-1 * dxc +  Huu^-1*-bu
        }
      }
    }
  }
    //遍历所有的关键帧中的帧（因为Wcu.t * Huu^-1存在滑窗中的观测中）,需要更新的话需要求出d_u，d_u = V^-1* (v − W.T * dxc )，
    // 这里的du =-Wcu.t * Huu^-1 * dxc +  Huu^-1*bu (Huu就是V,bu就是v) = Huu^-1*(bu -Wcu.t * dxc )也就求出了地图点逆深度的增量
  for (int iKF = 0; iKF < nKFs; ++iKF)
  {
    const int iX = m_iKF2X[iKF];
    if (iX == -1)
    {
      continue;
    }
    const int id = m_iKF2d[iKF];
    const Depth::InverseGaussian *ds = m_ds.data() + id;
    ubyte *uds = m_uds.data() + id;
    float *xds = m_xds.Data() + iX;
    const int Nx = static_cast<int>(m_KFs[iKF].m_xs.size());
    for (int ix = 0; ix < Nx; ++ix)
    {
      if (!(uds[ix] & LBA_FLAG_TRACK_UPDATE_BACK_SUBSTITUTION))
      {
        continue;
      }
      xds[ix] = -xds[ix];
      //if (Depth::InverseGaussian::Valid(xds[ix] + ds[ix].u())) {
      //  continue;
      //}
      //xds[ix] = 0.0f;
      //uds[ix] &= ~LBA_FLAG_TRACK_UPDATE_BACK_SUBSTITUTION;
    }
  }
  //遍历关键帧中的地图点以及所有子轨迹,将更新轨迹新的flags全设为0
  for (int iKF = 0; iKF < nKFs; ++iKF)
  {
    if (!(m_ucsKF[iKF] & LBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION))
    {
      continue;
    }
    m_ucsKF[iKF] &= ~LBA_FLAG_FRAME_UPDATE_TRACK_INFORMATION;//如果更新过了这位就归0
    ubyte *uds = m_uds.data() + m_iKF2d[iKF];
    KeyFrame &KF = m_KFs[iKF];
    const int Nx = static_cast<int>(KF.m_xs.size());
    for (int ix = 0; ix < Nx; ++ix)
    {
      uds[ix] &= ~LBA_FLAG_TRACK_UPDATE_INFORMATION;
    }
    const int NST = static_cast<int>(KF.m_STs.size());
    for (int iST = 0; iST < NST; ++iST)
    {
      KF.m_usST[iST] &= ~LBA_FLAG_TRACK_UPDATE_INFORMATION;
    }
  }
#ifdef LBA_DEBUG_GROUND_TRUTH_STATE
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
  PushDepthUpdates(m_xds, &m_xsGN);//
  m_x2GN = m_xsGN.SquaredLength();
//#ifdef CFG_DEBUG
#if 0
  UT::DebugStart();
  m_work.Set(m_xsGN);
  std::sort(m_work.Data(), m_work.End());
  UT::DebugStop();
#endif
}

#ifdef CFG_GROUND_TRUTH
void LocalBundleAdjustor::SolveBackSubstitutionGT(const std::vector<Depth::InverseGaussian> &ds,
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
#endif

bool LocalBundleAdjustor::EmbeddedMotionIteration() {
  const int pc = 6, pm = 9;
  const int Nc = m_CsLF.Size();
  const LA::Vector6f *xcs = (LA::Vector6f *) m_xsGN.Data();
  LA::Vector9f *xms = (LA::Vector9f *) (xcs + Nc);
  //const float eps = 0.0f;
  const float eps = FLT_EPSILON;
#if 0
//#if 1
  AlignedVector<LA::AlignedMatrix9x9f> Amus, Ambs;
  AlignedVector<LA::AlignedVector9f> bms;
  m_work.Resize((Amus.BindSize(Nc) + Ambs.BindSize(Nc - 1) + bms.BindSize(Nc)) / sizeof(float));
  Amus.Bind(m_work.Data(), Nc);
  Ambs.Bind(Amus.BindNext(), Nc - 1);
  bms.Bind(Ambs.BindNext(), Nc);

  LA::AlignedVector6f xc[2];
  LA::AlignedMatrix9x6f Amc;
  for (int ic = 0, r = 0; ic < Nc; ++ic, r = 1 - r) {
    const Camera::Factor &A = m_SAcmsLF[m_ic2LF[ic]];
    LA::AlignedVector9f &bm = bms[ic];
    A.m_Au.m_Amm.Get(&Amus[ic], &bm);
    xc[r].Set(xcs[ic]);
    A.m_Au.m_Acm.GetTranspose(Amc);
    LA::AlignedMatrix9x6f::AddAbTo(Amc, xc[r], bm);
    if (ic == 0) {
      continue;
    }
    const int _ic = ic - 1;
    Ambs[_ic] = A.m_Ab.m_Amm;
    A.m_Ab.m_Acm.GetTranspose(Amc);
    LA::AlignedMatrix9x6f::AddAbTo(Amc, xc[1 - r], bm);
    LA::AlignedMatrix9x6f::AddAbTo(A.m_Ab.m_Amc, xc[r], bms[_ic]);
  }

  LA::AlignedMatrix9x9f Am21, Mm21;
  LA::AlignedVector9f bm1;
  for (int ic1 = 0, ic2 = 1; ic1 < Nc; ic1 = ic2++) {
    LA::AlignedMatrix9x9f &Mm11 = Amus[ic1];
    if (!Mm11.InverseLDL(eps)) {
      //return false;
      Mm11.MakeZero();
      if (ic2 < Nc) {
        Ambs[ic1].MakeZero();
      }
      bms[ic1].MakeZero();
      continue;
    }
    bm1 = bms[ic1];
    LA::AlignedMatrix9x9f::Ab(Mm11, bm1, bms[ic1]);
    if (ic2 == Nc) {
      break;
    }
    Ambs[ic1].GetTranspose(Am21);
    LA::AlignedMatrix9x9f::ABT(Am21, Mm11, Mm21);
    Mm21.GetTranspose(Ambs[ic1]);
    LA::AlignedMatrix9x9f::SubtractABTFromUpper(Mm21, Am21, Amus[ic2]);
    LA::AlignedMatrix9x9f::SubtractAbFrom(Mm21, bm1, bms[ic2]);
  }
  for (int ic1 = Nc - 2, ic2 = ic1 + 1; ic1 >= 0; ic2 = ic1--) {
    LA::AlignedMatrix9x9f::SubtractAbFrom(Ambs[ic1], bms[ic2], bms[ic1]);
  }
  for (int ic = 0; ic < Nc; ++ic) {
    xms[ic].Set(bms[ic]);
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
  for (int ic1 = -1, ic2 = 0, imp1 = -pm, imp2 = 0, r = 0; ic2 < Nc;
       ic1 = ic2++, imp1 = imp2, imp2 += pm, r = 1 - r) {
    const Camera::Factor &_A = m_SAcmsLF[m_ic2LF[ic2]];
    Amm.Set(_A.m_Au.m_Amm.m_A);
    A.SetBlock(imp2, 0, Amm);
    float *_b = b.Data() + imp2;
    _A.m_Au.m_Amm.m_b.Get(_b);
    xc[r].Set(xcs[ic2]);
    _A.m_Au.m_Acm.GetTranspose(Amc);
    LA::AlignedMatrix9x6f::AddAbTo(Amc, xc[r], _b);
    if (ic2 == 0) {
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
  for (int ic = 0, imp = 0; ic < Nc; ++ic) {
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
      if (ic == Nc - 1) {
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

  for (int ic = 0, imp = 0; ic < Nc; ++ic, imp += pm) {
    memcpy(x[ic], b.Data() + imp, 36);
  }
  for (int ic = Nc - 1, imp = Nmr - 1, r = ic & 1; ic >= 0; --ic, r = 1 - r) {
    const int _ic = ic + 1;
    const int _Nmc = _ic == Nc ? pm : Nmc;
    float *xi = x[ic];
    if (_ic < Nc) {
      memcpy(xi + pm, x[_ic], 36);
    }
    for (int ip = pm - 1; ip >= 0; --ip, --imp) {
      xi[ip] -= SIMD::Dot(ip + 1, _Nmc, A[imp], xi);
    }
  }
  for (int ic = 0; ic < Nc; ++ic) {
    xms[ic].Set(x[ic]);
  }
#endif
  m_xsGN.MakeMinus(pc * Nc);
  ConvertMotionUpdates((float *) xms, &m_xv2s, &m_xba2s, &m_xbw2s);
  return true;
}


void LocalBundleAdjustor::EmbeddedPointIteration(const AlignedVector<Camera> &CsLF,
                                                 const AlignedVector<Rigid3D> &CsKF,
                                                 const std::vector<ubyte> &ucsKF,
                                                 const std::vector<ubyte> &uds,
                                                 std::vector<Depth::InverseGaussian> *ds) {
                                                   
//#ifdef CFG_DEBUG
#if 0
  {
    const int iKF = 37;
    const int ix = 21;
    LA::Vector3f Rx;
    LA::SymmetricMatrix2x2f W;
    const KeyFrame &KF = m_KFs[iKF];
    const FTR::Source &x = KF.m_xs[ix];
    m_zds.resize(0);
#ifdef CFG_STEREO
    if (x.m_xr.Valid()) {
      Rx.Set(x.m_x.x(), x.m_x.y(), 1.0f);
      //const float gyr = 1.0f;
      const float gyr = KF.m_Ards[ix].m_wx * 1.0e-2f / BA_WEIGHT_FEATURE;
      UT::Print("[%d] %e\n", KF.m_Cam_pose.m_iFrm, gyr);
      x.m_Wr.GetScaled(gyr, W);
      m_zds.push_back(Depth::Measurement(m_K.m_br, Rx, x.m_xr, W));
    }
#endif
    const int nKFs = static_cast<int>(m_KFs.size()), nLFs = static_cast<int>(m_LFs.size());
    const int Nc = nKFs + nLFs;
#ifdef CFG_STEREO
    m_t12s.Resize(Nc << 1);
#else
    m_t12s.Resize(Nc);
#endif
    for (int ic = 0; ic < Nc; ++ic) {
      const FRM::Frame *F = ic < nKFs ? (FRM::Frame *) &m_KFs[ic] : &m_LFs[ic - nKFs];
      const KeyFrame *KF = ic < nKFs ? (KeyFrame *) F : NULL;
      const LocalFrame *LF = KF ? NULL : (LocalFrame *) F;
      const int iz = F->SearchFeatureMeasurement(iKF, ix);
      if (iz == -1) {
        continue;
      }
      const Rigid3D T = (ic < nKFs ? m_CsKF[ic] : m_CsLF[ic - nKFs].m_Cam_pose) / m_CsKF[iKF];
      T.ApplyRotation(x.m_x, Rx);
#ifdef CFG_STEREO
      LA::AlignedVector3f *t = m_t12s.Data() + (ic << 1);
#else
      LA::AlignedVector3f *t = m_t12s.Data() + ic;
#endif
      T.GetTranslation(*t);
      const FTR::Measurement &z = F->m_zs[iz];
#ifdef CFG_STEREO
      if (z.m_z.Valid()) {
        //const float gyr = 1.0f;
        const float gyr = (KF ? KF->m_Azs[iz].m_wx * 1.0e-2f : LF->m_Lzs[iz].m_wx) / BA_WEIGHT_FEATURE;
        UT::Print("[%d] %e\n", F->m_Cam_pose.m_iFrm, gyr);
        z.m_W.GetScaled(gyr, W);
        m_zds.push_back(Depth::Measurement(t[0], Rx, z.m_z, W));
      }
      if (z.m_zr.Valid()) {
        LA::AlignedVector3f::apb(t[0], m_K.m_br, t[1]);
        //const float gyr = 1.0f;
        const float gyr = (KF ? KF->m_Azs[iz].m_wxr * 1.0e-2f : LF->m_Lzs[iz].m_wxr) / BA_WEIGHT_FEATURE;
        UT::Print("[%d] %e\n", F->m_Cam_pose.m_iFrm, gyr);
        z.m_Wr.GetScaled(gyr, W);
        m_zds.push_back(Depth::Measurement(t[1], Rx, z.m_zr, W));
      }
#else
      z.m_W.GetScaled(KF ? KF->m_Azs[iz].m_wx : LF->m_Lzs[iz].m_wx, W);
      m_zds.push_back(Depth::Measurement(*t, Rx, z.m_z, W));
#endif
    }
    Depth::InverseGaussian d = m_ds[m_iKF2d[iKF] + ix];
    Depth::Triangulate(1.0f, static_cast<int>(m_zds.size()), m_zds.data(), &d, &m_work, true);
  }
#endif
  std::vector<int> &iKF2X = m_idxsTmp1, &iX2d = m_idxsTmp2;
  const int nKFs = static_cast<int>(m_KFs.size());
  iKF2X.assign(nKFs, -1);
  iX2d.resize(0);

  int Nd = 0;
  for (int iKF = 0; iKF < nKFs; ++iKF) {
    if (!(ucsKF[iKF] & LBA_FLAG_FRAME_UPDATE_DEPTH)) {
      continue;
    }
    const ubyte *_uds = uds.data() + m_iKF2d[iKF];
    const int iX = static_cast<int>(iX2d.size()), Nx = static_cast<int>(m_KFs[iKF].m_xs.size());
    iKF2X[iKF] = iX;
    iX2d.resize(iX + Nx, -1);
    int *ix2d = iX2d.data() + iX;
    for (int ix = 0; ix < Nx; ++ix) {
      if (_uds[ix] & LBA_FLAG_TRACK_UPDATE_DEPTH) {
        ix2d[ix] = Nd++;
      }
    }
  }

  int Nt = 0;
  m_idxsTmp3.resize(Nd + Nd + 1);
  int *Nzs = m_idxsTmp3.data(), *id2z = Nzs + Nd;
  memset(Nzs, 0, sizeof(int) * Nd);
  const int Nc = nKFs + static_cast<int>(m_LFs.size());
  for (int ic = 0; ic < Nc; ++ic) {
    const FRM::Frame *F = ic < nKFs ? (FRM::Frame *) &m_KFs[ic] : &m_LFs[ic - nKFs];
#ifdef CFG_STEREO
    if (ic < nKFs) {
      const int iX = iKF2X[ic];
      if (iX != -1) {
        const KeyFrame *KF = (KeyFrame *) F;
        const int *ix2d = iX2d.data() + iX;
        const int Nx = static_cast<int>(KF->m_xs.size());
        for (int ix = 0; ix < Nx; ++ix) {
          const int id = ix2d[ix];
          if (id != -1 && KF->m_xs[ix].m_xr.Valid()) {
            ++Nzs[id];
          }
        }
      }
    }
#endif
    const int NZ = static_cast<int>(F->m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ) {
      const FRM::Measurement &Z = F->m_Zs[iZ];
      const int iX = iKF2X[Z.m_iKF];
      if (iX == -1) {
        continue;
      }
      bool t = false;
      const int *ix2d = iX2d.data() + iX;
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
        const FTR::Measurement &z = F->m_zs[iz];
        const int id = ix2d[z.m_ix];
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
  for (int id = 0; id < Nd; ++id) {
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
  for (int ic = 0; ic < Nc; ++ic) {
    const FRM::Frame *F = ic < nKFs ? (FRM::Frame *) &m_KFs[ic] : &m_LFs[ic - nKFs];
    const KeyFrame *KF = ic < nKFs ? (KeyFrame *) F : NULL;
    const LocalFrame *LF = KF ? NULL : (LocalFrame *) F;
#ifdef CFG_STEREO
    if (KF) {
      const int iX = iKF2X[ic];
      if (iX != -1) {
        const int *ix2d = iX2d.data() + iX;
        const int Nx = static_cast<int>(KF->m_xs.size());
        for (int ix = 0; ix < Nx; ++ix) {
          const int id = ix2d[ix];
          const FTR::Source &x = KF->m_xs[ix];
          if (id == -1 || x.m_xr.Invalid()) {
            continue;
          }
          Rx.Set(x.m_x.x(), x.m_x.y(), 1.0f);
          //x.m_Wr.GetScaled(KF->m_Ards[ix].m_wx, W);
          const LA::SymmetricMatrix2x2f &W = x.m_Wr;
          const int i = id2z[id] + Nzs[id]++;
          m_zds[i].Set(m_K.m_br, Rx, x.m_xr, W);
        }
      }
    }
#endif
    const Rigid3D &C = KF ? CsKF[ic] : CsLF[ic - nKFs].m_Cam_pose;
    const int NZ = static_cast<int>(F->m_Zs.size());
    for (int iZ = 0; iZ < NZ; ++iZ) {
      const FRM::Measurement &Z = F->m_Zs[iZ];
      const int iX = iKF2X[Z.m_iKF];
      if (iX == -1) {
        continue;
      }
      const int *ix2d = iX2d.data() + iX;
      bool found = false;
      for (int iz = Z.m_iz1; iz < Z.m_iz2 && !found; ++iz) {
        found = ix2d[F->m_zs[iz].m_ix] != -1;
      }
      if (!found) {
        continue;
      }
      const Rigid3D T = C / CsKF[Z.m_iKF];
      LA::AlignedVector3f *t = m_t12s.Data() + Nt++;
      T.GetTranslation(*t);
#ifdef CFG_STEREO
      LA::AlignedVector3f::apb(t[0], m_K.m_br, t[1]);
      ++Nt;
#endif
      const KeyFrame &_KF = m_KFs[Z.m_iKF];
      for (int iz = Z.m_iz1; iz < Z.m_iz2; ++iz) {
        const FTR::Measurement &z = F->m_zs[iz];
        const int id = ix2d[z.m_ix];
        if (id == -1) {
          continue;
        }
        T.ApplyRotation(_KF.m_xs[z.m_ix].m_x, Rx);
#ifdef CFG_STEREO
        if (z.m_z.Valid()) {
          //z.m_W.GetScaled(KF ? KF->m_Azs[iz].m_wx : LF->m_Lzs[iz].m_wx, W);
          const LA::SymmetricMatrix2x2f &W = z.m_W;
          const int i = id2z[id] + Nzs[id]++;
          m_zds[i].Set(t[0], Rx, z.m_z, W);
        }
        if (z.m_zr.Valid()) {
          //z.m_Wr.GetScaled(KF ? KF->m_Azs[iz].m_wxr : LF->m_Lzs[iz].m_wxr, W);
          const LA::SymmetricMatrix2x2f &W = z.m_Wr;
          const int i = id2z[id] + Nzs[id]++;
          m_zds[i].Set(t[1], Rx, z.m_zr, W);
        }
#else
        //z.m_W.GetScaled(KF ? KF->m_Azs[iz].m_wx : LF->m_Lzs[iz].m_wx, W);
        const LA::SymmetricMatrix2x2f &W = z.m_W;
        const int i = ++Nzs[id];
        m_zds[i].Set(*t, Rx, z.m_z, W);
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
    const KeyFrame &KF = m_KFs[iKF];
    const int Nx = static_cast<int>(KF.m_xs.size());
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
