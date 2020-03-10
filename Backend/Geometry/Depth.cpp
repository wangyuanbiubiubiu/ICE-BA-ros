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
#include "Depth.h"
#include "Parameter.h"
#include "VectorN.h"

//#ifdef CFG_DEBUG
#if 1
//#define DEPTH_TRI_VERBOSE  1
//#define DEPTH_TRI_VERBOSE  2
#endif
#define DEPTH_TRI_DOG_LEG

namespace Depth {

bool Triangulate(const float w, const LA::AlignedVector3f &t12, const Point2D &x1,
                 const Point2D &x2, InverseGaussian *d, const LA::SymmetricMatrix2x2f *Wx2) {
  float x;
  LA::Vector2f J, e, WJ;
  d->Initialize();
#if 0
  if (Wx2) {
    if (!Triangulate(t12, x1, x2, d)) {
      return false;
    }
  }
#endif
#ifdef DEPTH_TRI_VERBOSE
  const float f = 500.0f;
#if DEPTH_TRI_VERBOSE == 1
  UT::Print("e = %f", f * sqrtf((d->GetProjected(t12, x1) - x2).SquaredLength()));
#endif
#endif
#ifdef DEPTH_TRI_DOG_LEG
  float delta = DEPTH_TRI_DL_RADIUS_INITIAL;
#endif
  const float eps = 0.0f;
  for (int iIter = 0; iIter < DEPTH_TRI_MAX_ITERATIONS; ++iIter) {
    d->Project(t12, x1, e, J);
    e -= x2;
    if (Wx2) {
      LA::SymmetricMatrix2x2f::Ab(*Wx2, J, WJ);
      d->s2() = WJ.Dot(J);
    } else {
      d->s2() = J.Dot(J);
    }
    if (d->s2() <= eps) {
      return false;
    }
    d->s2() = 1.0f / d->s2();
    if (Wx2) {
      x = -d->s2() * WJ.Dot(e);
    } else {
      x = -d->s2() * J.Dot(e);
    }
#ifdef DEPTH_TRI_DOG_LEG
    const float xGN = x;
    const float F = Wx2 ? LA::SymmetricMatrix2x2f::MahalanobisDistance(*Wx2, e) :
                          e.SquaredLength();
    const float dBkp = d->u();
    bool update = true, converge = false;
    for (int iIterDL = 0; iIterDL < DEPTH_TRI_DL_MAX_ITERATIONS; ++iIterDL) {
      if (fabs(xGN) <= delta) {
        x = xGN;
      } else {
        x = x > 0.0f ? delta : -delta;
      }
      d->u() = x + d->u();
      const LA::Vector2f ea = d->GetProjected(t12, x1) - x2;
      const LA::Vector2f ep = e + J * x;
      const float dFa = F - (Wx2 ? LA::SymmetricMatrix2x2f::MahalanobisDistance(*Wx2, ea) :
                             ea.SquaredLength());
      const float dFp = F - (Wx2 ? LA::SymmetricMatrix2x2f::MahalanobisDistance(*Wx2, ep) :
                             ea.SquaredLength());
      const float rho = dFa > 0.0f && dFp > 0.0f ? dFa / dFp : -1.0f;
      if (rho < DEPTH_TRI_DL_GAIN_RATIO_MIN) {
        delta *= DEPTH_TRI_DL_RADIUS_FACTOR_DECREASE;
        if (delta < DEPTH_TRI_DL_RADIUS_MIN) {
          delta = DEPTH_TRI_DL_RADIUS_MIN;
        }
        d->u() = dBkp;
        update = false;
        converge = false;
        continue;
      } else if (rho > DEPTH_TRI_DL_GAIN_RATIO_MAX) {
        delta = std::max(delta, DEPTH_TRI_DL_RADIUS_FACTOR_INCREASE * static_cast<float>(fabs(x)));
        if (delta > DEPTH_TRI_DL_RADIUS_MAX) {
          delta = DEPTH_TRI_DL_RADIUS_MAX;
        }
      }
      update = true;
      converge = fabs(x) < DEPTH_TRI_CONVERGE;
      break;
    }
    if (!update || converge) {
      break;
    }
#else
    d->u() = x + d->u();
    if (fabs(x) < DEPTH_TRI_CONVERGE) {
      break;
    }
#endif
#if defined DEPTH_TRI_VERBOSE && DEPTH_TRI_VERBOSE == 2
    const std::string str = UT::String("%02d  ", iIter);
    if (iIter == 0) {
      UT::PrintSeparator();
      UT::Print("%se = %f\n", std::string(str.size(), ' ').c_str(),
                f * sqrtf((InverseGaussian(0.0f).GetProjected(t12, x1) - x2).SquaredLength()));
    }
    UT::Print("%se = %f  z = %f  x = %f\n", str.c_str(),
              f * sqrtf((d->GetProjected(t12, x1) - x2).SquaredLength()), 1.0f / d->u(), x);
#endif
  }
#if defined DEPTH_TRI_VERBOSE && DEPTH_TRI_VERBOSE == 1
  UT::Print(" --> %f  z = %f  s = %f", f * sqrtf((d->GetProjected(t12, x1) - x2).SquaredLength()),
            1.0f / d->u(), sqrtf(d->s2()));
  if (!d->Valid()) {
    UT::Print("  FAIL");
  }
  UT::Print("\n");
#endif
  d->s2() = DEPTH_VARIANCE_EPSILON + d->s2() * w;
  return d->Valid();
}

float ComputeError(const int N, const Measurement *zs, const InverseGaussian &d) {
  LA::Vector2f ei;
  float Se2 = 0.0f;
  for (int i = 0; i < N; ++i) {
    const Measurement &z = zs[i];
    d.Project(*z.m_t, z.m_Rx, ei);
    ei -= z.m_z;
    Se2 += ei.SquaredLength();
  }
  return sqrtf(Se2 / N);
}
//三角化,通过r(Uc0) = 归一化(Pnc0 - Uc0 * tc0c1) - 归一化(Rc0c1 *Pnc1),min F(Uc0) = 0.5*||r(Uc0)||^2（马氏距离下）求解左目特征点的深度
bool Triangulate(const float w/*1*/, const int N/*0 or 1*/, const Measurement *zs/*地图点的左右目观测*/, InverseGaussian *d/*特征点对应的逆深度*/,
                 AlignedVector<float> *work/*J部分*/, const bool initialized,
                 float *eAvg/*平均误差*/) {
  if (N == 0) {
    return false;
  }
  if (!initialized) {

    d->Initialize();//初始化深度,初始深度为5m
  }
  float a, b, x;
#ifdef DEPTH_TRI_DOG_LEG
  float F, Fa, Fp;
  const int Nx2 = N + N;
  LA::AlignedVectorXf J, e, ep/*理论下降值*/, _w;
  work->Resize((J.BindSize(Nx2) * 3 + (DEPTH_TRI_ROBUST ? _w.BindSize(N) : 0)) / sizeof(float));//扩容
  J.Bind(work->Data(), Nx2);
  e.Bind(J.BindNext(), Nx2);
  ep.Bind(e.BindNext(), Nx2);
  if (DEPTH_TRI_ROBUST) {
    _w.Bind(ep.BindNext(), N);
  }
  AlignedVector<LA::Vector2f> Jis(J.Data(), N, false);
  AlignedVector<LA::Vector2f> eis(e.Data(), N, false);
  AlignedVector<LA::Vector2f> epis(ep.Data(), N, false);
#else
  LA::Vector2f Ji, ei;
#endif
  LA::Vector2f WJi;
#ifdef DEPTH_TRI_VERBOSE
  const float f = 500.0f;
#if DEPTH_TRI_VERBOSE == 1
  UT::Print("e = %f", f * ComputeError(N, feat_measures, *d));
#else if DEPTH_TRI_VERBOSE == 2
  const InverseGaussian d0 = *d;
#endif
#endif
#ifdef DEPTH_TRI_DOG_LEG
  float delta = DEPTH_TRI_DL_RADIUS_INITIAL;
#endif
  const float eps = 0.0f;
  for (int iIter = 0; iIter < DEPTH_TRI_MAX_ITERATIONS; ++iIter) {
    a = b = 0.0f;
    for (int i = 0; i < N; ++i) {
      const Measurement &z = zs[i];/*地图点的左右目观测*/
#ifdef DEPTH_TRI_DOG_LEG
      LA::Vector2f &Ji = Jis[i], &ei = eis[i];
#endif
        // 投影 Pc0 Pc1代表相机坐标下的特征点 Pnc0 和 Pnc1代表了归一化以后的点,齐次转换就省略了 Uc0,Uc1表示两个坐标系下的逆深度
        // Pc0 = Rc0c1 * Pc1 + tc0c1 ==>>  (1/Uc0) * Pnc0 = Rc0c1 * (1/Uc1) * Pnc1 + tc0c1
        // ==>> Rc0c1 *Pnc1* (Uc0/Uc1) = Pnc0 - Uc0 * tc0c1
        // ==> 对Pnc0 - Uc0 * tc0c1进行归一化,因为之前对Pnc1也做了（Rc0c1 *Pnc1)归一化坐标,存在z.m_z里
      d->Project(*z.m_t/*-tc0_c1*/, z.m_Rx/* 左目的无畸变归一化坐标*/, ei, Ji/*残差*/);
      ei -= z.m_z;// r(Uc0) = 归一化(Pnc0 - Uc0 * tc0c1) - 归一化(Rc0c1 *Pnc1),min F(Uc0) = ||r(Uc0)||^2（马氏距离下）
      LA::SymmetricMatrix2x2f::Ab(z.m_W, Ji, WJi);//WJi = z.m_W *Ji
      if (DEPTH_TRI_ROBUST) {//huber核函数
        const float r2 = LA::SymmetricMatrix2x2f::MahalanobisDistance(z.m_W, ei);
        const float wi = ME::Weight<ME::FUNCTION_HUBER>(r2);
        WJi *= wi;
#ifdef DEPTH_TRI_DOG_LEG
        _w[i] = wi;
#endif
#if 0
//#if 1
        UT::Print("%d %f\n", i, wi);
#endif
      }
      a += WJi.Dot(Ji);//WJi.t *Ji 更新H矩阵(WJi的作为是马氏距离归一化),优化变量是逆深度,所以是1×1
      b += WJi.Dot(ei);//WJi.t * r 更新-b
#if 0
//#if 1
      UT::Print("%d %f %f\n", i, acc, b);
#endif
    }
    if (a <= eps) {
      return false;
    }
    d->s2() = 1.0f / a;//H^-1,协方差更新
    x = -d->s2() * b;// x = H^-1*-(-b) = H^-1*b 求出G-N法的增量
#ifdef DEPTH_TRI_DOG_LEG
    const float xGN = x;
    F = 0.0f;
    for (int i = 0; i < N; ++i) {
      F += LA::SymmetricMatrix2x2f::MahalanobisDistance(zs[i].m_W, eis[i]);//归一化一下(马氏距离)
    }
    const float dBkp = d->u();//优化变量的备份,用来回滚
    bool update = true, converge = false;
    for (int iIterDL = 0; iIterDL < DEPTH_TRI_DL_MAX_ITERATIONS; ++iIterDL) {//这里也没用dogleg啊,用的还是GN
      if (fabs(xGN) <= delta) {//信赖域内,那么就是一个无约束问题
        x = xGN;
      } else {
        x = x > 0.0f ? delta : -delta;
      }
      d->u() = x + d->u();//加上增量

      J.GetScaled(x, ep);//Jx = ep
      ep += e;
      Fa /*实际下降*/= Fp/*理论下降*/ = 0.0f;
      for (int i = 0; i < N; ++i) {
        const Measurement &z = zs[i];/*地图点的左右目观测*/
        LA::Vector2f &ei = epis[i];//Jx
        const float Fpi = LA::SymmetricMatrix2x2f::MahalanobisDistance(z.m_W, ei);//理论下降J*deltax再归一化一下
        d->Project(*z.m_t, z.m_Rx, ei);
        ei -= z.m_z;
        const float Fai = LA::SymmetricMatrix2x2f::MahalanobisDistance(z.m_W, ei);//实际下降的
        if (DEPTH_TRI_ROBUST) {
          Fp += Fpi * _w[i];
          Fa += Fai * _w[i];
        } else {
          Fp += Fpi;
          Fa += Fai;
        }
      }
      const float dFa = F - Fa, dFp = F - Fp;
      const float rho = dFa > 0.0f && dFp > 0.0f ? dFa / dFp : -1.0f;//实际下降/理论下降
      if (rho < DEPTH_TRI_DL_GAIN_RATIO_MIN) {//<0.25,如果大于0说明近似的不好,需要减小信赖域,减小近似的范围。如果<0就说明是错误的近似,那么就拒绝这次的增量
        delta *= DEPTH_TRI_DL_RADIUS_FACTOR_DECREASE;
        if (delta < DEPTH_TRI_DL_RADIUS_MIN) {
          delta = DEPTH_TRI_DL_RADIUS_MIN;
        }
        d->u() = dBkp;//放弃更新,回滚
        update = false;
        converge = false;
        continue;
      } else if (rho > DEPTH_TRI_DL_GAIN_RATIO_MAX) {//rho > 0.75,可以扩大信赖域半径
        delta = std::max(delta, DEPTH_TRI_DL_RADIUS_FACTOR_INCREASE * static_cast<float>(fabs(x)));
        if (delta > DEPTH_TRI_DL_RADIUS_MAX) {//信赖域上限
          delta = DEPTH_TRI_DL_RADIUS_MAX;
        }
      }
      update = true;
      converge = fabs(x) < DEPTH_TRI_CONVERGE;
      break;
    }
    if (!update || converge) {
      break;
    }
#else
    d->u() = x + d->u();
    if (fabs(x) < DEPTH_TRI_CONVERGE) {
      break;
    }
#endif
#if defined DEPTH_TRI_VERBOSE && DEPTH_TRI_VERBOSE == 2
    const std::string str = UT::String("%02d  ", iIter);
    if (iIter == 0) {
      UT::PrintSeparator();
      UT::Print("%se = %f  z = %f\n", std::string(str.size(), ' ').c_str(),
                f * ComputeError(N, feat_measures, d0), 1.0f / d0.u());
    }
    UT::Print("%se = %f  z = %f  x = %f\n", str.c_str(), f * ComputeError(N, feat_measures, *d),
              1.0f / d->u(), x);
#endif
  }
#if defined DEPTH_TRI_VERBOSE && DEPTH_TRI_VERBOSE == 1
  UT::Print(" --> %f  z = %f  s = %f", f * ComputeError(N, feat_measures, *d), 1.0f / d->u(), sqrtf(d->s2()));
  if (!d->Valid()) {
    UT::Print("  FAIL");
  }
  UT::Print("\n");
#endif
  d->s2() = DEPTH_VARIANCE_EPSILON + d->s2() * w;
  if (eAvg) {
    *eAvg = ComputeError(N, zs, *d);//计算平均误差,这次不是马氏距离了,是欧式距离
  }
  return d->Valid();
}

////三角化,通过r(Uc0) = 归一化(Pnc0 - Uc0 * tc0c1) - 归一化(Rc0c1 *Pnc1),min F(Uc0) = 0.5*||r(Uc0)||^2（马氏距离下）求解左目特征点的深度
bool Triangulateinit(const float w/*1*/, const int N/*0 or 1*/, const Measurement *zs/*地图点的左右目观测*/, InverseGaussian *d/*特征点对应的逆深度*/,
                     AlignedVector<float> *work/*J部分*/,const Rotation3D & Rclcr,const Point3D & tclcr_n, const bool initialized,
                     float *eAvg/*平均误差*/ )//-tlr
{
        if (N == 0) {
            return false;
        }
        if (!initialized)
        {

            LA::AlignedVector3f pr,f_temp;//R_r_l * f_l
            pr.Set(zs[0].m_z[0],zs[0].m_z[1],1.0f);//(Rc0c1 *Pnc1)归一化坐标


            pr = Rclcr.GetATb(Rclcr,zs[0].m_z/*(Rc0c1 *Pnc1)归一化坐标*/);//转回r系下,右目观测
            Eigen::Vector3f f_r{pr.x()/pr.z(),pr.y()/pr.z(),1.0f};//右目系下的归一化坐标
            Point2D R_x;
            R_x.Set(zs[0].m_Rx.x(),zs[0].m_Rx.y());//左目系下的归一化坐标
            f_temp = Rclcr.GetATb(Rclcr ,R_x);//R_r_l * f_l
            float depth;
            LA::AlignedVector3f t_rl = Rclcr.GetATb(Rclcr ,tclcr_n);
            Eigen::Vector3f t_rl_eigen{t_rl.x(),t_rl.y(),t_rl.z()};
            Eigen::Vector3f f_temp_eigen{f_temp.x(),f_temp.y(),f_temp.z()};
            Eigen::Matrix<float, 3, 2> A;
            A << f_temp_eigen, f_r;
            const Eigen::Matrix2f AtA = A.transpose() * A;
            if (AtA.determinant() < 1e-6) {
                // TODO(mingyu): figure the right threshold for float
                depth = 1000;  // acc very far point
                return false;
            }
            const Eigen::Vector2f depth2 = - AtA.inverse() * A.transpose() * t_rl_eigen;

            depth = fabs(depth2[0]);//左目深度

//            float depth;
//            Eigen::Matrix3f Rrl_eigen;
//            Eigen::Vector3f trl_eigen, f_l_eigen, f_r_eigen;
//            Eigen::Matrix4f Trl_eigen = Eigen::Matrix4f::Identity();
//
//            float Rclcr_f[3][3];
//            Rclcr.Get(Rclcr_f);
//            for (int i = 0; i < 3; ++i)
//                for (int j = 0; j < 3; ++j)
//                    Rrl_eigen(i,j) = Rclcr_f[j][i];
//
//            LA::AlignedVector3f t_rl = Rclcr.GetATb(Rclcr ,tclcr_n);
//            trl_eigen = Eigen::Vector3f{t_rl.x(),t_rl.y(),t_rl.z()};
//
//            Trl_eigen.block<3,3>(0,0) = Rrl_eigen;
//            Trl_eigen.block<3,1>(0,3) = trl_eigen;
//
//            LA::AlignedVector3f pr;//R_r_l * f_l
//            pr.Set(zs[0].m_z[0],zs[0].m_z[1],1.0f);//(Rc0c1 *Pnc1)归一化坐标
//            pr = Rclcr.GetATb(Rclcr,zs[0].m_z/*(Rc0c1 *Pnc1)归一化坐标*/);//转回r系下,右目观测
//            f_r_eigen = Eigen::Vector3f{pr.x()/pr.z(),pr.y()/pr.z(),1.0f};//右目系下的归一化坐标
//            f_l_eigen = Eigen::Vector3f{zs[0].m_Rx.x(),zs[0].m_Rx.y(),1.0f};//左目系下的归一化坐标
//
//            if(0)
//            {
//                Eigen::Matrix<float, 3, 2> A;
//                A << Rrl_eigen * f_l_eigen, f_r_eigen;
//                const Eigen::Matrix2f AtA = A.transpose() * A;
//                if (AtA.determinant() < 1e-6) {
//                    // TODO(mingyu): figure the right threshold for float
//                    depth = 1000;  // acc very far point
//                    return false;
//                }
//
//                const Eigen::Vector2f depth2 = - AtA.inverse() * A.transpose() * trl_eigen;
//
//                depth = fabs(depth2[0]);//左目深度
//            } else
//            {
//                Eigen::Matrix3Xd G_bearing_vectors;
//                Eigen::Matrix3Xd p_G_C_vector;
//                G_bearing_vectors.resize(Eigen::NoChange, 2);
//                p_G_C_vector.resize(Eigen::NoChange, 2);
//                Eigen::Vector3d p_G_fi;
//                Eigen::Matrix4d T_G_left_C = Eigen::Matrix4d::Identity();
//                Eigen::Matrix4d T_G_right_C = T_G_left_C * Trl_eigen.cast<double>().inverse();
//                G_bearing_vectors.col(0) = T_G_left_C.block<3,3>(0,0) * f_l_eigen.cast<double>();
//                p_G_C_vector.col(0) = T_G_left_C.block<3,1>(0,3);
//
//
//                G_bearing_vectors.col(1) = T_G_right_C.block<3,3>(0,0) * f_r_eigen.cast<double>();
//                p_G_C_vector.col(1) = T_G_right_C.block<3,1>(0,3);
//                if(linearTriangulateFromNViews(G_bearing_vectors, p_G_C_vector, p_G_fi))
//                    depth = (float)p_G_fi[2]; //因为现在左相机就是世界坐标系
//                else
//                    depth = 5.0f;
//
//
//            }
            d->Initialize(1.0f/depth);


        }
        float a, b, x;
#ifdef DEPTH_TRI_DOG_LEG
        float F, Fa, Fp;
        const int Nx2 = N + N;
        LA::AlignedVectorXf J, e, ep/*理论下降值*/, _w;
        work->Resize((J.BindSize(Nx2) * 3 + (DEPTH_TRI_ROBUST ? _w.BindSize(N) : 0)) / sizeof(float));//扩容
        J.Bind(work->Data(), Nx2);
        e.Bind(J.BindNext(), Nx2);
        ep.Bind(e.BindNext(), Nx2);
        if (DEPTH_TRI_ROBUST) {
            _w.Bind(ep.BindNext(), N);
        }
        AlignedVector<LA::Vector2f> Jis(J.Data(), N, false);
        AlignedVector<LA::Vector2f> eis(e.Data(), N, false);
        AlignedVector<LA::Vector2f> epis(ep.Data(), N, false);
#else
        LA::Vector2f Ji, ei;
#endif
        LA::Vector2f WJi;
#ifdef DEPTH_TRI_VERBOSE
        const float f = 500.0f;
#if DEPTH_TRI_VERBOSE == 1
  UT::Print("e = %f", f * ComputeError(N, feat_measures, *d));
#else if DEPTH_TRI_VERBOSE == 2
  const InverseGaussian d0 = *d;
#endif
#endif
#ifdef DEPTH_TRI_DOG_LEG
        float delta = DEPTH_TRI_DL_RADIUS_INITIAL;
#endif
        const float eps = 0.0f;
        for (int iIter = 0; iIter < DEPTH_TRI_MAX_ITERATIONS; ++iIter) {
            a = b = 0.0f;
            for (int i = 0; i < N; ++i) {
                const Measurement &z = zs[i];/*地图点的左右目观测*/
#ifdef DEPTH_TRI_DOG_LEG
                LA::Vector2f &Ji = Jis[i], &ei = eis[i];
#endif
                // 投影 Pc0 Pc1代表相机坐标下的特征点 Pnc0 和 Pnc1代表了归一化以后的点,齐次转换就省略了 Uc0,Uc1表示两个坐标系下的逆深度
                // Pc0 = Rc0c1 * Pc1 + tc0c1 ==>>  (1/Uc0) * Pnc0 = Rc0c1 * (1/Uc1) * Pnc1 + tc0c1
                // ==>> Rc0c1 *Pnc1* (Uc0/Uc1) = Pnc0 - Uc0 * tc0c1
                // ==> 对Pnc0 - Uc0 * tc0c1进行归一化,因为之前对Pnc1也做了（Rc0c1 *Pnc1)归一化坐标,存在z.m_z里
                d->Project(*z.m_t/*-tc0_c1*/, z.m_Rx/* 左目的无畸变归一化坐标*/, ei, Ji/*残差*/);
                ei -= z.m_z;// r(Uc0) = 归一化(Pnc0 - Uc0 * tc0c1) - 归一化(Rc0c1 *Pnc1),min F(Uc0) = ||r(Uc0)||^2（马氏距离下）
                LA::SymmetricMatrix2x2f::Ab(z.m_W, Ji, WJi);//WJi = z.m_W *Ji
                if (DEPTH_TRI_ROBUST) {//huber核函数
                    const float r2 = LA::SymmetricMatrix2x2f::MahalanobisDistance(z.m_W, ei);
                    const float wi = ME::Weight<ME::FUNCTION_HUBER>(r2);
                    WJi *= wi;
#ifdef DEPTH_TRI_DOG_LEG
                    _w[i] = wi;
#endif
#if 0
                    //#if 1
        UT::Print("%d %f\n", i, wi);
#endif
                }
                a += WJi.Dot(Ji);//WJi.t *Ji 更新H矩阵(WJi的作为是马氏距离归一化),优化变量是逆深度,所以是1×1
                b += WJi.Dot(ei);//WJi.t * r 更新-b
#if 0
                //#if 1
      UT::Print("%d %f %f\n", i, acc, b);
#endif
            }
            if (a <= eps) {
                return false;
            }
            d->s2() = 1.0f / a;//H^-1,协方差更新
            x = -d->s2() * b;// x = H^-1*-(-b) = H^-1*b 求出G-N法的增量
#ifdef DEPTH_TRI_DOG_LEG
            const float xGN = x;
            F = 0.0f;
            for (int i = 0; i < N; ++i) {
                F += LA::SymmetricMatrix2x2f::MahalanobisDistance(zs[i].m_W, eis[i]);//归一化一下(马氏距离)
            }
            const float dBkp = d->u();//优化变量的备份,用来回滚
            bool update = true, converge = false;
            for (int iIterDL = 0; iIterDL < DEPTH_TRI_DL_MAX_ITERATIONS; ++iIterDL) {//这里也没用dogleg啊,用的还是GN
                if (fabs(xGN) <= delta) {//信赖域内,那么就是一个无约束问题
                    x = xGN;
                } else {
                    x = x > 0.0f ? delta : -delta;
                }
                d->u() = x + d->u();//加上增量

                J.GetScaled(x, ep);//Jx = ep
                ep += e;
                Fa /*实际下降*/= Fp/*理论下降*/ = 0.0f;
                for (int i = 0; i < N; ++i) {
                    const Measurement &z = zs[i];/*地图点的左右目观测*/
                    LA::Vector2f &ei = epis[i];//Jx
                    const float Fpi = LA::SymmetricMatrix2x2f::MahalanobisDistance(z.m_W, ei);//理论下降J*deltax再归一化一下
                    d->Project(*z.m_t, z.m_Rx, ei);
                    ei -= z.m_z;
                    const float Fai = LA::SymmetricMatrix2x2f::MahalanobisDistance(z.m_W, ei);//实际下降的
                    if (DEPTH_TRI_ROBUST) {
                        Fp += Fpi * _w[i];
                        Fa += Fai * _w[i];
                    } else {
                        Fp += Fpi;
                        Fa += Fai;
                    }
                }
                const float dFa = F - Fa, dFp = F - Fp;
                const float rho = dFa > 0.0f && dFp > 0.0f ? dFa / dFp : -1.0f;//实际下降/理论下降
                if (rho < DEPTH_TRI_DL_GAIN_RATIO_MIN) {//<0.25,如果大于0说明近似的不好,需要减小信赖域,减小近似的范围。如果<0就说明是错误的近似,那么就拒绝这次的增量
                    delta *= DEPTH_TRI_DL_RADIUS_FACTOR_DECREASE;
                    if (delta < DEPTH_TRI_DL_RADIUS_MIN) {
                        delta = DEPTH_TRI_DL_RADIUS_MIN;
                    }
                    d->u() = dBkp;//放弃更新,回滚
                    update = false;
                    converge = false;
                    continue;
                } else if (rho > DEPTH_TRI_DL_GAIN_RATIO_MAX) {//rho > 0.75,可以扩大信赖域半径
                    delta = std::max(delta, DEPTH_TRI_DL_RADIUS_FACTOR_INCREASE * static_cast<float>(fabs(x)));
                    if (delta > DEPTH_TRI_DL_RADIUS_MAX) {//信赖域上限
                        delta = DEPTH_TRI_DL_RADIUS_MAX;
                    }
                }
                update = true;
                converge = fabs(x) < DEPTH_TRI_CONVERGE;
                break;
            }
            if (!update || converge) {
                break;
            }
#else
            d->u() = x + d->u();
    if (fabs(x) < DEPTH_TRI_CONVERGE) {
      break;
    }
#endif
#if defined DEPTH_TRI_VERBOSE && DEPTH_TRI_VERBOSE == 2
            const std::string str = UT::String("%02d  ", iIter);
    if (iIter == 0) {
      UT::PrintSeparator();
      UT::Print("%se = %f  z = %f\n", std::string(str.size(), ' ').c_str(),
                f * ComputeError(N, feat_measures, d0), 1.0f / d0.u());
    }
    UT::Print("%se = %f  z = %f  x = %f\n", str.c_str(), f * ComputeError(N, feat_measures, *d),
              1.0f / d->u(), x);
#endif
        }
#if defined DEPTH_TRI_VERBOSE && DEPTH_TRI_VERBOSE == 1
        UT::Print(" --> %f  z = %f  s = %f", f * ComputeError(N, feat_measures, *d), 1.0f / d->u(), sqrtf(d->s2()));
  if (!d->Valid()) {
    UT::Print("  FAIL");
  }
  UT::Print("\n");
#endif
        d->s2() = DEPTH_VARIANCE_EPSILON + d->s2() * w;
        if (eAvg) {
            *eAvg = ComputeError(N, zs, *d);//计算平均误差,这次不是马氏距离了,是欧式距离
        }
        return d->Valid();
    }

    //from maplab
    bool linearTriangulateFromNViews(
            const Eigen::Matrix3Xd& t_G_bv,//world系下的投影射线
            const Eigen::Matrix3Xd &p_G_C,//world系下的pose
            Eigen::Vector3d & p_G_P)//这点在左目坐标
    {

        const int num_measurements = t_G_bv.cols();
        if (num_measurements < 2) {
            return false;
        }

        // 1.) Formulate the geometrical problem
        // p_G_P + alpha[i] * t_G_bv[i] = p_G_C[i]      (+ alpha intended)
        // as linear system Ax = b, where
        // x = [p_G_P; alpha[0]; alpha[1]; ... ] and b = [p_G_C[0]; p_G_C[1]; ...]
        //
        // 2.) Apply the approximation AtAx = Atb
        // AtA happens to be composed of mostly more convenient blocks than A:
        // - Top left = N * Eigen::Matrix3d::Identity()
        // - Top right and bottom left = t_G_bv
        // - Bottom right = t_G_bv.colwise().squaredNorm().asDiagonal()

        // - Atb.head(3) = p_G_C.rowwise().sum()
        // - Atb.tail(N) = columnwise dot products between t_G_bv and p_G_C
        //               = t_G_bv.cwiseProduct(p_G_C).colwise().sum().transpose()
        //
        // 3.) Apply the Schur complement to solve after p_G_P only
        // AtA = [E B; C D] (same blocks as above) ->
        // (E - B * D.inverse() * C) * p_G_P = Atb.head(3) - B * D.inverse() * Atb.tail(N)

        const Eigen::MatrixXd BiD = t_G_bv *
                                    t_G_bv.colwise().squaredNorm().asDiagonal().inverse();
        const Eigen::Matrix3d AxtAx = num_measurements * Eigen::Matrix3d::Identity() -
                                      BiD * t_G_bv.transpose();
        const Eigen::Vector3d Axtbx = p_G_C.rowwise().sum() - BiD *
                                                              t_G_bv.cwiseProduct(p_G_C).colwise().sum().transpose();

        Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr = AxtAx.colPivHouseholderQr();
        static constexpr double kRankLossTolerance = 1e-5;
        qr.setThreshold(kRankLossTolerance);
        const size_t rank = qr.rank();
        if (rank < 3) {
            return false;
        }

        p_G_P = qr.solve(Axtbx);
        return true;
    }


}
