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
#ifndef _DEPTH_H_
#define _DEPTH_H_

#include "Parameter.h"
#include "Intrinsic.h"
#include "Rigid.h"
#include "BoundingBox.h"
#include "M-Estimator.h"
#include "AlignedVector.h"
#include <eigen3/Eigen/Dense>
#include <Eigen/Dense>
#include <eigen3/Eigen/QR>
namespace Depth {

class InverseGaussian {

 public:

  inline InverseGaussian() {}
  inline InverseGaussian(const float u) : m_u(u), m_s2(DEPTH_VARIANCE_INITIAL) {}
  inline InverseGaussian(const float u, const float s2) : m_u(u), m_s2(s2) {}

  inline const float& u () const { return m_u; }    inline float& u () { return m_u; }
  inline const float& s2() const { return m_s2; }   inline float& s2() { return m_s2; }

  inline bool operator == (const InverseGaussian &d) const { return u() == d.u() && s2() == d.s2(); }

  inline void Initialize() { u() = DEPTH_INITIAL; s2() = DEPTH_VARIANCE_INITIAL; }
  inline void Initialize(const float u) { this->u() = u; s2() = DEPTH_VARIANCE_INITIAL; }
  inline void Set(const float u, const float s2) { this->u() = u; this->s2() = s2; }
  inline void Set(const InverseGaussian &d) { u() = d.u(); s2() = d.s2();}
  inline void Propagate(const float dt) { s2() = DEPTH_VARIANCE_WALK * dt * dt + s2(); }
  inline void Update(const InverseGaussian &d) {
    const float Ss2 = s2() + d.s2();
    if (Ss2 < FLT_EPSILON) {
      return;
    }
    const float Ss2I = 1.0f / Ss2;
    u() = (s2() * d.u() + d.s2() * u()) * Ss2I;
    s2() = s2() * d.s2() * Ss2I;
  }
  static inline bool Valid(const float d) { return d > DEPTH_MIN && d < DEPTH_MAX; }
  inline bool Valid() const { return Valid(u()); }
  //inline bool Valid() const { return u() > 0.0f; }
  inline bool Invalid() const { return u() == 0.0f; }
  inline void Invalidate() { u() = 0.0f; }
  inline bool Converge() const { return s2() <= DEPTH_VARIANCE_CONVERGE; }

  inline void Project(const Rigid3D &T12/*这里应该表达成T21更准确一些*/, const Point2D &x1, LA::Vector2f &x2) const {
    Project(T12, x1, u(), x2);
  }
  inline void Project(const LA::AlignedVector3f &t12/*-tc0_c1*/, const Point2D &x1/*Pc0归一化坐标*/, LA::Vector2f &x2/*重投影误差*/) const {
    Project(t12/*-tc0_c1*/, x1/*Pc0归一化坐标*/, u()/*Pc0逆深度*/, x2/*重投影误差*/);
  }
  inline void Project(const LA::AlignedVector3f &t12, const LA::Vector3f &Rx1, LA::Vector2f &x2) const {
    Project(t12, Rx1, u(), x2);
  }

    // 投影 Pc0 Pc1代表相机坐标下的特征点 Pnc0 和 Pnc1代表了归一化以后的点,齐次转换就省略了 Uc0,Uc1表示两个坐标系下的逆深度
    // Pc0 = Rc0c1 * Pc1 + tc0c1 ==>>  (1/Uc0) * Pnc0 = Rc0c1 * (1/Uc1) * Pnc1 + tc0c1
    // ==>> Rc0c1 *Pnc1* (Uc0/Uc1) = Pnc0 - Uc0 * tc0c1
    // ==> 对Pnc0 - Uc0 * tc0c1进行归一化,因为之前对Pnc1也做了（Rc0c1 *Pnc1)归一化坐标,存在z.m_z里
  static inline void Project(const LA::AlignedVector3f &t12/*-tc0_c1*/, const Point2D &x1/*Pc0归一化坐标*/, const float d1/*逆深度*/,
                             LA::Vector2f &x2/*重投影误差*/) {
    const LA::AlignedVector3f t = t12 * d1;
    const float d12 = 1.0f / (t.z() + 1.0f);
    x2.x() = (x1.x() + t.x()) * d12;
    x2.y() = (x1.y() + t.y()) * d12;
  }
  static inline void Project(const LA::AlignedVector3f &t12, const LA::Vector3f &Rx1, const float d1,
                             LA::Vector2f &x2) {
    const LA::AlignedVector3f t = t12 * d1;
    const float d12 = 1.0f / (t.z() + Rx1.z());
    x2.x() = (Rx1.x() + t.x()) * d12;
    x2.y() = (Rx1.y() + t.y()) * d12;
  }
  static inline void Project(const Rigid3D &T12, const Point2D &x1, const float d1,
                             LA::Vector2f &x2) {
    const xp128f t = xp128f::get(x1.x(), x1.y(), 1.0f, d1);
    //const float z12 = SIMD::Sum(_mm_mul_ps(T12.r20_r21_r22_tz(), t));
    //if (z12 < FLT_EPSILON)
    //  return false;
    //const float d12 = 1.0f / z12;
    const float d12 = 1.0f / (T12.r20_r21_r22_tz() * t).vsum_all();
    x2.x() = (T12.r00_r01_r02_tx() * t).vsum_all() * d12;
    x2.y() = (T12.r10_r11_r12_ty() * t).vsum_all() * d12;
    //return true;
  }
  inline void Project(const Rigid3D &T12, const Point2D &x1, LA::Vector2f &x2, float &d2) const {
    const xp128f t = xp128f::get(x1.x(), x1.y(), 1.0f, u());
    //const float z12 = SIMD::Sum(_mm_mul_ps(T12.r20_r21_r22_tz(), t));
    //if (z12 < FLT_EPSILON)
    //  return false;
    //const float d12 = 1.0f / z12;
    const float d12 = 1.0f / (T12.r20_r21_r22_tz() * t).vsum_all();
    x2.x() = (T12.r00_r01_r02_tx() * t).vsum_all() * d12;
    x2.y() = (T12.r10_r11_r12_ty() * t).vsum_all() * d12;
    d2 = d12 * u();
    //return true;
  }
  inline void Project(const Rigid3D &T12, const Point2D &x1, LA::Vector2f &x2,
                      LA::Vector2f &Jx2) const {
    const xp128f t = xp128f::get(x1.x(), x1.y(), 1.0f, u());
    const float d12 = 1.0f / (T12.r20_r21_r22_tz() * t).vsum_all();
    x2.x() = (T12.r00_r01_r02_tx() * t).vsum_all() * d12;
    x2.y() = (T12.r10_r11_r12_ty() * t).vsum_all() * d12;
    Jx2.x() = (T12.tx() - x2.x() * T12.tz()) * d12;
    Jx2.y() = (T12.ty() - x2.y() * T12.tz()) * d12;
  }//看下面的注释吧,输入的参数是一模一样的,Rx1.z()因为是齐次表示,其实就是1
  inline void Project(const LA::AlignedVector3f &t12/*-tc0c1*/, const Point2D &x1, LA::Vector2f &x2,
                      LA::Vector2f &Jx2) const {
    const LA::AlignedVector3f t = t12 * u();
    const float d12 = 1.0f / (t.z() + 1.0f);
    x2.x() = (x1.x() + t.x()) * d12;
    x2.y() = (x1.y() + t.y()) * d12;
    Jx2.x() = (t12.x() - x2.x() * t12.z()) * d12;
    Jx2.y() = (t12.y() - x2.y() * t12.z()) * d12;
  }


  // 投影 Pc0 Pc1代表相机坐标下的特征点 Pnc0 和 Pnc1代表了归一化以后的点,齐次转换就省略了 Uc0,Uc1表示两个坐标系下的逆深度
  // Pc0 = Rc0c1 * Pc1 + tc0c1 ==>>  (1/Uc0) * Pnc0 = Rc0c1 * (1/Uc1) * Pnc1 + tc0c1
  // ==>> Rc0c1 *Pnc1* (Uc0/Uc1) = Pnc0 - Uc0 * tc0c1
  // ==> 对Pnc0 - Uc0 * tc0c1进行归一化,因为之前对Pnc1也做了（Rc0c1 *Pnc1)归一化坐标,存在z.m_z里
  inline void Project(const LA::AlignedVector3f &t12/*-tc0c1*/, const LA::Vector3f &Rx1/* 左目的无畸变归一化坐标*/, LA::Vector2f &x2/*残差*/,
                      LA::Vector2f &Jx2/*残差关于逆深度的雅克比*/) const {
    const LA::AlignedVector3f t = t12 * u();
    const float d12 = 1.0f / (t.z() + Rx1.z());
    x2.x() = (Rx1.x() + t.x()) * d12;      //Pnc0 - Uc0 * tc0c1进行归一化以后的0 1维
    x2.y() = (Rx1.y() + t.y()) * d12;
    Jx2.x() = (t12.x() - x2.x() * t12.z()) * d12;//div(x2.x())/div(u) 求残差关于逆深度的雅克比
    Jx2.y() = (t12.y() - x2.y() * t12.z()) * d12;//div(x2.y())/div(u)
  }
  inline void Project(const Rigid3D &T12/*这里给的变换明明是T21,这代码有毒*/, const Point2D &x1, LA::Vector2f &x2, float &d2,
                      LA::Vector2f &Jx2) const {
    const xp128f t = xp128f::get(x1.x(), x1.y(), 1.0f, u());
    const float d12 = 1.0f / (T12.r20_r21_r22_tz() * t).vsum_all();//这点在1中深度 / 这点在2中深度
    x2.x() = (T12.r00_r01_r02_tx() * t).vsum_all() * d12;
    x2.y() = (T12.r10_r11_r12_ty() * t).vsum_all() * d12;//投到2以后,2的归一化坐标
    d2 = d12 * u();//2中的逆深度
    // x2.x() = (r00 * x1.x() + r01 * x1.y() + r02 + tx() * u()) / (r20 * x1.x() + r21 * x1.y() + r22 + tx() * u())
    // div(x2.x())/div(u)就是下面这个 即kf中逆深度对于投影到LF中的归一化坐标的影响
    Jx2.x() = (T12.tx() - x2.x() * T12.tz()) * d12;//div(x2.x())/div(u)
    Jx2.y() = (T12.ty() - x2.y() * T12.tz()) * d12;//div(x2.y())/div(u)
  }
  inline void Project(const Rigid3D &T12, const Point2D &x1, LA::Vector2f &x2, float &d12, float &d2,
                      LA::Vector2f &Jx2) const {
    const xp128f t = xp128f::get(x1.x(), x1.y(), 1.0f, u());
    d12 = 1.0f / (T12.r20_r21_r22_tz() * t).vsum_all();
    x2.x() = (T12.r00_r01_r02_tx() * t).vsum_all() * d12;
    x2.y() = (T12.r10_r11_r12_ty() * t).vsum_all() * d12;
    d2 = d12 * u();
    Jx2.x() = (T12.tx() - x2.x() * T12.tz()) * d12;
    Jx2.y() = (T12.ty() - x2.y() * T12.tz()) * d12;
  }
  inline void Project(const Rigid3D &T12, const Point2D &x1, LA::Vector2f &x2, float &d12, float &d2,
                      LA::Vector2f &Jx2, LA::AlignedVector3f &R12x1) const {
    const xp128f t = xp128f::get(x1.x(), x1.y(), 1.0f, u());
    d12 = 1.0f / (T12.r20_r21_r22_tz() * t).vsum_all(&R12x1.z());
    x2.x() = (T12.r00_r01_r02_tx() * t).vsum_all(&R12x1.x()) * d12;
    x2.y() = (T12.r10_r11_r12_ty() * t).vsum_all(&R12x1.y()) * d12;
    d2 = d12 * u();
    Jx2.x() = (T12.tx() - x2.x() * T12.tz()) * d12;
    Jx2.y() = (T12.ty() - x2.y() * T12.tz()) * d12;
  }
  inline bool Project(const Rigid3D &T12, const Point2D &x1, const BoundingBox2D &B2,
                      Point2D &x2) const {
    const xp128f t = xp128f::get(x1.x(), x1.y(), 1.0f, u());
    const float z12 = (T12.r20_r21_r22_tz() * t).vsum_all();
    if (z12 < FLT_EPSILON) {
      return false;
    }
    const float d12 = 1.0f / z12;
    x2.x() = (T12.r00_r01_r02_tx() * t).vsum_all() * d12;
    x2.y() = (T12.r10_r11_r12_ty() * t).vsum_all() * d12;
    return B2.Inside(x2);
  }
  inline bool Project(const Rigid3D &T12, const Point2D &x1, const BoundingBox2D &B2, Point2D &x2,
                      LA::Vector2f &Jx2) const {
    const xp128f t = xp128f::get(x1.x(), x1.y(), 1.0f, u());
    const float z12 = (T12.r20_r21_r22_tz() * t).vsum_all();
    if (z12 < FLT_EPSILON) {
      return false;
    }
    const float d12 = 1.0f / z12;
    x2.x() = (T12.r00_r01_r02_tx() * t).vsum_all() * d12;
    x2.y() = (T12.r10_r11_r12_ty() * t).vsum_all() * d12;
    if (!B2.Inside(x2)) {
      return false;
    }
    Jx2.x() = (T12.tx() - x2.x() * T12.tz()) * d12;
    Jx2.y() = (T12.ty() - x2.y() * T12.tz()) * d12;
    return true;
  }
  inline bool Project(const Rigid3D &T12, const Point2D &x1, const BoundingBox2D &B2, Point2D &x2,
                      float &d2, LA::Vector2f &Jx2) const {
    const xp128f t = xp128f::get(x1.x(), x1.y(), 1.0f, u());
    const float z12 = (T12.r20_r21_r22_tz() * t).vsum_all();
    if (z12 < FLT_EPSILON) {
      return false;
    }
    const float d12 = 1.0f / z12;
    x2.x() = (T12.r00_r01_r02_tx() * t).vsum_all() * d12;
    x2.y() = (T12.r10_r11_r12_ty() * t).vsum_all() * d12;
    if (!B2.Inside(x2)) {
      return false;
    }
    d2 = u() * d12;
    Jx2.x() = (T12.tx() - x2.x() * T12.tz()) * d12;
    Jx2.y() = (T12.ty() - x2.y() * T12.tz()) * d12;
    return true;
  }
  inline Point2D GetProjected(const Rigid3D &T12, const Point2D &x1) const {
    Point2D x2;
    Project(T12, x1, x2);
    return x2;
  }
  inline Point2D GetProjected(const LA::AlignedVector3f &t12, const Point2D &x1) const {
    Point2D x2;
    Project(t12, x1, x2);
    return x2;
  }
  inline Point2D GetProjected(const LA::AlignedVector3f &t12, const LA::Vector3f &Rx1) const {
    Point2D x2;
    Project(t12, Rx1, x2);
    return x2;
  }
  inline float GetProjectedZ(const Rigid3D &T12, const Point2D &x1) const {
    const xp128f t = xp128f::get(x1.x(), x1.y(), 1.0f, u());
    return (T12.r20_r21_r22_tz() * t).vsum_all() / u();
  }
  inline bool ProjectD(const Rigid3D::Row &T12z/*(Tc0w(当前帧) * Twc0(关键帧)).row(z)*/, const Point2D &x1/*首次被观测到的归一化坐标*/, InverseGaussian &d2) const {
    const xp128f t1 = T12z.r0_r1_r2_t() * xp128f::get(x1.x(), x1.y(), 1.0f, u());//Pc(当前帧)[2]= u * (Tc0w(当前帧) * Twc0(关键帧)).row(2) * Pc0(关键帧)
    const float t2 = t1.vsum_012(), t3 = 1.0f / (t2 + t1[3]);//(t2 + t1[3] ) * (1/u() = Pc(当前帧)[2], 取倒数 d2.u() = t3* u()
    d2.u() = u() * t3;//这点在当前帧中的逆深度
    if (d2.u() < FLT_EPSILON) {
      return false;
    }
    const float j = t2 * t3 * t3;//这个应该是考虑的平移的影响来定义的协方差吧
    d2.s2() = j * j * s2();
    return true;
  }
  static inline void ProjectD(const Rigid3D &T21, const Point2D &x2, const float d2, float &d1,
                              LA::Vector2f &Jx2, float &Jd2, LA::AlignedVector6f &Jpr21) {
    const float z2 = 1.0f / d2;
    const xp128f X2 = xp128f::get(x2.x() * z2, x2.y() * z2, z2, 1.0f);
    const float z1 = (T21.r20_r21_r22_tz() * X2).vsum_all();
    d1 = 1.0f / z1;
    const float d12 = d1 * d1;
    Jpr21.v0123() = T21.r_20_21_22_x() * (-d12);
    Jpr21.v3() = (T21.r00_r01_r02_tx() * X2).vsum_all() * d12;
    Jpr21.v4() = (T21.r10_r11_r12_ty() * X2).vsum_all() * (-d12);
    Jpr21.v5() = 0.0f;
    Jx2.x() = Jpr21.v0() * z2;
    Jx2.y() = Jpr21.v1() * z2;
    Jd2 = (z2 - T21.tz()) * d12 * z2 * z2;
  }
  inline float GetProjectedD(const Rigid3D::Row &T12z, const Point2D &x1) const {
    const xp128f t = xp128f::get(x1.x(), x1.y(), 1.0f, u());
    return u() / (T12z.r0_r1_r2_t() * t).vsum_all();
  }
  inline void Print(const bool e = false, const bool n = true) const {
    if (Invalid()) {
      return;
    }
    if (e) {
      UT::Print("%e +- %e", u(), sqrtf(s2()));
    } else {
      UT::Print("%f +- %f", u(), sqrtf(s2()));
    }
    if (n) {
      UT::Print("\n");
    }
  }
  inline void Print(const std::string str, const bool e, const bool n) const {
    if (Invalid()) {
      return;
    }
    UT::Print("%s", str.c_str());
    Print(e, n);
  }
  inline bool AssertEqual(const InverseGaussian &d, const int verbose = 1,
                          const std::string str = "", const float eps = 0.0f) const {
    if (UT::AssertEqual(u(), d.u(), verbose, str + ".u", eps) &&
        UT::AssertEqual(s2(), d.s2(), verbose, str + ".s2", eps)) {
      return true;
    } else if (verbose) {
      UT::PrintSeparator();
      Print(verbose > 1);
      d.Print(verbose > 1);
    }
    return false;
  }

 protected:

  float m_u/*逆深度,0时代表不合法*/, m_s2/*协方差*/;
};

class InverseGaussianBeta : public InverseGaussian {

 public:

  inline const float& a () const { return m_a; }    inline float& a () { return m_a; }
  inline const float& b () const { return m_b; }    inline float& b () { return m_b; }

  inline void Initialize() { InverseGaussian::Initialize(); a() = b() = 10.0f; }
  inline void Initialize(const float u) { InverseGaussian::Initialize(u); a() = b() = 10.0f; }
  inline void Initialize(const float u, const float s2) { InverseGaussian::Set(u, s2); a() = b() = 10.0f; }
  inline void Initialize(const InverseGaussian &d) { InverseGaussian::Set(d); a() = b() = 10.0f; }
  inline bool Update(const InverseGaussian &d) {
    const float Ss2 = s2() + d.s2();
    if (Ss2 < FLT_EPSILON) {
      return true;
    }
    const float SabI = 1.0f / (a() + b()), Ss2I = 1.0f / Ss2, dz = d.u() - u(), dz2 = dz * dz;
    const float C1 = a() * SabI * exp(-dz * dz * 0.5f * Ss2I) * sqrtf(UT_1_2PI * Ss2I);
    const float C2 = b() * SabI / DEPTH_RANGE;
    const float SCI = 1.0f / (C1 + C2);
    const float Cn1 = C1 * SCI, Cn2 = C2 * SCI;
    const float w1 = 1.0f / s2(), w2 = 1.0f / d.s2();
    const float _s2 = 1.0f / (w2 + w1), m = _s2 * (d.u() * w2 + u() * w1);
    const float _u = Cn1 * m + Cn2 * u();
    s2() = Cn1 * (_s2 + m * m) + Cn2 * (s2() + u() * u()) - _u * _u;
    u() = _u;

    const float Sab1I = 1.0f / (a() + b() + 1.0f), Sab2I = 1.0f / (a() + b() + 2.0f);
    const float G1 = Cn1 * (a() + 1.0f) * Sab1I, G2 = Cn2 * a() * Sab1I;
    const float R1 = G1 + G2, R1I = 1.0f / R1;
    const float R2 = (G1 * (a() + 2.0f) + G2 * (a() + 1.0f)) * Sab2I;
    a() = (R2 - R1) / (R1 - R2 * R1I);
    b() = a() * (R1I - 1.0f);

    return Valid() && a() >= (a() + b()) * DEPTH_MIN_INLIER_RATIO;
  }
  inline void Print(const std::string str = "", const bool e = false, const bool n = true) const {
    InverseGaussian::Print(str, e, false);
    UT::Print("  pi = %.2f%% (%.2f %.2f)", UT::Percentage(a(), a() + b()), a(), b());
    if (n) {
      UT::Print("\n");
    }
  }

 protected:

  float m_a, m_b;
};
//深度的先验//J.t*(Wrobust * W)*J * deltax = -w * (Wrobust * W)*A.m_e => H * deltax = -b ，costFun(d) = ||d - m_d||^2 （m_w下的）
class Prior {
 public:
  class Factor {
   public:
    inline void MakeZero() { memset(this, 0, sizeof(Factor)); }
   public:
    float m_e/*实际逆深度 - 场景平均逆深度*/, m_F/*costFun*/, m_a/*hessian*/, m_b;//-b
  };
  class Reduction {
   public:
    float m_e, m_F, m_dF;
  };
 public:
  inline Prior(const float d, const float w) : m_d(d), m_w(w) {}
  template<int ME_FUNCTION>
  inline void GetFactor(const float w/*权重*/, const float d/*这个地图点的逆深度*/, Factor &A) const {
    A.m_e = d - m_d;//实际逆深度 - 场景平均逆深度
    const float r2 = m_w * A.m_e * A.m_e;//马氏距离下的残差
    const float _w = w * ME::Weight<ME_FUNCTION>(r2);//w*rho'(r2)
    A.m_F = _w * r2; //F就是代价函数,这里用鲁邦核修正过了 ||d - m_d||^2 （m_w下的）
    A.m_a = _w * m_w;//构造H矩阵 这里雅克比就是1,所以没有体现出来
    A.m_b = A.m_a * A.m_e;//J.t*(Wrobust * W)*J * deltax = -w * (Wrobust * W)*A.m_e => Hx = -b
  }
  inline float GetCost(const Factor &A, const float e) const {
    return A.m_a * e * e;
  }
  inline float GetCost(const Factor &A, const float xd, float &e) const {
    e = A.m_e + xd;
    return GetCost(A, e);
  }
  inline void GetReduction(const Factor &A, const float d, const float xd, Reduction &Ra,
                           Reduction &Rp) const {
    Ra.m_e = d - m_d;
    Rp.m_e = A.m_e + xd;
    Ra.m_dF = A.m_F - (Ra.m_F = GetCost(A, Ra.m_e));
    Rp.m_dF = A.m_F - (Rp.m_F = GetCost(A, Rp.m_e));
  }
 public:
  float m_d/*当前逆深度双目观测到的地图点的平均逆深度*/, m_w/*平均逆深度的信息矩阵*/;
};

class Measurement {
 public:
  inline Measurement() {}
  inline Measurement(const LA::AlignedVector3f &t, const LA::Vector3f &Rx, const Point2D &z,
                     const LA::SymmetricMatrix2x2f &W) { Set(t, Rx, z, W); }
  inline void Set(const LA::AlignedVector3f &t, const LA::Vector3f &Rx, const Point2D &z,
                  const LA::SymmetricMatrix2x2f &W) {
    m_t = &t;
    m_Rx = Rx;
    m_z = z;
    m_W = W;
  }
 public:
  const LA::AlignedVector3f *m_t;//-tc0_c1
  LA::Vector3f m_Rx;//关键帧中左目的无畸变归一化坐标
  Point2D m_z;//关键帧中右目的无畸变归一化坐标左乘了Rc0c1以后进行了归一化

  LA::SymmetricMatrix2x2f m_W;//右目特征点的畸变部分协方差,用来做马氏距离归一化
};

float ComputeError(const int N, const Measurement *zs, const InverseGaussian &d);
bool Triangulate(const float w, const LA::AlignedVector3f &t12, const Point2D &x1,
                 const Point2D &x2, InverseGaussian *d, const LA::SymmetricMatrix2x2f *Wx2 = NULL);
bool Triangulate(const float w, const int N, const Measurement *zs, InverseGaussian *d,
                 AlignedVector<float> *work, const bool initialized = true,
                 float *eAvg = NULL);
bool Triangulateinit(const float w, const int N, const Measurement *zs, InverseGaussian *d,
                     AlignedVector<float> *work,const Rotation3D & Rclcr,const Point3D & tclcr_n, const bool initialized = true,
                     float *eAvg = NULL);

    bool linearTriangulateFromNViews(
            const Eigen::Matrix3Xd & t_G_bv,//world系下的投影射线
            const Eigen::Matrix3Xd &p_G_C,//world系下的pose
            Eigen::Vector3d & p_G_P);//这点在左目坐标


}
#endif
