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
#include "Intrinsic.h"
#include "Parameter.h"
#include <Eigen/Core>

#include <iostream>
#ifdef CFG_DEBUG
//#define FTR_UNDIST_VERBOSE  1
//#define FTR_UNDIST_VERBOSE  2
#endif
#define FTR_UNDIST_DOG_LEG

void Intrinsic::UndistortionMap::Set(const Intrinsic &K) {
  const float sx = float(FTR_UNDIST_LUT_SIZE - 1) / K.w();
  const float sy = float(FTR_UNDIST_LUT_SIZE - 1) / K.h();
  m_fx = sx * K.k().m_fx; m_fy = sy * K.k().m_fy;//将图片缩放到FTR_UNDIST_LUT_SIZE×FTR_UNDIST_LUT_SIZEsize,然后进行畸变表的存储
  m_cx = sx * K.k().m_cx; m_cy = sy * K.k().m_cy;
  const float fxI = 1.0f / m_fx, fyI = 1.0f / m_fy;
  m_xns.Resize(FTR_UNDIST_LUT_SIZE, FTR_UNDIST_LUT_SIZE);//11*11这个在干吗？

#ifdef FTR_UNDIST_VERBOSE
  UT::PrintSeparator();
#endif
  Point2D xd, xn;
  float v;
  int x[2] = {FTR_UNDIST_LUT_SIZE / 2, FTR_UNDIST_LUT_SIZE / 2};
  int b[2][2] = {{x[0] - 1, x[0] + 1}, {x[1] - 1, x[1] + 1}};
  const int d[2] = {-1, 1};
  const int N = m_xns.Size();
  for(int i = 0, ix = 0, id = 0; i < N; ++i) {
    xd.x() = fxI * (x[0] - m_cx);
    xd.y() = fyI * (x[1] - m_cy);
    if (i == 0) {
      xn = xd;
    } else {
      xn = xd * v;
    }
    if (!K.Undistort(xd, &xn, NULL, NULL, true)) {
        continue;
//      exit(0);
    }
    v = sqrtf(xn.SquaredLength() / xd.SquaredLength());//大概的一个畸变比例
//#ifdef CFG_DEBUG
#if 0
    xn.x() = v;
#endif
    m_xns[x[1]][x[0]] = xn;//对应的无畸变的归一化坐标的值
//#ifdef CFG_DEBUG
#if 0
    //UT::Print("%d %d\n", x[0], x[1]);
    const float s = 0.5f;
    const int _w = K.gyr(), _h = K.h();
    const float _fx = K.fx() * s, _fy = K.fy() * s;
    const float _cx = (_w - 1) * 0.5f, _cy = (_h - 1) * 0.5f;
    CVD::Image<CVD::Rgb<ubyte> > I;
    I.resize(CVD::ImageRef(_w, _h));
    I.zero();
    UT::ImageDrawCross(I, int(_fx * xd.x() + _cx), int(_fy * xd.y() + _cy), 5, CVD::Rgb<ubyte>(255, 0, 0));
    UT::ImageDrawCross(I, int(_fx * xn.x() + _cx), int(_fy * xn.y() + _cy), 5, CVD::Rgb<ubyte>(0, 255, 0));
    UT::ImageSave(UT::String("D:/tmp/test%04d.jpg", i), I);
#endif
    //这里就是把 0,0 - 10,10范围内的都遍历一遍,这些像素点假设是带畸变的,m_xns保存的是对应的无畸变的归一话坐标的值
    if (x[ix] == b[ix][id]) {
      b[ix][id] += d[id];
      ix = 1 - ix;
      if (ix == 0) {
        id = 1 - id;
      }
    }
    x[ix] += d[id];
  }
#ifdef FTR_UNDIST_VERBOSE
  UT::PrintSeparator();
#endif
}
//去畸变,如果没有UndistortionMap畸变对照表的时候,需要将图片缩到FTR_UNDIST_LUT_SIZE*FTR_UNDIST_LUT_SIZE去做畸变对照。
//如果没有UndistortionMap输入或者生成UndistortionMap的时候,会用dogleg方法进行迭代求解。如果用了UndistortionMap作为初值的话,就用G-N求解即可
bool Intrinsic::Undistort(const Point2D &xd/*归一化坐标前两维*/, Point2D *xn/*待优化变量,最终反映的是无失真归一化坐标*/,
        LA::AlignedMatrix2x2f *JT/*畸变的雅克比J.t*/,UndistortionMap *UM/*畸变对应表*/, const bool initialized) const {
#ifdef CFG_DEBUG
  if (FishEye()) {
    UT::Error("TODO (haomin)\n");
  }
#endif
  if (!NeedUndistortion()) {//如果不需要去畸变
    *xn = xd;
    JT->MakeIdentity();
    return true;
  }
  float dr, j, dx2;
  LA::AlignedMatrix2x2f J;
  LA::SymmetricMatrix2x2f A;
  LA::AlignedMatrix2x2f AI;
  LA::Vector2f e, dx;//e就是残差
#ifdef FTR_UNDIST_DOG_LEG
  float dx2GN/*G-N解出的步长的欧式距离*/, dx2GD/*pU点的欧式距离*/, delta2/*信赖域半径*/, beta;
  LA::Vector2f dxGN/*G-N解出的增量*/, dxGD/*pU点*/;
  bool update, converge ,success;
    success = false;
#endif
  if (UM) {//如果预先做了畸变表,然么就用这里的畸变作为初值
    if (UM->Empty()) {
      UM->Set(*this);//如果没有畸变表，需要生成,因为只是做初值,所以就用FTR_UNDIST_LUT_SIZE*FTR_UNDIST_LUT_SIZE去做就好了,
      // key是畸变的像素坐标,value是无畸变的归一化坐标的值
    }
    *xn = UM->Get(xd);//用畸变表里对应的无畸变的归一化坐标作为初值
//#ifdef CFG_DEBUG
#if 0
    *xn = xd * xn->x();
#endif
  } else if (!initialized) {
    *xn = xd;//给个初值
  }
#if defined FTR_UNDIST_VERBOSE && FTR_UNDIST_VERBOSE == 1
  const Point2D _xd = m_k.GetNormalizedToImage(xd);
  UT::Print("x = %03d %03d", int(_xd.x() + 0.5f), int(_xd.y() + 0.5f));
  UT::Print("  e = %f", sqrtf((GetDistorted(*xn) - xd).SquaredLength() * m_k.m_fx * m_k.m_fy));
#endif
  const float *ds/*0_k1,1_k2,2_p1,3_p2,4_k3,5_k4,6_k5,7_k6*/ = m_k.m_ds, *jds = m_k.m_jds;//jds就是为了雅克比计算村的参数
  const float dx2Conv = FTR_UNDIST_CONVERGE * fxyI();//收敛条件
  //const float dx2Conv = FTR_UNDIST_CONVERGE * m_k.m_fxyI;
#ifdef FTR_UNDIST_DOG_LEG
  delta2 = FTR_UNDIST_DL_RADIUS_INITIAL;//初始化一下信赖域半径
#endif
  for (int iIter = 0; iIter < FTR_UNDIST_MAX_ITERATIONS; ++iIter)
  {//最大迭代10次
#if 0
//#if 1
    if (UT::Debugging()) {
      UT::Print("%d %e %e\n", iIter, xn->x(), xn->y());
    }
#endif
    ////针对k1,k2,k3,p1,p2而言 r^2 = x^2 + y^2
    ////残差2*1 归一化坐标的2维： min F(x,y) = 0.5*r(x,y)^2 约束： x^2+y^2 <=
    ///      r.x = x*(1 + k1*r^2 + k2*r^4 + k3*r^6) + 2*p1*x*y + p2*(r^2 + 2*x^2) - x0
    ////     r.y = y*(1 + k1*r^2 + k2*r^4 + k3*r^6) + 2*p2*x*y + p1*(r^2 + 2*y^2) - y0
    ////迭代法求,这里用的狗腿。下面是雅克比矩阵J,代码里是先给J矩阵径向部分然后再给切向部分
    ////J00 = div(r.x)/div(x) = (1 + k1*r^2 + k2*r^4 + k3*r^6) + (k1 + 2*k2*r^2 + 3*k3*r^4)*2*x^2 + 2*p1*y + 6*p2*x
    ////J01 = div(r.x)/div(y) = (k1 + 2*k2*r^2 + 3*k3*r^4)*x*y + 2*p1*x + 2*p2*y
    ////J10 = div(r.y)/div(x) = (k1 + 2*k2*r^2 + 3*k3*r^4)*x*y + 2*p1*x + 2*p2*y
    ////J11 = div(r.y)/div(y) = (1 + k1*r^2 + k2*r^4 + k3*r^6) + (k1 + 2*k2*r^2 + 3*k3*r^4)*2*y^2 + 6*p1*y + 2*p2*x
    const float x = xn->x(), x2 = x * x, y = xn->y(), y2 = y * y, xy = x * y;
    if(!m_fishEye)
    {
        const float r2 = x2 + y2, r4 = r2 * r2, r6 = r2 * r4;
        dr = ds[4] * r6 + ds[1] * r4 + ds[0] * r2 + 1.0f;// 1 + k1*r^2 + k2*r^4 + k3*r^6
        j = jds[4] * r4 + jds[1] * r2 + ds[0];//k1 + 2*k2*r^2 + 3*k3*r^4
        if (m_radial6)
        {
            const float dr2I = 1.0f / (ds[7] * r6 + ds[6] * r4 + ds[5] * r2 + 1.0f);
            dr *= dr2I;
            j = (j - (jds[7] * r4 + jds[6] * r2 + ds[5]) * dr) * dr2I;
        }
        //进行径向畸变
        e.x() = dr * x;//x*(1 + k1*r^2 + k2*r^4 + k3*r^6)
        e.y() = dr * y;//y*(1 + k1*r^2 + k2*r^4 + k3*r^6)
        //先对径向畸变的雅克比进行赋值
        J.m00() = j * (x2 + x2) + dr;//(1 + k1*r^2 + k2*r^4 + k3*r^6) + (k1 + 2*k2*r^2 + 3*k3*r^4)*2*x^2
        J.m01() = j * (xy + xy);//(k1 + 2*k2*r^2 + 3*k3*r^4)*x*y
        J.m11() = j * (y2 + y2) + dr;//(1 + k1*r^2 + k2*r^4 + k3*r^6) + (k1 + 2*k2*r^2 + 3*k3*r^4)*2*y^2
//#ifdef CFG_DEBUG
#if 0
        J.m00() = 1 - 3 * x2 - y2;
    J.m01() = -2 * xy;
    J.m11() = 1 - x2 - 3 * y2;
#endif
        if (m_tangential) {
            //切向畸变部分
            const float dx = jds[2] * xy + ds[3] * (r2 + x2 + x2);//2*p1*x*y + p2*(r^2 + 2*x^2)
            const float dy = jds[3] * xy + ds[2] * (r2 + y2 + y2);//2*p2*x*y + p1*(r^2 + 2*y^2)
            e.x() = dx + e.x();//在径向畸变后进行切向畸变
            e.y() = dy + e.y();
            const float d2x = jds[2] * x, d2y = jds[2] * y;//
            const float d3x = jds[3] * x, d3y = jds[3] * y;
            J.m00() = d3x + d3x + d3x + d2y + J.m00();// 2*p1*y + 6*p2*x + 径向雅克比
            J.m01() = d2x + d3y + J.m01();//2*p1*x + 2*p2*y + 径向雅克比
            J.m11() = d3x + d2y + d2y + d2y + J.m11();//6*p1*y + 2*p2*x + 径向雅克比
        }
        J.m10() = J.m01();
    }else
    {
        ////针对k1,k2,k2,k3的等距投影模型, r^2 = x^2 + y^2
        ////残差2*1 归一化坐标的2维： min F(x,y) = 0.5*r(x,y)^2 约束： x^2+y^2 <=
        ///      r.x = x*(theta*(1 + k1*theta^2 + k2*theta^4 + k3 *theta^6 + k4*theta^8)/r) - x0
        ////     r.x = y*(theta*(1 + k1*theta^2 + k2*theta^4 + k3 *theta^6 + k4*theta^8)/r) - y0
        // distortion first:
        const float u0 = x;//x
        const float u1 = y;//y
        const float r = sqrt(u0 * u0 + u1 * u1);//r = sqrt(x^2 + y^2)
        const float theta = atan(r);
        const float theta2 = theta * theta;
        const float theta4 = theta2 * theta2;
        const float theta6 = theta4 * theta2;
        const float theta8 = theta4 * theta4;
        const float thetad = theta * (1 + ds[0] * theta2 + ds[1] * theta4 + ds[2] * theta6 + ds[3] * theta8);

        const float scaling = (r > 1e-8) ? thetad / r : 1.0;
        e.x() = scaling * u0;
        e.y() = scaling * u1;

        if (r > 1e-8){
            // mostly matlab generated...
            float t2;//x^2
            float t3;//y^2
            float t4;//r^2 = x^2+y^2
            float t6;//theta
            float t7;//theta^2
            float t8;// 1/r
            float t9;//theta^4
            float t11;// 1 / ((x^2+y^2) + 1.0);
            float t17;// (((k1_ * theta^2 + k2_ * theta^4) + k3_ * t7 * t9) + k4_ * (t9 * t9)) + 1.0
            float t18;
            float t19;
            float t20;
            float t25;

            t2 = u0 * u0;
            t3 = u1 * u1;
            t4 = t2 + t3;
            t6 = atan(sqrt(t4));
            t7 = t6 * t6;
            t8 = 1.0 / sqrt(t4);
            t9 = t7 * t7;
            t11 = 1.0 / ((t2 + t3) + 1.0);
            t17 = (((ds[0] * t7 + ds[1] * t9) + ds[2] * t7 * t9) + ds[3] * (t9 * t9)) + 1.0;
            t18 = 1.0 / t4;
            t19 = 1.0 / sqrt(t4 * t4 * t4);
            t20 = t6 * t8 * t17;
            t25 = ((ds[1] * t6 * t7 * t8 * t11 * u1 * 4.0
                    + ds[2] * t6 * t8 * t9 * t11 * u1 * 6.0)
                   + ds[3] * t6 * t7 * t8 * t9 * t11 * u1 * 8.0)
                  + ds[0] * t6 * t8 * t11 * u1 * 2.0;
            t4 = ((ds[1] * t6 * t7 * t8 * t11 * u0 * 4.0
                   + ds[2] * t6 * t8 * t9 * t11 * u0 * 6.0)
                  + ds[3] * t6 * t7 * t8 * t9 * t11 * u0 * 8.0)
                 + ds[0] * t6 * t8 * t11 * u0 * 2.0;
            t7 = t11 * t17 * t18 * u0 * u1;
            J.m01() = (t7 + t6 * t8 * t25 * u0) - t6 * t17 * t19 * u0 * u1;
            J.m11() = ((t20 - t3 * t6 * t17 * t19) + t3 * t11 * t17 * t18)
                      + t6 * t8 * t25 * u1;
            J.m00() = ((t20 - t2 * t6 * t17 * t19) + t2 * t11 * t17 * t18)
                      + t6 * t8 * t4 * u0;
            J.m10() = (t7 + t6 * t8 * t4 * u1) - t6 * t17 * t19 * u0 * u1;
        } else {
            J.m00() = 1;
            J.m01() = 0;
            J.m11() = 1;
            J.m10() = 0;
        }
    }
    e -= xd;//计算残差
    LA::SymmetricMatrix2x2f::AAT(J, A);//构造H矩阵
//#ifdef CFG_DEBUG
#if 0
    A.Set(J.m00(), J.m01(), J.m11());
#endif
    const LA::Vector2f b = J * e;//构造Ax = -b
    if (!A.GetInverse(AI)) {//求解A逆
      //return false;
      break;
    }

    LA::AlignedMatrix2x2f::Ab(AI, b, dx);//求解-x

      dx.MakeMinus();//求出x
      dx2 = dx.SquaredLength();//x的欧式距离
#ifdef FTR_UNDIST_DOG_LEG
    if (!UM)
    {
      dxGN = dx;//G-N法求出的增量
      dx2GN = dx2;//G-N法求出的增量的欧式距离
      dx2GD = 0.0f;//dogleg
      const float F = e.SquaredLength();//dogleg迭代开始前残差的欧式距离
      const Point2D xnBkp = *xn;//待优化变量
      update = true;
      converge = false;//狗腿迭代次数
      for (int iIterDL = 0; iIterDL < FTR_UNDIST_DL_MAX_ITERATIONS; ++iIterDL) {
        if (dx2GN > delta2 && dx2GD == 0.0f) {//如果G-N增量在信赖域外且dx2GD还没有初始化
          const float bl = sqrtf(b.SquaredLength());//模长
          const LA::Vector2f g = b * (1.0f / bl);//梯度方向
          const LA::Vector2f Ag = A * g;
          const float xl = bl / g.Dot(Ag);//计算pU点的步长
          g.GetScaled(-xl, dxGD);//负梯度*步长,pU点
          dx2GD = xl * xl;//pU点的半径
#ifdef CFG_DEBUG
         UT::AssertEqual(dxGD.SquaredLength(), dx2GD);
#endif
        }
        //三种情况,1：GN极值点在域内直接变成无约束条件 2:GN和pU点都在域外,那么就在给pU点的步长一个比例因子(域半径/自己的步长^2（因为用的是最小2乘）),让它刚好落在域半径上
        //3:GN在域外,pU点在域内,那么增量就是GN极值点和pU点的连线与信赖域的交点
        if (dx2GN <= delta2) {//如果G-N的极值在信赖域内,那么就是一个无约束问题,就直接用GN法求出的增量就可以
          dx = dxGN;
          dx2 = dx2GN;
          beta = 1.0f;
        } else if (dx2GD >= delta2) {//如果G-N和pU点求的最优点都在信赖域外
          if (delta2 == 0.0f) {//信赖域为0,直接用pU点求得最优值
            dx = dxGD;
            dx2 = dx2GD;
          } else {
            dxGD.GetScaled(sqrtf(delta2 / dx2GD), dx);//乘比例因子
            dx2 = delta2;
          }
          beta = 0.0f;
        } else {//GN在域外,pU点在域内,那么增量就是GN极值点和pU点的连线与信赖域的交点
          const LA::Vector2f v = dxGN - dxGD;//方向
          const float d = dxGD.Dot(v), v2 = v.SquaredLength();
          //beta = float((-d + sqrt(double(d) * d + (delta2 - dx2GD) * double(v2))) / v2);
          beta = (-d + sqrtf(d * d + (delta2 - dx2GD) * v2)) / v2;//算得是在域外那段连线的长度
          dx = dxGD;
          dx += v * beta;
          dx2 = delta2;
        }
        *xn += dx;//加上这一次的增量
        const float dFa = F - (GetDistorted(*xn) - xd).SquaredLength();//实际下降的
        const float dFp = F - (e + J * dx).SquaredLength();//理论下降值,直接用J*dx近似下降值了
        const float rho = dFa > 0.0f && dFp > 0.0f ? dFa / dFp : -1.0f;//求实际/理论的比值,理论不可能为负,实际为负的时候拒绝这次更新
        //信赖域： Numerical Optimization 第二版 p69

          //rho < 0.25 如果大于0说明近似的不好,需要减小信赖域,减小近似的范围。如果<0就说明是错误的近似,那么就拒绝这次的增量
        if (rho < FTR_UNDIST_DL_GAIN_RATIO_MIN) {
          delta2 *= FTR_UNDIST_DL_RADIUS_FACTOR_DECREASE;
          if (delta2 < FTR_UNDIST_DL_RADIUS_MIN) {
            delta2 = FTR_UNDIST_DL_RADIUS_MIN;
          }
          *xn = xnBkp;//取消这次增量
          update = false;//不更新
          converge = false;
          continue;
        } else if (rho > FTR_UNDIST_DL_GAIN_RATIO_MAX) //rho > 0.75,可以扩大信赖域半径
        {
          delta2 = std::max(delta2, FTR_UNDIST_DL_RADIUS_FACTOR_INCREASE * dx2);
          if (delta2 > FTR_UNDIST_DL_RADIUS_MAX) {//信赖域半径最大值
            delta2 = FTR_UNDIST_DL_RADIUS_MAX;
          }
        }
        update = true;//
          const double chi2 = e.SquaredLength();;
          if (chi2 < dx2Conv * 1e5) {
              success = true;
          }
        converge = dx2 < dx2Conv;//增量小于阈值,认为收敛
        break;
      }
      if (!update || converge) {
          success = true;
        break;
      }
    } else
#endif
    {
      *xn += dx;
//      std::cout<<"iIter:"<<iIter<<" "<<"dx2:"<<dx2<<std::endl;
          const double chi2 = e.SquaredLength();;
          if (chi2 < dx2Conv * 1e5) {
              success = true;
          }
      if (dx2 < dx2Conv) {//收敛
          success = true;
        break;
      }
    }
#if defined FTR_UNDIST_VERBOSE && FTR_UNDIST_VERBOSE == 2
    const std::string str = UT::String("%02d  ", iIter);
    if (iIter == 0) {
      UT::PrintSeparator();
      m_k.GetNormalizedToImage(xd).Print(std::string(str.size(), ' ') + "x = ", false, true);
    }
    GetNormalizedToImage(xnBkp).Print(str + "x = ", false, false);
    UT::Print("  e = %f  dx = %f  beta = %f\n", sqrtf(F * fxy()), sqrtf(dx2 * fxy()), beta);
#endif
  }
  if (JT) {
    J.GetTranspose(*JT);
  }
#if defined FTR_UNDIST_VERBOSE && FTR_UNDIST_VERBOSE == 1
  UT::Print(" --> %f\n", sqrtf((GetDistorted(*xn) - xd).SquaredLength() * m_k.m_fx * m_k.m_fy));
#endif
  return true;
}
