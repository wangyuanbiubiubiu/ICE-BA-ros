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
//#ifdef CFG_DEBUG
//#define CFG_DEBUG_EIGEN
//#endif
#include "IMU.h"
#include <eigen3/Eigen/Dense>
namespace IMU {

void Delta::Factor::Auxiliary::Global::Set(const Jacobian::Global &J/*motion部分的雅克比*/, const Error &e,//motion部分的残差
                                           const float w, const Weight &W/*motion部分的信息矩阵*/, const float Tpv/*dt*/) {
  J.GetTranspose(m_JT);//m_Jrr1,m_Jrbw1,m_Jvr1,m_Jvv1,m_Jvba1,m_Jvbw1,m_Jpr1,m_Jpr2,m_Jpp1,m_Jpv1,m_Jpba1,m_Jpbw1的转置
  W.GetScaled(w, &m_W);
  const xp128f _Tpv = xp128f::get(Tpv);//dt
#ifdef CFG_IMU_FULL_COVARIANCE
  for (int i = 0; i < 5; ++i) {//m_W存储的顺序是R,v,p,ba,bw。
      //正常J.t * W是(rvpbabw)_i, (rvpbabw)_j
      //但是这里顺序变成了(prvbabw)_i, (prvbabw)_j,横排还是rvpbabw
      //所以m_JTW中第0行的第i个元素[0][i] = Jppi.t * W[2][i](==Wi2,为了和代码一致,我后面都用括号里这种形式表达了)
      //[1][i] = Jrri.t * W[i][0] + Jvri.t * W[i][1] + Jpri.t * W[i][2]
      //[2][i] = Jvvi.t * W[i][1] + Jpvi.t * W[i][2](这里为啥用的是dt)
      //[3][i] = Jvbai.t * W[i][1] + Jpbai.t * W[i][2] + W[i][3]
      //[4][i] = Jrbwi.t * W[i][0] + Jvbwi.t * W[i][1] + Jpbwi.t * W[i][2] + W[i][4]
      //[5][i] = -Jppi.t * W[i][2]
      //[6][i] = -Jrri.t * W[i][0] + Jprj.t * W[i][2]
      //[7][i] = -Jvvi.t * W[i][1]
      //[8][i] = -W[i][3]
      //[8][i] = -W[i][4]
    const LA::AlignedMatrix3x3f *Wi = m_W[i];//i对应的状态各自对应的那一行
    const LA::AlignedMatrix3x3f &Wir = Wi[0];//i和r部分的信息矩阵

    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrr1, Wir, m_JTW[1][i]);
    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrbw1, Wir, m_JTW[4][i]);
    m_JTW[1][i].GetMinus(m_JTW[6][i]);
    const LA::AlignedMatrix3x3f &Wiv = Wi[1];
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jvr1, Wiv, m_JTW[1][i]);
    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvv1, Wiv, m_JTW[2][i]);
    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvba1, Wiv, m_JTW[3][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jvbw1, Wiv, m_JTW[4][i]);
    m_JTW[2][i].GetMinus(m_JTW[7][i]);
    const LA::AlignedMatrix3x3f &Wip = Wi[2];
    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpp1, Wip, m_JTW[0][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpr1, Wip, m_JTW[1][i]);
    //LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpv1, Wip, m_JTW[2][i]);//???为什么不用这个而是下面这个,少了一个-Rciw啊
    LA::AlignedMatrix3x3f::AddsATo(_Tpv, m_JTW[0][i], m_JTW[2][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpba1, Wip, m_JTW[3][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpbw1, Wip, m_JTW[4][i]);
    m_JTW[0][i].GetMinus(m_JTW[5][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpr2, Wip, m_JTW[6][i]);
    m_JTW[3][i] += m_W[3][i];
    m_W[3][i].GetMinus(m_JTW[8][i]);
    m_JTW[4][i] += m_W[4][i];
    m_W[4][i].GetMinus(m_JTW[9][i]);
  }
//这里在算J.t*W*J的上三角存在m_A里
//这里使用的J.t*(J.t*w).t
//这里还做了一个行列的变化，因为最后行列都需要是(prvbabw)_i, (prvbabw)_j
//只计算上三角部分(这里&m_A[ 0], &m_A[ 9], &m_A[17], &m_A[24], &m_A[30],&m_A[35], &m_A[39], &m_A[42], &m_A[44], &m_A[45]
// 其实是&m_A[ 0], &m_A[ 9+1], &m_A[17+2], &m_A[24+3], &m_A[30+4],&m_A[35+5], &m_A[39+6], &m_A[42+7], &m_A[44+8], &m_A[45+9])
//i>= 代表是否需要保存
//A[0][i] = Jpp1.t * m_JTW[i][2].t p1对应的这行 i>=0
//A[1][i] = Jrr1.t * m_JTW[i][0].t + Jvr1.t * m_JTW[i][1].t + Jpr1.t * m_JTW[i][2].t r1对应的这行 i>=1
//A[2][i] = Jvv1.t * m_JTW[i][1].t + Jpv1.t * m_JTW[i][2].t v1对应的这行 i>=2
//A[3][i] = Jvba1.t * m_JTW[i][1].t + Jpba1.t * m_JTW[i][2].t + m_JTW[i][3].t ba1对应的这行 i>=3
//A[4][i] = Jrbw1.t * m_JTW[i][0].t + Jvbw1.t * m_JTW[i][1].t + Jpbw1.t * m_JTW[i][2].t + m_JTW[i][4].t bw1对应的这行 i>=4
//A[5][i] = -Jpp1.t * m_JTW[i][2].t p2对应的这行 i>=5
//A[6][i] = -Jrr1.t * m_JTW[i][0].t + Jpr2.t * m_JTW[i][2].t r2对应的这行 i>=6
//A[7][i] = -Jvv1.t * m_JTW[i][1].t v2对应的这行 i>=7
//A[8][i] = -m_JTW[i][3].t ba2对应的这行  i>=8
//A[9][i] = -m_JTW[i][4].t bw2对应的这行 i>=9

  LA::AlignedMatrix3x3f *A[10] = {&m_A[ 0], &m_A[ 9], &m_A[17], &m_A[24], &m_A[30],
                                  &m_A[35], &m_A[39], &m_A[42], &m_A[44], &m_A[45]};//这里是用的导推,就是45+9这样子,就是上三角
  const LA::AlignedVector3f *_e = (LA::AlignedVector3f *) &e;
  LA::AlignedMatrix3x3f WJ;
  for (int i = 0; i < 10; ++i)//遍历10行
  {
    const LA::AlignedMatrix3x3f *JTWi = m_JTW[i];
    if (i >= 1)
    {
      const LA::AlignedMatrix3x3f &JTWir = JTWi[0];
      LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrr1, JTWir, A[1][i], i == 1);
      if (i >= 4)
      {
        LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrbw1, JTWir, A[4][i], i == 4);
        if (i >= 6)
        {
          A[1][i].GetMinus(A[6][i]);
        }
      }
      const LA::AlignedMatrix3x3f &JTWiv = JTWi[1];
      LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jvr1, JTWiv, A[1][i], i == 1);
      if (i >= 2)
      {
        LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvv1, JTWiv, A[2][i], i == 2);
        if (i >= 3)
        {
          LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvba1, JTWiv, A[3][i], i == 3);
          if (i >= 4)
          {
            LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jvbw1, JTWiv, A[4][i], i == 4);
          }
          if (i >= 7)
          {
            A[2][i].GetMinus(A[7][i]);
          }
        }
      }
    }
    const LA::AlignedMatrix3x3f &JTWip = JTWi[2];
    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpp1, JTWip, A[0][i], i == 0);
    if (i >= 1)
    {
      LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpr1, JTWip, A[1][i], i == 1);
      if (i >= 2)
      {
        //LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpv1, JTWip, A[2][i], i == 2);
        LA::AlignedMatrix3x3f::AddsATo(_Tpv, A[0][i], A[2][i], i == 2);
        if (i >= 3)
        {
          LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpba1, JTWip, A[3][i], i == 3);
          if (i >= 4)
          {
            LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpbw1, JTWip, A[4][i], i == 4);
            if (i >= 5)
            {
              A[0][i].GetMinus(A[5][i]);
              if (i >= 6)
              {
                LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpr2, JTWip, A[6][i], i == 6);
              }
            }
          }
        }
      }
    }
    if (i >= 3)
    {
      JTWi[3].GetTranspose(WJ);
      A[3][i] += WJ;
      if (i == 3)
      {
        A[3][i].SetLowerFromUpper();
      } else if (i >= 8)
      {
        WJ.GetMinus(A[8][i]);
      }
      if (i >= 4)
      {
        JTWi[4].GetTranspose(WJ);
        A[4][i] += WJ;
        if (i == 4)
        {
          A[4][i].SetLowerFromUpper();
        } else if (i >= 9) {
          WJ.GetMinus(A[9][i]);
        }
      }
    }
    LA::AlignedVector3f &bi = m_b[i];//J.t*W*e
    bi.MakeZero();
    float *_bi = bi;
    for (int j = 0; j < 5; ++j)
    {
      LA::AlignedMatrix3x3f::AddAbTo(JTWi[j], _e[j], _bi);
    }
  }
#else
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrr1, m_W.m_Wr, m_JTWr1r);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrbw1, m_W.m_Wr, m_JTWbw1r);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvr1, m_W.m_Wv, m_JTWr1v);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvv1, m_W.m_Wv, m_JTWv1v);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvba1, m_W.m_Wv, m_JTWba1v);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvbw1, m_W.m_Wv, m_JTWbw1v);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpp1, m_W.m_Wp, m_JTWp1p);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpr1, m_W.m_Wp, m_JTWr1p);
  //LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpv1, m_W.m_Wp, m_JTWv1p);
  m_JTWp1p.GetScaled(_Tpv, m_JTWv1p);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpba1, m_W.m_Wp, m_JTWba1p);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpbw1, m_W.m_Wp, m_JTWbw1p);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpr2, m_W.m_Wp, m_JTWr2p);

  LA::SymmetricMatrix3x3f::ABT(m_JTWp1p, m_JT.m_Jpp1, m_Ap1p1);
  LA::AlignedMatrix3x3f::ABT(m_JTWp1p, m_JT.m_Jpr1, m_Ap1r1);
  //LA::AlignedMatrix3x3f::ABT(m_JTWp1p, m_JT.m_Jpv1, m_Ap1v1);
  m_Ap1p1.GetScaled(Tpv, m_Ap1v1);
  LA::AlignedMatrix3x3f::ABT(m_JTWp1p, m_JT.m_Jpba1, m_Ap1ba1);
  LA::AlignedMatrix3x3f::ABT(m_JTWp1p, m_JT.m_Jpbw1, m_Ap1bw1);
  m_Ap1p1.GetMinus(m_Ap1p2);
  LA::AlignedMatrix3x3f::ABT(m_JTWp1p, m_JT.m_Jpr2, m_Ap1r2);
  LA::AlignedMatrix3x3f::Ab(m_JTWp1p, e.m_ep, m_bp1);

  LA::SymmetricMatrix3x3f::ABT(m_JTWr1r, m_JT.m_Jrr1, m_Ar1r1);
  m_Ar1r1.GetMinus(m_Ar1r2);
  m_Ar2r2 = m_Ar1r1;
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWr1v, m_JT.m_Jvr1, m_Ar1r1);
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWr1p, m_JT.m_Jpr1, m_Ar1r1);
  LA::AlignedMatrix3x3f::ABT(m_JTWr1v, m_JT.m_Jvv1, m_Ar1v1);
  m_Ar1v1.GetMinus(m_Ar1v2);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWr1p, m_JT.m_Jpv1, m_Ar1v1);
  LA::AlignedMatrix3x3f::ABT(m_JTWr1v, m_JT.m_Jvba1, m_Ar1ba1);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWr1p, m_JT.m_Jpba1, m_Ar1ba1);
  LA::AlignedMatrix3x3f::ABT(m_JTWr1r, m_JT.m_Jrbw1, m_Ar1bw1);
  m_Ar1bw1.GetMinus(m_Abw1r2);  m_Abw1r2.Transpose();
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWr1v, m_JT.m_Jvbw1, m_Ar1bw1);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWr1p, m_JT.m_Jpbw1, m_Ar1bw1);
  m_Ap1r1.GetMinus(m_Ar1p2);  m_Ar1p2.Transpose();
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWr1p, m_JT.m_Jpr2, m_Ar1r2);
  LA::AlignedMatrix3x3f::Ab(m_JTWr1r, e.m_er, m_br1);
  m_br1.GetMinus(m_br2);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWr1v, e.m_ev, m_br1);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWr1p, e.m_ep, m_br1);

  LA::SymmetricMatrix3x3f::ABT(m_JTWv1v, m_JT.m_Jvv1, m_Av1v1);
  m_Av1v1.GetMinus(m_Av1v2);
  m_Av2v2 = m_Av1v1;
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWv1p, m_JT.m_Jpv1, m_Av1v1);
  LA::AlignedMatrix3x3f::ABT(m_JTWv1v, m_JT.m_Jvba1, m_Av1ba1);
  m_Av1ba1.GetMinus(m_Aba1v2);  m_Aba1v2.Transpose();
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWv1p, m_JT.m_Jpba1, m_Av1ba1);
  LA::AlignedMatrix3x3f::ABT(m_JTWv1v, m_JT.m_Jvbw1, m_Av1bw1);
  m_Av1bw1.GetMinus(m_Abw1v2);  m_Abw1v2.Transpose();
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWv1p, m_JT.m_Jpbw1, m_Av1bw1);
  m_Ap1v1.GetMinus(m_Av1p2);    m_Av1p2.Transpose();
  LA::AlignedMatrix3x3f::ABT(m_JTWv1p, m_JT.m_Jpr2, m_Av1r2);
  LA::AlignedMatrix3x3f::Ab(m_JTWv1v, e.m_ev, m_bv1);
  m_bv1.GetMinus(m_bv2);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWv1p, e.m_ep, m_bv1);

  LA::SymmetricMatrix3x3f::ABT(m_JTWba1v, m_JT.m_Jvba1, m_Aba1ba1);
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWba1p, m_JT.m_Jpba1, m_Aba1ba1);
  m_Aba1ba1.IncreaseDiagonal(m_W.m_wba);
  LA::AlignedMatrix3x3f::ABT(m_JTWba1v, m_JT.m_Jvbw1, m_Aba1bw1);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWba1p, m_JT.m_Jpbw1, m_Aba1bw1);
  m_Ap1ba1.GetMinus(m_Aba1p2);  m_Aba1p2.Transpose();
  LA::AlignedMatrix3x3f::ABT(m_JTWba1p, m_JT.m_Jpr2, m_Aba1r2);
  m_Aba1ba2 = -m_W.m_wba;
  e.m_eba.GetScaled(m_W.m_wba, m_bba1);
  m_bba1.GetMinus(m_bba2);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWba1v, e.m_ev, m_bba1);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWba1p, e.m_ep, m_bba1);

  LA::SymmetricMatrix3x3f::ABT(m_JTWbw1r, m_JT.m_Jrbw1, m_Abw1bw1);
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWbw1v, m_JT.m_Jvbw1, m_Abw1bw1);
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWbw1p, m_JT.m_Jpbw1, m_Abw1bw1);
  m_Abw1bw1.IncreaseDiagonal(m_W.m_wbw);
  m_Ap1bw1.GetMinus(m_Abw1p2);  m_Abw1p2.Transpose();
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWbw1p, m_JT.m_Jpr2, m_Abw1r2);
  m_Abw1bw2 = -m_W.m_wbw;
  e.m_ebw.GetScaled(m_W.m_wbw, m_bbw1);
  m_bbw1.GetMinus(m_bbw2);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWbw1r, e.m_er, m_bbw1);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWbw1v, e.m_ev, m_bbw1);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWbw1p, e.m_ep, m_bbw1);

  m_Ap2p2 = m_Ap1p1;
  m_Ap1r2.GetMinus(m_Ap2r2);
  m_bp1.GetMinus(m_bp2);

  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWr2p, m_JT.m_Jpr2, m_Ar2r2);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWr2p, e.m_ep, m_br2);

  m_Aba2ba2 = m_W.m_wba;
  m_Abw2bw2 = m_W.m_wbw;
#endif
}


void Delta::Factor::Auxiliary::RelativeLF::Set(const Jacobian::RelativeLF &J, const Error &e,
                                               const float w, const Weight &W, const float Tpv) {
  J.GetTranspose(m_JT);
  J.m_Jvr2.GetTranspose(m_Jvr2T);
  J.m_Jvv2.GetTranspose(m_Jvv2T);
  W.GetScaled(w, &m_W);
  const xp128f Jpv1 = xp128f::get(-Tpv);
#ifdef CFG_IMU_FULL_COVARIANCE
  for (int i = 0; i < 5; ++i) {
    const LA::AlignedMatrix3x3f *Wi = m_W[i];
    const LA::AlignedMatrix3x3f &Wir = Wi[0];
    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrr1, Wir, m_JTW[1][i]);
    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrbw1, Wir, m_JTW[4][i]);
    m_JTW[1][i].GetMinus(m_JTW[6][i]);
    const LA::AlignedMatrix3x3f &Wiv = Wi[1];
    LA::AlignedMatrix2x3f::ABT(J.m_JvgT, Wiv, m_JTWg[i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jvr1, Wiv, m_JTW[1][i]);
    //LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvv1, Wiv, m_JTW[2][i]);
    m_W[1][i].GetMinus(m_JTW[2][i]);
    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvba1, Wiv, m_JTW[3][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jvbw1, Wiv, m_JTW[4][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_Jvr2T, Wiv, m_JTW[6][i]);
    LA::AlignedMatrix3x3f::ABT(m_Jvv2T, Wiv, m_JTW[7][i]);
    const LA::AlignedMatrix3x3f &Wip = Wi[2];
    LA::AlignedMatrix2x3f::AddABTTo(J.m_JpgT, Wip, m_JTWg[i]);
    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpp1, Wip, m_JTW[0][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpr1, Wip, m_JTW[1][i]);
    //LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpv1, Wip, m_JTW[2][i]);
    LA::AlignedMatrix3x3f::AddsATo(Jpv1, m_W[2][i], m_JTW[2][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpba1, Wip, m_JTW[3][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpbw1, Wip, m_JTW[4][i]);
    m_JTW[0][i].GetMinus(m_JTW[5][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpr2, Wip, m_JTW[6][i]);
    m_JTW[3][i] += m_W[3][i];
    m_W[3][i].GetMinus(m_JTW[8][i]);
    m_JTW[4][i] += m_W[4][i];
    m_W[4][i].GetMinus(m_JTW[9][i]);
  }

  LA::AlignedMatrix2x3f::ABT(m_JTWg[1], J.m_JvgT, m_Agg);
  LA::AlignedMatrix2x3f::AddABTTo(m_JTWg[2], J.m_JpgT, m_Agg);
  const LA::AlignedVector3f *_e = (LA::AlignedVector3f *) &e;
  m_bg.MakeZero();
  for (int i = 0; i < 5; ++i) {
    LA::AlignedMatrix2x3f::AddAbTo(m_JTWg[i], _e[i], m_bg);
  }
  LA::AlignedMatrix3x3f *A[10] = {&m_A[ 0], &m_A[ 9], &m_A[17], &m_A[24], &m_A[30],
                                  &m_A[35], &m_A[39], &m_A[42], &m_A[44], &m_A[45]};
  LA::AlignedMatrix3x3f WJ;
  for (int i = 0; i < 10; ++i) {
    const LA::AlignedMatrix3x3f *JTWi = m_JTW[i], &JTWiv = JTWi[1], &JTWip = JTWi[2];
    LA::AlignedMatrix2x3f::ABT(J.m_JvgT, JTWiv, m_Agc[i]);
    LA::AlignedMatrix2x3f::AddABTTo(J.m_JpgT, JTWip, m_Agc[i]);
    if (i >= 1) {
      const LA::AlignedMatrix3x3f &JTWir = JTWi[0];
      LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrr1, JTWir, A[1][i], i == 1);
      if (i >= 4) {
        LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrbw1, JTWir, A[4][i], i == 4);
        if (i >= 6) {
          A[1][i].GetMinus(A[6][i]);
        }
      }
      LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jvr1, JTWiv, A[1][i], i == 1);
      if (i >= 2) {
        //LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvv1, JTWiv, A[2][i], i == 2);
        JTWiv.GetTranspose(A[2][i]);
        A[2][i].MakeMinus();
        if (i == 2) {
          A[2][i].SetLowerFromUpper();
        }
        if (i >= 3) {
          LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvba1, JTWiv, A[3][i], i == 3);
          if (i >= 4) {
            LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jvbw1, JTWiv, A[4][i], i == 4);
          }
          if (i >= 6) {
            LA::AlignedMatrix3x3f::AddABTTo(m_Jvr2T, JTWiv, A[6][i], i == 6);
            if (i >= 7) {
              LA::AlignedMatrix3x3f::ABT(m_Jvv2T, JTWiv, A[7][i], i == 7);
            }
          }
        }
      }
    }
    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpp1, JTWip, A[0][i], i == 0);
    if (i >= 1) {
      LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpr1, JTWip, A[1][i], i == 1);
      if (i >= 2) {
        //LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpv1, JTWip, A[2][i], i == 2);
        JTWip.GetTranspose(WJ);
        LA::AlignedMatrix3x3f::AddsATo(Jpv1, WJ, A[2][i], i == 2);
        if (i >= 3) {
          LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpba1, JTWip, A[3][i], i == 3);
          if (i >= 4) {
            LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpbw1, JTWip, A[4][i], i == 4);
            if (i >= 5) {
              A[0][i].GetMinus(A[5][i]);
              if (i >= 6) {
                LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpr2, JTWip, A[6][i], i == 6);
              }
            }
          }
        }
      }
    }
    if (i >= 3) {
      JTWi[3].GetTranspose(WJ);
      A[3][i] += WJ;
      if (i == 3) {
        A[3][i].SetLowerFromUpper();
      } else if (i >= 8) {
        WJ.GetMinus(A[8][i]);
      }
      if (i >= 4) {
        JTWi[4].GetTranspose(WJ);
        A[4][i] += WJ;
        if (i == 4) {
          A[4][i].SetLowerFromUpper();
        } else if (i >= 9) {
          WJ.GetMinus(A[9][i]);
        }
      }
    }
    LA::AlignedVector3f &bi = m_b[i];
    bi.MakeZero();
    float *_bi = bi;
    for (int j = 0; j < 5; ++j) {
      LA::AlignedMatrix3x3f::AddAbTo(JTWi[j], _e[j], _bi);
    }
  }
#else
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrr1, m_W.m_Wr, m_JTWr1r);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrbw1, m_W.m_Wr, m_JTWbw1r);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvr1, m_W.m_Wv, m_JTWr1v);
  //LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvv1, m_W.m_Wv, m_JTWv1v);
  m_W.m_Wv.GetMinus(m_JTWv1v);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvba1, m_W.m_Wv, m_JTWba1v);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvbw1, m_W.m_Wv, m_JTWbw1v);
  LA::AlignedMatrix3x3f::ABT(m_Jvr2T, m_W.m_Wv, m_JTWr2v);
  LA::AlignedMatrix3x3f::ABT(m_Jvv2T, m_W.m_Wv, m_JTWv2v);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpp1, m_W.m_Wp, m_JTWp1p);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpr1, m_W.m_Wp, m_JTWr1p);
  //LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpv1, m_W.m_Wp, m_JTWv1p);
  m_W.m_Wp.GetScaled(Jpv1, m_JTWv1p);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpba1, m_W.m_Wp, m_JTWba1p);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpbw1, m_W.m_Wp, m_JTWbw1p);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpr2, m_W.m_Wp, m_JTWr2p);
  LA::AlignedMatrix2x3f::ABT(J.m_JvgT, m_W.m_Wv, m_JTWgv);
  LA::AlignedMatrix2x3f::ABT(J.m_JpgT, m_W.m_Wp, m_JTWgp);

  LA::SymmetricMatrix3x3f::ABT(m_JTWp1p, m_JT.m_Jpp1, m_Ap1p1);
  LA::AlignedMatrix3x3f::ABT(m_JTWp1p, m_JT.m_Jpr1, m_Ap1r1);
  //LA::AlignedMatrix3x3f::ABT(m_JTWp1p, m_JT.m_Jpv1, m_Ap1v1);
  m_JTWp1p.GetScaled(Jpv1, m_Ap1v1);
  LA::AlignedMatrix3x3f::ABT(m_JTWp1p, m_JT.m_Jpba1, m_Ap1ba1);
  LA::AlignedMatrix3x3f::ABT(m_JTWp1p, m_JT.m_Jpbw1, m_Ap1bw1);
  m_Ap1p1.GetMinus(m_Ap1p2);
  LA::AlignedMatrix3x3f::ABT(m_JTWp1p, m_JT.m_Jpr2, m_Ap1r2);
  LA::AlignedMatrix3x3f::Ab(m_JTWp1p, e.m_ep, m_bp1);

  LA::SymmetricMatrix3x3f::ABT(m_JTWr1r, m_JT.m_Jrr1, m_Ar1r1);
  m_Ar1r1.GetMinus(m_Ar1r2);
  m_Ar2r2 = m_Ar1r1;
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWr1v, m_JT.m_Jvr1, m_Ar1r1);
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWr1p, m_JT.m_Jpr1, m_Ar1r1);
  //LA::AlignedMatrix3x3f::ABT(m_JTWr1v, m_JT.m_Jvv1, m_Ar1v1);
  //LA::AlignedMatrix3x3f::AddABTTo(m_JTWr1p, m_JT.m_Jpv1, m_Ar1v1);
  m_JTWr1p.GetScaled(Jpv1, m_Ar1v1);
  m_Ar1v1 -= m_JTWr1v;
  LA::AlignedMatrix3x3f::ABT(m_JTWr1v, m_JT.m_Jvba1, m_Ar1ba1);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWr1p, m_JT.m_Jpba1, m_Ar1ba1);
  LA::AlignedMatrix3x3f::ABT(m_JTWr1r, m_JT.m_Jrbw1, m_Ar1bw1);
  m_Ar1bw1.GetMinus(m_Abw1r2);  m_Abw1r2.Transpose();
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWr1v, m_JT.m_Jvbw1, m_Ar1bw1);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWr1p, m_JT.m_Jpbw1, m_Ar1bw1);
  m_Ap1r1.GetMinus(m_Ar1p2);  m_Ar1p2.Transpose();
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWr1v, m_Jvr2T, m_Ar1r2);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWr1p, m_JT.m_Jpr2, m_Ar1r2);
  LA::AlignedMatrix3x3f::ABT(m_JTWr1v, m_Jvv2T, m_Ar1v2);
  LA::AlignedMatrix3x3f::Ab(m_JTWr1r, e.m_er, m_br1);
  m_br1.GetMinus(m_br2);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWr1v, e.m_ev, m_br1);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWr1p, e.m_ep, m_br1);

  //LA::SymmetricMatrix3x3f::ABT(m_JTWv1v, m_JT.m_Jvv1, m_Av1v1);
  //LA::SymmetricMatrix3x3f::AddABTTo(m_JTWv1p, m_JT.m_Jpv1, m_Av1v1);
  m_Av1v1.Set(m_W.m_Wv);
  LA::SymmetricMatrix3x3f::AddsATo(Jpv1, m_JTWv1p, m_Av1v1);
  LA::AlignedMatrix3x3f::ABT(m_JTWv1v, m_JT.m_Jvba1, m_Av1ba1);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWv1p, m_JT.m_Jpba1, m_Av1ba1);
  LA::AlignedMatrix3x3f::ABT(m_JTWv1v, m_JT.m_Jvbw1, m_Av1bw1);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWv1p, m_JT.m_Jpbw1, m_Av1bw1);
  m_Ap1v1.GetMinus(m_Av1p2);    m_Av1p2.Transpose();
  LA::AlignedMatrix3x3f::ABT(m_JTWv1v, m_Jvr2T, m_Av1r2);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWv1p, m_JT.m_Jpr2, m_Av1r2);
  LA::AlignedMatrix3x3f::ABT(m_JTWv1v, m_Jvv2T, m_Av1v2);
  LA::AlignedMatrix3x3f::Ab(m_JTWv1v, e.m_ev, m_bv1);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWv1p, e.m_ep, m_bv1);

  LA::SymmetricMatrix3x3f::ABT(m_JTWba1v, m_JT.m_Jvba1, m_Aba1ba1);
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWba1p, m_JT.m_Jpba1, m_Aba1ba1);
  m_Aba1ba1.IncreaseDiagonal(m_W.m_wba);
  LA::AlignedMatrix3x3f::ABT(m_JTWba1v, m_JT.m_Jvbw1, m_Aba1bw1);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWba1p, m_JT.m_Jpbw1, m_Aba1bw1);
  m_Ap1ba1.GetMinus(m_Aba1p2);  m_Aba1p2.Transpose();
  LA::AlignedMatrix3x3f::ABT(m_JTWba1v, m_Jvr2T, m_Aba1r2);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWba1p, m_JT.m_Jpr2, m_Aba1r2);
  LA::AlignedMatrix3x3f::ABT(m_JTWba1v, m_Jvv2T, m_Aba1v2);
  m_Aba1ba2 = -m_W.m_wba;
  e.m_eba.GetScaled(m_W.m_wba, m_bba1);
  m_bba1.GetMinus(m_bba2);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWba1v, e.m_ev, m_bba1);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWba1p, e.m_ep, m_bba1);

  LA::SymmetricMatrix3x3f::ABT(m_JTWbw1r, m_JT.m_Jrbw1, m_Abw1bw1);
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWbw1v, m_JT.m_Jvbw1, m_Abw1bw1);
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWbw1p, m_JT.m_Jpbw1, m_Abw1bw1);
  m_Abw1bw1.IncreaseDiagonal(m_W.m_wbw);
  m_Ap1bw1.GetMinus(m_Abw1p2);  m_Abw1p2.Transpose();
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWbw1v, m_Jvr2T, m_Abw1r2);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWbw1p, m_JT.m_Jpr2, m_Abw1r2);
  LA::AlignedMatrix3x3f::ABT(m_JTWbw1v, m_Jvv2T, m_Abw1v2);
  m_Abw1bw2 = -m_W.m_wbw;
  e.m_ebw.GetScaled(m_W.m_wbw, m_bbw1);
  m_bbw1.GetMinus(m_bbw2);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWbw1r, e.m_er, m_bbw1);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWbw1v, e.m_ev, m_bbw1);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWbw1p, e.m_ep, m_bbw1);

  m_Ap2p2 = m_Ap1p1;
  m_Ap1r2.GetMinus(m_Ap2r2);
  m_bp1.GetMinus(m_bp2);

  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWr2v, m_Jvr2T, m_Ar2r2);
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWr2p, m_JT.m_Jpr2, m_Ar2r2);
  LA::AlignedMatrix3x3f::ABT(m_JTWr2v, m_Jvv2T, m_Ar2v2);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWr2v, e.m_ev, m_br2);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWr2p, e.m_ep, m_br2);

  LA::SymmetricMatrix3x3f::ABT(m_JTWv2v, m_Jvv2T, m_Av2v2);
  LA::AlignedMatrix3x3f::Ab(m_JTWv2v, e.m_ev, m_bv2);

  m_Aba2ba2 = m_W.m_wba;
  m_Abw2bw2 = m_W.m_wbw;

  LA::AlignedMatrix2x3f::ABT(m_JTWgv, J.m_JvgT, m_Agg);
  LA::AlignedMatrix2x3f::AddABTTo(m_JTWgp, J.m_JpgT, m_Agg);
  LA::AlignedMatrix2x3f::ABT(m_JTWgp, m_JT.m_Jpp1, m_Agp1);
  LA::AlignedMatrix2x3f::ABT(m_JTWgv, m_JT.m_Jvr1, m_Agr1);
  LA::AlignedMatrix2x3f::AddABTTo(m_JTWgp, m_JT.m_Jpr1, m_Agr1);
  m_JTWgp.GetScaled(Jpv1, m_Agv1);
  m_Agv1 -= m_JTWgv;
  LA::AlignedMatrix2x3f::ABT(m_JTWgv, m_JT.m_Jvba1, m_Agba1);
  LA::AlignedMatrix2x3f::AddABTTo(m_JTWgp, m_JT.m_Jpba1, m_Agba1);
  LA::AlignedMatrix2x3f::ABT(m_JTWgv, m_JT.m_Jvbw1, m_Agbw1);
  LA::AlignedMatrix2x3f::AddABTTo(m_JTWgp, m_JT.m_Jpbw1, m_Agbw1);
  m_Agp1.GetMinus(m_Agp2);
  LA::AlignedMatrix2x3f::ABT(m_JTWgv, m_Jvr2T, m_Agr2);
  LA::AlignedMatrix2x3f::AddABTTo(m_JTWgp, m_JT.m_Jpr2, m_Agr2);
  LA::AlignedMatrix2x3f::ABT(m_JTWgv, m_Jvv2T, m_Agv2);
  LA::AlignedMatrix2x3f::Ab(m_JTWgv, e.m_ev, m_bg);
  LA::AlignedMatrix2x3f::AddAbTo(m_JTWgp, e.m_ep, m_bg);
#endif
}

void Delta::Factor::Auxiliary::RelativeKF::Set(const Jacobian::RelativeKF &J, const Error &e,
                                               const float w, const Weight &W, const float Tpv) {
  J.GetTranspose(m_JT);
  W.GetScaled(w, &m_W);
  const xp128f Jpv1 = xp128f::get(-Tpv);
#ifdef CFG_IMU_FULL_COVARIANCE
  for (int i = 0; i < 5; ++i) {
    const LA::AlignedMatrix3x3f *Wi = m_W[i];
    const LA::AlignedMatrix3x3f &Wir = Wi[0];
    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrbw1, Wir, m_JTWc[2][i]);
    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrr2, Wir, m_JTWc[4][i]);
    const LA::AlignedMatrix3x3f &Wiv = Wi[1];
    LA::AlignedMatrix2x3f::ABT(J.m_JvgT, Wiv, m_JTWg[i]);
    //LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvv1, Wiv, m_JTWc[0][i]);
    m_W[1][i].GetMinus(m_JTWc[0][i]);
    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvba1, Wiv, m_JTWc[1][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jvbw1, Wiv, m_JTWc[2][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jvr2, Wiv, m_JTWc[4][i]);
    LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvv2, Wiv, m_JTWc[5][i]);
    const LA::AlignedMatrix3x3f &Wip = Wi[2];
    LA::AlignedMatrix2x3f::AddABTTo(J.m_JpgT, Wip, m_JTWg[i]);
    //LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpv1, Wip, m_JTW[2][i]);
    LA::AlignedMatrix3x3f::AddsATo(Jpv1, m_W[2][i], m_JTWc[0][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpba1, Wip, m_JTWc[1][i]);
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpbw1, Wip, m_JTWc[2][i]);
    //LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpp2, Wip, m_JTWc[3][i]);
    m_JTWc[3][i] = m_W[2][i];
    LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpr2, Wip, m_JTWc[4][i]);
    m_JTWc[1][i] += m_W[3][i];
    m_W[3][i].GetMinus(m_JTWc[6][i]);
    m_JTWc[2][i] += m_W[4][i];
    m_W[4][i].GetMinus(m_JTWc[7][i]);
  }

  LA::AlignedMatrix2x3f::ABT(m_JTWg[1], J.m_JvgT, m_Agg);
  LA::AlignedMatrix2x3f::AddABTTo(m_JTWg[2], J.m_JpgT, m_Agg);
  const LA::AlignedVector3f *_e = (LA::AlignedVector3f *) &e;
  m_bg.MakeZero();
  for (int i = 0; i < 5; ++i) {
    LA::AlignedMatrix2x3f::AddAbTo(m_JTWg[i], _e[i], m_bg);
  }
  LA::AlignedMatrix3x3f *Ac[8] = {&m_Ac[ 0], &m_Ac[ 7], &m_Ac[13], &m_Ac[18],
                                  &m_Ac[22], &m_Ac[25], &m_Ac[27], &m_Ac[28]};
  LA::AlignedMatrix3x3f WJ;
  for (int i = 0; i < 8; ++i) {
    const LA::AlignedMatrix3x3f *JTWi = m_JTWc[i], &JTWiv = JTWi[1], &JTWip = JTWi[2];
    LA::AlignedMatrix2x3f::ABT(J.m_JvgT, JTWiv, m_Agc[i]);
    LA::AlignedMatrix2x3f::AddABTTo(J.m_JpgT, JTWip, m_Agc[i]);
    const LA::AlignedMatrix3x3f &JTWir = JTWi[0];
    if (i >= 2) {
      LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrbw1, JTWir, Ac[2][i], i == 2);
      if (i >= 4) {
        LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrr2, JTWir, Ac[4][i], i == 4);
      }
    }
    //LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvv1, JTWiv, Ac[0][i], i == 0);
    JTWiv.GetTranspose(Ac[0][i]);
    Ac[0][i].MakeMinus();
    if (i == 0) {
      Ac[0][i].SetLowerFromUpper();
    }
    if (i >= 1) {
      LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvba1, JTWiv, Ac[1][i], i == 1);
      if (i >= 2) {
        LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jvbw1, JTWiv, Ac[2][i], i == 2);
      }
      if (i >= 4) {
        LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jvr2, JTWiv, Ac[4][i], i == 4);
        if (i >= 5) {
          LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvv2, JTWiv, Ac[5][i], i == 5);
        }
      }
    }
    //LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpv1, JTWip, Ac[0][i], i == 0);
    JTWip.GetTranspose(WJ);
    LA::AlignedMatrix3x3f::AddsATo(Jpv1, WJ, Ac[0][i], i == 0);
    if (i >= 1) {
      LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpba1, JTWip, Ac[1][i], i == 1);
      if (i >= 2) {
        LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpbw1, JTWip, Ac[2][i], i == 2);
        if (i >= 3) {
          //LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpp2, JTWip, Ac[3][i], i == 3);
          Ac[3][i] = WJ;
          if (i == 3) {
            Ac[3][i].SetLowerFromUpper();
          }
          if (i >= 4) {
            LA::AlignedMatrix3x3f::AddABTTo(m_JT.m_Jpr2, JTWip, Ac[4][i], i == 4);
          }
        }
      }
      JTWi[3].GetTranspose(WJ);
      Ac[1][i] += WJ;
      if (i == 1) {
        Ac[1][i].SetLowerFromUpper();
      } else if (i >= 6) {
        WJ.GetMinus(Ac[6][i]);
      }
      if (i >= 2) {
        JTWi[4].GetTranspose(WJ);
        Ac[2][i] += WJ;
        if (i == 2) {
          Ac[2][i].SetLowerFromUpper();
        } else if (i >= 7) {
          WJ.GetMinus(Ac[7][i]);
        }
      }
    }
    LA::AlignedVector3f &bi = m_bc[i];
    bi.MakeZero();
    float *_bi = bi;
    for (int j = 0; j < 5; ++j) {
      LA::AlignedMatrix3x3f::AddAbTo(JTWi[j], _e[j], _bi);
    }
  }
#else
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrbw1, m_W.m_Wr, m_JTWbw1r);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jrr2, m_W.m_Wr, m_JTWr2r);
  //LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvv1, m_W.m_Wv, m_JTWv1v);
  m_W.m_Wv.GetMinus(m_JTWv1v);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvba1, m_W.m_Wv, m_JTWba1v);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvbw1, m_W.m_Wv, m_JTWbw1v);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvr2, m_W.m_Wv, m_JTWr2v);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jvv2, m_W.m_Wv, m_JTWv2v);
  //LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpv1, m_W.m_Wp, m_JTWv1p);
  m_W.m_Wp.GetScaled(Jpv1, m_JTWv1p);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpba1, m_W.m_Wp, m_JTWba1p);
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpbw1, m_W.m_Wp, m_JTWbw1p);
  //LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpp2, m_W.m_Wp, m_JTWp2p);
  m_JTWp2p = m_W.m_Wp;
  LA::AlignedMatrix3x3f::ABT(m_JT.m_Jpr2, m_W.m_Wp, m_JTWr2p);
  LA::AlignedMatrix2x3f::ABT(J.m_JvgT, m_W.m_Wv, m_JTWgv);
  LA::AlignedMatrix2x3f::ABT(J.m_JpgT, m_W.m_Wp, m_JTWgp);


  //LA::SymmetricMatrix3x3f::ABT(m_JTWv1v, m_JT.m_Jvv1, m_Av1v1);
  //LA::SymmetricMatrix3x3f::AddABTTo(m_JTWv1p, m_JT.m_Jpv1, m_Av1v1);
  m_Av1v1.Set(m_W.m_Wv);
  LA::SymmetricMatrix3x3f::AddsATo(Jpv1, m_JTWv1p, m_Av1v1);
  LA::AlignedMatrix3x3f::ABT(m_JTWv1v, m_JT.m_Jvba1, m_Av1ba1);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWv1p, m_JT.m_Jpba1, m_Av1ba1);
  LA::AlignedMatrix3x3f::ABT(m_JTWv1v, m_JT.m_Jvbw1, m_Av1bw1);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWv1p, m_JT.m_Jpbw1, m_Av1bw1);
  //LA::AlignedMatrix3x3f::ABT(m_JTWv1p, m_JT.m_Jpp2, m_Av1p2);
  m_Av1p2 = m_JTWv1p;
  LA::AlignedMatrix3x3f::ABT(m_JTWv1v, m_JT.m_Jvr2, m_Av1r2);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWv1p, m_JT.m_Jpr2, m_Av1r2);
  LA::AlignedMatrix3x3f::ABT(m_JTWv1v, m_JT.m_Jvv2, m_Av1v2);
  LA::AlignedMatrix3x3f::Ab(m_JTWv1v, e.m_ev, m_bv1);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWv1p, e.m_ep, m_bv1);

  LA::SymmetricMatrix3x3f::ABT(m_JTWba1v, m_JT.m_Jvba1, m_Aba1ba1);
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWba1p, m_JT.m_Jpba1, m_Aba1ba1);
  m_Aba1ba1.IncreaseDiagonal(m_W.m_wba);
  LA::AlignedMatrix3x3f::ABT(m_JTWba1v, m_JT.m_Jvbw1, m_Aba1bw1);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWba1p, m_JT.m_Jpbw1, m_Aba1bw1);
  //LA::AlignedMatrix3x3f::ABT(m_JTWba1p, m_JT.m_Jpp2, m_Aba1p2);
  m_Aba1p2 = m_JTWba1p;
  LA::AlignedMatrix3x3f::ABT(m_JTWba1v, m_JT.m_Jvr2, m_Aba1r2);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWba1p, m_JT.m_Jpr2, m_Aba1r2);
  LA::AlignedMatrix3x3f::ABT(m_JTWba1v, m_JT.m_Jvv2, m_Aba1v2);
  m_Aba1ba2 = -m_W.m_wba;
  e.m_eba.GetScaled(m_W.m_wba, m_bba1);
  m_bba1.GetMinus(m_bba2);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWba1v, e.m_ev, m_bba1);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWba1p, e.m_ep, m_bba1);

  LA::SymmetricMatrix3x3f::ABT(m_JTWbw1r, m_JT.m_Jrbw1, m_Abw1bw1);
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWbw1v, m_JT.m_Jvbw1, m_Abw1bw1);
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWbw1p, m_JT.m_Jpbw1, m_Abw1bw1);
  m_Abw1bw1.IncreaseDiagonal(m_W.m_wbw);
  //LA::AlignedMatrix3x3f::ABT(m_JTWbw1p, m_JT.m_Jpp2, m_Abw1p2);
  m_Abw1p2 = m_JTWbw1p;
  LA::AlignedMatrix3x3f::ABT(m_JTWbw1r, m_JT.m_Jrr2, m_Abw1r2);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWbw1v, m_JT.m_Jvr2, m_Abw1r2);
  LA::AlignedMatrix3x3f::AddABTTo(m_JTWbw1p, m_JT.m_Jpr2, m_Abw1r2);
  LA::AlignedMatrix3x3f::ABT(m_JTWbw1v, m_JT.m_Jvv2, m_Abw1v2);
  m_Abw1bw2 = -m_W.m_wbw;
  e.m_ebw.GetScaled(m_W.m_wbw, m_bbw1);
  m_bbw1.GetMinus(m_bbw2);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWbw1r, e.m_er, m_bbw1);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWbw1v, e.m_ev, m_bbw1);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWbw1p, e.m_ep, m_bbw1);

  m_Ap2p2.Set(m_W.m_Wp);
  m_JTWr2p.GetTranspose(m_Ap2r2);
  LA::AlignedMatrix3x3f::Ab(m_W.m_Wp, e.m_ep, m_bp2);

  LA::SymmetricMatrix3x3f::ABT(m_JTWr2r, m_JT.m_Jrr2, m_Ar2r2);
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWr2v, m_JT.m_Jvr2, m_Ar2r2);
  LA::SymmetricMatrix3x3f::AddABTTo(m_JTWr2p, m_JT.m_Jpr2, m_Ar2r2);
  LA::AlignedMatrix3x3f::ABT(m_JTWr2v, m_JT.m_Jvv2, m_Ar2v2);
  LA::AlignedMatrix3x3f::Ab(m_JTWr2r, e.m_er, m_br2);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWr2v, e.m_ev, m_br2);
  LA::AlignedMatrix3x3f::AddAbTo(m_JTWr2p, e.m_ep, m_br2);

  LA::SymmetricMatrix3x3f::ABT(m_JTWv2v, m_JT.m_Jvv2, m_Av2v2);
  LA::AlignedMatrix3x3f::Ab(m_JTWv2v, e.m_ev, m_bv2);

  m_Aba2ba2 = m_W.m_wba;
  m_Abw2bw2 = m_W.m_wbw;

  LA::AlignedMatrix2x3f::ABT(m_JTWgv, J.m_JvgT, m_Agg);
  LA::AlignedMatrix2x3f::AddABTTo(m_JTWgp, J.m_JpgT, m_Agg);
  m_JTWgp.GetScaled(Jpv1, m_Agv1);
  m_Agv1 -= m_JTWgv;
  LA::AlignedMatrix2x3f::ABT(m_JTWgv, m_JT.m_Jvba1, m_Agba1);
  LA::AlignedMatrix2x3f::AddABTTo(m_JTWgp, m_JT.m_Jpba1, m_Agba1);
  LA::AlignedMatrix2x3f::ABT(m_JTWgv, m_JT.m_Jvbw1, m_Agbw1);
  LA::AlignedMatrix2x3f::AddABTTo(m_JTWgp, m_JT.m_Jpbw1, m_Agbw1);
  //LA::AlignedMatrix2x3f::ABT(m_JTWgp, m_JT.m_Jpp2, m_Agp2);
  m_Agp2 = m_JTWgp;
  LA::AlignedMatrix2x3f::ABT(m_JTWgv, m_JT.m_Jvr2, m_Agr2);
  LA::AlignedMatrix2x3f::AddABTTo(m_JTWgp, m_JT.m_Jpr2, m_Agr2);
  LA::AlignedMatrix2x3f::ABT(m_JTWgv, m_JT.m_Jvv2, m_Agv2);
  LA::AlignedMatrix2x3f::Ab(m_JTWgv, e.m_ev, m_bg);
  LA::AlignedMatrix2x3f::AddAbTo(m_JTWgp, e.m_ep, m_bg);
#endif
}
//如果有imu数据(已经被转到了左相机坐标系(Rc0_i*)下的观测),就用加速度数据求出东北天坐标系下左相机的一个初始朝向
void InitializeCamera(const AlignedVector<Measurement> &us, Camera &C) {
  const int N = us.Size();
  if (N == 0 || IMU_GRAVITY_EXCLUDED) {//如果没有imu数据或者不用重力
    C.MakeIdentity();
  } else {
    LA::AlignedVector3f g;//加速度测量
    g.MakeZero();
    for (int i = 0; i < N; ++i) {//这里需要静止才可以
      g += us[i].m_a;
    }
    g.Normalize();//对加速度数据进行归一化
    g.MakeMinus();//-g
    C.MakeIdentity(&g);//JPL的四元数,所以是Rc0w，这里tc0w是0
  }
}

void PreIntegrate(const AlignedVector<Measurement> &us/*当前帧和之间帧之间的所有imu测量*/, const float t1/*上一帧的时间戳*/, const float t2/*当前帧的时间戳*/,
                  const Camera &C1/*上一帧的状态*/, Delta *D, AlignedVector<float> *work, const bool jac,
                  const Measurement *u1/*上一帧最后一个imu测量*/, const Measurement *u2, const float eps) {
#ifdef CFG_DEBUG
  UT_ASSERT(imu_measures.Empty() || (imu_measures.Front().t() >= t1 && imu_measures.Back().t() <= t2));
#endif
  PreIntegrate(us.Data()/*当前帧和之间帧之间的所有imu测量*/, us.Size()/*imu测量的数量*/, t1/*上一帧的时间戳*/, t2/*当前帧的时间戳*/, C1/*上一帧的状态*/, D/*预计分的pvq*/, work, jac/*是否要输出雅克比*/, u1/*上一帧最后一个imu测量*/, u2, eps);
}

void PreIntegrate(const Measurement *us/*当前帧和之间帧之间的所有imu测量*/, const int N/*imu测量的数量*/, const float t1/*上一帧的时间戳*/, const float t2/*当前帧的时间戳*/,
                  const Camera &C1/*上一帧的状态*/, Delta *D, AlignedVector<float> *work, const bool jac/*是否要输出雅克比*/,
                  const Measurement *u1/*上一帧(有u2时就是上上帧)最后一个imu测量*/, const Measurement *u2/*当前帧第一个imu测量*/, const float eps) {
#ifdef CFG_DEBUG
  if (u1) {
    UT_ASSERT(u1->Valid() && u1->t() <= t1);
  }
  if (u2) {
    UT_ASSERT(u2->Valid() && u2->t() >= t2);
  }
#endif
  if (u1) {//如果给了上一帧最后一个imu数据,就记录一下
    D->m_u1 = *u1;
  } else {
    D->m_u1.Invalidate();
  }
  if (u2) {
    D->m_u2 = *u2;
  } else {
    D->m_u2.Invalidate();
  }
  D->m_ba = C1.m_ba;
  D->m_bw = C1.m_bw;

  D->m_v.MakeZero();
  D->m_p.MakeZero();
  D->m_Jvba.MakeZero();
  D->m_Jvbw.MakeZero();
  D->m_Jpba.MakeZero();
  D->m_Jpbw.MakeZero();

  int r1 = 0, r2 = 1;
  Quaternion dq, q;
  Rotation3D dR, RT[2];
  LA::AlignedMatrix3x3f Jrbw[2];
  q.MakeIdentity();
  RT[r1].MakeIdentity();
  RT[r2].MakeIdentity();
  Jrbw[r1].MakeZero();
  Jrbw[r2].MakeZero();

#ifdef CFG_IMU_FULL_COVARIANCE
  Delta::Transition F;//传播矩阵
  Delta::Covariance P[2];//协方差
  Delta::Covariance::DD U;
  P[r1].MakeZero();
  P[r2].MakeZero();
#endif

  float Tpg = 0.0f;
  float _t1/*前一次的imu测量的数据的时间戳*/, _t2;
  LA::AlignedVector3f a, adt, w, wdt;
  LA::AlignedMatrix3x3f Jr[2], Jrdt;
  const xp128f s = xp128f::get(0.5f);
  for (int i1 = -1, i2 = 0; i1 < N; i1 = i2++)
  {
    if (i1 != -1)
    {//i1 != -1时就全是当前的imu消息了
      const Measurement &_u1 = us[i1];//
      _t1 = _u1.t();//k时刻imu测量时间戳
      if (i2 != N)
      {
        const Measurement &_u2 = us[i2];
        _t2 = _u2.t();//k+1时刻imu测量时间戳
        a.xyzr() = (_u1.m_a.xyzr() + _u2.m_a.xyzr()) * s;//中值
        w.xyzr() = (_u1.m_w.xyzr() + _u2.m_w.xyzr()) * s;
      } else
      {
          _t2 = t2;
          if (u2)
          {
              Measurement::Interpolate(_u1, *u2, (_t1 + _t2) * 0.5f, a, w);
          } else
          {
              a = _u1.m_a;
              w = _u1.m_w;
          }
      }
    } else
    {
#ifdef CFG_DEBUG
        UT_ASSERT(i2 < N);
#endif
        const Measurement &_u2 = us[i2];//当前帧的第一个imu测量
        _t1 = t1;//k时刻imu测量时间戳
        _t2 = _u2.t();//k+1时刻imu测量时间戳
        if (u1)
        {
            Measurement::Interpolate(*u1/*上一帧最后一个imu测量*/, _u2/*当前帧的第一个imu测量*/, (_t1 + _t2) * 0.5f/*中间时间值*/, a, w);//a,w对应于k和k+1中间时刻的imu测量
        } else
        {
            a = _u2.m_a;
            w = _u2.m_w;
        }
    }
    if (_t1 == _t2)
    {
      continue;
    }
    a -= D->m_ba;//减去bias的影响
    w -= D->m_bw;
    const xp128f dt = xp128f::get(_t2 - _t1);//delta_t
    const xp128f dT = xp128f::get(_t1 - t1/*上一帧的时间戳*/);
    const xp128f dt_2 = xp128f::get(dt[0] * 0.5f);//0.5*delta_t
    Tpg = (dt_2[0] + (_t1 - t1)) * dt[0] + Tpg;// Tpg += (0.5dt + dt0)dt 就是0.5 (t2-t1)^2
//预计分pvq
    w.GetScaled(dt, wdt);//wdt = (w-bw)*delta_t
    //dR.SetRodrigues(wdt);
    //Rotation3D::ABT(dR, RT[r1], D->m_R);
    dq.SetRodrigues(wdt, eps);//so3 转四元数
    dR.SetQuaternion(dq);//dRc0k+1_c0k
    q = dq * q;//因为是JPL形式,所以是反着的 qc0k+1_c0i = qc0k+1_c0k* qc0k_c0i
    RT[r2].SetQuaternion(q);//Rc0k+1_c0i
    RT[r2].Transpose();//Rc0i_c0k+1

    const LA::AlignedMatrix3x3f RTdt = (RT[r1] + RT[r2]) * dt_2;//0.5*(Rc0i_c0k + Rc0i_c0k+1) *dt，为啥这里直接加了?这么加能满足正交性么
    const LA::AlignedVector3f dv = RTdt * a;//vc0k_c0k+1 = Rc0i_c0mid * (ac0mid-ba) * dt
    const LA::AlignedVector3f dp = D->m_v * dt + dv * dt_2;//pc0k_c0k+1 =vc0i_c0k * dt + 0.5 * ac0i * dt^2
    D->m_v += dv;
    D->m_p += dp;
//#ifdef CFG_DEBUG
#if 0
    if (UT::Debugging()) {
      UT::Print("%d %d\n", i1, i2);
      //(D->m_v * (1.0f / (_t2 - t1))).Print();
      D->m_v.Print();
      D->m_p.Print();
    }
#endif
    //////R部分的不确定性传播过程
    // 首先是旋转的传播过程 Rci_cj = ∏(下标k=i,上标j-1)exp[(w_mid -bw_k)*dt)]x
    //   如果给bw的bias一个扰动的话 = ∏(下标k=i,上标j-1)exp[(w_mid -bw_k -d_bw_k)*dt)]x
      //= ∏(下标k=i,上标j-1){exp[(w_mid -bw_k)*dt]x *exp[-Jr[(w_mid -bw_k)*dt]*d_bw_k*dt]x }
      //= &Rij* ∏(下标k=i,上标j-1){exp[ -&Rk+1_j.t * Jr[(w_mid -bw_k)*dt] * d_bw_k*dt]x } 其中&Rij代表 ∏(下标k=i,上标j-1){exp[(w_mid -bw_k)*dt]x}
   // so3下的 d(theta)i_j = ∑(下标k=i,上标j-1){-&Rk+1_j.t * Jr[(w_mid -bw_k)*dt] * d_bw_k*dt}
   // 李代数下的旋转误差传逆: d(theta)i_k+1 = &Rk_k+1.t(这个就是用w积分出的旋转dR) * d(theta)i_k + - Jr[(w_mid -bw_k)*dt] * dt * d_bw_k
  //////P,V部分的雅克比推导比较常规，就不打了,但我求的有两处对不上
    //
    if (jac)//计算一下误差传递矩阵
    {
      Rotation3D::GetRodriguesJacobian(wdt/*delta_th*/, Jrdt, eps);//算了一下右乘雅克比
      Jrdt *= dt;
      Jrbw[r2] = dR * Jrbw[r1] - Jrdt;//Jk+1 = Fk*Jk 这里是递推得到预积分后状态量对i时刻bw的雅克比
#if 0
      if (UT::Debugging()) {
        UT::PrintSeparator();
        Jrdt.Print("Jrdt = ", true);
        Jrbw[r2].Print("Jrbw = ", true);
      }
#endif
      const LA::AlignedMatrix3x3f dJvba = RTdt.GetMinus();
      const LA::AlignedMatrix3x3f dJpba = D->m_Jvba * dt + dJvba * dt_2;
      D->m_Jvba += dJvba;//d_v/d_ba = - Rc0i_c0mid * dt
      D->m_Jpba += dJpba;//d_p/d_ba = (d_v/d_ba) + -0.5 Rc0i_c0mid * dt^2
      a.GetScaled(dt_2, adt);//adt = 0.5 * (ac0mid-ba) * t
      SkewSymmetricMatrix::ABT(RT[r1], adt, Jr[r1]);//r1就是之前的状态 Jr[r1] = Rc0i_c0k * [-0.5*(ac0mid-ba)*t]x
      SkewSymmetricMatrix::ABT(RT[r2], adt, Jr[r2]);//Jr[r2] = Rc0i_c0k+1 * [-0.5*(ac0mid-ba)*t]x
      const LA::AlignedMatrix3x3f dJvbw = (Jr[r1] * Jrbw[r1]) + (Jr[r2] * Jrbw[r2]);//这里为啥是Jr[r2] * Jrbw[r2]而不是Jr[r2]*Jrdt
      const LA::AlignedMatrix3x3f dJpbw = D->m_Jvbw * dt + dJvbw * dt_2;
      D->m_Jvbw += dJvbw;
      D->m_Jpbw += dJpbw;

#ifdef CFG_IMU_FULL_COVARIANCE
      if (i1 != -1)
      {
        RT[r2].GetScaled(dt, F.m_Fdb.m_Frbw);
        F.m_Fdb.m_Frbw.MakeMinus();
        dv.GetMinus(F.m_Fdd.m_Fvr);
        F.m_Fdb.m_Fvba = dJvba;
        F.m_Fdb.m_Fvbw = dJvbw;
        dp.GetMinus(F.m_Fdd.m_Fpr);
        F.m_Fdd.m_Fpv = dt;
        F.m_Fdb.m_Fpba = dJpba;
        F.m_Fdb.m_Fpbw = dJpbw;
        Delta::Covariance::FPFT(F/*传递矩阵*/, P[r1]/*上个时刻协方差*/, &U, &P[r2]/*当前时刻协方差*/);//计算下一次的斜方差P[r2] = FP[r1]F.t
      }//前面算了状态量误差的传播,现在把噪声的传播也加上
      const float s2r = dt[0] * IMU_VARIANCE_GYROSCOPE_NOISE;
      const float s2v = dt[0] * IMU_VARIANCE_ACCELERATION_NOISE;
      const float s2p = dt[0] * dt_2[0] * s2v;
      const float s2ba = dt[0] * IMU_VARIANCE_ACCELERATION_BIAS_WALK;
      const float s2bw = dt[0] * IMU_VARIANCE_GYROSCOPE_BIAS_WALK;
      P[r2].IncreaseDiagonal(s2r, s2v, s2p, s2ba, s2bw);
//#ifdef CFG_DEBUG
#if 0
      static FILE *fp;
      if (UT::Debugging()) {
        UT::DebugStop();
        fp = fopen("D:/tmp/P.txt", "rb");
      }
      //UT::Print("%d %f\t", i2, P[r2].m_Pdd.m_Prr.m00());
      const Quaternion e_dq = UT::LoadB<Quaternion>(fp);
      const Rotation3D e_dR = UT::LoadB<Rotation3D>(fp);
      const Rotation3D e_R = UT::LoadB<Rotation3D>(fp);
      e_dq.AssertEqual(dq, 1, UT::String("dq[%d]", i2));
      e_dR.AssertEqual(dR, 1, UT::String("dR[%d]", i2));
      e_R.AssertEqual(D->m_R, 1, UT::String("R[%d]", i2));
      const Delta::EigenTransition e_F = UT::LoadB<Delta::EigenTransition>(fp);
      const Delta::EigenCovariance e_P = UT::LoadB<Delta::EigenCovariance>(fp);
      e_F.AssertEqual(F, 1, UT::String("F[%d]", i2));
      e_P.AssertEqual(P[r2], 1, UT::String("P[%d]", i2));
      if (i2 == N) {
        //UT::Print("\n");
        fclose(fp);
      }
#endif
#endif
    }

    r1 = r2;//置换一下,这样[2]就可以做循环预积分了
    r2 = 1 - r2;
  }
  D->m_RT = RT[r1];//这里实际是最后一次循环r2的位置
  D->m_Jrbw = Jrbw[r1];//保存一下递推得到的雅克比,方便后面使用
  D->m_Tvg = D->m_Tpv = t2 - t1;
  D->m_Tpg = Tpg;
  //D->m_Tpg = 0.5f * D.m_Tvg * D.m_Tvg;
  if (jac) {
#ifdef CFG_IMU_FULL_COVARIANCE
    P[r1].IncreaseDiagonal(IMU_VARIANCE_EPSILON_ROTATION,
                           IMU_VARIANCE_EPSILON_VELOCITY,
                           IMU_VARIANCE_EPSILON_POSITION,
                           IMU_VARIANCE_EPSILON_BIAS_ACCELERATION,
                           IMU_VARIANCE_EPSILON_BIAS_GYROSCOPE);
    D->m_W.Set(P[r1], work);//存储的顺序是R,v,p,ba,bw用协方差的逆设置一下信息矩阵
//#ifdef CFG_DEBUG
#if 0
    FILE *fp = fopen("D:/tmp/W.txt", "rb");
    const Delta::EigenCovariance e_P = UT::LoadB<Delta::EigenCovariance>(fp);
    const Delta::EigenWeight e_W = UT::LoadB<Delta::EigenWeight>(fp);
    e_P.AssertEqual(P[r1], 1, "P");
    //e_P.Get(&P[r1]);
    //D->m_W.Set(P[r1], work);
    e_W.AssertEqual(D->m_W, 1, "W");
    fclose(fp);
#endif
    const float w[5] = {IMU_WEIGHT_ROTATION, IMU_WEIGHT_VELOCITY, IMU_WEIGHT_POSITION,
                        IMU_WEIGHT_BIAS_ACCELERATION, IMU_WEIGHT_BIAS_GYROSCOPE};
    for (int i = 0; i < 5; ++i) {
      D->m_W[i][i] *= w[i];
      for (int j = i + 1; j < 5; ++j) {
        D->m_W[i][j] *= sqrtf(w[i] * w[j]);
        D->m_W[i][j].GetTranspose(D->m_W[j][i]);
      }
    }
#if 0
    UT::DebugStart();
    for (int i = 0; i < 5; ++i) {
      D->m_W[i][i].MakeDiagonal(1.0e7f);
      for (int j = i + 1; j < 5; ++j) {
        D->m_W[i][j].MakeZero();
        D->m_W[j][i].MakeZero();
      }
    }
    UT::DebugStop();
#endif
#else
    LA::AlignedMatrix3x3f JS;
    LA::SymmetricMatrix3x3f Sr, Sv, Sp;
    const xp128f s2na = xp128f::get(IMU_VARIANCE_ACCELERATION_NOISE);
    const xp128f s2nw = xp128f::get(IMU_VARIANCE_GYROSCOPE_NOISE);
    LA::SymmetricMatrix3x3f::sAAT(s2nw, D->m_Jrbw, JS, Sr);
    Sr.IncreaseDiagonal(IMU_VARIANCE_EPSILON_ROTATION);
    LA::SymmetricMatrix3x3f::sAAT(s2na, D->m_Jvba, JS, Sv);
    LA::SymmetricMatrix3x3f::AddsAATTo(s2nw, D->m_Jvbw, JS, Sv);
    Sv.IncreaseDiagonal(IMU_VARIANCE_EPSILON_VELOCITY);
    LA::SymmetricMatrix3x3f::sAAT(s2na, D->m_Jpba, JS, Sp);
    LA::SymmetricMatrix3x3f::AddsAATTo(s2nw, D->m_Jpbw, JS, Sp);
    Sp.IncreaseDiagonal(IMU_VARIANCE_EPSILON_POSITION);
    Sr *= UT::Inverse(IMU_WEIGHT_ROTATION);
    Sv *= UT::Inverse(IMU_WEIGHT_VELOCITY);
    Sp *= UT::Inverse(IMU_WEIGHT_POSITION);
    if (!Sr.GetInverse(D->m_W.m_Wr, 0.0f)) {
      D->m_W.m_Wr.MakeZero();
    }
    if (!Sv.GetInverse(D->m_W.m_Wv, 0.0f)) {
      D->m_W.m_Wv.MakeZero();
    }
    if (!Sp.GetInverse(D->m_W.m_Wp, 0.0f)) {
      D->m_W.m_Wp.MakeZero();
    }
    const float dT = (t2 - t1), dT2 = dT * dT;
    const float s2ba = IMU_VARIANCE_ACCELERATION_BIAS_WALK * dT2 +
                       IMU_VARIANCE_EPSILON_BIAS_ACCELERATION;
    D->m_W.m_wba = UT::Inverse(s2ba, IMU_WEIGHT_BIAS_ACCELERATION);
    const float s2bw = IMU_VARIANCE_GYROSCOPE_BIAS_WALK * dT2 +
                       IMU_VARIANCE_EPSILON_BIAS_GYROSCOPE;
    D->m_W.m_wbw = UT::Inverse(s2bw, IMU_WEIGHT_BIAS_GYROSCOPE);
//#ifdef CFG_DEBUG
#if 0
    if (UT::Debugging()) {
      UT::PrintSeparator();
      Sr.GetAlignedMatrix3x3f().Print();
      UT::PrintSeparator();
      D->m_W.m_Wr.Print();
    }
#endif
#endif
  }
}
//考虑了bias以后进行pvR传播
void Propagate(const Point3D &pu/*外参tc0_i*/, const Delta &D/*预积分部分*/, const Camera &C1/*上一帧的状态*/, Camera &C2/*当前帧的Twc*/, const float eps) {
  const LA::AlignedVector3f dba = C1.m_ba - D.m_ba;
  const LA::AlignedVector3f dbw = C1.m_bw - D.m_bw;
  const Rotation3D R1T = C1.m_Cam_pose.GetRotationTranspose();//Rwc0i
#ifdef CFG_IMU_FULL_COVARIANCE
  C2.m_Cam_pose = D.GetRotationMeasurement(dbw, eps).GetTranspose() / R1T;//Rc0j_w=Rc0k+1_c0i* Rc0iw
#else
  C2.m_Cam_pose = D.GetRotationMeasurement(dbw, eps) / R1T;
#endif
  const LA::AlignedVector3f dp = D.m_p + D.m_Jpba * dba + D.m_Jpbw * dbw;//考虑bias的影响
  //因为是用c0系来对齐重力系的,所以v和bias都左乘了Rc0i
  //twcj - twci + Rwcj*tc0i=  twbj - twbi + Rwci*tc0i
  //twcj = twbj - twbi + (Rwci-Rwci)*tc0i + twci
  //     = Vw_c0i*dt + -0.5*gw*dt^2 + Rwc0*pc0i_c0j(这个预积分实际上积的还是b的位移,但是是转到了c0系的) + (Rwci-Rwci)*tc0i + twci
  C2.m_p = (R1T - C2.m_Cam_pose.GetRotationTranspose()) * pu 
          +C1.m_p + C1.m_v * D.m_Tpv + R1T.GetApplied(dp);
  if (!IMU_GRAVITY_EXCLUDED) {
    C2.m_p.z() -= IMU_GRAVITY_MAGNITUDE * D.m_Tpg;//考虑重力
  }

  C2.m_Cam_pose.SetPosition(C2.m_p);//存的是tc0w
  const LA::AlignedVector3f dv = D.m_v + D.m_Jvba * dba + D.m_Jvbw * dbw;
  C2.m_v = C1.m_v + R1T.GetApplied(dv);
  if (!IMU_GRAVITY_EXCLUDED) {
    C2.m_v.z() -= IMU_GRAVITY_MAGNITUDE * D.m_Tvg;//vw_j = vw_i − gw*delta_t + Rwc0i * vc0i_c0j
  }
  //////////////////////////////////////////////////////////////////////////
  //C2.m_p = C1.m_p;
  //C2.m_Cam_pose.SetPosition(C2.m_p);
  //C2.m_v.MakeZero();
  //////////////////////////////////////////////////////////////////////////
  C2.m_ba = C1.m_ba;//bias
  C2.m_bw = C1.m_bw;
#ifdef CFG_DEBUG
  C2.m_Cam_pose.AssertOrthogonal();
#endif
}

#ifdef CFG_DEBUG_EIGEN
#define IMU_DELTA_EIGEN_DEBUG_JACOBIAN
void Delta::EigenGetErrorJacobian(const Camera &C1/*前一帧(i)状态*/, const Camera &C2/*后一帧(j)状态*/, const Point3D &pu,/*tc0_i*/
                                  EigenError *e_e, EigenJacobian::Global *e_J,
                                  const float eps) const {
  const EigenVector3f e_ba = EigenVector3f(m_ba), e_bw = EigenVector3f(m_bw);//预积分
  const EigenRotation3D e_RdT = EigenRotation3D(m_RT)/*预积分Rc0i_c0j*/, e_Rd = EigenRotation3D(e_RdT.transpose());/*预积分Rc0j_c0i*/
  const EigenVector3f e_vd = EigenVector3f(m_v)/*预积分vc0i_c0j*/, e_pd = EigenVector3f(m_p);/*预积分pc0i_c0j*/
  const EigenMatrix3x3f e_Jrbw = EigenMatrix3x3f(m_Jrbw);//dr/bw
  const EigenMatrix3x3f e_Jvba = EigenMatrix3x3f(m_Jvba)/*dv/da*/, e_Jvbw = EigenMatrix3x3f(m_Jvbw);/*dv/dw*/
  const EigenMatrix3x3f e_Jpba = EigenMatrix3x3f(m_Jpba)/*dp/da*/, e_Jpbw = EigenMatrix3x3f(m_Jpbw);/*dp/dw*/

  const EigenRotation3D e_R1 = C1.m_Cam_pose,/*Rc0iw*/ e_R1T = EigenRotation3D(e_R1.transpose());//Rwc0i
  const EigenRotation3D e_R2 = C2.m_Cam_pose,/*Rc0jw*/ e_R2T = EigenRotation3D(e_R2.transpose());//Rwc0j
  const EigenPoint3D e_p1 = EigenPoint3D(C1.m_p)/*twc0i*/, e_p2 = EigenPoint3D(C2.m_p);/*twc0j*/
  const EigenVector3f e_v1 = EigenVector3f(C1.m_v)/*Vwc0i*/, e_v2 = EigenVector3f(C2.m_v);/*Vwc0j*/
  const EigenVector3f e_ba1 = EigenVector3f(C1.m_ba)/*ba_wc0i*/, e_ba2 = EigenVector3f(C2.m_ba);/*ba_wc0j*/
  const EigenVector3f e_dba = EigenVector3f(e_ba1 - e_ba);//e_ba = ba_wc0i - m_ba 这里m_ba == ba_wc0i =ba_wc0j
  const EigenVector3f e_bw1 = EigenVector3f(C1.m_bw), e_bw2 = EigenVector3f(C2.m_bw);
  const EigenVector3f e_dbw = EigenVector3f(e_bw1 - e_bw);//e_bw = bw_wc0i - m_bw 这里m_bw == bw_wc0i =bw_wc0j
  const EigenVector3f e_pu = EigenVector3f(pu);/*tc0_i*/

  const EigenRotation3D e_R12 = EigenRotation3D(e_R2 * e_R1T);//e_R12 = Rc0jw * Rwc0i
  const EigenRotation3D e_R21 = EigenRotation3D(e_R12.transpose());//e_R21 = (Rc0jw * Rwc0i).t
  const EigenVector3f e_drbw = EigenVector3f(-e_Jrbw * e_dbw);// - dr/bw * (bw_wc0i - m_bw)
#ifdef CFG_IMU_FULL_COVARIANCE
  //const EigenRotation3D e_eR = EigenRotation3D(e_RdT * EigenRotation3D(e_drbw) * e_R12, eps);
  const EigenRotation3D e_eR = GetRotationMeasurement(C1, eps) / GetRotationState(C1, C2);

//GetRotationMeasurement(C1, eps)是m_RT*exp[-(m_Jrbw*(C1.m_bw - m_bw))]x.t =>
// Rc0i_c0j*exp[(m_Jrbw*(C1.m_bw - m_bw))]x 因为是jpl转的R,所以是exp[-th]
//GetRotationState(C1, C2) 是 Rc0i_w *Rc0j_w.t
// e_eR => Rc0i_c0j*exp[(m_Jrbw*(C1.m_bw - m_bw))]x * ((Rc0i_w *Rc0j_w.t)).t

  #else
  const EigenRotation3D e_eR = EigenRotation3D(e_R12 * e_RdT * EigenRotation3D(e_drbw));
#endif
  const EigenVector3f e_er = e_eR.GetRodrigues(eps);
  const EigenMatrix3x3f e_JrI = EigenRotation3D::GetRodriguesJacobianInverse(e_er, eps);
#ifdef CFG_IMU_FULL_COVARIANCE
  const EigenMatrix3x3f e_Jrr2 = EigenMatrix3x3f(e_JrI * e_eR * e_R1);
#else
  const EigenMatrix3x3f e_Jrr2 = EigenMatrix3x3f(e_JrI * e_R2);
#endif
  const EigenMatrix3x3f e_Jrr1 = EigenMatrix3x3f(-e_Jrr2);
#ifdef CFG_IMU_FULL_COVARIANCE
  const EigenMatrix3x3f e_Jrbw1 = EigenMatrix3x3f(-e_JrI * e_RdT *
                                                  EigenRotation3D::GetRodriguesJacobian(e_drbw, eps) *
                                                  e_Jrbw);
#else
  const EigenMatrix3x3f e_Jrbw1 = EigenMatrix3x3f(-e_JrI * e_R12 * e_RdT *
                                                  EigenRotation3D::GetRodriguesJacobian(e_drbw, eps) *
                                                  e_Jrbw);
#endif

  EigenVector3f e_v12 = EigenVector3f(e_v2 - e_v1);
  if (!IMU_GRAVITY_EXCLUDED) {
    e_v12.z() += IMU_GRAVITY_MAGNITUDE * m_Tvg;
  }
  const EigenMatrix3x3f e_Jvr1 = EigenMatrix3x3f(e_R1 * EigenSkewSymmetricMatrix(e_v12));
  e_v12 = EigenVector3f(e_R1 * e_v12);
  const EigenVector3f e_ev = EigenVector3f(e_v12 - (e_vd + e_Jvba * e_dba + e_Jvbw * e_dbw));
  const EigenMatrix3x3f e_Jvv1 = EigenMatrix3x3f(-e_R1);
  const EigenMatrix3x3f e_Jvba1 = EigenMatrix3x3f(-e_Jvba);
  const EigenMatrix3x3f e_Jvbw1 = EigenMatrix3x3f(-e_Jvbw);
  const EigenMatrix3x3f e_Jvv2 = e_R1;

  EigenVector3f e_p12 = EigenVector3f(e_R2T * e_pu + e_p2 - e_p1 - e_v1 * m_Tpv);
  if (!IMU_GRAVITY_EXCLUDED) {
    e_p12.z() += IMU_GRAVITY_MAGNITUDE * m_Tpg;
  }
  const EigenMatrix3x3f e_Jpr1 = EigenMatrix3x3f(e_R1 * EigenSkewSymmetricMatrix(e_p12));
  e_p12 = EigenVector3f(e_R1 * e_p12);
  const EigenVector3f e_ep = EigenVector3f(-e_pu + e_p12 - (e_pd + e_Jpba * e_dba +
                                                                   e_Jpbw * e_dbw));
  const EigenMatrix3x3f e_Jpp1 = EigenMatrix3x3f(-e_R1);
  const EigenMatrix3x3f e_Jpv1 = EigenMatrix3x3f(-e_R1 * m_Tpv);
  const EigenMatrix3x3f e_Jpba1 = EigenMatrix3x3f(-e_Jpba);
  const EigenMatrix3x3f e_Jpbw1 = EigenMatrix3x3f(-e_Jpbw);
  const EigenMatrix3x3f e_Jpp2 = e_R1;
  const EigenMatrix3x3f e_Jpr2 = EigenMatrix3x3f(-e_R1 * EigenSkewSymmetricMatrix(
                                                 EigenVector3f(e_R2T * e_pu)));

  const EigenVector3f e_eba = EigenVector3f(e_ba1 - e_ba2);
  const EigenMatrix3x3f e_Jbaba1 = EigenMatrix3x3f::Identity();
  const EigenMatrix3x3f e_Jbaba2 = EigenMatrix3x3f(-EigenMatrix3x3f::Identity());

  const EigenVector3f e_ebw = EigenVector3f(e_bw1 - e_bw2);
  const EigenMatrix3x3f e_Jbwbw1 = EigenMatrix3x3f::Identity();
  const EigenMatrix3x3f e_Jbwbw2 = EigenMatrix3x3f(-EigenMatrix3x3f::Identity());

#ifdef IMU_DELTA_EIGEN_DEBUG_JACOBIAN
  const float e_dr1Max = 0.1f;
  const float e_dr2Max = 0.1f;
  const float e_dp1Max = 0.01f;
  const float e_dp2Max = 0.01f;
  const float e_dv1Max = 0.01f;
  const float e_dv2Max = 0.01f;
  const float e_dba1Max = 0.01f;
  const float e_dbw1Max = 0.01f;
  //const float e_dr1Max = 0.0f;
  //const float e_dr2Max = 0.0f;
  //const float e_dp1Max = 0.0f;
  //const float e_dp2Max = 0.0f;
  //const float e_dv1Max = 0.0f;
  //const float e_dv2Max = 0.0f;
  //const float e_dba1Max = 0.0f;
  //const float e_dbw1Max = 0.0f;
#if 0
  EigenVector3f e_dr1, e_dr2;
  EigenRotation3D e_R1GT, e_R2GT;
  if (e_dr1Max == 0.0f && e_dr2Max == 0.0f) {
    e_dr1.setZero();
    e_R1GT = e_R1;
    e_dr2.setZero();
    e_R2GT = e_R2;
  } else if (e_dr1Max == 0.0f) {
    e_dr1.setZero();
    e_R1GT = e_R1;
    e_R2GT = e_Rd * e_R1GT;
    e_dr2 = EigenRotation3D(e_R2GT * e_R2.transpose()).GetRodrigues(eps);
  } else if (e_dr2Max == 0.0f) {
    e_dr2.setZero();
    e_R2GT = e_R2;
    e_R1GT = e_RdT * e_R2GT;
    e_dr1 = EigenRotation3D(e_R1GT * e_R1T).GetRodrigues(eps);
  } else {
    e_dr1 = EigenAxisAngle::GetRandom(e_dr1Max * UT_FACTOR_DEG_TO_RAD).GetRodrigues(eps);
    e_R1GT = EigenMatrix3x3f(EigenRotation3D(e_dr1) * e_R1);
    e_R2GT = e_Rd * e_R1GT;
    e_dr2 = EigenRotation3D(e_R2GT * e_R2.transpose()).GetRodrigues(eps);
  }
#else
  const EigenVector3f e_dr1 = EigenVector3f::GetRandom(e_dr1Max * UT_FACTOR_DEG_TO_RAD);
  const EigenVector3f e_dr2 = EigenVector3f::GetRandom(e_dr2Max * UT_FACTOR_DEG_TO_RAD);
  const EigenRotation3D e_R1GT = EigenRotation3D(e_R1 * EigenRotation3D(e_dr1));
  const EigenRotation3D e_R2GT = EigenRotation3D(e_R2 * EigenRotation3D(e_dr2));
#endif
  const EigenVector3f e_dp1 = EigenVector3f::GetRandom(e_dp1Max);
  const EigenVector3f e_dp2 = EigenVector3f::GetRandom(e_dp2Max);
  const EigenPoint3D e_p1GT = EigenVector3f(e_p1 + e_dp1);
  const EigenPoint3D e_p2GT = EigenVector3f(e_p2 + e_dp2);

  const EigenVector3f e_dv1 = EigenVector3f::GetRandom(e_dv1Max);
  const EigenVector3f e_dv2 = EigenVector3f::GetRandom(e_dv2Max);
  const EigenVector3f e_v1GT = EigenVector3f(e_v1 + e_dv1);
  const EigenVector3f e_v2GT = EigenVector3f(e_v2 + e_dv2);

  const EigenVector3f e_dba1 = EigenVector3f::GetRandom(e_dba1Max);
  const EigenVector3f e_ba1GT = EigenVector3f(e_ba1 + e_dba1);
  const EigenVector3f e_dbaGT = EigenVector3f(e_ba1GT - e_ba);

  const EigenVector3f e_dbw1 = EigenVector3f::GetRandom(e_dbw1Max * UT_FACTOR_DEG_TO_RAD);
  const EigenVector3f e_bw1GT = EigenVector3f(e_bw1 + e_dbw1);
  const EigenVector3f e_dbwGT = EigenVector3f(e_bw1GT - e_bw);

  const EigenVector3f e_drbwGT = EigenVector3f(-e_Jrbw * e_dbwGT);
#ifdef CFG_IMU_FULL_COVARIANCE
  const EigenVector3f e_erGT = EigenRotation3D(e_RdT * EigenRotation3D(e_drbwGT) *
                                               e_R2GT * e_R1GT.GetTranspose()).GetRodrigues(eps);
#else
  const EigenVector3f e_erGT = EigenRotation3D(e_R2GT * e_R1GT.GetTranspose() * e_RdT *
                                               EigenRotation3D(e_drbwGT)).GetRodrigues(eps);
#endif

  const EigenVector3f e_er1 = EigenVector3f(e_er - e_erGT);
  const EigenVector3f e_er2 = EigenVector3f(e_er1 + e_Jrr1 * e_dr1 + e_Jrr2 * e_dr2 +
                                                    e_Jrbw1 * e_dbw1);
  const float eps = 1.0e-3f;
  UT::AssertReduction(e_er1, e_er2, 1, "er", eps);
//#ifdef CFG_DEBUG
#if 0
  const EigenVector3f e_er2r1 = EigenVector3f(e_er1 + e_Jrr1 * e_dr1);
  UT::AssertReduction(e_er1, e_er2r1, 1, "err1");
  const EigenVector3f e_er2r2 = EigenVector3f(e_er1 + e_Jrr2 * e_dr2);
  UT::AssertReduction(e_er1, e_er2r2, 1, "err2");
  const EigenVector3f e_er2bw1 = EigenVector3f(e_er1 + e_Jrbw1 * e_dbw1);
  UT::AssertReduction(e_er1, e_er2bw1, 1, "erbw1");
#endif
  
  e_v12 = EigenVector3f(e_v2GT - e_v1GT);
  if (!IMU_GRAVITY_EXCLUDED)
    e_v12.z() += IMU_GRAVITY_MAGNITUDE * m_Tvg;
  const EigenVector3f e_evGT = EigenVector3f(e_R1GT * e_v12 - (e_vd + e_Jvba * e_dbaGT +
                                                               e_Jvbw * e_dbwGT));
  const EigenVector3f e_ev1 = EigenVector3f(e_ev - e_evGT);
  const EigenVector3f e_ev2 = EigenVector3f(e_ev1 + e_Jvr1 * e_dr1 + e_Jvv1 * e_dv1 +
                                            e_Jvv2 * e_dv2 + e_Jvba1 * e_dba1 + e_Jvbw1 * e_dbw1);
  UT::AssertReduction(e_ev1, e_ev2);

  e_p12 = EigenVector3f(e_R2GT.transpose() * e_pu + e_p2GT - e_p1GT - e_v1GT * m_Tpv);
  if (!IMU_GRAVITY_EXCLUDED)
    e_p12.z() += IMU_GRAVITY_MAGNITUDE * m_Tpg;
  const EigenVector3f e_epGT = EigenVector3f(-e_pu + e_R1GT * e_p12 - (e_pd + e_Jpba * e_dbaGT +
                                                                              e_Jpbw * e_dbwGT));
  const EigenVector3f e_ep1 = EigenVector3f(e_ep - e_epGT);
  const EigenVector3f e_ep2 = EigenVector3f(e_ep1 + e_Jpr1 * e_dr1 + e_Jpp1 * e_dp1 +
                                                    e_Jpv1 * e_dv1 + e_Jpr2 * e_dr2 +
                                                    e_Jpp2 * e_dp2 + e_Jpba1 * e_dba1 +
                                                    e_Jpbw1 * e_dbw1);
  UT::AssertReduction(e_ep1, e_ep2);
#endif
  e_e->m_er = e_er;
  e_J->m_Jr.setZero();
  e_J->m_Jr.block<3, 3>(0, 3) = e_Jrr1;
  e_J->m_Jr.block<3, 3>(0, 12) = e_Jrbw1;
  e_J->m_Jr.block<3, 3>(0, 18) = e_Jrr2;
  e_e->m_ev = e_ev;
  e_J->m_Jv.setZero();
  e_J->m_Jv.block<3, 3>(0, 3) = e_Jvr1;
  e_J->m_Jv.block<3, 3>(0, 6) = e_Jvv1;
  e_J->m_Jv.block<3, 3>(0, 9) = e_Jvba1;
  e_J->m_Jv.block<3, 3>(0, 12) = e_Jvbw1;
  e_J->m_Jv.block<3, 3>(0, 21) = e_Jvv2;
  e_e->m_ep = e_ep;
  e_J->m_Jp.setZero();
  e_J->m_Jp.block<3, 3>(0, 0) = e_Jpp1;
  e_J->m_Jp.block<3, 3>(0, 3) = e_Jpr1;
  e_J->m_Jp.block<3, 3>(0, 6) = e_Jpv1;
  e_J->m_Jp.block<3, 3>(0, 9) = e_Jpba1;
  e_J->m_Jp.block<3, 3>(0, 12) = e_Jpbw1;
  e_J->m_Jp.block<3, 3>(0, 15) = e_Jpp2;
  e_J->m_Jp.block<3, 3>(0, 18) = e_Jpr2;
  e_e->m_eba = e_eba;
  e_J->m_Jba.setZero();
  e_J->m_Jba.block<3, 3>(0, 9) = e_Jbaba1;
  e_J->m_Jba.block<3, 3>(0, 24) = e_Jbaba2;
  e_e->m_ebw = e_ebw;
  e_J->m_Jbw.setZero();
  e_J->m_Jbw.block<3, 3>(0, 12) = e_Jbwbw1;
  e_J->m_Jbw.block<3, 3>(0, 27) = e_Jbwbw2;
#ifdef CFG_DEBUG
  Error e;
  Jacobian::Global J;
  GetErrorJacobian(C1, C2, pu, &e, &J, eps);
  e_e->AssertEqual(e);
  e_J->AssertEqual(J);
  e_e->Set(e);
  e_J->Set(J);
#endif
}

void Delta::EigenGetErrorJacobian(const Camera &C1/*最老帧状态*/, const Camera &C2/*次老帧(j)状态*/, const Point3D &pu/*tc0_i*/,
                                  const Rotation3D &Rg/*kf的Rc0w*/, EigenError *e_e,
                                  EigenJacobian::RelativeLF *e_J, const float eps) const {
  const EigenVector3f e_ba = EigenVector3f(m_ba), e_bw = EigenVector3f(m_bw);//预积分
  const EigenRotation3D e_RdT = EigenRotation3D(m_RT)/*预积分Rc0i_c0j*/, e_Rd = EigenRotation3D(e_RdT.transpose());/*预积分Rc0j_c0i*/
  const EigenVector3f e_vd = EigenVector3f(m_v)/*预积分vc0i_c0j*/, e_pd = EigenVector3f(m_p);/*预积分pc0i_c0j*/
    const EigenMatrix3x3f e_Jrbw = EigenMatrix3x3f(m_Jrbw);//dr/bw
    const EigenMatrix3x3f e_Jvba = EigenMatrix3x3f(m_Jvba)/*dv/da*/, e_Jvbw = EigenMatrix3x3f(m_Jvbw);/*dv/dw*/
    const EigenMatrix3x3f e_Jpba = EigenMatrix3x3f(m_Jpba)/*dp/da*/, e_Jpbw = EigenMatrix3x3f(m_Jpbw);/*dp/dw*/

  const Rotation3D R1 = Rotation3D(C1.m_Cam_pose) / Rg/*Rcick = Rciw*Rckw.t*/, R2 = Rotation3D(C2.m_Cam_pose) / Rg;/*Rcjck = Rcjw*Rckw.t*/
  const Point3D p1 = Rg.GetApplied(C1.m_p)/*Rckw*twci*/, p2 = Rg.GetApplied(C2.m_p);/*Rckw*twcj*/
  const Point3D v1 = C1.m_Cam_pose.GetAppliedRotation(C1.m_v)/*Rciw*vwi*/, v2 = C2.m_Cam_pose.GetAppliedRotation(C2.m_v);/*Rcjw*vwj*/
  const EigenRotation3D e_Rg = EigenRotation3D(Rg)/*Rckw*/, e_RgT = EigenRotation3D(e_Rg.transpose());/*Rckw.t*/
  const EigenVector3f e_g(0.0f, 0.0f, IMU_GRAVITY_EXCLUDED ? 0.0f : -IMU_GRAVITY_MAGNITUDE);//重力在东北天的方向
  const EigenRotation3D e_R1 = EigenRotation3D(R1), e_R1T = EigenRotation3D(e_R1.transpose());
  const EigenRotation3D e_R2 = EigenRotation3D(R2), e_R2T = EigenRotation3D(e_R2.transpose());
  const EigenPoint3D e_p1 = EigenPoint3D(p1), e_p2 = EigenPoint3D(p2);
  const EigenVector3f e_v1 = EigenVector3f(v1), e_v2 = EigenVector3f(v2);
  const EigenVector3f e_ba1 = EigenVector3f(C1.m_ba), e_ba2 = EigenVector3f(C2.m_ba);
  const EigenVector3f e_dba = EigenVector3f(e_ba1 - e_ba);
  const EigenVector3f e_bw1 = EigenVector3f(C1.m_bw), e_bw2 = EigenVector3f(C2.m_bw);
  const EigenVector3f e_dbw = EigenVector3f(e_bw1 - e_bw);
  const EigenVector3f e_pu = EigenVector3f(pu);


  const EigenRotation3D e_R12 = EigenRotation3D(e_R2 * e_R1T);////e_R12 = Rcjck * Rckci
  const EigenRotation3D e_R21 = EigenRotation3D(e_R12.transpose());
  const EigenVector3f e_drbw = EigenVector3f(-e_Jrbw * e_dbw);
#ifdef CFG_IMU_FULL_COVARIANCE
  //const EigenRotation3D e_eR = EigenRotation3D(e_RdT * EigenRotation3D(e_drbw) * e_R12, eps);
  const EigenRotation3D e_eR = GetRotationMeasurement(C1, eps) / GetRotationState(C1, C2);//m_RT*exp[-(m_Jrbw*(C1.m_bw - m_bw))]x.t
#else //e_eR => Rc0i_c0j*exp[(m_Jrbw*(C1.m_bw - m_bw))]x * (Rc0j_w *Rc0i_w.t)

    const EigenRotation3D e_eR = EigenRotation3D(e_R12 * e_RdT * EigenRotation3D(e_drbw), eps);
#endif
  const EigenVector3f e_er = e_eR.GetRodrigues(eps);
  const EigenMatrix3x3f e_JrI = EigenRotation3D::GetRodriguesJacobianInverse(e_er, eps);
#ifdef CFG_IMU_FULL_COVARIANCE
  const EigenMatrix3x3f e_Jrr2 = EigenMatrix3x3f(e_JrI * (e_eR * e_R1/*Rcick*/));
#else
  const EigenMatrix3x3f e_Jrr2 = EigenMatrix3x3f(e_JrI * e_R2);
#endif
  const EigenMatrix3x3f e_Jrr1 = EigenMatrix3x3f(-e_Jrr2);
#ifdef CFG_IMU_FULL_COVARIANCE
  const EigenMatrix3x3f e_Jrbw1 = EigenMatrix3x3f(-e_JrI * e_RdT *
                                                  EigenRotation3D::GetRodriguesJacobian(e_drbw, eps) *
                                                  e_Jrbw);


#else
  const EigenMatrix3x3f e_Jrbw1 = EigenMatrix3x3f(-e_JrI * e_R12 * e_RdT *
                                                  EigenRotation3D::GetRodriguesJacobian(e_drbw, eps) *
                                                  e_Jrbw);
#endif

  EigenVector3f e_v12 = EigenVector3f(e_R2T * e_v2 - e_Rg * e_g * m_Tvg);
  const EigenMatrix3x3f e_Jvr1 = EigenMatrix3x3f(e_R1 * EigenSkewSymmetricMatrix(e_v12));
  e_v12 = EigenVector3f(e_R1 * e_v12 - e_v1);
  const EigenVector3f e_ev = EigenVector3f(e_v12 - (e_vd + e_Jvba * e_dba + e_Jvbw * e_dbw));
  const EigenMatrix3x3f e_Jvv1 = EigenMatrix3x3f(-EigenMatrix3x3f::Identity());
  const EigenMatrix3x3f e_Jvba1 = EigenMatrix3x3f(-e_Jvba);
  const EigenMatrix3x3f e_Jvbw1 = EigenMatrix3x3f(-e_Jvbw);
  const EigenMatrix3x3f e_Jvr2 = EigenMatrix3x3f(-e_R1 * EigenSkewSymmetricMatrix(
                                                 EigenVector3f(e_R2.transpose() * e_v2)));
  const EigenMatrix3x3f e_Jvv2 = e_R21;
  const EigenMatrix3x3f e_Jvrg = EigenMatrix3x3f(-e_R1 * e_Rg * EigenSkewSymmetricMatrix(e_g) *
                                                 m_Tvg);//重力的扰动

  EigenVector3f e_p12 = EigenVector3f(e_R2T * e_pu + e_p2 - e_p1 - e_Rg * e_g * m_Tpg);
  const EigenMatrix3x3f e_Jpr1 = EigenMatrix3x3f(e_R1 * EigenSkewSymmetricMatrix(e_p12));
  e_p12 = EigenVector3f(e_R1 * e_p12 - e_v1 * m_Tpv);
  const EigenVector3f e_ep = EigenVector3f(-e_pu + e_p12 - (e_pd + e_Jpba * e_dba +
                                                                   e_Jpbw * e_dbw));
  const EigenMatrix3x3f e_Jpp1 = EigenMatrix3x3f(-e_R1);
  const EigenMatrix3x3f e_Jpv1 = EigenMatrix3x3f(-EigenMatrix3x3f::Identity() * m_Tpv);
  const EigenMatrix3x3f e_Jpba1 = EigenMatrix3x3f(-e_Jpba);
  const EigenMatrix3x3f e_Jpbw1 = EigenMatrix3x3f(-e_Jpbw);
  const EigenMatrix3x3f e_Jpp2 = e_R1;
  const EigenMatrix3x3f e_Jpr2 = EigenMatrix3x3f(-e_R1 * EigenSkewSymmetricMatrix(
                                                 EigenVector3f(e_R2T * e_pu)));
  const EigenMatrix3x3f e_Jprg = EigenMatrix3x3f(-e_R1 * e_Rg * EigenSkewSymmetricMatrix(e_g) *
                                                 m_Tpg);

  const EigenVector3f e_eba = EigenVector3f(e_ba1 - e_ba2);
  const EigenMatrix3x3f e_Jbaba1 = EigenMatrix3x3f::Identity();
  const EigenMatrix3x3f e_Jbaba2 = EigenMatrix3x3f(-EigenMatrix3x3f::Identity());

  const EigenVector3f e_ebw = EigenVector3f(e_bw1 - e_bw2);
  const EigenMatrix3x3f e_Jbwbw1 = EigenMatrix3x3f::Identity();
  const EigenMatrix3x3f e_Jbwbw2 = EigenMatrix3x3f(-EigenMatrix3x3f::Identity());

#ifdef IMU_DELTA_EIGEN_DEBUG_JACOBIAN
  const float e_dr1Max = 0.1f;
  const float e_dr2Max = 0.1f;
  const float e_drgMax = 0.1f;
  const float e_dp1Max = 0.01f;
  const float e_dp2Max = 0.01f;
  const float e_dv1Max = 0.01f;
  const float e_dv2Max = 0.01f;
  const float e_dba1Max = 0.01f;
  const float e_dbw1Max = 0.01f;
  //const float e_dr1Max = 0.0f;
  //const float e_dr2Max = 0.0f;
  //const float e_drgMax = 0.0f;
  //const float e_dp1Max = 0.0f;
  //const float e_dp2Max = 0.0f;
  //const float e_dv1Max = 0.0f;
  //const float e_dv2Max = 0.0f;
  //const float e_dba1Max = 0.0f;
  //const float e_dbw1Max = 0.0f;
//#ifdef CFG_DEBUG
#if 0
  if (UT::Debugging()) {
    //*((float *) &e_dr1Max) = 0.0f;
    *((float *) &e_dr2Max) = 0.0f;
    *((float *) &e_dbw1Max) = 0.0f;
  }
#endif
  const EigenVector3f e_dr1 = EigenVector3f::GetRandom(e_dr1Max * UT_FACTOR_DEG_TO_RAD);
  const EigenVector3f e_dr2 = EigenVector3f::GetRandom(e_dr2Max * UT_FACTOR_DEG_TO_RAD);
  const EigenVector3f e_drg = EigenVector3f::GetRandom(e_drgMax * UT_FACTOR_DEG_TO_RAD);
  const EigenRotation3D e_R1GT = EigenRotation3D(e_R1 * EigenRotation3D(e_dr1));
  const EigenRotation3D e_R2GT = EigenRotation3D(e_R2 * EigenRotation3D(e_dr2));
  const EigenRotation3D e_RgGT = EigenRotation3D(e_Rg * EigenRotation3D(e_drg));
  const EigenVector3f e_dp1 = EigenVector3f::GetRandom(e_dp1Max);
  const EigenVector3f e_dp2 = EigenVector3f::GetRandom(e_dp2Max);
  const EigenPoint3D e_p1GT = EigenVector3f(e_p1 + e_dp1);
  const EigenPoint3D e_p2GT = EigenVector3f(e_p2 + e_dp2);

  const EigenVector3f e_dv1 = EigenVector3f::GetRandom(e_dv1Max);
  const EigenVector3f e_dv2 = EigenVector3f::GetRandom(e_dv2Max);
  const EigenVector3f e_v1GT = EigenVector3f(e_v1 + e_dv1);
  const EigenVector3f e_v2GT = EigenVector3f(e_v2 + e_dv2);

  const EigenVector3f e_dba1 = EigenVector3f::GetRandom(e_dba1Max);
  const EigenVector3f e_ba1GT = EigenVector3f(e_ba1 + e_dba1);
  const EigenVector3f e_dbaGT = EigenVector3f(e_ba1GT - e_ba);

  const EigenVector3f e_dbw1 = EigenVector3f::GetRandom(e_dbw1Max * UT_FACTOR_DEG_TO_RAD);
  const EigenVector3f e_bw1GT = EigenVector3f(e_bw1 + e_dbw1);
  const EigenVector3f e_dbwGT = EigenVector3f(e_bw1GT - e_bw);

  const EigenVector3f e_drbwGT = EigenVector3f(-e_Jrbw * e_dbwGT);
#ifdef CFG_IMU_FULL_COVARIANCE
  const EigenVector3f e_erGT = EigenRotation3D(e_RdT * EigenRotation3D(e_drbwGT) *
                                               e_R2GT * e_R1GT.GetTranspose()).GetRodrigues(eps);
#else
  const EigenVector3f e_erGT = EigenRotation3D(e_R2GT * e_R1GT.GetTranspose() * e_RdT *
                                               EigenRotation3D(e_drbwGT)).GetRodrigues(eps);
#endif

  const EigenVector3f e_er1 = EigenVector3f(e_er - e_erGT);
  const EigenVector3f e_er2 = EigenVector3f(e_er1 + e_Jrr1 * e_dr1 + e_Jrr2 * e_dr2 +
                                                    e_Jrbw1 * e_dbw1);
  const float _eps = 1.0e-3f;
  UT::AssertReduction(e_er1, e_er2, 1, "er", _eps);

  e_v12 = EigenVector3f(e_R2GT.transpose() * e_v2GT - e_RgGT * e_g * m_Tvg);
  e_v12 = EigenVector3f(e_R1GT * e_v12 - e_v1GT);
  const EigenVector3f e_evGT = EigenVector3f(e_v12 - (e_vd + e_Jvba * e_dbaGT + e_Jvbw * e_dbwGT));
  const EigenVector3f e_ev1 = EigenVector3f(e_ev - e_evGT);
  const EigenVector3f e_ev2 = EigenVector3f(e_ev1 + e_Jvr1 * e_dr1 + e_Jvv1 * e_dv1 +
                                            e_Jvr2 * e_dr2 + e_Jvv2 * e_dv2 +
                                            e_Jvba1 * e_dba1 + e_Jvbw1 * e_dbw1 +
                                            e_Jvrg * e_drg);
  UT::AssertReduction(e_ev1, e_ev2);

  e_p12 = EigenVector3f(e_R2GT.transpose() * e_pu + e_p2GT - e_p1GT - e_RgGT * e_g * m_Tpg);
  e_p12 = EigenVector3f(e_R1GT * e_p12 - e_v1GT * m_Tpv);
  const EigenVector3f e_epGT = EigenVector3f(-e_pu + e_p12 - (e_pd + e_Jpba * e_dbaGT +
                                                              e_Jpbw * e_dbwGT));
  const EigenVector3f e_ep1 = EigenVector3f(e_ep - e_epGT);
  const EigenVector3f e_ep2 = EigenVector3f(e_ep1 + e_Jpr1 * e_dr1 + e_Jpp1 * e_dp1 +
                                                    e_Jpv1 * e_dv1 + e_Jpr2 * e_dr2 +
                                                    e_Jpp2 * e_dp2 + e_Jpba1 * e_dba1 +
                                                    e_Jpbw1 * e_dbw1 + e_Jprg * e_drg);
  UT::AssertReduction(e_ep1, e_ep2);
#endif
  e_e->m_er = e_er;
  e_J->m_Jr.setZero();
  e_J->m_Jr.block<3, 3>(0, 3) = e_Jrr1;
  e_J->m_Jr.block<3, 3>(0, 12) = e_Jrbw1;
  e_J->m_Jr.block<3, 3>(0, 18) = e_Jrr2;
  e_e->m_ev = e_ev;
  e_J->m_Jv.setZero();
  e_J->m_Jv.block<3, 3>(0, 3) = e_Jvr1;
  e_J->m_Jv.block<3, 3>(0, 6) = e_Jvv1;
  e_J->m_Jv.block<3, 3>(0, 9) = e_Jvba1;
  e_J->m_Jv.block<3, 3>(0, 12) = e_Jvbw1;
  e_J->m_Jv.block<3, 3>(0, 18) = e_Jvr2;
  e_J->m_Jv.block<3, 3>(0, 21) = e_Jvv2;
  e_J->m_JvgT = EigenMatrix2x3f(e_Jvrg.block<3, 2>(0, 0).transpose());//z固定不动,只取前两列
  e_e->m_ep = e_ep;
  e_J->m_Jp.setZero();
  e_J->m_Jp.block<3, 3>(0, 0) = e_Jpp1;
  e_J->m_Jp.block<3, 3>(0, 3) = e_Jpr1;
  e_J->m_Jp.block<3, 3>(0, 6) = e_Jpv1;
  e_J->m_Jp.block<3, 3>(0, 9) = e_Jpba1;
  e_J->m_Jp.block<3, 3>(0, 12) = e_Jpbw1;
  e_J->m_Jp.block<3, 3>(0, 15) = e_Jpp2;
  e_J->m_Jp.block<3, 3>(0, 18) = e_Jpr2;
  e_J->m_JpgT = EigenMatrix2x3f(e_Jprg.block<3, 2>(0, 0).transpose());
  e_e->m_eba = e_eba;
  e_J->m_Jba.setZero();
  e_J->m_Jba.block<3, 3>(0, 9) = e_Jbaba1;
  e_J->m_Jba.block<3, 3>(0, 24) = e_Jbaba2;
  e_e->m_ebw = e_ebw;
  e_J->m_Jbw.setZero();
  e_J->m_Jbw.block<3, 3>(0, 12) = e_Jbwbw1;
  e_J->m_Jbw.block<3, 3>(0, 27) = e_Jbwbw2;
#ifdef CFG_DEBUG
  Error e;
  Jacobian::RelativeLF J;
  GetErrorJacobian(C1, C2, pu, Rg, &e, &J, eps);
  e_e->AssertEqual(e);
  e_J->AssertEqual(J, m_Tpv);
  e_e->Set(e);
  e_J->Set(J, m_Tpv);
#endif
}

void Delta::EigenGetErrorJacobian(const Camera &C1, const Camera &C2, const Point3D &pu,
                                  EigenError *e_e, EigenJacobian::RelativeKF *e_J,
                                  const float eps) const {
  const Rotation3D Rg = C1.m_Cam_pose;//Rc0w(KF)
  EigenGetErrorJacobian(C1, C2, pu, Rg/*Rc0w(KF)*/, e_e, e_J, eps);
  e_J->m_Jr.block<3, 6>(0, 0).setZero();
  e_J->m_Jv.block<3, 6>(0, 0).setZero();
  e_J->m_Jp.block<3, 6>(0, 0).setZero();
  e_J->m_Jba.block<3, 6>(0, 0).setZero();
  e_J->m_Jbw.block<3, 6>(0, 0).setZero();
}

void Delta::EigenGetFactor(const float gyr, const Camera &C1/*前一帧状态*/, const Camera &C2/*后一帧状态*/, const Point3D &pu/*tc0_i*/,
                           EigenFactor::Global *e_A, const float eps) const {
  EigenError e_e;
  EigenJacobian::Global e_J;
  EigenGetErrorJacobian(C1/*前一帧状态*/, C2/*后一帧状态*/, pu/*tc0_i*/, &e_e, &e_J, eps);
  EigenGetFactor(gyr, m_W, e_J, e_e, e_A);
}

void Delta::EigenGetFactor(const float gyr, const Camera &C1, const Camera &C2, const Point3D &pu,
                           const Rotation3D &Rg, EigenFactor::RelativeLF *e_A,
                           const float eps) const {
  EigenError e_e;
  EigenJacobian::RelativeLF e_J;
  EigenGetErrorJacobian(C1, C2, pu, Rg, &e_e, &e_J, eps);
  EigenGetFactor(gyr, m_W, e_J, e_e, e_A);
}

void Delta::EigenGetFactor(const float gyr, const Camera &C1, const Camera &C2, const Point3D &pu,
                           EigenFactor::RelativeKF *e_A, const float eps) const {
  EigenError e_e;
  EigenJacobian::RelativeKF e_J;
  EigenGetErrorJacobian(C1, C2, pu, &e_e, &e_J, eps);
  EigenGetFactor(gyr, m_W, e_J, e_e, e_A);
}

void Delta::EigenGetFactor(const float gyr, const Weight &W, const EigenJacobian::Global &e_J,
                           const EigenError &e_e, EigenFactor::Global *e_A) {
#ifdef CFG_IMU_FULL_COVARIANCE
  EigenMatrix15x30f e_J_;
  EigenVector15f e_e_;
  e_J.Get(&e_J_);
  e_e.Get(&e_e_);
  const EigenWeight e_W(gyr, W);
  const EigenMatrix30x15f e_JTW = EigenMatrix30x15f(e_J_.transpose() * e_W);
  const EigenMatrix30x30f e_A_ = EigenMatrix30x30f(e_JTW * e_J_);
  const EigenVector30f e_b = EigenVector30f(e_JTW * e_e_);//Jt*W*e
  const float e_F = e_e_.dot(e_W * e_e_);//costfun
  e_A->Set(e_A_, e_b, e_F);
#else
  const EigenMatrix3x3f e_Wr = EigenMatrix3x3f(gyr * EigenMatrix3x3f(W.m_Wr));
  const EigenMatrix3x3f e_Wv = EigenMatrix3x3f(gyr * EigenMatrix3x3f(W.m_Wv));
  const EigenMatrix3x3f e_Wp = EigenMatrix3x3f(gyr * EigenMatrix3x3f(W.m_Wp));
  const float wba = gyr * W.m_wba;
  const float wbw = gyr * W.m_wbw;
  const EigenMatrix3x30f e_WJr = EigenMatrix3x30f(e_Wr * e_J.m_Jr);
  const EigenMatrix3x30f e_WJv = EigenMatrix3x30f(e_Wv * e_J.m_Jv);
  const EigenMatrix3x30f e_WJp = EigenMatrix3x30f(e_Wp * e_J.m_Jp);
  const EigenMatrix3x30f e_wJba = EigenMatrix3x30f(wba * e_J.m_Jba);
  const EigenMatrix3x30f e_wJbw = EigenMatrix3x30f(wbw * e_J.m_Jbw);
  const float e_F = (e_Wr * e_e.m_er).dot(e_e.m_er) +
                    (e_Wv * e_e.m_ev).dot(e_e.m_ev) +
                    (e_Wp * e_e.m_ep).dot(e_e.m_ep) +
                    wba * e_e.m_eba.squaredNorm() +
                    wbw * e_e.m_ebw.squaredNorm();
  e_A->Set(EigenMatrix30x31f(e_WJr.transpose() * EigenMatrix3x31f(e_J.m_Jr, e_e.m_er) +
                             e_WJv.transpose() * EigenMatrix3x31f(e_J.m_Jv, e_e.m_ev) +
                             e_WJp.transpose() * EigenMatrix3x31f(e_J.m_Jp, e_e.m_ep) +
                             e_wJba.transpose() * EigenMatrix3x31f(e_J.m_Jba, e_e.m_eba) +
                             e_wJbw.transpose() * EigenMatrix3x31f(e_J.m_Jbw, e_e.m_ebw)), e_F);
#endif
}

void Delta::EigenGetFactor(const float gyr, const Weight &W, const EigenJacobian::RelativeLF &e_J,
                           const EigenError &e_e, EigenFactor::RelativeLF *e_A) {
  EigenGetFactor(gyr, W, EigenJacobian::Global(e_J), e_e, (EigenFactor::Global *) e_A);
#ifdef CFG_IMU_FULL_COVARIANCE
  EigenMatrix15x2f e_Jg;
  EigenMatrix15x30f e_Jc;
  EigenVector15f e_e_;
  e_J.Get(&e_Jg, &e_Jc);
  e_e.Get(&e_e_);
  const EigenWeight e_W(gyr, W);
  const EigenMatrix2x15f e_JgTW = EigenMatrix2x15f(e_Jg.transpose() * e_W);
  const EigenMatrix2x2f e_Agg = EigenMatrix2x2f(e_JgTW * e_Jg);
  const EigenMatrix2x30f e_Agc = EigenMatrix2x30f(e_JgTW * e_Jc);
  const EigenVector2f e_bg = EigenVector2f(e_JgTW * e_e_);
  e_A->m_Agg = e_Agg;
  e_Agc.Get(e_A->m_Agc1, e_A->m_Agm1, e_A->m_Agc2, e_A->m_Agm2);
  e_A->m_bg = e_bg;
#else
  const EigenMatrix3x3f e_Wv = EigenMatrix3x3f(gyr * EigenMatrix3x3f(W.m_Wv));
  const EigenMatrix3x3f e_Wp = EigenMatrix3x3f(gyr * EigenMatrix3x3f(W.m_Wp));
  const EigenMatrix3x31f e_WJv = EigenMatrix3x31f(e_Wv * EigenMatrix3x31f(e_J.m_Jv, e_e.m_ev));
  const EigenMatrix3x31f e_WJp = EigenMatrix3x31f(e_Wp * EigenMatrix3x31f(e_J.m_Jp, e_e.m_ep));
  const EigenMatrix2x31f e_Agc = EigenMatrix2x31f(e_J.m_JvgT * e_WJv + e_J.m_JpgT * e_WJp);
  e_Agc.Get(e_A->m_Agc1, e_A->m_Agm1, e_A->m_Agc2, e_A->m_Agm2, e_A->m_bg);
  e_A->m_Agg = e_J.m_JvgT * e_Wv * e_J.m_JvgT.transpose() +
               e_J.m_JpgT * e_Wp * e_J.m_JpgT.transpose();
#endif
}

Delta::EigenError Delta::EigenGetError(const EigenErrorJacobian &e_Je, const EigenVector30f e_x) {
  EigenError e_e;
  e_e.m_er = EigenVector3f(e_Je.m_e.m_er + e_Je.m_J.m_Jr * e_x);
  e_e.m_ev = EigenVector3f(e_Je.m_e.m_ev + e_Je.m_J.m_Jv * e_x);
  e_e.m_ep = EigenVector3f(e_Je.m_e.m_ep + e_Je.m_J.m_Jp * e_x);
  e_e.m_eba = EigenVector3f(e_Je.m_e.m_eba + e_Je.m_J.m_Jba * e_x);
  e_e.m_ebw = EigenVector3f(e_Je.m_e.m_ebw + e_Je.m_J.m_Jbw * e_x);
  return e_e;
}

float Delta::EigenGetCost(const float gyr, const Camera &C1, const Camera &C2, const Point3D &pu,
                          const EigenVector6f &e_xc1, const EigenVector9f &e_xm1,
                          const EigenVector6f &e_xc2, const EigenVector9f &e_xm2,
                          const float eps) const {
  EigenErrorJacobian e_Je;
  EigenGetErrorJacobian(C1, C2, pu, &e_Je.m_e, &e_Je.m_J, eps);
  const EigenVector30f e_x(e_xc1, e_xm1, e_xc2, e_xm2);
  const EigenError e_e = EigenGetError(e_Je, e_x);
#ifdef CFG_IMU_FULL_COVARIANCE
  EigenVector15f e_e_;
  e_e.Get(&e_e_);
  const EigenWeight e_W(m_W);
  const float F = gyr * (e_W * e_e_).dot(e_e_);
#else
  const float F = gyr * ((EigenMatrix3x3f(m_W.m_Wr) * e_e.m_er).dot(e_e.m_er) +
                       (EigenMatrix3x3f(m_W.m_Wv) * e_e.m_ev).dot(e_e.m_ev) +
                       (EigenMatrix3x3f(m_W.m_Wp) * e_e.m_ep).dot(e_e.m_ep) +
                       m_W.m_wba * e_e.m_eba.squaredNorm() +
                       m_W.m_wbw * e_e.m_ebw.squaredNorm());
#endif
  return F;
}
#endif

}
