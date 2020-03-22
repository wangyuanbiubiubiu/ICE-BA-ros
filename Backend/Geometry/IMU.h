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
#ifndef _IMU_H_
#define _IMU_H_

#include "Camera.h"
#include "Parameter.h"
#include "AlignedVector.h"
#include <eigen3/Eigen/Dense>
namespace IMU {
class Measurement {
 public:
  inline const float& t() const { return m_w.r(); }
  inline       float& t()       { return m_w.r(); }
  inline bool operator < (const float t) const { return this->t() < t; }//重载了<，所以是按时间戳来排序
  inline bool Valid() const { return m_a.Valid(); }
  inline bool Invalid() const { return m_a.Invalid(); }
  inline void Invalidate() { m_a.Invalidate(); }
  static inline void Interpolate(const Measurement &u1/*k时刻imu测量*/, const Measurement &u2/*k+1时刻imu测量*/, const float t/*0.5*(tk + tk+!)*/,
                                 LA::AlignedVector3f &a, LA::AlignedVector3f &w) {
    const xp128f w1 = xp128f::get((u2.t() - t) / (u2.t() - u1.t()));//就是插值获得它们中间时刻对应的imu测量
    const xp128f w2 = xp128f::get(1.0f - w1[0]);
    a.xyzr() = (u1.m_a.xyzr() * w1) + (u2.m_a.xyzr() * w2);
    w.xyzr() = (u1.m_w.xyzr() * w1) + (u2.m_w.xyzr() * w2);
  }
  inline void Print(const bool e = false) const {
    if (e) {
      UT::Print("%e  %e %e %e  %e %e %e\n", t(), m_a.x(), m_a.y(), m_a.z(),
                                                 m_w.x(), m_w.y(), m_w.z());
    } else {
      UT::Print("%f  %f %f %f  %f %f %f\n", t(), m_a.x(), m_a.y(), m_a.z(),
                                                 m_w.x(), m_w.y(), m_w.z());
    }
  }
 public:
  LA::AlignedVector3f m_a, m_w;//acc 和 gry的测量值,转到了(Rc0_i*)左相机坐标系下,其中m_w的第3维度存的是时间戳
};

class Delta {
 public:
  class Transition {
   public:
    class DD {
     public:
      //LA::AlignedMatrix3x3f m_Fvr, m_Fpr, m_Fpv;
      SkewSymmetricMatrix m_Fvr/*dv/dR*/, m_Fpr/*dp/dR*/;
      xp128f m_Fpv;/*dp/dv*/
    };
    class DB {
     public:
      LA::AlignedMatrix3x3f m_Frbw/*dR/dbw*/, m_Fvba/*dv/dba*/, m_Fvbw/*dv/dbw*/, m_Fpba/*dp/dba*/, m_Fpbw/*dp/dbw*/;
    };
   public:
    DD m_Fdd;/*传递矩阵Rvp和Rvp部分*/
    DB m_Fdb;
  };
  class Covariance {
   public:
    class DD {
     public:
      inline void GetUpper(float **P) const {
        memcpy(&P[0][0], &m_Prr.m00(), 12);
        memcpy(&P[0][3], &m_Prv.m00(), 12);
        memcpy(&P[0][6], &m_Prp.m00(), 12);
        memcpy(&P[1][1], &m_Prr.m11(), 8);
        memcpy(&P[1][3], &m_Prv.m10(), 12);
        memcpy(&P[1][6], &m_Prp.m10(), 12);
        P[2][2] = m_Prr.m22();
        memcpy(&P[2][3], &m_Prv.m20(), 12);
        memcpy(&P[2][6], &m_Prp.m20(), 12);
        memcpy(&P[3][3], &m_Pvv.m00(), 12);
        memcpy(&P[3][6], &m_Pvp.m00(), 12);
        memcpy(&P[4][4], &m_Pvv.m11(), 8);
        memcpy(&P[4][6], &m_Pvp.m10(), 12);
        P[5][5] = m_Pvv.m22();
        memcpy(&P[5][6], &m_Pvp.m20(), 12);
        memcpy(&P[6][6], &m_Ppp.m00(), 12);
        memcpy(&P[7][7], &m_Ppp.m11(), 8);
        P[8][8] = m_Ppp.m22();
      }
      inline void IncreaseDiagonal(const float s2r, const float s2v, const float s2p) {
        m_Prr.IncreaseDiagonal(s2r);
        m_Pvv.IncreaseDiagonal(s2v);
        m_Ppp.IncreaseDiagonal(s2p);
      }
      inline void SetLowerFromUpper() {
        m_Prr.SetLowerFromUpper();
        m_Prv.GetTranspose(m_Pvr);
        m_Prp.GetTranspose(m_Ppr);
        m_Pvv.SetLowerFromUpper();
        m_Pvp.GetTranspose(m_Ppv);
        m_Ppp.SetLowerFromUpper();
      }
     public:
      LA::AlignedMatrix3x3f m_Prr, m_Prv, m_Prp;
      LA::AlignedMatrix3x3f m_Pvr, m_Pvv, m_Pvp;
      LA::AlignedMatrix3x3f m_Ppr, m_Ppv, m_Ppp;
    };
    class BD {
     public:
      LA::AlignedMatrix3x3f m_Pbav, m_Pbap;
      LA::AlignedMatrix3x3f m_Pbwr, m_Pbwv, m_Pbwp;
    };
    class DB {
     public:
      inline void Get(float **P, const int j) const {
        const int jbw = j + 3;
        memset(P[0] + j, 0, 12);              memcpy(P[0] + jbw, &m_Prbw.m00(), 12);
        memset(P[1] + j, 0, 12);              memcpy(P[1] + jbw, &m_Prbw.m10(), 12);
        memset(P[2] + j, 0, 12);              memcpy(P[2] + jbw, &m_Prbw.m20(), 12);
        memcpy(P[3] + j, &m_Pvba.m00(), 12);  memcpy(P[3] + jbw, &m_Pvbw.m00(), 12);
        memcpy(P[4] + j, &m_Pvba.m10(), 12);  memcpy(P[4] + jbw, &m_Pvbw.m10(), 12);
        memcpy(P[5] + j, &m_Pvba.m20(), 12);  memcpy(P[5] + jbw, &m_Pvbw.m20(), 12);
        memcpy(P[6] + j, &m_Ppba.m00(), 12);  memcpy(P[6] + jbw, &m_Ppbw.m00(), 12);
        memcpy(P[7] + j, &m_Ppba.m10(), 12);  memcpy(P[7] + jbw, &m_Ppbw.m10(), 12);
        memcpy(P[8] + j, &m_Ppba.m20(), 12);  memcpy(P[8] + jbw, &m_Ppbw.m20(), 12);
      }
      inline void GetTranspose(BD *P) const {
        m_Prbw.GetTranspose(P->m_Pbwr);
        m_Pvba.GetTranspose(P->m_Pbav);
        m_Pvbw.GetTranspose(P->m_Pbwv);
        m_Ppba.GetTranspose(P->m_Pbap);
        m_Ppbw.GetTranspose(P->m_Pbwp);
      }
     public:
      LA::AlignedMatrix3x3f m_Prbw;
      LA::AlignedMatrix3x3f m_Pvba, m_Pvbw;
      LA::AlignedMatrix3x3f m_Ppba, m_Ppbw;
    };
    class BB {
     public:
      inline void GetUpper(float **P, const int i, const int j) const {
        P[i][j] = P[i + 1][j + 1] = P[i + 2][j + 2] = m_Pbaba;
        P[i + 3][j + 3] = P[i + 4][j + 4] = P[i + 5][j + 5] = m_Pbwbw;
        memset(P[i] + j + 1, 0, 20);
        memset(P[i + 1] + j + 2, 0, 16);
        memset(P[i + 2] + j + 3, 0, 12);
        memset(P[i + 3] + j + 4, 0, 8);
        P[i + 4][j + 5] = 0.0f;
      }
      inline void IncreaseDiagonal(const float s2ba, const float s2bw) {
        m_Pbaba += s2ba;
        m_Pbwbw += s2bw;
      }
     public:
      float m_Pbaba, m_Pbwbw;
    };
   public:
    inline void MakeZero() { memset(this, 0, sizeof(Covariance)); }
    inline void IncreaseDiagonal(const float s2r, const float s2v, const float s2p,
                                 const float s2ba, const float s2bw) {
      m_Pdd.IncreaseDiagonal(s2r, s2v, s2p);
      m_Pbb.IncreaseDiagonal(s2ba, s2bw);
    }
    inline void SetLowerFromUpper() {
      m_Pdd.SetLowerFromUpper();
      //m_Pdb.GetTranspose(&m_Pbd);
    }
   public:
    static inline void ABT(const Transition::DD &A/*传递矩阵*/, const DD &B/*上个时刻协方差*/, DD *ABT) {
      ABT->m_Prr = B.m_Prr;//pvR中,R只会收到角度的扰动,为I
      ABT->m_Prv = B.m_Prv;
      ABT->m_Prp = B.m_Prp;
      //LA::AlignedMatrix3x3f::ABT(A.m_Fvr, B.m_Prr, ABT->m_Pvr);
      SkewSymmetricMatrix::AB(A.m_Fvr, B.m_Prr, ABT->m_Pvr);//ABT->m_Pvr = [A.m_Fvr]x * B.m_Prr
      ABT->m_Pvr += B.m_Pvr;////ABT->m_Pvr = [A.m_Fvr]x * B.m_Prr + B.m_Pvr
      //LA::AlignedMatrix3x3f::ABT(A.m_Fvr, B.m_Pvr, ABT->m_Pvv);
      SkewSymmetricMatrix::AB(A.m_Fvr, B.m_Prv, ABT->m_Pvv);//ABT->m_Pvv = [A.m_Fvr]x * B.m_Prv
      ABT->m_Pvv += B.m_Pvv;////ABT->m_Pvv = [A.m_Fvr]x * B.m_Prv + B.m_Pvv
      //LA::AlignedMatrix3x3f::ABT(A.m_Fvr, B.m_Ppr, ABT->m_Pvp);
      SkewSymmetricMatrix::AB(A.m_Fvr, B.m_Prp, ABT->m_Pvp);//ABT->m_Pvp = [A.m_Fvr]x * B.m_Prp
      ABT->m_Pvp += B.m_Pvp;////ABT->m_Pvp = [A.m_Fvr]x * B.m_Prp + B.m_Pvp
      //LA::AlignedMatrix3x3f::ABT(A.m_Fpr, B.m_Prr, ABT->m_Ppr);
      SkewSymmetricMatrix::AB(A.m_Fpr, B.m_Prr, ABT->m_Ppr); //ABT->m_Ppr = [A.m_Fpr]x * B.m_Prr
      //LA::AlignedMatrix3x3f::AddABTTo(A.m_Fpv, B.m_Prv, ABT->m_Ppr);
      LA::AlignedMatrix3x3f::AddsATo(A.m_Fpv, B.m_Pvr, ABT->m_Ppr);// ABT->m_Ppr =[A.m_Fpr]x * B.m_Prr + A.m_Fpv*B.m_Pvr
      ABT->m_Ppr += B.m_Ppr;////ABT->m_Ppr = [A.m_Fpr]x * B.m_Prr  + A.m_Fpv*B.m_Pvr + B.m_Ppr
      //LA::AlignedMatrix3x3f::ABT(A.m_Fpr, B.m_Pvr, ABT->m_Ppv);
      SkewSymmetricMatrix::AB(A.m_Fpr, B.m_Prv, ABT->m_Ppv);//ABT->m_Ppv = [A.m_Fpr]x * B.m_Prv
      //LA::AlignedMatrix3x3f::AddABTTo(A.m_Fpv, B.m_Pvv, ABT->m_Ppv);
      LA::AlignedMatrix3x3f::AddsATo(A.m_Fpv, B.m_Pvv, ABT->m_Ppv);// ABT->m_Ppv =  [A.m_Fpr]x * B.m_Prv + A.m_Fpv*B.m_Pvv
      ABT->m_Ppv += B.m_Ppv;//// ABT->m_Ppv =  [A.m_Fpr]x * B.m_Prv + A.m_Fpv*B.m_Pvv + B.m_Ppv
      //LA::AlignedMatrix3x3f::ABT(A.m_Fpr, B.m_Ppr, ABT->m_Ppp);
      SkewSymmetricMatrix::AB(A.m_Fpr, B.m_Prp, ABT->m_Ppp);//ABT->m_Ppp = [A.m_Fpr]x * B.m_Prp
      //LA::AlignedMatrix3x3f::AddABTTo(A.m_Fpv, B.m_Ppv, ABT->m_Ppp);
      LA::AlignedMatrix3x3f::AddsATo(A.m_Fpv, B.m_Pvp, ABT->m_Ppp);//ABT->m_Ppp = [A.m_Fpr]x * B.m_Prp + A.m_Fpv*B.m_Pvp
      ABT->m_Ppp += B.m_Ppp;////ABT->m_Ppp = [A.m_Fpr]x * B.m_Prp + A.m_Fpv*B.m_Pvp + B.m_Ppp
    }
    //static inline void ABT(const Transition::DD &A, const BD &B, DB *ABT) {
    //  B.m_Pbwr.GetTranspose(ABT->m_Prbw);
    //  B.m_Pbav.GetTranspose(ABT->m_Pvba);
    //  B.m_Pbwv.GetTranspose(ABT->m_Pvbw);
    //  LA::AlignedMatrix3x3f::AddABTTo(A.m_Fvr, B.m_Pbwr, ABT->m_Pvbw);
    //  B.m_Pbap.GetTranspose(ABT->m_Ppba);
    //  LA::AlignedMatrix3x3f::AddABTTo(A.m_Fpv, B.m_Pbav, ABT->m_Ppba);
    //  B.m_Pbwp.GetTranspose(ABT->m_Ppbw);
    //  LA::AlignedMatrix3x3f::AddABTTo(A.m_Fpr, B.m_Pbwr, ABT->m_Ppbw);
    //  LA::AlignedMatrix3x3f::AddABTTo(A.m_Fpv, B.m_Pbwv, ABT->m_Ppbw);
    //}
    static inline void AB(const Transition::DD &A/*传递矩阵pRv和pRv部分*/, const DB &B/*上个时刻协方差pRv和ba,bw部分*/, DB *AB/*FP的pRv和ba,bw部分*/) {
      AB->m_Prbw = B.m_Prbw;//没有m_Prba是因为这个总是0
      AB->m_Pvba = B.m_Pvba;
      AB->m_Pvbw = B.m_Pvbw;
      SkewSymmetricMatrix::AddABTo(A.m_Fvr, B.m_Prbw, AB->m_Pvbw);//AB->m_Pvbw = B.m_Pvbw + [A.m_Fvr]x * B.m_Prbw
      AB->m_Ppba = B.m_Ppba;
      LA::AlignedMatrix3x3f::AddsATo(A.m_Fpv, B.m_Pvba, AB->m_Ppba);//AB->m_Ppba = A.m_Fpv * B.m_Pvba + B.m_Ppba
      AB->m_Ppbw = B.m_Ppbw;
      SkewSymmetricMatrix::AddABTo(A.m_Fpr, B.m_Prbw, AB->m_Ppbw);
      LA::AlignedMatrix3x3f::AddsATo(A.m_Fpv, B.m_Pvbw, AB->m_Ppbw);//AB->m_Ppbw =  B.m_Ppbw +  [A.m_Fvr]x * B.m_Pvbw + A.m_Fpv * B.m_Pvbw
    }
    static inline void AddABTTo(const Transition::DB &A, const DB &B, DD *ABT) {
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Frbw, B.m_Prbw, ABT->m_Prr);//把传递矩阵pRv和ba,bw 与上个时刻协方差ba,bw和pRv部分 继续构造,刚才ABT里少加了这部分
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Frbw, B.m_Pvbw, ABT->m_Prv);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Frbw, B.m_Ppbw, ABT->m_Prp);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Fvbw, B.m_Prbw, ABT->m_Pvr);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Fvba, B.m_Pvba, ABT->m_Pvv);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Fvbw, B.m_Pvbw, ABT->m_Pvv);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Fvba, B.m_Ppba, ABT->m_Pvp);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Fvbw, B.m_Ppbw, ABT->m_Pvp);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Fpbw, B.m_Prbw, ABT->m_Ppr);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Fpba, B.m_Pvba, ABT->m_Ppv);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Fpbw, B.m_Pvbw, ABT->m_Ppv);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Fpba, B.m_Ppba, ABT->m_Ppp);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Fpbw, B.m_Ppbw, ABT->m_Ppp);
    }
    static inline void AddABTTo(const Transition::DB &A, const BB &B, DB *ABT) {
      const xp128f Bbaba = xp128f::get(B.m_Pbaba);
      const xp128f Bbwbw = xp128f::get(B.m_Pbwbw);
      LA::AlignedMatrix3x3f::AddsATo(Bbwbw, A.m_Frbw, ABT->m_Prbw);
      LA::AlignedMatrix3x3f::AddsATo(Bbaba, A.m_Fvba, ABT->m_Pvba);
      LA::AlignedMatrix3x3f::AddsATo(Bbwbw, A.m_Fvbw, ABT->m_Pvbw);
      LA::AlignedMatrix3x3f::AddsATo(Bbaba, A.m_Fpba, ABT->m_Ppba);
      LA::AlignedMatrix3x3f::AddsATo(Bbwbw, A.m_Fpbw, ABT->m_Ppbw);
    }
    static inline void ABT(const Transition &A/*传递矩阵*/, const Covariance &B/*上个时刻协方差*/, DD *ABTdd/*FP的pRv和pRv部分*/, DB *ABTdb/*FP的pRv和ba,bw部分*/) {
      ABT(A.m_Fdd/*传递矩阵pRv和pRv部分*/, B.m_Pdd/*上个时刻协方差pRv和pRv部分*/, ABTdd/*FP的pRv和pRv部分*/);//构造FP的pRv部分
      //ABT(A.m_Fdd, B.m_Pbd, ABTdb);
      AB(A.m_Fdd/*传递矩阵pRv和pRv部分*/, B.m_Pdb/*上个时刻协方差pRv和ba,bw部分*/, ABTdb/*FP的pRv和ba,bw部分*/);
      AddABTTo(A.m_Fdb/*传递矩阵pRv和ba,bw部分*/, B.m_Pdb/*上个时刻协方差pRv和ba,bw部分*/, ABTdd/*FP的pRv和pRv部分*/);
      AddABTTo(A.m_Fdb/*传递矩阵pRv和ba,bw部分*/, B.m_Pbb/*上个时刻协方差pRv和pRv部分*/, ABTdb/*FP的pRv和ba,bw部分*/);
    }
    static inline void ABTToUpper(const DD &A/*FP的pRv和pRv部分*/, const Transition::DD &B/*传递矩阵pRv和pRv部分*/, DD *ABT) {
      ABT->m_Prr = A.m_Prr;
      //LA::AlignedMatrix3x3f::ABT(A.m_Prr, B.m_Fvr, ABT->m_Prv);
      SkewSymmetricMatrix::ABT(A.m_Prr, B.m_Fvr, ABT->m_Prv);
      ABT->m_Prv += A.m_Prv;
      //LA::AlignedMatrix3x3f::ABT(A.m_Prr, B.m_Fpr, ABT->m_Prp);
      SkewSymmetricMatrix::ABT(A.m_Prr, B.m_Fpr, ABT->m_Prp);
      //LA::AlignedMatrix3x3f::AddABTTo(A.m_Prv, B.m_Fpv, ABT->m_Prp);
      LA::AlignedMatrix3x3f::AddsATo(B.m_Fpv, A.m_Prv, ABT->m_Prp);
      ABT->m_Prp += A.m_Prp;
      ABT->m_Pvv = A.m_Pvv;
      //LA::AlignedMatrix3x3f::AddABTToUpper(A.m_Pvr, B.m_Fvr, ABT->m_Pvv);
      SkewSymmetricMatrix::AddABTToUpper(A.m_Pvr, B.m_Fvr, ABT->m_Pvv);
      //LA::AlignedMatrix3x3f::ABT(A.m_Pvr, B.m_Fpr, ABT->m_Pvp);
      SkewSymmetricMatrix::ABT(A.m_Pvr, B.m_Fpr, ABT->m_Pvp);
      //LA::AlignedMatrix3x3f::AddABTTo(A.m_Pvv, B.m_Fpv, ABT->m_Pvp);
      LA::AlignedMatrix3x3f::AddsATo(B.m_Fpv, A.m_Pvv, ABT->m_Pvp);
      ABT->m_Pvp += A.m_Pvp;
      ABT->m_Ppp = A.m_Ppp;
      //LA::AlignedMatrix3x3f::AddABTToUpper(A.m_Ppr, B.m_Fpr, ABT->m_Ppp);
      SkewSymmetricMatrix::AddABTToUpper(A.m_Ppr, B.m_Fpr, ABT->m_Ppp);
      //LA::AlignedMatrix3x3f::AddABTToUpper(A.m_Ppv, B.m_Fpv, ABT->m_Ppp);
      LA::AlignedMatrix3x3f::AddsAToUpper(B.m_Fpv, A.m_Ppv, ABT->m_Ppp);
    }
    static inline void AddABTToUpper(const DB &A, const Transition::DB &B, DD *ABT) {
      LA::AlignedMatrix3x3f::AddABTToUpper(A.m_Prbw, B.m_Frbw, ABT->m_Prr);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Prbw, B.m_Fvbw, ABT->m_Prv);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Prbw, B.m_Fpbw, ABT->m_Prp);
      LA::AlignedMatrix3x3f::AddABTToUpper(A.m_Pvba, B.m_Fvba, ABT->m_Pvv);
      LA::AlignedMatrix3x3f::AddABTToUpper(A.m_Pvbw, B.m_Fvbw, ABT->m_Pvv);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Pvba, B.m_Fpba, ABT->m_Pvp);
      LA::AlignedMatrix3x3f::AddABTTo(A.m_Pvbw, B.m_Fpbw, ABT->m_Pvp);
      LA::AlignedMatrix3x3f::AddABTToUpper(A.m_Ppba, B.m_Fpba, ABT->m_Ppp);
      LA::AlignedMatrix3x3f::AddABTToUpper(A.m_Ppbw, B.m_Fpbw, ABT->m_Ppp);
    }
    static inline void ABTToUpper(const DD &Add/*FP的pRv和pRv部分*/, const DB &Adb/*FP的pRv和ba,bw部分*/, const Transition &B/*传递矩阵*/, DD *ABT) {
      ABTToUpper(Add/*FP的pRv和pRv部分*/, B.m_Fdd/*传递矩阵pRv和pRv部分*/, ABT);//在算FPF.t的pRv和pRv中pRv影响的部分
      AddABTToUpper(Adb/*FP的pRv和ba,bw部分*/, B.m_Fdb/*FP的ba,bw和pRv部分*/, ABT);//在算FPF.t的pRv和pRv中ba,bw影响的部分
    }
    static inline void FPFT(const Transition &F/*传递矩阵*/, const Covariance &P/*上个时刻协方差*/, DD *U, Covariance *FPFT/*当前时刻协方差*/) {
      ABT(F/*传递矩阵*/, P/*上个时刻协方差*/, U/*FP的pRv和pRv部分*/, &FPFT->m_Pdb/*FP的pRv和ba,bw部分*/);//计算FP pRv这三行的值
      ABTToUpper(*U/*FP的pRv和pRv部分*/, FPFT->m_Pdb/*FP的pRv和ba,bw部分*/, F/*传递矩阵*/, &FPFT->m_Pdd);//计算FPF.t pRv左上角的值
      FPFT->m_Pbb = P.m_Pbb;//右下角
      FPFT->SetLowerFromUpper();
    }
   public:
    DD m_Pdd;/*矩阵块的Rvp和Rvp部分*/
    DB m_Pdb;/*矩阵块的Rvp和ba,bw部分*/
    //BD m_Pbd;
    BB m_Pbb;/*矩阵块的ba,bw和ba,bw部分*/
  };
#ifdef CFG_IMU_FULL_COVARIANCE
  class Weight {
   public:
    inline const LA::AlignedMatrix3x3f* operator [] (const int i) const { return m_W[i]; }
    inline       LA::AlignedMatrix3x3f* operator [] (const int i)       { return m_W[i]; }
    inline void Set(const Covariance &P, AlignedVector<float> *work) {
      work->Resize(15 * 15);
      float *_P[15];
      _P[0] = work->Data();
      for (int i = 1; i < 15; ++i) {
        _P[i] = _P[i - 1] + 15;
      }
      P.m_Pdd.GetUpper(_P);//获取上三角矩阵块的Rvp和Rvp部分部分的数据
      P.m_Pdb.Get(_P, 9);//获取上三角矩阵块的Rvp和ba,bw部分部分的数据
      P.m_Pbb.GetUpper(_P, 9, 9);//获取上三角矩阵块的ba,bw和ba,bw部分部分的数据

      if (LA::LS::InverseLDL<float>(15, _P)) {
        for (int i = 0, _i = 0; i < 5; ++i) {
          float *W0 = _P[_i++], *W1 = _P[_i++], *W2 = _P[_i++];
          for (int j = 0, _j = 0; j < 5; ++j, _j += 3) {
            m_W[i][j].Set(W0 + _j, W1 + _j, W2 + _j);
          }
        }
      } else {
        MakeZero();
      }
    }
    inline void GetScaled(const float w, Weight *W) const {
      const xp128f _w = xp128f::get(w);
      GetScaled(_w, W);
    }
    inline void GetScaled(const xp128f &w, Weight *W) const {
      for (int i = 0; i < 5; ++i) {
        m_W[i][i].GetScaledToUpper(w, W->m_W[i][i]);
        W->m_W[i][i].SetLowerFromUpper();
        for (int j = i + 1; j < 5; ++j) {
          m_W[i][j].GetScaled(w, W->m_W[i][j]);
          W->m_W[i][j].GetTranspose(W->m_W[j][i]);
        }
      }
    }
    inline void MakeZero() { memset(this, 0, sizeof(Weight)); }
    inline void SetLowerFromUpper() {
      for (int i = 0; i < 5; ++i) {
        m_W[i][i].SetLowerFromUpper();
        for (int j = i + 1; j < 5; ++j) {
          m_W[i][j].GetTranspose(m_W[j][i]);
        }
      }
    }
    inline bool AssertEqual(const Weight &W, const int verbose = 1,
                            const std::string str = "", const bool norm = true) const {
      bool scc = true;
      LA::AlignedMatrix3x3f W1, W2;
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
          W1 = m_W[i][j];
          W2 = W[i][j];
          if (norm) {
            const float n1 = sqrt(W1.SquaredFrobeniusNorm());
            const float n2 = sqrt(W2.SquaredFrobeniusNorm());
            const float n = std::max(n1, n2);
            if (n == 0.0f)
              continue;
            const float s = 1.0f / n;
            W1 *= s;
            W2 *= s;
          }
          scc = W1.AssertEqual(W2, verbose, str + UT::String(".W[%d][%d]", i, j)) && scc;
        }
      }
      return scc;
    }
   public:
    LA::AlignedMatrix3x3f m_W[5][5];//权重矩阵
  };
#else
  class Weight {
   public:
    inline void GetScaled(const float s, Weight *W) const {
      const xp128f _s = xp128f::get(s);
      GetScaled(_s, W);
    }
    inline void GetScaled(const xp128f &s, Weight *W) const {
      m_Wr.GetScaled(s, W->m_Wr);
      m_Wv.GetScaled(s, W->m_Wv);
      m_Wp.GetScaled(s, W->m_Wp);
      W->m_wba = s[0] * m_wba;
      W->m_wbw = s[0] * m_wbw;
    }
    inline void SetLowerFromUpper() {
      m_Wr.SetLowerFromUpper();
      m_Wv.SetLowerFromUpper();
      m_Wp.SetLowerFromUpper();
    }
    inline bool AssertEqual(const Weight &W, const int verobse = 1,
                            const std::string str = "", const bool norm = true) const {
      bool scc = true;
      scc = AssertEqual(m_Wr, W.m_Wr, verobse, str + ".Wr", norm) && scc;
      scc = AssertEqual(m_Wv, W.m_Wv, verobse, str + ".Wv", norm) && scc;
      scc = AssertEqual(m_Wp, W.m_Wp, verobse, str + ".Wp", norm) && scc;
      scc = UT::AssertEqual(m_wba, W.m_wba, verobse, str + ".wba") && scc;
      scc = UT::AssertEqual(m_wbw, W.m_wbw, verobse, str + ".wbw") && scc;
      return scc;
    }
    static inline bool AssertEqual(const LA::AlignedMatrix3x3f &W1,
                                   const LA::AlignedMatrix3x3f &W2,
                                   const int verbose = 1, const std::string str = "",
                                   const bool norm = true) {
      if (norm) {
        const float n1 = sqrt(W1.SquaredFrobeniusNorm());
        const float n2 = sqrt(W2.SquaredFrobeniusNorm());
        const float n = std::max(n1, n2);
        if (n == 0.0f)
          return true;
        const float s = 1.0f / n;
        const LA::AlignedMatrix3x3f W1_ = W1 * s;
        const LA::AlignedMatrix3x3f W2_ = W2 * s;
        return AssertEqual(W1_, W2_, verbose, str, false);
      } else {
        return W1.AssertEqual(W2, verbose, str);
      }
    }
   public:
    union {
      struct { LA::AlignedMatrix3x3f m_Wr, m_Wv, m_Wp; };
      struct { float m_data1[3], m_wba, m_data2[7], m_wbw; };
    };
  };
#endif

  class Error {
   public:
    inline bool Valid() const { return m_er.Valid(); }
    inline bool Invalid() const { return m_er.Invalid(); }
    inline void Invalidate() { m_er.Invalidate(); }
    inline void Print(const bool e = false, const bool l = false) const {
      UT::PrintSeparator();
      m_er.Print(" er = ", e, l, true);
      m_ev.Print(" ev = ", e, l, true);
      m_ep.Print(" ep = ", e, l, true);
      m_eba.Print("eba = ", e, l, true);
      m_ebw.Print("ebw = ", e, l, true);
    }
    inline void Print(const std::string str, const bool e, const bool l) const {
      UT::PrintSeparator();
      const std::string _str(str.size(), ' ');
      m_er.Print( str + " er = ", e, l, true);
      m_ev.Print(_str + " ev = ", e, l, true);
      m_ep.Print(_str + " ep = ", e, l, true);
      m_eba.Print(_str + "eba = ", e, l, true);
      m_ebw.Print(_str + "ebw = ", e, l, true);
    }
   public:
    LA::AlignedVector3f m_er, m_ev, m_ep, m_eba, m_ebw;//motion部分的残差具体看IMU.h的注释
  };
  class Jacobian {
   public:
    class Gravity {
    public:
      LA::AlignedMatrix2x3f m_JvgT, m_JpgT;
    };
    class FirstMotion {
     public:
      inline void GetTranspose(FirstMotion &JT) const {
        m_Jrbw1.GetTranspose(JT.m_Jrbw1);
        m_Jvv1.GetTranspose(JT.m_Jvv1);
        m_Jvba1.GetTranspose(JT.m_Jvba1);
        m_Jvbw1.GetTranspose(JT.m_Jvbw1);
        m_Jpv1.GetTranspose(JT.m_Jpv1);
        m_Jpba1.GetTranspose(JT.m_Jpba1);
        m_Jpbw1.GetTranspose(JT.m_Jpbw1);
      }
     public:
      LA::AlignedMatrix3x3f m_Jrbw1;//div(e_r)/div(bwi)
      LA::AlignedMatrix3x3f m_Jvv1/*div(e_v)/div(v_wi)*/, m_Jvba1/*div(e_v)/div(bai)*/, m_Jvbw1;/*div(e_v)/div(bwi)*/
      LA::AlignedMatrix3x3f m_Jpv1/*div(e_p)/div(v_wi)*/, m_Jpba1/*div(e_p)/div(bai)*/, m_Jpbw1;/*div(e_p)/div(bwi)*/
    };
    class Global : public FirstMotion {
     public:
      inline void GetTranspose(Global &JT) const {
        FirstMotion::GetTranspose(JT);
        m_Jrr1.GetTranspose(JT.m_Jrr1);
        m_Jvr1.GetTranspose(JT.m_Jvr1);
        m_Jpp1.GetTranspose(JT.m_Jpp1);
        m_Jpr1.GetTranspose(JT.m_Jpr1);
        m_Jpr2.GetTranspose(JT.m_Jpr2);
      }
     public:
      LA::AlignedMatrix3x3f m_Jrr1;//div(e_r)/div(Rciw)
      LA::AlignedMatrix3x3f m_Jvr1;//div(e_v)/div(Rciw)
      LA::AlignedMatrix3x3f m_Jpp1;/*div(e_p)/div(twci)*/
      LA::AlignedMatrix3x3f m_Jpr1;/*div(e_p)/div(Rciw)*/
      LA::AlignedMatrix3x3f m_Jpr2;/*div(e_p)/div(Rcjw)*/
    };
    class RelativeLF : public Gravity, public Global {
     public:
      LA::AlignedMatrix3x3f m_Jvr2/*div(e_v)/div(Rcjw)*/, m_Jvv2;/*div(e_v)/div(v_wj)*/
    };
    class RelativeKF : public Gravity, public FirstMotion {
     public:
      inline void GetTranspose(RelativeKF &JT) const {
        FirstMotion::GetTranspose(JT);
        m_Jrr2.GetTranspose(JT.m_Jrr2);
        m_Jvr2.GetTranspose(JT.m_Jvr2);
        m_Jvv2.GetTranspose(JT.m_Jvv2);
        m_Jpp2.GetTranspose(JT.m_Jpp2);
        m_Jpr2.GetTranspose(JT.m_Jpr2);
      }
     public:
      LA::AlignedMatrix3x3f m_Jrr2;//div(e_r)/div(Rcjw)
      LA::AlignedMatrix3x3f m_Jvr2/*div(e_v)/div(Rcjw)*/, m_Jvv2;/*div(e_v)/div(v_wj)*/
      LA::AlignedMatrix3x3f m_Jpp2;/*div(e_p)/div(twcj)*/
      LA::AlignedMatrix3x3f m_Jpr2;/*div(e_p)/div(Rcjw)*/
    };
  };
  class ErrorJacobian {
   public:
    Error m_e;//motion部分的残差
    Jacobian::Global m_J;//motion部分的雅克比
  };
  class Factor {
   public:
    class Unitary {
     public:
      inline void Set(const LA::AlignedMatrix3x3f *Ap, const LA::AlignedMatrix3x3f *Ar,
                      const LA::AlignedMatrix3x3f *Av, const LA::AlignedMatrix3x3f *Aba,
                      const LA::AlignedMatrix3x3f *Abw, const LA::AlignedVector3f *b) {
        m_Acc.Set(Ap, Ar, b);
        m_Acm.Set(Ap + 2, Ar + 2);
        m_Amm.Set(Av + 2, Aba + 2, Abw + 2, b + 2);
      }
      inline void Set(const LA::AlignedMatrix3x3f *Av, const LA::AlignedMatrix3x3f *Aba,
                      const LA::AlignedMatrix3x3f *Abw, const LA::AlignedVector3f *b) {
        m_Acc.MakeZero();
        m_Acm.MakeZero();
        m_Amm.Set(Av, Aba, Abw, b);
      }
      inline void MakeMinus() {
        m_Acc.MakeMinus();
        m_Acm.MakeMinus();
        m_Amm.MakeMinus();
      }
      static inline void AmB(const Unitary &A, const Unitary &B, Unitary &AmB) {
        Camera::Factor::Unitary::CC::AmB(A.m_Acc, B.m_Acc, AmB.m_Acc);
        Camera::Factor::Unitary::CM::AmB(A.m_Acm, B.m_Acm, AmB.m_Acm);
        Camera::Factor::Unitary::MM::AmB(A.m_Amm, B.m_Amm, AmB.m_Amm);
      }
     public:
      Camera::Factor::Unitary::CC m_Acc;//这一帧的pose自己和自己的H以及自己对应的-b
      Camera::Factor::Unitary::CM m_Acm;//这一帧的pose和这一帧的motion的H
      Camera::Factor::Unitary::MM m_Amm;//这一帧的motion自己和自己的H以及自己对应的-b
    };
    class Auxiliary {
     public:
      class Global {
       public:
        void Set(const Jacobian::Global &J, const Error &e, const float w, const Weight &W,
                 const float Tpv);
        inline void Get(Unitary *A11/*H中i时刻的c,m和i时刻的c,m,以及对应的-b*/, Unitary *A22/*H中j时刻的c,m和j时刻的c,m,以及对应的-b*/
                , Camera::Factor::Binary *A12/*H中i时刻的c,m和j时刻的c,m*/) const {
#ifdef CFG_IMU_FULL_COVARIANCE//imu约束的H是30*30的,矩阵就四大块,左上角就是前一时刻的c,m，右下角就是后一时刻的cm，左下右上的就是前一时刻和后一时刻的
          //const LA::AlignedMatrix3x3f *A[10] = {&m_A[ 0], &m_A[ 9], &m_A[17], &m_A[24], &m_A[30],
          //                                      &m_A[35], &m_A[39], &m_A[42], &m_A[44], &m_A[45]};
          A11->Set(m_A, m_A + 9, m_A + 17, m_A + 24, m_A + 30, m_b);//上三角部分的左上角前一帧的自己和自己对应的块以及对应的b
          A22->Set(m_A + 40, m_A + 44, m_A + 47, m_A + 49, m_A + 50, m_b + 5);//上三角部分的右下角的后一帧的自己和自己对应的块以及对应的b
          A12->Set(m_A + 5, m_A + 14, m_A + 22, m_A + 29, m_A + 35);//上三角部分的右上角,也就是非自己和自己的H部分
#else
          A11->m_Acc.m_A.Set(m_Ap1p1, m_Ap1r1, m_Ar1r1);
          A11->m_Acm.Set(m_Ap1v1, m_Ap1ba1, m_Ap1bw1, m_Ar1v1, m_Ar1ba1, m_Ar1bw1);
          A12->m_Acc.Set(m_Ap1p2, m_Ap1r2, m_Ar1p2, m_Ar1r2);
          A12->m_Acm.m_Arv = m_Ar1v2;
          A11->m_Acc.m_b.Set(m_bp1, m_br1);

          A11->m_Amm.m_A.Set(m_Av1v1, m_Av1ba1, m_Av1bw1, m_Aba1ba1, m_Aba1bw1, m_Abw1bw1);
          A12->m_Amc.Set(m_Av1p2, m_Av1r2, m_Aba1p2, m_Aba1r2, m_Abw1p2, m_Abw1r2);
          A12->m_Amm.m_Amv.Set(m_Av1v2, m_Aba1v2, m_Abw1v2);
          A12->m_Amm.m_Ababa = m_Aba1ba2;
          A12->m_Amm.m_Abwbw = m_Abw1bw2;
          A11->m_Amm.m_b.Set(m_bv1, m_bba1, m_bbw1);

          A22->m_Acc.m_A.Set(m_Ap2p2, m_Ap2r2, m_Ar2r2);
          A22->m_Acm.MakeZero();
          A22->m_Acc.m_b.Set(m_bp2, m_br2);
          A22->m_Amm.m_A.Set(m_Av2v2, m_Aba2ba2, m_Abw2bw2);
          A22->m_Amm.m_b.Set(m_bv2, m_bba2, m_bbw2);
#endif
        }
       public:
        Jacobian::Global m_JT;//m_Jrr1,m_Jrbw1,m_Jvr1,m_Jvv1,m_Jvba1,m_Jvbw1,m_Jpr1,m_Jpr2,m_Jpp1,m_Jpv1,m_Jpba1,m_Jpbw1的转置
        Weight m_W;
#ifdef CFG_IMU_FULL_COVARIANCE
        LA::AlignedMatrix3x3f m_JTW[10][5]/*存储的J.t*W*/, m_A[55];//J.t*W*J的上三角部分
        LA::AlignedVector3f m_b[10];//J.t*e部分
#else
        LA::AlignedMatrix3x3f m_JTWr1r, m_JTWbw1r;
        LA::AlignedMatrix3x3f m_JTWr1v, m_JTWv1v, m_JTWba1v, m_JTWbw1v;
        LA::AlignedMatrix3x3f m_JTWp1p, m_JTWr1p, m_JTWv1p, m_JTWba1p, m_JTWbw1p, m_JTWr2p;
        LA::SymmetricMatrix3x3f m_Ap1p1, m_Ar1r1, m_Av1v1, m_Aba1ba1, m_Abw1bw1;
        LA::SymmetricMatrix3x3f m_Ap2p2, m_Ar2r2, m_Av2v2;
        LA::AlignedVector3f m_bp1, m_br1, m_bv1, m_bba1, m_bbw1, m_bp2, m_br2, m_bv2, m_bba2, m_bbw2;
        LA::AlignedMatrix3x3f m_Ap1r1, m_Ap1v1, m_Ap1ba1,  m_Ap1bw1,  m_Ap1p2,  m_Ap1r2;
        LA::AlignedMatrix3x3f          m_Ar1v1, m_Ar1ba1,  m_Ar1bw1,  m_Ar1p2,  m_Ar1r2,  m_Ar1v2;
        LA::AlignedMatrix3x3f                   m_Av1ba1,  m_Av1bw1,  m_Av1p2,  m_Av1r2,  m_Av1v2;
        LA::AlignedMatrix3x3f                             m_Aba1bw1, m_Aba1p2, m_Aba1r2, m_Aba1v2;
        LA::AlignedMatrix3x3f                                        m_Abw1p2, m_Abw1r2, m_Abw1v2;
        LA::AlignedMatrix3x3f                                                   m_Ap2r2;
        float m_Aba1ba2, m_Abw1bw2, m_Aba2ba2, m_Abw2bw2;
#endif
      };
      class RelativeLF : public Global {
       public:
        void Set(const Jacobian::RelativeLF &J, const Error &e, const float w, const Weight &W,
                 const float Tpv);
       public:
        LA::AlignedMatrix3x3f m_Jvr2T, m_Jvv2T;
#ifdef CFG_IMU_FULL_COVARIANCE
        LA::AlignedMatrix2x3f m_JTWg[5];
#else
        LA::AlignedMatrix3x3f m_JTWr2v, m_JTWv2v;
        LA::AlignedMatrix2x3f m_JTWgv, m_JTWgp;
        LA::AlignedMatrix3x3f m_Ar2v2;
#endif
        LA::SymmetricMatrix2x2f m_Agg;
#ifdef CFG_IMU_FULL_COVARIANCE
        LA::AlignedMatrix2x3f m_Agc[10];//g和ci(pi,ri),mi(vi bai bwi)，cj(pj,rj),mj(vj baj bgj)
#else
        LA::AlignedMatrix2x3f m_Agp1, m_Agr1, m_Agv1, m_Agba1, m_Agbw1;
        LA::AlignedMatrix2x3f m_Agp2, m_Agr2, m_Agv2;
#endif
        LA::Vector2f m_bg;
      };
      class RelativeKF {
       public:
        void Set(const Jacobian::RelativeKF &J, const Error &e, const float w, const Weight &W,
                 const float Tpv);
#ifdef CFG_IMU_FULL_COVARIANCE
        inline void Get(Unitary *A11, Unitary *A22, Camera::Factor::Binary *A12) const {
          A11->Set(m_Ac, m_Ac + 7, m_Ac + 13, m_bc);
          A22->Set(m_Ac + 21, m_Ac + 25, m_Ac + 28, m_Ac + 30, m_Ac + 31, m_bc + 3);
          A12->Set(m_Ac + 3, m_Ac + 10, m_Ac + 16);
        }
#endif
       public:
        Jacobian::RelativeKF m_JT;
        Weight m_W;
#ifdef CFG_IMU_FULL_COVARIANCE
        LA::AlignedMatrix3x3f m_JTWc[8][5], m_Ac[36];
        LA::AlignedVector3f m_bc[8];//m1,c2,m2
        LA::AlignedMatrix2x3f m_JTWg[5];
        LA::SymmetricMatrix2x2f m_Agg;//e_JgTW * e_Jg
        LA::AlignedMatrix2x3f m_Agc[8];//g和m1(v1 ba1 bw1)，c2(p1,r1),m2(v2 ba2 bg2)
        LA::Vector2f m_bg;//g
#else
        LA::AlignedMatrix3x3f m_JTWbw1r, m_JTWr2r;
        LA::AlignedMatrix3x3f m_JTWv1v, m_JTWba1v, m_JTWbw1v, m_JTWr2v, m_JTWv2v;
        LA::AlignedMatrix3x3f m_JTWv1p, m_JTWba1p, m_JTWbw1p, m_JTWp2p, m_JTWr2p;
        LA::AlignedMatrix2x3f m_JTWgv, m_JTWgp;
        LA::SymmetricMatrix3x3f m_Av1v1, m_Aba1ba1, m_Abw1bw1, m_Ap2p2, m_Ar2r2, m_Av2v2;
        LA::AlignedVector3f m_bv1, m_bba1, m_bbw1, m_bp2, m_br2, m_bv2, m_bba2, m_bbw2;
        LA::AlignedMatrix3x3f m_Av1ba1,  m_Av1bw1,  m_Av1p2,  m_Av1r2,  m_Av1v2;
        LA::AlignedMatrix3x3f           m_Aba1bw1, m_Aba1p2, m_Aba1r2, m_Aba1v2;
        LA::AlignedMatrix3x3f                      m_Abw1p2, m_Abw1r2, m_Abw1v2;
        LA::AlignedMatrix3x3f                                 m_Ap2r2;
        LA::AlignedMatrix3x3f                                          m_Ar2v2;
        float m_Aba1ba2, m_Abw1bw2, m_Aba2ba2, m_Abw2bw2;
        LA::SymmetricMatrix2x2f m_Agg;
        LA::AlignedMatrix2x3f m_Agv1, m_Agba1, m_Agbw1, m_Agp2, m_Agr2, m_Agv2;
        LA::Vector2f m_bg;
#endif
      };
    };
   public:
    inline void MakeZero() { memset(this, 0, sizeof(Factor)); }
   public:
    ErrorJacobian m_Je;
    union {
      struct { float m_data[21], m_F/*motion部分的cost:马氏距离下的残差*/; };
      struct { Unitary m_A11,//这里存的是前一帧自己的运动和pose之间的约束,也就是Hc1c1,Hc1m1,Hm1m1,-bc1,-bm1
              m_A22; };//这里存的是后一帧自己的运动和pose之间的约束,也就是Hc2c2,Hc2m2,Hm2m2,-bc2,-bm2
    };
  };
  class Reduction {
   public:
    Error m_e;
    float m_F, m_dF;
  };
  class ESError : public LA::AlignedVector3f {
   public:
    inline ESError() {}
    inline ESError(const LA::AlignedVector3f &e, const float s = 1.0f) {
      if (s == 1.0f) {
        *((LA::AlignedVector3f *) this) = e;
      } else {
        e.GetScaled(s, *this);
      }
    }
    inline void Print(const bool l = true) const {
      if (l) {
        UT::Print("%f %f %f", x(), y(), z());
      } else {
        UT::Print("%.2f %.2f %.2f", x(), y(), z());
      }
    }
  };
  class ES : public UT::ES<float, int> {
   public:
    inline void Initialize() {
      UT::ES<float, int>::Initialize();
      m_ESr.Initialize();
      m_ESp.Initialize();
      m_ESv.Initialize();
      m_ESba.Initialize();
      m_ESbw.Initialize();
    }
    inline void Accumulate(const Error &e, const float F, const int iFrm = -1) {
      UT::ES<float, int>::Accumulate(F, F, iFrm);
      m_ESr.Accumulate(ESError(e.m_er, UT_FACTOR_RAD_TO_DEG), -1.0f, iFrm);
      m_ESp.Accumulate(ESError(e.m_ep), -1.0f, iFrm);
      m_ESv.Accumulate(ESError(e.m_ev), -1.0f, iFrm);
      m_ESba.Accumulate(ESError(e.m_eba), -1.0f, iFrm);
      m_ESbw.Accumulate(ESError(e.m_ebw, UT_FACTOR_RAD_TO_DEG), -1.0f, iFrm);
    }
    inline void Print(const std::string str = "", const bool l = true) const {
      if (!Valid()) {
        return;
      }
      UT::ES<float, int>::Print(str + "ed = ", true, l);
      const std::string _str(str.size() + 17, ' ');
      if (m_ESr.Valid()) {
        m_ESr.Print(_str + "er  = ", false, l);
      }
      if (m_ESp.Valid()) {
        m_ESp.Print(_str + "ep  = ", false, l);
      }
      if (m_ESv.Valid()) {
        m_ESv.Print(_str + "ev  = ", false, l);
      }
      if (m_ESba.Valid()) {
        m_ESba.Print(_str + "eba = ", false, l);
      }
      if (m_ESbw.Valid()) {
        m_ESbw.Print(_str + "ebw = ", false, l);
      }
    }
  public:
    UT::ES<ESError, int> m_ESr, m_ESp, m_ESv, m_ESba, m_ESbw;
  };
 public:

  inline bool Valid() const { return m_RT.Valid(); }
  inline bool Invalid() const { return m_RT.Invalid(); }
  inline void Invalidate() { m_RT.Invalidate(); }

#ifdef CFG_IMU_FULL_COVARIANCE
  inline Rotation3D GetRotationState(const Camera &C1, const Camera &C2) const {
    return Rotation3D(C1.m_Cam_pose) / C2.m_Cam_pose;
  }
  inline Rotation3D GetRotationMeasurement(const Camera &C1, const float eps) const {
    return m_RT / Rotation3D(m_Jrbw * (C1.m_bw - m_bw), eps);//m_RT*exp[-(m_Jrbw*(C1.m_bw - m_bw))]x.t 因为是jpl转的R,所以是exp[-th]
  }
  inline Rotation3D GetRotationMeasurement(const LA::AlignedVector3f &dbw, const float eps) const {
    return m_RT / Rotation3D(m_Jrbw * dbw/*w*/, eps);//消除bias gyr的影响
  }
  inline LA::AlignedVector3f GetRotationError(const Camera &C1, const Camera &C2,
                                              const float eps) const {
    const Rotation3D eR = GetRotationMeasurement(C1, eps) / GetRotationState(C1, C2);
    return eR.GetRodrigues(eps);
  }
#else
  inline Rotation3D GetRotationState(const Camera &C1, const Camera &C2) const {
    return Rotation3D(C2.m_Cam_pose) / C1.m_Cam_pose;
  }
  inline Rotation3D GetRotationMeasurement(const Camera &C1, const float eps) const {
    return Rotation3D(m_Jrbw * (C1.m_bw - m_bw), eps) / m_RT;
  }
  inline LA::AlignedVector3f GetRotationError(const Camera &C1, const Camera &C2,
                                              const float eps) const {
    const Rotation3D eR = GetRotationState(C1, C2) / GetRotationMeasurement(C1, eps);
    return eR.GetRodrigues();
  }
#endif

  inline LA::AlignedVector3f GetVelocityMeasurement(const Camera &C1) const {
    return m_v + m_Jvba * (C1.m_ba - m_ba) + m_Jvbw * (C1.m_bw - m_bw);//假设ba,bw的影响都是线性的了
  }
  inline LA::AlignedVector3f GetVelocityState(const Camera &C1, const Camera &C2) const {
    LA::AlignedVector3f dv = C2.m_v - C1.m_v;// Rc0w*(v_wj - v_wji + g*dt)
    if (!IMU_GRAVITY_EXCLUDED) {
      dv.z() += IMU_GRAVITY_MAGNITUDE * m_Tvg;
    }
    return C1.m_Cam_pose.GetAppliedRotation(dv);
  }
  inline LA::AlignedVector3f GetVelocityError(const Camera &C1, const Camera &C2) const {
    return GetVelocityState(C1, C2) - GetVelocityMeasurement(C1);
  }

  inline LA::AlignedVector3f GetPositionState(const Camera &C1, const Camera &C2,
                                              const Point3D &pu) const {
    LA::AlignedVector3f dp = C2.m_p - C1.m_p - C1.m_v * m_Tpv;
    if (!IMU_GRAVITY_EXCLUDED) {
      dp.z() += IMU_GRAVITY_MAGNITUDE * m_Tpg;
    }
    dp += C2.m_Cam_pose.GetAppliedRotationInversely(pu);
    dp = C1.m_Cam_pose.GetAppliedRotation(dp);
    dp -= pu;
    return dp;
  }
  inline LA::AlignedVector3f GetPositionMeasurement(const Camera &C1) const {
    return m_p + m_Jpba * (C1.m_ba - m_ba) + m_Jpbw * (C1.m_bw - m_bw);
  }
  inline LA::AlignedVector3f GetPositionError(const Camera &C1, const Camera &C2,
                                              const Point3D &pu) const {
    return GetPositionState(C1, C2, pu) - GetPositionMeasurement(C1);
  }
  
#ifdef CFG_DEBUG
  inline void DebugSetMeasurement(const Camera &C1, const Camera &C2, const Point3D &pu, const float eps) {
    const LA::AlignedVector3f dba = C1.m_ba - m_ba;
    const LA::AlignedVector3f dbw = C1.m_bw - m_bw;
    m_RT = GetRotationState(C1, C2) * Rotation3D(m_Jrbw * dbw, eps);
    m_v = GetVelocityState(C1, C2) - (m_Jvba * dba + m_Jvbw * dbw);
    m_p = GetPositionState(C1, C2, pu) - (m_Jpba * dba + m_Jpbw * dbw);
  }
#endif
  inline void GetError(const Camera &C1, const Camera &C2, const Point3D &pu, Error &e,
                       const float eps) const {
    e.m_er = GetRotationError(C1, C2, eps);
    e.m_ev = GetVelocityError(C1, C2);
    e.m_ep = GetPositionError(C1, C2, pu);
    e.m_eba = C1.m_ba - C2.m_ba;
    e.m_ebw = C1.m_bw - C2.m_bw;
  }
  inline Error GetError(const Camera &C1, const Camera &C2, const Point3D &pu,
                        const float eps) const {
    Error e;
    GetError(C1, C2, pu, e, eps);
    return e;
  }
  static inline void GetError(const ErrorJacobian &Je, const LA::AlignedVector3f *xp1,
                              const LA::AlignedVector3f *xr1, const LA::AlignedVector3f *xv1,
                              const LA::AlignedVector3f *xba1, const LA::AlignedVector3f *xbw1, 
                              const LA::AlignedVector3f *xp2, const LA::AlignedVector3f *xr2,
                              const LA::AlignedVector3f *xv2, const LA::AlignedVector3f *xba2,
                              const LA::AlignedVector3f *xbw2, Error &e) {
#ifdef CFG_DEBUG
    UT_ASSERT(xp1 || xr1 || xv1 || xba1 || xbw1 || xp2 || xr2 || xv2 || xba2 || xbw2);
#endif
    e = Je.m_e;
    if (xp1) {
      if (xp2) {
        const LA::AlignedVector3f dxp = *xp1 - *xp2;
        LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jpp1, dxp, (float *) &e.m_ep);
      } else {
        LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jpp1, *xp1, (float *) &e.m_ep);
      }
    } else if (xp2) {
      LA::AlignedMatrix3x3f::SubtractAbFrom(Je.m_J.m_Jpp1, *xp2, (float *) &e.m_ep);
    }
    if (xr1) {
      if (xr2) {
        const LA::AlignedVector3f dxr = *xr1 - *xr2;
        LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jrr1, dxr, (float *) &e.m_er);
        LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jpr2, *xr2, (float *) &e.m_ep);
      } else {
        LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jrr1, *xr1, (float *) &e.m_er);
      }
      LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jvr1, *xr1, (float *) &e.m_ev);
      LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jpr1, *xr1, (float *) &e.m_ep);
    } else if (xr2) {
      LA::AlignedMatrix3x3f::SubtractAbFrom(Je.m_J.m_Jrr1, *xr2, (float *) &e.m_er);
      LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jpr2, *xr2, (float *) &e.m_ep);
    }
    if (xv1) {
      if (xv2) {
        const LA::AlignedVector3f dxv = *xv1 - *xv2;
        LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jvv1, dxv, (float *) &e.m_ev);
      } else {
        LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jvv1, *xv1, (float *) &e.m_ev);
      }
      LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jpv1, *xv1, (float *) &e.m_ep);
    } else if (xv2) {
      LA::AlignedMatrix3x3f::SubtractAbFrom(Je.m_J.m_Jvv1, *xv2, (float *) &e.m_ev);
    }
    if (xba1) {
      LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jvba1, *xba1, (float *) &e.m_ev);
      LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jpba1, *xba1, (float *) &e.m_ep);
      e.m_eba += *xba1;
    }
    if (xbw1) {
      LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jrbw1, *xbw1, (float *) &e.m_er);
      LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jvbw1, *xbw1, (float *) &e.m_ev);
      LA::AlignedMatrix3x3f::AddAbTo(Je.m_J.m_Jpbw1, *xbw1, (float *) &e.m_ep);
      e.m_ebw += *xbw1;
    }
    if (xba2) {
      e.m_eba -= *xba2;
    }
    if (xbw2) {
      e.m_ebw -= *xbw2;
    }
  }
  inline void GetError(const Jacobian::RelativeLF &J, const LA::Vector2f &xg,
                       const LA::AlignedVector3f &xp1, const LA::AlignedVector3f &xr1,
                       const LA::AlignedVector3f &xv1, const LA::AlignedVector3f &xba1,
                       const LA::AlignedVector3f &xbw1, const LA::AlignedVector3f &xp2,
                       const LA::AlignedVector3f &xr2, const LA::AlignedVector3f &xv2,
                       const LA::AlignedVector3f &xba2, const LA::AlignedVector3f &xbw2,
                       Error &e) const {
    GetError(J, xg, xp1, xr1, xv1, xba1, xbw1, xp2, xr2, xv2, xba2, xbw2, m_Tpv, e);
  }
  static inline void GetError(const Jacobian::RelativeLF &J, const LA::Vector2f &xg,
                              const LA::AlignedVector3f &xp1, const LA::AlignedVector3f &xr1,
                              const LA::AlignedVector3f &xv1, const LA::AlignedVector3f &xba1,
                              const LA::AlignedVector3f &xbw1, const LA::AlignedVector3f &xp2,
                              const LA::AlignedVector3f &xr2, const LA::AlignedVector3f &xv2,
                              const LA::AlignedVector3f &xba2, const LA::AlignedVector3f &xbw2,
                              const float Tpv, Error &e) {
    const LA::AlignedVector3f dxr = xr1 - xr2;
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jrr1, dxr, (float *) &e.m_er);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jrbw1, xbw1, (float *) &e.m_er);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jvr1, xr1, (float *) &e.m_ev);
    //LA::AlignedMatrix3x3f::AddAbTo(J.m_Jvv1, xv1, (float *) &e.m_ev);
    e.m_ev -= xv1;
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jvba1, xba1, (float *) &e.m_ev);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jvbw1, xbw1, (float *) &e.m_ev);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jvr2, xr2, (float *) &e.m_ev);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jvv2, xv2, (float *) &e.m_ev);
    LA::AlignedMatrix2x3f::AddATbTo(J.m_JvgT, xg, e.m_ev);
    const LA::AlignedVector3f dxp = xp1 - xp2;
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jpp1, dxp, (float *) &e.m_ep);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jpr1, xr1, (float *) &e.m_ep);
    //LA::AlignedMatrix3x3f::AddAbTo(J.m_Jpv1, xv1, e.m_ep);
    e.m_ep -= (xv1 * Tpv);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jpba1, xba1, (float *) &e.m_ep);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jpbw1, xbw1, (float *) &e.m_ep);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jpr2, xr2, (float *) &e.m_ep);
    LA::AlignedMatrix2x3f::AddATbTo(J.m_JpgT, xg, e.m_ep);
    e.m_eba += xba1;
    e.m_eba -= xba2;
    e.m_ebw += xbw1;
    e.m_ebw -= xbw2;
  }
  inline void GetError(const Jacobian::RelativeKF &J, const LA::Vector2f &xg,
                       const LA::AlignedVector3f &xv1, const LA::AlignedVector3f &xba1,
                       const LA::AlignedVector3f &xbw1, const LA::AlignedVector3f &xp2,
                       const LA::AlignedVector3f &xr2, const LA::AlignedVector3f &xv2,
                       const LA::AlignedVector3f &xba2, const LA::AlignedVector3f &xbw2,
                       Error &e) const {
    GetError(J, xg, xv1, xba1, xbw1, xp2, xr2, xv2, xba2, xbw2, m_Tpv, e);
  }

  static inline void GetError(const Jacobian::RelativeKF &J, const LA::Vector2f &xg,
                              const LA::AlignedVector3f &xv1, const LA::AlignedVector3f &xba1,
                              const LA::AlignedVector3f &xbw1, const LA::AlignedVector3f &xp2,
                              const LA::AlignedVector3f &xr2, const LA::AlignedVector3f &xv2,
                              const LA::AlignedVector3f &xba2, const LA::AlignedVector3f &xbw2,
                              const float Tpv, Error &e) {
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jrbw1, xbw1, (float *) &e.m_er);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jrr2, xr2, (float *) &e.m_er);
    //LA::AlignedMatrix3x3f::AddAbTo(J.m_Jvv1, xv1, (float *) &e.m_ev);
    e.m_ev -= xv1;
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jvba1, xba1, (float *) &e.m_ev);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jvbw1, xbw1, (float *) &e.m_ev);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jvr2, xr2, (float *) &e.m_ev);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jvv2, xv2, (float *) &e.m_ev);
    LA::AlignedMatrix2x3f::AddATbTo(J.m_JvgT, xg, e.m_ev);
    //LA::AlignedMatrix3x3f::AddAbTo(J.m_Jpv1, xv1, e.m_ep);
    e.m_ep -= (xv1 * Tpv);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jpba1, xba1, (float *) &e.m_ep);
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jpbw1, xbw1, (float *) &e.m_ep);
    //LA::AlignedMatrix3x3f::AddAbTo(J.m_Jpp2, xp2, (float *) &e.m_ep);
    e.m_ep += xp2;
    LA::AlignedMatrix3x3f::AddAbTo(J.m_Jpr2, xr2, (float *) &e.m_ep);
    LA::AlignedMatrix2x3f::AddATbTo(J.m_JpgT, xg, e.m_ep);
    e.m_eba += xba1;
    e.m_eba -= xba2;
    e.m_ebw += xbw1;
    e.m_ebw -= xbw2;
  }
//#define  WYA_DEBUG
#ifdef WYA_DEBUG
///只是我为了debug用一下


        template <typename Derived>
        static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
        {
            Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
            ans << typename Derived::Scalar(0), -q(2), q(1),
                    q(2), typename Derived::Scalar(0), -q(0),
                    -q(1), q(0), typename Derived::Scalar(0);
            return ans;
        }

        static Eigen::Matrix3f ComputeJl(const Eigen::Vector3f & omega)
        {


            float theta;
            Eigen::Matrix3f so3 = expAndTheta(omega, &theta);

            Eigen::Matrix3f Omega = hat(omega);
            Eigen::Matrix3f Omega_sq = Omega*Omega;
            Eigen::Matrix3f V;

            if(theta<1e-10)
            {
                V = so3.matrix();
                //Note: That is an accurate expansion!
            }
            else
            {
                float theta_sq = theta*theta;
                V = (Eigen::Matrix3f::Identity()
                     + (1-cos(theta))/(theta_sq)*Omega
                     + (theta-sin(theta))/(theta_sq*theta)*Omega_sq);
            }
            return V;

        }

        static Eigen::Matrix3f eigenskewSymmetric(const Eigen::Vector3f &q)
        {
            Eigen::Matrix3f ans;
            ans << 0, -q(2), q(1),
                    q(2), 0, -q(0),
                    -q(1), q(0), 0;
            return ans;
        }


        static Eigen::Matrix3f ComputeJlInv(const Eigen::Vector3f &e_w) {
            //return EigenMatrix3x3f(Rotation3D::GetRodriguesJacobianInverse(e_w.GetAlignedVector3f()));
            const Eigen::Matrix3f e_S = skewSymmetric(e_w);
            const float th = sqrtf(e_w.transpose() *e_w);
            float th_2 = 0.5*th;
            Eigen::Vector3f a = e_w/th ;
            const Eigen::Matrix3f a_S = skewSymmetric(a);
            if (th < 1e-10) {
                return Eigen::Matrix3f(Eigen::Matrix3f::Identity() + 0.5f * e_S);
            } else {

                return Eigen::Matrix3f(th_2 * (cosf(th_2)/sinf(th_2))* Eigen::Matrix3f::Identity()
                                    + (1-(th_2 * (cosf(th_2)/sinf(th_2))))*a*a.transpose() - (th_2)*a_S);

            }
        }
        static Eigen::Matrix3f ComputeJrInv(const Eigen::Vector3f & so3_r)
        {
            return ComputeJlInv(-so3_r);
        }

        static Eigen::Matrix3f ComputeJr(const Eigen::Vector3f & so3_r)
        {
            return ComputeJl(-so3_r);
        }


        static Eigen::Matrix3f expAndTheta(const Eigen::Vector3f & omega, float * theta)
        {
            *theta = omega.norm();
            float half_theta = 0.5*(*theta);

            float imag_factor;
            float real_factor = cos(half_theta);
            if((*theta)<1e-10)
            {
                float theta_sq = (*theta)*(*theta);
                float theta_po4 = theta_sq*theta_sq;
                imag_factor = 0.5-0.0208333*theta_sq+0.000260417*theta_po4;
            }
            else
            {
                float sin_half_theta = sin(half_theta);
                imag_factor = sin_half_theta/(*theta);
            }

            return Eigen::Matrix3f(Eigen::Quaternionf(real_factor,
                                                      imag_factor*omega.x(),
                                                      imag_factor*omega.y(),
                                                      imag_factor*omega.z()).toRotationMatrix());
        }

        static Eigen::Matrix3f hat(const Eigen::Vector3f & v)
        {
            Eigen::Matrix3f Omega;
            Omega <<  0, -v(2),  v(1)
                    ,  v(2),     0, -v(0)
                    , -v(1),  v(0),     0;
            return Omega;
        }

        static Eigen::Matrix3f SetRodrigues(const Eigen::Vector3f &w) {


            //w = theta * n
            const Eigen::Vector3f w2{w[0]*w[0],w[1]*w[1],w[2]*w[2]};
            const float th2 = w.transpose() *w, th = sqrtf(th2);//theta
            if (th < 1e-10) {
                const float s = 1.0f / sqrtf(th2 + 4.0f);
                return Eigen::Matrix3f(Eigen::Quaternionf(s + s,
                                                          w[0] * s,
                                                          w[1] * s,
                                                          w[2] * s).toRotationMatrix());
            }
            const float t1 = sinf(th) / th, t2 = (1.0f - cosf(th)) / th2, t3 = 1.0f - t2 * th2;//cos(theta)
            const Eigen::Vector3f t1w = w * t1;//sin(theta) * n
            const Eigen::Vector3f t2w2 = w2 * t2;//(1-cos(theta)) * n * n.t 的对角线部分
            const float t2wx = t2 * w.x();//(1-cos(theta)) * n * n.t非对角线部分
            const float t2wxy = t2wx * w.y();
            const float t2wxz = t2wx * w.z();
            const float t2wyz = t2 * w.y() * w.z();//R = cos(theta)*I + sin(theta) * [n]x + (1-cos(theta)) * n * n.t
            Eigen::Matrix3f R;
            R<<t3 + t2w2.x(),t2wxy + t1w.z(),t2wxz - t1w.y(),
                    t2wxy - t1w.z(),t3 + t2w2.y(),t2wyz + t1w.x(),
                    t2wxz + t1w.y(),t2wyz - t1w.x(),t3 + t2w2.z();
            return R.transpose();
        }
#endif

//imu的预积分约束,这里只求了m_Jrr1,m_Jrbw1,m_Jvr1,m_Jvv1,m_Jvba1,m_Jvbw1,m_Jpr1,m_Jpr2,m_Jpp1,m_Jpv1,m_Jpba1,m_Jpbw1
//残差:e_r,e_v,e_p,e_ba,e_bw
//优化变量 Rciw,Rcjw,p_wci,p_wcj,v_wi,v_wj,bai,baj,bwi,bwj
// m_Jrr2 = -m_Jrr1, m_Jvv2 = -m_Jvv1 ,m_Jpp2 = - m_Jpp1,这里没有赋值,但是之后雅克比部分肯定是会给的
//他都是用的jpl的被动表示,比如GetRodrigues,GetRodriguesJacobianInverse都是反着的R = exp[-th]x (Indirect Kalman Filter for 3D Attitude Estimation)
//// e_r = -ln{预积分的Rij * exp[Jrbw *(bwi - z_bw)]x * Rcjw * Rciw.t}v
//  Rciw右乘扰动 => -ln{预积分的Rij * exp[Jrbw *(bwi - z_bw)]x * Rcjw * (Rciw*exp[-th]x).t}v
//                    = -ln{预积分的Rij * exp[Jrbw *(bwi - z_bw)]x * Rcjw * exp[th]x*Rciw.t}v  //exp[th]x前面这堆东西 = eR * Rciw
// 利用伴随性质 exp[Ad(R)*th]x *R => R* exp[th]x => -ln{exp[eR * Rciw*th]x*eR * Rciw *Rciw.t}v
////  BCH展开 div(e_r)/div(Rciw)  = - Jl_inv(-e_r)* eR * Rciw = - Jr_inv(e_r)* eR * Rciw
// Rcjw右乘扰动 =>  -ln{预积分的Rij * exp[Jrbw *(bwi - z_bw)]x * Rcjw * exp[-th]x* Rciw.t}v
//// 和上面一样的步骤 div(e_r)/div(Rcjw)   = Jr_inv(e_r)* eR * Rciw (不过这里没有赋值这个)
// bwi加一个扰动 => -ln{预积分的Rij * exp[Jrbw *(dbw + bwi - z_bw)]x * Rcjw * Rciw.t}v
//             = -ln{预积分的Rij  * exp[Jl(Jrbw *(bwi - z_bw))*Jrbw *dbw]x * exp[Jrbw *(bwi - z_bw)]x * Rcjw * Rciw.t}v
//  利用伴随性质 = -ln{ exp[预积分的Rij * Jl*Jrbw *dbw]x * 预积分的Rij * exp[Jrbw *(bwi - z_bw)]x * Rcjw * Rciw.t}v
//BCH展开      = - (Jl_inv(-e_r)*预积分的Rij * Jl(Jrbw *(bwi - z_bw))*Jrbw *dbw + -er)
// div(e_r)/div(bwi) = - Jl_inv(-e_r)*预积分的Rij * Jl*Jrbw
//// div(e_r)/div(bwi)   = - Jr_inv(e_r)*预积分的Rij * Jr(-Jrbw *(bwi - z_bw))*Jrbw

////  e_v = Rciw*(v_wj - v_wi + g*t) - (m_v + m_Jvba * (bai - m_ba) + m_Jvbw * (bwi - m_bw))
// Rciw右乘扰动 div(e_v)/div(Rciw) = Rciw * exp[-th]x(v_wj - v_wi + g*dt) 我这里就简写了
////    正交矩阵性质[Ab]xA = A[b]x  = Rciw * [v_wj - v_wi + g*dt]x = [Rciw*(v_wj - v_wji + g*dt)]x * Rciw
////       div(e_v)/div(v_wi) = -Rciw
////       div(e_v)/div(v_wj) = Rciw  (不过这里没有赋值这个)
////       div(e_v)/div(bai) = -m_Jvba
////       div(e_v)/div(bwi) = -m_Jvbw

 //twcj - twci + Rwcj*tc0i=  twbj - twbi + Rwci*tc0i
//twcj = twbj - twbi + (Rwci-Rwci)*tc0i + twci
//     = Vw_c0i*dt + -0.5*gw*dt^2 + Rwc0*pc0i_c0j + (Rwci-Rwci)*tc0i + twci

// 0 = twcj - twci - Vw_c0i*dt +0.5*gw*dt^2 - Rwc0i*pc0i_c0j - (Rwci-Rwci)*tc0i
//  同时左乘Rciw,并且考虑了bias的扰动就是e_p
////  e_p = Rciw*(p_wcj - p_wci - v_wci*dt + 0.5*g*dt^2) + Rciw * Rcjw.t*tc0i - tc0i - (m_p + m_Jpba * (bai - m_ba) + m_Jpbw * (bwi - m_bw))
// Rciw右乘扰动 div(e_p)/div(Rciw) = Rciw * exp[-th]x(p_wcj - p_wci - v_wci*dt + 0.5*g*dt^2 + Rcjw.t*tc0i)
//                               = Rciw * [p_wcj - p_wci - v_wci*dt + 0.5*g*dt^2 + Rcjw.t*tc0i]x
//// 正交矩阵性质[Ab]xA = A[b]x  = [Rciw*(p_wcj - p_wci - v_wci*dt + 0.5*g*dt^2 + Rcjw.t*tc0i)]x*Rciw
// Rcjw右乘扰动 div(e_p)/div(Rcjw) = Rciw * (Rcjw*exp[-th]x).t*tc0i
//                            = - Rciw * [Rcjw.t*tc0i]x
////                           = - [Rciw* Rcjw.t*tc0i]x * Rciw
////       div(e_p)/div(v_wi) = -Rciw * dt
////       div(e_p)/div(p_wci) = -Rciw
////       div(e_p)/div(p_wcj) = Rciw  (不过这里没有赋值这个)
////       div(e_p)/div(bai) = -m_Jvba
////       div(e_p)/div(bwi) = -m_Jvbw
//// e_ba = bai - baj
//// e_bw = bwi - bwj 这两个的雅克比就是i是1,j是-1
inline void GetErrorJacobian(const Camera &C1/*前一帧状态*/, const Camera &C2/*后一帧状态*/, const Point3D &pu,/*tc0_i*/
                               Error *e, Jacobian::Global *J, const float eps) const {

#ifdef CFG_IMU_FULL_COVARIANCE
    const Rotation3D dR = GetRotationState(C1, C2);//Rciw * Rcjw.t
    const LA::AlignedVector3f drbw = m_Jrbw * (C1.m_bw - m_bw);
    const Rotation3D eR = m_RT / Rotation3D(drbw, eps) / dR;//预积分的Rij * (exp[-Jrbw *(bwi - z_bw)]x).t * (Rciw * Rcjw.t).t
                                                            //预积分的Rij * exp[Jrbw *(bwi - z_bw)]x * Rcjw * Rciw.t
    eR.GetRodrigues(e->m_er, eps);//这里SO3转so3是-th
    Rotation3D::GetRodriguesJacobianInverse(e->m_er, J->m_Jrr1, eps);// J->m_Jrr1 = Jr_inv(e_r) 它这里用的是右乘雅克比
    Rotation3D::GetRodriguesJacobian(drbw.GetMinus(), J->m_Jrbw1, eps);//J->m_Jrbw1 = Jr(-Jrbw *(bwi - z_bw))
    J->m_Jrr1.MakeMinus();//J->m_Jrr1 = - Jr_inv(e_r)
    J->m_Jrbw1 = J->m_Jrr1 * m_RT * J->m_Jrbw1 * m_Jrbw;//div(e_r)/div(bwi) = - Jr_inv(e_r)*预积分的Rij * Jr(-Jrbw *(bwi - z_bw))*Jrbw
    J->m_Jrr1 = J->m_Jrr1 * eR * C1.m_Cam_pose;//div(e_r)/div(Rciw) = - Jr_inv(e_r)* eR * Rciw
      //debug用

#ifdef WYA_DEBUG
      Eigen::Matrix3f Tc1w_eigen,Tc2w_eigen,m_RTeigen;
      Tc1w_eigen<<C1.m_Cam_pose.r00(),C1.m_Cam_pose.r01(),C1.m_Cam_pose.r02(),
              C1.m_Cam_pose.r10(),C1.m_Cam_pose.r11(),C1.m_Cam_pose.r12(),
              C1.m_Cam_pose.r20(),C1.m_Cam_pose.r21(),C1.m_Cam_pose.r22();

      Tc2w_eigen<<C2.m_Cam_pose.r00(),C2.m_Cam_pose.r01(),C2.m_Cam_pose.r02(),
              C2.m_Cam_pose.r10(),C2.m_Cam_pose.r11(),C2.m_Cam_pose.r12(),
              C2.m_Cam_pose.r20(),C2.m_Cam_pose.r21(),C2.m_Cam_pose.r22();

      m_RTeigen<<m_RT.m00(),m_RT.m01(),m_RT.m02(),
              m_RT.m10(),m_RT.m11(),m_RT.m12(),//预积分的R12
              m_RT.m20(),m_RT.m21(),m_RT.m22();

      Eigen::Matrix3f dR_eigen = Tc1w_eigen * Tc2w_eigen.transpose();
      Eigen::Matrix3f m_Jrbw_eigen;
      m_Jrbw_eigen<<m_Jrbw.m00(),m_Jrbw.m01(),m_Jrbw.m02(),
              m_Jrbw.m10(),m_Jrbw.m11(),m_Jrbw.m12(),
              m_Jrbw.m20(),m_Jrbw.m21(),m_Jrbw.m22();
      Eigen::Vector3f drbw_eigen = m_Jrbw_eigen * Eigen::Vector3f{(C1.m_bw - m_bw).x(),(C1.m_bw - m_bw).y(),(C1.m_bw - m_bw).z()};

      Eigen::Matrix3f eR_eigen = m_RTeigen* SetRodrigues(drbw_eigen) * Tc2w_eigen * Tc1w_eigen.transpose();
      Eigen::AngleAxisf tempr(eR_eigen);
      Eigen::Vector3f m_er_eigen{-tempr.axis()[0]*tempr.angle(),-tempr.axis()[1]*tempr.angle(),-tempr.axis()[2]*tempr.angle()};
      Eigen::Matrix3f m_Jrr1_eigen =- ComputeJlInv(-m_er_eigen) * eR_eigen * Tc1w_eigen;

    std::cout<<"dR: \n"
    <<dR.m00()<<" "<<dR.m01()<<" "<<dR.m02()<<"\n"<<
      dR.m10()<<" "<<dR.m11()<<" "<<dR.m12()<<"\n"<<
      dR.m20()<<" "<<dR.m21()<<" "<<dR.m22()<<"\n";

      std::cout<<"dR_eigen: \n"<<dR_eigen<<"\n";

      std::cout<<"eR: \n"
               <<eR.m00()<<" "<<eR.m01()<<" "<<eR.m02()<<"\n"
               <<eR.m10()<<" "<<eR.m11()<<" "<<eR.m12()<<"\n"
               <<eR.m20()<<" "<<eR.m21()<<" "<<eR.m22()<<"\n";
      std::cout<<"eR_eigen: \n"<<eR_eigen<<"\n";

      std::cout<<"m_er: \n"
               <<e->m_er.x()<<" "<<e->m_er.y()<<" "<<e->m_er.z()<<"\n";
      std::cout<<"m_er_eigen: \n"
               <<m_er_eigen.x()<<" "<<m_er_eigen.y()<<" "<<m_er_eigen.z()<<"\n";


      std::cout<<"m_Jrr1: \n"
               <<J->m_Jrr1.m00()<<" "<<J->m_Jrr1.m01()<<" "<<J->m_Jrr1.m02()<<"\n"
               <<J->m_Jrr1.m10()<<" "<<J->m_Jrr1.m11()<<" "<<J->m_Jrr1.m12()<<"\n"
               <<J->m_Jrr1.m20()<<" "<<J->m_Jrr1.m21()<<" "<<J->m_Jrr1.m22()<<"\n";

      std::cout<<"m_Jrr1_eigen: \n"<<m_Jrr1_eigen<<"\n";

      //v
            LA::AlignedVector3f dv = C2.m_v - C1.m_v;// (v_wj - v_wji + g*dt)
            if (!IMU_GRAVITY_EXCLUDED) {
                dv.z() += IMU_GRAVITY_MAGNITUDE * m_Tvg;
            }
            Eigen::Vector3f dv_eigen{dv[0],dv[1],dv[2]};

            Eigen::Matrix3f m_Jvr1_eigen = Tc1w_eigen*eigenskewSymmetric(dv_eigen);

#endif
#else
    const Rotation3D dR = GetRotationState(C1, C2);
    const LA::AlignedVector3f drbw = m_Jrbw * (C1.m_bw - m_bw);
    const Rotation3D eR = dR * m_RT, eRub = eR / Rotation3D(drbw);
    eRub.GetRodrigues(e->m_er, eps);
    Rotation3D::GetRodriguesJacobianInverse(e->m_er, J->m_Jrr1, eps);
    Rotation3D::GetRodriguesJacobian(drbw.GetMinus(), J->m_Jrbw1, eps);
    J->m_Jrr1.MakeMinus();
    J->m_Jrbw1 = J->m_Jrr1 * eR * J->m_Jrbw1 * m_Jrbw;
    J->m_Jrr1 = J->m_Jrr1 * C2.m_Cam_pose;

#endif

    e->m_ev = GetVelocityState(C1, C2);//Rciw*(v_wj - v_wji + g*dt)
    SkewSymmetricMatrix::AB(e->m_ev, C1.m_Cam_pose, J->m_Jvr1);//div(e_r)/div(Rciw) = [Rciw*(v_wj - v_wji + g*dt)]x * Rciw

    e->m_ev -= GetVelocityMeasurement(C1);//预积分的测量值,考虑了bias误差的影响m_v + m_Jvba * (bai - m_ba) + m_Jvbw * (bwi - m_bw)
    C1.m_Cam_pose.GetMinus(J->m_Jvv1);//-Rciw
    m_Jvba.GetMinus(J->m_Jvba1);//div(e_r)/div(bai) = -m_Jvba
    m_Jvbw.GetMinus(J->m_Jvbw1);//div(e_r)/div(bwi) = -m_Jvbw

    e->m_ep = C2.m_p - C1.m_p - C1.m_v * m_Tpv;//e_p = p_wcj - p_wci - v_wji*dt
    if (!IMU_GRAVITY_EXCLUDED) {
      e->m_ep.z() += IMU_GRAVITY_MAGNITUDE * m_Tpg;//e_p = p_wcj - p_wci - v_wji*dt+ g*dt
    }
    e->m_ep = C1.m_Cam_pose.GetAppliedRotation(e->m_ep);//e_p = Rciw*(p_wcj - p_wci - v_wji*dt + g*dt)
#ifdef CFG_IMU_FULL_COVARIANCE
    const LA::AlignedVector3f R21pu = dR.GetApplied(pu);//Rciw * Rcjw.t*tc0i
#else
    const LA::AlignedVector3f R21pu = dR.GetAppliedInversely(pu);
#endif
    e->m_ep += R21pu;//e_p = Rciw*(p_wcj - p_wci - v_wji*dt + 0.5*g*dt^2) + Rciw * Rcjw.t*tc0i
    //div(e_p)/div(Rciw) = [Rciw*(p_wcj - p_wci - v_wci*dt + 0.5*g*dt^2 + Rcjw.t*tc0i)]x * Rciw
    SkewSymmetricMatrix::AB(e->m_ep, C1.m_Cam_pose, J->m_Jpr1);
    e->m_ep -= pu; //e_p = Rciw*(p_wcj - p_wci - v_wji*dt + 0.5*g*dt^2) + Rciw * Rcjw.t*tc0i - tc0i
    e->m_ep -= GetPositionMeasurement(C1);//e_p = Rciw*(p_wcj - p_wci - v_wji*dt + 0.5*g*dt^2) + Rciw * (Rcjw.t - Rciw.t)*tc0i - (m_p + m_Jpba * (bai - m_ba) + m_Jpbw * (bwi - m_bw))
    C1.m_Cam_pose.GetMinus(J->m_Jpp1);//div(e_p)/div(p_wci) = -Rciw
    C1.m_Cam_pose.GetScaled(-m_Tpv, J->m_Jpv1);//div(e_p)/div(v_wi) = -Rciw * dt
    m_Jpba.GetMinus(J->m_Jpba1);//div(e_p)/div(bai) = -m_Jvba
    m_Jpbw.GetMinus(J->m_Jpbw1);//div(e_p)/div(bwi) = -m_Jvbw
    //div(e_p)/div(Rcjw) = [ - Rciw* Rcjw.t*tc0i]x * Rciw
    SkewSymmetricMatrix::ATB(R21pu, C1.m_Cam_pose, J->m_Jpr2);
    e->m_eba = C1.m_ba - C2.m_ba;
    e->m_ebw = C1.m_bw - C2.m_bw;

    //J->m_Jpp1.MakeZero();
    //J->m_Jpr1.MakeZero();
    //J->m_Jpba1.MakeZero();
    //J->m_Jpbw1.MakeZero();
    //J->m_Jpr2.MakeZero();
  }

//imu的预积分约束,之前优化的时候,优化变量都是相对于世界坐标系的,而边缘化时,优化变量是相对于相对运动最小的关键帧坐标系中的:也就是从W转到了K
//上面的上面有类似推导的注释了,这里就给结果了,这个函数是针对要边缘化的这帧不是关键帧的情况,那么就是ij的pose,motion还有g都要优化
// 不求对i pose的雅克比,因为i就是关键帧,约束还在关键帧和j帧帧中
//残差:e_r,e_v,e_p,e_ba,e_bw
//优化变量 Rcik,Rcjk,p_kci,p_kcj,,v_i,v_j,bai,baj,bwi,bwj,gw(东北天下的重力)
//// e_r = -ln{预积分的Rij * exp[Jrbw *(bwi - z_bw)]x * Rcjk * Rcik.t}v
//   div(e_r)/div(Rcik)  = - Jl_inv(-e_r)* eR * Rcik = - Jr_inv(e_r)* eR * Rcik
//   div(e_r)/div(Rcjk)   = Jr_inv(e_r)* eR * Rcik
//   div(e_r)/div(bwi)   = - Jr_inv(e_r)*预积分的Rij * Jr(-Jrbw *(bwi - z_bw))*Jrbw

////  e_v = Rcik*(Rcjk.t * v_j(是Rcjw*vw_j) - Rckw * gw*dt) - v_i(是Rciw*vw_i) - (m_v + m_Jvba * (bai - m_ba) + m_Jvbw * (bwi - m_bw))
//    div(e_v)/div(Rcik)  = Rcik*[Rcjk.t * v_j - Rckw * gw*t]x = [Rcik*Rcjk.t * v_j - Rckw * gw*t]x*Rcik
//    div(e_v)/div(Rcjk) = -Rcik*[Rcjk.t * v_j]x = -[Rcik  * Rcjk.t * v_j]x*Rcik
//    div(e_v)/div(v_i) = -I
//    div(e_v)/div(v_j) = Rcik*Rcjk.t
//    div(e_v)/div(bai) = -m_Jvba
//    div(e_v)/div(bwi) = -m_Jvbw
//    div(e_v)/div(gw) = Rcik*(- Rckw *exp[-th]x* gw*dt) = -Rciw*[gw]x*dt
//    代码里存的是div(e_v)/div(gw).t = - (Rciw*[gw*dt]x).t = = - [gw*dt]x* Rciw.t叉乘转换= - Rciw.t * [Rciw*gw*dt]x 固定z所以第行列是0

////  e_p = - tc0i  + Rcik *(Rcjk.t*tc0i + (pkj(是Rkw*pwi + tkw) - pki)(这里代码里是Rkw*(pwi-pwj)因为tkw的部分消去了) - Rckw * gw*0.5*dt^2) - v_i*dt- (m_p + m_Jpba * (bai - m_ba) + m_Jpbw * (bwi - m_bw))
//    div(e_p)/div(Rcik)  = Rcik * [Rcjk.t*tc0i + pkj - pki - Rckw * gw*0.5*dt^2]x
//    div(e_p)/div(Rcjk) = - Rcik * [Rcjk.t*tc0i]x = - [Rcik * Rcjk.t*tc0i]x * Rcik
//    div(e_p)/div(v_wi) = -I * dt
//    div(e_p)/div(pki) = -Rcik = -I
//    div(e_p)/div(pkj) = Rcik = I
//    div(e_p)/div(bai) = -m_Jvba
//    div(e_p)/div(bwi) = -m_Jvbw
//    div(e_p)/div(gw) = Rcik*(- Rckw *exp[-th]x* gw*0.5*dt^2) = -Rcik*Rckw*[gw*0.5*dt^2]x = -Rciw*[gw*0.5*dt^2]x 固定z所以第3列是0
//    代码里存的是div(e_p)/div(gw).t = - (Rciw*[gw*0.5*dt^2]x).t = = - [gw*0.5*dt^2]x* Rciw.t叉乘转换= - Rciw.t * [gw*0.5*dt^2]x 固定z所以第行列是0
//// e_ba = bai - baj
//// e_bw = bwi - bwj 这两个的雅克比就是i是1,j是-1
inline void GetErrorJacobian(const Camera &C1/*最老帧的状态*/, const Camera &C2/*次老帧的相机状态*/,
        const Point3D &pu,/*tc0_i*/const Rotation3D &Rg/*上一次滑窗时的Tc0w(参考关键帧)*/, Error *e, Jacobian::RelativeLF *J,/*因子*/
                               const float eps) const {
#ifdef CFG_IMU_FULL_COVARIANCE
    const Rotation3D dR = GetRotationState(C1, C2);//Rciw * Rcjw.t
    const LA::AlignedVector3f drbw = m_Jrbw * (C1.m_bw - m_bw);//
    //预积分的Rij * exp[Jrbw *(bwi - z_bw)]x * Rcjw * Rciw.t(数值上来说w和k这里是一样的)
    const Rotation3D eR = m_RT / Rotation3D(drbw, eps) / dR;
    eR.GetRodrigues(e->m_er, eps);
    Rotation3D::GetRodriguesJacobianInverse(e->m_er, J->m_Jrr1, eps);// J->m_Jrr1 = Jr_inv(e_r) 它这里用的是右乘雅克比
    Rotation3D::GetRodriguesJacobian(drbw.GetMinus(), J->m_Jrbw1, eps);//J->m_Jrbw1 = Jr(-Jrbw *(bwi - z_bw))
    J->m_Jrr1.MakeMinus();// J->m_Jrr1 = Jr_inv(e_r)
    //div(e_r)/div(bwi) = - Jr_inv(e_r)*预积分的Rij * Jr(-Jrbw *(bwi - z_bw))*Jrbw
    J->m_Jrbw1 = J->m_Jrr1 * m_RT * J->m_Jrbw1 * m_Jrbw;//
    J->m_Jrr1 = J->m_Jrr1 * eR;// J->m_Jrr1 = Jr_inv(e_r) * eR
    const Rotation3D R1T = Rg / C1.m_Cam_pose; // Rc0w(参考关键帧) * Rc0w(最老帧).t = Rc0(参考关键帧)c0(最老帧) = Rkci
    J->m_Jrr1 = LA::AlignedMatrix3x3f::GetABT(J->m_Jrr1, R1T);// J->m_Jrr1 = - Jr_inv(e_r)* eR * Rcik
#else
    const Rotation3D dR = GetRotationState(C1, C2);
    const LA::AlignedVector3f drbw = m_Jrbw * (C1.m_bw - m_bw);
    const Rotation3D eR = dR * m_RT, eRub = eR / Rotation3D(drbw, eps);
    eRub.GetRodrigues(e->m_er, eps);
    Rotation3D::GetRodriguesJacobianInverse(e->m_er, J->m_Jrr1, eps);
    Rotation3D::GetRodriguesJacobian(drbw.GetMinus(), J->m_Jrbw1, eps);
    J->m_Jrr1.MakeMinus();
    J->m_Jrbw1 = J->m_Jrr1 * eR * J->m_Jrbw1 * m_Jrbw;
    const Rotation3D R2T = Rg / C2.m_Cam_pose;
    J->m_Jrr1 = LA::AlignedMatrix3x3f::GetABT(J->m_Jrr1, R2T);
#endif

    C1.m_Cam_pose.ApplyRotation(C2.m_v, e->m_ev);//m_ev = Rciw * v_wj = Rcik  * Rcjk.t * v_j(定义成Rcjw *v_wj)- (m_v + m_Jvba * (bai - m_ba) + m_Jvbw * (bwi - m_bw))
#ifdef CFG_IMU_FULL_COVARIANCE
    J->m_Jvv2 = dR;//div(e_v)/div(v_j) = Rcik*Rcjk.t
#else
    dR.LA::AlignedMatrix3x3f::GetTranspose(J->m_Jvv2);
#endif
    LA::AlignedMatrix3x3f::Ab(J->m_Jvv2, pu, (float *) &e->m_ep);//m_ep = Rcik * Rcjk.t*tc0_i (tc0_i这里看上面的注释,有说过)
    //const Rotation3D R1 = Rotation3D(C1.m_Cam_pose) / Rg;
    const Rotation3D R1 = R1T.GetTranspose();//Rcik
    SkewSymmetricMatrix::ATB(e->m_ev, R1, J->m_Jvr2);//div(e_v)/div(Rcjk) = -[Rcik  * Rcjk.t * v_j]x*Rcik
    SkewSymmetricMatrix::ATB(e->m_ep, R1, J->m_Jpr2);//div(e_p)/div(Rcjk) = -[Rcik * Rcjk.t*tc0i]x * Rcik
    if (IMU_GRAVITY_EXCLUDED) {
      J->m_JvgT.Invalidate();
      J->m_JpgT.Invalidate();
    } else {
      const LA::AlignedVector3f g1 = C1.m_Cam_pose.GetColumn2();
      const LA::AlignedVector3f dv = g1 * (m_Tvg * IMU_GRAVITY_MAGNITUDE);
      e->m_ev += dv;//m_ev = Rciw * v_wj - Rciw*gw*dt = Rciw*(v_wj - gw*dt)
      SkewSymmetricMatrix::ATBT(C1.m_Cam_pose, dv, J->m_JvgT);//div(e_v)/div(gw).t = - Rciw.t * [Rciw*gw*dt]x
      const LA::AlignedVector3f dp = g1 * (m_Tpg * IMU_GRAVITY_MAGNITUDE);
      e->m_ep += dp;//
      SkewSymmetricMatrix::ATBT(C1.m_Cam_pose, dp, J->m_JpgT);//div(e_p)/div(gw).t = - Rciw.t * [gw*0.5*dt^2]x
    }
    SkewSymmetricMatrix::AB(e->m_ev, R1, J->m_Jvr1);
    const LA::AlignedVector3f v1 = C1.m_Cam_pose.GetAppliedRotation(C1.m_v);
    e->m_ev -= v1;//m_ev = Rciw*(v_wj - gw*dt) - Rciw*v_wi
    e->m_ev -= GetVelocityMeasurement(C1);//
    //J->m_Jvv1.MakeDiagonal(-1.0);
#ifdef CFG_DEBUG
    J->m_Jvv1.Invalidate();
#endif
    m_Jvba.GetMinus(J->m_Jvba1);//后面就不注释了,都能和我的注释对上
    m_Jvbw.GetMinus(J->m_Jvbw1);

    e->m_ep += C1.m_Cam_pose.GetAppliedRotation(C2.m_p - C1.m_p);
    SkewSymmetricMatrix::AB(e->m_ep, R1, J->m_Jpr1);
    e->m_ep -= v1 * m_Tpv;
    e->m_ep -= pu;
    e->m_ep -= GetPositionMeasurement(C1);
    R1.GetMinus(J->m_Jpp1);
    //J->m_Jpv1.MakeDiagonal(-m_Tpv);
#ifdef CFG_DEBUG
    J->m_Jpv1.Invalidate();
#endif
    m_Jpba.GetMinus(J->m_Jpba1);
    m_Jpbw.GetMinus(J->m_Jpbw1);

    e->m_eba = C1.m_ba - C2.m_ba;
    e->m_ebw = C1.m_bw - C2.m_bw;
  }
//imu的预积分约束,之前优化的时候,优化变量都是相对于世界坐标系的,而边缘化时,优化变量是相对于相对运动最小的关键帧坐标系中的:也就是从W转到了K
//上面的上面有类似推导的注释了,这里就给结果了,这个函数是针对要边缘化的这帧同时也是关键帧的情况,那么它的最近帧就是它自己,即Rcik,pcik都是0,
// 不求对i pose的雅克比,因为i就是关键帧,约束还在关键帧和j帧帧中
//残差:e_r,e_v,e_p,e_ba,e_bw
//优化变量 Rcik,Rcjk,p_kci,p_kcj,v_i,v_j,bai,baj,bwi,bwj,gw(东北天下的重力)
//// e_r = -ln{预积分的Rij * exp[Jrbw *(bwi - z_bw)]x * Rcjk * Rcik.t}v
//   div(e_r)/div(Rcik)  = - Jr_inv(e_r)* eR * Rcik
//   div(e_r)/div(Rcjk)   = Jr_inv(e_r)* eR * Rcik
//   div(e_r)/div(bwi)   = - Jr_inv(e_r)*预积分的Rij * Jr(-Jrbw *(bwi - z_bw))*Jrbw

////  e_v = Rcik(在这种情况是I)*(Rcjk.t * v_j(是Rcjw*vw_j) - Rckw * gw*dt) - v_i(是Rciw*vw_i) - (m_v + m_Jvba * (bai - m_ba) + m_Jvbw * (bwi - m_bw))
//    div(e_v)/div(Rcik)  = Rcik*[Rcjk.t * v_j - Rckw * gw*t]x = [Rcjk.t * v_j - Rckw * gw*t]x
//    div(e_v)/div(Rcjk) = -Rcik*[Rcjk.t * v_j]x = -[Rcjk.t * v_j]x
//    div(e_v)/div(v_i) = -I
//    div(e_v)/div(v_j) = Rcik*Rcjk.t = Rcjk.t
//    div(e_v)/div(bai) = -m_Jvba
//    div(e_v)/div(bwi) = -m_Jvbw
//    div(e_v)/div(gw) = Rcik*(- Rckw *exp[-th]x* gw*dt) = -Rcik*Rckw*[gw]x*dt  = - Rckw*[gw*dt]x 固定z所以第3列是0
//    代码里存的是div(e_v)/div(gw).t = - (Rckw*[gw*dt]x).t =  -[gw*dt]x* Rckw.t叉乘转换= -Rckw.t * [Rckw*gw*dt]x 固定z所以第行列是0

////  e_p = - tc0i  + Rcik(在这种情况是I) *(Rcjk.t*tc0i + (pkj(是Rkw*pwi + tkw) - pki)(这里代码里是Rkw*(pwi-pwj)因为tkw的部分消去了) - Rckw * gw*0.5*dt^2) - v_i*dt- (m_p + m_Jpba * (bai - m_ba) + m_Jpbw * (bwi - m_bw))
//    div(e_p)/div(Rcik)  = Rcik * [Rcjk.t*tc0i + pkj - pki - Rckw * gw*0.5*dt^2]x = [Rcjk.t*tc0i + pkj - pki - Rckw * gw*0.5*dt^2]x
//    div(e_p)/div(Rcjk) = - Rcik * [Rcjk.t*tc0i]x = -[Rcjk.t*tc0i]x
//    div(e_p)/div(v_wi) = -I * dt
//    div(e_p)/div(pki) = -Rcik = -I
//    div(e_p)/div(pkj) = Rcik = I
//    div(e_p)/div(bai) = -m_Jvba
//    div(e_p)/div(bwi) = -m_Jvbw
//    div(e_p)/div(gw) = Rcik*(- Rckw *exp[-th]x* gw*0.5*dt^2) = -Rcik*Rckw*[gw*0.5*dt^2]x = -Rckw*[gw*0.5*dt^2]x 固定z所以第3列是0
//    代码里存的是div(e_p)/div(gw).t = - (Rckw*[gw*0.5*dt^2]x).t =-[gw*0.5*dt^2]x* Rckw.t叉乘转换=- Rckw.t * [Rckw*gw*0.5*dt^2]x 固定z所以第行列是0
//// e_ba = bai - baj
//// e_bw = bwi - bwj 这两个的雅克比就是i是1,j是-1
  inline void GetErrorJacobian(const Camera &C1/*前一帧状态*/, const Camera &C2/*后一帧状态*/, const Point3D &pu,/*tc0_i*/
                               Error *e, Jacobian::RelativeKF *J, const float eps) const {
#ifdef CFG_IMU_FULL_COVARIANCE
    const Rotation3D dR = GetRotationState(C1, C2);//Rciw * Rcjw.t
    const LA::AlignedVector3f drbw = m_Jrbw * (C1.m_bw - m_bw);
    const Rotation3D eR = m_RT / Rotation3D(drbw, eps) / dR;//预积分的Rij * (exp[-Jrbw *(bwi - z_bw)]x).t * (Rciw * Rcjw.t).t
                                                            //预积分的Rij * exp[Jrbw *(bwi - z_bw)]x * Rcjw * Rciw.t(数值上来说w和k这里是一样的)
    eR.GetRodrigues(e->m_er, eps);//这里SO3转so3是-th
    Rotation3D::GetRodriguesJacobianInverse(e->m_er, J->m_Jrr2, eps);// J->m_Jrr2 = Jr_inv(e_r) 它这里用的是右乘雅克比
    Rotation3D::GetRodriguesJacobian(drbw.GetMinus(), J->m_Jrbw1, eps);//J->m_Jrbw1 = Jr(-Jrbw *(bwi - z_bw))
    J->m_Jrbw1 = J->m_Jrr2 * m_RT * J->m_Jrbw1 * m_Jrbw;//div(e_r)/div(bwi) = Jr_inv(e_r)*预积分的Rij * Jr(-Jrbw *(bwi - z_bw))*Jrbw
    J->m_Jrbw1.MakeMinus();//div(e_r)/div(bwi) = -Jr_inv(e_r)*预积分的Rij * Jr(-Jrbw *(bwi - z_bw))*Jrbw
    J->m_Jrr2 = J->m_Jrr2 * eR;//div(e_r)/div(Rcjk)   = Jr_inv(e_r)* eR * Rcik(这种情况为I)
#else
    const Rotation3D dR = GetRotationState(C1, C2);
    const LA::AlignedVector3f drbw = m_Jrbw * (C1.m_bw - m_bw);
    const Rotation3D eR = dR * m_RT, eRub = eR / Rotation3D(drbw, eps);
    eRub.GetRodrigues(e->m_er, eps);
    Rotation3D::GetRodriguesJacobianInverse(e->m_er, J->m_Jrr2, eps);
    Rotation3D::GetRodriguesJacobian(drbw.GetMinus(), J->m_Jrbw1, eps);
    J->m_Jrbw1 = J->m_Jrr2 * eR * J->m_Jrbw1 * m_Jrbw;
    J->m_Jrbw1.MakeMinus();
    const Rotation3D R2T = Rotation3D(C1.m_Cam_pose) / C2.m_Cam_pose;
    J->m_Jrr2 = LA::AlignedMatrix3x3f::GetABT(J->m_Jrr2, R2T);
#endif

    C1.m_Cam_pose.ApplyRotation(C2.m_v, e->m_ev);//m_ev = Rciw * v_wj = Rcik(在这种情况是I) * Rcjk.t * v_j(定义成Rcjw *v_wj)= Rcjk.t * v_j
#ifdef CFG_IMU_FULL_COVARIANCE
    J->m_Jvv2 = dR;//div(e_v)/div(v_wj) = Rciw * Rcjw.t = Rcik * Rcjk.t = Rcjk.t
#else
    dR.LA::AlignedMatrix3x3f::GetTranspose(J->m_Jvv2);
#endif
    LA::AlignedMatrix3x3f::Ab(J->m_Jvv2, pu, (float *) &e->m_ep);//m_ep = Rcik * Rcjk.t*tc0_i = Rcjk.t*tc0_i (tc0_i这里看上面的注释,有说过)
    SkewSymmetricMatrix::GetTranspose(e->m_ev, J->m_Jvr2);//div(e_v)/div(Rcjk) = -[Rcjk.t * v_j]x
    SkewSymmetricMatrix::GetTranspose(e->m_ep, J->m_Jpr2);//div(e_p)/div(Rcjw) = -[ Rcjk.t*tc0_i]x 这个好像有些出入啊
    if (IMU_GRAVITY_EXCLUDED) {
      J->m_JvgT.Invalidate();
      J->m_JpgT.Invalidate();
    } else {
      const LA::AlignedVector3f g1 = C1.m_Cam_pose.GetColumn2();//重力gw = [0,0,-9.81] ，
      const LA::AlignedVector3f dv = g1 * (m_Tvg * IMU_GRAVITY_MAGNITUDE);//考虑了重力扰动,对应的是这个-Rckw*gw*dt,负号挪到g里后Rckw*gw*dt
      e->m_ev += dv;//m_ev = Rckw * v_wj - Rckw*gw*dt = Rcik*(Rcjk.t * v_j - * gw*dt)
      SkewSymmetricMatrix::ATBT(C1.m_Cam_pose, dv, J->m_JvgT);//div(e_v)/div(gw).t =  Rckw.t*([-Rckw*gw*dt]x).t = -Rckw.t * [Rckw*gw*dt]x
      const LA::AlignedVector3f dp = g1 * (m_Tpg * IMU_GRAVITY_MAGNITUDE);
      e->m_ep += dp;//m_ep =  Rcjk.t*tc0_i - Rckw * gw*0.5*dt^2
      SkewSymmetricMatrix::ATBT(C1.m_Cam_pose, dp, J->m_JpgT);// div(e_v)/div(gw).t = -Rckw.t * [Rckw*gw*0.5*dt^2]x
    }
    const LA::AlignedVector3f v1 = C1.m_Cam_pose.GetAppliedRotation(C1.m_v);//残差具体的和我注释对应,就不写了
    e->m_ev -= v1;
    e->m_ev -= GetVelocityMeasurement(C1);
    //J->m_Jvv1.MakeDiagonal(-1.0);
#ifdef CFG_DEBUG
    J->m_Jvv1.Invalidate();
#endif
    m_Jvba.GetMinus(J->m_Jvba1);
    m_Jvbw.GetMinus(J->m_Jvbw1);

    e->m_ep += C1.m_Cam_pose.GetAppliedRotation(C2.m_p - C1.m_p);
    e->m_ep -= v1 * m_Tpv;
    e->m_ep -= pu;
    e->m_ep -= GetPositionMeasurement(C1);
    //J->m_Jpv1.MakeDiagonal(-m_Tpv);
    //J->m_Jpp2.MakeIdentity();
#ifdef CFG_DEBUG
    J->m_Jpv1.Invalidate();
    J->m_Jpp2.Invalidate();
#endif
    m_Jpba.GetMinus(J->m_Jpba1);
    m_Jpbw.GetMinus(J->m_Jpbw1);

    e->m_eba = C1.m_ba - C2.m_ba;
    e->m_ebw = C1.m_bw - C2.m_bw;

  }
  inline void GetFactor(const float w, const Camera &C1, const Camera &C2, const Point3D &pu,
                        Factor *A, Camera::Factor::Binary *A12, Factor::Auxiliary::Global *U,
                        const float eps) const {
    GetErrorJacobian(C1/*前一帧状态*/, C2/*后一帧状态*/, pu/*tc0_i*/, &A->m_Je.m_e/*motion部分的残差*/, &A->m_Je.m_J/*motion部分的雅克比*/, eps);
    A->m_F = GetCost(w, A->m_Je.m_e/*motion部分的残差*/);//计算一下cost:马氏距离下的残差
    //这一步在求H = J.t*W*J，存储在m_A,和-b = J.t*W*e存储在m_b，不过没有对得上具体位置,这个也不重要,最终的H里的顺序是p r v ba bw，
    U->Set(A->m_Je.m_J/*motion部分的雅克比*/, A->m_Je.m_e/*motion部分的残差*/, w, m_W/*预积分的协方差的逆得到的信息矩阵*/, m_Tpv/*dt*/);
    U->Get(&A->m_A11/*H中i时刻的c,m和i时刻的c,m,以及对应的-b*/, &A->m_A22/*H中j时刻的c,m和j时刻的c,m,以及对应的-b*/, A12/*H中i时刻的c,m和j时刻的c,m*/);
  }
  inline void GetFactor(const float w, const Camera &C1/*最老帧的状态*/, const Camera &C2/*次老帧的相机状态*/,
          const Point3D &pu/*tc0_i*/,const Rotation3D &Rg, /*上一次滑窗时的Tc0w(参考关键帧)*/
          Error *e, Jacobian::RelativeLF *J,
                        Factor::Auxiliary::RelativeLF *U/*因子*/, const float eps) const {
    GetErrorJacobian(C1/*最老帧的状态*/, C2/*次老帧的相机状态*/, pu/*tc0_i*/, Rg/*上一次滑窗时的Tc0w(参考关键帧)*/, e, J, eps);
    U->Set(*J, *e, w, m_W, m_Tpv);
  }
  inline void GetFactor(const float w, const Camera &C1, const Camera &C2, const Point3D &pu,
                        Error *e, Jacobian::RelativeKF *J, Factor::Auxiliary::RelativeKF *U,
                        const float eps) const {
    GetErrorJacobian(C1/*最老帧的状态*/, C2/*次老帧的相机状态*/, pu/*tc0_i*/, e/*残差*/, J/*雅克比*/, eps);
    U->Set(*J/*雅克比*/, *e, w, m_W/*协方差*/, m_Tpv/*dt*/);//这一步在求H = J.t*W*J，存储在m_A,和-b = J.t*W*e存储在m_b
  }
  inline float GetCost(const float w, const Error &e) const {
    return GetCost(w, m_W/*预积分得到的信息矩阵*/, e);
  }
  static inline float GetCost(const float w, const Weight &W/*信息矩阵*/, const Error &e) {//马氏距离下的残差
#ifdef CFG_IMU_FULL_COVARIANCE
    LA::AlignedVector3f We;
    float F = 0.0f;
    const LA::AlignedVector3f *_e = (LA::AlignedVector3f *) &e;
    for (int i = 0; i < 5; ++i) {
      We.MakeZero();
      const LA::AlignedMatrix3x3f *Wi = W[i];
      for (int j = 0; j < 5; ++j) {
        LA::AlignedMatrix3x3f::AddAbTo(Wi[j], _e[j], (float *) &We);
      }
      F += _e[i].Dot(We);
    }
    return w * F;
#else
    return gyr * (LA::SymmetricMatrix3x3f::MahalanobisDistance(W.m_Wr, e.m_er) +
                LA::SymmetricMatrix3x3f::MahalanobisDistance(W.m_Wv, e.m_ev) +
                LA::SymmetricMatrix3x3f::MahalanobisDistance(W.m_Wp, e.m_ep) +
                W.m_wba * e.m_eba.SquaredLength() +
                W.m_wbw * e.m_ebw.SquaredLength());
#endif
  }
  inline float GetCost(const float w, const ErrorJacobian &Je, const LA::AlignedVector3f *xp1,
                       const LA::AlignedVector3f *xr1, const LA::AlignedVector3f *xv1,
                       const LA::AlignedVector3f *xba1, const LA::AlignedVector3f *xbw1,
                       const LA::AlignedVector3f *xp2, const LA::AlignedVector3f *xr2,
                       const LA::AlignedVector3f *xv2, const LA::AlignedVector3f *xba2,
                       const LA::AlignedVector3f *xbw2, Error &e) const {
    GetError(Je, xp1, xr1, xv1, xba1, xbw1, xp2, xr2, xv2, xba2, xbw2, e);
    return GetCost(w, e);
  }
  inline void GetReduction(const float w, const Factor &A, const Camera &C1, const Camera &C2,
                           const Point3D &pu, const LA::AlignedVector3f *xp1,
                           const LA::AlignedVector3f *xr1, const LA::AlignedVector3f *xv1,
                           const LA::AlignedVector3f *xba1, const LA::AlignedVector3f *xbw1,
                           const LA::AlignedVector3f *xp2, const LA::AlignedVector3f *xr2, 
                           const LA::AlignedVector3f *xv2, const LA::AlignedVector3f *xba2,
                           const LA::AlignedVector3f *xbw2, Reduction &Ra, Reduction &Rp,
                           const float eps) const {
    GetError(C1, C2, pu, Ra.m_e, eps);
    GetError(A.m_Je, xp1, xr1, xv1, xba1, xbw1, xp2, xr2, xv2, xba2, xbw2, Rp.m_e);
    Ra.m_dF = A.m_F - (Ra.m_F = GetCost(w, Ra.m_e));
    Rp.m_dF = A.m_F - (Rp.m_F = GetCost(w, Rp.m_e));
  }

  inline bool AssertEqual(const Delta &D, const int verbose = 1, const std::string str = "",
    const bool normWeight = true) const {
    bool scc = true;
    scc = m_ba.AssertEqual(D.m_ba, verbose, str + ".m_ba") && scc;
    scc = m_bw.AssertEqual(D.m_bw, verbose, str + ".m_bw") && scc;
    scc = m_RT.AssertEqual(D.m_RT, verbose, str + ".m_R") && scc;
    scc = m_v.AssertEqual(D.m_v, verbose, str + ".m_v") && scc;
    scc = m_p.AssertEqual(D.m_p, verbose, str + ".m_p") && scc;
    scc = m_Jrbw.AssertEqual(D.m_Jrbw, verbose, str + ".m_Jrbw") && scc;
    scc = m_Jvba.AssertEqual(D.m_Jvba, verbose, str + ".m_Jvba") && scc;
    scc = m_Jvbw.AssertEqual(D.m_Jvbw, verbose, str + ".m_Jvbw") && scc;
    scc = m_Jpba.AssertEqual(D.m_Jpba, verbose, str + ".m_Jpba") && scc;
    scc = m_Jpbw.AssertEqual(D.m_Jpbw, verbose, str + ".m_Jpbw") && scc;
    scc = m_W.AssertEqual(D.m_W, verbose, str + ".m_W", normWeight) && scc;
    scc = UT::AssertEqual(m_Tvg, D.m_Tvg, verbose, str + ".m_Tvg") && scc;
    scc = UT::AssertEqual(m_Tpv, D.m_Tpv, verbose, str + ".m_Tpv") && scc;
    scc = UT::AssertEqual(m_Tpg, D.m_Tpg, verbose, str + ".m_Tpg") && scc;
    return scc;
  }
  inline void Print(const bool e = false) const {
    UT::PrintSeparator();
    m_ba.Print("  ba = ", e, false, true);
    m_bw.Print("  bw = ", e, false, true);
    m_RT.Print("  RT = ", e);
    m_v.Print("   v = ", e, false, true);
    m_p.Print("   p = ", e, false, true);
    m_Jrbw.Print("Jrbw = ", e);
    m_Jvba.Print("Jvba = ", e);
    m_Jvbw.Print("Jvbw = ", e);
    m_Jpba.Print("Jpba = ", e);
    m_Jpbw.Print("Jpbw = ", e);
    if (e) {
      UT::Print(" Tvg = %e\n", m_Tvg);
    } else {
      UT::Print(" Tvg = %f\n", m_Tvg);
    }
  }
 public:
  Measurement m_u1/*上一帧最后一个imu测量*/, m_u2;
  LA::AlignedVector3f m_ba, m_bw;//bias
  Rotation3D m_RT;//Rc0i_c0j
  LA::AlignedVector3f m_v, m_p;//vc0i_c0j,pc0i_c0j
  LA::AlignedMatrix3x3f m_Jrbw, m_Jvba, m_Jvbw, m_Jpba, m_Jpbw;//i时刻泰勒展开,然后递推得到预积分后状态量对i时刻ba,bw的雅克比
  Weight m_W;//信息矩阵
  float m_Tvg, m_Tpv/*两帧之间的时间戳*/, m_Tpg/*连加以后近似于m_Tpv^2*/, m_r;
#ifdef CFG_DEBUG_EIGEN
 public:
  class EigenTransition : public EigenMatrix15x15f {
   public:
    class DD : public EigenMatrix9x9f {
     public:
      inline DD() {}
      inline DD(const Eigen::Matrix<float, 9, 9> &e_P) {
        *((EigenMatrix9x9f *) this) = e_P;
      }
      inline DD(const Transition::DD &F) { Set(F); }
      inline void Set(const Transition::DD &F) {
        setIdentity();
        //block<3, 3>(3, 0) = EigenMatrix3x3f(F.m_Fvr);
        //block<3, 3>(6, 0) = EigenMatrix3x3f(F.m_Fpr);
        //block<3, 3>(6, 3) = EigenMatrix3x3f(F.m_Fpv);
        block<3, 3>(3, 0) = EigenMatrix3x3f(F.m_Fvr.GetAlignedMatrix3x3f());
        block<3, 3>(6, 0) = EigenMatrix3x3f(F.m_Fpr.GetAlignedMatrix3x3f());
        block<3, 3>(6, 3) = EigenMatrix3x3f(F.m_Fpv[0]);
      }
      inline bool AssertEqual(const Transition::DD &F, const int verbose = 1,
                              const std::string str = "") const {
        bool scc = true;
        const EigenMatrix3x3f e_I = EigenMatrix3x3f::Identity();
        scc = EigenMatrix3x3f(block<3, 3>(0, 0)).AssertEqual(e_I, verbose, str + ".Frr") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(0, 3)).AssertZero(verbose, str + ".Frv") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(0, 6)).AssertZero(verbose, str + ".Frp") && scc;
        //scc = EigenMatrix3x3f(block<3, 3>(3, 0)).AssertEqual(F.m_Fvr, verbose, str + ".Fvr") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 0)).AssertEqual(F.m_Fvr.GetAlignedMatrix3x3f(), verbose, str + ".Fvr") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 3)).AssertEqual(e_I, verbose, str + ".Fvv") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 6)).AssertZero(verbose, str + ".Fvp") && scc;
        //scc = EigenMatrix3x3f(block<3, 3>(6, 0)).AssertEqual(F.m_Fpr, verbose, str + ".Fpr") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(6, 0)).AssertEqual(F.m_Fpr.GetAlignedMatrix3x3f(), verbose, str + ".Fpr") && scc;
        //scc = EigenMatrix3x3f(block<3, 3>(6, 3)).AssertEqual(F.m_Fpv, verbose, str + ".Fpv") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(6, 3)).AssertEqual(EigenMatrix3x3f(F.m_Fpv[0]), verbose, str + ".Fpv") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(6, 6)).AssertEqual(e_I, verbose, str + ".Fpp") && scc;
        return scc;
      }
    };
    class DB : public EigenMatrix9x6f {
     public:
      inline DB() {}
      inline DB(const Eigen::Matrix<float, 9, 6> &e_P) {
        *((EigenMatrix9x6f *) this) = e_P;
      }
      inline DB(const Transition::DB &F) { Set(F); }
      inline void Set(const Transition::DB &F) {
        block<3, 3>(0, 0) = EigenMatrix3x3f::Zero();
        block<3, 3>(0, 3) = EigenMatrix3x3f(F.m_Frbw);
        block<3, 3>(3, 0) = EigenMatrix3x3f(F.m_Fvba);
        block<3, 3>(3, 3) = EigenMatrix3x3f(F.m_Fvbw);
        block<3, 3>(6, 0) = EigenMatrix3x3f(F.m_Fpba);
        block<3, 3>(6, 3) = EigenMatrix3x3f(F.m_Fpbw);
      }
      inline bool AssertEqual(const Transition::DB &F, const int verbose = 1,
                              const std::string str = "") const {
        bool scc = true;
        scc = EigenMatrix3x3f(block<3, 3>(0, 0)).AssertZero(verbose, str + ".Frba") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(0, 3)).AssertEqual(F.m_Frbw, verbose, str + ".Frbw") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 0)).AssertEqual(F.m_Fvba, verbose, str + ".Fvba") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 3)).AssertEqual(F.m_Fvbw, verbose, str + ".Fvbw") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(6, 0)).AssertEqual(F.m_Fpba, verbose, str + ".Fpba") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(6, 3)).AssertEqual(F.m_Fpbw, verbose, str + ".Fpbw") && scc;
        return scc;
      }
    };
   public:
    inline EigenTransition() {}
    inline EigenTransition(const Eigen::Matrix<float, 15, 15, Eigen::RowMajor> &e_F) {
      *((EigenMatrix15x15f *) this) = e_F;
    }
    inline void Set(const Transition &F) {
      block<9, 9>(0, 0) = DD(F.m_Fdd);
      block<9, 6>(0, 9) = DB(F.m_Fdb);
      block<6, 9>(9, 0).setZero();
      block<6, 6>(9, 9).setIdentity();
    }
    inline bool AssertEqual(const Transition &F, const int verbose = 1,
                            const std::string str = "") const {
      bool scc = true;
      scc = DD(block<9, 9>(0, 0)).AssertEqual(F.m_Fdd, verbose, str) && scc;
      scc = DB(block<9, 6>(0, 9)).AssertEqual(F.m_Fdb, verbose, str) && scc;
      scc = EigenMatrix6x9f(block<6, 9>(9, 0)).AssertZero(verbose, str) && scc;
      scc = EigenMatrix6x6f(block<6, 6>(9, 9)).AssertEqual(EigenMatrix6x6f::Identity(), verbose, str) && scc;
      return scc;
    }
  };
  class EigenCovariance : public EigenMatrix15x15f {
   public:
    class DD : public EigenMatrix9x9f {
     public:
      inline DD() {}
      inline DD(const Eigen::Matrix<float, 9, 9> &e_P) {
        *((EigenMatrix9x9f *) this) = e_P;
      }
      inline DD(const Covariance::DD &P) { Set(P); }
      inline void Set(const Covariance::DD &P) {
        block<3, 3>(0, 0) = EigenMatrix3x3f(P.m_Prr);
        block<3, 3>(0, 3) = EigenMatrix3x3f(P.m_Prv);
        block<3, 3>(0, 6) = EigenMatrix3x3f(P.m_Prp);
        block<3, 3>(3, 0) = EigenMatrix3x3f(P.m_Pvr);
        block<3, 3>(3, 3) = EigenMatrix3x3f(P.m_Pvv);
        block<3, 3>(3, 6) = EigenMatrix3x3f(P.m_Pvp);
        block<3, 3>(6, 0) = EigenMatrix3x3f(P.m_Ppr);
        block<3, 3>(6, 3) = EigenMatrix3x3f(P.m_Ppv);
        block<3, 3>(6, 6) = EigenMatrix3x3f(P.m_Ppp);
      }
      inline void Get(Covariance::DD *P) const {
        P->m_Prr = EigenMatrix3x3f(block<3, 3>(0, 0)).GetAlignedMatrix3x3f();
        P->m_Prv = EigenMatrix3x3f(block<3, 3>(0, 3)).GetAlignedMatrix3x3f();
        P->m_Prp = EigenMatrix3x3f(block<3, 3>(0, 6)).GetAlignedMatrix3x3f();
        P->m_Pvr = EigenMatrix3x3f(block<3, 3>(3, 0)).GetAlignedMatrix3x3f();
        P->m_Pvv = EigenMatrix3x3f(block<3, 3>(3, 3)).GetAlignedMatrix3x3f();
        P->m_Pvp = EigenMatrix3x3f(block<3, 3>(3, 6)).GetAlignedMatrix3x3f();
        P->m_Ppr = EigenMatrix3x3f(block<3, 3>(6, 0)).GetAlignedMatrix3x3f();
        P->m_Ppv = EigenMatrix3x3f(block<3, 3>(6, 3)).GetAlignedMatrix3x3f();
        P->m_Ppp = EigenMatrix3x3f(block<3, 3>(6, 6)).GetAlignedMatrix3x3f();
      }
      inline bool AssertEqual(const Covariance::DD &P, const int verbose = 1,
                              const std::string str = "") const {
        bool scc = true;
        scc = EigenMatrix3x3f(block<3, 3>(0, 0)).AssertEqual(P.m_Prr, verbose, str + ".Prr") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(0, 3)).AssertEqual(P.m_Prv, verbose, str + ".Prv") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(0, 6)).AssertEqual(P.m_Prp, verbose, str + ".Prp") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 0)).AssertEqual(P.m_Pvr, verbose, str + ".Pvr") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 3)).AssertEqual(P.m_Pvv, verbose, str + ".Pvv") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 6)).AssertEqual(P.m_Pvp, verbose, str + ".Pvp") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(6, 0)).AssertEqual(P.m_Ppr, verbose, str + ".Ppr") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(6, 3)).AssertEqual(P.m_Ppv, verbose, str + ".Ppv") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(6, 6)).AssertEqual(P.m_Ppp, verbose, str + ".Ppp") && scc;
        return scc;
      }
    };
    class DB : public EigenMatrix9x6f {
     public:
      inline DB() {}
      inline DB(const Eigen::Matrix<float, 9, 6> &e_P) {
        *((EigenMatrix9x6f *) this) = e_P;
      }
      inline DB(const Covariance::DB &P) { Set(P); }
      inline void Set(const Covariance::DB &P) {
        block<3, 3>(0, 0) = EigenMatrix3x3f::Zero();
        block<3, 3>(0, 3) = EigenMatrix3x3f(P.m_Prbw);
        block<3, 3>(3, 0) = EigenMatrix3x3f(P.m_Pvba);
        block<3, 3>(3, 3) = EigenMatrix3x3f(P.m_Pvbw);
        block<3, 3>(6, 0) = EigenMatrix3x3f(P.m_Ppba);
        block<3, 3>(6, 3) = EigenMatrix3x3f(P.m_Ppbw);
      }
      inline void Get(Covariance::DB *P) const {
        P->m_Prbw = EigenMatrix3x3f(block<3, 3>(0, 3)).GetAlignedMatrix3x3f();
        P->m_Pvba = EigenMatrix3x3f(block<3, 3>(3, 0)).GetAlignedMatrix3x3f();
        P->m_Pvbw = EigenMatrix3x3f(block<3, 3>(3, 3)).GetAlignedMatrix3x3f();
        P->m_Ppba = EigenMatrix3x3f(block<3, 3>(6, 0)).GetAlignedMatrix3x3f();
        P->m_Ppbw = EigenMatrix3x3f(block<3, 3>(6, 3)).GetAlignedMatrix3x3f();
      }
      inline bool AssertEqual(const Covariance::DB &P, const int verbose = 1,
                              const std::string str = "") const {
        bool scc = true;
        scc = EigenMatrix3x3f(block<3, 3>(0, 0)).AssertZero(verbose, str + ".Prba") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(0, 3)).AssertEqual(P.m_Prbw, verbose, str + ".Prbw") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 0)).AssertEqual(P.m_Pvba, verbose, str + ".Pvba") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 3)).AssertEqual(P.m_Pvbw, verbose, str + ".Pvbw") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(6, 0)).AssertEqual(P.m_Ppba, verbose, str + ".Ppba") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(6, 3)).AssertEqual(P.m_Ppbw, verbose, str + ".Ppbw") && scc;
        return scc;
      }
    };
    class BD : public EigenMatrix6x9f {
     public:
      inline BD() {}
      inline BD(const Eigen::Matrix<float, 6, 9> &e_P) {
        *((EigenMatrix6x9f *) this) = e_P;
      }
      inline BD(const Covariance::BD &P) { Set(P); }
      inline void Set(const Covariance::BD &P) {
        block<3, 3>(0, 0) = EigenMatrix3x3f::Zero();
        block<3, 3>(0, 3) = EigenMatrix3x3f(P.m_Pbav);
        block<3, 3>(0, 6) = EigenMatrix3x3f(P.m_Pbap);
        block<3, 3>(3, 0) = EigenMatrix3x3f(P.m_Pbwr);
        block<3, 3>(3, 3) = EigenMatrix3x3f(P.m_Pbwv);
        block<3, 3>(3, 6) = EigenMatrix3x3f(P.m_Pbwp);
      }
      inline void Get(Covariance::BD *P) const {
        P->m_Pbav = EigenMatrix3x3f(block<3, 3>(0, 3)).GetAlignedMatrix3x3f();
        P->m_Pbap = EigenMatrix3x3f(block<3, 3>(0, 6)).GetAlignedMatrix3x3f();
        P->m_Pbwr = EigenMatrix3x3f(block<3, 3>(3, 0)).GetAlignedMatrix3x3f();
        P->m_Pbwv = EigenMatrix3x3f(block<3, 3>(3, 3)).GetAlignedMatrix3x3f();
        P->m_Pbwp = EigenMatrix3x3f(block<3, 3>(3, 6)).GetAlignedMatrix3x3f();
      }
      inline bool AssertEqual(const Covariance::BD &P, const int verbose = 1,
                              const std::string str = "") const {
        bool scc = true;
        scc = EigenMatrix3x3f(block<3, 3>(0, 0)).AssertZero(verbose, str + ".Pbar") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(0, 3)).AssertEqual(P.m_Pbav, verbose, str + ".Pbav") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(0, 6)).AssertEqual(P.m_Pbap, verbose, str + ".Pbap") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 0)).AssertEqual(P.m_Pbwr, verbose, str + ".Pbwr") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 3)).AssertEqual(P.m_Pbwv, verbose, str + ".Pbwv") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 6)).AssertEqual(P.m_Pbwp, verbose, str + ".Pbwp") && scc;
        return scc;
      }
    };
    class BB : public EigenMatrix6x6f {
     public:
      inline BB() {}
      inline BB(const Eigen::Matrix<float, 6, 6> &e_P) {
        *((EigenMatrix6x6f *) this) = e_P;
      }
      inline BB(const Covariance::BB &P) { Set(P); }
      inline void Set(const Covariance::BB &P) {
        EigenMatrix6x6f &e_P = *this;
        e_P.setZero();
        e_P(0, 0) = e_P(1, 1) = e_P(2, 2) = P.m_Pbaba;
        e_P(3, 3) = e_P(4, 4) = e_P(5, 5) = P.m_Pbwbw;
      }
      inline void Get(Covariance::BB *P) const {
        const EigenMatrix6x6f &e_P = *this;
        P->m_Pbaba = e_P(0, 0);
        P->m_Pbwbw = e_P(3, 3);
      }
      inline bool AssertEqual(const Covariance::BB &P, const int verbose = 1,
                              const std::string str = "") const {
        bool scc = true;
        scc = EigenMatrix3x3f(block<3, 3>(0, 0)).AssertEqual(EigenMatrix3x3f(P.m_Pbaba),
                                                             verbose, str + ".Pbaba") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(0, 3)).AssertZero(verbose, str + ".Pbabw") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 0)).AssertZero(verbose, str + ".Pbwba") && scc;
        scc = EigenMatrix3x3f(block<3, 3>(3, 3)).AssertEqual(EigenMatrix3x3f(P.m_Pbwbw),
                                                             verbose, str + ".Pbwbw") && scc;
        return scc;
      }
    };
   public:
    inline EigenCovariance() {}
    inline EigenCovariance(const Eigen::Matrix<float, 15, 15, Eigen::RowMajor> &e_P) {
      *((EigenMatrix15x15f *) this) = e_P;
    }
    inline void Set(const Covariance &P) {
      block<9, 9>(0, 0) = DD(P.m_Pdd);
      block<9, 6>(0, 9) = DB(P.m_Pdb);
      //block<6, 9>(9, 0) = BD(P.m_Pbd);
      block<6, 9>(9, 0) = block<9, 6>(0, 9).transpose();
      block<6, 6>(9, 9) = BB(P.m_Pbb);
    }
    inline void Get(Covariance *P) const {
      DD(block<9, 9>(0, 0)).Get(&P->m_Pdd);
      DB(block<9, 6>(0, 9)).Get(&P->m_Pdb);
      BB(block<6, 6>(9, 9)).Get(&P->m_Pbb);
    }
    inline bool AssertEqual(const Covariance &P, const int verbose = 1,
                            const std::string str = "") const {
      bool scc = true;
      scc = DD(block<9, 9>(0, 0)).AssertEqual(P.m_Pdd, verbose, str) && scc;
      scc = DB(block<9, 6>(0, 9)).AssertEqual(P.m_Pdb, verbose, str) && scc;
      //scc = BD(block<6, 9>(9, 0)).AssertEqual(P.m_Pbd, verbose, str) && scc;
      scc = BB(block<6, 6>(9, 9)).AssertEqual(P.m_Pbb, verbose, str) && scc;
      return scc;
    }
  };
  class EigenWeight : public EigenMatrix15x15f {
   public:
    inline EigenWeight() {}
    inline EigenWeight(const Eigen::Matrix<float, 15, 15, Eigen::RowMajor> &e_W) { *this = e_W; }
    inline EigenWeight(const Weight &W) { Set(W); }
    inline EigenWeight(const float gyr, const Weight &W) { Set(gyr, W); }
    inline void operator = (const Eigen::Matrix<float, 15, 15, Eigen::RowMajor> &e_W) {
      *((Eigen::Matrix<float, 15, 15, Eigen::RowMajor> *) this) = e_W;
    }
    inline void Set(const EigenCovariance &e_P) {
      *this = e_P.ldlt().solve(EigenMatrix15x15f::Identity());
    }
    inline void Set(const Weight &W) {
      setZero();
#ifdef CFG_IMU_FULL_COVARIANCE
      for (int i = 0, _i = 0; i < 5; ++i, _i += 3) {
        const LA::AlignedMatrix3x3f *Wi = W[i];
        for (int j = 0, _j = 0; j < 5; ++j, _j += 3) {
          block<3, 3>(_i, _j) = EigenMatrix3x3f(Wi[j]);
        }
      }
#else
      EigenMatrix15x15f &e_W = *this;
      e_W.block<3, 3>(0, 0) = EigenMatrix3x3f(W.m_Wr);
      e_W.block<3, 3>(3, 3) = EigenMatrix3x3f(W.m_Wv);
      e_W.block<3, 3>(6, 6) = EigenMatrix3x3f(W.m_Wp);
      e_W(9, 9) = e_W(10, 10) = e_W(11, 11) = W.m_wba;
      e_W(12, 12) = e_W(13, 13) = e_W(14, 14) = W.m_wbw;
#endif
    }
    inline void Set(const float gyr, const Weight &W) { Set(W); *this *= gyr; }
    inline bool AssertEqual(const Weight &W, const int verbose = 1,
                            const std::string str = "") const {
      bool scc = true;
      LA::AlignedMatrix3x3f Wij;
      for (int i = 0, _i = 0; i < 5; ++i, _i += 3) {
        for (int j = 0, _j = 0; j < 5; ++j, _j += 3) {
#ifdef CFG_IMU_FULL_COVARIANCE
          Wij = W[i][j];
#else
          if (i == j) {
            switch (i) {
            case 0: Wij = W.m_Wr; break;
            case 1: Wij = W.m_Wv; break;
            case 2: Wij = W.m_Wp; break;
            case 3: Wij.SetDiagonal(W.m_wba); break;
            case 4: Wij.SetDiagonal(W.m_wbw); break;
            }
          } else {
            Wij.MakeZero();
          }
#endif
          scc = EigenMatrix3x3f(block<3, 3>(_i, _j)).AssertEqual(Wij, verbose,
            str + UT::String(".W[%d][%d]", i, j)) && scc;
        }
      }
      return scc;
    }
  };
  class EigenError {
   public:
    inline EigenError() {}
    inline EigenError(const Error &e) { Set(e); }
    inline void Set(const Error &e) {
      m_er = EigenVector3f(e.m_er);
      m_ev = EigenVector3f(e.m_ev);
      m_ep = EigenVector3f(e.m_ep);
      m_eba = EigenVector3f(e.m_eba);
      m_ebw = EigenVector3f(e.m_ebw);
    }
    inline void Get(EigenVector15f *e_e) const {
      e_e->block<3, 1>(0, 0) = m_er;
      e_e->block<3, 1>(3, 0) = m_ev;
      e_e->block<3, 1>(6, 0) = m_ep;
      e_e->block<3, 1>(9, 0) = m_eba;
      e_e->block<3, 1>(12, 0) = m_ebw;
    }
    inline bool AssertEqual(const Error &e, const int verbose = 1,
                            const std::string str = "",
                            const float epsAbs = 0.0f, const float epsRel = 0.0f) const {
      EigenError e_e;
      e_e.Set(e);
      return AssertEqual(e_e, verbose, str, epsAbs, epsRel);
    }
    inline bool AssertEqual(const EigenError &e_e, const int verbose = 1,
                            const std::string str = "",
                            const float epsAbs = 0.0f, const float epsRel = 0.0f) const {
      bool scc = true;
      scc = m_er.AssertEqual(e_e.m_er, verbose, str + ".m_er", epsAbs, epsRel) && scc;
      scc = m_ev.AssertEqual(e_e.m_ev, verbose, str + ".m_ev", epsAbs, epsRel) && scc;
      scc = m_ep.AssertEqual(e_e.m_ep, verbose, str + ".m_ep", epsAbs, epsRel) && scc;
      scc = m_eba.AssertEqual(e_e.m_eba, verbose, str + ".m_eba", epsAbs, epsRel) && scc;
      scc = m_ebw.AssertEqual(e_e.m_ebw, verbose, str + ".m_ebw", epsAbs, epsRel) && scc;
      return scc;
    }
   public:
    EigenVector3f m_er, m_ev, m_ep, m_eba, m_ebw;
  };
  class EigenJacobian {
   public:
    class Gravity {
     public:
      inline void Set(const Jacobian::Gravity &J) {
        m_JvgT = J.m_JvgT;
        m_JpgT = J.m_JpgT;
      }
      inline void Get(EigenMatrix15x2f *e_J) const {
        e_J->setZero();
        e_J->block<3, 2>(3, 0) = m_JvgT.transpose();
        e_J->block<3, 2>(6, 0) = m_JpgT.transpose();
      }
      inline bool AssertEqual(const Jacobian::Gravity &J, const int verbose = 1,
                              const std::string str = "") const {
        EigenJacobian::Gravity e_J;
        e_J.Set(J);
        return AssertEqual(e_J, verbose, str);
      }
      inline bool AssertEqual(const EigenJacobian::Gravity &e_J, const int verbose = 1,
                              const std::string str = "") const {
        bool scc = true;
        scc = m_JvgT.AssertEqual(e_J.m_JvgT, verbose, str + ".m_JvgT") && scc;
        scc = m_JpgT.AssertEqual(e_J.m_JpgT, verbose, str + ".m_JpgT") && scc;
        return scc;
      }
     public:
      EigenMatrix2x3f m_JvgT, m_JpgT;
    };
    class Global {
     public:
      inline void Set(const Jacobian::FirstMotion &J) {
        m_Jr.setZero();
        m_Jr.block<3, 3>(0, 12) = EigenMatrix3x3f(J.m_Jrbw1);
        m_Jv.setZero();
        m_Jv.block<3, 3>(0, 6) = EigenMatrix3x3f(J.m_Jvv1);
        m_Jv.block<3, 3>(0, 9) = EigenMatrix3x3f(J.m_Jvba1);
        m_Jv.block<3, 3>(0, 12) = EigenMatrix3x3f(J.m_Jvbw1);
        m_Jp.setZero();
        m_Jp.block<3, 3>(0, 6) = EigenMatrix3x3f(J.m_Jpv1);
        m_Jp.block<3, 3>(0, 9) = EigenMatrix3x3f(J.m_Jpba1);
        m_Jp.block<3, 3>(0, 12) = EigenMatrix3x3f(J.m_Jpbw1);
        m_Jba.setZero();
        m_Jba.block<3, 3>(0, 9) = EigenMatrix3x3f::Identity();
        m_Jbw.setZero();
        m_Jbw.block<3, 3>(0, 12) = EigenMatrix3x3f::Identity();
      }
      inline void Set(const Jacobian::Global &J) {
        Set(Jacobian::FirstMotion(J));
        m_Jr.block<3, 3>(0, 3) = EigenMatrix3x3f(J.m_Jrr1);
        m_Jr.block<3, 3>(0, 18) = -EigenMatrix3x3f(J.m_Jrr1);
        m_Jv.block<3, 3>(0, 3) = EigenMatrix3x3f(J.m_Jvr1);
        m_Jv.block<3, 3>(0, 21) = EigenMatrix3x3f(J.m_Jvv1.GetMinus());
        m_Jp.block<3, 3>(0, 0) = EigenMatrix3x3f(J.m_Jpp1);
        m_Jp.block<3, 3>(0, 3) = EigenMatrix3x3f(J.m_Jpr1);
        m_Jp.block<3, 3>(0, 15) = EigenMatrix3x3f(J.m_Jpp1.GetMinus());
        m_Jp.block<3, 3>(0, 18) = EigenMatrix3x3f(J.m_Jpr2);
        m_Jba.block<3, 3>(0, 24) = -EigenMatrix3x3f::Identity();
        m_Jbw.block<3, 3>(0, 27) = -EigenMatrix3x3f::Identity();
      }
      inline void Get(EigenMatrix15x30f *e_J) const {
        e_J->block<3, 30>(0, 0) = m_Jr;
        e_J->block<3, 30>(3, 0) = m_Jv;
        e_J->block<3, 30>(6, 0) = m_Jp;
        e_J->block<3, 30>(9, 0) = m_Jba;
        e_J->block<3, 30>(12, 0) = m_Jbw;
      }
      inline bool AssertEqual(const Jacobian::Global &J, const int verbose = 1,
                              const std::string str = "") const {
        EigenJacobian::Global e_J;
        e_J.Set(J);
        return AssertEqual(e_J, verbose, str);
      }
      inline bool AssertEqual(const EigenJacobian::Global &e_J, const int verbose = 1,
                              const std::string str = "") const {
        bool scc = true;
        scc = m_Jr.AssertEqual(e_J.m_Jr, verbose, str + ".m_Jr") && scc;
        scc = m_Jv.AssertEqual(e_J.m_Jv, verbose, str + ".m_Jv") && scc;
        scc = m_Jp.AssertEqual(e_J.m_Jp, verbose, str + ".m_Jp") && scc;
        scc = m_Jba.AssertEqual(e_J.m_Jba, verbose, str + ".m_Jba") && scc;
        scc = m_Jbw.AssertEqual(e_J.m_Jbw, verbose, str + ".m_Jbw") && scc;
        return scc;
      }
     public:
      EigenMatrix3x30f m_Jr, m_Jv, m_Jp, m_Jba, m_Jbw;
    };
    class RelativeLF : public Gravity, public Global {
     public:
      inline RelativeLF() {}
      inline RelativeLF(const Jacobian::RelativeLF &J, const float Tpv) { Set(J, Tpv); }
      inline void Set(const Jacobian::RelativeLF &J, const float Tpv) {
        Gravity::Set(J);
        Global::Set(Jacobian::Global(J));
        m_Jv.block<3, 3>(0, 6) = EigenMatrix3x3f(-EigenMatrix3x3f::Identity());
        m_Jv.block<3, 3>(0, 18) = EigenMatrix3x3f(J.m_Jvr2);
        m_Jv.block<3, 3>(0, 21) = EigenMatrix3x3f(J.m_Jvv2);
        m_Jp.block<3, 3>(0, 6) = EigenMatrix3x3f(-EigenMatrix3x3f::Identity() * Tpv);
      }
      inline void Get(EigenMatrix15x2f *e_Jg, EigenMatrix15x30f *e_Jc) const {
        Gravity::Get(e_Jg);
        Global::Get(e_Jc);
      }
      inline bool AssertEqual(const Jacobian::RelativeLF &J, const float Tpv,
                              const int verbose = 1, const std::string str = "") const {
        EigenJacobian::RelativeLF e_J;
        e_J.Set(J, Tpv);
        return AssertEqual(e_J, verbose, str);
      }
      inline bool AssertEqual(const EigenJacobian::RelativeLF &e_J, const int verbose = 1,
                              const std::string str = "") const {
        bool scc = true;
        scc = Gravity::AssertEqual(e_J, verbose, str) && scc;
        scc = Global::AssertEqual(e_J, verbose, str) && scc;
        return scc;
      }
    };
    class RelativeKF : public RelativeLF {
     public:
      inline RelativeKF() {}
      inline RelativeKF(const Jacobian::RelativeKF &J, const float Tpv) { Set(J, Tpv); }
      inline void Set(const Jacobian::RelativeKF &J, const float Tpv) {
        Gravity::Set(Jacobian::Gravity(J));
        Global::Set(Jacobian::FirstMotion(J));
        m_Jr.block<3, 3>(0, 18) = EigenMatrix3x3f(J.m_Jrr2);
        m_Jv.block<3, 3>(0, 6) = EigenMatrix3x3f(-EigenMatrix3x3f::Identity());
        m_Jv.block<3, 3>(0, 18) = EigenMatrix3x3f(J.m_Jvr2);
        m_Jv.block<3, 3>(0, 21) = EigenMatrix3x3f(J.m_Jvv2);
        m_Jp.block<3, 3>(0, 6) = EigenMatrix3x3f(-EigenMatrix3x3f::Identity() * Tpv);
        m_Jp.block<3, 3>(0, 15) = EigenMatrix3x3f::Identity();
        m_Jp.block<3, 3>(0, 18) = EigenMatrix3x3f(J.m_Jpr2);
        m_Jba.block<3, 3>(0, 24) = -EigenMatrix3x3f::Identity();
        m_Jbw.block<3, 3>(0, 27) = -EigenMatrix3x3f::Identity();
      }
      inline bool AssertEqual(const Jacobian::RelativeKF &J, const float Tpv,
                              const int verbose = 1, const std::string str = "") const {
        EigenJacobian::RelativeKF e_J;
        e_J.Set(J, Tpv);
        return RelativeLF::AssertEqual(e_J, verbose, str);
      }
    };
  };
  class EigenErrorJacobian {
   public:
    inline void Set(const ErrorJacobian &Je) {
      m_e.Set(Je.m_e);
      m_J.Set(Je.m_J);
    }
    inline bool AssertEqual(const ErrorJacobian &Je, const int verbose = 1,
                            const std::string str = "") const {
      bool scc = true;
      scc = m_e.AssertEqual(Je.m_e, verbose, str) && scc;
      scc = m_J.AssertEqual(Je.m_J, verbose, str) && scc;
      return scc;
    }
   public:
    EigenError m_e;
    EigenJacobian::Global m_J;
  };
  class EigenFactor {
   public:
    class Global {
     public:
      inline void operator *= (const float gyr) {
        m_Ac1c1 *= gyr; m_Ac1m1 *= gyr; m_Ac1c2 *= gyr; m_Ac1m2 *= gyr; m_bc1 *= gyr;
        m_Am1m1 *= gyr; m_Am1c2 *= gyr; m_Am1m2 *= gyr; m_bm1 *= gyr;
        m_Ac2c2 *= gyr; m_Ac2m2 *= gyr; m_bc2 *= gyr;
        m_Am2m2 *= gyr; m_bm2 *= gyr;
      }
      inline void Set(const EigenMatrix30x31f &A, const float F) {
        Set(EigenMatrix30x30f(A.block<30, 30>(0, 0)), EigenVector30f(A.block<30, 1>(0, 30)), F);
      }
      inline void Set(const EigenMatrix30x30f &A, const EigenVector30f &b, const float F) {
        m_Ac1c1 = A.block<6, 6>(0, 0);
        m_Ac1m1 = A.block<6, 9>(0, 6);
        m_Ac1c2 = A.block<6, 6>(0, 15);
        m_Ac1m2 = A.block<6, 9>(0, 21);
        m_bc1 = b.block<6, 1>(0, 0);
        m_Am1m1 = A.block<9, 9>(6, 6);
        m_Am1c2 = A.block<9, 6>(6, 15);
        m_Am1m2 = A.block<9, 9>(6, 21);
        m_bm1 = b.block<9, 1>(6, 0);
        m_Ac2c2 = A.block<6, 6>(15, 15);
        m_Ac2m2 = A.block<6, 9>(15, 21);
        m_bc2 = b.block<6, 1>(15, 0);
        m_Am2m2 = A.block<9, 9>(21, 21);
        m_bm2 = b.block<9, 1>(21, 0);
        m_F = F;
      }
      inline void Set(const Factor::Unitary &A11, const Factor::Unitary &A22,
                      const Camera::Factor::Binary &A12, const float F) {
        m_Ac1c1 = A11.m_Acc.m_A;
        m_Ac1m1 = A11.m_Acm;
        m_Ac1c2 = A12.m_Acc;
#ifdef CFG_IMU_FULL_COVARIANCE
        m_Ac1m2 = A12.m_Acm;
#else
        m_Ac1m2.setZero();
        m_Ac1m2.block<3, 3>(3, 0) = EigenMatrix3x3f(A12.m_Acm.m_Arv);
#endif
        m_bc1 = A11.m_Acc.m_b;

        m_Am1m1 = A11.m_Amm.m_A;
        m_Am1c2 = A12.m_Amc;
#ifdef CFG_IMU_FULL_COVARIANCE
        m_Am1m2 = A12.m_Amm;
#else
        m_Am1m2.setZero();
        m_Am1m2.block<9, 3>(0, 0) = EigenMatrix9x3f(A12.m_Amm.m_Amv);
        m_Am1m2.block<3, 3>(3, 3) = EigenMatrix3x3f(A12.m_Amm.m_Ababa);
        m_Am1m2.block<3, 3>(6, 6) = EigenMatrix3x3f(A12.m_Amm.m_Abwbw);
#endif
        m_bm1 = A11.m_Amm.m_b;

        m_Ac2c2 = A22.m_Acc.m_A;
        m_Ac2m2 = A22.m_Acm;
        m_bc2 = A22.m_Acc.m_b;

        m_Am2m2 = A22.m_Amm.m_A;
        m_bm2 = A22.m_Amm.m_b;

        m_F = F;
      }
      inline void Set(const Factor &A, const Camera::Factor::Binary &A12) {
        Set(A.m_A11, A.m_A22, A12, A.m_F);
      }
      inline void Get(EigenMatrix30x30f &e_A, EigenVector30f &e_b) const {
        e_A.block<6, 6>(0, 0) = m_Ac1c1;
        e_A.block<6, 9>(0, 6) = m_Ac1m1;
        e_A.block<6, 6>(0, 15) = m_Ac1c2;
        e_A.block<6, 9>(0, 21) = m_Ac1m2;
        e_b.block<6, 1>(0, 0) = m_bc1;
        e_A.block<9, 9>(6, 6) = m_Am1m1;
        e_A.block<9, 6>(6, 15) = m_Am1c2;
        e_A.block<9, 9>(6, 21) = m_Am1m2;
        e_b.block<9, 1>(6, 0) = m_bm1;
        e_A.block<6, 6>(15, 15) = m_Ac2c2;
        e_A.block<6, 9>(15, 21) = m_Ac2m2;
        e_b.block<6, 1>(15, 0) = m_bc2;
        e_A.block<9, 9>(21, 21) = m_Am2m2;
        e_b.block<9, 1>(21, 0) = m_bm2;
        e_A.SetLowerFromUpper();
      }
      inline bool AssertEqual(const Factor &A, const Camera::Factor::Binary &A12,
                              const int verbose = 1, const std::string str = "") const {
        Global e_A;
        e_A.Set(A, A12);
        return AssertEqual(e_A, verbose, str);
      }
      inline bool AssertEqual(const Factor::Unitary &A11, const Factor::Unitary &A22,
                              const Camera::Factor::Binary &A12, const float F,
                              const int verbose = 1, const std::string str = "") const {
        Global e_A;
        e_A.Set(A11, A22, A12, F);
        return AssertEqual(e_A, verbose, str);
      }
      inline bool AssertEqual(const Global &A, const int verbose = 1,
                              const std::string str = "") const {
        bool scc = true;
        scc = m_Ac1c1.AssertEqual(A.m_Ac1c1, verbose, str + ".m_Ac1c1") && scc;
        scc = m_Ac1m1.AssertEqual(A.m_Ac1m1, verbose, str + ".m_Ac1m1") && scc;
        scc = m_Ac1c2.AssertEqual(A.m_Ac1c2, verbose, str + ".m_Ac1c2") && scc;
        scc = m_Ac1m2.AssertEqual(A.m_Ac1m2, verbose, str + ".m_Ac1m2") && scc;
        scc = m_bc1.AssertEqual(A.m_bc1, verbose, str + ".m_bc1") && scc;

        scc = m_Am1m1.AssertEqual(A.m_Am1m1, verbose, str + ".m_Am1m1") && scc;
        scc = m_Am1c2.AssertEqual(A.m_Am1c2, verbose, str + ".m_Am1c2") && scc;
        scc = m_Am1m2.AssertEqual(A.m_Am1m2, verbose, str + ".m_Am1m2") && scc;
        scc = m_bm1.AssertEqual(A.m_bm1, verbose, str + ".m_bm1") && scc;

        scc = m_Ac2c2.AssertEqual(A.m_Ac2c2, verbose, str + ".m_Ac2c2") && scc;
        scc = m_Ac2m2.AssertEqual(A.m_Ac2m2, verbose, str + ".m_Ac2m2") && scc;
        scc = m_bc2.AssertEqual(A.m_bc2, verbose, str + ".m_bc2") && scc;

        scc = m_Am2m2.AssertEqual(A.m_Am2m2, verbose, str + ".m_Am2m2") && scc;
        scc = m_bm2.AssertEqual(A.m_bm2, verbose, str + ".m_bm2") && scc;

        scc = UT::AssertEqual(m_F, A.m_F, verbose, str + ".m_F") && scc;
        return scc;
      }


     public:
      EigenMatrix6x6f m_Ac1c1;//H中r1x1r的因子 我现在都是说的行列的索引
      EigenMatrix6x9f m_Ac1m1;
      EigenMatrix6x6f m_Ac1c2;
      EigenMatrix6x9f m_Ac1m2;
      EigenVector6f m_bc1;
      EigenMatrix9x9f m_Am1m1;
      EigenMatrix9x6f m_Am1c2;
      EigenMatrix9x9f m_Am1m2;
      EigenVector9f m_bm1;
      EigenMatrix6x6f m_Ac2c2;
      EigenMatrix6x9f m_Ac2m2;
      EigenVector6f m_bc2;
      EigenMatrix9x9f m_Am2m2;
      EigenVector9f m_bm2;
      float m_F;
    };
    class RelativeLF : public Global {
     public:
      inline void Set(const Factor::Auxiliary::RelativeLF &A, const float F) {
#ifdef CFG_IMU_FULL_COVARIANCE
        Factor::Unitary A11, A22;
        Camera::Factor::Binary A12;
        A.Get(&A11, &A22, &A12);
        Global::Set(A11, A22, A12, F);
        m_Agg = EigenMatrix2x2f(A.m_Agg);
        m_Agc1 = EigenMatrix2x6f(A.m_Agc[0], A.m_Agc[1]);
        m_Agm1 = EigenMatrix2x9f(A.m_Agc[2], A.m_Agc[3], A.m_Agc[4]);
        m_Agc2 = EigenMatrix2x6f(A.m_Agc[5], A.m_Agc[6]);
        m_Agm2 = EigenMatrix2x9f(A.m_Agc[7], A.m_Agc[8], A.m_Agc[9]);
        m_bg = EigenVector2f(A.m_bg);
#else
        m_Ac1c1 = EigenMatrix6x6f(A.m_Ap1p1, A.m_Ap1r1, A.m_Ar1r1);
        m_Ac1m1 = EigenMatrix6x9f(A.m_Ap1v1, A.m_Ap1ba1, A.m_Ap1bw1,
                                  A.m_Ar1v1, A.m_Ar1ba1, A.m_Ar1bw1);
        m_Ac1c2 = EigenMatrix6x6f(A.m_Ap1p2, A.m_Ap1r2, A.m_Ar1p2, A.m_Ar1r2);
        m_Ac1m2.setZero();
        m_Ac1m2.block<3, 3>(3, 0) = EigenMatrix3x3f(A.m_Ar1v2);
        m_bc1 = EigenVector6f(A.m_bp1, A.m_br1);
        m_Am1m1 = EigenMatrix9x9f(A.m_Av1v1, A.m_Av1ba1, A.m_Av1bw1, A.m_Aba1ba1, A.m_Aba1bw1,
                                  A.m_Abw1bw1);
        m_Am1c2 = EigenMatrix9x6f(A.m_Av1p2, A.m_Av1r2, A.m_Aba1p2, A.m_Aba1r2, A.m_Abw1p2,
                                  A.m_Abw1r2);
        m_Am1m2.setZero();
        m_Am1m2.block<3, 3>(0, 0) = EigenMatrix3x3f(A.m_Av1v2);
        m_Am1m2.block<3, 3>(3, 0) = EigenMatrix3x3f(A.m_Aba1v2);
        m_Am1m2.block<3, 3>(3, 3) = EigenMatrix3x3f(A.m_Aba1ba2);
        m_Am1m2.block<3, 3>(6, 0) = EigenMatrix3x3f(A.m_Abw1v2);
        m_Am1m2.block<3, 3>(6, 6) = EigenMatrix3x3f(A.m_Abw1bw2);
        m_bm1 = EigenVector9f(A.m_bv1, A.m_bba1, A.m_bbw1);
        m_Ac2c2 = EigenMatrix6x6f(A.m_Ap2p2, A.m_Ap2r2, A.m_Ar2r2);
        m_Ac2m2.setZero();
        m_Ac2m2.block<3, 3>(3, 0) = EigenMatrix3x3f(A.m_Ar2v2);
        m_bc2 = EigenVector6f(A.m_bp2, A.m_br2);
        m_Am2m2.setZero();
        m_Am2m2.block<3, 3>(0, 0) = EigenMatrix3x3f(A.m_Av2v2);
        m_Am2m2.block<3, 3>(3, 3) = EigenMatrix3x3f(A.m_Aba2ba2);
        m_Am2m2.block<3, 3>(6, 6) = EigenMatrix3x3f(A.m_Abw2bw2);
        m_bm2 = EigenVector9f(A.m_bv2, A.m_bba2, A.m_bbw2);
        m_F = F;
        m_Agg = EigenMatrix2x2f(A.m_Agg);
        m_Agc1 = EigenMatrix2x6f(A.m_Agp1, A.m_Agr1);
        m_Agm1 = EigenMatrix2x9f(A.m_Agv1, A.m_Agba1, A.m_Agbw1);
        m_Agc2 = EigenMatrix2x6f(A.m_Agp2, A.m_Agr2);
        m_Agm2.setZero();
        m_Agm2.block<2, 3>(0, 0) = EigenMatrix2x3f(A.m_Agv2);
        m_bg = EigenVector2f(A.m_bg);
#endif
      }
      inline void Get(EigenMatrix2x2f &e_Agg, EigenMatrix2x30f &e_Agc, EigenMatrix30x30f &e_Acc,
                      EigenVector2f &e_bg, EigenVector30f &e_bc) const {
        Global::Get(e_Acc, e_bc);
        e_Agg = m_Agg;
        e_Agc.block<2, 6>(0, 0) = m_Agc1;
        e_Agc.block<2, 9>(0, 6) = m_Agm1;
        e_Agc.block<2, 6>(0, 15) = m_Agc2;
        e_Agc.block<2, 9>(0, 21) = m_Agm2;
        e_bg = m_bg;
      }
      inline bool AssertEqual(const Factor::Auxiliary::RelativeLF &A, const int verbose = 1,
                              const std::string str = "") const {
        RelativeLF e_A;
        e_A.Set(A, m_F);
        return AssertEqual(e_A, verbose, str);
      }
      inline bool AssertEqual(const RelativeLF &e_A, const int verbose = 1,
                              const std::string str = "") const {
        bool scc = true;
        scc = Global::AssertEqual(e_A, verbose, str) && scc;
        scc = m_Agg.AssertEqual(e_A.m_Agg, verbose, str + ".m_Agg") && scc;
        scc = m_Agc1.AssertEqual(e_A.m_Agc1, verbose, str + ".m_Agc1") && scc;
        scc = m_Agm1.AssertEqual(e_A.m_Agm1, verbose, str + ".m_Agm1") && scc;
        scc = m_Agc2.AssertEqual(e_A.m_Agc2, verbose, str + ".m_Agc2") && scc;
        scc = m_Agm2.AssertEqual(e_A.m_Agm2, verbose, str + ".m_Agm2") && scc;
        scc = m_bg.AssertEqual(e_A.m_bg, verbose, str + ".m_bg") && scc;
        return scc;
      }
     public:
      EigenMatrix2x2f m_Agg;//重力和重力的H
      EigenMatrix2x6f m_Agc1;
      EigenMatrix2x9f m_Agm1;
      EigenMatrix2x6f m_Agc2;
      EigenMatrix2x9f m_Agm2;
      EigenVector2f m_bg;
    };
    class RelativeKF : public RelativeLF {
     public:
      inline void Set(const Factor::Auxiliary::RelativeKF &A, const float F) {
        m_Ac1c1.setZero();
        m_Ac1m1.setZero();
        m_Ac1c2.setZero();
        m_Ac1m2.setZero();
        m_bc1.setZero();
#ifdef CFG_IMU_FULL_COVARIANCE
        Factor::Unitary A11, A22;
        Camera::Factor::Binary A12;
        A.Get(&A11, &A22, &A12);
        Global::Set(A11, A22, A12, F);
        m_Agg = EigenMatrix2x2f(A.m_Agg);
        m_Agc1.setZero();
        m_Agm1 = EigenMatrix2x9f(A.m_Agc[0], A.m_Agc[1], A.m_Agc[2]);
        m_Agc2 = EigenMatrix2x6f(A.m_Agc[3], A.m_Agc[4]);
        m_Agm2 = EigenMatrix2x9f(A.m_Agc[5], A.m_Agc[6], A.m_Agc[7]);
        m_bg = EigenVector2f(A.m_bg);
#else
        m_Am1m1 = EigenMatrix9x9f(A.m_Av1v1, A.m_Av1ba1, A.m_Av1bw1, A.m_Aba1ba1, A.m_Aba1bw1,
                                  A.m_Abw1bw1);
        m_Am1c2 = EigenMatrix9x6f(A.m_Av1p2, A.m_Av1r2, A.m_Aba1p2, A.m_Aba1r2, A.m_Abw1p2,
                                  A.m_Abw1r2);
        m_Am1m2.setZero();
        m_Am1m2.block<3, 3>(0, 0) = EigenMatrix3x3f(A.m_Av1v2);
        m_Am1m2.block<3, 3>(3, 0) = EigenMatrix3x3f(A.m_Aba1v2);
        m_Am1m2.block<3, 3>(3, 3) = EigenMatrix3x3f(A.m_Aba1ba2);
        m_Am1m2.block<3, 3>(6, 0) = EigenMatrix3x3f(A.m_Abw1v2);
        m_Am1m2.block<3, 3>(6, 6) = EigenMatrix3x3f(A.m_Abw1bw2);
        m_bm1 = EigenVector9f(A.m_bv1, A.m_bba1, A.m_bbw1);
        m_Ac2c2 = EigenMatrix6x6f(A.m_Ap2p2, A.m_Ap2r2, A.m_Ar2r2);
        m_Ac2m2.setZero();
        m_Ac2m2.block<3, 3>(3, 0) = EigenMatrix3x3f(A.m_Ar2v2);
        m_bc2 = EigenVector6f(A.m_bp2, A.m_br2);
        m_Am2m2.setZero();
        m_Am2m2.block<3, 3>(0, 0) = EigenMatrix3x3f(A.m_Av2v2);
        m_Am2m2.block<3, 3>(3, 3) = EigenMatrix3x3f(A.m_Aba2ba2);
        m_Am2m2.block<3, 3>(6, 6) = EigenMatrix3x3f(A.m_Abw2bw2);
        m_bm2 = EigenVector9f(A.m_bv2, A.m_bba2, A.m_bbw2);
        m_F = F;
        m_Agg = EigenMatrix2x2f(A.m_Agg);
        m_Agc1.setZero();
        m_Agm1 = EigenMatrix2x9f(A.m_Agv1, A.m_Agba1, A.m_Agbw1);
        m_Agc2 = EigenMatrix2x6f(A.m_Agp2, A.m_Agr2);
        m_Agm2.setZero();
        m_Agm2.block<2, 3>(0, 0) = EigenMatrix2x3f(A.m_Agv2);
        m_bg = EigenVector2f(A.m_bg);
#endif
      }
      inline bool AssertEqual(const Factor::Auxiliary::RelativeKF &A, const int verbose = 1,
                              const std::string str = "") const {
        RelativeKF e_A;
        e_A.Set(A, m_F);
        return RelativeLF::AssertEqual(e_A, verbose, str);
      }
    };
  };
  void EigenGetErrorJacobian(const Camera &C1, const Camera &C2, const Point3D &pu,
                             EigenError *e_e, EigenJacobian::Global *e_J, const float eps) const;
  void EigenGetErrorJacobian(const Camera &C1, const Camera &C2, const Point3D &pu,
                             const Rotation3D &Rg, EigenError *e_e, EigenJacobian::RelativeLF *e_J,
                             const float eps) const;
  void EigenGetErrorJacobian(const Camera &C1, const Camera &C2, const Point3D &pu,
                             EigenError *e_e, EigenJacobian::RelativeKF *e_J,
                             const float eps) const;
  void EigenGetFactor(const float gyr, const Camera &C1, const Camera &C2, const Point3D &pu,
                      EigenFactor::Global *e_A, const float eps) const;
  void EigenGetFactor(const float gyr, const Camera &C1, const Camera &C2, const Point3D &pu,
                      const Rotation3D &Rg, EigenFactor::RelativeLF *e_A, const float eps) const;
  void EigenGetFactor(const float gyr, const Camera &C1, const Camera &C2, const Point3D &pu,
                      EigenFactor::RelativeKF *e_A, const float eps) const;
  static void EigenGetFactor(const float gyr, const Weight &W, const EigenJacobian::Global &e_J,
                             const EigenError &e_e, EigenFactor::Global *e_A);
  static void EigenGetFactor(const float gyr, const Weight &W, const EigenJacobian::RelativeLF &e_J,
                             const EigenError &e_e, EigenFactor::RelativeLF *e_A);
  static EigenError EigenGetError(const EigenErrorJacobian &e_Je, const EigenVector30f e_x);
  float EigenGetCost(const float gyr, const Camera &C1, const Camera &C2, const Point3D &pu,
                     const EigenVector6f &e_xc1, const EigenVector9f &e_xm1,
                     const EigenVector6f &e_xc2, const EigenVector9f &e_xm2,
                     const float eps) const;
#endif
};

void InitializeCamera(const AlignedVector<Measurement> &us, Camera &C);
void PreIntegrate(const AlignedVector<Measurement> &us, const float t1, const float t2,
                  const Camera &C1, Delta *D, AlignedVector<float> *work, const bool jac/* = true*/,
                  const Measurement *u1/* = NULL*/, const Measurement *u2/* = NULL*/,
                  const float eps);
void PreIntegrate(const Measurement *us, const int N, const float t1, const float t2,
                  const Camera &C1, Delta *D, AlignedVector<float> *work, const bool jac/* = true*/,
                  const Measurement *u1/* = NULL*/, const Measurement *u2/* = NULL*/,
                  const float eps);
void Propagate(const Point3D &pu, const Delta &D, const Camera &C1, Camera &C2,
               const float eps);

};

#endif
