/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Feb 3, 2015
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file implementation/EquidistantDistortion.hpp
 * @brief Header implementation file for the EquidistantDistortion class.
 * @author Stefan Leutenegger
 */


#include <Eigen/LU>
#include <iostream>

/// \brief okvis Main namespace of this package.
namespace vio {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {
    int FTR_UNDIST_DL_MAX_ITERATIONS              = 10;
    float FTR_UNDIST_DL_RADIUS_INITIAL              = 1.0f;      // 1.0^2
    float FTR_UNDIST_DL_RADIUS_MIN                  = 1.0e-10f;  // 0.00001^2
    float FTR_UNDIST_DL_RADIUS_MAX                  = 1.0e4f;    // 100.0^2
    float FTR_UNDIST_DL_RADIUS_FACTOR_INCREASE      = 9.0f;      // 3.0^2
    float FTR_UNDIST_DL_RADIUS_FACTOR_DECREASE      = 0.25f;     // 0.5^2
    float FTR_UNDIST_DL_GAIN_RATIO_MIN              = 0.25f;
    float FTR_UNDIST_DL_GAIN_RATIO_MAX              = 0.75f;
// The default constructor with all zero ki
EquidistantDistortion::EquidistantDistortion()
        : k1_(0.0),
          k2_(0.0),
          k3_(0.0),
          k4_(0.0)
{
    parameters_.setZero();
}

// Constructor initialising ki
EquidistantDistortion::EquidistantDistortion(float k1, float k2, float k3,
                                             float k4)
{
    parameters_[0] = k1;
    parameters_[1] = k2;
    parameters_[2] = k3;
    parameters_[3] = k4;
    k1_ = k1;
    k2_ = k2;
    k3_ = k3;
    k4_ = k4;
}

bool EquidistantDistortion::setParameters(const Eigen::VectorXd & parameters)
{
    if (parameters.cols() != NumDistortionIntrinsics) {
        return false;
    }
    parameters_ = parameters.cast<float>();;
    k1_ = parameters[0];
    k2_ = parameters[1];
    k3_ = parameters[2];
    k4_ = parameters[3];
    return true;
}

bool EquidistantDistortion::distort(const Eigen::Vector2d & pointUndistorted,
                                    Eigen::Vector2d * pointDistorted) const
{
    // distortion only:
    const float u0 = pointUndistorted[0];
    const float u1 = pointUndistorted[1];
    const float r = sqrt(u0 * u0 + u1 * u1);
    const float theta = atan(r);
    const float theta2 = theta * theta;
    const float theta4 = theta2 * theta2;
    const float theta6 = theta4 * theta2;
    const float theta8 = theta4 * theta4;
    const float thetad = theta
                          * (1 + k1_ * theta2 + k2_ * theta4 + k3_ * theta6 + k4_ * theta8);

    const float scaling = (r > 1e-8) ? thetad / r : 1.0;
    (*pointDistorted)[0] = scaling * u0;
    (*pointDistorted)[1] = scaling * u1;
    return true;
}

bool EquidistantDistortion::distort(const Eigen::Vector2f & pointUndistorted,
                                    Eigen::Vector2f * pointDistorted) const
{
    // distortion only:
    const float u0 = pointUndistorted[0];
    const float u1 = pointUndistorted[1];
    const float r = sqrt(u0 * u0 + u1 * u1);
    const float theta = atan(r);
    const float theta2 = theta * theta;
    const float theta4 = theta2 * theta2;
    const float theta6 = theta4 * theta2;
    const float theta8 = theta4 * theta4;
    const float thetad = theta
                         * (1 + k1_ * theta2 + k2_ * theta4 + k3_ * theta6 + k4_ * theta8);

    const float scaling = (r > 1e-8) ? thetad / r : 1.0;
    (*pointDistorted)[0] = scaling * u0;
    (*pointDistorted)[1] = scaling * u1;
    return true;
}

bool EquidistantDistortion::distort(const Eigen::Vector2d & pointUndistorted,
                                    Eigen::Vector2d * pointDistorted,
                                    Eigen::Matrix2d * pointJacobian,
                                    Eigen::Matrix2Xd * parameterJacobian) const
{
    // distortion first:
    const float u0 = pointUndistorted[0];
    const float u1 = pointUndistorted[1];
    const float r = sqrt(u0 * u0 + u1 * u1);
    const float theta = atan(r);
    const float theta2 = theta * theta;
    const float theta4 = theta2 * theta2;
    const float theta6 = theta4 * theta2;
    const float theta8 = theta4 * theta4;
    const float thetad = theta
                          * (1 + k1_ * theta2 + k2_ * theta4 + k3_ * theta6 + k4_ * theta8);

    const float scaling = (r > 1e-8) ? thetad / r : 1.0;
    (*pointDistorted)[0] = scaling * u0;
    (*pointDistorted)[1] = scaling * u1;

    Eigen::Matrix2d & J = *pointJacobian;
    if (r > 1e-8) {
        // mostly matlab generated...
        float t2;
        float t3;
        float t4;
        float t6;
        float t7;
        float t8;
        float t9;
        float t11;
        float t17;
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
        t17 = (((k1_ * t7 + k2_ * t9) + k3_ * t7 * t9) + k4_ * (t9 * t9)) + 1.0;
        t18 = 1.0 / t4;
        t19 = 1.0 / sqrt(t4 * t4 * t4);
        t20 = t6 * t8 * t17;
        t25 = ((k2_ * t6 * t7 * t8 * t11 * u1 * 4.0
                + k3_ * t6 * t8 * t9 * t11 * u1 * 6.0)
               + k4_ * t6 * t7 * t8 * t9 * t11 * u1 * 8.0)
              + k1_ * t6 * t8 * t11 * u1 * 2.0;
        t4 = ((k2_ * t6 * t7 * t8 * t11 * u0 * 4.0
               + k3_ * t6 * t8 * t9 * t11 * u0 * 6.0)
              + k4_ * t6 * t7 * t8 * t9 * t11 * u0 * 8.0)
             + k1_ * t6 * t8 * t11 * u0 * 2.0;
        t7 = t11 * t17 * t18 * u0 * u1;
        J(0, 1) = (t7 + t6 * t8 * t25 * u0) - t6 * t17 * t19 * u0 * u1;
        J(1, 1) = ((t20 - t3 * t6 * t17 * t19) + t3 * t11 * t17 * t18)
                  + t6 * t8 * t25 * u1;
        J(0, 0) = ((t20 - t2 * t6 * t17 * t19) + t2 * t11 * t17 * t18)
                  + t6 * t8 * t4 * u0;
        J(1, 0) = (t7 + t6 * t8 * t4 * u1) - t6 * t17 * t19 * u0 * u1;

        if (parameterJacobian) {
            Eigen::Matrix2Xd & Ji = *parameterJacobian;
            Ji.resize(2,NumDistortionIntrinsics);
            // mostly matlab generated...
            float t6;
            float t2;
            float t3;
            float t8;
            float t10;

            t6 = u0 * u0 + u1 * u1;
            t2 = atan(sqrt(t6));
            t3 = t2 * t2;
            t8 = t3 * t3;
            t6 = 1.0 / sqrt(t6);
            t10 = t8 * t8;
            Ji(0, 0) = t2 * t3 * t6 * u0;
            Ji(1, 0) = t2 * t3 * t6 * u1;
            Ji(0, 1) = t2 * t8 * t6 * u0;
            Ji(1, 1) = t2 * t8 * t6 * u1;
            Ji(0, 2) = t2 * t3 * t8 * t6 * u0;
            Ji(1, 2) = t2 * t3 * t8 * t6 * u1;
            Ji(0, 3) = t2 * t6 * t10 * u0;
            Ji(1, 3) = t2 * t6 * t10 * u1;

        }
    } else {
        // handle limit case for [u0,u1]->0
        if (parameterJacobian) {
            parameterJacobian->resize(2,NumDistortionIntrinsics);
            parameterJacobian->setZero();
        }
        J.setIdentity();
    }

    return true;
}


bool EquidistantDistortion::distort(const Eigen::Vector2f & pointUndistorted,
                                    Eigen::Vector2f* pointDistorted,
                                    Eigen::Matrix2f * pointJacobian,
                                    Eigen::Matrix2Xf * parameterJacobian) const
{


    ////针对k1,k2,k2,k3的等距投影模型, r^2 = x^2 + y^2
    ////残差2*1 归一化坐标的2维： min F(x,y) = 0.5*r(x,y)^2 约束： x^2+y^2 <=
    ///      r.x = x*(theta*(1 + k1*theta^2 + k2*theta^4 + k3 *theta^6 + k4*theta^8)/r) - x0
    ////     r.x = y*(theta*(1 + k1*theta^2 + k2*theta^4 + k3 *theta^6 + k4*theta^8)/r) - y0
    // distortion first:
    const float u0 = pointUndistorted[0];//x
    const float u1 = pointUndistorted[1];//y
    const float r = sqrt(u0 * u0 + u1 * u1);//r = sqrt(x^2 + y^2)
    const float theta = atan(r);
    const float theta2 = theta * theta;
    const float theta4 = theta2 * theta2;
    const float theta6 = theta4 * theta2;
    const float theta8 = theta4 * theta4;
    const float thetad = theta
                         * (1 + k1_ * theta2 + k2_ * theta4 + k3_ * theta6 + k4_ * theta8);

    const float scaling = (r > 1e-8) ? thetad / r : 1.0;
    (*pointDistorted)[0] = scaling * u0;
    (*pointDistorted)[1] = scaling * u1;

    Eigen::Matrix2f & J = *pointJacobian;
    if (r > 1e-8) {
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
        t17 = (((k1_ * t7 + k2_ * t9) + k3_ * t7 * t9) + k4_ * (t9 * t9)) + 1.0;
        t18 = 1.0 / t4;
        t19 = 1.0 / sqrt(t4 * t4 * t4);
        t20 = t6 * t8 * t17;
        t25 = ((k2_ * t6 * t7 * t8 * t11 * u1 * 4.0
                + k3_ * t6 * t8 * t9 * t11 * u1 * 6.0)
               + k4_ * t6 * t7 * t8 * t9 * t11 * u1 * 8.0)
              + k1_ * t6 * t8 * t11 * u1 * 2.0;
        t4 = ((k2_ * t6 * t7 * t8 * t11 * u0 * 4.0
               + k3_ * t6 * t8 * t9 * t11 * u0 * 6.0)
              + k4_ * t6 * t7 * t8 * t9 * t11 * u0 * 8.0)
             + k1_ * t6 * t8 * t11 * u0 * 2.0;
        t7 = t11 * t17 * t18 * u0 * u1;
        J(0, 1) = (t7 + t6 * t8 * t25 * u0) - t6 * t17 * t19 * u0 * u1;
        J(1, 1) = ((t20 - t3 * t6 * t17 * t19) + t3 * t11 * t17 * t18)
                  + t6 * t8 * t25 * u1;
        J(0, 0) = ((t20 - t2 * t6 * t17 * t19) + t2 * t11 * t17 * t18)
                  + t6 * t8 * t4 * u0;
        J(1, 0) = (t7 + t6 * t8 * t4 * u1) - t6 * t17 * t19 * u0 * u1;

        if (parameterJacobian) {
            Eigen::Matrix2Xf & Ji = *parameterJacobian;
            Ji.resize(2,NumDistortionIntrinsics);
            // mostly matlab generated...
            float t6;
            float t2;
            float t3;
            float t8;
            float t10;

            t6 = u0 * u0 + u1 * u1;
            t2 = atan(sqrt(t6));
            t3 = t2 * t2;
            t8 = t3 * t3;
            t6 = 1.0 / sqrt(t6);
            t10 = t8 * t8;
            Ji(0, 0) = t2 * t3 * t6 * u0;
            Ji(1, 0) = t2 * t3 * t6 * u1;
            Ji(0, 1) = t2 * t8 * t6 * u0;
            Ji(1, 1) = t2 * t8 * t6 * u1;
            Ji(0, 2) = t2 * t3 * t8 * t6 * u0;
            Ji(1, 2) = t2 * t3 * t8 * t6 * u1;
            Ji(0, 3) = t2 * t6 * t10 * u0;
            Ji(1, 3) = t2 * t6 * t10 * u1;

        }
    } else {
        // handle limit case for [u0,u1]->0
        if (parameterJacobian) {
            parameterJacobian->resize(2,NumDistortionIntrinsics);
            parameterJacobian->setZero();
        }
        J.setIdentity();
    }

    return true;
}


bool EquidistantDistortion::distortWithExternalParameters(
        const Eigen::Vector2d & pointUndistorted,
        const Eigen::VectorXd & parameters, Eigen::Vector2d * pointDistorted,
        Eigen::Matrix2d * pointJacobian, Eigen::Matrix2Xd * parameterJacobian) const
{
    // decompose parameters

    const float k1 = parameters[0];
    const float k2 = parameters[1];
    const float k3 = parameters[2];
    const float k4 = parameters[3];
    // distortion first:
    const float u0 = pointUndistorted[0];
    const float u1 = pointUndistorted[1];
    const float r = sqrt(u0 * u0 + u1 * u1);
    const float theta = atan(r);
    const float theta2 = theta * theta;
    const float theta4 = theta2 * theta2;
    const float theta6 = theta4 * theta2;
    const float theta8 = theta4 * theta4;
    const float thetad = theta
                          * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

    const float scaling = (r > 1e-8) ? thetad / r : 1.0;
    (*pointDistorted)[0] = scaling * u0;
    (*pointDistorted)[1] = scaling * u1;

    Eigen::Matrix2d & J = *pointJacobian;
    if (r > 1e-8) {
        // mostly matlab generated...
        float t2;
        float t3;
        float t4;
        float t6;
        float t7;
        float t8;
        float t9;
        float t11;
        float t17;
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
        t17 = (((k1 * t7 + k2 * t9) + k3 * t7 * t9) + k4 * (t9 * t9)) + 1.0;
        t18 = 1.0 / t4;
        t19 = 1.0 / sqrt(t4 * t4 * t4);
        t20 = t6 * t8 * t17;
        t25 = ((k2 * t6 * t7 * t8 * t11 * u1 * 4.0
                + k3 * t6 * t8 * t9 * t11 * u1 * 6.0)
               + k4 * t6 * t7 * t8 * t9 * t11 * u1 * 8.0)
              + k1 * t6 * t8 * t11 * u1 * 2.0;
        t4 = ((k2 * t6 * t7 * t8 * t11 * u0 * 4.0
               + k3 * t6 * t8 * t9 * t11 * u0 * 6.0)
              + k4 * t6 * t7 * t8 * t9 * t11 * u0 * 8.0)
             + k1 * t6 * t8 * t11 * u0 * 2.0;
        t7 = t11 * t17 * t18 * u0 * u1;
        J(0, 0) = (t7 + t6 * t8 * t25 * u0) - t6 * t17 * t19 * u0 * u1;
        J(1, 0) = ((t20 - t3 * t6 * t17 * t19) + t3 * t11 * t17 * t18)
                  + t6 * t8 * t25 * u1;
        J(0, 1) = ((t20 - t2 * t6 * t17 * t19) + t2 * t11 * t17 * t18)
                  + t6 * t8 * t4 * u0;
        J(1, 1) = (t7 + t6 * t8 * t4 * u1) - t6 * t17 * t19 * u0 * u1;
        if (parameterJacobian) {
            Eigen::Matrix2Xd & Ji = *parameterJacobian;
            Ji.resize(2,NumDistortionIntrinsics);
            // mostly matlab generated...
            float t6;
            float t2;
            float t3;
            float t8;
            float t10;

            t6 = u0 * u0 + u1 * u1;
            t2 = atan(sqrt(t6));
            t3 = t2 * t2;
            t8 = t3 * t3;
            t6 = 1.0 / sqrt(t6);
            t10 = t8 * t8;
            Ji(0, 0) = t2 * t3 * t6 * u0;
            Ji(1, 0) = t2 * t3 * t6 * u1;
            Ji(0, 1) = t2 * t8 * t6 * u0;
            Ji(1, 1) = t2 * t8 * t6 * u1;
            Ji(0, 2) = t2 * t3 * t8 * t6 * u0;
            Ji(1, 2) = t2 * t3 * t8 * t6 * u1;
            Ji(0, 3) = t2 * t6 * t10 * u0;
            Ji(1, 3) = t2 * t6 * t10 * u1;

        }
    } else {
        // handle limit case for [u0,u1]->0
        if (parameterJacobian) {
            parameterJacobian->resize(2,NumDistortionIntrinsics);
            parameterJacobian->setZero();
        }
        J.setIdentity();
    }

    return true;
}
bool EquidistantDistortion::undistort(const Eigen::Vector2d & pointDistorted,
                                      Eigen::Vector2d * pointUndistorted) const
{
    // this is expensive: we solve with Gauss-Newton...
    Eigen::Vector2d x_bar = pointDistorted;  // initialise at distorted point
    const int n = 5;  // just 5 iterations max.
    Eigen::Matrix2d E;  // error Jacobian

    bool success = false;
    for (int i = 0; i < n; i++) {

        Eigen::Vector2d x_tmp;

        distort(x_bar, &x_tmp, &E);

        Eigen::Vector2d e(pointDistorted - x_tmp);//Hx = b=>
        Eigen::Vector2d du = (E.transpose() * E).inverse() * E.transpose() * e;

        x_bar += du;


        const double chi2 = e.dot(e);
        if (chi2 < 1e-2) {
            success = true;
        }
        //std::cout<<"chi2"<<chi2<<std::endl;
        if (chi2 < 1e-15) {
            success = true;
            break;
        }

    }
    *pointUndistorted = x_bar;

    return success;
}


bool EquidistantDistortion::undistort(const Eigen::Vector2f & pointDistorted,
                                      Eigen::Vector2f * pointUndistorted) const
{

    float dx2GN/*G-N解出的步长的欧式距离*/, dx2GD/*pU点的欧式距离*/, delta2/*信赖域半径*/, beta;
    Eigen::Vector2f dxGN/*G-N解出的增量*/, dxGD/*pU点*/;
    bool update, converge;
    float dr, j, dx2;

    delta2 = 1.0;
    // this is expensive: we solve with Gauss-Newton...
    Eigen::Vector2f x_bar = pointDistorted;  // initialise at distorted point
    const int n = 30;  // just 5 iterations max.
    const int UNDIST_DL_MAX_ITERATIONS = 4;
    Eigen::Matrix2f E;  // error Jacobian

    bool success = false;
    for (int i = 0; i < n; i++)
    {

        Eigen::Vector2f x_tmp;

        distort(x_bar, &x_tmp, &E);

        Eigen::Vector2f e(x_tmp -pointDistorted);
        Eigen::Vector2f dx = -(E.transpose() * E).inverse() * E.transpose() * e;
        Eigen::Vector2f b = -E.transpose() * e;
        Eigen::Matrix2f A = -E.transpose() * E;

        x_bar += dx;

//
//        dx2 = dx.squaredNorm();//x的欧式距离
//        //by wya
//       dxGN = dx;//G-N法求出的增量
//        dx2GN = dx2;//G-N法求出的增量的欧式距离
//        dx2GD = 0.0f;//dogleg
//        const float F = e.squaredNorm();//dogleg迭代开始前残差的欧式距离
//        const Eigen::Vector2f xnBkp = x_bar;//待优化变量
//        update = true;
//        converge = false;//狗腿迭代次数
//        for (int iIterDL = 0; iIterDL < UNDIST_DL_MAX_ITERATIONS; ++iIterDL)
//        {
//            if (dx2GN > delta2 && dx2GD == 0.0f) {//如果G-N增量在信赖域外且dx2GD还没有初始化
//                const float bl = sqrtf(b.squaredNorm());//模长
//                const Eigen::Vector2f g = b * (1.0f / bl);//梯度方向
//                const Eigen::Vector2f Ag = A * g;
//                const float xl = bl / g.dot(Ag);//计算pU点的步长
//                dxGD = g * -xl;//负梯度*步长,pU点
//                dx2GD = xl * xl;//pU点的半径
//
//            }
//            //三种情况,1：GN极值点在域内直接变成无约束条件 2:GN和pU点都在域外,那么就在给pU点的步长一个比例因子(域半径/自己的步长^2（因为用的是最小2乘）),让它刚好落在域半径上
//            //3:GN在域外,pU点在域内,那么增量就是GN极值点和pU点的连线与信赖域的交点
//            if (dx2GN <= delta2) {//如果G-N的极值在信赖域内,那么就是一个无约束问题,就直接用GN法求出的增量就可以
//                dx = dxGN;
//                dx2 = dx2GN;
//                beta = 1.0f;
//            } else if (dx2GD >= delta2) {//如果G-N和pU点求的最优点都在信赖域外
//                if (delta2 == 0.0f) {//信赖域为0,直接用pU点求得最优值
//                    dx = dxGD;
//                    dx2 = dx2GD;
//                } else {
//                    dx =  dxGD * sqrtf(delta2 / dx2GD);//乘比例因子
//                    dx2 = delta2;
//                }
//                beta = 0.0f;
//            } else {//GN在域外,pU点在域内,那么增量就是GN极值点和pU点的连线与信赖域的交点
//                const Eigen::Vector2f v = dxGN - dxGD;//方向
//                const float d = dxGD.dot(v), v2 = v.squaredNorm();
//                //beta = float((-d + sqrt(double(d) * d + (delta2 - dx2GD) * double(v2))) / v2);
//                beta = (-d + sqrtf(d * d + (delta2 - dx2GD) * v2)) / v2;//算得是在域外那段连线的长度
//                dx = dxGD;
//                dx += v * beta;
//                dx2 = delta2;
//            }
//            x_bar += dx;//加上这一次的增量
//            Eigen::Vector2f x_d_temp;
//            distort(x_bar, &x_d_temp);
//            const float dFa = F - (x_d_temp - pointDistorted).squaredNorm();//实际下降的
//            const float dFp = F - (e + ( E.transpose()) * dx).squaredNorm();//理论下降值,直接用J*dx近似下降值了
//            const float rho = dFa > 0.0f && dFp > 0.0f ? dFa / dFp : -1.0f;//求实际/理论的比值,理论不可能为负,实际为负的时候拒绝这次更新
//            //信赖域： Numerical Optimization 第二版 p69
//
//            //rho < 0.25 如果大于0说明近似的不好,需要减小信赖域,减小近似的范围。如果<0就说明是错误的近似,那么就拒绝这次的增量
//            if (rho < FTR_UNDIST_DL_GAIN_RATIO_MIN) {
//                delta2 *= FTR_UNDIST_DL_RADIUS_FACTOR_DECREASE;
//                if (delta2 < FTR_UNDIST_DL_RADIUS_MIN) {
//                    delta2 = FTR_UNDIST_DL_RADIUS_MIN;
//                }
//                x_bar = xnBkp;//取消这次增量
//                update = false;//不更新
//                converge = false;
//                continue;
//            } else if (rho > FTR_UNDIST_DL_GAIN_RATIO_MAX) //rho > 0.75,可以扩大信赖域半径
//            {
//                delta2 = std::max(delta2, FTR_UNDIST_DL_RADIUS_FACTOR_INCREASE * dx2);
//                if (delta2 > FTR_UNDIST_DL_RADIUS_MAX) {//信赖域半径最大值
//                    delta2 = FTR_UNDIST_DL_RADIUS_MAX;
//                }
//            }
//            update = true;//
//
//            converge = dx2 < 1e-12;//增量小于阈值,认为收敛
//            break;
//        }
//        if (!update || converge) {
//            break;
//        }
//
//        const double chi2 = e.dot(e);
//        if (chi2 < 1e-5) {
//            success = true;
//        }
        const double chi2 = e.dot(e);

        if (chi2 < 1e-8) {
            success = true;
            break;
        }


    }
    *pointUndistorted = x_bar;
    if(converge)
        success = true;


    return success;
}

bool EquidistantDistortion::undistort(const Eigen::Vector2d & pointDistorted,
                                      Eigen::Vector2d * pointUndistorted,
                                      Eigen::Matrix2d * pointJacobian) const
{
    // this is expensive: we solve with Gauss-Newton...
    Eigen::Vector2d x_bar = pointDistorted;  // initialise at distorted point
    const int n = 5;  // just 5 iterations max.
    Eigen::Matrix2d E;  // error Jacobian

    bool success = false;
    for (int i = 0; i < n; i++) {

        Eigen::Vector2d x_tmp;

        distort(x_bar, &x_tmp, &E);

        Eigen::Vector2d e(pointDistorted - x_tmp);
        Eigen::Vector2d dx = (E.transpose() * E).inverse() * E.transpose() * e;

        x_bar += dx;

        const double chi2 = e.dot(e);
        if (chi2 < 1e-2) {
            success = true;
        }
        if (chi2 < 1e-15) {
            success = true;
            break;
        }

    }
    *pointUndistorted = x_bar;

    // the Jacobian of the inverse map is simply the inverse Jacobian.
    *pointJacobian = E.inverse();

    return success;
}

bool EquidistantDistortion::undistort(const Eigen::Vector2f & pointDistorted,
                                      Eigen::Vector2f * pointUndistorted,
                                      Eigen::Matrix2f * pointJacobian) const
{
    // this is expensive: we solve with Gauss-Newton...
    Eigen::Vector2f x_bar = pointDistorted;  // initialise at distorted point
    const int n = 5;  // just 5 iterations max.
    Eigen::Matrix2f E;  // error Jacobian

    bool success = false;
    for (int i = 0; i < n; i++) {

        Eigen::Vector2f x_tmp;

        distort(x_bar, &x_tmp, &E);

        Eigen::Vector2f e(pointDistorted - x_tmp);
        Eigen::Vector2f dx = (E.transpose() * E).inverse() * E.transpose() * e;

        x_bar += dx;

        const double chi2 = e.dot(e);
        if (chi2 < 1e-2) {
            success = true;
        }
        if (chi2 < 1e-15) {
            success = true;
            break;
        }

    }
    *pointUndistorted = x_bar;

    // the Jacobian of the inverse map is simply the inverse Jacobian.
    *pointJacobian = E.inverse();

    return success;
}

}  // namespace cameras
}  // namespace okvis
