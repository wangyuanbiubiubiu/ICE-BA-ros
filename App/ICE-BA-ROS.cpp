/******************************************************************************
 * Copyright 2017 Baidu Robotic Vision Authors. All Rights Reserved.
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
#include "IBA/IBA.h"
#include "feature_utils.h"
#include "image_utils.h"
#include "xp_quaternion.h"
#include "param.h"  // calib
#include "basic_datatype.h"
#include "iba_helper.h"
#include "pose_viewer.h"
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <opencv2/core.hpp>

//ros相关
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <tf/transform_broadcaster.h>
#include "../ros_visualization/visualization.h"
#include <eigen3/Eigen/Dense>

#include <thread>
#include <mutex>
#include <condition_variable>

#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <queue>
#include <memory>

//闭环
#include "LoopClosing.h"
#include "CameraBase.hpp"
#include "NCameraSystem.hpp"
#include "ParamsInit.h"

DEFINE_bool(show_track_img,true,"output trackimg(ros msg)");
DEFINE_string(config_file, "", "像vinsfuison,给总配置文件的路径");
DEFINE_int32(grid_row_num, 1, "Number of rows of detection grids");
DEFINE_int32(grid_col_num, 1, "Number of cols of detection grids");
DEFINE_int32(max_num_per_grid, 150, "Max number of points per grid");
DEFINE_double(feat_quality, 0.07, "Tomasi-Shi feature quality level");
DEFINE_double(feat_min_dis, 10, "Tomasi-Shi feature minimal distance");
DEFINE_bool(not_use_fast, false, "Whether or not use FAST");
DEFINE_int32(pyra_level, 2, "Total pyramid levels");
DEFINE_int32(start_idx, 0, "The image index of the first detection (from 0)");
DEFINE_int32(end_idx, -1, "The image index of the last detection");
DEFINE_double(uniform_radius, 40, "< 5 disables uniformaty enforcement");
DEFINE_int32(ft_len, 125, "The feature track length threshold when dropout kicks in");
DEFINE_double(ft_droprate, 0.05, "The drop out rate when acc feature track exceeds ft_len");
DEFINE_bool(show_feat_only, false, "wether or not show detection results only");
DEFINE_int32(fast_thresh, 10, "FAST feature threshold (only meaningful if use_fast=true)");
DEFINE_double(min_feature_distance_over_baseline_ratio,
              4, "Used for slave image feature detection");
DEFINE_double(max_feature_distance_over_baseline_ratio,
              3000, "Used for slave image feature detection");
DEFINE_string(iba_param_path, "", "iba parameters path");
DEFINE_string(gba_result_path, "", "Save the gab_result");
DEFINE_string(result_folder, "", "Save the result");
DEFINE_bool(stereo, false, "monocular or stereo mode");
DEFINE_bool(save_feature, false, "Save features to .dat file");
DEFINE_string(image0_topic, "", "左目话题");
DEFINE_string(image1_topic, "", "右目话题");
DEFINE_string(imu_topic, "", "imu话题");
DEFINE_bool(LoopClosure, true, "use LoopClosure?");
DEFINE_bool(GetGT, true, "GTdata");
DEFINE_bool(UseIMU,true, "use imu data?");
DEFINE_string(GT_path, "", "");
DEFINE_string(dbow3_voc_path, "", "dbow3_voc_path,must set!");
DEFINE_int32(mono_init_th,3,"大于3次认为ok");


std::queue<IBA::RelativeConstraint> loop_info;

std::queue<IBA::CurrentFrame> CFs_buf;
std::queue<IBA::KeyFrame> KFs_buf;
std::mutex m_buf,m_syn_buf,m_solver_buf,m_loop_buf;
std::condition_variable con;
std::unique_ptr<XP::FeatureTrackDetector> feat_track_detector_ptr;
std::unique_ptr<XP::ImgFeaturePropagator> slave_img_feat_propagator_ptr;
std::vector<float> total_time;

std::unique_ptr<LC::LoopClosing> LoopCloser_ptr;

XP::DuoCalibParam duo_calib_param;
//绘图器
XP::PoseViewer pose_viewer;
//求解器初始化
IBA::Solver solver;
//外参
Eigen::Matrix4f T_Cl_Cr;
Eigen::Matrix4d T_Cl_Cr_d;
Eigen::Matrix4d Tbc0;
//用于LBA回调的绘图器
Eigen::Vector3f last_position = Eigen::Vector3f::Zero();
float travel_dist = 0.f;
double offset_ts = 0;
bool set_relative_pose = false;
bool init_first_track = true;

Eigen::Matrix4d relative_pose; //Twc(first) * Twc.inv(gt)
// Create masks based on FOVs computed from intrinsics
std::vector<cv::Mat_<uchar> > masks(2);

float prev_img_time_stamp = 0.0f;
// load previous image//左目之前的特征点和描述子
std::vector<cv::KeyPoint> pre_image_key_points;
cv::Mat pre_image_features;
int mono_init_count = 0;
bool need_mono_init = true;

std::map<int,Eigen::Vector3f> cur_Mp_info; //前一帧追踪到的地图点的3d坐标
std::vector<std::tuple<int,int,cv::KeyPoint>> cur_track;//地图点id,左右目,观测 ;
Eigen::Matrix4f last_pose_Twc0;

bool process_frontend()
{
    while (true)
    {
        std::vector<std::pair<std::vector<XP::ImuData>, stereo_Img_info_f >> measurements;

        std::unique_lock <std::mutex> lk(m_buf);
        con.wait(lk, [&] {
            return (measurements = getMeasurements()).size() != 0;//将目前所有的测量取出来
        });

        lk.unlock();

        //处理多组观测
        for (auto &measurement : measurements)
        {
            stereo_Img_info_f cur_img_info = measurement.second;
            // get timestamp from image file name (s)
            //这里用的时间戳都是相对于imu第一帧的时间戳,time_stamp为左图像时间戳
            const float img_time_stamp = cur_img_info.first;
            std::vector<XP::ImuData> imu_meas = measurement.first;//两帧图像之间imu的原始测量数据
            cv::Mat img_in_raw;
            if(cur_img_info.second.first.type() == CV_8UC1 )
                img_in_raw = cur_img_info.second.first;
            else if(cur_img_info.second.first.type() == CV_8UC3)
            {
                cv::cvtColor(cur_img_info.second.first, img_in_raw, CV_BGR2GRAY);
            }
            CHECK_EQ(img_in_raw.rows, duo_calib_param.Camera.img_size.height);
            CHECK_EQ(img_in_raw.cols, duo_calib_param.Camera.img_size.width);
            cv::Mat img_in_smooth;
            cv::blur(img_in_raw, img_in_smooth, cv::Size(3, 3));//图像去噪
            if (img_in_smooth.rows == 0) {
                std::cerr << "Cannot smooth " <<std::endl;
                return EXIT_FAILURE;
            }
            // load slave image
            cv::Mat slave_img_smooth;  // for visualization later

            if (FLAGS_stereo) //也就是双目的情况
            {//读取右目相机,一样做降噪处理
                cv::Mat slave_img_in;
                if(cur_img_info.second.second.type() == CV_8UC1 )
                    slave_img_in = cur_img_info.second.second;
                else if(cur_img_info.second.second.type() == CV_8UC3)
                {
                    cv::cvtColor(cur_img_info.second.second, slave_img_in, CV_BGR2GRAY);
                }
                cv::blur(slave_img_in, slave_img_smooth, cv::Size(3, 3));
            }

            std::vector<cv::KeyPoint> key_pnts;//左目提取到的特征点
            cv::Mat orb_feat;//左目orb描述子
            cv::Mat pre_img_in_smooth;
            // load slave image
            std::vector<cv::KeyPoint> key_pnts_slave;
            cv::Mat orb_feat_slave;
            if(init_first_track) //首次追踪
            {
                // first frame
                //第一帧左相机提取特征点
                //输入左相机图片,图像掩码,最大提取的特征点数量,金字塔层数,fast点阈值,特征点,描述子
                //主要就是提点,计算描述子
                feat_track_detector_ptr->detect(img_in_smooth,
                                                masks[0],
                                                FLAGS_max_num_per_grid * FLAGS_grid_row_num * FLAGS_grid_col_num,
                                                FLAGS_pyra_level,
                                                FLAGS_fast_thresh,
                                                &key_pnts,
                                                nullptr);
                feat_track_detector_ptr->build_img_pyramids(img_in_smooth,//构建图像金字塔,存到前一帧缓存器中
                                                            XP::FeatureTrackDetector::BUILD_TO_PREV);
            } else
            {

                VLOG(1) << "pre_image_key_points.size(): " << pre_image_key_points.size();
                const int request_feat_num = FLAGS_max_num_per_grid * FLAGS_grid_row_num * FLAGS_grid_col_num;//最大提取的特征点数量
                feat_track_detector_ptr->build_img_pyramids(img_in_smooth,
                                                            XP::FeatureTrackDetector::BUILD_TO_CURR);//存储当前左目的金字塔
                if (imu_meas.size() > 1) {//如果有imu的话
                    // Here we simply the transformation chain to rotation only and assume zero translation
                    cv::Matx33f old_R_new;
                    XP::XpQuaternion I_new_q_I_old;  // The rotation between the new {I} and old {I}
                    //RK4预积分,算出Ri(cur)_i(pre)
                    for (size_t i = 1; i < imu_meas.size(); ++i) {
                        XP::XpQuaternion q_end;
                        XP::IntegrateQuaternion(imu_meas[i - 1].ang_v/*k时刻的角速度*/,
                                                imu_meas[i].ang_v/*k+1时刻的角速度*/,
                                                I_new_q_I_old,
                                                imu_meas[i].time_stamp - imu_meas[i - 1].time_stamp/*delta_t*/,
                                                &q_end);
                        I_new_q_I_old = q_end;
                    }
                    Eigen::Matrix3f I_new_R_I_old = I_new_q_I_old.ToRotationMatrix();
                    Eigen::Matrix4f I_T_C =//左相机到imu的变换
                            duo_calib_param.Imu.D_T_I.inverse() * duo_calib_param.Camera.D_T_C_lr[0];//Ti_d * Td_cl = Ti_cl
                    Eigen::Matrix3f I_R_C = I_T_C.topLeftCorner<3, 3>();//外参旋转部分
                    //imu测出是imu系前后两帧之间的旋转，需要转到左相机坐标系的旋转
                    Eigen::Matrix3f C_new_R_C_old = I_R_C.transpose() * I_new_R_I_old * I_R_C;// Rcl_i * Ri(cur)_i(pre) * Ri_cl = Rcl(cur)_cl(pre)
                    //Rcl(pre)_cl(cur)
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            old_R_new(j, i) = C_new_R_C_old(i, j);
                        }
                    }

                    if (VLOG_IS_ON(1)) {
                        XP::XpQuaternion C_new_q_C_old;
                        C_new_q_C_old.SetFromRotationMatrix(C_new_R_C_old);
                        VLOG(1) << "C_new_R_C_old = \n" << C_new_R_C_old;
                        VLOG(1) << "ea =\n" << C_new_q_C_old.ToEulerRadians() * 180 / M_PI;
                    }
                    feat_track_detector_ptr->optical_flow_and_detect(masks[0]/*左相机掩码*/,
                                                                     pre_image_features/*左相机上一帧检测到的特征点的描述子*/,
                                                                     pre_image_key_points/*左相机上一帧检测到的特征点*/,
                                                                     request_feat_num/*最大要求提取的特征点数量*/,
                                                                     FLAGS_pyra_level/*金字塔层*/,
                                                                     FLAGS_fast_thresh/*fast阈值*/,
                                                                     &key_pnts/*左相机当前帧提取到的特征点*/,
                                                                     nullptr/*左相机当前帧提取到的特征点对应的描述子*/,
                                                                     duo_calib_param.Camera.fishEye,
                                                                     cv::Vec2f(0, 0),  // shift init pixels
                                                                     &duo_calib_param.Camera.cv_camK_lr[0]/*cv形式的左右相机内参*/,
                                                                     &duo_calib_param.Camera.cv_dist_coeff_lr[0]/*左右相机的畸变参数*/,
                                                                     &old_R_new/*Rcl(pre)_cl(cur)左相机当前帧到前一帧的旋转*/);
                } else {
                    feat_track_detector_ptr->optical_flow_and_detect(masks[0],
                                                                     pre_image_features,
                                                                     pre_image_key_points,
                                                                     request_feat_num,
                                                                     FLAGS_pyra_level,
                                                                     FLAGS_fast_thresh,
                                                                     &key_pnts,
                                                                     nullptr);
                }
                feat_track_detector_ptr->update_img_pyramids();//更新金字塔buffer
                VLOG(1) << "after OF key_pnts.size(): " << key_pnts.size() << " requested # "
                        << FLAGS_max_num_per_grid * FLAGS_grid_row_num * FLAGS_grid_col_num;
            }

            //如果是双目的话
            if (slave_img_smooth.rows > 0)
            {
                CHECK(orb_feat_slave.empty());
                auto det_slave_img_start = std::chrono::high_resolution_clock::now();
                //输入右左目图片，左目提取的特征点,左右目外参,右目提取的特征点,描述子,是否要输出debug信息
                //用svo的块匹配的思路算出右相机中特征点的坐标,并且计算描述子
                slave_img_feat_propagator_ptr->PropagateFeatures(slave_img_smooth,  // cur 右目
                                                                 img_in_smooth,  // ref 左目
                                                                 key_pnts, //左目提取的特征点
                                                                 T_Cl_Cr,  // T_ref_cur //左右目外参
                                                                 &key_pnts_slave,
                                                                 nullptr,
                                                                 false);  // draw_debug
                VLOG(1) << "detect slave key_pnts.size(): " << key_pnts_slave.size() << " takes "
                        << std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::high_resolution_clock::now() - det_slave_img_start).count() / 1e3
                        << " ms";
            }
            //根据class_id(与地图点id一致)进行排序,不过本来就是这个顺序吧
            std::sort(key_pnts.begin(), key_pnts.end(), cmp_by_class_id);
            std::sort(key_pnts_slave.begin(), key_pnts_slave.end(), cmp_by_class_id);

            if(init_first_track)
                init_first_track = false;
            else
            {
                if(FLAGS_show_track_img)
                    pubTrackImage(img_in_smooth,slave_img_smooth,pre_image_key_points,key_pnts,key_pnts_slave,(double)img_time_stamp + offset_ts);
            }
            if(!FLAGS_stereo)//单目简单的判断一下第一次需要开始吧
            {//算一下运动
                mono_begin_compute(pre_image_key_points,key_pnts,duo_calib_param.Camera.cv_camK_lr[0],
                                   &duo_calib_param.Camera.cv_dist_coeff_lr[0],
                                   duo_calib_param.Camera.fishEye,
                                   masks[0],&mono_init_count);
                if(need_mono_init && mono_init_count >= FLAGS_mono_init_th )//大于5次就认为在激励,开始追踪吧
                {
                    need_mono_init = false;
                }
                else if(need_mono_init)
                {
                    pre_image_key_points = key_pnts;
                    pre_image_features = orb_feat.clone();

                    prev_img_time_stamp = img_time_stamp;
                    continue;
                }
            }

            // push to IBA
            IBA::CurrentFrame CF;
            IBA::KeyFrame KF;
            //输入左相机特征点,右相机特征点,imu测量,左目时间戳,当前帧,关键帧
            //进行点管理(新旧地图点的观测更新),以及关键帧判断以及生成
            create_iba_frame(key_pnts, key_pnts_slave, imu_meas, img_time_stamp, &CF, &KF);

            if(FLAGS_LoopClosure)
            {
                if(KF.iFrm != -1)
                {

                    std::vector<cv::KeyPoint> loop_key_pnts;//左目提取到的特征点
                    cv::Mat loop_orb_feat;//左目orb描述子

                    feat_track_detector_ptr->detect_for_loop(img_in_smooth,
                                                             masks[0],
                                                             1000,
                                                             FLAGS_pyra_level,
                                                             FLAGS_fast_thresh,
                                                             &loop_key_pnts,
                                                             &loop_orb_feat);

                    cv::Mat cur_orb_feat;
                    feat_track_detector_ptr->ComputeDescriptors(img_in_smooth,&key_pnts,&cur_orb_feat);
                    LoopCloser_ptr ->InsertKeyFrame(std::shared_ptr<LC::KeyFrame> (new LC::KeyFrame(KF.iFrm,key_pnts,cur_orb_feat,loop_key_pnts,loop_orb_feat,img_in_smooth)));
                }
            }

            m_solver_buf.lock();
            CFs_buf.push(CF);
            if(KF.iFrm != -1)
                KFs_buf.push(KF);
            m_solver_buf.unlock();


            pre_image_key_points = key_pnts;
            pre_image_features = orb_feat.clone();

            prev_img_time_stamp = img_time_stamp;
        }

    }
}

void process_backend()
{
    while(1)
    {
        IBA::CurrentFrame CF;
        IBA::KeyFrame KF;
        CF.iFrm = -1;
        KF.iFrm = -1;
        m_solver_buf.lock();
        if (!CFs_buf.empty())
        {
            CF = CFs_buf.front();
            CFs_buf.pop();
            if(!KFs_buf.empty())
            {
                if( CF.iFrm == KFs_buf.front().iFrm)//KF里的索引只可能>= cf的
                {
                    KF = KFs_buf.front();
                    KFs_buf.pop();
                }
            }
        }
        m_solver_buf.unlock();
        if(CF.iFrm != -1)
        {
            if(FLAGS_LoopClosure)
            {
                std::vector<IBA::RelativeConstraint> loop_Relative_priors;
                m_loop_buf.lock();
                while(!loop_info.empty())
                {
                    loop_Relative_priors.push_back(loop_info.front());
                    loop_info.pop();
                }
                m_loop_buf.unlock();
                //将当前帧以及关键帧(如果有的话)放进求解器
                if(!loop_Relative_priors.empty())
                {
                    for (int i = 0; i < loop_Relative_priors.size(); ++i)
                    {
                        solver.PushRelativeConstraint(loop_Relative_priors[i]);
                    }
                    solver.Wakeup_GBA();
                }
            }

            if(total_time.size() > CF.iFrm)//先这么存一下吧,先不考虑上限的问题
                total_time[CF.iFrm] = CF.t;
            else
            {
                total_time.resize(total_time.size()+300);
                total_time[CF.iFrm] = CF.t;
            }

            if(!FLAGS_UseIMU)
            {
                Get_Track(CF,cur_track);

                if(SolveMulticamPnP(cur_track,cur_Mp_info,last_pose_Twc0))
                {
                    Eigen::Matrix3f pose_pnp_Rc0w = last_pose_Twc0.block<3,3>(0,0).transpose();
                    Eigen::Vector3f pose_pnp_twc0 = last_pose_Twc0.block<3,1>(0,3);
                    for (int i = 0; i < 3; ++i)
                    {
                        CF.Cam_state.Cam_pose.p[i] = pose_pnp_twc0[i];
                        for (int j = 0; j < 3; ++j) {
                            CF.Cam_state.Cam_pose.R[i][j] = pose_pnp_Rc0w(i,j);
                        }

                    }
                    KF.Cam_pose = CF.Cam_state.Cam_pose;
                }
            }
            solver.PushCurrentFrame(CF, KF.iFrm == -1 ? nullptr : &KF);//先说明一下,我习惯的求解增量的表达是Hx=b,但是这里是Hx=-b
//          show pose
            pose_viewer.displayTo("trajectory");
            cv::waitKey(1);
        }

    }
}


//euroc测试代码
int main(int argc, char** argv)
{
    //初始化glog相关程序
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InstallFailureSignalHandler();

    ros::init(argc, argv, "ice_ba");
    ros::NodeHandle n("~");

    registerPub(n);

    total_time.resize(4000);

    //如果没有给image文件夹的话
    if (FLAGS_config_file.empty() )//
    {
        google::ShowUsageWithFlags(argv[0]);
        return -1;
    }

    if(!FLAGS_result_folder.empty())
    {
        if (!fs::is_directory(FLAGS_result_folder + "/dat")) {
            fs::create_directories(FLAGS_result_folder + "/dat");
        }

    }

    try {//输出标定参数（内外参,立体矫正,去完畸变后的映射）
        load_Parameters(FLAGS_config_file, duo_calib_param);
    } catch (...){
        LOG(ERROR) << "Load calibration file error";
        return -1;
    }


    int Num_Cam = FLAGS_stereo ? 2:1;
    for (int lr = 0; lr < Num_Cam; ++lr) {
        float fov;
        //输出每一目的mask和视场角
        if (XP::generate_cam_mask(duo_calib_param.Camera.cv_camK_lr[lr],
                                  duo_calib_param.Camera.cv_dist_coeff_lr[lr],
                                  duo_calib_param.Camera.fishEye,
                                  duo_calib_param.Camera.img_size,
                                  &masks[lr],
                                  &fov)) {
            std::cout << "camera " << lr << " fov: " << fov << " deg\n";
        }
    }

    if(FLAGS_LoopClosure)
    {
        LoopCloser_ptr.reset(new LC::LoopClosing(FLAGS_dbow3_voc_path,
                                                 duo_calib_param.Camera.cameraK_lr[0],  // ref_camK 左相机内参
                                                 duo_calib_param.Camera.cv_dist_coeff_lr[0],  // ref_dist_coeff 左相机畸变
                                                 duo_calib_param.Camera.fishEye,
                                                 masks[0]));
    }

    //初始化特征提取器
    feat_track_detector_ptr.reset(new XP::FeatureTrackDetector(FLAGS_ft_len/*图像大小*/,
                                                               FLAGS_ft_droprate/*图像大小*/,
                                                               !FLAGS_not_use_fast/*是否用fast点*/,
                                                               FLAGS_uniform_radius/*图像大小*/,
                                                               duo_calib_param.Camera.img_size/*图像大小*/));

    if(FLAGS_stereo)
    {
        //配置左右相机的投影和畸变模型,目前只支持针孔+radtan
        slave_img_feat_propagator_ptr.reset(new XP::ImgFeaturePropagator(
                duo_calib_param.Camera.cameraK_lr[1],  // cur_camK 右相机内参
                duo_calib_param.Camera.cameraK_lr[0],  // ref_camK 左相机内参
                duo_calib_param.Camera.cv_dist_coeff_lr[1],  // cur_dist_coeff 右相机畸变
                duo_calib_param.Camera.cv_dist_coeff_lr[0],  // ref_dist_coeff 左相机畸变
                duo_calib_param.Camera.fishEye,
                masks[1],//右相机的掩码
                FLAGS_pyra_level,//所有金字塔层数
                FLAGS_min_feature_distance_over_baseline_ratio,//特征点最小深度比例,用于极线搜索
                FLAGS_max_feature_distance_over_baseline_ratio));//特征点最大深度比例,用于极线搜索)

        //双目外参
        T_Cl_Cr = T_Cl_Cr_d.cast<float>();

    }
    //绘制前清除画布
    pose_viewer.set_clear_canvas_before_draw(true);
    if (FLAGS_save_feature) {
        //存储外参,内参文件
        IBA::SaveCalibration(FLAGS_result_folder + "/calibration.dat", to_iba_calibration(duo_calib_param));
    }

    //左右目的内外参,,是否要输出细节内容,是否输出debug信息,,参数所在位置
    solver.Create(to_iba_calibration(duo_calib_param),
                  257,
                  IBA_VERBOSE_NONE,
                  IBA_DEBUG_NONE,
                  257,
                  FLAGS_iba_param_path,//iba的配置文件
                  "" /* iba directory */);

    if(FLAGS_LoopClosure)
    {
        LoopCloser_ptr->SetCallback([&](const vector<Eigen::Matrix4f> & rKFpose/*参考关键帧Twc*/,
                                        const Eigen::Matrix4f & lKFpose,vector<int> riFrm,int liFrm)
        {
            pubLoopCamPose(lKFpose.cast<double>());
            for (int k = 0; k < rKFpose.size(); ++k)
            {
                LA::AlignedMatrix6x6f S;
                IBA::RelativeConstraint Z;

                Eigen::Matrix4f Tlr = lKFpose.inverse() * rKFpose[k];////Tc0(观测关键帧)c0(参考关键帧)
                float pose_f[3][4];
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 4; ++j)
                    {
                        pose_f[i][j] = Tlr(i,j);
                    }

                }
                Rigid3D T;
                T.Set(pose_f);////Tc0(观测关键帧)c0(参考关键帧)

                Z.iFrm1 = riFrm[k];
                Z.iFrm2 = liFrm;
                T.Rotation3D::Get(Z.T.R);
                T.GetPosition().Get(Z.T.p);
                S.MakeDiagonal(LOOP_S2P, LOOP_S2R);//固定一下参考关键帧
                S.Get(Z.S.S);
                m_loop_buf.lock();
                loop_info.push(Z);
                m_loop_buf.unlock();
            }


        });

    }
    //对GBA求解器设置回调函数m_callback,用来可视化
    solver.SetCallbackGBA([&](const int iFrm,/*最新一帧的id*/ const float ts/*最新一帧的时间戳*/)
    {

        if(FLAGS_LoopClosure)
        {
            IBA::Global_Map GM;
            solver.GetUpdateGba(&GM);
            LoopCloser_ptr->UpdateKfInfo(GM);
        }

        std::vector<std::pair<int,IBA::CameraPose>> total_kfs = solver.Get_Total_KFs();
        std::vector<std::pair<double,Eigen::Matrix4d>> kf_poses;
        kf_poses.resize(total_kfs.size());
        for (int kf_id = 0; kf_id < total_kfs.size(); ++kf_id)
        {
            kf_poses[kf_id].first = (double)total_time[total_kfs[kf_id].first] + offset_ts;
            kf_poses[kf_id].second = Eigen::Matrix4d::Identity();

            for (int i = 0; i < 3; ++i) {
                kf_poses[kf_id].second(i, 3) = (double)total_kfs[kf_id].second.p[i];
                for (int j = 0; j < 3; ++j) {
                    kf_poses[kf_id].second(i,j) = (double)total_kfs[kf_id].second.R[j][i];
                }
            }
        }
        pubKFsPose(kf_poses);
    });


    //对LBA求解器设置回调函数m_callback,用来可视化
    solver.SetCallbackLBA([&](const int iFrm,/*最新一帧的id*/ const float ts/*最新一帧的时间戳*/)
    {
        // as we may be able to send out information directly in the callback arguments
        IBA::SlidingWindow sliding_window;
        //获得LBA中的滑窗中更新了的普通帧以及更新了的关键帧还有更新了的地图点
        solver.GetSlidingWindow(&sliding_window);
        const IBA::CameraIMUState& X = sliding_window.CsLF.back();//最新的一帧
        const IBA::CameraPose& C = X.Cam_pose;
        Eigen::Matrix4f W_vio_T_S = Eigen::Matrix4f::Identity();  // Twc0最新的
        for (int i = 0; i < 3; ++i) {
            W_vio_T_S(i, 3) = C.p[i];
            for (int j = 0; j < 3; ++j) {
                W_vio_T_S(i, j) = C.R[j][i];  //因为存储的是C.R里是Rc0w,所以要转成Rwc0     Cam_state.R is actually R_SW
            }
        }
        Eigen::Matrix4d Twc0 = W_vio_T_S.cast<double>();
        Eigen::Matrix<float, 9, 1> speed_and_biases;
        Eigen::Matrix<double , 9, 1> speed_and_biases_d;
        for (int i = 0; i < 3; ++i) {
            speed_and_biases(i) = X.v[i];
            speed_and_biases(i + 3) = X.ba[i];
            speed_and_biases(i + 6) = X.bw[i];

            speed_and_biases_d(i) = (double)X.v[i];
            speed_and_biases_d(i + 3) = (double)X.ba[i];
            speed_and_biases_d(i + 6) = (double)X.bw[i];
        }

        Eigen::Vector3f cur_position = W_vio_T_S.topRightCorner(3, 1);
        travel_dist += (cur_position - last_position).norm();
        last_position = cur_position;
        pose_viewer.addPose(W_vio_T_S, speed_and_biases, travel_dist);

        CHECK_EQ(sliding_window.iFrms.size(),sliding_window.CsLF.size());

        double cur_time =(double)total_time[sliding_window.iFrms.back()] + offset_ts;

        if(!FLAGS_UseIMU)
        {
            vector<int> track_idx;
            for (int i = 0; i < cur_track.size(); ++i)
                track_idx.push_back(std::get<0>(cur_track[i]));
            solver.Get_cur_Mps(track_idx,cur_Mp_info);
            last_pose_Twc0 = W_vio_T_S;
        }

        //ROS pub
//        pubUpdatePointClouds(sliding_window.Xs,cur_time);
        pubLatestCameraPose(Twc0,speed_and_biases_d.block<3,1>(0,0),cur_time);
        pubTF(Twc0,cur_time);


    });


    //启动求解器
    solver.Start();



    //左右相机同步
    std::thread sync_thread{sync_stereo};
    //前端
    std::thread measurement_process{process_frontend};
    //后端
    std::thread Backend_process{process_backend};
    //
    ros::Subscriber sub_imu = n.subscribe(FLAGS_imu_topic, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_img0 = n.subscribe(FLAGS_image0_topic, 100, img0_callback);
    ros::Subscriber sub_img1 = n.subscribe(FLAGS_image1_topic, 100, img1_callback);

    ros::Subscriber sub_odom = n.subscribe("/ice_ba/imu_pose", 1000, odom_callback);

    ros::spin();


    return 0;
}