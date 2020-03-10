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
namespace fs = boost::filesystem;
using std::string;
using std::vector;
using stereo_Img = std::pair<cv::Mat,cv::Mat>;
using stereo_Img_info_d = std::pair<double,stereo_Img>;
using stereo_Img_info_f = std::pair<float,stereo_Img>;
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
DEFINE_string(result_folder, "", "Save the result");
DEFINE_bool(stereo, false, "monocular or stereo mode");
DEFINE_bool(save_feature, false, "Save features to .dat file");
DEFINE_string(image0_topic, "", "左目话题");
DEFINE_string(image1_topic, "", "右目话题");
DEFINE_string(imu_topic, "", "imu话题");
DEFINE_bool(LoopClosure, true, "use LoopClosure?");
DEFINE_bool(GetGT, true, "GTdata");
DEFINE_string(GT_path, "", "");
DEFINE_string(dbow3_voc_path, "", "dbow3_voc_path,must set!");


std::queue<IBA::RelativeConstraint> loop_info;
std::queue<sensor_msgs::ImuConstPtr> imu_buf;
std::queue<sensor_msgs::ImageConstPtr> img0_buf;
std::queue<sensor_msgs::ImageConstPtr> img1_buf;

std::queue<IBA::CurrentFrame> CFs_buf;
std::queue<IBA::KeyFrame> KFs_buf;
std::queue<stereo_Img_info_d> stereo_buf;
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
bool set_offset_time = false;
bool set_relative_pose = false;
bool init_first_track = true;

Eigen::Matrix4d relative_pose; //Twc(first) * Twc.inv(gt)
// Create masks based on FOVs computed from intrinsics
std::vector<cv::Mat_<uchar> > masks(2);

float prev_img_time_stamp = 0.0f;
// load previous image//左目之前的特征点和描述子
std::vector<cv::KeyPoint> pre_image_key_points;
cv::Mat pre_image_features;

int idx = 1;
struct Data;
vector<Data> benchmark;
int GT_skip = 0;//等稳定了
int init = 0;
Eigen::Quaterniond baseRgt;
Eigen::Vector3d baseTgt;
tf::Transform trans;


struct Data
{
    Data(FILE *f)
    {
        if (fscanf(f, " %lf,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", &t,
                   &px, &py, &pz,
                   &qw, &qx, &qy, &qz,
                   &vx, &vy, &vz,
                   &wx, &wy, &wz,
                   &ax, &ay, &az) != EOF)
        {
            t /= 1e9;
            px += -7.48903e-02;//euroc的真值是TRM,TMb是的R是单位阵,所以我在这里直接加了t
            py += 1.84772e-02;
            pz += 1.20209e-01;

        }
    }
    double t;
    float px, py, pz;
    float qw, qx, qy, qz;
    float vx, vy, vz;
    float wx, wy, wz;
    float ax, ay, az;
};

void odom_callback(const nav_msgs::OdometryConstPtr &odom_msg)
{
    if(!FLAGS_GetGT)
        return;

    if (odom_msg->header.stamp.toSec() > benchmark.back().t)
        return;

    for (; idx < static_cast<int>(benchmark.size()) && benchmark[idx].t <= odom_msg->header.stamp.toSec(); idx++)
        ;

    if (init++ < GT_skip)
    {
        baseRgt = Eigen::Quaterniond(odom_msg->pose.pose.orientation.w,
                              odom_msg->pose.pose.orientation.x,
                              odom_msg->pose.pose.orientation.y,
                              odom_msg->pose.pose.orientation.z) *
                Eigen::Quaterniond(benchmark[idx - 1].qw,
                              benchmark[idx - 1].qx,
                              benchmark[idx - 1].qy,
                              benchmark[idx - 1].qz).inverse();
        baseTgt = Eigen::Vector3d{odom_msg->pose.pose.position.x,
                           odom_msg->pose.pose.position.y,
                           odom_msg->pose.pose.position.z} -
                  baseRgt * Eigen::Vector3d{benchmark[idx - 1].px, benchmark[idx - 1].py , benchmark[idx - 1].pz};
        return;
    }

    nav_msgs::Odometry odometry;
    Eigen::Vector3d tmp_T = baseTgt + baseRgt * Eigen::Vector3d{benchmark[idx - 1].px, benchmark[idx - 1].py, benchmark[idx - 1].pz};
    Eigen::Quaterniond tmp_R = baseRgt * Eigen::Quaterniond{benchmark[idx - 1].qw,
                                              benchmark[idx - 1].qx,
                                              benchmark[idx - 1].qy,
                                              benchmark[idx - 1].qz};

    Eigen::Vector3d tmp_V = baseRgt * Eigen::Vector3d{benchmark[idx - 1].vx,
                                        benchmark[idx - 1].vy,
                                        benchmark[idx - 1].vz};

    Eigen::Matrix4d GT_Twb = Eigen::Matrix4d::Identity();
    GT_Twb.block<3,3>(0,0) = tmp_R.toRotationMatrix();
    GT_Twb.block<3,1>(0,3) = tmp_T;
    Eigen::Matrix4d GT_Twc0 = GT_Twb * Tbc0;
    pubGTCamPose(GT_Twc0,benchmark[idx - 1].t);
}

void load_Parameters(const std::string &config_path,XP::DuoCalibParam &calib_param)
{
    int pn = config_path.find_last_of('/');
    std::string config_folder = config_path.substr(0, pn);
    YAML::Node config_yaml = YAML::LoadFile(config_path);
    std::vector<double> v_double;
    v_double = config_yaml["body_T_imu"]["data"].as<std::vector<double>>();
    Eigen::Matrix4d b_t_i = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(&v_double[0]);


    FLAGS_stereo = config_yaml["num_of_cam"].as<int>() == 2;
    if(FLAGS_stereo == false)//暂时只有双目版本开源,开源的虽然有单目脚本,但是单目用起来是飞,得做一下初始化以及三角化点的深度
        LOG(FATAL) << "unsupported mono-vio";
    //话题
    FLAGS_image0_topic = config_yaml["image0_topic"].as<string>();
    FLAGS_image1_topic = config_yaml["image1_topic"].as<string>();
    FLAGS_imu_topic = config_yaml["imu_topic"].as<string>();
    std::cout<<"image0_topic: "<<FLAGS_image0_topic<<std::endl
             <<"image1_topic: "<<FLAGS_image1_topic<<std::endl
             <<"imu_topic: "<<FLAGS_imu_topic<<std::endl;

    FLAGS_LoopClosure = config_yaml["loop_closure"].as<int>() == 1;
    FLAGS_dbow3_voc_path = config_yaml["dbow3_voc_path"].as<string>();

    IMU_VARIANCE_ACCELERATION_NOISE = float(std::pow(config_yaml["acc_n"].as<double>(),2));
    IMU_VARIANCE_ACCELERATION_BIAS_WALK = float(std::pow(config_yaml["acc_w"].as<double>(),2));
    IMU_VARIANCE_GYROSCOPE_NOISE = float(std::pow(config_yaml["gyr_n"].as<double>(),2));
    IMU_VARIANCE_GYROSCOPE_BIAS_WALK = float(std::pow(config_yaml["gyr_w"].as<double>(),2));
    IMU_GRAVITY_MAGNITUDE = config_yaml["g_norm"].as<float>();

    FLAGS_GetGT = config_yaml["show_GT"].as<int>() == 1;
    if(FLAGS_GetGT)
    {
        FLAGS_GT_path =  config_yaml["GT_path"].as<string>();
        GT_skip = config_yaml["GT_SKIP"].as<int>();
    }

    int Num_Cam = FLAGS_stereo ? 2:1;
    std::vector<float> v_float;
    Eigen::Matrix4d b_t_c0,b_t_c1;
    for (int cam_id = 0; cam_id < Num_Cam; ++cam_id)
    {
        std::string cam_string = "cam" + std::to_string(cam_id) + "_calib";

        std::string cam_yaml = config_folder + "/" + config_yaml[cam_string].as<string>();
        YAML::Node cam_calib = YAML::LoadFile(cam_yaml);
        //内参
        v_float = cam_calib["intrinsics"].as<std::vector<float>>();
        calib_param.Camera.cv_camK_lr[cam_id] << v_float[0], 0, v_float[2],
                0, v_float[1], v_float[3],
                0, 0, 1;
        calib_param.Camera.cameraK_lr[cam_id] << v_float[0], 0, v_float[2],
                0, v_float[1], v_float[3],
                0, 0, 1;
        //畸变 //TODO:Equi
        v_double = cam_calib["distortion_coefficients"].as<std::vector<double>>();
        calib_param.Camera.cv_dist_coeff_lr[cam_id] = (cv::Mat_<float>(8, 1) << static_cast<float>(v_double[0]), static_cast<float>(v_double[1]),
                static_cast<float>(v_double[2]), static_cast<float>(v_double[3]), 0.0, 0.0, 0.0, 0.0);

        string distort_model = cam_calib["distortion_model"].as<string>();
        if(distort_model == "equidistant")
            calib_param.Camera.fishEye = true;
        else
            calib_param.Camera.fishEye = false;
        //外参
        v_double = cam_calib["body_T_cam"]["data"].as<std::vector<double>>();


        if(cam_id == 0)
            b_t_c0 = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(&v_double[0]);
        else
            b_t_c1 = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(&v_double[0]);
    }
    Tbc0 = b_t_c0;
    std::cout<<"fisheye?"<<calib_param.Camera.fishEye<<std::endl;

    // ASL {B}ody frame is the IMU
    // {D}evice frame is the left camera
    //百度这个代码的表示方式：d_t_cam0就是Tdcam0,即相机0到设备坐标系的变换,这里把设备坐标系固连于左相机坐标系
    Eigen::Matrix4d d_t_cam0 = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d d_t_b = d_t_cam0 * b_t_c0.inverse();// Tdb = Tdc0 * Tc0b本体坐标系到设备坐标系
    if(FLAGS_stereo)
    {
        Eigen::Matrix4d d_t_cam1 = d_t_b * b_t_c1;//Tdc1 右相机到设备坐标系
        calib_param.Camera.D_T_C_lr[1] = d_t_cam1.cast<float>();
    }
    Eigen::Matrix4d d_t_imu = d_t_b * b_t_i; //imu坐标系到设备坐标系
    //设置关于设备坐标系的参数
    calib_param.Camera.D_T_C_lr[0] = Eigen::Matrix4f::Identity();
    // Image size 图像大小
    std::vector<int> v_int = config_yaml["resolution"].as<std::vector<int>>();
    calib_param.Camera.img_size = cv::Size(v_int[0], v_int[1]);
    // IMU
    calib_param.Imu.accel_TK = Eigen::Matrix3f::Identity();
    calib_param.Imu.accel_bias = Eigen::Vector3f::Zero();
    calib_param.Imu.gyro_TK = Eigen::Matrix3f::Identity();
    calib_param.Imu.gyro_bias = Eigen::Vector3f::Zero();
    calib_param.Imu.accel_noise_var = Eigen::Vector3f{0.0016, 0.0016, 0.0016};
    calib_param.Imu.angv_noise_var = Eigen::Vector3f{0.0001, 0.0001, 0.0001};
    calib_param.Imu.D_T_I = d_t_imu.cast<float>();
    calib_param.device_id = "ASL";
    calib_param.sensor_type = XP::DuoCalibParam::SensorType::UNKNOWN;

    //进行双目立体矫正,事先设置好左右目的remap
//    calib_param.initUndistortMap(calib_param.Camera.img_size);




}



inline bool cmp_by_class_id(const cv::KeyPoint& lhs, const cv::KeyPoint& rhs)  {
    return lhs.class_id < rhs.class_id;
}

template <typename T>
void InitPOD(T& t) {
    memset(&t, 0, sizeof(t));
}
//输入左相机特征点,右相机特征点,imu测量,左目时间戳,当前帧,关键帧
//进行点管理,生成当前帧(imu数据,对于老地图点的观测),并且判断是否要生成关键帧(阈值为看到老地图点的多少或者需要生成新地图点的多少),
// 如果要生成关键帧，将新的地图点push进Xs保存到关键帧数据结构中
bool create_iba_frame(const vector<cv::KeyPoint>& kps_l,
                      const vector<cv::KeyPoint>& kps_r,
                      const vector<XP::ImuData>& imu_samples,
                      const float rig_time,
                      IBA::CurrentFrame* ptrCF, IBA::KeyFrame* ptrKF) {

    CHECK(std::is_sorted(kps_l.begin(), kps_l.end(), cmp_by_class_id));
    CHECK(std::is_sorted(kps_r.begin(), kps_r.end(), cmp_by_class_id));
    CHECK(std::includes(kps_l.begin(), kps_l.end(), kps_r.begin(), kps_r.end(), cmp_by_class_id));

    // IBA will handle *unknown* initial depth values
    //IBA将处理*未知*初始深度值
    IBA::Depth kUnknownDepth;
    kUnknownDepth.d = 0.0f;
    kUnknownDepth.s2 = 0.0f;
    static int last_added_point_id = -1;//用来判断是否是新地图点
    static int iba_iFrm = 0;//全局frame_idx 从0开始
    auto kp_it_l = kps_l.cbegin(), kp_it_r = kps_r.cbegin();

    IBA::CurrentFrame& CF = *ptrCF;
    IBA::KeyFrame& KF = *ptrKF;

    CF.iFrm = iba_iFrm;
    InitPOD(CF.Cam_state);//初始化一下C中的值 // needed to ensure the dumped frame deterministic even for unused field
    CF.Cam_state.Cam_pose.R[0][0] = CF.Cam_state.v[0] = CF.Cam_state.ba[0] = CF.Cam_state.bw[0] = FLT_MAX;
    // MapPointMeasurement, process in ascending class id, left camera to right
    // Note the right keypoints is acc subset of the left ones
    IBA::MapPointMeasurement mp_mea;//地图点的测量
    InitPOD(mp_mea);
    //这里没有加权,协方差直接是单位阵
    mp_mea.x.S[0][0] = mp_mea.x.S[1][1] = 1.f;
    mp_mea.x.S[0][1] = mp_mea.x.S[1][0] = 0.f;

    //第一帧的时候这里是跳过的,因为还没有last_added_point_id
    //class_id小于上次最后一个地图点id,说明这个点是老地图点的测量,先处理老地图点的观测
    //这里需要一点,KF的class_id不是连续的,因为追踪过程是普通帧追踪上一帧的地图点,而这个地图点非kf里的地图点,比如说这帧有新的地图点,75,76,77,但是
    //地图点太少,不算关键帧,到了能判定关键帧的时候,可能这个关键帧看到的是74,76,78---，那么75,77是没有的,last_added_point_id是上一个关键帧的最后一个点的id
    for (; kp_it_l != kps_l.cend() && kp_it_l->class_id <= last_added_point_id; ++kp_it_l)
    {
        //将左侧特征点的id,测量值,均赋值给地图点的测量
        mp_mea.idx = kp_it_l->class_id;
        mp_mea.x.x[0] = kp_it_l->pt.x;
        mp_mea.x.x[1] = kp_it_l->pt.y;
        mp_mea.right = false;
        CF.feat_measures.push_back(mp_mea);
        //kp_it_r里的点数量<= kp_it_l点的数量,所以class_id要么和l的相等要么提前于l
        //如果右目也观测到了地图点,将右目状态更新,也push进去
        if (kp_it_r != kps_r.cend() && kp_it_r->class_id == kp_it_l->class_id) {
            mp_mea.x.x[0] = kp_it_r->pt.x;
            mp_mea.x.x[1] = kp_it_r->pt.y;
            mp_mea.right = true;
            CF.feat_measures.push_back(mp_mea);
            ++kp_it_r;
        }
    }
    //imu_samples数据类型转成CF.imu_measure所需要的数据类型
    std::transform(imu_samples.begin(), imu_samples.end(), std::back_inserter(CF.imu_measures), XP::to_iba_imu);
    CF.t = rig_time;//当前帧时间戳
    CF.d = kUnknownDepth;
    //需要一个新关键帧的条件:2个
    // 1: std::distance(kp_it_l, kps_l.end()) >= 20说明的是没观测到的新的地图点比较多,大于20个
    // 2: CF.feat_measures.size() < 20说明的是当前帧观测到的地图点数量太少了，不足20个
    bool need_new_kf = std::distance(kp_it_l, kps_l.end()) >= (kps_l.size()/3) || CF.feat_measures.size() < (kps_l.size()/3);
    if (std::distance(kp_it_l, kps_l.end()) == 0)
        need_new_kf = false;
    if (!need_new_kf) KF.iFrm = -1;//如果不需要关键帧
    else
        LOG(INFO) << "new keyframe " << KF.iFrm;

    if (!need_new_kf)
    {
        KF.iFrm = -1;
        //  to make it deterministic
        InitPOD(KF.Cam_pose);//初始化一下,全赋0
        InitPOD(KF.d);
    } else {
        //如果需要增加关键帧的话,就把帧id,pose,对于老地图点的观测给关键帧
        KF.iFrm = CF.iFrm;
        KF.Cam_pose = CF.Cam_state.Cam_pose;
        // MapPointMeasurement, duplication of CF
        KF.feat_measures = CF.feat_measures;
        // MapPoint
        //现在kp_it_l已经遍历到新的地图点处了,后面的都是新的地图点
        for(; kp_it_l != kps_l.cend(); ++kp_it_l) {
            IBA::MapPoint mp;
            InitPOD(mp.X);
            mp.X.idx = kp_it_l->class_id;//全局id
            mp.X.X[0] = FLT_MAX;
            //mappoint的观测,哪一帧,像素坐标,是哪一目看到的都记录下来,push进这个地图点的数据结构里，注意新的地图点的观测是不push进KF.feat_measures的
            mp_mea.iFrm = iba_iFrm;
            mp_mea.x.x[0] = kp_it_l->pt.x;
            mp_mea.x.x[1] = kp_it_l->pt.y;
            mp_mea.right = false;
            mp.feat_measures.push_back(mp_mea);

            if (kp_it_r != kps_r.cend() && kp_it_r->class_id == kp_it_l->class_id) {
                mp_mea.x.x[0] = kp_it_r->pt.x;
                mp_mea.x.x[1] = kp_it_r->pt.y;
                mp_mea.right = true;
                mp.feat_measures.push_back(mp_mea);
                kp_it_r++;
            } else {
                LOG(WARNING) << "add new feature point " << kp_it_l->class_id << " only found in left image";
            }
            //将新的地图点push进,也就是说关键帧数据结构中Xs只存由它首次观测到的新地图点
            KF.Xs.push_back(mp);
        }
        last_added_point_id = std::max(KF.Xs.back().X.idx, last_added_point_id);//最后一个新的地图点的id
        KF.d = kUnknownDepth;
    }
    ++iba_iFrm;//全局frame_idx更新
    return true;
}

std::vector<std::pair<std::vector<XP::ImuData>, stereo_Img_info_f >> getMeasurements()
{
    std::vector<std::pair<std::vector<XP::ImuData>, stereo_Img_info_f >> measurements;

    while (true)
    {
        if (imu_buf.empty() || stereo_buf.empty())
            return measurements;

        if (imu_buf.back()->header.stamp.toSec() <= stereo_buf.front().first)
        {
//            std::cout << "wait for imu, only should happen at the beginning sum_of_wait: "<< std::endl;
            return measurements;
        }

        if (imu_buf.front()->header.stamp.toSec() >= stereo_buf.front().first)
        {
//            std::cout << "throw img, only should happen at the beginning" << std::endl;
            stereo_buf.pop();
            continue;
        }
        stereo_Img_info_d stereo_info = stereo_buf.front();
        stereo_buf.pop();
        XP::ImuData imu_sample;
        std::vector<XP::ImuData> IMUs;
        IMUs.reserve(10);
        while (imu_buf.front()->header.stamp.toSec() < stereo_info.first )
        {
            imu_sample.time_stamp = (float)(imu_buf.front()->header.stamp.toSec() - offset_ts);
            imu_sample.ang_v(0) = (float)imu_buf.front()->angular_velocity.x;
            imu_sample.ang_v(1) = (float)imu_buf.front()->angular_velocity.y;
            imu_sample.ang_v(2) = (float)imu_buf.front()->angular_velocity.z;
            imu_sample.accel(0) = (float)imu_buf.front()->linear_acceleration.x;
            imu_sample.accel(1) = (float)imu_buf.front()->linear_acceleration.y;
            imu_sample.accel(2) = (float)imu_buf.front()->linear_acceleration.z;
            IMUs.push_back(imu_sample);
            imu_buf.pop();
        }

        stereo_Img_info_f stereo_info_f;
        stereo_info_f.second = stereo_info.second;
        stereo_info_f.first = (float)(stereo_info.first -= offset_ts);
        measurements.emplace_back(IMUs, stereo_info_f);
    }
    return measurements;
}

bool process_frontend()
{
    while (true)
    {
        std::vector<std::pair<std::vector<XP::ImuData>, stereo_Img_info_f >> measurements;

        std::unique_lock <std::mutex> lk(m_buf);
        con.wait(lk, [&] {
            return (measurements = getMeasurements()).size() != 0;//将目前所有的测量取出来
        });
        if (measurements.size() > 1) {
            std::cout << "1 getMeasurements size: " << measurements.size()
                      << " imu sizes: " << measurements[0].first.size()
                      << " stereo_buf size: " << stereo_buf.size()
                      << " imu_buf size: " << imu_buf.size() << std::endl;
        }
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

            if(init_first_track)
                init_first_track = false;
            else
            {
                if(FLAGS_show_track_img)
                    pubTrackImage(img_in_smooth,slave_img_smooth,pre_image_key_points,key_pnts,key_pnts_slave,(double)img_time_stamp + offset_ts);
            }

            pre_image_key_points = key_pnts;
            pre_image_features = orb_feat.clone();

            prev_img_time_stamp = img_time_stamp;
        }

    }
}


void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    //设置一下第一个imu的时间
    if (!set_offset_time) {
        set_offset_time = true;
        offset_ts = imu_msg->header.stamp.toSec();
    }
    m_syn_buf.lock();
    imu_buf.push(imu_msg);
    m_syn_buf.unlock();
    con.notify_one();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

// extract images with same timestamp from two topics
void sync_stereo()
{
    while(1)
    {
        if(FLAGS_stereo)
        {
            cv::Mat image0, image1;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();
                // sync tolerance
                double sync_tolerance = 0.008;
                if(time0 < time1 - sync_tolerance)
                {
                    img0_buf.pop();
                    printf("throw img0\n");
                }
                else if(time0 > time1 + sync_tolerance)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    image0 = getImageFromMsg(img0_buf.front());
                    image1 = getImageFromMsg(img1_buf.front());
                    img0_buf.pop();
                    img1_buf.pop();
                }
            }
            m_buf.unlock();
            if(!image1.empty())
            {
                m_syn_buf.lock();
                stereo_buf.push(std::make_pair(time,std::make_pair(image0,image1)));
                m_syn_buf.unlock();
                con.notify_one();
            }
        }
        else
        {
            //暂时不支持单目
        }
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}


bool command()
{

    while(true)
    {

        char c = getchar();

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);


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
            if(total_time.size() > CF.iFrm)//先这么存一下吧,先不考虑上限的问题
                total_time[CF.iFrm] = CF.t;
            else
            {
                total_time.resize(total_time.size()+300);
                total_time[CF.iFrm] = CF.t;
            }

            solver.PushCurrentFrame(CF, KF.iFrm == -1 ? nullptr : &KF);//先说明一下,我习惯的求解增量的表达是Hx=b,但是这里是Hx=-b

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

    if(FLAGS_GetGT)
    {
        string csv_file = FLAGS_GT_path;
        std::cout << "load ground truth " << csv_file << std::endl;
        FILE *f = fopen(csv_file.c_str(), "r");
        if (f==NULL)
        {
            ROS_WARN("can't load ground truth; wrong path");
            //std::cerr << "can't load ground truth; wrong path " << csv_file << std::endl;
            return 0;
        }
        char tmp[10000];
        if (fgets(tmp, 10000, f) == NULL)
        {
            ROS_WARN("can't load ground truth; no data available");
        }
        while (!feof(f))
            benchmark.emplace_back(f);
        fclose(f);
        benchmark.pop_back();
        ROS_INFO("Data loaded: %d", (int)benchmark.size());
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
    T_Cl_Cr = duo_calib_param.Camera.D_T_C_lr[0].inverse() * duo_calib_param.Camera.D_T_C_lr[1];
    T_Cl_Cr_d = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            T_Cl_Cr_d(i,j) = (double)T_Cl_Cr(i,j);
        }
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
        Eigen::Matrix4d Twc0 = Eigen::Matrix4d::Identity();  // Twc0最新的
        for (int i = 0; i < 3; ++i) {
            W_vio_T_S(i, 3) = C.p[i];
            Twc0(i, 3) = (double)C.p[i];
            for (int j = 0; j < 3; ++j) {
                W_vio_T_S(i, j) = C.R[j][i];  //因为存储的是C.R里是Rc0w,所以要转成Rwc0     Cam_state.R is actually R_SW
                Twc0(i,j) = (double)C.R[j][i];
            }
        }
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

        //ROS pub
        pubLatestCameraPose(Twc0,speed_and_biases_d.block<3,1>(0,0),cur_time);
        pubTF(Twc0,cur_time);
    });


    //启动求解器
    solver.Start();

//    std::thread keyboard_command_process;
//    keyboard_command_process = std::thread(command);

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