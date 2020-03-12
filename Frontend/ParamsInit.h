//
// Created by wya on 2020/3/11.
//

#ifndef ICE_BA_PARAMSINIT_H
#define ICE_BA_PARAMSINIT_H
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "PnpPoseEstimator.h"
DECLARE_int32(lc_num_ransac_iters);
DECLARE_bool(lc_nonlinear_refinement_p3p);
DECLARE_double(lc_ransac_pixel_sigma);
DECLARE_int32(lc_min_inlier_count);
DECLARE_double(lc_min_inlier_ratio);
DECLARE_bool(lc_use_random_pnp_seed);
DECLARE_int32(Match_count);

namespace fs = boost::filesystem;
using std::string;
using std::vector;
using stereo_Img = std::pair<cv::Mat,cv::Mat>;
using stereo_Img_info_d = std::pair<double,stereo_Img>;
using stereo_Img_info_f = std::pair<float,stereo_Img>;

int idx = 1;
struct Data;
vector<Data> benchmark;
int GT_skip = 0;//等稳定了
int init = 0;
Eigen::Quaterniond baseRgt;
Eigen::Vector3d baseTgt;
tf::Transform trans;

extern Eigen::Matrix4d T_Cl_Cr_d;
extern Eigen::Matrix4d Tbc0;

std::queue<sensor_msgs::ImuConstPtr> imu_buf;
std::queue<sensor_msgs::ImageConstPtr> img0_buf;
std::queue<sensor_msgs::ImageConstPtr> img1_buf;
std::queue<stereo_Img_info_d> stereo_buf;

extern std::mutex m_buf,m_syn_buf;
extern std::condition_variable con;
extern double offset_ts;
bool set_offset_time = false;
int MIN_PARALLAX = 20;
std::shared_ptr<vio::cameras::NCameraSystem> Ncamera_ptr;
DECLARE_bool(stereo);
DECLARE_bool(UseIMU);
DECLARE_bool(LoopClosure);
DECLARE_bool(GetGT);

DECLARE_string(image0_topic);
DECLARE_string(image1_topic);
DECLARE_string(imu_topic);
DECLARE_string(iba_param_path);
DECLARE_string(dbow3_voc_path);
DECLARE_string(GT_path);
DECLARE_string(gba_result_path);
///这里就是放一些无用的东西

//配置文件读取
void load_Parameters(const std::string &config_path,XP::DuoCalibParam &calib_param)
{
    int pn = config_path.find_last_of('/');
    std::string config_folder = config_path.substr(0, pn);
    YAML::Node config_yaml = YAML::LoadFile(config_path);
    std::vector<double> v_double;
    v_double = config_yaml["body_T_imu"]["data"].as<std::vector<double>>();
    Eigen::Matrix4d b_t_i = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(&v_double[0]);


    FLAGS_stereo = config_yaml["num_of_cam"].as<int>() == 2;

    FLAGS_gba_result_path = config_yaml["GBA_result_path"].as<string>();

    FLAGS_UseIMU = config_yaml["imu"].as<int>() == 1;
    //话题
    FLAGS_image0_topic = config_yaml["image0_topic"].as<string>();
    if(FLAGS_stereo)
        FLAGS_image1_topic = config_yaml["image1_topic"].as<string>();
    if(FLAGS_UseIMU)
        FLAGS_imu_topic = config_yaml["imu_topic"].as<string>();
    std::cout<<"image0_topic: "<<FLAGS_image0_topic<<std::endl
             <<"image1_topic: "<<FLAGS_image1_topic<<std::endl
             <<"imu_topic: "<<FLAGS_imu_topic<<std::endl;

    if(FLAGS_stereo)
        FLAGS_iba_param_path = "../config/config_of_stereo.txt";
    else
        FLAGS_iba_param_path = "../config/config_of_mono.txt";


    FLAGS_LoopClosure = config_yaml["loop_closure"].as<int>() == 1;
    FLAGS_dbow3_voc_path = config_yaml["dbow3_voc_path"].as<string>();

    if(FLAGS_UseIMU)
    {
        IMU_VARIANCE_ACCELERATION_NOISE = float(std::pow(config_yaml["acc_n"].as<double>(),2));
        IMU_VARIANCE_ACCELERATION_BIAS_WALK = float(std::pow(config_yaml["acc_w"].as<double>(),2));
        IMU_VARIANCE_GYROSCOPE_NOISE = float(std::pow(config_yaml["gyr_n"].as<double>(),2));
        IMU_VARIANCE_GYROSCOPE_BIAS_WALK = float(std::pow(config_yaml["gyr_w"].as<double>(),2));
        IMU_GRAVITY_MAGNITUDE = config_yaml["g_norm"].as<float>();
    }
    else
    {
        IMU_GRAVITY_EXCLUDED = true;
        LBA_PROPAGATE_CAMERA = false;
    }

    FLAGS_GetGT = config_yaml["show_GT"].as<int>() == 1;
    if(FLAGS_GetGT)
    {
        FLAGS_GT_path =  config_yaml["GT_path"].as<string>();
        GT_skip = config_yaml["GT_SKIP"].as<int>();

        string csv_file = FLAGS_GT_path;
        std::cout << "load ground truth " << csv_file << std::endl;
        FILE *f = fopen(csv_file.c_str(), "r");
        if (f==NULL)
        {
            ROS_WARN("can't load ground truth; wrong path");
            std::cerr << "can't load ground truth; wrong path " << csv_file << std::endl;
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
    std::vector<float> v_float;
    vector<Eigen::Matrix4d> b_t_c;//b_t_c0,b_t_c1;
    b_t_c.resize(Num_Cam);
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
            b_t_c[0] = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(&v_double[0]);
        else
            b_t_c[1] = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(&v_double[0]);
    }
    Tbc0 = b_t_c[0];
    T_Cl_Cr_d = b_t_c[0].inverse() * b_t_c[1];


    std::cout<<"fisheye?"<<calib_param.Camera.fishEye<<std::endl;

    // ASL {B}ody frame is the IMU
    // {D}evice frame is the left camera
    //百度这个代码的表示方式：d_t_cam0就是Tdcam0,即相机0到设备坐标系的变换,这里把设备坐标系固连于左相机坐标系
    Eigen::Matrix4d d_t_cam0 = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d d_t_b = d_t_cam0 * b_t_c[0].inverse();// Tdb = Tdc0 * Tc0b本体坐标系到设备坐标系
    if(FLAGS_stereo)
    {
        Eigen::Matrix4d d_t_cam1 = d_t_b * b_t_c[1];//Tdc1 右相机到设备坐标系
        calib_param.Camera.D_T_C_lr[1] = d_t_cam1.cast<float>();
    }
    Eigen::Matrix4d d_t_imu = d_t_b * b_t_i; //imu坐标系到设备坐标系
    //设置关于设备坐标系的参数
    calib_param.Camera.D_T_C_lr[0] = Eigen::Matrix4f::Identity();
    // Image size 图像大小
    std::vector<int> v_int = config_yaml["resolution"].as<std::vector<int>>();
    calib_param.Camera.img_size = cv::Size(v_int[0], v_int[1]);
    if(FLAGS_UseIMU)
    {
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
    }

    if(!FLAGS_UseIMU)//不用imu的话,双目就得用PNP给个位姿初值
    {
        Ncamera_ptr.reset(new vio::cameras::NCameraSystem());
        for (int i = 0; i < Num_Cam; ++i )
        {
            std::shared_ptr<const vio::cameras::CameraBase> curcam;
            std::shared_ptr<const Eigen::Matrix4d> b_T_c_ptr(new Eigen::Matrix4d(b_t_c[i]));

            if(calib_param.Camera.fishEye)
            {
                curcam.reset(new vio::cameras::PinholeCamera<
                        vio::cameras::EquidistantDistortion>(
                        calib_param.Camera.img_size.width,
                        calib_param.Camera.img_size.height,
                        calib_param.Camera.cameraK_lr[i](0, 0),  // focalLength[0],
                        calib_param.Camera.cameraK_lr[i](1, 1),  // focalLength[1],
                        calib_param.Camera.cameraK_lr[i](0, 2),  // principalPoint[0],
                        calib_param.Camera.cameraK_lr[i](1, 2),  // principalPoint[1],
                        vio::cameras::EquidistantDistortion(
                                calib_param.Camera.cv_dist_coeff_lr[i](0),
                                calib_param.Camera.cv_dist_coeff_lr[i](1),
                                calib_param.Camera.cv_dist_coeff_lr[i](2),
                                calib_param.Camera.cv_dist_coeff_lr[i](3))));

                Ncamera_ptr->addCamera(b_T_c_ptr,curcam,vio::cameras::NCameraSystem::Equidistant,false);

            }
            else if (calib_param.Camera.cv_dist_coeff_lr[i].rows == 8)//初始化右相机的针孔投影、rantan畸变模型
            {
                curcam.reset(new vio::cameras::PinholeCamera<
                        vio::cameras::RadialTangentialDistortion8>(
                        calib_param.Camera.img_size.width,
                        calib_param.Camera.img_size.height,
                        calib_param.Camera.cameraK_lr[i](0, 0),  // focalLength[0], fu
                        calib_param.Camera.cameraK_lr[i](1, 1),  // focalLength[1], fv
                        calib_param.Camera.cameraK_lr[i](0, 2),  // principalPoint[0], cx
                        calib_param.Camera.cameraK_lr[i](1, 2),  // principalPoint[1], cy
                        vio::cameras::RadialTangentialDistortion8(//目前只有randtan模型
                                calib_param.Camera.cv_dist_coeff_lr[i](0),
                                calib_param.Camera.cv_dist_coeff_lr[i](1),
                                calib_param.Camera.cv_dist_coeff_lr[i](2),
                                calib_param.Camera.cv_dist_coeff_lr[i](3),
                                calib_param.Camera.cv_dist_coeff_lr[i](4),
                                calib_param.Camera.cv_dist_coeff_lr[i](5),
                                calib_param.Camera.cv_dist_coeff_lr[i](6),
                                calib_param.Camera.cv_dist_coeff_lr[i](7))));
                Ncamera_ptr->addCamera(b_T_c_ptr,curcam,vio::cameras::NCameraSystem::RadialTangential8,false);
            } else if (calib_param.Camera.cv_dist_coeff_lr[i].rows == 4) {
                curcam.reset(new vio::cameras::PinholeCamera<
                        vio::cameras::RadialTangentialDistortion>(
                        calib_param.Camera.img_size.width,
                        calib_param.Camera.img_size.height,
                        calib_param.Camera.cameraK_lr[i](0, 0),  // focalLength[0], fu
                        calib_param.Camera.cameraK_lr[i](1, 1),  // focalLength[1], fv
                        calib_param.Camera.cameraK_lr[i](0, 2),  // principalPoint[0], cx
                        calib_param.Camera.cameraK_lr[i](1, 2),  // principalPoint[1], cy
                        vio::cameras::RadialTangentialDistortion(
                                calib_param.Camera.cv_dist_coeff_lr[i](0),
                                calib_param.Camera.cv_dist_coeff_lr[i](1),
                                calib_param.Camera.cv_dist_coeff_lr[i](2),
                                calib_param.Camera.cv_dist_coeff_lr[i](3))));
                Ncamera_ptr->addCamera(b_T_c_ptr,curcam,vio::cameras::NCameraSystem::RadialTangential,false);
            } else {
                LOG(FATAL) << "Dist model unsupported for cam";
            }
        }

    }


}


bool SolveMulticamPnP(const std::vector<std::tuple<int,int,cv::KeyPoint>> &cur_track,const std::map<int,Eigen::Vector3f> &cur_Mp_info,
        Eigen::Matrix4f & Twc_pnp)
{
    bool success = false;
    if(cur_Mp_info.size() == 0)
    {

    } else
    {
        int num_matches = 0;
        Eigen::Matrix2Xf measurements;
        vector<int> measurement_camera_indices;
        Eigen::Matrix3Xf G_landmark_positions;

        for (int i = 0; i < cur_track.size(); ++i)
        {
            int Mpidx = std::get<0>(cur_track[i]);
            if(cur_Mp_info.count(Mpidx))
            {
                num_matches++;
            }
        }
        measurements.resize(Eigen::NoChange, num_matches);
        measurement_camera_indices.resize(num_matches);
        G_landmark_positions.resize(Eigen::NoChange, num_matches);
        num_matches = 0;
        for (int i = 0; i < cur_track.size(); ++i)
        {
            int Mpidx = std::get<0>(cur_track[i]);
            if(cur_Mp_info.count(Mpidx))
            {
                cv::KeyPoint pt_ = std::get<2>(cur_track[i]);
                measurements.col(num_matches) = Eigen::Vector2f{pt_.pt.x,pt_.pt.y};
                measurement_camera_indices[num_matches] = std::get<1>(cur_track[i]);
                G_landmark_positions.col(num_matches) = cur_Mp_info.find(Mpidx)->second;
                num_matches++;
            }
        }


        geometric_vision::PnpPoseEstimator pose_estimator(
                FLAGS_lc_nonlinear_refinement_p3p, FLAGS_lc_use_random_pnp_seed);

        double inlier_ratio;
        int num_inliers;
        std::vector<double> inlier_distances_to_model;
        int num_iters;
        std::vector<int> inliers;
//        std::cout<<"初始位姿:"<<Twc_pnp<<std::endl;
        pose_estimator.absoluteMultiPoseRansacPinholeCam(
                measurements,measurement_camera_indices, G_landmark_positions,
                FLAGS_lc_ransac_pixel_sigma, FLAGS_lc_num_ransac_iters, Ncamera_ptr,
                Twc_pnp, &inliers, &inlier_distances_to_model, &num_iters);
        CHECK_EQ(inliers.size(), inlier_distances_to_model.size());
        num_inliers = static_cast<int>(inliers.size());
        inlier_ratio = static_cast<double>(num_inliers) /
                       static_cast<double>(G_landmark_positions.cols());
        if (inlier_ratio >= (FLAGS_lc_min_inlier_ratio) || num_inliers >= FLAGS_lc_min_inlier_count )
        {
            success = true;
//            std::cout<<"pnp后位姿:"<<Twc_pnp<<std::endl;
        } else
        {
            std::cout<<"fuck:?"<<std::endl;
        }
    }

    return success;
}

std::vector<std::pair<std::vector<XP::ImuData>, stereo_Img_info_f >> getMeasurements()
{
    std::vector<std::pair<std::vector<XP::ImuData>, stereo_Img_info_f >> measurements;

    while (true)
    {
        if(FLAGS_UseIMU)
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
        else
        {
            if (stereo_buf.empty())
                return measurements;

            stereo_Img_info_d stereo_info = stereo_buf.front();
            stereo_buf.pop();
            std::vector<XP::ImuData> IMUs;
            stereo_Img_info_f stereo_info_f;
            stereo_info_f.second = stereo_info.second;
            stereo_info_f.first = (float)(stereo_info.first -= offset_ts);
            measurements.emplace_back(IMUs, stereo_info_f);
        }
    }
    return measurements;
}

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
//            px += -7.48903e-02;//euroc的真值是TRM,TMb是的R是单位阵,所以我在这里直接加了t
//            py += 1.84772e-02;
//            pz += 1.20209e-01;

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


float compensatedParallax2(const std::vector<cv::Point2f> & pre_bearing_vec, const std::vector<cv::Point2f> & cur_bearing_vec)
{
    float Parallax = 0;
    float Parallax_num = 0;

    for(int i = 0; i < pre_bearing_vec.size(); ++i)
    {
        float ans = 0;
        Eigen::Vector3f p_j{cur_bearing_vec[i].x,cur_bearing_vec[i].y,1.0f};

        float u_j = p_j(0);
        float v_j = p_j(1);

        Eigen::Vector3f p_i{pre_bearing_vec[i].x,pre_bearing_vec[i].y,1.0f};
        Eigen::Vector3f p_i_comp;


        p_i_comp = p_i;
        float dep_i = p_i(2);
        float u_i = p_i(0) / dep_i;
        float v_i = p_i(1) / dep_i;
        float du = u_i - u_j, dv = v_i - v_j;

        float dep_i_comp = p_i_comp(2);
        float u_i_comp = p_i_comp(0) / dep_i_comp;
        float v_i_comp = p_i_comp(1) / dep_i_comp;
        float du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

        ans = std::max(ans, std::sqrt(std::min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));
        Parallax += ans;
        Parallax_num++;
    }

    return Parallax / float(Parallax_num);
}


void mono_begin_compute(std::vector<cv::KeyPoint> pre_key_pnts,std::vector<cv::KeyPoint> cur_key_pnts,
                        const cv::Matx33f& K,
                        const cv::Mat_<float>* dist_coeffs_ptr,
                        const bool fisheye,
                        const cv::Mat_<uchar> &  cam_mask,int *init_count)
{
    std::vector<cv::Point2f> pre_uv;
    std::vector<cv::Point2f> cur_uv;
    std::vector<cv::KeyPoint>::iterator prePt_iter = pre_key_pnts.begin();
    std::vector<cv::KeyPoint>::iterator curPt_iter = cur_key_pnts.begin();

    while(prePt_iter != pre_key_pnts.end() && curPt_iter != cur_key_pnts.end())
    {
        if(prePt_iter->class_id == curPt_iter->class_id)
        {
            pre_uv.push_back(prePt_iter->pt);
            cur_uv.push_back(curPt_iter->pt);
            prePt_iter++;curPt_iter++;
        }
        else if(prePt_iter->class_id < curPt_iter->class_id)
            prePt_iter++;
        else
            curPt_iter++;
    }

    assert(cur_uv.size() == pre_uv.size());
    double track_ratio = (double)cur_uv.size() / (double)pre_key_pnts.size();

    bool valid_track = track_ratio < 0.95 && track_ratio > 0.2;//也不能全追踪上

    std::vector<cv::Point2f> pre_feat_undistorted;
    std::vector<cv::Point2f> cur_feat_undistorted;
    if(pre_uv.empty())
    {
        (*init_count) = 0;
    } else
    {
        if(fisheye)
        {
            cv::Vec4f distortion_coeffs;
            for (int i = 0; i < distortion_coeffs.rows; ++i)
                distortion_coeffs[i] = (*dist_coeffs_ptr)(i);
            cv::fisheye::undistortPoints(pre_uv,//去畸变
                                         pre_feat_undistorted,
                                         K, distortion_coeffs);

            cv::fisheye::undistortPoints(cur_uv,//去畸变
                                         cur_feat_undistorted,
                                         K, distortion_coeffs);
        }
        else
        {
            cv::undistortPoints(pre_uv,//去畸变
                                pre_feat_undistorted,
                                K, *dist_coeffs_ptr);

            cv::undistortPoints(cur_uv,//去畸变
                                cur_feat_undistorted,
                                K, *dist_coeffs_ptr);
        }

        float focal_len = (K(0,0) + K(1,1)) / 2.0f;
        float aver_Parallax = compensatedParallax2(pre_feat_undistorted,cur_feat_undistorted) * focal_len;
//        std::cout<<aver_Parallax<<" "<<track_ratio<<std::endl;
        if(aver_Parallax > MIN_PARALLAX && valid_track)
            (*init_count) ++;
        else
        if((*init_count) > 0)
            (*init_count) --;
    }

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
    bool need_new_kf;
    if(FLAGS_stereo)
        need_new_kf = std::distance(kp_it_l, kps_l.end()) >= (kps_l.size()/3) || CF.feat_measures.size() < (kps_l.size()/3);
    else
        need_new_kf = std::distance(kp_it_l, kps_l.end()) >= (kps_l.size()/2) || CF.feat_measures.size() < (kps_l.size()/3);

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

void Get_Track(const IBA::CurrentFrame & CF,std::vector<std::tuple<int,int,cv::KeyPoint>> & cur_track)
{
    cur_track.clear();
    for (int i = 0; i < CF.feat_measures.size() ; ++i)
    {
        cv::KeyPoint cur_kp;
        cur_kp.pt.x = CF.feat_measures[i].x.x[0];
        cur_kp.pt.y = CF.feat_measures[i].x.x[1];
        cur_kp.class_id = CF.feat_measures[i].idx;
        if(CF.feat_measures[i].right)
            cur_track.push_back(std::make_tuple(CF.feat_measures[i].idx,1,cur_kp));
        else
            cur_track.push_back(std::make_tuple(CF.feat_measures[i].idx,0,cur_kp));
    }
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if(FLAGS_UseIMU)
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
                //设置一下第一个imu的时间
                if (!set_offset_time && !FLAGS_UseIMU) {
                    set_offset_time = true;
                    offset_ts = time;
                }

                m_syn_buf.lock();
                stereo_buf.push(std::make_pair(time,std::make_pair(image0,image1)));
                m_syn_buf.unlock();
                con.notify_one();
            }
        }
        else
        {
            cv::Mat image0;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.toSec();

                time = img0_buf.front()->header.stamp.toSec();
                image0 = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
            }

            m_buf.unlock();
            m_syn_buf.lock();
            stereo_buf.push(std::make_pair(time,std::make_pair(image0,image0)));//单目就都塞一张图吧
            m_syn_buf.unlock();
            con.notify_one();


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




#endif //ICE_BA_PARAMSINIT_H
