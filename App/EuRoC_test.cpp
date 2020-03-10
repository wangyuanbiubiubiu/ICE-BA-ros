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

#include <algorithm>
#include <string>
#include <fstream>
#include <vector>

#include "LoopClosing.h"



namespace fs = boost::filesystem;
using std::string;
using std::vector;
DEFINE_string(imgs_folder, "", "The folder containing l and r folders, and the calib.yaml");
DEFINE_int32(grid_row_num, 1, "Number of rows of detection grids");
DEFINE_int32(grid_col_num, 1, "Number of cols of detection grids");
DEFINE_int32(max_num_per_grid, 70, "Max number of points per grid");
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
DEFINE_string(gba_camera_save_path, "", "Save the camera states to when finished");
DEFINE_bool(stereo, false, "monocular or stereo mode");
DEFINE_bool(save_feature, false, "Save features to .dat file");

std::unique_ptr<LC::LoopClosing> LoopCloser_ptr;
std::queue<IBA::RelativeConstraint> loop_info;
std::mutex m_loop_buf;

//输入数据集所在位置
//输出左右相机各自的图片名称
size_t load_image_data(const string& image_folder,
                       std::vector<string> &limg_name,
                       std::vector<string> &rimg_name) {
  LOG(INFO) << "Loading " << image_folder;
  //时间戳和图片的对应
  std::string l_path = image_folder + "/mav0/cam0/data.csv";
  std::string r_path = image_folder + "/mav0/cam1/data.csv";
  //右相机图片所在位置
  std::string r_img_prefix = image_folder + "/mav0/cam1/data/";
  std::ifstream limg_file(l_path);
  std::ifstream rimg_file(r_path);
  if (!limg_file.is_open() || !rimg_file.is_open()) {
    LOG(WARNING) << image_folder << " cannot be opened";
    return 0;
  }
  std::string line;
  std::string time;
  while (getline(limg_file,line)) {
    if (line[0] == '#')
      continue;
    std::istringstream is(line);
    int i = 0;
    while (getline(is, time, ',')){
      bool is_exist = boost::filesystem::exists(r_img_prefix + time + ".png");
      if (i == 0 && is_exist){
        limg_name.push_back(time + ".png");
        rimg_name.push_back(time + ".png");
      }
      i++;
    }
  }
  limg_file.close();
  rimg_file.close();
  LOG(INFO)<< "loaded " << limg_name.size() << " images";
  return limg_name.size();
}

//读取imu数据
size_t load_imu_data(const string& imu_file_str,
                     std::list<XP::ImuData>* imu_samples_ptr,
                     uint64_t &offset_ts_ns) {
  CHECK(imu_samples_ptr != NULL);
  LOG(INFO) << "Loading " << imu_file_str;
  std::ifstream imu_file(imu_file_str.c_str());
  if (!imu_file.is_open()) {
    LOG(WARNING) << imu_file_str << " cannot be opened";
    return 0;
  }
  std::list<XP::ImuData>& imu_samples = *imu_samples_ptr;
  imu_samples.clear();
  // read imu data
  std::string line;
  std::string item;
  double c[6];
  uint64_t t;
  bool set_offset_time = false;
  while (getline(imu_file,line))
  {
    if (line[0] == '#')
      continue;
    std::istringstream is(line);
    int i = 0;
    while (getline(is, item, ',')) {
      std::stringstream ss;
      ss << item;
      if (i == 0)
        ss >> t;
      else
        ss >> c[i-1];
      i++;
    }
    //设置一下第一个的时间
    if (!set_offset_time) {
      set_offset_time = true;
      offset_ts_ns = t;
    }
    XP::ImuData imu_sample;
    float _t_100us = (t - offset_ts_ns)/1e5;
    imu_sample.time_stamp = _t_100us/1e4;

    imu_sample.ang_v(0) = c[0];
    imu_sample.ang_v(1) = c[1];
    imu_sample.ang_v(2) = c[2];
    imu_sample.accel(0) = c[3];
    imu_sample.accel(1) = c[4];
    imu_sample.accel(2) = c[5];

    VLOG(3) << "accel " << imu_sample.accel.transpose()
            << " gyro " << imu_sample.ang_v.transpose();
    imu_samples.push_back(imu_sample);
  }
  imu_file.close();
  LOG(INFO)<< "loaded " << imu_samples.size() << " imu samples";
  return imu_samples.size();
}
//读取相机参数
//输入数据集的根目录
//输出标定参数（内外参,立体矫正,去完畸变后的映射）
void load_asl_calib(const std::string &asl_path,
                    XP::DuoCalibParam &calib_param) {
  std::string cam0_yaml = asl_path + "/mav0/cam0/sensor.yaml";
  std::string cam1_yaml = asl_path + "/mav0/cam1/sensor.yaml";
  std::string imu0_yaml = asl_path + "/mav0/imu0/sensor.yaml";
  //左右相机,imu的yaml节点
  YAML::Node cam0_calib = YAML::LoadFile(cam0_yaml);
  YAML::Node cam1_calib = YAML::LoadFile(cam1_yaml);
  YAML::Node imu0_calib = YAML::LoadFile(imu0_yaml);
  // intrinsics
  //左相机的内参
  std::vector<float> v_float = cam0_calib["intrinsics"].as<std::vector<float>>();
  //左相机的和右相机的内参K矩阵
  calib_param.Camera.cv_camK_lr[0] << v_float[0], 0, v_float[2],
      0, v_float[1], v_float[3],
      0, 0, 1;
  calib_param.Camera.cameraK_lr[0] << v_float[0], 0, v_float[2],
      0, v_float[1], v_float[3],
      0, 0, 1;
  v_float = cam1_calib["intrinsics"].as<std::vector<float>>();
  calib_param.Camera.cv_camK_lr[1] << v_float[0], 0, v_float[2],
      0, v_float[1], v_float[3],
      0, 0, 1;
  calib_param.Camera.cameraK_lr[1] << v_float[0], 0, v_float[2],
      0, v_float[1], v_float[3],
      0, 0, 1;
  // distortion_coefficients
  //左右相机畸变参数,不过这里默认是radtan畸变模型了,equi模型呢?
  std::vector<double> v_double = cam0_calib["distortion_coefficients"].as<std::vector<double>>();
  calib_param.Camera.cv_dist_coeff_lr[0] = (cv::Mat_<float>(8, 1) << static_cast<float>(v_double[0]), static_cast<float>(v_double[1]),
      static_cast<float>(v_double[2]), static_cast<float>(v_double[3]), 0.0, 0.0, 0.0, 0.0);
  v_double = cam1_calib["distortion_coefficients"].as<std::vector<double>>();
  calib_param.Camera.cv_dist_coeff_lr[1] = (cv::Mat_<float>(8, 1) << static_cast<float>(v_double[0]), static_cast<float>(v_double[1]),
      static_cast<float>(v_double[2]), static_cast<float>(v_double[3]), 0.0, 0.0, 0.0, 0.0);
  //TBS
  //左右相机,imu到本体坐标系的外参,本体系在euroc中固连于imu上
  //Tbc0,Tbc1
  v_double = cam0_calib["T_BS"]["data"].as<std::vector<double>>();
  Eigen::Matrix4d b_t_c0 = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(&v_double[0]);
  v_double = cam1_calib["T_BS"]["data"].as<std::vector<double>>();
  Eigen::Matrix4d b_t_c1 = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(&v_double[0]);
  v_double = imu0_calib["T_BS"]["data"].as<std::vector<double>>();
  Eigen::Matrix4d b_t_i = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(&v_double[0]);
  // ASL {B}ody frame is the IMU
  // {D}evice frame is the left camera
  //百度这个代码的表示方式：d_t_cam0就是Tdcam0,即相机0到设备坐标系的变换,这里把设备坐标系固连于左相机坐标系
  Eigen::Matrix4d d_t_cam0 = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d d_t_b = d_t_cam0 * b_t_c0.inverse();// Tdb = Tdc0 * Tc0b本体坐标系到设备坐标系
  Eigen::Matrix4d d_t_cam1 = d_t_b * b_t_c1;//Tdc1 右相机到设备坐标系
  Eigen::Matrix4d d_t_imu = d_t_b * b_t_i; //imu坐标系到设备坐标系
  //设置关于设备坐标系的参数
  calib_param.Camera.D_T_C_lr[0] = Eigen::Matrix4f::Identity();
  calib_param.Camera.D_T_C_lr[1] = d_t_cam1.cast<float>();
  // Image size 图像大小
  std::vector<int> v_int = cam0_calib["resolution"].as<std::vector<int>>();
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
//  calib_param.initUndistortMap(calib_param.Camera.img_size);
}

float get_timestamp_from_img_name(const string& img_name,
                                  uint64_t offset_ns) {
  string ts_ns_string = fs::path(img_name).stem().string();
  int64_t offset_t = boost::lexical_cast<uint64_t>(ts_ns_string) - offset_ns;
  int64_t t = offset_t/1e5;
  return static_cast<float>(t)/1e4;
}

bool convert_to_asl_timestamp(const string& file_in,
                              const string& file_out,
                              uint64_t offset_ns) {
  FILE *fp_in = fopen(file_in.c_str(), "r");
  FILE *fp_out = fopen(file_out.c_str(), "w");
  if (!fp_in || !fp_out) {
    LOG(ERROR) << "convert to asl timestamp error";
    return false;
  }
  float t;
  float x, y, z;
  float qx, qy, qz, qw;
  while (fscanf(fp_in, "%f %f %f %f %f %f %f %f", &t, &x, &y, &z, &qx, &qy, &qz, &qw) == 8) {
    double t_s = t + static_cast<double>(offset_ns*1e-9);
    fprintf(fp_out, "%lf %f %f %f %f %f %f %f\n", t_s, x, y, z, qx, qy, qz, qw);
  }


//    while (fscanf(fp_in, "%f %f %f %f %f %f %f %f", &t, &x, &y, &z, &qx, &qy, &qz, &qw) == 8) {
//    double t_s = t + static_cast<double>(offset_ns*1e-9);
//    fprintf(fp_out, "%f %f %f\n", x, y, z);
//  }
  fclose(fp_in);
  fclose(fp_out);
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
//  std::cout<<"asd"<<std::distance(kp_it_l, kps_l.end())<<" afsa:"<<CF.feat_measures.size()<<std::endl;
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

//euroc测试代码
int main(int argc, char** argv) {
    //初始化glog相关程序
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InstallFailureSignalHandler();
    //如果没有给image文件夹的话
    if (FLAGS_imgs_folder.empty())//
    {
        google::ShowUsageWithFlags(argv[0]);
        return -1;
    }

  vector<string> img_file_paths;//左相机的图片全局路径
  vector<string> slave_img_file_paths;//右相机的图片全局路径
  vector<string> iba_dat_file_paths;//？dat文件的保存文件名称




  constexpr int reserve_num = 5000;
  img_file_paths.reserve(reserve_num);
  slave_img_file_paths.reserve(reserve_num);
  //数据集存放的位置
  fs::path p(FLAGS_imgs_folder + "/mav0/cam0");
  if (!fs::is_directory(p)) {
    LOG(ERROR) << p << " is not acc directory";
    return -1;
  }
  //暂时还不知道这个dat文件是干啥用的
  //但是会建立一个存储这个的文件夹
  if (!fs::is_directory(FLAGS_imgs_folder + "/dat")) {
    fs::create_directories(FLAGS_imgs_folder + "/dat");
  }
  vector<string> limg_name, rimg_name;
  //得到图片名称
  load_image_data(FLAGS_imgs_folder, limg_name, rimg_name);
  for (int i=0; i<limg_name.size(); i++)
  {
    string l_png = p.string() + "/data/" + limg_name[i];
    img_file_paths.push_back(l_png);
    slave_img_file_paths.push_back(FLAGS_imgs_folder + "/mav0/cam1/data/" + rimg_name[i]);
    iba_dat_file_paths.push_back(FLAGS_imgs_folder + "/dat/" + limg_name[i] + ".dat");
  }
  //如果只运行单目的话
  if (!FLAGS_stereo) {
    slave_img_file_paths.clear();
  }

  if (img_file_paths.size() == 0) {
    LOG(ERROR) << "No image files for detection";
    return -1;
  }
  XP::DuoCalibParam duo_calib_param;
  try {//输出标定参数（内外参,立体矫正,去完畸变后的映射）
    load_asl_calib(FLAGS_imgs_folder, duo_calib_param);
  } catch (...){
    LOG(ERROR) << "Load calibration file error";
    return -1;
  }
  // Create masks based on FOVs computed from intrinsics
  std::vector<cv::Mat_<uchar> > masks(2);
  for (int lr = 0; lr < 2; ++lr) {
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

    LoopCloser_ptr.reset(new LC::LoopClosing("/home/wya/ICE-BA_ws/src/ICE-BA/vocab/orbvoc.dbow3",
                                             duo_calib_param.Camera.cameraK_lr[0],  // ref_camK 左相机内参
                                             duo_calib_param.Camera.cv_dist_coeff_lr[0],  // ref_dist_coeff 左相机畸变
                                             duo_calib_param.Camera.fishEye,
                                             masks[0]));



  // Load IMU samples to predict OF point locations
  std::list<XP::ImuData> imu_samples;
  std::string imu_file = FLAGS_imgs_folder + "/mav0/imu0/data.csv";
  uint64_t offset_ts_ns;
  //读取所有imu数据(时间戳,加速度,陀螺仪数据),第一个imu时间戳为设备启始时间,用陀螺仪来估计旋转
  if (load_imu_data(imu_file, &imu_samples, offset_ts_ns) > 0) {
    std::cout << "Load imu data. Enable OF prediciton with gyro\n";
  } else {
    std::cout << "Cannot load imu data.\n";
    return -1;
  }
  // Adjust end image index for detection
  if (FLAGS_end_idx < 0 || FLAGS_end_idx > img_file_paths.size()) {
    FLAGS_end_idx = img_file_paths.size();//图片停止时的索引
  }

  FLAGS_start_idx = std::max(0, FLAGS_start_idx);//图片开始时的索引
  // remove all frames before the first IMU data
  // offset_ts_ns是imu第一帧的原始时间戳,开始的id一定要在有了imu数据时才可以
  while (FLAGS_start_idx < FLAGS_end_idx && get_timestamp_from_img_name(img_file_paths[FLAGS_start_idx], offset_ts_ns) <= imu_samples.front().time_stamp)
    FLAGS_start_idx++;

  //初始化特征提取器
  XP::FeatureTrackDetector feat_track_detector(FLAGS_ft_len/*图像大小*/,
                                               FLAGS_ft_droprate/*图像大小*/,
                                               !FLAGS_not_use_fast/*是否用fast点*/,
                                               FLAGS_uniform_radius/*图像大小*/,
                                               duo_calib_param.Camera.img_size/*图像大小*/);

  //配置左右相机的投影和畸变模型,目前只支持针孔+radtan
  XP::ImgFeaturePropagator slave_img_feat_propagator(
      duo_calib_param.Camera.cameraK_lr[1],  // cur_camK 右相机内参
      duo_calib_param.Camera.cameraK_lr[0],  // ref_camK 左相机内参
      duo_calib_param.Camera.cv_dist_coeff_lr[1],  // cur_dist_coeff 右相机畸变
      duo_calib_param.Camera.cv_dist_coeff_lr[0],  // ref_dist_coeff 左相机畸变
      duo_calib_param.Camera.fishEye,
      masks[1],//右相机的掩码
      FLAGS_pyra_level,//所有金字塔层数
      FLAGS_min_feature_distance_over_baseline_ratio,//特征点最小深度比例,用于极线搜索
      FLAGS_max_feature_distance_over_baseline_ratio);//特征点最大深度比例,用于极线搜索




  //双目外参
  const Eigen::Matrix4f T_Cl_Cr =
      duo_calib_param.Camera.D_T_C_lr[0].inverse() * duo_calib_param.Camera.D_T_C_lr[1];
  //绘图器
  XP::PoseViewer pose_viewer;
    //绘制前清除画布
  pose_viewer.set_clear_canvas_before_draw(true);

  //求解器初始化
  IBA::Solver solver;
  Eigen::Vector3f last_position = Eigen::Vector3f::Zero();
  float travel_dist = 0.f;
  if (FLAGS_save_feature) {
      //存储外参,内参文件
    IBA::SaveCalibration(FLAGS_imgs_folder + "/calibration.dat", to_iba_calibration(duo_calib_param));
  }

    //左右目的内外参,,是否要输出细节内容,是否输出debug信息,,参数所在位置
  solver.Create(to_iba_calibration(duo_calib_param),
                257,
                IBA_VERBOSE_NONE,
                IBA_DEBUG_NONE,
                257,
                FLAGS_iba_param_path,//iba的配置文件
                "" /* iba directory */);

    //对LBA求解器设置回调函数m_callback,用来可视化
    solver.SetCallbackGBA([&](const int iFrm,/*最新一帧的id*/ const float ts/*最新一帧的时间戳*/)
      {
          IBA::Global_Map GM;
          solver.GetUpdateGba(&GM);
          LoopCloser_ptr->UpdateKfInfo(GM);
      });


    LoopCloser_ptr->SetCallback([&](const vector<Eigen::Matrix4f> & rKFpose/*参考关键帧Twc*/,
                                    const Eigen::Matrix4f & lKFpose,vector<int> riFrm,int liFrm)
    {
        for (int k = 0; k < rKFpose.size(); ++k)
        {
//            std::cout<<rKFpose[k]<<std::endl;
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


  //对LBA求解器设置回调函数m_callback,用来可视化
  solver.SetCallbackLBA([&](const int iFrm,/*最新一帧的id*/ const float ts/*最新一帧的时间戳*/)
  {
#ifndef __DUO_VIO_TRACKER_NO_DEBUG__
    VLOG(1) << "===== start ibaCallback at ts = " << ts;
#endif
    // as we may be able to send out information directly in the callback arguments
    IBA::SlidingWindow sliding_window;
    //获得LBA中的滑窗中更新了的普通帧以及更新了的关键帧还有更新了的地图点
    solver.GetSlidingWindow(&sliding_window);
    const IBA::CameraIMUState& X = sliding_window.CsLF.back();//最新的一帧
    const IBA::CameraPose& C = X.Cam_pose;
    Eigen::Matrix4f W_vio_T_S = Eigen::Matrix4f::Identity();  // W_vio_T_S
    for (int i = 0; i < 3; ++i) {
      W_vio_T_S(i, 3) = C.p[i];
      for (int j = 0; j < 3; ++j) {
        W_vio_T_S(i, j) = C.R[j][i];  //因为存储的是C.R里是Rc0w,所以要转成Rwc0     Cam_state.R is actually R_SW
      }
    }

    Eigen::Matrix<float, 9, 1> speed_and_biases;
    for (int i = 0; i < 3; ++i) {
      speed_and_biases(i) = X.v[i];
      speed_and_biases(i + 3) = X.ba[i];
      speed_and_biases(i + 6) = X.bw[i];
    }

      Eigen::Vector3f cur_position = W_vio_T_S.topRightCorner(3, 1);
    travel_dist += (cur_position - last_position).norm();
    last_position = cur_position;
    pose_viewer.addPose(W_vio_T_S, speed_and_biases, travel_dist);
  });


    //启动求解器
    solver.Start();

  float prev_time_stamp = 0.0f;
  // load previous image//左目之前的特征点和描述子
  std::vector<cv::KeyPoint> pre_image_key_points;
  cv::Mat pre_image_features;
  for (int it_img = FLAGS_start_idx; it_img < FLAGS_end_idx; ++it_img)//开始处理所有图片
  {
    VLOG(0) << " start detection at ts = " << fs::path(img_file_paths[it_img]).stem().string();
    auto read_img_start = std::chrono::high_resolution_clock::now();
    cv::Mat img_in_raw;
    img_in_raw = cv::imread(img_file_paths[it_img], CV_LOAD_IMAGE_GRAYSCALE);//读取左相机图片
    CHECK_EQ(img_in_raw.rows, duo_calib_param.Camera.img_size.height);
    CHECK_EQ(img_in_raw.cols, duo_calib_param.Camera.img_size.width);
    cv::Mat img_in_smooth;
    cv::blur(img_in_raw, img_in_smooth, cv::Size(3, 3));//图像去噪
    if (img_in_smooth.rows == 0) {
      LOG(ERROR) << "Cannot load " << img_file_paths[it_img];
      return -1;
    }
    // get timestamp from image file name (s)
      //这里用的时间戳都是相对于imu第一帧的时间戳,time_stamp为左图像时间戳
    const float time_stamp = get_timestamp_from_img_name(img_file_paths[it_img], offset_ts_ns);
    std::vector<cv::KeyPoint> key_pnts;//左目提取到的特征点
    cv::Mat orb_feat;//左目orb描述子
    cv::Mat pre_img_in_smooth;
    // load slave image
    cv::Mat slave_img_smooth;  // for visualization later
    std::vector<cv::KeyPoint> key_pnts_slave;
    cv::Mat orb_feat_slave;
    std::vector<XP::ImuData> imu_meas;//两帧图像之间imu的原始测量数据

    // Get the imu measurements within prev_img_time_stamp and time_stamp to compute old_R_new
    //获取prev_time_stamp和time_stamp中的imu测量值，以计算old_R_new
    imu_meas.reserve(10);
    for (auto it_imu = imu_samples.begin(); it_imu != imu_samples.end(); )
    {
      if (it_imu->time_stamp < time_stamp) {
        imu_meas.push_back(*it_imu);
        it_imu++;
        imu_samples.pop_front();
      } else {
        break;
      }
    }
    VLOG(1) << "imu_meas size = " << imu_meas.size();
    VLOG(1) << "img ts prev -> curr " << prev_time_stamp << " -> " << time_stamp;
    if (imu_meas.size() > 0) {
      VLOG(1) << "imu ts prev -> curr " << imu_meas.front().time_stamp
              << " -> " << imu_meas.back().time_stamp;
    }

    if (!slave_img_file_paths.empty()) //也就是双目的情况
    {//读取右目相机,一样做降噪处理
      if (!slave_img_file_paths[it_img].empty()) {
        cv::Mat slave_img_in;
        slave_img_in = cv::imread(slave_img_file_paths[it_img], CV_LOAD_IMAGE_GRAYSCALE);
        cv::blur(slave_img_in, slave_img_smooth, cv::Size(3, 3));
      }
    }
    // use optical flow  from the 1st frame
    if (it_img != FLAGS_start_idx)//第二帧之后用光流进行左目前后帧的追踪
    {
      CHECK(it_img >= 1);
      VLOG(1) << "pre_image_key_points.size(): " << pre_image_key_points.size();
      const int request_feat_num = FLAGS_max_num_per_grid * FLAGS_grid_row_num * FLAGS_grid_col_num;//最大提取的特征点数量
      feat_track_detector.build_img_pyramids(img_in_smooth,
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
        feat_track_detector.optical_flow_and_detect(masks[0]/*左相机掩码*/,
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
        feat_track_detector.optical_flow_and_detect(masks[0],
                                                    pre_image_features,
                                                    pre_image_key_points,
                                                    request_feat_num,
                                                    FLAGS_pyra_level,
                                                    FLAGS_fast_thresh,
                                                    &key_pnts,
                                                    nullptr);
      }
      feat_track_detector.update_img_pyramids();//更新金字塔buffer
      VLOG(1) << "after OF key_pnts.size(): " << key_pnts.size() << " requested # "
              << FLAGS_max_num_per_grid * FLAGS_grid_row_num * FLAGS_grid_col_num;
    } else
        {//第一帧
      // first frame
      //第一帧左相机提取特征点
      //输入左相机图片,图像掩码,最大提取的特征点数量,金字塔层数,fast点阈值,特征点,描述子
      //主要就是提点,计算描述子
      feat_track_detector.detect(img_in_smooth,
                                 masks[0],
                                 FLAGS_max_num_per_grid * FLAGS_grid_row_num * FLAGS_grid_col_num,
                                 FLAGS_pyra_level,
                                 FLAGS_fast_thresh,
                                 &key_pnts,
                                 nullptr);
      feat_track_detector.build_img_pyramids(img_in_smooth,//构建图像金字塔,存到前一帧缓存器中
                                             XP::FeatureTrackDetector::BUILD_TO_PREV);
    }

    //如果是双目的话
    if (slave_img_smooth.rows > 0)
    {
      CHECK(orb_feat_slave.empty());
      auto det_slave_img_start = std::chrono::high_resolution_clock::now();
      //输入右左目图片，左目提取的特征点,左右目外参,右目提取的特征点,描述子,是否要输出debug信息
      //用svo的块匹配的思路算出右相机中特征点的坐标,并且计算描述子
      slave_img_feat_propagator.PropagateFeatures(slave_img_smooth,  // cur 右目
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
    create_iba_frame(key_pnts, key_pnts_slave, imu_meas, time_stamp, &CF, &KF);
    //将当前帧以及关键帧(如果有的话)放进求解器
    if(KF.iFrm != -1)
    {

//        std::vector<cv::KeyPoint> loop_key_pnts;//左目提取到的特征点
//        cv::Mat loop_orb_feat;//左目orb描述子
//
//        feat_track_detector.detect_for_loop(img_in_smooth,
//                                        masks[0],
//                                        1000,
//                                        FLAGS_pyra_level,
//                                        FLAGS_fast_thresh,
//                                        &loop_key_pnts,
//                                        &loop_orb_feat);
//
//
        cv::Mat cur_orb_feat;
        feat_track_detector.ComputeDescriptors(img_in_smooth,&key_pnts,&cur_orb_feat);

//        std::shared_ptr<LC::KeyFrame> (new LC::KeyFrame(KF.iFrm,key_pnts,cur_orb_feat,loop_key_pnts,loop_orb_feat,img_in_smooth));
        LoopCloser_ptr ->InsertKeyFrame(std::shared_ptr<LC::KeyFrame> (new LC::KeyFrame(KF.iFrm,key_pnts,cur_orb_feat,key_pnts,cur_orb_feat)));
    }
    solver.PushCurrentFrame(CF, KF.iFrm == -1 ? nullptr : &KF);//先说明一下,我习惯的求解增量的表达是Hx=b,

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



      // 但是它里面的表达是Hx=-b,所以我注释的时候b都是标的-b,反正就记住是反的就好了
    if (FLAGS_save_feature) {
      IBA::SaveCurrentFrame(iba_dat_file_paths[it_img], CF, KF);
    }
    pre_image_key_points = key_pnts;
    pre_image_features = orb_feat.clone();
    // show pose
    pose_viewer.displayTo("trajectory");
    cv::waitKey(1);
    prev_time_stamp = time_stamp;
  }
    std::string temp_file = "/tmp/" + std::to_string(offset_ts_ns) + ".txt";
    solver.SaveCamerasGBA(temp_file, false /* append */, true /* pose only */);
    solver.Stop();
    solver.Destroy();

  // for comparsion with asl groundtruth
  convert_to_asl_timestamp(temp_file, FLAGS_gba_camera_save_path, offset_ts_ns);
  return 0;
}
