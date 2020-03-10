/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include "CameraPoseVisualization.h"
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "../Backend/IBA/IBA_datatype.h"

void registerPub(ros::NodeHandle &n);

void pubUpdatePointClouds(const std::vector<IBA::Point3D> & UpdatePointClouds,double t);

void pubLoopCamPose(const Eigen::Matrix4d &Twc0);

void pubGTCamPose(const Eigen::Matrix4d &Twc0,double t);

void pubTF(const Eigen::Matrix4d & Twc0, double t);


void pubTrackImage(const cv::Mat &cur_l_img, const cv::Mat &cur_r_img ,std::vector<cv::KeyPoint> &pre_l_kp,
                   std::vector<cv::KeyPoint> &cur_l_kp, std::vector<cv::KeyPoint> &cur_r_kp,double t);

void pubLatestCameraPose(const Eigen::Matrix4d & Twc0, const Eigen::Vector3d &V, double t);

void pubKFsPose(const std::vector<std::pair<double,Eigen::Matrix4d>> & kf_poses);
