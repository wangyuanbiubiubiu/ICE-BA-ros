# ICE-BA
## ICE-BA-ros  
If you want to use the ros version, set(USE_ROS true)   #true or false  
stereo-vio:   
![rviz](https://github.com/wangyuanbiubiubiu/ICE-BA-ros/blob/master/config/ice-ba.png)  
mono-vio: (note: If you want to use a mono-vio, you need to give the camera enough movement to get started ,see`mono_begin_compute` And I also added the initialization version of vinsmono, but I needed to rely on Ceres for SFM BA.git checkout use_vinsmono_init)   
![rviz](https://github.com/wangyuanbiubiubiu/ICE-BA-ros/blob/master/config/iceba-mono.png)
## How to build
    export ROS_VERSION=kinetic
    export CATKIN_WS=~/ICE-BA_ws
    mkdir -p $CATKIN_WS/src
    cd $CATKIN_WS
    catkin init
    catkin config --merge-devel
    catkin config --extend /opt/ros/$ROS_VERSION
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
    cd src
    git clone https://github.com/wangyuanbiubiubiu/ICE-BA-ros.git
    cd ICE-BA-ros #git checkout use_vinsmono_init ，if you want to use vinsmono init
    bash build_thirdparty.sh #opengv use to compute PNP（Muticam pnp）
    cd $CATKIN_WS
    catkin build ice_ba_ros
## How to run
    cd $CATKIN_WS
    source devel/setup.bash
    roslaunch ice_ba_ros ice_ba_rviz.launch 
    ./src/ICE-BA-ros/scripts/run_ice_ba_ros.sh
    rosbag play --pause MH_05_difficult.bag 
## ICE-BA 的中文注释  
基本全都注释了,推荐先看LBA流程,再看滑窗边缘化老帧的思路，以及滑窗是如何将先验给到GBA和LBA的,最后看GBA的部分  

## ICE-BA: Incremental, Consistent and Efficient Bundle Adjustment for Visual-Inertial SLAM  
We present ICE-BA, an incremental, consistent and efficient bundle adjustment for visual-inertial SLAM, which takes feature tracks, IMU measurements and optionally the loop constraints as input, performs in parallel both local BA over the sliding window and global BA over all keyframes, and outputs camera pose and updated map points for each frame in real-time. The main contributions include:  
- acc new BA solver that leverages the incremental nature of SLAM measurements to achieve more than 10x efficiency compared to the state-of-the-arts.
- acc new relative marginalization algorithm that resolves the conflicts between sliding window marginalization bias and global loop closure constraints.

Beside the backend solver, the library also provides an optic flow based frontend, which can be easily replaced by other more complicated frontends like ORB-SLAM2.  

The original implementation of our ICE-BA is at https://github.com/ZJUCVG/EIBA, which only performs global BA and does not support IMU input.  

**Authors:** Haomin Liu, Mingyu Chen, Yingze Bao, Zhihao Wang  
**Related Publications:**  
Haomin Liu, Mingyu Chen, Guofeng Zhang, Hujun Bao and Yingze Bao. ICE-BA: Incremental, Consistent and Efficient Bundle Adjustment for
Visual-Inertial SLAM. (Accepted by CVPR 2018).**[PDF](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_ICE-BA_Incremental_Consistent_CVPR_2018_paper.pdf)**.  
Haomin Liu, Chen Li, Guojun Chen, Guofeng Zhang, Michael Kaess and Hujun Bao. Robust Keyframe-based Dense SLAM with an RGB-D Camera [J]. arXiv preprint arXiv:1711.05166, 2017. [arXiv report].**[PDF](https://arxiv.org/abs/1711.05166)**.  


## 1. License
Licensed under the Apache License, Version 2.0.  
Refer to LISENCE for more details.

## 2. Prerequisites
We have tested the library in **Ubuntu 14.04** and **Ubuntu 16.04**.  
The following dependencies are needed:
### boost
sudo apt-get install libboost-dev libboost-thread-dev libboost-filesystem-dev 

### Eigen
sudo apt-get install libeigen3-dev

### Glog
https://github.com/google/glog

### Gflags
https://github.com/gflags/gflags

### OpenCV
We use OpenCV 3.0.0.  
https://opencv.org/

### Yaml
https://github.com/jbeder/yaml-cpp

### brisk
https://github.com/gwli/brisk

## 3. Build
cd ice-ba  
chmod +x build.sh  
./build.sh

## 4. Run
We provide examples to run ice-ba with [EuRoC dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets#downloads).  

### run ICE-BA stereo
Run ICE-BA in stereo mode. Please refer to scripts/run_ice_ba_stereo.sh for more details about how to run the example.  

### run ICE-BA monocular
Run ICE-BA in monocular mode. Please refer to scripts/run_ice_ba_mono.sh for more details about how to run the example.  

### run back-end only
Front-end results can be saved into files. Back-end only mode loads these files and runs backend only.  
Please refer to scripts/run_backend_only.sh for more details about how to run the example.  

## 5. Contribution
You are very welcome to contribute to ICE-BA.
Baidu requires the contributors to e-sign [CLA (Contributor License Agreement)](https://gist.github.com/tanzhongyibidu/6605bdef5f7bb03b9084dd8fed027037) before making acc Pull Request.  We have the CLA binding to Github so it will pop up before creating acc PR.

