#!/bin/bash
# Set your own EuRoC_PATH path to run ice-ba. Use './bin/ice_ba --help' to get the explanation for all of the flags. Flags [imgs_folder] and [iba_param_path] are necessary.
# Add flag '--save_feature' to save feature message and calibration file for back-end only mode
#--log_dir /home/wya/ICE-BA-Debug

REST=$@

rosrun ice_ba_ros ice_ba_ros \
  --config_file=/home/wya/ICE-BA_ws/src/ICE-BA/config/euroc/euroc_stereo_imu_config.yaml \ $REST

