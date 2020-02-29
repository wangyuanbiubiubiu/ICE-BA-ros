#!/bin/bash

# Set your own EuRoC_PATH path to run ice-ba. Use './bin/ice_ba --help' to get the explanation for all of the flags. Flags [imgs_folder] and [iba_param_path] are necessary.
# Add flag '--save_feature' to save feature message and calibration file for back-end only mode
#--log_dir /home/wya/ICE-BA-Debug
EuRoC_PATH=~/dataset/EuRoC

mkdir $EuRoC_PATH/result

cmd="../bin/ice_ba --imgs_folder $EuRoC_PATH/MH_05_difficult --v=0 --start_idx 0 --end_idx -1 --iba_param_path ../config/config_of_stereo.txt  --gba_camera_save_path $EuRoC_PATH/result/MH_05_difficult.txt --stereo --save_feature"
echo $cmd
eval $cmd
