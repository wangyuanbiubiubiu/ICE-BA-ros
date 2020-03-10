//
// Created by wya on 2020/3/5.
//

#ifndef ICE_BA_LOOPCLOSING_H
#define ICE_BA_LOOPCLOSING_H

#include <eigen3/Eigen/Dense>
#include <list>
#include <queue>
#include <mutex>
#include <thread>
#include <memory>
#include "../thirdparty/DBoW3/src/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/absolute_pose/NoncentralAbsoluteAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include "KeyFrame.h"
#include "FeatureMatcher.h"
#include "IBA_datatype.h"
#include "../Frontend/cameras/PinholeCamera.hpp"  // for vio::cameras::CameraBase
#include "../Frontend/cameras/EquidistantDistortion.hpp"
#include "../Frontend/cameras/RadialTangentialDistortion.hpp"
#include "../Frontend/cameras/RadialTangentialDistortion8.hpp"
#include <glog/logging.h>
#include <opencv2/video/tracking.hpp>

namespace LC
{

    /// 回环检测线程
    class LoopClosing
    {

    public:
        typedef std::function<void(const vector<Eigen::Matrix4f> & rKFpose,
                const Eigen::Matrix4f & lKFpose,vector<int> riFrm,int liFrm)> LoopCallback;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        LoopClosing(const string & voc_path);

        LoopClosing(const string & voc_path,
        const Eigen::Matrix3f& left_camK,
        const cv::Mat_<float>& left_cv_dist_coeff,
        const bool fisheye,
        const cv::Mat_<uchar>& left_mask);

        ~LoopClosing(){ finished = true;}

        void UpdateKfInfo(const IBA::Global_Map & UpdateGM);

        void InsertKeyFrame(shared_ptr<KeyFrame> kf);

        bool CheckNewKeyFrames();

        // Main function
        void Run();

        vector<shared_ptr<KeyFrame>> GetConnectedKeyFrames(int iFrm);

        bool DetectLoop(shared_ptr<KeyFrame> &kf);

        bool CorrectLoop(Eigen::Matrix4f & pnp_Twc);

        bool ComputeOptimizedPose(Eigen::Matrix4f & PnP_Twc);

        void SetCallback(const LoopCallback & loop_callback);

    protected:
        map<DBoW3::EntryId, shared_ptr<KeyFrame>> checkedKFs;    // keyframes that are recorded.

        int maxKFId = 0;
        std::shared_ptr<DBoW3::Database> kfDB = nullptr;
        std::shared_ptr<DBoW3::Vocabulary> voc = nullptr;
        shared_ptr<KeyFrame> candidateKF = nullptr;
        vector<shared_ptr<KeyFrame>> candidate_co_KFs;
        std::vector<list<shared_ptr<KeyFrame>> > mvInvertedFile;
        // loop kf queue
        deque<shared_ptr<KeyFrame>> KFqueue;
        vector<shared_ptr<KeyFrame>> allKF;
        shared_ptr<KeyFrame> mCurrentKF = nullptr;


        std::vector<shared_ptr<MapPoint>> All_Mps; //所有的地图点
        std::vector<int> Gid_2_idx; //[G_id] = 在All_Mps中的索引

        bool finished = false;

        vector<shared_ptr<KF_info>> All_KF_Info;
        std::map<int,int> iFrm_2_idx;//[iFrm] = 在All_KF_Info中的索引

        //左相机
        std::shared_ptr<vio::cameras::CameraBase> cam_left_;

        mutex mutex_GM;
        mutex mutexKFQueue;
        thread mainLoop;
        int kfGap = 20;
        float historyminScore = 0.0002;
        float minScoreAccept = 0.06;

        LoopCallback m_callback;
    };
}

#endif //ICE_BA_LOOPCLOSING_H
