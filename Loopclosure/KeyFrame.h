//
// Created by wya on 2020/3/5.
//

#ifndef ICE_BA_KEYFRAME_H
#define ICE_BA_KEYFRAME_H

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
#include <memory>

namespace LC
{
    struct MapPoint
    {
        Eigen::Vector3f Pw;
        int G_id;//这个是前端的地图点的id
        bool valid = false;
    };

    struct KF_info
    {
        std::vector<int>  CovisibleKFs;
        Eigen::Matrix4f Twc;
        int iFrm;//这个是前端的地图点的id
        bool valid = false;
    };

    using namespace cv;
    using namespace std;
    class KeyFrame
    {

        public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW


        KeyFrame(int iFrm_,std::vector<cv::KeyPoint> & kps, cv::Mat & Descriptors,
                 std::vector<cv::KeyPoint> & loop_kps, cv::Mat & loop_Descriptors, cv::Mat & img);

        KeyFrame(int iFrm_,std::vector<cv::KeyPoint> & kps, cv::Mat & Descriptors,
                 std::vector<cv::KeyPoint> & loop_kps, cv::Mat & loop_Descriptors);

        std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

        void ComputeBoW(std::shared_ptr<DBoW3::Vocabulary> voc);


        vector<shared_ptr<MapPoint>> GetMapPointMatches();
        int id;
        int iFrm;//全局帧的id
        cv::Mat mDescriptors;
        std::vector<cv::KeyPoint> mKPs;
        //BoW
        DBoW3::BowVector mBowVec; //
        DBoW3::FeatureVector mFeatVec; //

        cv::Mat mloop_Descriptors;
        std::vector<cv::KeyPoint> mloop_KPs;
        //BoW
        DBoW3::BowVector mloop_BowVec; //
        DBoW3::FeatureVector mloop_FeatVec; //

        mutex mMutexFeatures;

        vector<shared_ptr<MapPoint>> mvpMapPoints;

        cv::Mat mImg; //only debug
    protected:
        static unsigned long nextId;  // next id
    };
}

#endif //ICE_BA_KEYFRAME_H
