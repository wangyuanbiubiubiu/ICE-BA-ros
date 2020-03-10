//
// Created by wya on 2020/3/5.
//

#ifndef ICE_BA_FEATUREMATCHER_H
#define ICE_BA_FEATUREMATCHER_H

#include "KeyFrame.h"
#include "../Frontend/cameras/PinholeCamera.hpp"  // for vio::cameras::CameraBase
#include <opencv2/flann/miniflann.hpp>
namespace LC
{
    /**
     * Match structure
     */
    struct Match {
        Match(int _index1 = -1, int _index2 = -1, int _dist = -1) : index1(_index1), index2(_index2), dist(_dist) {}

        int index1 = -1;
        int index2 = -1;
        int dist = -1;
    };

    class FeatureMatcher {

    public:
        FeatureMatcher(float nnRatio = 0.6, bool checkRot = true) :
                nnRatio(nnRatio), checkOrientation(checkRot) {}

        inline void SetRatio(float ratio){nnRatio = ratio;}

        inline void SetTH(int th ){TH_LOW = th;}

        /// the distance of two descriptors
        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

        int SearchByBoW(shared_ptr<KeyFrame> pKF1,vector<shared_ptr<KeyFrame>> pKF2_KFs,
                std::vector<int> &matches,const std::vector<shared_ptr<MapPoint>> &All_Mps,const std::vector<int> &Gid_2_idx);



        int SearchByBoW(shared_ptr<KeyFrame> pKF1/*新帧*/, shared_ptr<KeyFrame> pKF2/*老帧*/, std::vector<int> &matches,
                        const std::vector<shared_ptr<MapPoint>> & All_Mps, const std::vector<int> &Gid_2_idx);


        int SearchByProjection(shared_ptr<KeyFrame> pKF1/*新帧*/,const vector<std::pair<Eigen::Vector3f,vector<cv::Mat>>> &Mp_info,
            std::vector<int> &matches,std::shared_ptr<vio::cameras::CameraBase> camera_ptr,const int pixel_distance,const Eigen::Matrix4f &Twc);

        /**
         * Search by bag-of-words model
         * @param frame1
         * @param frame2
         * @param matches
         * @return
         */

        void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

        // draw matches, will block until user press a key, return the cv::waitkey code
        int DrawMatches( shared_ptr<KeyFrame> frame1/*新帧*/, shared_ptr<KeyFrame> frame2/*老帧*/, std::vector<int> &matches );

    private:
        float nnRatio = 0.6;
        bool checkOrientation = true;

        int Max_Near_num = 20;
        // configuation
        int TH_LOW = 50;
        const int TH_HIGH = 100;
        const int HISTO_LENGTH = 30;

    };
}
#endif //ICE_BA_FEATUREMATCHER_H
