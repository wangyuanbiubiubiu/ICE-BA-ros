//
// Created by wya on 2020/3/6.
//

#ifndef ICE_BA_PNPPOSEESTIMATOR_H
#define ICE_BA_PNPPOSEESTIMATOR_H


#include <memory>
#include <vector>

#include <Eigen/Core>
#include <glog/logging.h>

#include "../Frontend/cameras/PinholeCamera.hpp"  // for vio::cameras::CameraBase
#include "../Frontend/cameras/NCameraSystem.hpp"

namespace geometric_vision {

    class PnpPoseEstimator {
    public:
        explicit PnpPoseEstimator(bool run_nonlinear_refinement)
                : random_seed_(true),
                  run_nonlinear_refinement_(run_nonlinear_refinement) {}
        /// This constructor should be used for when a deterministic seed (set from
        /// outside) is necessary, such as for testing.
        PnpPoseEstimator(bool run_nonlinear_refinement, bool random_seed)
                : random_seed_(random_seed),
                  run_nonlinear_refinement_(run_nonlinear_refinement) {}


        bool absolutePoseRansacPinholeCam(
                const Eigen::Matrix2Xf& measurements,
                const Eigen::Matrix3Xf& G_landmark_positions, double pixel_sigma,
                int max_ransac_iters, std::shared_ptr<vio::cameras::CameraBase> camera_ptr,
                Eigen::Matrix4f & Twc, std::vector<int>* inliers,
                std::vector<double>* inlier_distances_to_model, int* num_iters,bool use_prior_pose = false);

        bool absolutePoseRansac(const Eigen::Matrix2Xf& measurements,
                const Eigen::Matrix3Xf& G_landmark_positions, double ransac_threshold,
                int max_ransac_iters, std::shared_ptr<vio::cameras::CameraBase> camera_ptr,
                Eigen::Matrix4f & Twc, std::vector<int>* inliers,std::vector<double>* inlier_distances_to_model,
                int* num_iters,bool use_prior_pose = false);


        bool absoluteMultiPoseRansacPinholeCam(
                const Eigen::Matrix2Xf& measurements,
                const std::vector<int>& measurement_camera_indices,
                const Eigen::Matrix3Xf& G_landmark_positions, double pixel_sigma,
                int max_ransac_iters,std::shared_ptr<vio::cameras::NCameraSystem> ncamera_ptr,
                Eigen::Matrix4f & Twc, std::vector<int>* inliers,
                std::vector<double>* inlier_distances_to_model, int* num_iters);

        bool absoluteMultiPoseRansac(
                const Eigen::Matrix2Xf& measurements,
                const std::vector<int>& measurement_camera_indices,
                const Eigen::Matrix3Xf& G_landmark_positions, double ransac_threshold,
                int max_ransac_iters,std::shared_ptr<vio::cameras::NCameraSystem> ncamera_ptr,
                Eigen::Matrix4f & Twc, std::vector<int>* inliers,
                std::vector<double>* inlier_distances_to_model, int* num_iters);

    private:
        /// Whether to let RANSAC pick a timestamp-based random seed or not. If false,
        /// a seed can be set with srand().
        const bool random_seed_;

        /// Run nonlinear refinement over all inliers.
        const bool run_nonlinear_refinement_;
    };

}


#endif //ICE_BA_PNPPOSEESTIMATOR_H
