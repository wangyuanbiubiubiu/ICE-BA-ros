//
// Created by wya on 2020/3/6.
//

#include "PnpPoseEstimator.h"
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/absolute_pose/NoncentralAbsoluteAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>


namespace geometric_vision
{

    bool PnpPoseEstimator::absolutePoseRansacPinholeCam(
            const Eigen::Matrix2Xf& measurements,
            const Eigen::Matrix3Xf& G_landmark_positions, double pixel_sigma,
            int max_ransac_iters, std::shared_ptr<vio::cameras::CameraBase> camera_ptr,
            Eigen::Matrix4f & Twc, std::vector<int>* inliers,
            std::vector<double>* inlier_distances_to_model, int* num_iters,bool use_prior_pose)
    {
        CHECK_NOTNULL(inliers);
        CHECK_NOTNULL(inlier_distances_to_model);
        CHECK_NOTNULL(num_iters);
        CHECK_EQ(measurements.cols(), G_landmark_positions.cols());

        double focal_length = 0;
        Eigen::VectorXd cam_intrinsics;
        camera_ptr->getIntrinsics(&cam_intrinsics);
        focal_length = (cam_intrinsics[0] + cam_intrinsics[1]) / 2.0;


        const double ransac_threshold = 1.0 - cos(atan(pixel_sigma / focal_length));

        return absolutePoseRansac(measurements,G_landmark_positions, ransac_threshold,
                                       max_ransac_iters, camera_ptr, Twc, inliers,
                                       inlier_distances_to_model, num_iters,use_prior_pose);
    }




    bool PnpPoseEstimator::absolutePoseRansac(
            const Eigen::Matrix2Xf& measurements,
            const Eigen::Matrix3Xf& G_landmark_positions, double ransac_threshold,
            int max_ransac_iters, std::shared_ptr<vio::cameras::CameraBase> camera_ptr,
            Eigen::Matrix4f & Twc, std::vector<int>* inliers,std::vector<double>* inlier_distances_to_model,
            int* num_iters,bool use_prior_pose)
    {
        CHECK_NOTNULL(inliers);
        CHECK_NOTNULL(num_iters);
        CHECK_EQ(measurements.cols(), G_landmark_positions.cols());

        opengv::points_t points;
        opengv::bearingVectors_t bearing_vectors;
        points.resize(measurements.cols());
        bearing_vectors.resize(measurements.cols());


        for (int i = 0; i < measurements.cols(); ++i)
        {
            Eigen::Vector3f f_left;//左相机的归一化坐标

            camera_ptr->backProject(measurements.col(i), &f_left);
            bearing_vectors[i] = f_left.cast<double>();
            bearing_vectors[i].normalize();
            points[i] = G_landmark_positions.col(i).cast<double>();
        }

        opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearing_vectors,
                                                              points);

        if(use_prior_pose)
        {
            Eigen::Matrix4d  Twc_d = Twc.cast<double>();
            Eigen::Matrix3d Rwc_d = Twc_d.block<3,3>(0,0);
            Eigen::Vector3d twc_d = Twc_d.block<3,1>(0,3);
            adapter.setR(Rwc_d);
            adapter.sett(twc_d);
        }

        opengv::sac::Ransac<
                opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
        std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
                absposeproblem_ptr(
                new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
                        adapter, opengv::sac_problems::absolute_pose::
                        AbsolutePoseSacProblem::EPNP,
                        random_seed_));
        ransac.sac_model_ = absposeproblem_ptr;
        ransac.threshold_ = ransac_threshold;
        ransac.max_iterations_ = max_ransac_iters;
        bool ransac_success = ransac.computeModel();
        CHECK_EQ(ransac.inliers_.size(), ransac.inlier_distances_to_model_.size());

        if (ransac_success) {
            // Optional nonlinear model refinement over all inliers.
            Eigen::Matrix<double, 3, 4> final_model = ransac.model_coefficients_;
            if (run_nonlinear_refinement_) {
                absposeproblem_ptr->optimizeModelCoefficients(ransac.inliers_,
                                                              ransac.model_coefficients_,
                                                              final_model);
            }

            Eigen::Matrix4d  Twc_d;
            Twc_d.setIdentity();
            Twc_d.block<3,1>(0,3) = final_model.rightCols(1);
            Twc_d.block<3,3>(0,0) = final_model.leftCols(3);
            Twc = Twc_d.cast<float>();
        }

        *inlier_distances_to_model = ransac.inlier_distances_to_model_;
        *inliers = ransac.inliers_;
        *num_iters = ransac.iterations_;
        return ransac_success;
    }

    bool PnpPoseEstimator::absoluteMultiPoseRansacPinholeCam(
            const Eigen::Matrix2Xf& measurements,
            const std::vector<int> & measurement_camera_indices,
            const Eigen::Matrix3Xf& G_landmark_positions, double pixel_sigma,
            int max_ransac_iters,std::shared_ptr<vio::cameras::NCameraSystem> ncamera_ptr,
            Eigen::Matrix4f & Twc, std::vector<int>* inliers,
            std::vector<double>* inlier_distances_to_model, int* num_iters)
    {
        CHECK_NOTNULL(inliers);
        CHECK_NOTNULL(inlier_distances_to_model);
        CHECK_NOTNULL(num_iters);
        CHECK_EQ(measurements.cols(), G_landmark_positions.cols());
        CHECK_EQ(measurements.cols(), static_cast<int>(measurement_camera_indices.size()));

        const int num_cameras = ncamera_ptr->numCameras();
        double focal_length = 0;

        for (int camera_index = 0; camera_index < num_cameras; ++camera_index)
        {
            std::shared_ptr<const vio::cameras::CameraBase> camera_ptr = ncamera_ptr->cameraGeometry(camera_index);
            Eigen::VectorXd cam_intrinsics;
            camera_ptr->getIntrinsics(&cam_intrinsics);
            focal_length = (cam_intrinsics[0] + cam_intrinsics[1]) / 2.0;
        }

        focal_length /= (2.0 * static_cast<double>(num_cameras));


        const double ransac_threshold = 1.0 - cos(atan(pixel_sigma / focal_length));

        return absoluteMultiPoseRansac(measurements, measurement_camera_indices,
                                       G_landmark_positions, ransac_threshold,
                                       max_ransac_iters, ncamera_ptr, Twc, inliers,
                                       inlier_distances_to_model, num_iters);
    }

    bool PnpPoseEstimator::absoluteMultiPoseRansac(
            const Eigen::Matrix2Xf& measurements,
            const std::vector<int>& measurement_camera_indices,
            const Eigen::Matrix3Xf& G_landmark_positions, double ransac_threshold,
            int max_ransac_iters,std::shared_ptr<vio::cameras::NCameraSystem> ncamera_ptr,
            Eigen::Matrix4f & Twc, std::vector<int>* inliers,
            std::vector<double>* inlier_distances_to_model, int* num_iters)
    {

        CHECK_NOTNULL(inliers);
        CHECK_NOTNULL(inlier_distances_to_model);
        CHECK_NOTNULL(num_iters);
        CHECK_EQ(measurements.cols(), G_landmark_positions.cols());
        CHECK_EQ(measurements.cols(), static_cast<int>(measurement_camera_indices.size()));

        // Fill in camera information from NCamera.
        // Rotation matrix for each camera.
        opengv::rotations_t cam_rotations;
        opengv::translations_t cam_translations;

        const int num_cameras = ncamera_ptr->numCameras();

        cam_rotations.resize(num_cameras);
        cam_translations.resize(num_cameras);


        Eigen::Matrix4d Twb = Twc.cast<double>() * ncamera_ptr->T_SC(0)->inverse();
        for (int camera_index = 0; camera_index < num_cameras; ++camera_index)
        {
            std::shared_ptr<const Eigen::Matrix4d> T_B_C_ptr = ncamera_ptr->T_SC(camera_index);
            cam_rotations[camera_index] = T_B_C_ptr->block<3,3>(0,0);
            cam_translations[camera_index] = T_B_C_ptr->block<3,1>(0,3);
        }

        opengv::points_t points;
        opengv::bearingVectors_t bearing_vectors;
        points.resize(measurements.cols());
        bearing_vectors.resize(measurements.cols());
        for (int i = 0; i < measurements.cols(); ++i)
        {
            // Figure out which camera this corresponds to, and reproject it in the
            // correct camera.
            int camera_index = measurement_camera_indices[i];
            Eigen::Vector3f f_left;//左相机的归一化坐标
            ncamera_ptr->cameraGeometry(camera_index)
                    ->backProject(measurements.col(i), &f_left);
            bearing_vectors[i] = f_left.cast<double>();
            bearing_vectors[i].normalize();
            points[i] = G_landmark_positions.col(i).cast<double>();
        }
        // Basically same as the Central, except measurement_camera_indices, which
        // assigns a camera index to each bearing_vector, and cam_offsets and
        // cam_rotations, which describe the position and orientation of the cameras
        // with respect to the body frame.
        opengv::absolute_pose::NoncentralAbsoluteAdapter adapter(
                bearing_vectors, measurement_camera_indices, points, cam_translations,
                cam_rotations);
        opengv::sac::Ransac<
                opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
        std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
                absposeproblem_ptr(
                new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(adapter,
                                                                                opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::GP3P,//GP3P,edit by wya
                                                                                random_seed_));
        ransac.sac_model_ = absposeproblem_ptr;
        ransac.threshold_ = ransac_threshold;
        ransac.max_iterations_ = max_ransac_iters;
        bool ransac_success = ransac.computeModel();

//  std::cout<<"ransac是否成功了？:"<<ransac_success<<std::endl;
        CHECK_EQ(ransac.inliers_.size(), ransac.inlier_distances_to_model_.size());

        if (ransac_success) {
            // Optional nonlinear model refinement over all inliers.
            Eigen::Matrix<double, 3, 4> final_model = ransac.model_coefficients_;
            if (run_nonlinear_refinement_) {
                absposeproblem_ptr->optimizeModelCoefficients(ransac.inliers_,
                                                              ransac.model_coefficients_,
                                                              final_model);
            }
            Eigen::Matrix4d  Twb_d;
            Twb_d.setIdentity();
            Twb_d.block<3,1>(0,3) = final_model.rightCols(1);
            Twb_d.block<3,3>(0,0) = final_model.leftCols(3);
            Twc = (Twb_d * (*ncamera_ptr->T_SC(0))).cast<float>();
        }

        *inliers = ransac.inliers_;
        *inlier_distances_to_model = ransac.inlier_distances_to_model_;
        *num_iters = ransac.iterations_;

        return ransac_success;
    }



}