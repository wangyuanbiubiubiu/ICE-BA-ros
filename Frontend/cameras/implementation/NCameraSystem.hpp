/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Apr 1, 2015
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file implementation/NCameraSystem.hpp
 * @brief Header implementation file for the NCameraSystem class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */


/// \brief okvis Main namespace of this package.
namespace vio {
/// \brief cameras Namespace for camera-related functionality.
    namespace cameras {

// Default constructor
        NCameraSystem::NCameraSystem()
        {
        }
// Construct with vector of extrinsics and geometries
        NCameraSystem::NCameraSystem(
                const std::vector<std::shared_ptr<const Eigen::Matrix4d>> & T_SC,
                const std::vector<std::shared_ptr<const cameras::CameraBase>> & cameraGeometries,
                const std::vector<DistortionType>& distortionTypes,
                bool computeOverlaps)
                : T_SC_(T_SC),
                  cameraGeometries_(cameraGeometries),
                  distortionTypes_(distortionTypes)
        {
            assert(T_SC.size() == cameraGeometries.size());
            assert(T_SC.size() == distortionTypes.size());

            if (computeOverlaps) {
                this->computeOverlaps();
            }
        }
        NCameraSystem::~NCameraSystem()
        {
        }

// Reset with vector of extrinsics and geometries
        void NCameraSystem::reset(
                const std::vector<std::shared_ptr<const Eigen::Matrix4d>> & T_SC,
                const std::vector<std::shared_ptr<const cameras::CameraBase>> & cameraGeometries,
                const std::vector<DistortionType>& distortionTypes,
                bool computeOverlaps)
        {
            assert(T_SC.size() == cameraGeometries.size());
            assert(T_SC.size() == distortionTypes.size());

            T_SC_ = T_SC;
            cameraGeometries_ = cameraGeometries;
            distortionTypes_ = distortionTypes;

            // recompute overlaps if requested
            if (computeOverlaps) {
                this->computeOverlaps();
            }
        }

// Reset with vector of extrinsics and geometries
        void NCameraSystem::addCamera(
                std::shared_ptr<const Eigen::Matrix4d> T_SC,
                std::shared_ptr<const cameras::CameraBase> cameraGeometry,
                DistortionType distortionType,
                bool computeOverlaps)
        {
            T_SC_.push_back(T_SC);
            cameraGeometries_.push_back(cameraGeometry);
            distortionTypes_.push_back(distortionType);

            // recompute overlaps if requested
            if (computeOverlaps) {
                this->computeOverlaps();
            }
        }

        void NCameraSystem::computeOverlaps()
        {
            //我不用这个
        }

// get the pose of the IMU frame S with respect to the camera cameraIndex
        std::shared_ptr<const Eigen::Matrix4d> NCameraSystem::T_SC(
                int cameraIndex) const
        {
            assert(cameraIndex < T_SC_.size());

            return T_SC_[cameraIndex];
        }

//get the camera geometry of camera cameraIndex
        std::shared_ptr<const cameras::CameraBase> NCameraSystem::cameraGeometry(
                int cameraIndex) const
        {
            assert(cameraIndex < cameraGeometries_.size());

            return cameraGeometries_[cameraIndex];
        }

// get the distortion type of cmaera cameraIndex
        inline NCameraSystem::DistortionType NCameraSystem::distortionType(int cameraIndex) const
        {
            assert(cameraIndex < cameraGeometries_.size());

            return distortionTypes_[cameraIndex];
        }

// Get the overlap mask
        const cv::Mat NCameraSystem::overlap(int cameraIndexSeenBy,
                                             int cameraIndex) const
        {
            assert(cameraIndexSeenBy < T_SC_.size());
            assert(cameraIndex < T_SC_.size());
            assert(overlapComputationValid());

            return overlapMats_[cameraIndexSeenBy][cameraIndex];
        }

// Can the first camera see parts of the FOV of the second camera?
        bool NCameraSystem::hasOverlap(int cameraIndexSeenBy,
                                       int cameraIndex) const
        {
            assert(cameraIndexSeenBy < T_SC_.size());
            assert(cameraIndex < T_SC_.size());
            assert(overlapComputationValid());

            return overlaps_[cameraIndexSeenBy][cameraIndex];
        }

        bool NCameraSystem::overlapComputationValid() const {
            assert(T_SC_.size() == cameraGeometries_.size());

            if(overlaps_.size() != cameraGeometries_.size()) {
                return false;
            }
            if(overlapMats_.size() != cameraGeometries_.size()) {
                return false;
            }

            // also check for each element
            for(int i= 0; i<overlaps_.size(); ++i){
                if(overlaps_[i].size() != cameraGeometries_.size()) {
                    return false;
                }
                if(overlapMats_[i].size() != cameraGeometries_.size()) {
                    return false;
                }
            }
            return true;
        }

        int NCameraSystem::numCameras() const {
            return cameraGeometries_.size();
        }

    }  // namespace cameras
}  // namespace okvis