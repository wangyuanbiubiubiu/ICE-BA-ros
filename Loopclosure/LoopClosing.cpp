//
// Created by wya on 2020/3/5.
//

#include "LoopClosing.h"
#include "PnpPoseEstimator.h"

DEFINE_int32(
        lc_num_ransac_iters, 100,
        "Maximum number of ransac iterations for absolute pose recovery.");
DEFINE_bool(
        lc_nonlinear_refinement_p3p, true,
        "If nonlinear refinement on all ransac inliers should be run.");
DEFINE_double(lc_ransac_pixel_sigma, 2.0, "Pixel sigma for ransac.");
DEFINE_int32(lc_min_inlier_count, 7, "Minimum inlier count for loop closure.");
DEFINE_double(
        lc_min_inlier_ratio, 0.3, "Minimum inlier ratio for loop closure.");
DEFINE_bool(lc_use_random_pnp_seed, true, "Use random seed for pnp RANSAC.");
DEFINE_int32(Match_count, 10, "Minimum inlier count for loop closure.");

namespace LC
{

    LoopClosing::LoopClosing(const string & voc_path)
    {
        voc = shared_ptr<DBoW3::Vocabulary>(new DBoW3::Vocabulary());
        voc->load(voc_path);
        if ( voc->empty() )
        {
            cerr<<"Vocabulary does not exist."<<endl;
        } else
        {
            std::cout<<"load voc finish"<<std::endl;
        }

        kfDB = shared_ptr<DBoW3::Database>(new DBoW3::Database(*voc));
        mainLoop = thread(&LoopClosing::Run, this);
    }

    LoopClosing::LoopClosing(const std::string &voc_path,
                             const Eigen::Matrix3f &left_camK ,
                             const cv::Mat_<float> &left_cv_dist_coeff, const bool fisheye,
                             const cv::Mat_<uchar> &left_mask)
    {



        voc = shared_ptr<DBoW3::Vocabulary>(new DBoW3::Vocabulary());
        voc->load(voc_path);
        if ( voc->empty() )
        {
            cerr<<"Vocabulary does not exist."<<endl;
        } else
        {
            std::cout<<"load voc finish"<<std::endl;
        }

        //初始化左相机的针孔投影、rantan畸变模型
        if(fisheye)
        {
            cam_left_.reset(new vio::cameras::PinholeCamera<
                    vio::cameras::EquidistantDistortion>(
                    left_mask.cols,
                    left_mask.rows,
                    left_camK(0, 0),  // focalLength[0],
                    left_camK(1, 1),  // focalLength[1],
                    left_camK(0, 2),  // principalPoint[0],
                    left_camK(1, 2),  // principalPoint[1],
                    vio::cameras::EquidistantDistortion(
                            left_cv_dist_coeff(0),
                            left_cv_dist_coeff(1),
                            left_cv_dist_coeff(2),
                            left_cv_dist_coeff(3))));
        }
        else if (left_cv_dist_coeff.rows == 8) {
            cam_left_.reset(new vio::cameras::PinholeCamera<
                    vio::cameras::RadialTangentialDistortion8>(
                    left_mask.cols,
                    left_mask.rows,
                    left_camK(0, 0),  // focalLength[0],
                    left_camK(1, 1),  // focalLength[1],
                    left_camK(0, 2),  // principalPoint[0],
                    left_camK(1, 2),  // principalPoint[1],
                    vio::cameras::RadialTangentialDistortion8(
                            left_cv_dist_coeff(0),
                            left_cv_dist_coeff(1),
                            left_cv_dist_coeff(2),
                            left_cv_dist_coeff(3),
                            left_cv_dist_coeff(4),
                            left_cv_dist_coeff(5),
                            left_cv_dist_coeff(6),
                            left_cv_dist_coeff(7))));
        } else if (left_cv_dist_coeff.rows == 4) {
            cam_left_.reset(new vio::cameras::PinholeCamera<
                    vio::cameras::RadialTangentialDistortion>(
                    left_mask.cols,
                    left_mask.rows,
                    left_camK(0, 0),  // focalLength[0],
                    left_camK(1, 1),  // focalLength[1],
                    left_camK(0, 2),  // principalPoint[0],
                    left_camK(1, 2),  // principalPoint[1],
                    vio::cameras::RadialTangentialDistortion(
                            left_cv_dist_coeff(0),
                            left_cv_dist_coeff(1),
                            left_cv_dist_coeff(2),
                            left_cv_dist_coeff(3))));
        } else {
            LOG(FATAL) << "Dist model unsupported for cam_left_";
        }
        cam_left_->setMask(left_mask);
        kfDB = shared_ptr<DBoW3::Database>(new DBoW3::Database(*voc));
        mainLoop = thread(&LoopClosing::Run, this);
    }

    //需要更新关键帧pose,共视帧,地图点
    void LoopClosing::UpdateKfInfo(const IBA::Global_Map & UpdateGM)
    {
        unique_lock<mutex> GM_lock(mutex_GM);
        //更新关键帧pose,共视帧
        for (int i = 0; i < UpdateGM.CsKF.size(); ++i)
        {
            int iFrm = UpdateGM.iFrmsKF[i];
            //转换一下pose
            Eigen::Matrix4f kf_pose;
            kf_pose.setIdentity();
            for (int k = 0; k < 3; ++k)
            {
                kf_pose(k, 3) = UpdateGM.CsKF[i].p[k];
                for (int j = 0; j < 3; ++j) {
                    kf_pose(k,j) = UpdateGM.CsKF[i].R[j][k];
                }

            }

            assert(All_KF_Info[iFrm_2_idx.find(iFrm)->second]->iFrm == iFrm);
            All_KF_Info[iFrm_2_idx.find(iFrm)->second]->Twc = kf_pose;
            All_KF_Info[iFrm_2_idx.find(iFrm)->second]->CovisibleKFs = UpdateGM.CovisibleKFs[i];
            All_KF_Info[iFrm_2_idx.find(iFrm)->second]->valid = true;

        }
        for (int k = 0; k < UpdateGM.Xs.size(); ++k)
        {
            int G_idx = UpdateGM.Xs[k].idx;

            assert(All_Mps[Gid_2_idx[G_idx]]->G_id == G_idx);
            All_Mps[Gid_2_idx[G_idx]]->Pw = Eigen::Vector3f{UpdateGM.Xs[k].X[0],UpdateGM.Xs[k].X[1],UpdateGM.Xs[k].X[2]};
            All_Mps[Gid_2_idx[G_idx]]->valid = true;
        }
    }

    void LoopClosing::InsertKeyFrame(shared_ptr<KeyFrame> kf) {
        kf->ComputeBoW(voc);

        unique_lock<mutex> GM_lock(mutex_GM);
        //地图点的索引
        Gid_2_idx.resize(kf->mKPs.back().class_id + 1,-1);
        for (int i = 0; i < kf->mKPs.size(); ++i)
        {
            int G_idx = kf->mKPs[i].class_id;
            if(Gid_2_idx[G_idx] == -1)
            {
                Gid_2_idx[G_idx] = All_Mps.size();
                All_Mps.push_back(std::shared_ptr<LC::MapPoint> (new LC::MapPoint()));
                All_Mps[Gid_2_idx[G_idx]]->G_id = G_idx;
            }

        }
        //关键帧的索引
        iFrm_2_idx.insert(make_pair(kf->iFrm,All_KF_Info.size()));
        All_KF_Info.push_back(std::shared_ptr<LC::KF_info> (new LC::KF_info()));
        All_KF_Info[iFrm_2_idx.find(kf->iFrm)->second]->iFrm = kf->iFrm;
        allKF.push_back(kf);

        unique_lock<mutex> lock(mutexKFQueue);
        if(kf->iFrm!=0)//关键帧还是会保存第0帧,只不过不会在词袋里加第0帧
        {
            KFqueue.push_back(kf);
        }
    }

    bool LoopClosing::CheckNewKeyFrames()
    {
        unique_lock<mutex> lock(mutexKFQueue);
        return(!KFqueue.empty());
    }


    void LoopClosing::Run()
    {

        while (1) {
            {
                if (finished) {
                    break;
                }

                // get the oldest one
                unique_lock<mutex> lock(mutexKFQueue);
                if (KFqueue.empty()) {
                    lock.unlock();
                    usleep(5000);
                    continue;
                }

//                while(KFqueue.size() > 1)//只闭环最新的,不过也可以注释掉,改成闭环最老的
//                {
//                    mCurrentKF = KFqueue.front();
//                    DBoW3::EntryId id = kfDB->add(mCurrentKF->mBowVec, mCurrentKF->mFeatVec);
//                    mCurrentKF->mloop_Descriptors.release();
//                    mCurrentKF->mloop_KPs.clear();
//                    mCurrentKF->mloop_FeatVec.clear();
//                    maxKFId = id;
//                    checkedKFs[id] = mCurrentKF;
//                    KFqueue.pop_front();
//                }
                mCurrentKF = KFqueue.front();
                KFqueue.pop_front();

                if (KFqueue.size() > 20)
                    KFqueue.clear();

            }

            if (DetectLoop(mCurrentKF))
            {
                Eigen::Matrix4f PnP_Twc;
                if(CorrectLoop(PnP_Twc))
                {
                    if(ComputeOptimizedPose(PnP_Twc))
                    {
                        vector<Eigen::Matrix4f> r_poses;
                        vector<int> r_iFrms;
                        for (int i = 0; i < candidate_co_KFs.size(); ++i)
                        {
                            r_poses.push_back(All_KF_Info[iFrm_2_idx.find(candidate_co_KFs[i]->iFrm)->second]->Twc);
                            r_iFrms.push_back(candidate_co_KFs[i]->iFrm);
                        }
                        if (m_callback) {
                            m_callback(r_poses,PnP_Twc,r_iFrms,mCurrentKF->iFrm);
                        }
                    }
                }
            }

            usleep(5000);
        }

    }

    vector<shared_ptr<KeyFrame>> LoopClosing::GetConnectedKeyFrames(int iFrm)
    {
        vector<shared_ptr<KeyFrame>> Co_KFs;
        std::vector<int> CovisibleKFs = All_KF_Info[iFrm_2_idx.find(iFrm)->second]->CovisibleKFs;
        for (int i = 0; i < CovisibleKFs.size(); ++i)
        {
            int CovisibleKF_iFrm = CovisibleKFs[i];
            assert(allKF[iFrm_2_idx.find(CovisibleKF_iFrm)->second]->iFrm == CovisibleKF_iFrm);
            if(All_KF_Info[iFrm_2_idx.find(CovisibleKF_iFrm)->second]->valid)
                Co_KFs.push_back(allKF[iFrm_2_idx.find(CovisibleKF_iFrm)->second]);
        }
        return Co_KFs;
    }

    bool LoopClosing::DetectLoop(shared_ptr<KeyFrame> &kf)
    {

        DBoW3::QueryResults results;
        kfDB->query(kf->mloop_BowVec, results, 1, maxKFId - kfGap);//前10帧的

        if (results.empty()) {
            DBoW3::EntryId id = kfDB->add(kf->mBowVec, kf->mFeatVec);
            kf->mloop_Descriptors.release();
            kf->mloop_KPs.clear();
            kf->mloop_FeatVec.clear();
            maxKFId = id;
            checkedKFs[id] = kf;
            return false;
        }
        DBoW3::Result r = results[0];
        candidateKF = checkedKFs[r.Id];
        float minScore = 1; //老帧和它共视帧之间的得分
        unique_lock<mutex> lock(mutex_GM);
        vector<shared_ptr<KeyFrame>> co_KFs = GetConnectedKeyFrames(checkedKFs[r.Id]->iFrm);//找老帧的共视帧,因为新帧的共视帧可能还没更新
        lock.unlock();
        for (int i = 0; i < co_KFs.size(); ++i)
        {
            if(co_KFs[i]->iFrm == kf->iFrm)
            {
                DBoW3::EntryId id = kfDB->add(kf->mBowVec, kf->mFeatVec);
                kf->mloop_Descriptors.release();
                kf->mloop_KPs.clear();
                kf->mloop_FeatVec.clear();
                maxKFId = id;
                checkedKFs[id] = kf;
                return false;
            }
            const DBoW3::BowVector &BowVec = co_KFs[i]->mloop_BowVec;
            float score = voc->score(BowVec, checkedKFs[r.Id]->mBowVec);
            // 更新最低得分
            if(score<minScore)
                minScore = score;

        }
        if(minScore < historyminScore)
            historyminScore = minScore;

//        minScoreAccept = (minScore == 1)? minScoreAccept : minScore;
        if (r.Score < minScore)
        {
            DBoW3::EntryId id = kfDB->add(kf->mBowVec, kf->mFeatVec);
            maxKFId = id;
            checkedKFs[id] = kf;
            kf->mloop_Descriptors.release();
            kf->mloop_KPs.clear();
            kf->mloop_FeatVec.clear();
            return false;
        }
        else if(r.Score < minScoreAccept)
        {
            DBoW3::EntryId id = kfDB->add(kf->mBowVec, kf->mFeatVec);
            maxKFId = id;
            checkedKFs[id] = kf;
            candidateKF = checkedKFs[r.Id];

            return true;
        }

        // detected a possible loop
        candidateKF = checkedKFs[r.Id];
        return true;   // don't add into database
    }

    //计算一个大概的位姿初值
    bool LoopClosing::CorrectLoop(Eigen::Matrix4f & PnP_Twc)
    { //没用cv的是因为,cv只支持radtan的ransac,所以就用opengv的,输入的是归一化的坐标

        // We compute first ORB matches for each candidate
        FeatureMatcher matcher(0.75, true);
        bool success = false;


        shared_ptr<KeyFrame> pKF = candidateKF;
        vector<int> matches;
        unique_lock<mutex> lock(mutex_GM);

        vector<shared_ptr<KeyFrame>> co_KFs = GetConnectedKeyFrames(pKF->iFrm);//找老帧的共视帧,因为新帧的共视帧可能还没更新

        vector<shared_ptr<KeyFrame>> candidateKFs;
        candidateKFs.push_back(pKF);
        candidateKFs.insert(candidateKFs.end(), co_KFs.begin(), co_KFs.end());
        int nmatches = matcher.SearchByBoW(mCurrentKF, pKF, matches,All_Mps,Gid_2_idx);
        lock.unlock();
        if (nmatches < FLAGS_Match_count)
        {
            matcher.SetRatio(0.75);
            matcher.SetTH(50);
            matches.clear();
            unique_lock<mutex> lock(mutex_GM);
            nmatches = matcher.SearchByBoW(mCurrentKF, candidateKFs, matches,All_Mps,Gid_2_idx);
            lock.unlock();
            if(nmatches < FLAGS_Match_count+5)
            {
                return false;
            }

        }

        Eigen::Matrix2Xf measurements;
        Eigen::Matrix3Xf G_landmark_positions;

        measurements.resize(Eigen::NoChange, nmatches);
        G_landmark_positions.resize(Eigen::NoChange,nmatches);
        unique_lock<mutex> gmlock(mutex_GM);
        int idx = 0;
        for (int i = 0; i < matches.size(); ++i)
        {
            if(matches[i] != -1)
            {
                measurements.col(idx) = Eigen::Vector2f{mCurrentKF->mloop_KPs[i].pt.x,mCurrentKF->mloop_KPs[i].pt.y};
                G_landmark_positions.col(idx++) = All_Mps[Gid_2_idx[matches[i]]]->Pw;
                assert(All_Mps[Gid_2_idx[matches[i]]]->valid);
            }
        }
        PnP_Twc = All_KF_Info[iFrm_2_idx.find(mCurrentKF->iFrm)->second]->Twc;
        gmlock.unlock();
        geometric_vision::PnpPoseEstimator pose_estimator(
                FLAGS_lc_nonlinear_refinement_p3p, FLAGS_lc_use_random_pnp_seed);

        double inlier_ratio;
        int num_inliers;
        std::vector<double> inlier_distances_to_model;
        int num_iters;
        std::vector<int> inliers;
        pose_estimator.absolutePoseRansacPinholeCam(
                measurements, G_landmark_positions,
                FLAGS_lc_ransac_pixel_sigma*4, FLAGS_lc_num_ransac_iters, cam_left_,
                PnP_Twc, &inliers, &inlier_distances_to_model, &num_iters,true);
        CHECK_EQ(inliers.size(), inlier_distances_to_model.size());
        num_inliers = static_cast<int>(inliers.size());
        inlier_ratio = static_cast<double>(num_inliers) /
                       static_cast<double>(G_landmark_positions.cols());
        if (inlier_ratio >= (FLAGS_lc_min_inlier_ratio/2.0f) || num_inliers >= FLAGS_lc_min_inlier_count )
        {
            success = true;
        }

        return success;
    }

    bool LoopClosing::ComputeOptimizedPose(Eigen::Matrix4f & PnP_Twc)
    {

        bool success = false;
        shared_ptr<KeyFrame> pKF = candidateKF;
        vector<int> matches;
        candidate_co_KFs.clear();
        vector<shared_ptr<KeyFrame>> co_candidateKFs;
        set<int> co_candidateKFs_iFrm;//用来统计一下已经存在的关键帧

        vector<std::pair<Eigen::Vector3f,vector<cv::Mat>>> Mp_info;//只保存有效的地图点以及这个点在这些共视帧中的描述子
        map<int,int> Mp_idxs; //key是地图点的全局id,value是地图点在Mp_info中的索引

        unique_lock<mutex> lock(mutex_GM);
        //将候选帧的共视以及共视的共视的地图都取出来,过滤在这步做了,这样后面就可以不用做了
        //先取出候选帧的关键帧
        vector<shared_ptr<KeyFrame>> co_KFs = GetConnectedKeyFrames(pKF->iFrm);
        candidate_co_KFs.push_back(pKF);
        co_candidateKFs.push_back(pKF);
        co_candidateKFs_iFrm.insert(pKF->iFrm);
        {
            for (int i = 0; i < co_KFs.size(); ++i)//遍历所有候选帧的关键帧的关键帧
            {
                if(!co_candidateKFs_iFrm.count(co_KFs[i]->iFrm))
                {
                    co_candidateKFs.push_back(co_KFs[i]);
                    candidate_co_KFs.push_back(co_KFs[i]);
                    co_candidateKFs_iFrm.insert(co_KFs[i]->iFrm);
                }
                vector<shared_ptr<KeyFrame>> co_co_KFs = GetConnectedKeyFrames(co_KFs[i]->iFrm);
                for (int j = 0; j < co_co_KFs.size(); ++j)
                {
                    if(!co_candidateKFs_iFrm.count(co_co_KFs[j]->iFrm))
                    {
                        co_candidateKFs.push_back(co_co_KFs[j]);
                        co_candidateKFs_iFrm.insert(co_co_KFs[j]->iFrm);
                    }

                }
            }
        }


        {
            //遍历所有共视关键帧的地图点,将对应的描述子保存下来(如果多个帧都观测到的话,就保存多个描述子)
            for (int k = 0; k < co_candidateKFs.size(); ++k)
                for (int i = 0; i < co_candidateKFs[k]->mKPs.size(); ++i)
                    if(Mp_idxs.count(co_candidateKFs[k]->mKPs[i].class_id))//其他关键帧也有这个地图点的id
                    {
                        int mp_idx = Mp_idxs.find(co_candidateKFs[k]->mKPs[i].class_id)->second;
                        Mp_info[mp_idx].second.push_back(co_candidateKFs[k]->mDescriptors.row(i));
                    } else//需要添加新的地图点
                    {
                        if(All_Mps[Gid_2_idx[co_candidateKFs[k]->mKPs[i].class_id]]->valid)//先判断一下这个地图点是否有效
                        {
                            Eigen::Vector3f Mp_W = All_Mps[Gid_2_idx[co_candidateKFs[k]->mKPs[i].class_id]]->Pw;
                            vector<cv::Mat> first_Descriptor;
                            first_Descriptor.push_back(co_candidateKFs[k]->mDescriptors.row(i));
                            Mp_idxs.insert(make_pair(co_candidateKFs[k]->mKPs[i].class_id,Mp_info.size()));
                            Mp_info.push_back(make_pair(Mp_W,first_Descriptor));
                        }
                    }
        }


        lock.unlock();
        FeatureMatcher matcher(0.75, true);
        int find_distance = 15;//10像素范围内查找
        int nmatches = matcher.SearchByProjection(mCurrentKF, Mp_info, matches,cam_left_,find_distance,PnP_Twc);
        if(nmatches < FLAGS_Match_count*2)
        {
            matches.clear();
            nmatches = matcher.SearchByProjection(mCurrentKF, Mp_info, matches,cam_left_,find_distance*2,PnP_Twc);
        }
        if(nmatches >= FLAGS_Match_count*2)
        {
            Eigen::Matrix2Xf measurements;
            Eigen::Matrix3Xf G_landmark_positions;

            measurements.resize(Eigen::NoChange, nmatches);
            G_landmark_positions.resize(Eigen::NoChange,nmatches);
            int idx = 0;
            for (int i = 0; i < matches.size(); ++i)
            {
                if(matches[i] > -1)
                {
                    measurements.col(idx) = Eigen::Vector2f{mCurrentKF->mloop_KPs[i].pt.x,mCurrentKF->mloop_KPs[i].pt.y};
                    G_landmark_positions.col(idx++) = Mp_info[matches[i]].first;
                }
            }

            geometric_vision::PnpPoseEstimator pose_estimator(
                    FLAGS_lc_nonlinear_refinement_p3p, FLAGS_lc_use_random_pnp_seed);

            double inlier_ratio;
            int num_inliers;
            std::vector<double> inlier_distances_to_model;
            int num_iters;
            std::vector<int> inliers;
            pose_estimator.absolutePoseRansacPinholeCam(
                    measurements, G_landmark_positions,
                    FLAGS_lc_ransac_pixel_sigma, FLAGS_lc_num_ransac_iters*5, cam_left_,
                    PnP_Twc, &inliers, &inlier_distances_to_model, &num_iters,true);
            CHECK_EQ(inliers.size(), inlier_distances_to_model.size());
            num_inliers = static_cast<int>(inliers.size());
            inlier_ratio = static_cast<double>(num_inliers) /
                           static_cast<double>(G_landmark_positions.cols());
            if (inlier_ratio >= FLAGS_lc_min_inlier_ratio || num_inliers >= FLAGS_lc_min_inlier_count+3 )
            {
                success = true;
            }

        }
        return success;
    }


    void LoopClosing::SetCallback(const LoopCallback & loop_callback)
    {
        m_callback = loop_callback;
    }
}
