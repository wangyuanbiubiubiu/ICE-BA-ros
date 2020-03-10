//
// Created by wya on 2020/3/5.
//

#include "FeatureMatcher.h"

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>


namespace LC
{

    int FeatureMatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist=0;

        // 8*32=256bit

        for(int i=0; i<8; i++, pa++, pb++)
        {
            unsigned  int v = *pa ^ *pb;        // 相等为0,不等为1
            // 下面的操作就是计算其中bit为1的个数了,这个操作看上面的链接就好
            // 其实我觉得也还阔以直接使用8bit的查找表,然后做32次寻址操作就完成了;不过缺点是没有利用好CPU的字长
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

    int FeatureMatcher::SearchByBoW(shared_ptr<KeyFrame> pKF1/*新帧*/,vector<shared_ptr<KeyFrame>> pKF2_KFs/*老帧的共视帧*/ ,
            std::vector<int> &matches,const std::vector<shared_ptr<MapPoint>> &All_Mps,const std::vector<int> &Gid_2_idx)
    {
        const vector<cv::KeyPoint> &vKps1 = pKF1->mloop_KPs;
        const DBoW3::FeatureVector &vFeatVec1 = pKF1->mloop_FeatVec;
//        const vector<shared_ptr<MapPoint>> vpMapPoints1 = pKF1->GetMapPointMatches();
        const cv::Mat &Descriptors1 = pKF1->mloop_Descriptors;
// 保存匹配结果
        matches = vector<int>(vKps1.size(),-1);
        int nmatches = 0;
        int last_nmatches = 0;
        for (int k = 0; k <pKF2_KFs.size() ; ++k)
        {

            std::vector<bool> draw_matches;
            draw_matches = vector<bool>(vKps1.size(),false);
            const vector<cv::KeyPoint> &vKps2 = pKF2_KFs[k]->mKPs;
            const DBoW3::FeatureVector &vFeatVec2 = pKF2_KFs[k]->mFeatVec;
            const cv::Mat &Descriptors2 = pKF2_KFs[k]->mDescriptors;

            vector<bool> vbMatched2(vKps2.size(),false);

            // 旋转直方图
            vector<int> rotHist[HISTO_LENGTH];
            for(int i=0;i<HISTO_LENGTH;i++)
                rotHist[i].reserve(500);

            const float factor = HISTO_LENGTH/360.0f;


            DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
            DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
            DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
            DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

            while(f1it != f1end && f2it != f2end)
            {
                if(f1it->first == f2it->first)//步骤1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
                {
                    // 步骤2：遍历KF中属于该node的特征点
                    for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                    {

                        const size_t idx1 = f1it->second[i1];
                        if(matches[idx1] != -1)
                        {
                            continue;
                        }

                        const cv::Mat &d1 = Descriptors1.row(idx1);

                        int bestDist1=256;
                        int bestIdx2 =-1 ;
                        int bestDist2=256;

                        // 步骤3：遍历KF2中属于该node的特征点，找到了最佳匹配点
                        for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                        {
                            const size_t idx2 = f2it->second[i2];

                            if(!All_Mps[Gid_2_idx[vKps2[idx2].class_id]]->valid)//地图点没有效深度
                                continue;

                            // 如果已经有匹配的点，或者遍历到的特征点对应的地图点无效
                            if(vbMatched2[idx2])
                                continue;

                            const cv::Mat &d2 = Descriptors2.row(idx2);

                            int dist = DescriptorDistance(d1,d2);

                            if(dist<bestDist1)
                            {
                                bestDist2=bestDist1;
                                bestDist1=dist;
                                bestIdx2=idx2;
                            }
                            else if(dist<bestDist2)
                            {
                                bestDist2=dist;
                            }
                        }

                        // 步骤4：根据阈值 和 角度投票剔除误匹配
                        // 详见SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)函数步骤4
                        if(bestDist1<TH_LOW)
                        {
                            if(static_cast<float>(bestDist1)<nnRatio*static_cast<float>(bestDist2))
                            {

//                                matches[idx1] = bestIdx2;//特征点位置
                                matches[idx1] = vKps2[bestIdx2].class_id;
                                vbMatched2[bestIdx2]=true;
                                draw_matches[idx1] = true;
                                if(checkOrientation)
                                {
                                    float rot = vKps1[idx1].angle - vKps2[bestIdx2].angle;
                                    if(rot<0.0)
                                        rot+=360.0f;
                                    int bin = round(rot*factor);
                                    if(bin==HISTO_LENGTH)
                                        bin=0;
                                    assert(bin>=0 && bin<HISTO_LENGTH);
                                    rotHist[bin].push_back(idx1);
                                }
                                nmatches++;
                            }
                        }
                    }

                    f1it++;
                    f2it++;
                }
                else if(f1it->first < f2it->first)
                {
                    f1it = vFeatVec1.lower_bound(f2it->first);
                }
                else
                {
                    f2it = vFeatVec2.lower_bound(f1it->first);
                }
            }

            // 旋转检查
            if(checkOrientation)
            {
                int ind1=-1;
                int ind2=-1;
                int ind3=-1;

                ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

                for(int i=0; i<HISTO_LENGTH; i++)
                {
                    if(i==ind1 || i==ind2 || i==ind3)
                        continue;
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        matches[rotHist[i][j]] = -1;
                        draw_matches[rotHist[i][j]] = false;
                        nmatches--;
                    }
                }
            }
            if(false && (nmatches - last_nmatches) > 0)
            {

                cv::Mat img1 = pKF1->mImg;
                cv::Mat img2 = pKF2_KFs[k]->mImg;

                cv::Mat imTrack;
                int cols = img1.cols;
                cv::hconcat(img1, img2, imTrack);//左右图拼接到一起
                cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);


                for (int i = 0; i < matches.size(); ++i)
                {
                    if(matches[i] == -1 || !draw_matches[i])
                        continue;

                    cv::circle(imTrack, cv::Point2f(pKF1->mloop_KPs[i].pt), 1,
                               cv::Scalar(0, 250, 0), 2);

                    cv::circle(imTrack, cv::Point2f(pKF2_KFs[k]->mKPs[matches[i]].pt.x + cols, pKF2_KFs[k]->mKPs[matches[i]].pt.y), 1,
                               cv::Scalar(0, 250, 0), 2);

                    cv::line(imTrack, cv::Point2f(pKF1->mloop_KPs[i].pt),
                             cv::Point2f(pKF2_KFs[k]->mKPs[matches[i]].pt.x + cols, pKF2_KFs[k]->mKPs[matches[i]].pt.y),
                             cv::Scalar(0, 250, 0),
                             1);
                }



                cv::imwrite("/home/wya/ICE-BA-Debug/Loop/loop_img/" + std::to_string(pKF1->iFrm) + "&&" + std::to_string(pKF2_KFs[k]->iFrm)+".jpg",imTrack);
            }

            last_nmatches = nmatches;

        }


        return nmatches;
    }

    int FeatureMatcher::SearchByBoW(shared_ptr<KeyFrame> pKF1/*新帧*/, shared_ptr<KeyFrame> pKF2/*老帧*/, std::vector<int> &matches,
            const std::vector<shared_ptr<MapPoint>> &All_Mps,const std::vector<int> &Gid_2_idx)
    {

        const vector<cv::KeyPoint> &vKps1 = pKF1->mloop_KPs;
        const DBoW3::FeatureVector &vFeatVec1 = pKF1->mloop_FeatVec;
//        const vector<shared_ptr<MapPoint>> vpMapPoints1 = pKF1->GetMapPointMatches();
        const cv::Mat &Descriptors1 = pKF1->mloop_Descriptors;


        const vector<cv::KeyPoint> &vKps2 = pKF2->mKPs;
        const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
        const cv::Mat &Descriptors2 = pKF2->mDescriptors;

        // 保存匹配结果
        matches = vector<int>(vKps1.size(),-1);
        vector<bool> vbMatched2(vKps2.size(),false);

        // 旋转直方图
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);

        const float factor = HISTO_LENGTH/360.0f;

        int nmatches = 0;

        DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while(f1it != f1end && f2it != f2end)
        {
            if(f1it->first == f2it->first)//步骤1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
            {
                // 步骤2：遍历KF中属于该node的特征点
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];


                    const cv::Mat &d1 = Descriptors1.row(idx1);

                    int bestDist1=256;
                    int bestIdx2 =-1 ;
                    int bestDist2=256;

                    // 步骤3：遍历KF2中属于该node的特征点，找到了最佳匹配点
                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        const size_t idx2 = f2it->second[i2];

                        if(!All_Mps[Gid_2_idx[vKps2[idx2].class_id]]->valid)//地图点没有效深度
                            continue;

                        // 如果已经有匹配的点，或者遍历到的特征点对应的地图点无效
                        if(vbMatched2[idx2])
                            continue;

                        const cv::Mat &d2 = Descriptors2.row(idx2);

                        int dist = DescriptorDistance(d1,d2);

                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdx2=idx2;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    }

                    // 步骤4：根据阈值 和 角度投票剔除误匹配
                    // 详见SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)函数步骤4
                    if(bestDist1<TH_LOW)
                    {
                        if(static_cast<float>(bestDist1)<nnRatio*static_cast<float>(bestDist2))
                        {
                            matches[idx1] = vKps2[bestIdx2].class_id;
//                            matches[idx1] = bestIdx2;
                            vbMatched2[bestIdx2]=true;

                            if(checkOrientation)
                            {
                                float rot = vKps1[idx1].angle - vKps2[bestIdx2].angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(idx1);
                            }
                            nmatches++;
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        // 旋转检查
        if(checkOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    matches[rotHist[i][j]] = -1;
                    nmatches--;
                }
            }
        }

        return nmatches;
    }

    //投影下来再匹配一次，因为词袋没有考虑几何约束。
    int FeatureMatcher::SearchByProjection(shared_ptr<KeyFrame> pKF1/*新帧*/,const vector<std::pair<Eigen::Vector3f,vector<cv::Mat>>> &Mp_info,
                                           std::vector<int> &matches,std::shared_ptr<vio::cameras::CameraBase> camera_ptr,
                                           const int pixel_distance,const Eigen::Matrix4f &Twc)
    {
        const vector<cv::KeyPoint> &vKps1 = pKF1->mloop_KPs;
        const cv::Mat &Descriptors1 = pKF1->mloop_Descriptors;
        // 保存匹配结果
        matches = vector<int>(vKps1.size(),-1);
        int nmatches = 0;

        std::map<int,vector<int>> Mp_matches_dist;//记录的是每一个地图点都和哪些特征点是匹配的,因为可能出现多个特征点匹配到一个地图点的情况,或者多个地图点匹配到一个特征点的情况
        std::map<int,int> Mp_matches;
        std::map<int,map<int,vector<int>>> Kps1_matches; //第一个int:pKF1->mloop_KPs中对应的id,第二个int:dist,第三个int:这个dist对应的所有地图点id

        cv::flann::KDTreeIndexParams indexParams(2);
        cv::flann::SearchParams params(128);
        std::vector<cv::Point2f> pKF1_kps;
        for (int i = 0; i < vKps1.size(); ++i)
        {
            pKF1_kps.push_back(vKps1[i].pt);
        }
        cv::flann::Index Kd_database = cv::flann::Index(cv::Mat(pKF1_kps).reshape(1), indexParams);//构造一下新帧所有的特征点的kd树,方便查找
        //遍历所有地图点,将它投到图像上和周围的特征点进行匹配，记录所有的得分
        int findss = 0;
        for (int j = 0; j < Mp_info.size(); ++j)
        {
            Eigen::Vector3f Pw = Mp_info[j].first;
            Eigen::Vector4f Pw_h{Pw[0],Pw[1],Pw[2],1.0f};
            Eigen::Vector4f Pc =  Twc.inverse() * Pw_h;

            Eigen::Vector3f Cam_bearing_vec = {Pc[0]/Pc[2],Pc[1]/Pc[2],1.0f};//利用旋转进行一个坐标对齐
            Eigen::Vector2f Mp_cam_measure;
            vio::cameras::CameraBase::ProjectionStatus result = camera_ptr->project(Cam_bearing_vec,&Mp_cam_measure);
            if(result == vio::cameras::CameraBase::ProjectionStatus::Successful)//如果投影成功的话
            {
//                std::cout<<"地图点投影下来的:"<<Mp_cam_measure[0]<<" "<<Mp_cam_measure[1]<<std::endl;
                vector<float> query;
                query.push_back(Mp_cam_measure[0]);
                query.push_back(Mp_cam_measure[1]);
                int k = Max_Near_num; //找最多10个最近的帧吧
                std::vector<size_t > near_Vertex;
                std::vector<int> near_idx(k,-1);//找到点的索引
                vector<float> dists(k);

                Kd_database.radiusSearch(query, near_idx, dists,pixel_distance, Max_Near_num,params);

                //遍历每一个附近的特征点,并且遍历这个地图点在多帧中的匹配
                for (int i = 0; i < near_idx.size(); ++i)//convert
                {
                    if(near_idx[i] == -1)
                        break;

                    const cv::Mat &d1 = Descriptors1.row(near_idx[i]);
                    int bestDist=256;
                    for (int l = 0; l < Mp_info[j].second.size(); ++l)//遍历所有的地图点的描述子,取最好的一组
                    {
                        const cv::Mat &d2 = Mp_info[j].second[l];
                        int dist = DescriptorDistance(d1,d2);
                        if(dist < bestDist)
                            bestDist = dist;
                    }
                    if( bestDist < TH_LOW)
                    {
                        //将dist加进Kps1_matches中
                        if(Kps1_matches.count(near_idx[i]))//其他关键帧也有这个地图点的id
                        {
                            if(Kps1_matches.find(near_idx[i])->second.count(bestDist))//有别的地图点也是和这个特征点是这个距离
                            {
                                Kps1_matches.find(near_idx[i])->second.find(bestDist)->second.push_back(j);
                            } else
                            {
                                vector<int> init_idx;
                                init_idx.push_back(j);
                                Kps1_matches.find(near_idx[i])->second.insert(make_pair(bestDist,init_idx));
                            }
                        }
                        else//需要添加新的地图点
                        {
                            vector<int> init_idx;
                            init_idx.push_back(j);
                            map<int,vector<int>> init_kp_idx;
                            init_kp_idx.insert(make_pair(bestDist,init_idx));
                            Kps1_matches.insert(make_pair(near_idx[i],init_kp_idx));
                        }
                    }

                }

            }
        }

        //将特征点存储的显著性强的点加进Mp_matches_dist,Mp_matches中
        {
            auto Kps1_matches_iter = Kps1_matches.begin();
            while (Kps1_matches_iter != Kps1_matches.end())//遍历所有特征点的匹配
            {
                int cur_kps1_idx = Kps1_matches_iter->first;
                if (Kps1_matches_iter->second.begin()->second.size() != 1)//最佳距离对应多个地图点的话,直接放弃所有匹配
                {
                    Kps1_matches_iter++;
                    continue;
                } else if (Kps1_matches_iter->second.size() == 1)//只有一个距离且这个距离也只对应于一个地图点的话,那就是直接对应上了
                {
                    if (Kps1_matches_iter->second.begin()->second.size() == 1)//只有一个特征点和一个地图点匹配,那么就是匹配上了
                    {
                        int cur_dist = Kps1_matches_iter->second.begin()->first;
                        if (Mp_matches_dist.count(Kps1_matches_iter->second.begin()->second[0]))//说明地图点已经和别的特征点匹配上了
                        {
                            int bestDist1 = Mp_matches_dist.find(
                                    Kps1_matches_iter->second.begin()->second[0])->second[0];
                            int bestDist2 = Mp_matches_dist.find(
                                    Kps1_matches_iter->second.begin()->second[0])->second[1];
                            if (cur_dist < bestDist1) {
                                Mp_matches_dist.find(
                                        Kps1_matches_iter->second.begin()->second[0])->second[1] = bestDist1;
                                Mp_matches_dist.find(
                                        Kps1_matches_iter->second.begin()->second[0])->second[0] = cur_dist;
                                Mp_matches.find(Kps1_matches_iter->second.begin()->second[0])->second = cur_kps1_idx;
                            } else if (cur_dist < bestDist2) {
                                Mp_matches_dist.find(
                                        Kps1_matches_iter->second.begin()->second[0])->second[1] = cur_dist;
                            }
                        } else {
                            Mp_matches.insert(
                                    make_pair(Kps1_matches_iter->second.begin()->second[0], Kps1_matches_iter->first));
                            //将地图点id,距离插入
                            vector<int> init_dist;
                            init_dist.push_back(cur_dist);
                            init_dist.push_back(256);
                            Mp_matches_dist.insert(make_pair(Kps1_matches_iter->second.begin()->second[0], init_dist));
                        }
                    }
                } else {
                    auto Kps1_dist_iter = Kps1_matches_iter->second.begin();
                    int cur_dist1 = Kps1_dist_iter->first;
                    Kps1_dist_iter++;
                    int cur_dist2 = Kps1_dist_iter->first;
                    if (static_cast<float>(cur_dist1) <
                        nnRatio * static_cast<float>(cur_dist2))//显著性还不错,将这个加入Mp_matches中
                    {
                        if (Mp_matches_dist.count(Kps1_matches_iter->second.begin()->second[0]))//说明地图点已经和别的特征点匹配上了
                        {
                            int bestDist1 = Mp_matches_dist.find(
                                    Kps1_matches_iter->second.begin()->second[0])->second[0];
                            int bestDist2 = Mp_matches_dist.find(
                                    Kps1_matches_iter->second.begin()->second[0])->second[1];
                            if (cur_dist1 < bestDist1) {
                                Mp_matches_dist.find(
                                        Kps1_matches_iter->second.begin()->second[0])->second[1] = bestDist1;
                                Mp_matches_dist.find(
                                        Kps1_matches_iter->second.begin()->second[0])->second[0] = cur_dist1;
                                Mp_matches.find(Kps1_matches_iter->second.begin()->second[0])->second = cur_kps1_idx;
                            } else if (cur_dist1 < bestDist2) {
                                Mp_matches_dist.find(
                                        Kps1_matches_iter->second.begin()->second[0])->second[1] = cur_dist1;
                            }
                        } else {
                            Mp_matches.insert(
                                    make_pair(Kps1_matches_iter->second.begin()->second[0], Kps1_matches_iter->first));
                            //将地图点id,距离插入
                            vector<int> init_dist;
                            init_dist.push_back(cur_dist1);
                            init_dist.push_back(256);
                            Mp_matches_dist.insert(make_pair(Kps1_matches_iter->second.begin()->second[0], init_dist));
                        }
                    }
                }


                Kps1_matches_iter++;
            }
        }
        assert(Mp_matches.size() == Mp_matches_dist.size());
        {
            auto Mp_matches_iter =  Mp_matches.begin();
            auto Mp_matches_dist_iter =  Mp_matches_dist.begin();
            while (Mp_matches_iter != Mp_matches.end())//遍历所有特征点的匹配
            {
                int kp1_idx = Mp_matches_iter->second;
                int cur_Mp_idx = Mp_matches_iter->first;
                if(matches[kp1_idx] == -2)//一对多的情况直接不考虑吧
                {
                    Mp_matches_iter++;
                    Mp_matches_dist_iter++;
                    continue;
                }
                int dist1 = Mp_matches_dist_iter->second[0];
                int dist2 = Mp_matches_dist_iter->second[1];
                if (static_cast<float>(dist1) <
                    nnRatio * static_cast<float>(dist2))//显著性还不错,应该是匹配的
                {
                    if(matches[kp1_idx] != -1)
                    {
                        matches[kp1_idx] = -2;//说明是一对多的情况,那么这个kp点很危险,还是不要了
                        nmatches--;
                    } else
                    {
                        matches[kp1_idx] = cur_Mp_idx;
                        nmatches++;
                    }
                }


                Mp_matches_iter++;
                Mp_matches_dist_iter++;
            }
        }

        int test_match = 0;
        for (int m = 0; m < matches.size(); ++m)
        {
            if(matches[m] > -1)
                test_match++;

        }
        assert(nmatches == test_match);
        return nmatches;
    }


    // 取出直方图中值最大的三个index, 但是注意如果出现了"一枝独秀"的情况,最后返回的结果中肯呢过和会出现-1
    void FeatureMatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1=0;
        int max2=0;
        int max3=0;

        for(int i=0; i<L; i++)
        {
            const int s = histo[i].size();
            if(s>max1)
            {
                max3=max2;
                max2=max1;
                max1=s;
                ind3=ind2;
                ind2=ind1;
                ind1=i;
            }
            else if(s>max2)
            {
                max3=max2;
                max2=s;
                ind3=ind2;
                ind2=i;
            }
            else if(s>max3)
            {
                max3=s;
                ind3=i;
            }
        }

        // 如果差距太大了,说明次优的非常不好,这里就索性放弃了
        if(max2<0.1f*(float)max1)
        {
            ind2=-1;
            ind3=-1;
        }
        else if(max3<0.1f*(float)max1)
        {
            ind3=-1;
        }
    }

    int FeatureMatcher::DrawMatches(shared_ptr<KeyFrame> f1, shared_ptr<KeyFrame> f2, std::vector<int> &matches)
    {

        cv::Mat img1 = f1->mImg;
        cv::Mat img2 = f2->mImg;

        cv::Mat imTrack;
        int cols = img1.cols;
        cv::hconcat(img1, img2, imTrack);//左右图拼接到一起
        cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);


        for (int i = 0; i < matches.size(); ++i)
        {
            if(matches[i] == -1)
                continue;

            cv::circle(imTrack, cv::Point2f(f1->mloop_KPs[i].pt), 1,
                       cv::Scalar(0, 250, 0), 2);

            cv::circle(imTrack, cv::Point2f(f2->mKPs[matches[i]].pt.x + cols, f2->mKPs[matches[i]].pt.y), 1,
                       cv::Scalar(0, 250, 0), 2);

            cv::line(imTrack, cv::Point2f(f1->mloop_KPs[i].pt),
                     cv::Point2f(f2->mKPs[matches[i]].pt.x + cols, f2->mKPs[matches[i]].pt.y),
                     cv::Scalar(0, 250, 0),
                     1);
        }


        for (auto &feat: f1->mloop_KPs) {
                cv::circle(imTrack, feat.pt, 1, cv::Scalar(0, 0, 250), 2);
        }
        for (auto &feat: f2->mKPs) {
                cv::circle(imTrack, cv::Point2f(feat.pt.x + cols, feat.pt.y), 1, cv::Scalar(0, 0, 250), 2);
        }

        cv::imwrite("/home/wya/ICE-BA-Debug/Loop/loop_img/" + std::to_string(f1->iFrm) + "&&" + std::to_string(f2->iFrm)+".jpg",imTrack);

        return 0;
    }
}
