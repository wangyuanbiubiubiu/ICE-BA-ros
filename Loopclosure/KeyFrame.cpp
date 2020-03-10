//
// Created by wya on 2020/3/5.
//

#include "KeyFrame.h"

namespace LC
{


    unsigned long KeyFrame::nextId = 0;



    KeyFrame::KeyFrame(int iFrm_,  std::vector<cv::KeyPoint> &kps,  cv::Mat &Descriptors,
                       std::vector<cv::KeyPoint> & loop_kps, cv::Mat & loop_Descriptors,cv::Mat &img) :
            iFrm(iFrm_),mKPs(kps),mDescriptors(Descriptors),mImg(img),mloop_KPs(loop_kps),mloop_Descriptors(loop_Descriptors)
    {
        id = nextId++;
    }

    KeyFrame::KeyFrame(int iFrm_,  std::vector<cv::KeyPoint> &kps,  cv::Mat &Descriptors,
                       std::vector<cv::KeyPoint> & loop_kps, cv::Mat & loop_Descriptors) :
            iFrm(iFrm_),mKPs(kps),mDescriptors(Descriptors),mloop_KPs(loop_kps),mloop_Descriptors(loop_Descriptors)
    {
        id = nextId++;
    }

    //将描述子转换为描述子向量，其实本质上是cv:Mat->std:vector
    std::vector<cv::Mat> KeyFrame::toDescriptorVector(const cv::Mat &Descriptors)
    {
        //存储转换结果的向量
        std::vector<cv::Mat> vDesc;

        //创建保留空间
        vDesc.reserve(Descriptors.rows);
        //对于每一个特征点的描述子
        for (int j=0;j<Descriptors.rows;j++)
            //从描述子这个矩阵中抽取出来存到向量中
            vDesc.push_back(Descriptors.row(j));

        //返回转换结果
        return vDesc;
    }

    void KeyFrame::ComputeBoW(std::shared_ptr<DBoW3::Vocabulary> voc) {
        if(mBowVec.empty() || mFeatVec.empty())
        {
            vector<cv::Mat> vCurrentDesc = toDescriptorVector(mDescriptors);

            voc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }

        if(mloop_BowVec.empty() || mloop_FeatVec.empty())//只用来搜索,这个是没有3d点对应的
        {
            vector<cv::Mat> vCurrentDesc = toDescriptorVector(mloop_Descriptors);

            voc->transform(vCurrentDesc, mloop_BowVec, mloop_FeatVec, 4);
        }
    }


// 获取当前关键帧的具体的地图点
    vector<shared_ptr<MapPoint>> KeyFrame::GetMapPointMatches()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpMapPoints;
    }


}
