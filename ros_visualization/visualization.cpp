//
// Created by wya on 2020/3/1.
//

#include "visualization.h"

DECLARE_bool(stereo);
DECLARE_string(gba_result_path);
extern Eigen::Matrix4d T_Cl_Cr_d,Tbc0;;
nav_msgs::Path lba_path,GT_path;
ros::Publisher pub_lba_path,pub_gba_path,pub_GT_path,pub_camera_pose,pub_camera_pose_visual,pub_imu_pose,
pub_point_cloud,pub_image_track,pub_loop_campose,pub_loop_camera_pose_visual,pub_GT_pose,pub_GT_camera_pose_visual;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
CameraPoseVisualization loopcameraposevisual(0, 0, 1, 1);
CameraPoseVisualization GTcameraposevisual(1, 1, 0, 1);
void registerPub(ros::NodeHandle &n)
{
    pub_lba_path = n.advertise<nav_msgs::Path>("lba_path", 1000);
    pub_gba_path = n.advertise<nav_msgs::Path>("gba_path", 1000);
    pub_GT_path = n.advertise<nav_msgs::Path>("GT_path", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    pub_camera_pose = n.advertise<nav_msgs::Odometry>("camera_pose", 1000);
    pub_imu_pose = n.advertise<nav_msgs::Odometry>("imu_pose", 1000);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_GT_pose = n.advertise<nav_msgs::Odometry>("GT_pose", 1000);
    pub_GT_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("GT_camera_pose_visual", 1000);
    pub_loop_campose = n.advertise<nav_msgs::Odometry>("loop_camera_pose", 1000);
    pub_loop_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("loop_camera_pose_visual", 1000);

    pub_image_track = n.advertise<sensor_msgs::Image>("image_track", 1000);

    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);
    GTcameraposevisual.setScale(0.1);
    GTcameraposevisual.setLineWidth(0.01);
    loopcameraposevisual.setScale(0.1);
    loopcameraposevisual.setLineWidth(0.01);
}

void pubUpdatePointClouds(const std::vector<IBA::Point3D> & UpdatePointClouds,double t)
{
    sensor_msgs::PointCloud point_cloud;
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(t);
    point_cloud.header = header;
    for (int i = 0; i < UpdatePointClouds.size(); ++i)
    {
        geometry_msgs::Point32 p;
        p.x = UpdatePointClouds[i].X[0];
        p.y = UpdatePointClouds[i].X[1];
        p.z = UpdatePointClouds[i].X[2];
        point_cloud.points.push_back(p);
    }
    pub_point_cloud.publish(point_cloud);
}

void pubLoopCamPose(const Eigen::Matrix4d &Twc0)
{
    Eigen::Matrix3d Rwc0 = Twc0.block<3,3>(0,0);
    Eigen::Vector3d twc0 = Twc0.block<3,1>(0,3);
    Eigen::Quaterniond qwc0(Rwc0);
    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time::now();
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = twc0.x();
    odometry.pose.pose.position.y = twc0.y();
    odometry.pose.pose.position.z = twc0.z();
    odometry.pose.pose.orientation.x = qwc0.x();
    odometry.pose.pose.orientation.y = qwc0.y();
    odometry.pose.pose.orientation.z = qwc0.z();
    odometry.pose.pose.orientation.w = qwc0.w();


    pub_loop_campose.publish(odometry);


    loopcameraposevisual.reset();
    loopcameraposevisual.add_pose(twc0, qwc0);
    if(FLAGS_stereo)
    {
        Eigen::Matrix4d Twc1 = Twc0 * T_Cl_Cr_d;
        Eigen::Matrix3d Rwc1 = Twc1.block<3,3>(0,0);
        Eigen::Vector3d twc1 = Twc1.block<3,1>(0,3);
        Eigen::Quaterniond qwc1(Rwc1);

        loopcameraposevisual.add_pose(twc1, qwc1);
    }
    loopcameraposevisual.publish_by(pub_loop_camera_pose_visual, odometry.header);
}

void pubTrackImage(const cv::Mat &cur_l_img, const cv::Mat &cur_r_img ,std::vector<cv::KeyPoint> &pre_l_kp,
                   std::vector<cv::KeyPoint> &cur_l_kp, std::vector<cv::KeyPoint> &cur_r_kp,double t)
{

    cv::Mat imTrack;
    if(FLAGS_stereo)
    {
        int cols = cur_l_img.cols;
        cv::hconcat(cur_l_img, cur_r_img, imTrack);//左右图拼接到一起
        cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);

        std::vector<std::pair<cv::Point2f,cv::Point2f>> stereo_matches;

        std::vector<cv::KeyPoint>::iterator leftPt_iter = cur_l_kp.begin();
        std::vector<cv::KeyPoint>::iterator rightPt_iter = cur_r_kp.begin();
        while(leftPt_iter != cur_l_kp.end() && rightPt_iter != cur_r_kp.end())
        {
            if(leftPt_iter->class_id == rightPt_iter->class_id)
            {
                stereo_matches.push_back(std::make_pair(leftPt_iter->pt,rightPt_iter->pt));
                leftPt_iter++;rightPt_iter++;
            }
            else if(leftPt_iter->class_id < rightPt_iter->class_id)
                leftPt_iter++;
            else
                rightPt_iter++;
        }


        for (size_t j = 0; j < stereo_matches.size(); j++)
        {
            //左图画出特征点
            cv::Point2f leftPt = stereo_matches[j].first;

            cv::circle(imTrack, leftPt, 2, cv::Scalar(0, 0, 255 ), 2);
            //右图画出特征点
            cv::Point2f rightPt = stereo_matches[j].second;
            rightPt.x += cols;//因为拼接到一起所以右图特征点x坐标应该加上左图长度
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);

        }
    }
    else
    {
        imTrack = cur_l_img.clone();
        cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);
    }


    std::vector<std::pair<cv::Point2f,cv::Point2f>> LK_matches;
    std::vector<cv::KeyPoint>::iterator prePt_iter = pre_l_kp.begin();
    std::vector<cv::KeyPoint>::iterator curPt_iter = cur_l_kp.begin();

    while(prePt_iter != pre_l_kp.end() && curPt_iter != cur_l_kp.end())
    {
        if(prePt_iter->class_id == curPt_iter->class_id)
        {
            LK_matches.push_back(std::make_pair(prePt_iter->pt,curPt_iter->pt));
            prePt_iter++;curPt_iter++;
        }
        else if(prePt_iter->class_id < curPt_iter->class_id)
            prePt_iter++;
        else
            curPt_iter++;
    }

    for (int i = 0; i < LK_matches.size(); ++i) {
        cv::arrowedLine(imTrack, LK_matches[i].second,
                        LK_matches[i].first, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }

    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(t);
    sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "rgb8", imTrack).toImageMsg();
    pub_image_track.publish(imgTrackMsg);
}

void pubGTCamPose(const Eigen::Matrix4d &Twc0,double t)
{
    Eigen::Matrix3d Rwc0 = Twc0.block<3,3>(0,0);
    Eigen::Vector3d twc0 = Twc0.block<3,1>(0,3);
    Eigen::Quaterniond qwc0(Rwc0);
    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(t);
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = twc0.x();
    odometry.pose.pose.position.y = twc0.y();
    odometry.pose.pose.position.z = twc0.z();
    odometry.pose.pose.orientation.x = qwc0.x();
    odometry.pose.pose.orientation.y = qwc0.y();
    odometry.pose.pose.orientation.z = qwc0.z();
    odometry.pose.pose.orientation.w = qwc0.w();

    pub_GT_pose.publish(odometry);

    GTcameraposevisual.reset();
    GTcameraposevisual.add_pose(twc0, qwc0);
    if(FLAGS_stereo)
    {
        Eigen::Matrix4d Twc1 = Twc0 * T_Cl_Cr_d;
        Eigen::Matrix3d Rwc1 = Twc1.block<3,3>(0,0);
        Eigen::Vector3d twc1 = Twc1.block<3,1>(0,3);
        Eigen::Quaterniond qwc1(Rwc1);

        GTcameraposevisual.add_pose(twc1, qwc1);
    }
    GTcameraposevisual.publish_by(pub_GT_camera_pose_visual, odometry.header);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = odometry.header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose = odometry.pose.pose;
    GT_path.header = odometry.header;
    GT_path.header.frame_id = "world";
    GT_path.poses.push_back(pose_stamped);
    pub_GT_path.publish(GT_path);
}

void pubTF(const Eigen::Matrix4d & Twc0, double t)
{
    std_msgs::Header header;
    header.stamp= ros::Time(t);
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;


    Eigen::Matrix4d Twb0 = Twc0 * Tbc0.inverse();
    Eigen::Matrix3d Rwb0 = Twb0.block<3,3>(0,0);
    Eigen::Vector3d twb0 = Twb0.block<3,1>(0,3);
    Eigen::Quaterniond qwb0(Rwb0);

    transform.setOrigin(tf::Vector3(twb0(0),
                                    twb0(1),
                                    twb0(2)));
    q.setW(qwb0.w());
    q.setX(qwb0.x());
    q.setY(qwb0.y());
    q.setZ(qwb0.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));


    Eigen::Matrix3d Rbc0 = Tbc0.block<3,3>(0,0);
    Eigen::Vector3d tbc0 = Tbc0.block<3,1>(0,3);
    Eigen::Quaterniond qbc0(Rbc0);
    // camera frame
    transform.setOrigin(tf::Vector3(tbc0.x(),
                                    tbc0.y(),
                                    tbc0.z()));
    q.setW(qbc0.w());
    q.setX(qbc0.x());
    q.setY(qbc0.y());
    q.setZ(qbc0.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));

}

void pubLatestCameraPose(const Eigen::Matrix4d & Twc0, const Eigen::Vector3d &V, double t)
{

    Eigen::Matrix3d Rwc0 = Twc0.block<3,3>(0,0);
    Eigen::Vector3d twc0 = Twc0.block<3,1>(0,3);
    Eigen::Quaterniond qwc0(Rwc0);
    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(t);
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = twc0.x();
    odometry.pose.pose.position.y = twc0.y();
    odometry.pose.pose.position.z = twc0.z();
    odometry.pose.pose.orientation.x = qwc0.x();
    odometry.pose.pose.orientation.y = qwc0.y();
    odometry.pose.pose.orientation.z = qwc0.z();
    odometry.pose.pose.orientation.w = qwc0.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();

    pub_camera_pose.publish(odometry);
    Eigen::Matrix4d Twb0 = Twc0 * Tbc0.inverse();

    {

        Eigen::Matrix3d Rwb0 = Twb0.block<3,3>(0,0);
        Eigen::Vector3d twb0 = Twb0.block<3,1>(0,3);
        Eigen::Quaterniond qwb0(Rwb0);
        nav_msgs::Odometry IMUodometry;
        IMUodometry.header.stamp = ros::Time(t);
        IMUodometry.header.frame_id = "world";
        IMUodometry.pose.pose.position.x = twb0.x();
        IMUodometry.pose.pose.position.y = twb0.y();
        IMUodometry.pose.pose.position.z = twb0.z();
        IMUodometry.pose.pose.orientation.x = qwb0.x();
        IMUodometry.pose.pose.orientation.y = qwb0.y();
        IMUodometry.pose.pose.orientation.z = qwb0.z();
        IMUodometry.pose.pose.orientation.w = qwb0.w();
        pub_imu_pose.publish(IMUodometry);
    }

    cameraposevisual.reset();
    cameraposevisual.add_pose(twc0, qwc0);
    if(FLAGS_stereo)
    {
        Eigen::Matrix4d Twc1 = Twc0 * T_Cl_Cr_d;
        Eigen::Matrix3d Rwc1 = Twc1.block<3,3>(0,0);
        Eigen::Vector3d twc1 = Twc1.block<3,1>(0,3);
        Eigen::Quaterniond qwc1(Rwc1);

        cameraposevisual.add_pose(twc1, qwc1);
    }
    cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = odometry.header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose = odometry.pose.pose;
    lba_path.header = odometry.header;
    lba_path.header.frame_id = "world";
    lba_path.poses.push_back(pose_stamped);
    pub_lba_path.publish(lba_path);

}

void pubKFsPose(const std::vector<std::pair<double,Eigen::Matrix4d>> & kf_poses)
{
    if(!FLAGS_gba_result_path.empty())
    {
        std::ofstream cleanfoutC(FLAGS_gba_result_path, std::ios::out);
        cleanfoutC.close();
        std::ofstream foutC(FLAGS_gba_result_path, std::ios::app);
        foutC.setf(std::ios::fixed, std::ios::floatfield);
        for (int i = 0; i < kf_poses.size(); ++i)
        {
            Eigen::Matrix3d Rwc0 = kf_poses[i].second.block<3,3>(0,0);
            Eigen::Vector3d twc0 = kf_poses[i].second.block<3,1>(0,3);
            Eigen::Quaterniond qwc0(Rwc0);
            foutC.precision(0);
            foutC << kf_poses[i].first * 1e9 << " ";
            foutC.precision(6);
            foutC << twc0.x() << " "
                  << twc0.y() << " "
                  << twc0.z() << " "
                  << qwc0.x() << " "
                  << qwc0.y() << " "
                  << qwc0.z() << " "
                  << qwc0.w() << std::endl;
            
        }
        foutC.close();
    }


    nav_msgs::Path gba_path;
    for (int i = 0; i < kf_poses.size(); ++i)
    {
        Eigen::Matrix3d Rwc0 = kf_poses[i].second.block<3,3>(0,0);
        Eigen::Vector3d twc0 = kf_poses[i].second.block<3,1>(0,3);
        Eigen::Quaterniond qwc0(Rwc0);
        nav_msgs::Odometry odometry;
        odometry.header.stamp = ros::Time(kf_poses[i].first);
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = twc0.x();
        odometry.pose.pose.position.y = twc0.y();
        odometry.pose.pose.position.z = twc0.z();
        odometry.pose.pose.orientation.x = qwc0.x();
        odometry.pose.pose.orientation.y = qwc0.y();
        odometry.pose.pose.orientation.z = qwc0.z();
        odometry.pose.pose.orientation.w = qwc0.w();
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = odometry.header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        gba_path.header = odometry.header;
        gba_path.header.frame_id = "world";
        gba_path.poses.push_back(pose_stamped);
        pub_gba_path.publish(gba_path);

    }
}