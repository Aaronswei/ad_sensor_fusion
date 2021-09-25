#include <sensor_msgs/NavSatFix.h>
#include "utility.h"
class GpsConvert : public ParamServer{
private:
    ros::Subscriber subImu;
    ros::Subscriber subOdom;
    ros::Subscriber subGps;
    ros::Publisher pubOdom;
    ros::Publisher pubOdomPath;
    nav_msgs::Path odomPath;
    std::vector<double> vecTimeGps;
    std::vector<bool> vecGpsFlag;
    bool publishGpsData;
    ofstream outFileRtkPose;
public:
    GpsConvert(): 

    //订阅发布数据
    publishGpsData(false){
        subOdom     = nh.subscribe<nav_msgs::Odometry> (gpsRawTopic, 2000, &GpsConvert::odometryHandler,  this, ros::TransportHints().tcpNoDelay());
        if(gpsRawTopic != gpsTopic){
            subGps      = nh.subscribe<sensor_msgs::NavSatFix>(gpsFixTopic, 2000, &GpsConvert::gpsHandler,  this, ros::TransportHints().tcpNoDelay());
            pubOdom     = nh.advertise<nav_msgs::Odometry> (gpsTopic,2000);
            publishGpsData = true;
        }
        pubOdomPath = nh.advertise<nav_msgs::Path> ("/odom/path", 1);
        odomPath.header.frame_id = "/map";
        //是否保留轨迹
        if(saveTrajectory){
            outFileRtkPose.open("/data_work/bag/rtk_pose.txt",ios::binary | ios::trunc| ios::in | ios::out);
        }
    }
    ~GpsConvert(){
        if(saveTrajectory){
            outFileRtkPose.close();
        }
    }


    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg){
        int index = -1;
        double time = 0.1;
        double timeStamp = odomMsg->header.stamp.toSec();
        if(publishGpsData){
            for (int i = 0; i < vecTimeGps.size(); ++i)
            {
                double time_diff = timeStamp - vecTimeGps[i];
                if(time_diff < 0 && abs(time_diff) < time){
                    index = i;
                    break;
                }
                else if(time_diff < 0){
                    break;
                }
                else if(time_diff < time){
                    time = time_diff;
                    index = i;
                }
            }
            if (-1 != index)
            {
                nav_msgs::Odometry odom = *odomMsg;
                if (vecGpsFlag[index])
                {
                    odom.pose.covariance[0] = 0.00001;
                    odom.pose.covariance[7] = 0.00001;
                    odom.pose.covariance[14] = 0.00001;

                }
                else{
                    odom.pose.covariance[0] = 999;
                    odom.pose.covariance[7] = 999;
                    odom.pose.covariance[14] = 999;
                }
                pubOdom.publish(odom);
                vecGpsFlag.erase(vecGpsFlag.begin(), vecGpsFlag.begin() + index + 1);
                vecTimeGps.erase(vecTimeGps.begin(),vecTimeGps.begin()+index+1);
            }
        }
        
        static double last_path_time = -1;
        if (timeStamp - last_path_time > 0.1)
        {
            last_path_time = timeStamp;
            geometry_msgs::PoseStamped pose_odm;
            pose_odm.header.stamp = odomMsg->header.stamp;
            pose_odm.header.frame_id = "/map";
            pose_odm.pose.position = odomMsg->pose.pose.position;
            pose_odm.pose.orientation = odomMsg->pose.pose.orientation;
            odomPath.poses.push_back(pose_odm);
        
            if(pubOdomPath.getNumSubscribers() != 0){
                odomPath.header.stamp = odomMsg->header.stamp;
                odomPath.header.frame_id = "/map";
                pubOdomPath.publish(odomPath);
            }
        }
        if(saveTrajectory){
            outFileRtkPose << std::setprecision(19) << odomMsg->header.stamp.toSec() << " " << odomMsg->pose.pose.position.x << " "
            << odomMsg->pose.pose.position.y << " " << odomMsg->pose.pose.position.z << " " << odomMsg->pose.pose.orientation.x << " "
            << odomMsg->pose.pose.orientation.y << " " << odomMsg->pose.pose.orientation.z << " " << odomMsg->pose.pose.orientation.w << "\n";
        }
    }

    void gpsHandler(const sensor_msgs::NavSatFix::ConstPtr& gpsMsg){
        sensor_msgs::NavSatFix gpsData = *gpsMsg;
        vecTimeGps.push_back(gpsMsg->header.stamp.toSec());
        vecGpsFlag.push_back((sensor_msgs::NavSatStatus::STATUS_FIX == gpsMsg->status.status));
    }
};


int main(int argc, char** argv){
    ros::init(argc, argv, "lidar");
    GpsConvert gps_tool;
    ROS_INFO("\033[1;32m----> GPS Converter Started.\033[0m");
    ros::spin();
    return 0;
}