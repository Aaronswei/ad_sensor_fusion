#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
using namespace std;

typedef pcl::PointXYZI PointType;

enum class SensorType { VELODYNE, OUSTER, Pandar40P };

class ParamServer
{
public:

    ros::NodeHandle nh;
    std::string PROJECT_NAME;
    std::string robot_id;

    //Topics
    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string gpsTopic;
    string gpsRawTopic;
    string odomResetTopic;
    //Frames
    string lidarFrame;
    string baselinkFrame;
    string odometryFrame;
    string mapFrame;

    // GPS Settings
    std::string gpsFixTopic;
    bool useGpsData;
    bool useGpsPoseinterpolation;
    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;
    bool saveTrajectory;
    // Save pcd
    bool savePCD;
    string savePCDDirectory;

    // Lidar Sensor Configuration
    SensorType sensor;
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;

    // IMU
    bool imuRotEqualBase;
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;
    float imuRPYWeight;
    float imuRate;
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;
    vector<double> extTransVL;
    vector<double> extROTVI;

    Eigen::Matrix3d extRot;
    Eigen::Matrix3d extRotImu2Base;
    Eigen::Matrix3d extRPY;
    Eigen::Matrix3d extRPYInv;
    Eigen::Vector3d extTransLidar2Base;
    Eigen::Vector3d extTrans;
    Eigen::Quaterniond extQRPY;
    Eigen::Quaterniond extQImu2Base;

    float errorThreshold;

    // LOAM
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;
    float mappingLeafSize;

    float z_tollerance; 
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;
    double mappingProcessInterval;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold; 
    float surroundingkeyframeAddingAngleThreshold; 
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;
    
    // Loop closure
    bool  loopClosureEnableFlag;
    float loopClosureFrequency;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    ParamServer()
    {
        nh.param<std::string>("/PROJECT_NAME", PROJECT_NAME, "sam");
        nh.param<std::string>("/robot_id", robot_id, "roboat");

        nh.param<std::string>(PROJECT_NAME + "/pointCloudTopic", pointCloudTopic, "points_raw");
        nh.param<std::string>(PROJECT_NAME + "/imuTopic", imuTopic, "imu_raw");
        nh.param<std::string>(PROJECT_NAME + "/odomTopic", odomTopic, "odometry/imu");
        nh.param<std::string>(PROJECT_NAME + "/gpsTopic", gpsTopic, "odometry/gps");
        nh.param<std::string>(PROJECT_NAME + "/gpsRawTopic", gpsRawTopic, "/localization/odom");
        nh.param<std::string>(PROJECT_NAME + "/odomResetTopic", odomResetTopic, "/reset_lidar_osometry");
        nh.param<std::string>(PROJECT_NAME + "/lidarFrame", lidarFrame, "base_link");
        nh.param<std::string>(PROJECT_NAME + "/baselinkFrame", baselinkFrame, "base_link");
        nh.param<std::string>(PROJECT_NAME + "/odometryFrame", odometryFrame, "odom");
        nh.param<std::string>(PROJECT_NAME + "/mapFrame", mapFrame, "map");

        nh.param<std::string>(PROJECT_NAME + "/gpsFixTopic", gpsFixTopic, "/cgi610/nav_fix");

        nh.param<bool>(PROJECT_NAME + "/saveTrajectory", saveTrajectory, false);
        nh.param<bool>(PROJECT_NAME + "/useGpsData", useGpsData, false);
        nh.param<bool>(PROJECT_NAME + "/useGpsPoseinterpolation", useGpsPoseinterpolation, false);
        nh.param<bool>(PROJECT_NAME + "/useImuHeadingInitialization", useImuHeadingInitialization, false);
        nh.param<bool>(PROJECT_NAME + "/useGpsElevation", useGpsElevation, false);
        nh.param<float>(PROJECT_NAME + "/gpsCovThreshold", gpsCovThreshold, 2.0);
        nh.param<float>(PROJECT_NAME + "/poseCovThreshold", poseCovThreshold, 25.0);

        nh.param<bool>(PROJECT_NAME + "/savePCD", savePCD, false);
        nh.param<std::string>(PROJECT_NAME + "/savePCDDirectory", savePCDDirectory, "/Downloads/LOAM/");

        std::string sensorStr;
        nh.param<std::string>(PROJECT_NAME + "/sensor", sensorStr, "");
        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE;
        }
        else if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;
        }
        else if(sensorStr == "pandar40p")
        {
            sensor = SensorType::Pandar40P;
        }
        else
        {
            ROS_ERROR_STREAM(
                "Invalid sensor type (must be either 'velodyne' or 'ouster'): " << sensorStr);
            ros::shutdown();
        }

        nh.param<int>(PROJECT_NAME + "/N_SCAN", N_SCAN, 16);
        nh.param<int>(PROJECT_NAME + "/Horizon_SCAN", Horizon_SCAN, 1800);
        nh.param<int>(PROJECT_NAME + "/downsampleRate", downsampleRate, 1);
        nh.param<float>(PROJECT_NAME + "/lidarMinRange", lidarMinRange, 1.0);
        nh.param<float>(PROJECT_NAME + "/lidarMaxRange", lidarMaxRange, 1000.0);

        nh.param<bool>(PROJECT_NAME + "/imuRotEqualBase", imuRotEqualBase, false);
        nh.param<float>(PROJECT_NAME + "/imuAccNoise", imuAccNoise, 0.01);
        nh.param<float>(PROJECT_NAME + "/imuGyrNoise", imuGyrNoise, 0.001);
        nh.param<float>(PROJECT_NAME + "/imuAccBiasN", imuAccBiasN, 0.0002);
        nh.param<float>(PROJECT_NAME + "/imuGyrBiasN", imuGyrBiasN, 0.00003);
        nh.param<float>(PROJECT_NAME + "/imuGravity", imuGravity, 9.80511);
        nh.param<float>(PROJECT_NAME + "/imuRPYWeight", imuRPYWeight, 0.01);
        nh.param<float>(PROJECT_NAME + "/imuRate", imuRate, 100);
        
        nh.param<vector<double>>(PROJECT_NAME + "/extrinsicRot", extRotV, vector<double>());
        nh.param<vector<double>>(PROJECT_NAME + "/extrinsicRPY", extRPYV, vector<double>());
        nh.param<vector<double>>(PROJECT_NAME + "/extrinsicTrans", extTransV, vector<double>());
        nh.param<vector<double>>(PROJECT_NAME + "/extrinsicTransLidar2Base", extTransVL, vector<double>());    
        nh.param<vector<double>>(PROJECT_NAME + "/extrinsicRotImu2Base", extROTVI, vector<double>());    
        extTransLidar2Base = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransVL.data(), 3, 1);
        extRotImu2Base  =  Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extROTVI.data(), 3, 3);
        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY);
        extRPYInv = extRPY.inverse();
        extQImu2Base = Eigen::Quaterniond(extRotImu2Base);
        
        nh.param<float>(PROJECT_NAME + "/errorThreshold", errorThreshold, 2.0);
        nh.param<float>(PROJECT_NAME + "/edgeThreshold", edgeThreshold, 0.1);
        nh.param<float>(PROJECT_NAME + "/surfThreshold", surfThreshold, 0.1);
        nh.param<int>(PROJECT_NAME + "/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
        nh.param<int>(PROJECT_NAME + "/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

        nh.param<float>(PROJECT_NAME + "/odometrySurfLeafSize", odometrySurfLeafSize, 0.5);
        nh.param<float>(PROJECT_NAME + "/mappingCornerLeafSize", mappingCornerLeafSize, 0.5);
        nh.param<float>(PROJECT_NAME + "/mappingSurfLeafSize", mappingSurfLeafSize, 0.5);
        nh.param<float>(PROJECT_NAME + "/mappingLeafSize", mappingLeafSize, 0.5);

        nh.param<float>(PROJECT_NAME + "/z_tollerance", z_tollerance, FLT_MAX);
        nh.param<float>(PROJECT_NAME + "/rotation_tollerance", rotation_tollerance, FLT_MAX);

        nh.param<int>(PROJECT_NAME + "/numberOfCores", numberOfCores, 2);
        nh.param<double>(PROJECT_NAME + "/mappingProcessInterval", mappingProcessInterval, 0.15);

        nh.param<float>(PROJECT_NAME + "/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
        nh.param<float>(PROJECT_NAME + "/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        nh.param<float>(PROJECT_NAME + "/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
        nh.param<float>(PROJECT_NAME + "/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

        nh.param<bool>(PROJECT_NAME + "/loopClosureEnableFlag", loopClosureEnableFlag, false);
        nh.param<float>(PROJECT_NAME + "/loopClosureFrequency", loopClosureFrequency, 1.0);
        nh.param<int>(PROJECT_NAME + "/surroundingKeyframeSize", surroundingKeyframeSize, 50);
        nh.param<float>(PROJECT_NAME + "/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh.param<float>(PROJECT_NAME + "/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        nh.param<int>(PROJECT_NAME + "/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        nh.param<float>(PROJECT_NAME + "/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

        nh.param<float>(PROJECT_NAME + "/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>(PROJECT_NAME + "/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>(PROJECT_NAME + "/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

        usleep(100);
    }

    sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
    {
        sensor_msgs::Imu imu_out = imu_in;

        // rotate acceleration
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRotImu2Base * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRotImu2Base * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();
        if(!imuRotEqualBase){
            // rotate roll pitch yaw
            Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
            Eigen::Quaterniond q_final = q_from * extQImu2Base;
            imu_out.orientation.x = q_final.x();
            imu_out.orientation.y = q_final.y();
            imu_out.orientation.z = q_final.z();
            imu_out.orientation.w = q_final.w();

            if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
            {
                ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
                ros::shutdown();
            }
        }
        return imu_out;
    }
};


sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

template<typename T>
double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}


template<typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}


template<typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}


template<typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}


float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

#endif
