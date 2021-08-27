#include <ros/ros.h>
#include <iostream>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <time.h>

#include "lidar_imu_sync/lidar_imu_calib.hpp"

using namespace std;

class LidarIMUSync{
    public:

        LidarIMUSync(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
        void lidarCallback(const sensor_msgs::PointCloud2ConstPtr &msg);
        void imuCallback(const sensor_msgs::ImuConstPtr &msg);
    private:
        queue<sensor_msgs::PointCloud2ConstPtr> lidar_buffer_;
        queue<sensor_msgs::ImuConstPtr> imu_buffer_;

        ros::NodeHandle nh_;
        ros::NodeHandle nh_private_;
        ros::Subscriber lidar_sub_;
        ros::Subscriber imu_sub_;
            // initialize caliber
        LidarIMUCalib caliber_;

};