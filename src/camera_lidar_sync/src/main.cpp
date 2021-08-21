#include "camera_lidar_sync/camera_lidar_sync.hpp"


int main(int argc, char** argv)
{
    ros::init(argc, argv, "camera_lidar_sync");

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    ROS_INFO("HELLO ROS, This is camera lidar sync");

    CameraLidarSync camera_lidar_sync(nh, nh_private);

    ros::spin();
    return 0;
}