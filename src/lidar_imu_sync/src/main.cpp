#include "lidar_imu_sync/lidar_imu_sync.hpp"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_imu_sync");
    ros::NodeHandle nh, nh_private("~");

    ROS_INFO("HELLO ROS, This is lidar imu sync");

    LidarIMUSync lidar_imu_sync(nh, nh_private);

    ros::spin();
    return 0;
}