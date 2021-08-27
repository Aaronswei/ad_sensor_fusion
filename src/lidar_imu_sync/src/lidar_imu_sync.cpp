#include "lidar_imu_sync/lidar_imu_sync.hpp"

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

using namespace std;

LidarIMUSync::LidarIMUSync(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
    : nh_(nh),
      nh_private_(nh_private)
{
    lidar_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 10, &LidarIMUSync::lidarCallback, this);
    imu_sub_ = nh_.subscribe<sensor_msgs::Imu>("/cgi610/imu", 100, &LidarIMUSync::imuCallback, this);

}

void LidarIMUSync::lidarCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    //读取激光数据并进行解析
    CloudT::Ptr cloud(new CloudT);
    pcl::fromROSMsg(*msg, *cloud);
    LidarData data;
    data.cloud = cloud;
    data.stamp = msg->header.stamp.toSec();
    //将激光雷达数据追加到标定系统中
    caliber_.addLidarData(data);

    // 解析IMU数据
    while (imu_buffer_.size() != 0)
    {
        ImuData data;
        //计算IMU的加速度信息
        data.acc = Eigen::Vector3d(imu_buffer_.front()->linear_acceleration.x,
                                   imu_buffer_.front()->linear_acceleration.y,
                                   imu_buffer_.front()->linear_acceleration.z);
        //IMU的角速度信息
        data.gyr = Eigen::Vector3d(imu_buffer_.front()->angular_velocity.x,
                                   imu_buffer_.front()->angular_velocity.y,
                                   imu_buffer_.front()->angular_velocity.z);
        //IMU的姿态信息
        data.rot = Eigen::Quaterniond(imu_buffer_.front()->orientation.w,
                                      imu_buffer_.front()->orientation.x,
                                      imu_buffer_.front()->orientation.y,
                                      imu_buffer_.front()->orientation.z);
        data.stamp = imu_buffer_.front()->header.stamp.toSec();
        //将IMU解析后的加速度、角速度、姿态信息追加到标定数据中
        caliber_.addImuData(data);
        //将追加后的数据从imu队列中删除掉
        imu_buffer_.pop();
    }
    // calib
    //执行标定函数
    Eigen::Vector3d rpy = caliber_.calib();
    //沿Z轴的旋转向量、沿Y轴的旋转向量、沿X轴的旋转向量
    Eigen::Matrix3d rot = Eigen::Matrix3d(Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX()));
    cout << "result euler angle(RPY) : " << rpy[0] << " " << rpy[1] << " " << rpy[2] << endl;
    cout << "result extrinsic rotation matrix : " << endl;
    cout << rot << endl;
}

void LidarIMUSync::imuCallback(const sensor_msgs::ImuConstPtr &msg)
{
    imu_buffer_.push(msg);
}

