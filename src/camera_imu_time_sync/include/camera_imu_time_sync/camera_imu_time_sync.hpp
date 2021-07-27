#ifndef _CAMERA_IMU_TIME_SYNC_HPP_
#define _CAMERA_IMU_TIME_SYNC_HPP_

#include "camera_imu_time_sync/cdkf.h"
#include "camera_imu_time_sync/utils.hpp"

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>

#include <std_msgs/Float64.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

class CameraImuTimeSync {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CameraImuTimeSync(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);

 private:

  void imuCallback(const sensor_msgs::ImuConstPtr& msg);

  void imageCallback(const sensor_msgs::ImageConstPtr& msg);

  void setupCDKF();

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  image_transport::ImageTransport it_;

  ros::Subscriber imu_sub_;
  image_transport::Subscriber image_sub_;

  ros::Publisher delta_t_pub_;
  ros::Publisher offset_pub_;
  image_transport::Publisher image_pub_;

  bool stamp_on_arrival_;
  double max_imu_data_age_s_;
  int delay_by_n_frames_;
  double focal_length_;
  bool calc_offset_;

  std::unique_ptr<CDKF> cdkf_;

  IMUList imu_rotations_;
};

#endif