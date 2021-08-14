#include "camera_imu_time_sync/camera_imu_time_sync.hpp"
#include "camera_imu_time_sync/utils.hpp"

#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>


CameraImuTimeSync::CameraImuTimeSync(const ros::NodeHandle& nh,
                           const ros::NodeHandle& nh_private)
    : nh_(nh),
      nh_private_(nh_private),
      it_(nh_private_),
      stamp_on_arrival_(false),
      max_imu_data_age_s_(2.0),
      delay_by_n_frames_(5),
      focal_length_(460.0),
      calc_offset_(true) {
  nh_private_.param("stamp_on_arrival", stamp_on_arrival_, stamp_on_arrival_);
  nh_private_.param("max_imu_data_age_s", max_imu_data_age_s_, max_imu_data_age_s_);
  nh_private_.param("delay_by_n_frames", delay_by_n_frames_, delay_by_n_frames_);
  nh_private_.param("focal_length", focal_length_, focal_length_);
  nh_private_.param("calc_offset", calc_offset_, calc_offset_);

  setupCDKF();
  
  constexpr int kImageQueueSize = 10;
  constexpr int kImuQueueSize = 100;
  constexpr int kFloatQueueSize = 100;

  imu_sub_ = nh_private_.subscribe("/cgi610/imu", kImuQueueSize, &CameraImuTimeSync::imuCallback, this);
std::cout << __LINE__ << " " << __FILE__ << " imu correct"  << std::endl;
  image_sub_ = it_.subscribe("/raw_image", kImageQueueSize, &CameraImuTimeSync::imageCallback, this);
  std::cout << __LINE__ << " " << __FILE__ << " image subscribe"  << std::endl;
  image_pub_ = it_.advertise("output/image", kImageQueueSize);
std::cout << __LINE__ << " " << __FILE__ << " image publish "  << std::endl;

  delta_t_pub_ = nh_private_.advertise<std_msgs::Float64>("delta_t", kFloatQueueSize);

  if (calc_offset_) {
    offset_pub_ = nh_private_.advertise<std_msgs::Float64>("offset", kFloatQueueSize);
  }
}

void CameraImuTimeSync::setupCDKF() {
  CDKF::Config config;

  nh_private_.param("verbose", config.verbose, config.verbose);
  nh_private_.param("mah_threshold", config.mah_threshold, config.mah_threshold);

  nh_private_.param("inital_delta_t", config.inital_delta_t, config.inital_delta_t);
  nh_private_.param("inital_offset", config.inital_offset, config.inital_offset);

  nh_private_.param("inital_delta_t_sd", config.inital_delta_t_sd, config.inital_delta_t_sd);
  nh_private_.param("inital_offset_sd", config.inital_offset_sd, config.inital_offset_sd);

  nh_private_.param("timestamp_sd", config.timestamp_sd, config.timestamp_sd);

  nh_private_.param("delta_t_sd", config.delta_t_sd, config.delta_t_sd);
  nh_private_.param("offset_sd", config.offset_sd, config.offset_sd);

  cdkf_ = std::unique_ptr<CDKF>(new CDKF(config));

}


void CameraImuTimeSync::imuCallback(const sensor_msgs::ImuConstPtr& msg) {
  static sensor_msgs::Imu prev_msg;
  static bool first_msg = true;
  
  if (first_msg) {
    first_msg = false;
    imu_rotations_.emplace_back(msg->header.stamp, Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
    prev_msg = *msg;
    std::cout << __LINE__ << " " << __FILE__ << " " << imu_rotations_.size() << std::endl;
    return;
  }

  if (prev_msg.header.stamp >= msg->header.stamp) {
    ROS_WARN(
        "Your imu messages are not monotonically increasing, expect garbage results.");
  }

  // integrate imu reading
  double half_delta_t = (msg->header.stamp - prev_msg.header.stamp).toSec() / 2.0;

  Eigen::Quaterniond delta_angle =
      Eigen::AngleAxisd(half_delta_t * (msg->angular_velocity.x + prev_msg.angular_velocity.x),
                        Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(half_delta_t * (msg->angular_velocity.y + prev_msg.angular_velocity.y),
                        Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(half_delta_t * (msg->angular_velocity.z + prev_msg.angular_velocity.z),
                        Eigen::Vector3d::UnitZ());

  imu_rotations_.emplace_back(
      prev_msg.header.stamp + ros::Duration(half_delta_t),
      imu_rotations_.back().second * delta_angle);
  imu_rotations_.back().second.normalize();

  // clear old data
  while ((imu_rotations_.back().first - imu_rotations_.front().first).toSec() > max_imu_data_age_s_) {
    imu_rotations_.pop_front();
  }

  prev_msg = *msg;
}

void CameraImuTimeSync::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
  ros::Time stamp;
  if (stamp_on_arrival_) {
    stamp = ros::Time::now();
  } else {
    stamp = msg->header.stamp;
  }

  static std::list<cv_bridge::CvImage> images;
  cv_bridge::CvImagePtr image = cv_bridge::toCvCopy(msg, "mono8");

  // fire the image back out with minimal lag
  if (images.size() >= (delay_by_n_frames_ - 1)) {
    std_msgs::Float64 delta_t, offset;
    cdkf_->getSyncedTimestamp(stamp, &(image->header.stamp), &(delta_t.data), &(offset.data));
    image_pub_.publish(image->toImageMsg());
    delta_t_pub_.publish(delta_t);
    if (calc_offset_) {
      offset_pub_.publish(offset);
    }
  }

  image->header.stamp = stamp;

  // delay by a few messages to ensure IMU messages span needed range
  images.push_back(*image);
  std::cout << "delay by a few messages "<< images.size() << std::endl;

  if (images.size() < delay_by_n_frames_) {
    cdkf_->rezeroTimestamps(images.front().header.stamp, true);
    return;
  }

  if (calc_offset_ && (imu_rotations_.size() < 2)) {
    return;
  }
  std::cout << __LINE__ << " " << __FILE__ << " " << imu_rotations_.size() << std::endl;

  double image_angle = 0.0;
  if (calc_offset_) {
    image_angle = calcAngleBetweenImages(images.begin()->image, std::next(images.begin())->image, focal_length_);
  }
  std::cout << __LINE__ << " " << __FILE__ << " " << imu_rotations_.size() << std::endl;

  // actually run filter
  cdkf_->predictionUpdate(std::next(images.begin())->header.stamp);
  cdkf_->measurementUpdate(images.begin()->header.stamp, std::next(images.begin())->header.stamp, image_angle, imu_rotations_, calc_offset_);

  images.pop_front();
}



int main(int argc, char** argv) {
  ros::init(argc, argv, "camera_imu_time_sync");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  ROS_INFO("HELLO ROS");
  CameraImuTimeSync camera_imu_time_sync(nh, nh_private);

  ros::spin();

  return 0;
}
