#ifndef CDKF_TIME_AUTOSYNC_H
#define CDKF_TIME_AUTOSYNC_H

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <Eigen/Eigen>

#include "camera_imu_time_sync/sigma_points.h"
#include "camera_imu_time_sync/state_accessors.h"
#include "camera_imu_time_sync/state_data.h"

template <typename Type>
  using AlignedList = std::list<Type, Eigen::aligned_allocator<Type>>;
  using IMUList = AlignedList<std::pair<ros::Time, Eigen::Quaterniond>>;

class CDKF {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Config {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool verbose = false;
    double mah_threshold = 10.0;

    // initial values
    double inital_delta_t = 0.05;
    double inital_offset = 0.0;

    // initial noise values
    double inital_timestamp_sd = 0.1;
    double inital_delta_t_sd = 0.1;
    double inital_offset_sd = 0.1;

    // measurement noise
    double timestamp_sd = 0.02;
    double angular_velocity_sd = 0.03;

    // process noise
    double delta_t_sd = 0.0001;
    double offset_sd = 0.0001;
  };

  CDKF(const Config& config);

  // make all stored timestamps relative to this one, called periodically to
  // prevent loss in precision
  void rezeroTimestamps(const ros::Time& new_zero_timestamp,
                        bool first_init = false);

  // sync the measured timestamp based on the current filter state
  void getSyncedTimestamp(const ros::Time& received_timestamp,
                        ros::Time* synced_timestamp, double* delta_t, double* offset);

  void predictionUpdate(const ros::Time& received_timestamp);

  void measurementUpdate(const ros::Time& prev_stamp,
                         const ros::Time& current_stamp,
                         const double image_angular_velocity,
                         const IMUList& imu_rotations, const bool calc_offset);

  static void stateToMeasurementEstimate(
      const IMUList& imu_rotations, const ros::Time zero_stamp,
      bool calc_offset, const Eigen::VectorXd& input_state,
      const Eigen::VectorXd& noise,
      Eigen::Ref<Eigen::VectorXd> estimated_measurement);

  static void propergateState(const Eigen::VectorXd& noise,
                              Eigen::Ref<Eigen::VectorXd> current_state);

  static Eigen::Quaterniond getInterpolatedImuAngle(
      const IMUList& imu_rotations, const ros::Time& stamp);

  static double getImuAngleChange(const IMUList& imu_rotations,
                                  const ros::Time& start_stamp,
                                  const ros::Time& end_stamp);

 private:
  ros::Time zero_timestamp_;

  double mah_threshold_;
  bool verbose_;

  Eigen::VectorXd state_;
  Eigen::MatrixXd cov_;

  Eigen::VectorXd prediction_noise_sd_;
  Eigen::VectorXd measurement_noise_sd_;
};

#endif  // STATE_DATA_TIME_AUTOSYNC_H
