#include "camera_imu_time_sync/cdkf.h"

CDKF::CDKF(const Config& config) {
  state_.resize(kStateSize, 1);
  accessS(state_, DELTA_T).array() = config.inital_delta_t;
  accessS(state_, OFFSET).array() = config.inital_offset;

  Eigen::MatrixXd inital_sd(kStateSize, 1);
  accessS(inital_sd, STATE_TIMESTAMP).array() = config.inital_timestamp_sd;
  accessS(inital_sd, DELTA_T).array() = config.inital_delta_t_sd;
  accessS(inital_sd, OFFSET).array() = config.inital_offset_sd;

  cov_.resize(kStateSize, kStateSize);
  cov_.setZero();
  cov_.diagonal() = inital_sd.array() * inital_sd.array();

  // set noise sd
  prediction_noise_sd_.resize(kStateSize, 1);
  accessS(prediction_noise_sd_, DELTA_T).array() = config.delta_t_sd;
  accessS(prediction_noise_sd_, OFFSET).array() = config.offset_sd;

  measurement_noise_sd_.resize(kMeasurementSize, 1);
  accessM(measurement_noise_sd_, MEASURED_TIMESTAMP).array() =
      config.timestamp_sd;
  accessM(measurement_noise_sd_, ANGULAR_VELOCITY).array() =
      config.angular_velocity_sd;

  std::cout << __FILE__ << " " <<__LINE__ << std::endl;
  mah_threshold_ = config.mah_threshold;
  verbose_ = config.verbose;
}

// make all stored timestamps relative to this one, called periodically to
// prevent loss in precision
void CDKF::rezeroTimestamps(const ros::Time& new_zero_timestamp,
                            bool first_init) {
  if (!first_init) {
    double time_diff = (new_zero_timestamp - zero_timestamp_).toSec();
    accessS(state_, STATE_TIMESTAMP).array() -= time_diff;
  }

  zero_timestamp_ = new_zero_timestamp;
}

// sync the measured timestamp based on the current filter state
void CDKF::getSyncedTimestamp(const ros::Time& received_timestamp,
                              ros::Time* synced_timestamp, double* delta_t, double* offset) {
  *synced_timestamp =
      zero_timestamp_ + ros::Duration(accessS(state_, STATE_TIMESTAMP)[0]);

  *delta_t = accessS(state_, DELTA_T)[0];
  *offset = accessS(state_, OFFSET)[0];

  // account for sync being some frames behind
  int num_frames =
      std::round((received_timestamp - *synced_timestamp).toSec() / *delta_t);

  if (std::abs(num_frames) > 10) {
    ROS_WARN_STREAM("Timesync is now off by "
                    << num_frames
                    << " frames, something must be going horribly wrong");
  }
  *synced_timestamp += ros::Duration(num_frames * *delta_t);
  *synced_timestamp += ros::Duration(*offset);

  if (verbose_) {
    ROS_INFO_STREAM("Input Timestamp: " << received_timestamp.sec << "."
                                        << std::setfill('0') << std::setw(9)
                                        << received_timestamp.nsec);
    ROS_INFO_STREAM("Output Timestamp: " << synced_timestamp->sec << "."
                                         << std::setfill('0') << std::setw(9)
                                         << synced_timestamp->nsec);
  }
}

void CDKF::predictionUpdate(const ros::Time& received_timestamp) {
  if (verbose_) {
    ROS_INFO_STREAM("Initial State: \n" << state_.transpose());
    ROS_INFO_STREAM("Initial Cov: \n" << cov_);
  }

  // work out how many frames to go forward to guard against drops
  int num_frames = std::round(((received_timestamp - zero_timestamp_).toSec() -
                               accessS(state_, STATE_TIMESTAMP)[0]) /
                              accessS(state_, DELTA_T)[0]);
  if (num_frames < 1) {
    num_frames = 1;
  }
  num_frames = 1;

  for (size_t i = 0; i < num_frames; ++i) {
    StateSigmaPoints sigma_points(state_, cov_, prediction_noise_sd_,
                                  CDKF::propergateState);

    sigma_points.calcEstimatedMean(&state_);
    sigma_points.calcEstimatedCov(&cov_);
  }

  if (verbose_) {
    ROS_INFO_STREAM("Predicted State: \n" << state_.transpose());
    ROS_INFO_STREAM("Predicted Cov: \n" << cov_);
  }
}

void CDKF::measurementUpdate(const ros::Time& prev_stamp,
                             const ros::Time& current_stamp,
                             const double image_angular_velocity,
                             const IMUList& imu_rotations,
                             const bool calc_offset) {
  // convert tracked points to measurement
  Eigen::VectorXd real_measurement(kMeasurementSize);
  accessM(real_measurement, MEASURED_TIMESTAMP).array() =
      (current_stamp - zero_timestamp_).toSec();

  accessM(real_measurement, ANGULAR_VELOCITY).array() = image_angular_velocity;

  if (verbose_) {
    ROS_INFO_STREAM("Measured Values: \n" << real_measurement.transpose());
  }

  // create sigma points
  MeasurementSigmaPoints sigma_points(
      state_, cov_, measurement_noise_sd_, CDKF::stateToMeasurementEstimate,
      imu_rotations, zero_timestamp_, calc_offset);

  // get mean and cov
  Eigen::VectorXd predicted_measurement;
  sigma_points.calcEstimatedMean(&predicted_measurement);
  Eigen::MatrixXd innovation;
  sigma_points.calcEstimatedCov(&innovation);

  if (verbose_) {
    ROS_INFO_STREAM("Predicted Measurements: \n"
                    << predicted_measurement.transpose());
  }

  // calc mah distance
  Eigen::VectorXd diff = real_measurement - predicted_measurement;
  double mah_dist = std::sqrt(diff.transpose() * innovation.inverse() * diff);
  if (mah_dist > mah_threshold_) {
    ROS_WARN("Outlier detected, measurement rejected");
    return;
  }

  Eigen::MatrixXd cross_cov;
  sigma_points.calcEstimatedCrossCov(&cross_cov);

  Eigen::MatrixXd gain = cross_cov * innovation.inverse();

  const Eigen::VectorXd state_diff =
      gain * (real_measurement - predicted_measurement);

  state_ += state_diff;

  cov_ -= gain * innovation * gain.transpose();

  if (verbose_) {
    ROS_INFO_STREAM("Updated State: \n" << state_.transpose());
    ROS_INFO_STREAM("Updated Cov: \n" << cov_);
  }

  // guard against precision issues
  constexpr double kMaxTime = 10000.0;
  if (accessS(state_, STATE_TIMESTAMP)[0] > kMaxTime) {
    rezeroTimestamps(current_stamp);
  }
}

void CDKF::stateToMeasurementEstimate(
    const IMUList& imu_rotations, const ros::Time zero_stamp, bool calc_offset,
    const Eigen::VectorXd& input_state, const Eigen::VectorXd& noise,
    Eigen::Ref<Eigen::VectorXd> estimated_measurement) {
  ros::Time end_stamp =
      zero_stamp + ros::Duration(accessS(input_state, STATE_TIMESTAMP)[0]);

  // this doesn't guard against the case of a dropped frame, but the angular
  // velocity will probably still be pretty similar
  ros::Time start_stamp =
      end_stamp - ros::Duration(accessS(input_state, DELTA_T)[0]);

  accessM(estimated_measurement, ANGULAR_VELOCITY).array() =
      accessM(noise, ANGULAR_VELOCITY)[0];

  if (calc_offset) {
    end_stamp += ros::Duration(accessS(input_state, OFFSET)[0]);
    accessM(estimated_measurement, ANGULAR_VELOCITY).array() +=
        getImuAngleChange(imu_rotations, start_stamp, end_stamp);
  }

  accessM(estimated_measurement, MEASURED_TIMESTAMP) =
      accessS(input_state, STATE_TIMESTAMP) + accessM(noise, STATE_TIMESTAMP);
}

void CDKF::propergateState(const Eigen::VectorXd& noise,
                           Eigen::Ref<Eigen::VectorXd> current_state) {
  accessS(current_state, DELTA_T) += accessS(noise, DELTA_T);
  accessS(current_state, OFFSET) += accessS(noise, OFFSET);

  accessS(current_state, STATE_TIMESTAMP) += accessS(current_state, DELTA_T);
}

Eigen::Quaterniond CDKF::getInterpolatedImuAngle(const IMUList& imu_rotations,
                                                 const ros::Time& stamp) {
  IMUList::const_iterator prev_it, it;
  prev_it = imu_rotations.begin();

  // find location of starting stamp
  for (it = imu_rotations.begin(); ((std::next(it) != imu_rotations.end()) &&
                                    ((stamp - it->first).toSec() > 0.0));
       prev_it = it++)
    ;

  // interpolate to get angle
  Eigen::Quaterniond angle;
  if (prev_it->first == it->first) {
    angle = it->second;
  } else {
    const double delta_t = (it->first - prev_it->first).toSec();
    double w = (stamp - prev_it->first).toSec() / delta_t;

    // don't extrapolate
    if (w < 0.0) {
      ROS_WARN("Trying to get Imu data from too far in the past");
      w = 0.0;
    } else if (w > 1.0) {
      ROS_WARN("Trying to get Imu data from too far in the future");
      w = 1.0;
    }

    angle = prev_it->second.slerp(w, it->second);
  }

  return angle;
}

double CDKF::getImuAngleChange(const IMUList& imu_rotations,
                               const ros::Time& start_stamp,
                               const ros::Time& end_stamp) {
  Eigen::Quaterniond start_angle =
      getInterpolatedImuAngle(imu_rotations, start_stamp);
  Eigen::Quaterniond end_angle =
      getInterpolatedImuAngle(imu_rotations, end_stamp);
  Eigen::AngleAxisd diff_angle(start_angle.inverse() * end_angle);

  return diff_angle.angle();
}
