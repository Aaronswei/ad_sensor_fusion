#ifndef SIGMA_POINTS_TIME_AUTOSYNC_H
#define SIGMA_POINTS_TIME_AUTOSYNC_H

#include <Eigen/Eigen>

#include "camera_imu_time_sync/state_data.h"

class SigmaPointsBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SigmaPointsBase(const size_t vector_size, const size_t state_size,
                  const size_t noise_size)
      : L_(state_size + noise_size),
        wm0_((h2_ - L_) / h2_),
        sigma_zero_(vector_size),
        sigma_plus_(sigma_zero_.size(), state_size + noise_size),
        sigma_minus_(sigma_zero_.size(), state_size + noise_size) {}

  void calcEstimatedMean(Eigen::VectorXd* mean) const {
    *mean = wm0_ * sigma_zero_;
    for (size_t i = 0; i < sigma_plus_.cols(); ++i) {
      *mean += wmi_ * sigma_plus_.col(i);
      *mean += wmi_ * sigma_minus_.col(i);
    }
  }

  void calcEstimatedCov(Eigen::MatrixXd* cov) const {
    const Eigen::MatrixXd alpha = sigma_plus_ - sigma_minus_;
    const Eigen::MatrixXd beta =
        (sigma_plus_ + sigma_minus_).colwise() - 2 * sigma_zero_;

    *cov = wc1i_ * alpha * alpha.transpose() + wc2i_ * beta * beta.transpose();

    //keep things bounded and sane
    cov->diagonal() = cov->diagonal().array().max(1e-20);
    cov->diagonal() = cov->diagonal().array().min(1e20);
  }

 protected:
  // static constexpr double h_ = std::sqrt(3.0);
  // static constexpr double h2_ = h_ * h_;
  // static constexpr double wmi_ = 1.0 / (2.0 * h2_);
  // static constexpr double wc1i_ = 1.0 / (4.0 * h2_);
  // static constexpr double wc2i_ = (h2_ - 1.0) / (4.0 * h2_ * h2_);
  double h_ = std::sqrt(3.0);
  double h2_ = h_ * h_;
  double wmi_ = 1.0 / (2.0 * h2_);
  double wc1i_ = 1.0 / (4.0 * h2_);
  double wc2i_ = (h2_ - 1.0) / (4.0 * h2_ * h2_);
  const double L_;
  const double wm0_;

  Eigen::VectorXd sigma_zero_;
  Eigen::MatrixXd sigma_plus_;
  Eigen::MatrixXd sigma_minus_;
};

class StateSigmaPoints : public SigmaPointsBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename FunctionPtr, class... Ts>
  StateSigmaPoints(const Eigen::VectorXd& state, const Eigen::MatrixXd& cov,
                   const Eigen::VectorXd& noise_sd,
                   FunctionPtr propergation_function, Ts... args)
      : SigmaPointsBase(state.size(), state.size(), noise_sd.size()) {
    sigma_zero_ = state;

    Eigen::LLT<Eigen::MatrixXd> cov_llt(cov);
    if (cov_llt.info() == Eigen::NumericalIssue) {
      ROS_ERROR("Calculation of sqrt mat failed, your totally screwed");
      ROS_ERROR_STREAM("cov: \n" << cov);
    }

    Eigen::MatrixXd sqrt_matrix = cov_llt.matrixL();
    sigma_plus_.leftCols(state.size()) = (h_ * sqrt_matrix).colwise() + state;
    sigma_plus_.rightCols(noise_sd.size()).colwise() = state;

    sigma_minus_.leftCols(state.size()) = (-h_ * sqrt_matrix).colwise() + state;
    sigma_minus_.rightCols(noise_sd.size()).colwise() = state;

    Eigen::VectorXd added_noise(noise_sd.size());
    added_noise.setZero();

    // propagate the state
    propergation_function(args..., added_noise, sigma_zero_);
    for (size_t i = 0; i < sigma_zero_.size(); ++i) {
      propergation_function(args..., added_noise, sigma_plus_.col(i));
      propergation_function(args..., added_noise, sigma_minus_.col(i));
    }

    // add in the augmented state elements and propagate them
    const size_t offset = sigma_zero_.size();
    for (size_t i = 0; i < added_noise.size(); ++i) {
      added_noise.setZero();
      added_noise[i] = h_ * noise_sd[i];
      propergation_function(args..., added_noise, sigma_plus_.col(i + offset));
      added_noise[i] = -h_ * noise_sd[i];
      propergation_function(args..., added_noise, sigma_minus_.col(i + offset));
    }
  }
};

class MeasurementSigmaPoints : public SigmaPointsBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename FunctionPtr, class... Ts>
  MeasurementSigmaPoints(const Eigen::VectorXd& state,
                         const Eigen::MatrixXd& cov,
                         const Eigen::VectorXd& noise_sd,
                         FunctionPtr measurement_estimation_function,
                         Ts... args)
      : SigmaPointsBase(NUM_MEASUREMENT_ELEMENTS,
            state.size(), noise_sd.size()) {
    Eigen::VectorXd added_noise(noise_sd.size());
    added_noise.setZero();

    measurement_estimation_function(args..., state, added_noise, sigma_zero_);

    // state measurement estimates
    Eigen::LLT<Eigen::MatrixXd> cov_llt(cov);
    if (cov_llt.info() == Eigen::NumericalIssue) {
      ROS_ERROR("Calculation of sqrt mat failed, your totally screwed");
    }

    sqrt_matrix_ = cov_llt.matrixL();

    for (size_t i = 0; i < state.size(); ++i) {
      measurement_estimation_function(args..., state + h_ * sqrt_matrix_.col(i),
                                      added_noise, sigma_plus_.col(i));
      measurement_estimation_function(args..., state - h_ * sqrt_matrix_.col(i),
                                      added_noise, sigma_minus_.col(i));
    }
    // augemented state measurement estimates
    const size_t offset = state.size();
    for (size_t i = 0; i < added_noise.size(); ++i) {
      added_noise.setZero();
      added_noise[i] = h_ * noise_sd[i];
      measurement_estimation_function(args..., state, added_noise,
                                      sigma_plus_.col(i + offset));
      added_noise[i] = -h_ * noise_sd[i];
      measurement_estimation_function(args..., state, added_noise,
                                      sigma_minus_.col(i + offset));
    }
  }

  void calcEstimatedCrossCov(Eigen::MatrixXd* cross_cov) const {
    *cross_cov =
        std::sqrt(wc1i_) * sqrt_matrix_ *
        (sigma_plus_ - sigma_minus_).leftCols(sqrt_matrix_.cols()).transpose();
  }

 private:
  Eigen::MatrixXd sqrt_matrix_;
};

#endif  // TIME_AUTOSYNC_SIGMA_POINTS_H
