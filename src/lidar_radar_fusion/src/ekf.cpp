#include "ekf.hpp"
#include "tools.hpp"

#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

FusionEKF::FusionEKF() {
	is_initialized_ = false;

	previous_timestamp_ = 0;

	// initializing matrices
	R_radar_ = MatrixXd(4, 4);
	H_jacobian = MatrixXd(4, 4);

	//measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0, 0, 
				0, 0.0009, 0, 0,
				0, 0, 0.09, 0,
				0, 0, 0, 0.0009;

	// Radar - jacobian matrix
	H_jacobian <<   0, 0, 0, 0,
					0, 0, 0, 0,
					0, 0, 0, 0,
					0, 0, 0, 0;

	// initialize the kalman filter variables
	ekf_.P_ = MatrixXd(4, 4);
	ekf_.P_ <<  1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1000, 0,
				0, 0, 0, 1000;

	ekf_.F_ = MatrixXd(4, 4);
	ekf_.F_ <<  1, 0, 1, 0,
				0, 1, 0, 1,
				0, 0, 1, 0,
				0, 0, 0, 1;

	// Initialize process noise covariance matrix
	ekf_.Q_ = MatrixXd(4, 4);
	ekf_.Q_ << 0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0;

	// Initialize ekf state
	ekf_.x_ = VectorXd(4);
	ekf_.x_ << 1, 1, 1, 1;

	// set measurement noises
	noise_ax = 9;
	noise_ay = 9;
}

FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(float x, float y, float vx, float vy, long long timestamp_) {

	Eigen::VectorXd raw_measurements_ = Eigen::VectorXd(4);
	raw_measurements_ << x, y, vx, vy;

	//Initialization
	if (!is_initialized_) {

		// first measurement
		ekf_.x_ = VectorXd(4);

		ekf_.x_ << x, y, vx, vy;  // x, y, vx, vy

		// previous_timestamp_ = timestamp_;

		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}

	//  Prediction

	/*
	* Update the state transition matrix F according to the new elapsed time.
	- Time is measured in seconds.
	* Update the process noise covariance matrix.
	* Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
	*/

	// compute the time elapsed between the current and previous measurements
	// float dt = (timestamp_ - previous_timestamp_) / 1000000.0;  //  in seconds
	float dt = timestamp_ / 1000.0;  //  in seconds
	// previous_timestamp_ = timestamp_;

	float dt_2 = dt * dt;
	float dt_3 = dt_2 * dt;
	float dt_4 = dt_3 * dt;

	// Modify the F matrix so that the time is integrated
	ekf_.F_(0, 2) = dt;
	ekf_.F_(1, 3) = dt;

	//set the process covariance matrix Q
	ekf_.Q_ = MatrixXd(4, 4);
	ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
				0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
				dt_3 / 2 * noise_ax, 0, dt_2*noise_ax, 0,
				0, dt_3 / 2 * noise_ay, 0, dt_2*noise_ay;
	
	if (dt > 0.001)
		ekf_.Predict();

	//Update
	/*
	* Use the sensor type to perform the update step.
	* Update the state and covariance matrices.
	*/

	//H_jacobian = tools.CalculateJacobian(ekf_.x_);
	ekf_.H_ = H_jacobian;
	ekf_.R_ = R_radar_;
	ekf_.UpdateEKF(raw_measurements_);
	// ekf_.Update(raw_measurements_);

}