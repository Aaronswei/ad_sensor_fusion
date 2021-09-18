#pragma once
#ifndef _EKF_HPP_
#define _EKF_HPP_

#include <string>
#include <fstream>
#include <vector>

#include "Eigen/Dense"
#include "measurement.hpp"
#include "tools.hpp"
#include "kalmanFilter.hpp"

class FusionEKF {
public:

	FusionEKF();	//Constructor.
	virtual ~FusionEKF();	//Destructor.

	void ProcessMeasurement(float x, float y, float vx, float vy, long long timestamp_);	// Run the whole flow of the Kalman Filter from here.

	KalmanFilter ekf_;  //Kalman Filter update and prediction math lives in here.

private:
	// check whether the tracking toolbox was initiallized or not (first measurement)
	bool is_initialized_;

	// previous timestamp
	long long previous_timestamp_;

	// tool object used to compute Jacobian and RMSE
	Tools tools;
	Eigen::MatrixXd R_laser_;    // laser measurement noise
	Eigen::MatrixXd R_radar_;    // radar measurement noise
	Eigen::MatrixXd H_laser_;    // measurement function for laser
	Eigen::MatrixXd H_jacobian;         // measurement function for radar

	//acceleration noise components
	float noise_ax;
	float noise_ay;
};

#endif