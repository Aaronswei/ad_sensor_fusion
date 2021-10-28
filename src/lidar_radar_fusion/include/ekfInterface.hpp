#pragma once
#ifndef _EKF_INTERFACE_HPP_
#define _EKF_INTERFACE_HPP_

#include "ekf.hpp"
#include "tools.hpp"
#include "measurement.hpp"
#include "groundTruth.hpp"

#include "Eigen/Dense"
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class EKF_API
{
public:
	EKF_API();
	~EKF_API();

	void process(std::vector<Measurement> measurement_pack_list);

private:
	// Create a Fusion EKF instance
	FusionEKF *fusionEKF;

	Tools *tools_;
	// used to compute the RMSE later
	std::vector<VectorXd> estimations;
};

#endif

