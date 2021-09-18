#pragma once
#ifndef _GROUNDTRUTH_HPP_
#define _GROUNDTRUTH_HPP_

#include "Eigen/Dense"

class GroundTruthPackage {
public:
	long long timestamp_;

	enum SensorType {
		LASER,
		RADAR
	} sensor_type_;

	Eigen::VectorXd gt_values_;

};

#endif