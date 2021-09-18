#pragma once
#ifndef _MEASUREMENT_HPP_
#define _MEASUREMENT_HPP_

#include "Eigen/Dense"

class Measurement {
public:
	long long timestamp_;

	enum SensorType {
		LASER,
		RADAR
	} sensor_type_;

	Eigen::VectorXd raw_measurements_;
};

#endif
