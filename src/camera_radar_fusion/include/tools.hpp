#pragma once
#ifndef _TOOLS_HPP_
#define _TOOLS_HPP_

#include <vector>
#include "Eigen/Dense"

class Tools
{
public:
	/*Constructor.*/
	Tools();

	/*Destructor.*/
	virtual ~Tools();

	/*
	A helper method to calculate RMSE.
	*/
	Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, const std::vector<Eigen::VectorXd> &ground_truth);

	/**
	A helper method to calculate Jacobians.
	*/
	Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd& x_state);
};

#endif