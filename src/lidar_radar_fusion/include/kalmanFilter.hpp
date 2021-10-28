#pragma once
#ifndef _KALMAN_FILTER_HPP_
#define _KALMAN_FILTER_HPP_

#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class KalmanFilter
{
public:

	KalmanFilter();	// Constructor
	virtual ~KalmanFilter();  // Destructor

	/**
	* Init Initializes Kalman filter
	* @param x_in Initial state
	* @param P_in Initial state covariance
	* @param F_in Transition matrix
	* @param H_in Measurement matrix
	* @param R_in Measurement covariance matrix
	* @param Q_in Process covariance matrix
	*/
	void Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
		MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in);

	/**
	* Prediction Predicts the state and the state covariance
	* using the process model
	* @param delta_T Time between k and k+1 in s
	*/
	void Predict();

	/**
	* Updates the state by using standard Kalman Filter equations
	* @param z The measurement at k+1
	*/
	void Update(const VectorXd &z);

	/**
	* Updates the state by using Extended Kalman Filter equations
	* @param z The measurement at k+1
	*/
	void UpdateEKF(const VectorXd &z);

	/**
	* General kalman filter update operations
	* @param y the update prediction error
	*/
	void UpdateRoutine(const VectorXd &y);

public:
	VectorXd x_;		// state vector
	MatrixXd P_;		// state covariance matrix
	MatrixXd F_;		// state transistion matrix
	MatrixXd Q_;		// process covariance matrix
	MatrixXd H_;		// measurement matrix
	MatrixXd R_;		// measurement covariance matrix

};

#endif