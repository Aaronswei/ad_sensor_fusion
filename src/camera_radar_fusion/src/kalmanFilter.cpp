#include "kalmanFilter.hpp"
#include "utils.hpp"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter(){}
KalmanFilter::~KalmanFilter(){}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
	MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in)
{
	x_ = x_in;
	P_ = P_in;
	F_ = F_in;
	H_ = H_in;
	R_ = R_in;
	Q_ = Q_in;
}

void KalmanFilter::Predict() {
	// Predict the state
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;

}

void KalmanFilter::UpdateRoutine(const VectorXd& y) {

	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();

	// Compute Kalman gain
	MatrixXd K = P_ * Ht * Si;

	// Update estimate
	x_ = x_ + K * y;
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
	std::cout << __LINE__ << " "<< __FILE__  << std::endl;
}

void KalmanFilter::Update(const VectorXd &z) {
	/*
	* update the state by using Kalman Filter equations
	*/
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;

	//MatrixXd Ht = H_.transpose();
	//MatrixXd S = H_ * P_ * Ht + R_;
	//MatrixXd Si = S.inverse();
	//
	//// Compute Kalman gain
	//MatrixXd PHt = P_ * Ht;
	//MatrixXd K = PHt * Si;

	////  new estimate
	//x_ = x_ + (K * y);
	//long x_size = x_.size();
	//MatrixXd I = MatrixXd::Identity(x_size, x_size);
	//P_ = (I - K * H_) * P_;
	UpdateRoutine(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	/*
	* update the state by using Extended Kalman Filter equations
	*/
	float PI = 3.1416;
	VectorXd Hx = VectorXd(4);
	float px = x_(0);
	float py = x_(1);
	float vx = x_(2);
	float vy = x_(3);

	float rho = sqrt(pow(px, 2) + pow(py, 2));
	float phi = 0.0; // this will be default case if px is 0
	if (fabs(px) > 0.0001) {
		phi = atan2(py, px);
	}
	float rho_dot = 0; // this will be default case if rho is 0
	if (fabs(rho) > 0.0001) {
		rho_dot = (px*vx + py*vy) / rho;
	}
	Hx << rho, phi, rho_dot, 0;
	VectorXd y = z - Hx;

	// normalize result to -pi and pi
	//if (fabs(y(1))>3.14) {
	//	y(1) = fmod(y(1), 2 * 3.14);
	//}
	y(1) = AngleNormalization(y(1));

	//MatrixXd Ht = H_.transpose();
	//MatrixXd S = H_ * P_ * Ht + R_;
	//MatrixXd Si = S.inverse();
	//MatrixXd PHt = P_ * Ht;
	//MatrixXd K = PHt * Si;
	std::cout << __LINE__ << " "<< __FILE__  << std::endl;
	////  new estimate
	//x_ = x_ + (K * y);
	//long x_size = x_.size();
	//MatrixXd I = MatrixXd::Identity(x_size, x_size);
	//P_ = (I - K * H_) * P_;
	UpdateRoutine(y);
}