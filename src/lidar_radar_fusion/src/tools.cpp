#include <iostream>
#include "tools.hpp"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
							  const vector<VectorXd> &ground_truth) {

	size_t num_estimations = estimations.size();
	size_t num_ground_truths = ground_truth.size();

	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	// sanity check of input validity
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if (estimations.size() != ground_truth.size() || estimations.size() < 1) {
		std::cout << "Cannot compute RMSE metric. Invalid input size." << std::endl;
		return rmse;
	}

	// accumulate residuals
	for (size_t i = 0; i < estimations.size(); ++i) {
		VectorXd residual = estimations[i] - ground_truth[i];
		
		//coefficient-wise multiplication
		residual = residual.array() * residual.array();
		rmse += residual;
	}

	// compute mean
	rmse /= estimations.size();

	// compute squared root
	rmse = rmse.array().sqrt();

	return rmse;

}


MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3, 4);

	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//pre-compute a set of terms which recur in the Jacobian to avoid repeated calculation
	float c1 = px*px + py*py;
	float c2 = sqrt(c1);
	float c3 = (c1*c2);

	//check division by zero
	if (fabs(c1) < 0.0001) {
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		Hj.fill(0.0);
		return Hj;
	}

	//compute the Jacobian matrix
	Hj << (px / c2), (py / c2), 0, 0,
		-(py / c1), (px / c1), 0, 0,
		py*(vx*py - vy*px) / c3, px*(px*vy - py*vx) / c3, px / c2, py / c2;

	return Hj;
}
