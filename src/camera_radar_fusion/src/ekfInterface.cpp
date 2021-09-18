#include "ekfInterface.hpp"

using namespace std;

EKF_API::EKF_API()
{
	fusionEKF = new FusionEKF();
	tools_ = new Tools();
}

EKF_API::~EKF_API()
{
	delete(fusionEKF);
}

void EKF_API::process(vector<Measurement> measurement_pack_list)
{
	
	//Call the EKF-based fusion
	size_t N = measurement_pack_list.size();
	for (size_t k = 0; k < N; ++k) {
		// start filtering from the second frame (the speed is unknown in the first
		// frame)



		fusionEKF->ProcessMeasurement(measurement_pack_list[k].raw_measurements_[0], 
									measurement_pack_list[k].raw_measurements_[1], 
									0,
									0,
									measurement_pack_list[k].timestamp_);

		estimations.push_back(fusionEKF->ekf_.x_);
	}

	cout << "EKF Done!" << endl;
}