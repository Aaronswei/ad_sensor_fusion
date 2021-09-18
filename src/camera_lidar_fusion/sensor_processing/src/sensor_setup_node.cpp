#include <ros/ros.h>
#include <sensor_processing_lib/sensor_fusion.h>

int main(int argc, char **argv){
	ros::init(argc, argv, "sensor_setup_node");
	sensor_processing::SensorFusion sensor_setup(
		ros::NodeHandle(), ros::NodeHandle("~"));
	ros::spin();

	return 0;
}
