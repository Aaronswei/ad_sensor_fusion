
#include <ros/ros.h>
#include <tracking_lib/ukf.h>

int main(int argc, char **argv){
	ros::init(argc, argv, "tracking_node");
	tracking::UnscentedKF tracker(
		ros::NodeHandle(), ros::NodeHandle("~"));
	ros::spin();

	return 0;
}
