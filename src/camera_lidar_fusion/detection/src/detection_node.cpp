
#include <ros/ros.h>
#include <detection_lib/dbscan.h>

int main(int argc, char **argv){
	ros::init(argc, argv, "detection_node");
	detection::DbScan detector(
		ros::NodeHandle(), ros::NodeHandle("~"));
	ros::spin();

	return 0;
}
