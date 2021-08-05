#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pcl/common/common_headers.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>

#include "point_clouds_icp/icp.h"


DEFINE_int32(icp_max_neighbours, 50 ,"");
DEFINE_double(icp_radius, 10, "");
DEFINE_string(icp_source_cloud_filename, "/disk3/sensor_fusion/camera_imu_time_sync/datasets/pcl_pcd/1621586141.699102.pcd", "");
DEFINE_string(icp_target_cloud_filename, "/disk3/sensor_fusion/camera_imu_time_sync/datasets/pcl_pcd/1621586141.799182.pcd", "");
DEFINE_string(icp_aligned_cloud_filename, "/tmp/icp.pcd", "");
DEFINE_double(icp_source_filter_size, 5, "");
DEFINE_double(icp_target_filter_size, 0, "");
DEFINE_bool(icp_save_aligned_cloud, true, "");
DEFINE_bool(icp_visualize_clouds, true, "");
DEFINE_string(icp_frame_id, "", "");

void parseIcpParameters(IcpParameters* params) {
  params->max_neighbours = FLAGS_icp_max_neighbours;
  params->radius = FLAGS_icp_radius;
  params->source_cloud_filename = FLAGS_icp_source_cloud_filename;
  params->target_cloud_filename = FLAGS_icp_target_cloud_filename;
  params->aligned_cloud_filename = FLAGS_icp_aligned_cloud_filename;
  params->visualize_clouds = FLAGS_icp_visualize_clouds;
  params->save_aligned_cloud = FLAGS_icp_save_aligned_cloud;
  params->frame_id = FLAGS_icp_frame_id;
}

typedef pcl::PointXYZ PointType;

void loadPointClouds(const std::string source_cloud_filename,
                     const std::string target_cloud_filename,
                     pcl::PointCloud<PointType>::Ptr source_cloud,
                     pcl::PointCloud<PointType>::Ptr target_cloud) {
  LOG(INFO) << "Loading source cloud: " << source_cloud_filename;
  pcl::io::loadPCDFile<PointType>(source_cloud_filename, *source_cloud);
  CHECK(source_cloud);
  CHECK(source_cloud->size() > 0u);

  LOG(INFO) << "Loading target cloud: " << target_cloud_filename;
  pcl::io::loadPCDFile<PointType>(target_cloud_filename, *target_cloud);
  CHECK(target_cloud);
  CHECK(target_cloud->size() > 0u);
}


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  ros::init(argc, argv, "ICP");
  ros::Time::init();

  // Load the ICP parameters.
  IcpParameters icp_params;
  parseIcpParameters(&icp_params);

  // Load source and target pointcloud.
  pcl::PointCloud<PointType>::Ptr source_cloud =
      boost::make_shared<pcl::PointCloud<PointType> >();
  pcl::PointCloud<PointType>::Ptr target_cloud =
      boost::make_shared<pcl::PointCloud<PointType> >();
  loadPointClouds(icp_params.source_cloud_filename, icp_params.target_cloud_filename,
                  source_cloud, target_cloud);

  // Run ICP.
  Icp icp(icp_params);
  icp.evaluate(source_cloud, target_cloud);

  return 0;
}
