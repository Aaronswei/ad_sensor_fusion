#include "point_clouds_icp/icp.h"

#include <string>

#include <glog/logging.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>



Icp::Icp(const IcpParameters& params)
  : params_(params) {}

void Icp::evaluate(
    pcl::PointCloud<PointType>::Ptr source_cloud,
    pcl::PointCloud<PointType>::Ptr target_cloud) {
  CHECK(source_cloud);
  CHECK(target_cloud);

  //调用pcl中的ICP库，一般需要两个变量，即source和target
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  // ICP Settings
  // icp.setMaxCorrespondenceDistance(); // 设置最大对应点的欧式距离，只有对应点之间的距离小于该设定值的对应点才作为ICP计算的对应点，default：1.7976931348623157e+308，基本上对所有点都计算了匹配点。
  
  //icp迭代条件设定，满足如下其中一个，即停止迭代
  icp.setMaximumIterations(100); // 设置最大迭代次数，迭代停止条件之一
  // icp.setTransformationEpsilon(1e-6); // 设置前后两次迭代的转换矩阵的最大epsilion，一旦两次迭代小于最大容差，泽认为已经收敛到最优解，迭代停止， default： 0。迭代停止条件之二
  // icp.setEuclideanFitnessEpsilon(1e-6);  //设置前后两次迭代的点对的欧式距离均值的最大容差，default：-std::numeric_limits::max ()。迭代终止条件之三
  // icp.setRANSACIterations(0);


  icp.setInputSource(source_cloud); // 设定source点云
  icp.setInputTarget(target_cloud); // 设定target点云
  LOG(INFO) << "TransformationEpsilon: " << icp.getTransformationEpsilon();
  LOG(INFO) << "MaxCorrespondenceDistance: " << icp.getMaxCorrespondenceDistance();
  LOG(INFO) << "RANSACOutlierRejectionThreshold: " << icp.getRANSACOutlierRejectionThreshold(); //获取 RANSAC算法剔除错误估计的阈值

  pcl::PointCloud<PointType>::Ptr aligned_source =
      boost::make_shared<pcl::PointCloud<PointType>>();
  icp.align(*aligned_source);  //执行ICP转换，并保存对齐后的点云
  LOG(INFO) << "Final transformation: " << std::endl << icp.getFinalTransformation();  //获取最终的配准的转化矩阵，即原始点云到目标点云的刚体变换
  if (icp.hasConverged()) {  //获取收敛状态，只要迭代过程符合上述终止条件之一，该函数返回true
    LOG(INFO) << "ICP converged." << std::endl
              << "The score is " << icp.getFitnessScore();  //用于获取迭代结束后目标点云和配准后的点云的最近点之间距离的均值
  } else {
    LOG(INFO) << "ICP did not converge.";
  }

  if (params_.save_aligned_cloud) {
    LOG(INFO) << "Saving aligned source cloud to: " << params_.aligned_cloud_filename;
    pcl::io::savePCDFile(params_.aligned_cloud_filename, *aligned_source);
  }

  if (params_.visualize_clouds) {
    source_cloud->header.frame_id = params_.frame_id;
    target_cloud->header.frame_id = params_.frame_id;
    aligned_source->header.frame_id = params_.frame_id;
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer
        (new pcl::visualization::PCLVisualizer ("ICP: source(red), target(green), aligned(blue)"));
    viewer->setBackgroundColor(255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        source_cloud_handler(source_cloud, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        target_cloud_handler(target_cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        aligned_source_handler(aligned_source, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(source_cloud, source_cloud_handler, "source");
    viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_cloud_handler, "target");
    viewer->addPointCloud<pcl::PointXYZ>(aligned_source, aligned_source_handler, "aligned source");
    viewer->spin();
  }
}

