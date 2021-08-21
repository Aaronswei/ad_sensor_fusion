#ifndef _CAMERA_LIDAR_SYNC_HPP_
#define _CAMERA_LIDAR_SYNC_HPP_

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/features/don.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/registration/ndt.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/Image.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudRadar;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudCluster;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
typedef message_filters::Subscriber<sensor_msgs::Image> image_sub_type;
typedef message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub_type;

class CameraLidarSync
{
public:
    CameraLidarSync(const ros::NodeHandle& nh,
                           const ros::NodeHandle& nh_private);

    // void pointCloudCallback(const PointCloud::ConstPtr& msg);
    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &laser_scan);

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);


private:
    PointCloud prePointCloud_;
    PointCloud curPointCloud_;

    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    image_transport::ImageTransport it_;


    ros::Subscriber imu_sub_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher pub_img_;

    ros::Subscriber pointcloud_sub_;

    ///The smallest scale to use in the DoN filter.
    double scale1_;
    ///The largest scale to use in the DoN filter.
    double scale2_;
    ///The minimum DoN magnitude to threshold by
    double threshold_;
    ///segment scene into clusters with given distance tolerance using euclidean clustering
    double segradius_;

    int delay_by_n_frames_;
    double max_image_age_s_;

    int img_height_;
    int img_width_;
    Eigen::Matrix4d laser_to_cam_;
    Eigen::Matrix4d camera_intrinsics_;

    std::vector<sensor_msgs::Image> img_vec_;
    
    std::list<cv_bridge::CvImage> images_;
    std::vector<pcl::PointCloud<pcl::PointXYZI> > point_cloud_lists_;
    std::vector<double> time_vec_;

    message_filters::Synchronizer<MySyncPolicy> *sync_;

};
#endif
