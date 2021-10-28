#ifndef __LIDAR_RADAR_UTILS_HPP__
#define __LIDAR_RADAR_UTILS_HPP__
#include <pcl/ModelCoefficients.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "opencv2/opencv.hpp"


namespace lidar_radar_fusion {
    #define PRINT_VEC(vec) \
    for(const auto& item:vec) {\
        std::cout << item << std::endl;\
    }

    // #define PRINT_MAP(map) \
    // for(const auto& item:map) {\
    //     std::cout <<"key:" << item->first << "value:" << item->second << std::endl;\
    // }

    typedef long long TimeStamp;
    typedef pcl::PointXYZ Point;
    typedef pcl::PointCloud<Point> PointCloud;

    struct ObjectInfo {
        int id = -1;
        TimeStamp time_stamp = -1;

        Eigen::Vector3d position{0.0,0.0,0.0};
    };

    struct TrackObject {
        int track_id = -1;
        int miss = 0;
        int life = 0;

        Eigen::Vector3d obs_point;
        Eigen::Vector3d obs_vel;
        Eigen::Vector3d obs_acc;
    };


    void GetFileLists(const std::string& path, std::vector<cv::String>& lists);
    std::vector<std::string> Split(const std::string& str, const std::string& delim);
    // helper function to normalize angles:
    double AngleNormalization(double angle);

}
#endif // __LIDAR_RADAR_UTILS_HPP__