#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include "opencv2/opencv.hpp"
namespace radar_camera_spatial_sync {
    
    const float CAR_HEIGHT = 1.75; //m
    const unsigned int IMAGE_WIDTH =1920;//pixel
    const unsigned int IMAGE_HEIGHT =1080;


    typedef struct {
        double timestamp = 0.0;
        unsigned int track_id = -1;
        int miss = 0;
        int life = 0;
        Eigen::Vector3d position = {0.0, 0.0, 0.0};
        Eigen::Vector3d velocity = {0.0, 0.0, 0.0};
    } RadarInfo_t;

    void GetFileLists(const std::string& img_path, 
                      const std::string& label_path,
                      std::vector<cv::String>& img_lists,
                      std::vector<cv::String>& label_lists);

    void LoadImage(const std::string& img_path, cv::Mat& img);

    std::vector<std::string> Split(const std::string& str, 
                                   const std::string& delim);

    void LoadTXTLabel(const std::string& label_path, 
                      const std::string& label_name, 
                      std::vector<cv::Rect2d>& dets);

    void LoadRadarInfo(const std::string& path,
                       std::vector<RadarInfo_t>& radar_objects);

    void ExecuteEstimate3Dvs3DPose(const std::vector<cv::Point3d>& camera_points, 
                                   const std::vector<cv::Point3d>& radar_points, 
                                   cv::Mat& rot_matrix, 
                                   cv::Mat& trans_matrix,
                                   std::vector<float>& reproject_error);
}

#endif // __UTILS_HPP__