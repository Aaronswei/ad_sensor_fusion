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
    void Point2Camera(const std::vector<cv::Point3d>& radar_points, 
                      std::vector<cv::Point3d>& camera_points, 
                      cv::Mat& rot_matrix, 
                      cv::Mat& trans_matrix);

}

// helper function to normalize angles:
double AngleNormalization(double angle);

void check_files(std::ifstream& in_file, std::string& in_name);

void check_arguments(int argc, char* argv[]);

std::string path_leaf(std::string const & path);
template<typename Out> void split(const std::string &s, char delim, Out result);
std::vector<std::string> split(const std::string &s, char delim);
// bool exists(std::string name);

float overlap(float x1, float w1, float x2, float w2);
float box_intersection(float a[4], float b[4]);
float box_union(float a[4], float b[4]);
float box_iou(float a[4], float b[4]);
std::vector<int> vector_union(std::vector <int> a, std::vector <int> b);

#endif // __UTILS_HPP__