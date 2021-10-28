
#include <iostream>
#include <string>
#include <vector>
#include <glog/logging.h>
#include <boost/filesystem.hpp>
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SVD"

#include "utils.hpp"
namespace lidar_radar_fusion {



void GetFileLists(const std::string& path, 
                  std::vector<cv::String>& lists) {
    if(path.empty()) {
        LOG(INFO) << "invalib path ... ";
        return;
    }

    lists.clear();

    cv::glob(path, lists);

    return;
}

void LoadPointCloudFrame(const std::string& path, PointCloud& lidar_frame) {

    lidar_frame.clear();
    LOG(INFO) << "loading file: " << path;
    if(-1 == pcl::io::loadPCDFile<pcl::PointXYZ>(path, lidar_frame)) {
        LOG(ERROR) << "could not load data: " << path;
        return;
    }

    return;
}


std::vector<std::string> Split(const std::string& str, 
                               const std::string& delim) {
    std::vector<std::string> res;
    if ("] [" == str) return res;
    //先将要切割的字符串从string类型转换为char*类型
    char* strs = new char[str.length() + 1];
    strcpy(strs, str.c_str());

    char* des = new char[delim.length() + 1];
    strcpy(des, delim.c_str());

    char* p = strtok(strs, des);
    while (p) {
        std::string s = p; //分割得到的字符串转换为string类型
        res.push_back(s); //存入结果数组
        p = strtok(NULL, des);
    }

    delete[] strs;
    delete[] des;
    return res;
}

// void LoadTXTLabel(const std::string& label_path, 
//                   const std::string& label_name, 
//                   std::vector<cv::Rect2d>& dets) {
//     dets.clear();
//     cv::Rect2d bbox;
//     std::string path = label_path + "/" + label_name;
//     std::ifstream file(path, std::ios::in);
//     if(!file.is_open()) {
//         LOG(INFO) << "could not load file: " << path;
//         return;
//     }

//     std::string line;
//     while(std::getline(file, line)) {
//         // LOG(INFO) << "line: " << line;
//         std::vector<std::string> split_info = Split(line, " ");
//         bbox.width = std::stod(split_info[3]) * IMAGE_WIDTH;
//         bbox.height = std::stod(split_info[4]) * IMAGE_HEIGHT ;
//         bbox.x = std::stod(split_info[1]) * IMAGE_WIDTH - bbox.width * 0.5;
//         bbox.y = std::stod(split_info[2]) * IMAGE_HEIGHT - bbox.height * 0.5;

//         dets.emplace_back(bbox);
//     }

// }

// void LoadRadarInfo(const std::string& path, std::vector<RadarInfo_t>& radar_objects) {
//     radar_objects.clear();
//     std::ifstream file(path, std::ios::in);
//     if(!file.is_open()) {
//         LOG(INFO) << "could not load radar: " << path;
//         return;
//     }

//     std::string dummy;
//     file >> dummy; 
    
//     file >> dummy >> dummy;   
//     file >> dummy >> dummy;  
//     file >> dummy >> dummy; 
    
//     if (dummy == "tracks[]") {  
//         LOG(INFO) << " radar frame id is empty";
//     } else {
//         file >> dummy;  
//     }
//     while (file >> dummy) {  
//         RadarInfo_t object;
//         file >> dummy >> object.track_id;
//         file >> dummy >> dummy >> dummy;
//         file >> dummy >> object.position[0];
//         file >> dummy >> object.position[1];
//         file >> dummy >> object.position[2];
//         file >> dummy;
//         file >> dummy >> dummy;
//         file >> dummy >> dummy;
//         file >> dummy >> dummy;
//         file >> dummy;
//         file >> dummy >> dummy;
//         file >> dummy >> dummy;
//         file >> dummy >> dummy;
//         file >> dummy;
//         file >> dummy >> dummy;
//         file >> dummy >> dummy;
//         file >> dummy >> dummy;
//         file >> dummy;
//         file >> dummy >> object.velocity[0];
//         file >> dummy >> object.velocity[1];
//         file >> dummy >> object.velocity[2];
//         file >> dummy;
//         file >> dummy >> dummy;
//         file >> dummy >> dummy;
//         file >> dummy >> dummy;
//         radar_objects.push_back(object);
//     }

//     file.close();
// }

double AngleNormalization(double angle) {
    // Constrain to less than pi
    while (angle > M_PI) angle -= 2.0 * M_PI;

    // Constrain to greater than -pi
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

}
