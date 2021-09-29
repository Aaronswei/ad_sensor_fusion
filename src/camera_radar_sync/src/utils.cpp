
#include <iostream>
#include <string>
#include <vector>
#include <glog/logging.h>
#include <boost/filesystem.hpp>
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SVD"

#include "utils.hpp"
namespace radar_camera_spatial_sync {

void GetFileLists(const std::string& img_path, 
                  const std::string& label_path,
                  std::vector<cv::String>& img_lists,
                  std::vector<cv::String>& label_lists) {
    if(img_path.empty() || label_path.empty()) {
        LOG(INFO) << "invalib img_path or label_path ... ";
        return;
    }

    img_lists.clear();
    label_lists.clear();

    cv::glob(img_path, img_lists);
    cv::glob(label_path, label_lists);

    return;
}

void LoadImage(const std::string& img_path, cv::Mat& img) {
    img = cv::imread(img_path, true);
    if(img.empty()) {
        LOG(INFO) << "could not load image: " << img_path;
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

void LoadTXTLabel(const std::string& label_path, 
                  const std::string& label_name, 
                  std::vector<cv::Rect2d>& dets) {
    dets.clear();
    cv::Rect2d bbox;
    std::string path = label_path + "/" + label_name;
    std::ifstream file(path, std::ios::in);
    if(!file.is_open()) {
        LOG(INFO) << "could not load file: " << path;
        return;
    }

    std::string line;
    while(std::getline(file, line)) {
        // LOG(INFO) << "line: " << line;
        std::vector<std::string> split_info = Split(line, " ");
        bbox.width = std::stod(split_info[3]) * IMAGE_WIDTH;
        bbox.height = std::stod(split_info[4]) * IMAGE_HEIGHT ;
        bbox.x = std::stod(split_info[1]) * IMAGE_WIDTH - bbox.width * 0.5;
        bbox.y = std::stod(split_info[2]) * IMAGE_HEIGHT - bbox.height * 0.5;

        dets.emplace_back(bbox);
    }

}

void LoadRadarInfo(const std::string& path, std::vector<RadarInfo_t>& radar_objects) {
    radar_objects.clear();
    std::ifstream file(path, std::ios::in);
    if(!file.is_open()) {
        LOG(INFO) << "could not load radar: " << path;
        return;
    }

    std::string dummy;
    file >> dummy; 
    
    file >> dummy >> dummy;   
    file >> dummy >> dummy;  
    file >> dummy >> dummy; 
    
    if (dummy == "tracks[]") {  
        LOG(INFO) << " radar frame id is empty";
    } else {
        file >> dummy;  
    }
    while (file >> dummy) {  
        RadarInfo_t object;
        file >> dummy >> object.track_id;
        file >> dummy >> dummy >> dummy;
        file >> dummy >> object.position[0];
        file >> dummy >> object.position[1];
        file >> dummy >> object.position[2];
        file >> dummy;
        file >> dummy >> dummy;
        file >> dummy >> dummy;
        file >> dummy >> dummy;
        file >> dummy;
        file >> dummy >> dummy;
        file >> dummy >> dummy;
        file >> dummy >> dummy;
        file >> dummy;
        file >> dummy >> dummy;
        file >> dummy >> dummy;
        file >> dummy >> dummy;
        file >> dummy;
        file >> dummy >> object.velocity[0];
        file >> dummy >> object.velocity[1];
        file >> dummy >> object.velocity[2];
        file >> dummy;
        file >> dummy >> dummy;
        file >> dummy >> dummy;
        file >> dummy >> dummy;
        radar_objects.push_back(object);
    }

    file.close();
}

void ExecuteEstimate3Dvs3DPose(const std::vector<cv::Point3d>& camera_points,
                               const std::vector<cv::Point3d>& radar_points, 
                               cv::Mat& rot_matrix, cv::Mat& trans_matrix,
                               std::vector<float>& reproject_error)
{
    float repro_error_thresh = 40.0;
    reproject_error.resize(3);
    // center of mass
    cv::Point3d p1, p2;
    //求质心
    int N = camera_points.size();
    if (N < 4)
    {
        return;
    }

    for (int i = 0; i < N; i++)
    {
        p1 += camera_points[i];
        p2 += radar_points[i];
    }

    p1 = cv::Point3d(cv::Vec3d(p1) / N);
    p2 = cv::Point3d(cv::Vec3d(p2) / N);
    // remove the center
    std::vector<cv::Point3d> q1(N), q2(N);
    //去质心
    for (int i = 0; i < N; i++)
    {
        q1[i] = camera_points[i] - p1;
        q2[i] = radar_points[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    // std::cout << "W=" << W << std::endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // std::cout << "U=" << U << std::endl;
    // std::cout << "V=" << V << std::endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    if (R_.determinant() < 0)
    {
        R_ = -R_;
    }
    // std::cout << " 3D23D:radar to camera rotation " << R_ << std::endl;
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);
 
    //计算重投影误差，将第二组点投影到第一组坐标系下
    float mean_repro_error = 0.0;
    float error_tmp = 0.0;

    float error_tmp_x = 0.0;
    float error_tmp_y = 0.0;
    float error_tmp_z = 0.0;

    std::vector<float> error_vec;
    std::vector<cv::Point3d> inlier_camera_points;
    std::vector<cv::Point3d> inlier_radar_points;
    for (int idx = 0; idx < N; ++idx)
    {
        Eigen::Vector3d temp_p_w = R_ * Eigen::Vector3d(radar_points[idx].x, radar_points[idx].y, radar_points[idx].z) + t_;

        Eigen::Vector3d error = temp_p_w - Eigen::Vector3d(camera_points[idx].x, camera_points[idx].y, camera_points[idx].z);
        error_tmp = sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2]);
        if (error_tmp <= repro_error_thresh)
        {
            inlier_radar_points.push_back(radar_points[idx]);
            inlier_camera_points.push_back(camera_points[idx]);
        }
        error_vec.push_back(error_tmp);
        mean_repro_error += error_tmp;
        error_tmp_x += std::abs(error[0]);
        error_tmp_y += std::abs(error[1]);
        error_tmp_z += std::abs(error[2]);

    }

    mean_repro_error /= N;

    reproject_error[0] = error_tmp_x / N;
    reproject_error[1] = error_tmp_y / N;
    reproject_error[2] = error_tmp_z / N;

    // convert to cv::Mat
    rot_matrix = (cv::Mat_<double>(3, 3) <<
        R_(0, 0), R_(0, 1), R_(0, 2),
        R_(1, 0), R_(1, 1), R_(1, 2),
        R_(2, 0), R_(2, 1), R_(2, 2));
    trans_matrix = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

}
