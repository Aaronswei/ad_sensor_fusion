#include <iostream>
#include <glog/logging.h>
#include "utils.hpp"
#include "ekfInterface.hpp"
#include "radar_camera_spatial_sync.hpp"

using RadarCameraSpatialSyncPtr = std::shared_ptr<radar_camera_spatial_sync::RadarCameraSpatialSync>;

int main() {
    std::string calib_file_path = "./datasets/practice_2_4_camera_radar_data/front_left_right_camera.yml";
    std::string left_camera_path = "./datasets/practice_2_4_camera_radar_data/left_camera";
    std::string left_camera_label = "./datasets/practice_2_4_camera_radar_data/label_left_camera";
    std::string right_camera_path = "./datasets/practice_2_4_camera_radar_data/right_camera";
    std::string right_camera_label = "./datasets/practice_2_4_camera_radar_data/label_right_camera";
    std::string front_radar_path = "./datasets/practice_2_4_camera_radar_data/radar_front_center";
    EKF_API *ekf_api = new EKF_API();
    //left
    std::vector<cv::String> left_camera_datas, left_camera_labels;
    radar_camera_spatial_sync::GetFileLists(left_camera_path,left_camera_label,left_camera_datas, left_camera_labels);
    if(left_camera_datas.size() != left_camera_labels.size()) {
        LOG(INFO) << "invalib data size ... ";
        return -1;
    }
    //right
    std::vector<cv::String> right_camera_datas, right_camera_labels;
    radar_camera_spatial_sync::GetFileLists(right_camera_path,right_camera_label,right_camera_datas, right_camera_labels);
    if(right_camera_datas.size() != right_camera_labels.size()) {
        LOG(INFO) << "invalib data size ... ";
        return -1;
    }

    bool ret_status = false;
    RadarCameraSpatialSyncPtr radar_camera_spatial_sync_ptr(new radar_camera_spatial_sync::RadarCameraSpatialSync());
    ret_status = radar_camera_spatial_sync_ptr->SetCalibFilePath(calib_file_path);
    
    ret_status = radar_camera_spatial_sync_ptr->Init();
    if(!ret_status) {
        return -1;
    }


    if(left_camera_datas.size() != right_camera_datas.size()) {
        LOG(INFO) << "left camera data size is not equal to right camera data size ... ";
        return -1;
    }

    std::vector<cv::Point3d> radar_points_vec, left_camera_points_vec;
    for(size_t idx = 0; idx < left_camera_datas.size(); ++idx) {
        //left
        cv::Mat left_img;
        radar_camera_spatial_sync::LoadImage(left_camera_datas[idx], left_img);

        std::vector<std::string> img_path_split = radar_camera_spatial_sync::Split(left_camera_datas[idx],"/");

        std::string img_name = img_path_split[img_path_split.size() - 1];
        std::string temp_img_name = img_name;
        std::string label_name = temp_img_name.replace(temp_img_name.rfind("."), 4,".txt");

        std::vector<cv::Rect2d> left_dets;
        radar_camera_spatial_sync::LoadTXTLabel(left_camera_label,label_name,left_dets);

        //right
        cv::Mat right_img;
        radar_camera_spatial_sync::LoadImage(right_camera_path + "/" + img_name, right_img);
        std::vector<cv::Rect2d> right_dets;
        radar_camera_spatial_sync::LoadTXTLabel(right_camera_label,label_name,right_dets);

        //radar
        std::vector<radar_camera_spatial_sync::RadarInfo_t> radar_infos;
        radar_camera_spatial_sync::LoadRadarInfo(front_radar_path + "/" + label_name, radar_infos);
        
        // run
        std::vector<cv::Point3d> radar_points, left_camera_points;
        ret_status = radar_camera_spatial_sync_ptr->Run(left_img,right_img,left_dets,right_dets, radar_infos, left_camera_points,radar_points);
        if(!ret_status) {
            return -1;
        }

        if((radar_points.size() == left_camera_points.size())&&(1 == radar_points.size())) {
            radar_points_vec.emplace_back(radar_points[0]);
            left_camera_points_vec.emplace_back(left_camera_points[0]);
        }
    }

    // calculate transform between left camera and radar
    LOG(INFO) << "radar_points_vec.size() = "<< radar_points_vec.size();
    LOG(INFO) << "left_camera_points = "<< left_camera_points_vec.size();

    cv::Mat rotation, translation;
    std::vector<float> error;
    radar_camera_spatial_sync::ExecuteEstimate3Dvs3DPose(left_camera_points_vec,radar_points_vec,rotation,translation,error);
    std::cout <<"rotation of radar to left camera: "<< std::endl << rotation << std::endl;
    std::vector<cv::Point3d> result_points;
    radar_camera_spatial_sync::Point2Camera(radar_points_vec, result_points, rotation, translation);

    std::vector<Measurement> measurement_pack_list;

    for(auto &result : result_points)
    {
        Measurement temp;
		temp.raw_measurements_ = VectorXd(2);
        temp.raw_measurements_ << result.x, result.z;
        temp.timestamp_ = 50; //ms
        measurement_pack_list.push_back(temp); 
    }
    ekf_api->process(measurement_pack_list);
    return 0;
}