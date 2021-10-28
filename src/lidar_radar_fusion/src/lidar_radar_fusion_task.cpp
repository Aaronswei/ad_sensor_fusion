#include <stdlib.h>
#include <vector>
#include <glog/logging.h>
#include "opencv2/opencv.hpp"
#include "lidar_radar_fusion_task.hpp"

using namespace lidar_radar_fusion;

LidarRadarFusionTask::LidarRadarFusionTask() {
    _lidar_data_list.clear();
    _radar_data_list.clear();

    lidar_radar_fusion = std::make_shared<LidarRadarFusion>();

}

LidarRadarFusionTask::~LidarRadarFusionTask() {
    std::vector<cv::String>().swap(_lidar_data_list);
    std::vector<cv::String>().swap(_radar_data_list);

}

bool LidarRadarFusionTask::LidarRadarFusionTask::Init(const std::vector<cv::String>& lidar_data_list,
                                                      const std::vector<cv::String>& radar_data_list) {
    if(lidar_data_list.empty() || radar_data_list.empty()) {
        LOG(INFO) << "no data list ...";
        return false;
    }
    _lidar_data_list.clear();
    _radar_data_list.clear();

    _lidar_data_list.reserve(lidar_data_list.size());
    _radar_data_list.reserve(radar_data_list.size());

    _lidar_data_list.assign(lidar_data_list.begin(),lidar_data_list.end());
    _radar_data_list.assign(radar_data_list.begin(),radar_data_list.end());
    
    ParseDataPath(_lidar_data_list, _lidar_data_list_map);
    LOG(INFO) << _lidar_data_list_map.size();
    // for(const auto& item:_lidar_data_list_map) {
    //     std::cout <<"key:" << item.first << "; value:" << item.second << std::endl;\
    // }
    ParseDataPath(_radar_data_list, _radar_data_list_map);
    LOG(INFO) << _radar_data_list_map.size();

    // time sync
    _lidar_data_time_sync.clear();
    _radar_data_time_sync.clear();
    DataSync();
    
    //fusion
    lidar_radar_fusion->Init(_lidar_data_time_sync,_radar_data_time_sync);

    _is_initialized = true;
    return true;
}

bool LidarRadarFusionTask::LidarRadarFusionTask::Run() {
    if(!_is_initialized) {
        LOG(INFO) << __LINE__ << "please initialize LidarRadarFusionTask firstly ...";
        return false;
    }

    lidar_radar_fusion->Run();

    return true;
}

void LidarRadarFusionTask::ParseDataPath(const std::vector<cv::String>& data_list, 
                                         std::unordered_map<std::string, std::string>& data_list_map) {
    if(data_list.empty()) {
        LOG(INFO) << __LINE__ << "no data list ...";
        return;
    }

    for(const auto& item:data_list) {
        std::string tmp(item);
        std::string file_name = tmp.substr(tmp.size() - 20,-1);
        std::string path = tmp.substr(0,tmp.size() - 20);
        data_list_map.insert(std::pair<std::string, std::string>(file_name,path));
    }

}

void LidarRadarFusionTask::DataSync() {
    if(_lidar_data_list_map.empty() || _radar_data_list_map.empty()) {
        LOG(INFO) << __LINE__ ;
        return;
    }

    const TimeStamp diff_thre = 50000; // us
    std::unordered_map<std::string, std::string>::const_iterator lidar_iter = _lidar_data_list_map.begin();
    std::unordered_map<std::string, std::string>::const_iterator radar_iter = _radar_data_list_map.begin();
    for(;lidar_iter != _lidar_data_list_map.end();lidar_iter++) {
        TimeStamp lidar_time_stamp = atoll(lidar_iter->first.substr(0, lidar_iter->first.size()-4).c_str());


        for(;radar_iter != _radar_data_list_map.end();radar_iter++) {
            TimeStamp radar_time_stamp = atoll(radar_iter->first.substr(0, radar_iter->first.size()-4).c_str());
            if(std::abs(lidar_time_stamp - radar_time_stamp) <= diff_thre ) {
                std::cout <<"lidar_time_stamp:"<< lidar_time_stamp <<";radar_time_stamp:"<< radar_time_stamp << std::endl;

                std::vector<ObjectInfo> lidar_objects;
                LoadObjectsFromFile(lidar_time_stamp,lidar_iter->second + lidar_iter->first, lidar_objects,true);
                _lidar_data_time_sync.insert(std::make_pair(lidar_time_stamp,lidar_objects));

                std::vector<ObjectInfo> radar_objects;
                LoadObjectsFromFile(radar_time_stamp,radar_iter->second + radar_iter->first, radar_objects);
                _radar_data_time_sync.insert(std::make_pair(radar_time_stamp,radar_objects));
            }

        }

    } 

}

void LidarRadarFusionTask::LoadObjectsFromFile(const TimeStamp& time_stamp,
                                               const std::string& file_path, 
                                               std::vector<ObjectInfo>& objects, 
                                               bool is_lidar) {
    if(file_path.empty()) {
        return;
    }

    std::ifstream file(file_path, std::ios::in);
    if(!file.is_open()) {
        LOG(INFO) << "could not load file: " << file_path;
        return;
    }

    std::string line;
    ObjectInfo object_info;
    int id = -1;
    while(std::getline(file, line)) {
        ++id;
        // LOG(INFO) << "line: " << line;
        std::vector<std::string> split_info;
        if(is_lidar) {
            split_info = Split(line, " ");
        } else {
            split_info = Split(line, ",");
        }
        object_info.id = id;
        object_info.time_stamp = time_stamp;
        object_info.position << std::stod(split_info[0]),std::stod(split_info[1]),0.0; // object points pf radar do not contain Z value
    }


}