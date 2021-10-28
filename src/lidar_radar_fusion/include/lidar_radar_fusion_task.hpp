#ifndef __LIDAR_RADAR_FUSION_TASK_HPP__
#define __LIDAR_RADAR_FUSION_TASK_HPP__
#include "utils.hpp"
#include "lidar_radar_fusion.hpp"

namespace lidar_radar_fusion {
    
    class LidarRadarFusionTask {
    public:

        LidarRadarFusionTask();
        ~LidarRadarFusionTask();

        bool Init(const std::vector<cv::String>& lidar_data_list,
                  const std::vector<cv::String>& radar_data_list);
        bool Run();
    private:
        void ParseDataPath(const std::vector<cv::String>& data_list, 
                           std::unordered_map<std::string, std::string>& data_list_map);
        void DataSync();
        void LoadObjectsFromFile(const TimeStamp& time_stamp,const std::string& file_path, std::vector<ObjectInfo>& objects, bool is_lidar = false);
    private:
        bool _is_initialized = false;
        std::vector<cv::String> _lidar_data_list;
        std::vector<cv::String> _radar_data_list;

        std::unordered_map<std::string, std::string> _lidar_data_list_map; // <file_name,path>
        std::unordered_map<std::string, std::string> _radar_data_list_map; // <file_name,path>

        std::map<TimeStamp, std::vector<ObjectInfo> > _lidar_data_time_sync;
        std::map<TimeStamp, std::vector<ObjectInfo> > _radar_data_time_sync;

        std::shared_ptr<LidarRadarFusion> lidar_radar_fusion;;

    };
}

#endif // __LIDAR_RADAR_FUSION_TASK_HPP__