#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utils.hpp"
#include "lidar_radar_fusion_task.hpp"
using namespace lidar_radar_fusion;

int main() 
{
    const std::string lidar_data_path = "/home/kavin/data/practice_2_6_radar_lidar_fusion/lidar_data_txt";
    const std::string radar_data_path = "/home/kavin/data/practice_2_6_radar_lidar_fusion/radar_data_txt";

    // load lidar data
    std::vector<cv::String> lidar_data_list;
    GetFileLists(lidar_data_path, lidar_data_list);
    std::cout << "lidar_data_list.size() = " << lidar_data_list.size() << std::endl;

    // load rada data
    std::vector<cv::String> radar_data_list;
    GetFileLists(radar_data_path,radar_data_list);
    std::cout << "radar_data_list.size() = " << radar_data_list.size() << std::endl;
    
    bool ret = false;
    std::shared_ptr<LidarRadarFusionTask> task = std::make_shared<LidarRadarFusionTask>();
    ret = task->Init(lidar_data_list,radar_data_list);
    ret = task->Run();
    

    return 0;
}