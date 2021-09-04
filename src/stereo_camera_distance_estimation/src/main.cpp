#include <iostream>
#include <glog/logging.h>
#include "stereo_camera_distance_estimation.hpp"

int main() {
    std::string calib_file_path = "/home/kavin/data/stereo_camera_data/front_left_right_camera.yml";
    std::string left_camera_path = "/home/kavin/data/stereo_camera_data/left_camera";
    std::string left_camera_label = "/home/kavin/data/stereo_camera_data/label_left_camera";
    std::string right_camera_path = "/home/kavin/data/stereo_camera_data/right_camera";
    std::string right_camera_label = "/home/kavin/data/stereo_camera_data/label_right_camera";
    
    bool ret_status = false;
    stereo_camera::StereoCameraDistance stereo_distance;
    ret_status = stereo_distance.SetCameraDataPath(calib_file_path,
                                                   left_camera_path,
                                                   left_camera_label,
                                                   right_camera_path,
                                                   right_camera_label);
    
    ret_status = stereo_distance.Init();
    if(!ret_status) {
        return -1;
    }

    ret_status = stereo_distance.Run();
    if(!ret_status) {
        return -1;
    }

    return 0;
}