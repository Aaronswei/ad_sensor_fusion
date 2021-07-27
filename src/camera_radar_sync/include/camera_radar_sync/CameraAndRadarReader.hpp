#ifndef _CAMERA_AND_RADAR_READER_HPP_
#define _CAMERA_AND_RADAR_READER_HPP_

#include <vector>
#include <opencv2/opencv.hpp>


class CameraAndRadarReader
{

public:
    CameraAndRadarReader(std::string basedir);
    void getRadarData(cv::string radar_file);
    void getImages(std::vector<cv::String> image_files);

    void cameraAndRadarSync();

private:
    

};

#endif