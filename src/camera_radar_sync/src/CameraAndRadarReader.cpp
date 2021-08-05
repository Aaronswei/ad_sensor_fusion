#include "camera_radar_sync/CameraAndRadarReader.hpp"

CameraAndRadarReader::CameraAndRadarReader(std::string basedir)
{

}

void CameraAndRadarReader::getRadarData(cv::string radar_file)
{

}

void CameraAndRadarReader::getImages(std::vector<cv::String> image_files)
{

}

void CameraAndRadarReader::cameraAndRadarSynchronize()
{
    // checking to verify if the buffers are not empty
    if (!(radarBuffer_.empty()) && !(frameBuffer_.empty())) {
        firstFrame_ = frameBuffer_.front();
        currentFrameID_ = firstFrame_->header.frame_id;
        currentRadarFrameID_ = radarBuffer_.front();
        // plot using the first queue element from both queues
        sync();                                            
        radarBuffer_.pop();
        frameBuffer_.pop();
        publish();
    } else if (radarBuffer_.empty()) {                     // Checking if Radar data is empty publish Camera data
        publishonedata();
    } else if (frameBuffer_.empty()) {
        ROS_INFO("Camera data is Empty");
    }
}

void CameraAndRadarReader::sync() {
    std::vector<int> cartCoOrdinates;
    cartCoOrdinates = polarToCartesianConverter_.getImageCoordinates(currentRadarFrameID_);
    cv::circle(firstFrame_->image, cv::Point2i(cartCoOrdinates[0], cartCoOrdinates[1]), 5, cv::Scalar(0, 0, 255), 4, 5);
}


void CameraAndRadarReader::publishonedata() {
    firstFrame_ = frameBuffer_.front();
    publish();
}

void CameraAndRadarReader::publish() {
    ROS_INFO("Publishing Fused Data");
    imagePublish_.publish(firstFrame_->toImageMsg());
}