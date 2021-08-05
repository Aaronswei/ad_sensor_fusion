// Copyright 2019 KPIT  [legal/copyright]

#ifndef _POLARTOCARTCONVERTER_HPP_
#define _POLARTOCARTCONVERTER_HPP_

#include <bits/stdc++.h>
#include <rosbag/view.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <queue>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "camera_radar_sync/RadarMsg.h"

class PolarToCartConverter {
 protected:
    int yaw_;
    int imgCols_;
    int imgRows_;
    int lateralOffset_;
    int radarheight_;
    int radarLateralOffset_;
    int cameraToRadarDist_;
    double height_;
    double pitch_;
    double pixelWidth_;
    double Pixelheight_;
    double focalLength_;
 public:
    PolarToCartConverter();
    std::vector<int> getImageCoordinates(const camera_radar_fusion::RadarMsg::ConstPtr& radar_msg);
    double realWorldDistanceToImageRow(double f32_distance);
    double realWorldWidthToImageCol(double f32_distance, double f32_width);
};

#endif  // _POLARTOCARTCONVERTER_HPP_
