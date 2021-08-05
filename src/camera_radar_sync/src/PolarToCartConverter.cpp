#include "camera_radar_sync/PolarToCartConverter.hpp"

PolarToCartConverter::PolarToCartConverter() {
    height_ = 1.46;
    pitch_ = -0.023;
    yaw_ = 0;
    pixelWidth_ = pow(3.75, -6);
    Pixelheight_ = pow(3.75, -6);  //
    imgCols_ = 1280;
    imgRows_ = 960;
    focalLength_ = pow(6, -3);
    lateralOffset_ = 0;
    radarheight_ = 0;
    radarLateralOffset_ = 0;
    cameraToRadarDist_ = 1.65;
}

std::vector<int> PolarToCartConverter::getImageCoordinates(const camera_radar_sync::RadarMsg::ConstPtr& radar_msg) {
    std::vector<int> camCoord_;
    int u32_Cameraheight_ = height_;
    int  u32_Radarheight_ = radarheight_;
    double  u32_RadarRange = radar_msg->radarRange_;
    int  u32_cameraToRadarDist_ance = cameraToRadarDist_;
    double  f32_RadarAngle = radar_msg->radarAngle_;
    f32_RadarAngle = -f32_RadarAngle;
    int  s32_CameraOffset = lateralOffset_;
    int  s32_RadarOffset = radarLateralOffset_;
    int  f32_Camerayaw__rad = yaw_;
    double f32_distanceObj;
    double u32_RadarRangeLong = u32_RadarRange * cos(f32_RadarAngle);
    double f32_distanceObj_temp = ((u32_cameraToRadarDist_ance + u32_RadarRangeLong) / cos(f32_Camerayaw__rad))+\
    sin(f32_Camerayaw__rad)*((u32_cameraToRadarDist_ance + u32_RadarRangeLong) * tan(f32_Camerayaw__rad)
    - (u32_RadarRangeLong * tan(f32_RadarAngle) + s32_CameraOffset - s32_RadarOffset));

    if (u32_Radarheight_ == 0)
        f32_distanceObj = f32_distanceObj_temp;
    else
        f32_distanceObj = f32_distanceObj_temp * u32_Cameraheight_ / (u32_Cameraheight_ - u32_Radarheight_);
    
    double f32_widthObj = -cos(f32_Camerayaw__rad) * ((u32_cameraToRadarDist_ance + u32_RadarRangeLong) *\
    tan(f32_Camerayaw__rad) -(u32_RadarRangeLong * tan(f32_RadarAngle) + s32_CameraOffset - s32_RadarOffset));
    long double u32YCord = realWorldDistanceToImageRow(f32_distanceObj);
    long double u32XCord = realWorldWidthToImageCol(f32_distanceObj_temp, f32_widthObj);
    camCoord_.reserve(2);
    camCoord_.push_back(u32XCord);
    camCoord_.push_back(u32YCord);
    return camCoord_;
}

double PolarToCartConverter::realWorldDistanceToImageRow(double f32_distance) {
    int u32_SensorRows = imgRows_;
    double s32Cameraheight_ = height_;
    double f32_Camerapitch__rad = pitch_;
    double f32_CameraFocusLength = focalLength_;
    double f32_SensorPixelheight_ = Pixelheight_;
    double f32_ImageRow = u32_SensorRows / 2 + (s32Cameraheight_ * cos(f32_Camerapitch__rad) + f32_distance *\
    sin(f32_Camerapitch__rad)) / (f32_distance * cos(f32_Camerapitch__rad) - s32Cameraheight_ *
    sin(f32_Camerapitch__rad)) * f32_CameraFocusLength / f32_SensorPixelheight_;
    return f32_ImageRow;
}

double PolarToCartConverter::realWorldWidthToImageCol(double f32_distance, double f32_width) {
    int u32_SensorCols = imgCols_;
    double s32Cameraheight_ = height_;
    double f32_CameraFocusLength = focalLength_;
    double f32_SensorpixelWidth_ = pixelWidth_;
    double f32_width_Pixel = ((f32_CameraFocusLength * f32_width) / sqrt(static_cast<float>(s32Cameraheight_ *
    s32Cameraheight_)+ static_cast<float>(f32_distance * f32_distance))) / f32_SensorpixelWidth_;
    double f32_ImageCol = static_cast<float>((static_cast<float>(u32_SensorCols / 2) - f32_width_Pixel));
    return (f32_ImageCol);
}

