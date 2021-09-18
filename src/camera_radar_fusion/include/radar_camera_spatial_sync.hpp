#ifndef __STEREO_CAMERA_DISTANCE_ESTIMATION_HPP_
#define __STEREO_CAMERA_DISTANCE_ESTIMATION_HPP_
#include <string>
#include <memory>
#include <glog/logging.h>
#include <boost/filesystem.hpp>
#include "opencv2/opencv.hpp"
#include "utils.hpp"


namespace radar_camera_spatial_sync {

    const float DISTANCE_MAX = 80.0;// m
    const double RADAR_POINT_LONG_MIN = 2.0;// m
    const double RADAR_POINT_LONG_MAX = 60.0; 
    const double RADAR_POINT_LATERAL_MIN = -0.5; 
    const double RADAR_POINT_LATERAL_MAX = 0.5; 



    struct StereoCalib {
        int height;
        int width;
        cv::Mat R, T, Q;
        cv::Mat R1, R2;
        cv::Mat P1, P2;
        cv::Mat M1, M2;
        cv::Mat D1, D2;

        cv::Mat map1x, map1y, map2x, map2y;
        void rectify(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& remap_left_img, cv::Mat& remap_right_img);
    };

    class RadarCameraSpatialSync {
    public:
        RadarCameraSpatialSync();
        ~RadarCameraSpatialSync() = default;

        bool SetCalibFilePath(const std::string& calib_file_path) ;
        bool Init();
        bool Run(const cv::Mat& left_img,
                 const cv::Mat& right_img,
                 const std::vector<cv::Rect2d>& left_dets,
                 const std::vector<cv::Rect2d>& right_dets,
                 const std::vector<RadarInfo_t>& radar_infos,
                 std::vector<cv::Point3d>& stere_camera_points,
                 std::vector<cv::Point3d>& radar_points);

    private:
        //! load calib file
        bool LoadStereoCalib();

        void EstimateDistance(const cv::Mat& left_img,
                              const cv::Mat& right_img, 
                              const std::vector<cv::Rect2d>& left_bboxes,
                              const std::vector<cv::Rect2d>& right_bboxes);
        void UndistortRectInfo(const std::vector<cv::Rect2d>& left_bboxes, 
                               const std::vector<cv::Rect2d>& right_bboxes);
        void CalculateDistance(const cv::Mat& left_img,
                               const cv::Mat& right_img);
        void DistanceLimitation();

        void RadarFilter(std::vector<RadarInfo_t>& radar_infos);


        bool _is_initialized = false;
        std::shared_ptr<StereoCalib> _stereo_calib;

        std::string _calib_file_path;

        std::vector<cv::String> _left_camera_imgs;
        std::vector<cv::String> _left_camera_labels;

        std::vector<cv::String> _right_camera_imgs;
        std::vector<cv::String> _right_camera_labels;

        std::vector<cv::Rect2d> _undistort_left_bboxes;
        std::vector<cv::Rect2d> _undistort_right_bboxes;
        std::vector<cv::Point3d> _stereo_distance_results;
    };

}


#endif // __STEREO_CAMERA_DISTANCE_ESTIMATION_HPP_