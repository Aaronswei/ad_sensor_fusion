#ifndef __STEREO_CAMERA_DISTANCE_ESTIMATION_HPP_
#define __STEREO_CAMERA_DISTANCE_ESTIMATION_HPP_
#include <string>
#include <memory>
#include <glog/logging.h>
#include <boost/filesystem.hpp>


#include "opencv2/opencv.hpp"
namespace stereo_camera {

    

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


    class StereoCameraDistance {
    public:
        StereoCameraDistance();
        ~StereoCameraDistance() = default;

        bool SetCameraDataPath(const std::string& calib_file_path,
                               const std::string& left_camera_path, 
                               const std::string& left_camera_label,
                               const std::string& right_camera_path, 
                               const std::string& right_camera_label);
        bool Init();
        bool Run();

    private:
        //! load calib file
        bool LoadStereoCalib();
        bool LoadDataFiles();

        void LoadTXTLabel(const std::string& label_path, const std::string& label_name, std::vector<cv::Rect2d>& dets);

        void LoadImage(const std::string& img_path, cv::Mat& img);

        void EstimateDistance(const cv::Mat& left_img,
                              const cv::Mat& right_img, 
                              const std::vector<cv::Rect2d>& left_bboxes,
                              const std::vector<cv::Rect2d>& right_bboxes);
        void UndistortRectInfo(const std::vector<cv::Rect2d>& left_bboxes, 
                               const std::vector<cv::Rect2d>& right_bboxes);
        void CalculateDistance(const cv::Mat& left_img,
                               const cv::Mat& right_img);
        std::vector<std::string> Split(const std::string& str, 
                                       const std::string& delim);
        void DistanceLimitation();


        size_t _img_nums = 0;
        bool _is_initialized = false;
        std::shared_ptr<StereoCalib> _stereo_calib;

        std::string _calib_file_path;
        std::string _left_camera_path;
        std::string _left_camera_label_path;
        std::string _right_camera_path;
        std::string _right_camera_label_path;

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