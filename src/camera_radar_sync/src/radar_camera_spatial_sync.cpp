#include <fstream>
#include "eigen3/Eigen/Dense"
#include "opencv2/core/eigen.hpp"
#include "radar_camera_spatial_sync.hpp"

using namespace radar_camera_spatial_sync;


void StereoCalib::rectify(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& remap_left_img, cv::Mat& remap_right_img) {
    cv::remap(left_img, remap_left_img, map1x, map1y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    cv::remap(right_img, remap_right_img, map2x, map2y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
}

RadarCameraSpatialSync::RadarCameraSpatialSync()
    :_stereo_calib(nullptr), _calib_file_path("") {
    _stereo_calib.reset(new StereoCalib());

}


bool RadarCameraSpatialSync::SetCalibFilePath(const std::string& calib_file_path) {
    _calib_file_path = calib_file_path;
    return true;

}

bool RadarCameraSpatialSync::Init() {

    bool ret_status = false;
    ret_status = LoadStereoCalib();
    if(!ret_status) {
        LOG(INFO) << "Please check calib file ... ";
        return ret_status;
    }


    _is_initialized = true;

    return true;
}

bool RadarCameraSpatialSync::Run(const cv::Mat& left_img,
                                const cv::Mat& right_img,
                                const std::vector<cv::Rect2d>& left_dets,
                                const std::vector<cv::Rect2d>& right_dets,
                                const std::vector<RadarInfo_t>& radar_infos,
                                std::vector<cv::Point3d>& left_camera_points,
                                std::vector<cv::Point3d>& radar_points) {
    if(!_is_initialized) {
        return false;
    }
    _stereo_distance_results.clear();
    left_camera_points.clear();
    radar_points.clear();
    // estimate distance
    EstimateDistance(left_img,right_img, left_dets, right_dets);
    // LOG(INFO) << "radar_infos = "<< radar_infos.size();
	// convert 3d postion from stereo to left
	Eigen::Matrix3d R1_inv;
	cv::cv2eigen(_stereo_calib->R1.inv(), R1_inv);
    for (auto &iter : _stereo_distance_results) {
        Eigen::Vector3d stereo_point_cam(iter.x,iter.y,iter.z);
        Eigen::Vector3d left_camera_point = R1_inv * stereo_point_cam;
        iter.x = left_camera_point[0];
        iter.y = left_camera_point[1];
        iter.z = left_camera_point[2];
	}
    left_camera_points.reserve(_stereo_distance_results.size());
    left_camera_points.swap(_stereo_distance_results);

    //radar filter
    std::vector<RadarInfo_t> radar_infos_filter = radar_infos;
    RadarFilter(radar_infos_filter);
    for(auto& iter:radar_infos_filter) {
        cv::Point3d radar_point_cv;
        radar_point_cv.x = iter.position[0];
        radar_point_cv.y = iter.position[1];
        radar_point_cv.z = CAR_HEIGHT/2;
        radar_points.emplace_back(radar_point_cv);
    }
    // LOG(INFO) << "radar_infos_filter = "<< radar_infos_filter.size();
    // LOG(INFO) << "_stereo_distance_results = "<< stere_camera_points.size();

    for(size_t idx = 0;idx < left_camera_points.size(); ++idx) {
        LOG(INFO) << "radar_point:"<<"(" << radar_points[idx].x<< ", " << radar_points[idx].y <<"," << radar_points[idx].z << ")";
        LOG(INFO) << "left_camera_points:"<<"(" << left_camera_points[idx].x << ", " << left_camera_points[idx].y <<"," << left_camera_points[idx].z << ")";
    }

    return true;
}


bool RadarCameraSpatialSync::LoadStereoCalib() {
    cv::FileStorage fs(_calib_file_path, cv::FileStorage::READ);
    if(!fs.isOpened()) {
        LOG(WARNING) <<"nofile: " << _calib_file_path;
        return false;
    }

    fs["height"] >> _stereo_calib->height;
    fs["width"] >> _stereo_calib->width;
    fs["R1"] >> _stereo_calib->R1;
    fs["R2"] >> _stereo_calib->R2;
    fs["P1"] >> _stereo_calib->P1;
    fs["P2"] >> _stereo_calib->P2;
    fs["Q"] >> _stereo_calib->Q;
    fs["M1"] >> _stereo_calib->M1;
    fs["M2"] >> _stereo_calib->M2;
    fs["D1"] >> _stereo_calib->D1;
    fs["D2"] >> _stereo_calib->D2;

    fs["R"] >> _stereo_calib->R;
    fs["T"] >> _stereo_calib->T;
    if(_stereo_calib->T.rows != 3) {
        _stereo_calib->T = _stereo_calib->T.t();
    }
    fs.release();

    StereoCalib& calib = *_stereo_calib;
    if(calib.map1x.empty() || calib.map1y.empty() || calib.map2x.empty() || calib.map2y.empty()) {
        cv::initUndistortRectifyMap(calib.M1, calib.D1, calib.R1, calib.P1, cv::Size(calib.width, calib.height), CV_16SC2, calib.map1x, calib.map1y);
        cv::initUndistortRectifyMap(calib.M2, calib.D2, calib.R2, calib.P2, cv::Size(calib.width, calib.height), CV_16SC2, calib.map2x, calib.map2y);
    }

    return true;
}


void RadarCameraSpatialSync::UndistortRectInfo(const std::vector<cv::Rect2d>& left_bboxes, const std::vector<cv::Rect2d>& right_bboxes) {
    for(size_t idx = 0; idx < left_bboxes.size(); ++idx) {
        // left 
        std::vector<cv::Point2d> corner_points;
        corner_points.emplace_back(cv::Point2d(left_bboxes[idx].x, left_bboxes[idx].y));
        corner_points.emplace_back(cv::Point2d(left_bboxes[idx].x + left_bboxes[idx].width, left_bboxes[idx].y + left_bboxes[idx].height));

		std::vector<cv::Point2d> undistort_corner_points;
		cv::undistortPoints(corner_points, undistort_corner_points, _stereo_calib->M1, _stereo_calib->D1, _stereo_calib->R1, _stereo_calib->P1);

        cv::Rect2d undistort_rect(undistort_corner_points[0].x, undistort_corner_points[0].y,
                                  undistort_corner_points[1].x - undistort_corner_points[0].x,
                                  undistort_corner_points[1].y - undistort_corner_points[0].y);
        _undistort_left_bboxes.emplace_back(undistort_rect);

        //right
        corner_points.clear();
		undistort_corner_points.clear();

        corner_points.emplace_back(cv::Point2d(right_bboxes[idx].x, right_bboxes[idx].y));
        corner_points.emplace_back(cv::Point2d(right_bboxes[idx].x + right_bboxes[idx].width, right_bboxes[idx].y + right_bboxes[idx].height));

		cv::undistortPoints(corner_points, undistort_corner_points, _stereo_calib->M1, _stereo_calib->D1, _stereo_calib->R1, _stereo_calib->P1);

        undistort_rect = cv::Rect2d(undistort_corner_points[0].x, undistort_corner_points[0].y,
                                    undistort_corner_points[1].x - undistort_corner_points[0].x,
                                    undistort_corner_points[1].y - undistort_corner_points[0].y);
        _undistort_right_bboxes.emplace_back(undistort_rect);
    }
}

void RadarCameraSpatialSync::CalculateDistance(const cv::Mat& left_img,
                                             const cv::Mat& right_img) {

    for(size_t idx = 0; idx < _undistort_left_bboxes.size(); ++idx) {
        std::vector<cv::Point2d> left_center_points, right_center_points;
        left_center_points.emplace_back(cv::Point2d(_undistort_left_bboxes[idx].x + _undistort_left_bboxes[idx].width * 0.5,
										            _undistort_left_bboxes[idx].y + _undistort_left_bboxes[idx].height * 0.5));

        right_center_points.emplace_back(cv::Point2d(_undistort_right_bboxes[idx].x + _undistort_right_bboxes[idx].width * 0.5,
										             _undistort_right_bboxes[idx].y + _undistort_right_bboxes[idx].height * 0.5));

        std::vector<cv::Point3d> triangle_points_3d;
	    cv::Mat triangle_points_camera;
	    cv::triangulatePoints(_stereo_calib->P1, _stereo_calib->P2, left_center_points, right_center_points, triangle_points_camera);

        cv::Point3d pt_3d;
        double w = triangle_points_camera.at<double>(3, 0);
        pt_3d.x = triangle_points_camera.at<double>(0, 0) / w;
        pt_3d.y = triangle_points_camera.at<double>(1, 0) / w;
        pt_3d.z = triangle_points_camera.at<double>(2, 0) / w;

        _stereo_distance_results.emplace_back(pt_3d);
    }
}

void RadarCameraSpatialSync::DistanceLimitation() {
    for (auto iter = _stereo_distance_results.begin(); iter != _stereo_distance_results.end();) {
		if (iter->z > DISTANCE_MAX) {
			iter = _stereo_distance_results.erase(iter);
			continue;
		}
		++iter;
	}
}

void RadarCameraSpatialSync::EstimateDistance(const cv::Mat& left_img,
                                            const cv::Mat& right_img, 
                                            const std::vector<cv::Rect2d>& left_bboxes,
                                            const std::vector<cv::Rect2d>& right_bboxes) {
    _undistort_left_bboxes.clear();
    _undistort_right_bboxes.clear();

    //undistort rect
    UndistortRectInfo(left_bboxes, right_bboxes);
    //calculate depth
    CalculateDistance(left_img,right_img);
    //limit distance 
    DistanceLimitation();
}

void RadarCameraSpatialSync::RadarFilter(std::vector<RadarInfo_t>& radar_infos) {

    for(auto iter= radar_infos.begin(); iter!= radar_infos.end();) {
        if(iter->position[0] < RADAR_POINT_LONG_MIN || 
           iter->position[0] > RADAR_POINT_LONG_MAX ||
           iter->position[1] < RADAR_POINT_LATERAL_MIN ||
           iter->position[1] > RADAR_POINT_LATERAL_MAX) {

               iter = radar_infos.erase(iter);
               continue;
        }
        ++iter;
    }

    return;
}
