#include <fstream>
#include "stereo_camera_distance_estimation.hpp"

using namespace stereo_camera;


void StereoCalib::rectify(const cv::Mat& left_img, const cv::Mat& right_img, cv::Mat& remap_left_img, cv::Mat& remap_right_img) {
    cv::remap(left_img, remap_left_img, map1x, map1y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    cv::remap(right_img, remap_right_img, map2x, map2y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
}


StereoCameraDistance::StereoCameraDistance()
    :_stereo_calib(nullptr), _calib_file_path(""), _left_camera_path(""), _left_camera_label_path(""), 
    _right_camera_path(""), _right_camera_label_path(""){
    _stereo_calib.reset(new StereoCalib());

}

bool StereoCameraDistance::SetCameraDataPath(const std::string& calib_file_path,
                                             const std::string& left_camera_path, 
                                             const std::string& left_camera_label,
                                             const std::string& right_camera_path, 
                                             const std::string& right_camera_label) {
    _calib_file_path = calib_file_path;
    _left_camera_path = left_camera_path;
    _left_camera_label_path = left_camera_label;
    _right_camera_path = right_camera_path;
    _right_camera_label_path = right_camera_label;

    return true;

}

bool StereoCameraDistance::Init() {

    bool ret_status = false;
    ret_status = LoadStereoCalib();
    if(!ret_status) {
        LOG(INFO) << "Please check calib file ... ";
        return ret_status;
    }

    ret_status = LoadDataFiles();
    if(!ret_status) {
        return ret_status;
    }

    _is_initialized = true;

    return true;
}

bool StereoCameraDistance::Run() {
    if(!_is_initialized) {
        return false;
    }

    _stereo_distance_results.clear();
    _stereo_distance_results.reserve(_img_nums);
    for(size_t idx = 0; idx < _img_nums; ++idx) {
        const std::string left_img_path = _left_camera_imgs[idx];
        std::vector<std::string> img_path_split = Split(left_img_path,"/");

        std::string img_name = img_path_split[img_path_split.size() - 1];
        // LOG(INFO) << " img_name:" << img_name;
        std::string temp_img_name = img_name;
        std::string label_name = temp_img_name.replace(temp_img_name.rfind("."), 4,".txt");
        // left info
        cv::Mat left_img;
        LoadImage(left_img_path,left_img);
        std::vector<cv::Rect2d> left_detections;
        LoadTXTLabel(_left_camera_label_path, label_name, left_detections);

        // for(auto iter:left_detections) {
        //     cv::rectangle(left_img,iter, cv::Scalar(0,255,0));
        // }
        // cv::imshow("win" , left_img);
        // cv::waitKey();

        //right info
        cv::Mat right_img;
        LoadImage(_right_camera_path + "/" + img_name, right_img);
        std::vector<cv::Rect2d> right_detections;
        LoadTXTLabel(_right_camera_label_path, label_name, right_detections);

        // estimate distance
        EstimateDistance(left_img,right_img, left_detections, right_detections);
    }

    for(auto& iter:_stereo_distance_results) {
        LOG(INFO) << "x = " << iter.x << "; y = " << iter.y <<"; z = " << iter.z;
    }

    return true;
    

}


bool StereoCameraDistance::LoadStereoCalib() {
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

bool StereoCameraDistance::LoadDataFiles() {
    _left_camera_imgs.clear();
    _right_camera_imgs.clear();
    //! left camera
    cv::glob(_left_camera_path, _left_camera_imgs);
    cv::glob(_left_camera_label_path, _left_camera_labels);
    assert(_left_camera_imgs.size() == _left_camera_labels.size());
    LOG(INFO) << " left image size = " << _left_camera_imgs.size()<< " ; left image label size = " << _left_camera_labels.size();
    // for(size_t idx = 0; idx < _left_camera_imgs.size(); ++idx) {
    //     LOG(INFO) << " left image path: " << _left_camera_imgs[idx]<< " ; left image label path: " << _left_camera_labels[idx];
    // }
    //right camera
    cv::glob(_right_camera_path, _right_camera_imgs);
    cv::glob(_right_camera_label_path, _right_camera_labels);
    assert(_right_camera_imgs.size() == _right_camera_labels.size());
    LOG(INFO) << " right image size = " << _right_camera_imgs.size()<< " ; right image label size = " << _right_camera_labels.size();
    // for(size_t idx = 0; idx < _left_camera_imgs.size(); ++idx) {
    //     LOG(INFO) << " right image path: " << _right_camera_imgs[idx]<< " ; right image label path: " << _right_camera_labels[idx];
    // }

    // load labels of left camera and right camera
    assert(_left_camera_imgs.size() == _right_camera_imgs.size());
    assert(_left_camera_labels.size() == _right_camera_labels.size());

    _img_nums = _left_camera_imgs.size();
    return true;
 }

 void StereoCameraDistance::LoadTXTLabel(const std::string& label_path, const std::string& label_name, std::vector<cv::Rect2d>& dets) {
    dets.clear();
    cv::Rect2d bbox;
    std::string path = label_path + "/" + label_name;
    std::ifstream file(path, std::ios::in);
    if(!file.is_open()) {
        LOG(INFO) << "could not load file: " << path;
        return;
    }

    std::string line;
    while(std::getline(file, line)) {
        LOG(INFO) << "line: " << line;
        std::vector<std::string> split_info = Split(line, " ");
        bbox.width = std::stod(split_info[3]) * _stereo_calib->width;
        bbox.height = std::stod(split_info[4]) * _stereo_calib->height ;
        bbox.x = std::stod(split_info[1]) * _stereo_calib->width - bbox.width * 0.5;
        bbox.y = std::stod(split_info[2]) * _stereo_calib->height - bbox.height * 0.5;

        dets.emplace_back(bbox);
    }

}

void StereoCameraDistance::LoadImage(const std::string& img_path, cv::Mat& img) {
    img = cv::imread(img_path, true);
    if(img.empty()) {
        LOG(INFO) << "could not load image: " << img_path;
        return;
    }
}

std::vector<std::string> StereoCameraDistance::Split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> res;
    if ("] [" == str) return res;
    //先将要切割的字符串从string类型转换为char*类型
    char* strs = new char[str.length() + 1];
    strcpy(strs, str.c_str());

    char* des = new char[delim.length() + 1];
    strcpy(des, delim.c_str());

    char* p = strtok(strs, des);
    while (p) {
        std::string s = p; //分割得到的字符串转换为string类型
        res.push_back(s); //存入结果数组
        p = strtok(NULL, des);
    }

    delete[] strs;
    delete[] des;
    return res;
}

void StereoCameraDistance::UndistortRectInfo(const std::vector<cv::Rect2d>& left_bboxes, const std::vector<cv::Rect2d>& right_bboxes) {
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

void StereoCameraDistance::CalculateDistance(const cv::Mat& left_img,
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

void StereoCameraDistance::DistanceLimitation() {
    for (auto iter = _stereo_distance_results.begin(); iter != _stereo_distance_results.end();) {
		if (iter->z > 80.0) {
			iter = _stereo_distance_results.erase(iter);
			continue;
		}
		++iter;
	}
}

void StereoCameraDistance::EstimateDistance(const cv::Mat& left_img,
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