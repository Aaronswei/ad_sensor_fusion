#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <Eigen/Eigen>

#include <pcl/filters/random_sample.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

double calcAngleBetweenPointclouds(const pcl::PointCloud<pcl::PointXYZ>& prev_pointcloud, const pcl::PointCloud<pcl::PointXYZ>& pointcloud);

double calcAngleBetweenImages(const cv::Mat& prev_image, const cv::Mat& image, float focal_length);

#endif
