//
// Created by rdcas on 2020/4/20.
//

#ifndef CAM_LIDAR_CALIBRATION_PROJECTOR_H
#define CAM_LIDAR_CALIBRATION_PROJECTOR_H

#include "ros/ros.h"
#include "std_msgs/Int8.h"
#include <termios.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "pcl_ros/transforms.h"
#include "camera_lidar_fusion/point_xyzir.h"
#include <Eigen/Dense>
#include <ros/package.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/flann.h>
#include <opencv2/opencv.hpp>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>


namespace extrinsic_calibration {
    class projector {
    private:

        cv::Mat undistort_map1, undistort_map2;

        image_transport::Publisher image_publisher;

        tf::TransformBroadcaster tf_br;

        double colmap[50][3] =  {{0,0,0.5385},{0,0,0.6154},{0,0,0.6923},
                                 {0,0,0.7692},{0,0,0.8462},{0,0,0.9231},
                                 {0,0,1.0000},{0,0.0769,1.0000},{0,0.1538,1.0000},
                                 {0,0.2308,1.0000},{0,0.3846,1.0000},{0,0.4615,1.0000},
                                 {0,0.5385,1.0000},{0,0.6154,1.0000},{0,0.6923,1.0000},
                                 {0,0.7692,1.0000},{0,0.8462,1.0000},{0,0.9231,1.0000},
                                 {0,1.0000,1.0000},{0.0769,1.0000,0.9231},{0.1538,1.0000,0.8462},
                                 {0.2308,1.0000,0.7692},{0.3077,1.0000,0.6923},{0.3846,1.0000,0.6154},
                                 {0.4615,1.0000,0.5385},{0.5385,1.0000,0.4615},{0.6154,1.0000,0.3846},
                                 {0.6923,1.0000,0.3077},{0.7692,1.0000,0.2308},{0.8462,1.0000,0.1538},
                                 {0.9231,1.0000,0.0769},{1.0000,1.0000,0},{1.0000,0.9231,0},
                                 {1.0000,0.8462,0},{1.0000,0.7692,0},{1.0000,0.6923,0},
                                 {1.0000,0.6154,0},{1.0000,0.5385,0},{1.0000,0.4615,0},
                                 {1.0000,0.3846,0},{1.0000,0.3077,0},{1.0000,0.2308,0},
                                 {1.0000,0.1538,0},{1.0000,0.0769,0},{1.0000,0,0},
                                 {0.9231,0,0},{0.8462,0,0},{0.7692,0,0},{0.6923,0,0}};

        struct initial_parameters
        {
            std::string camera_topic;
            std::string lidar_topic;
            bool fisheye_model;
            int lidar_ring_count;
            std::pair<int,int> grid_size;
            int square_length; // in millimetres
            std::pair<int,int> board_dimension; // in millimetres
            std::pair<int,int> cb_translation_error; // in millimetres
            cv::Mat cameramat;
            int distcoeff_num;
            cv::Mat distcoeff;
            std::pair<int,int> image_size;
        }i_params;

        struct Rot_Trans
        {
            double roll; // Joint (Rotation and translation) optimization variables
            double pitch;
            double yaw;
            double x;
            double y;
            double z;
            std::string to_string() const
            {
                return std::string("{") +  "roll:"+std::to_string(roll) +", pitch:"+std::to_string(pitch)
                        +", yaw:"+std::to_string(yaw) +", x:"+std::to_string(x) +", y:"+std::to_string(y)
                        +", z:"+std::to_string(z) +"}";
            }
        }rot_trans;

        void sensor_info_callback(const sensor_msgs::Image::ConstPtr &img,
                                             const sensor_msgs::PointCloud2::ConstPtr &pc);

        void organized_pointcloud(pcl::PointCloud<pcl::PointXYZIR>::Ptr input_pointcloud,
                                             pcl::PointCloud<pcl::PointXYZIR>::Ptr organized_pc);

        void init_params();

        double * converto_imgpts(double x, double y, double z);

        void matrix_to_transfrom(Eigen::MatrixXf & matrix, tf::Transform & trans);

        void undistort_img(cv::Mat original_img, cv::Mat undistort_img);

    public:
        projector();

    };
}
#endif //CAM_LIDAR_CALIBRATION_PROJECTOR_H
