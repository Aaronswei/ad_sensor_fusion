#include "camera_lidar_sync/camera_lidar_sync.hpp"



CameraLidarSync::CameraLidarSync(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
    : nh_(nh),
      nh_private_(nh_private),
      it_(nh_private_),
      scale1_(0.5),
      scale2_(5.0),
      threshold_(0.4),
      segradius_(10.0),
      delay_by_n_frames_(10),
      max_image_age_s_(2),
      img_height_(1080),
      img_width_(1920)
{
    //设定初始的相机内参
    /*
    相机信息：
        型号为on 0233，FOV60
        采用lvds转usb传输，会出现信号延时以及帧率不稳定的问题
    相机内参：
        fx: 1998.7356 
        fy: 1991.8909 
        cx: 851.8287 
        cy: 424.0041
    相机的畸变参数为：
        distortion_parameters:
        k1: -5.9837810961927029e-01
        k2: 3.8342172770055183e-01
        p1: 9.6107542719565779e-03
        p2: 1.2766832075472282e-02
        k3: -1.8572618846e-01
    感兴趣的话，可以对图像进行去畸变处理，代码如下：
        // 去畸变并保留最大图
        cv::Size img_size(img_width_, img_height_);
        cv::initUndistortRectifyMap(camera_intrinsics_,distcoeff_,cv::Mat(),
                cv::getOptimalNewCameraMatrix(camera_intrinsics_,distcoeff_,img_size, 1, img_size, 0),  //getOptimalNewCameraMatrix是调节视场大小，为1时视场大小不变，小于1时缩放视场
                img_size, CV_16SC2, undistort_map1, undistort_map2);
    */
    camera_intrinsics_ << 1998.7356, 0.000000000000e+00, 851.8287, 0.000000000000e+00, 0.000000000000e+00, 1991.8909, 424.0041, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00, 0.0, 0.0, 0.0, 1.0;
    // laser_to_cam_<<0.08083690,   	-0.99662126,   	-0.01167763,   	-0.59623674,   
    //                 0.04087759,   	0.01505343,   	-0.99906721,   	-0.38569012,   
    //                 0.99591426,  	0.08030508,   	0.04197840,   	-0.59186737,  
    //                 0.00000000,   	0.00000000,   	0.00000000,   	1.00000000;
    //设定激光到相机的外参
    /*
    Lidar型号：禾赛40p激光雷达
    安装位置：车顶部
    laser_to_cam_（，0:2）旋转
    laser_to_cam_（，3）平移
    */
    laser_to_cam_<<0.08083690,   	-0.99662126,   	-0.01167763,   	-0.19623674,   
                    0.04087759,   	0.01505343,   	-0.99906721,   	-0.26569012,   
                    0.99591426,  	0.08030508,   	0.04197840,   	-0.59186737,  
                    0.00000000,   	0.00000000,   	0.00000000,   	1.00000000;
    //订阅rosbag中topic为raw_image的图像流，长度为20，在imagcallback函数中进行处理
    image_sub_ = it_.subscribe("/raw_image", 20, &CameraLidarSync::imageCallback, this);
    //订阅rosbag中topic为velodyne_points的点云流，长度为10，在pointCloudCallback函数中进行处理
    pointcloud_sub_ = nh_private_.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 10, &CameraLidarSync::pointCloudCallback, this);
    //发布结果，topic为project_cloud_image
    pub_img_ = it_.advertise("/project_cloud_image", 20);
}

void CameraLidarSync::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    ros::Time stamp;
    stamp = msg->header.stamp;
    //读取msg中的图像数据和时间戳，并将信息放在list容器中
    //CvImagePtr自带时间戳参数，所以，不需要单独添加
    /*
    对于某些常用的编码，cv_bridge提供了可选的color或pixel depth的转换：
    mono8: CV_8UC1, grayscale image
    mono16: CV_16UC1, 16-bit grayscale image
    bgr8: CV_8UC3, color image with blue-green-red color order
    rgb8: CV_8UC3, color image with red-green-blue color order
    bgra8: CV_8UC4, BGR color image with an alpha channel
    rgba8: CV_8UC4, RGB color image with an alpha channel
    其中mono8和bgr8是大多数OpenCV函数所期望的图像编码格式。
    */
    cv_bridge::CvImagePtr image = cv_bridge::toCvCopy(msg, "mono8");
    image->header.stamp = stamp;
    images_.push_back(*image);

    // clear old data 
    /*
    删除旧的数据，由于采用list容器，所以容器会不断的追加数据，早期存放的数据就会失效，
    所以，如果图像容器中的最前和最后两张图片的时间差超过两秒钟，就删除容器中的最开始的图片
    以保证list容器中的数据时间差小于2s。
    这种保存多张图像的做法主要有几点：
    1）可以防止由于传输延时导致的时间戳间隔不固定；
    2）可以防止帧率不稳定导致的丢帧；
    3）可以对该容器内的图像进行时间同步；
    */ 
    while (((images_.back()).header.stamp - (images_.front()).header.stamp).toSec() > max_image_age_s_) {
        images_.pop_front();
    }

}


// void CameraLidarSync::pointCloudCallback(const PointCloud::ConstPtr& msg)
void CameraLidarSync::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &laser_scan)
{
    int index = -1;
    double time = laser_scan->header.stamp.toSec();
    /*
    作业1：这里并未对容器内的时间进行同步，可以先对容器内的时间进行同步
    提示：参考上次课camera_imu的容器内时间同步
    */
    //判断容器内的图像是否满足足够的数量，这里选择10帧，主要是为了方便与激光雷达10帧所对应
    if(images_.size() < delay_by_n_frames_)
    {
        return;
    }
    //这里的图像并未做过自校正和同步，会影响后面评价效果，但由于这段数据是车辆静止时采集的，
    //所以，图像的帧率对整体影响不大

    //设定最小相机与激光时间差，这里的0.05表示50毫秒，由于图像为20Hz，即50毫秒，lidar为10Hz
    float time_min = 0.05;

    //找出图像list中与点云帧相对时间差小于time_min的图像，感兴趣的可以试验一下，等于50毫秒的情况
    int count = 0;
    for (auto image_ : images_)
    {
        float time_diff = time - image_.header.stamp.toSec();
        //为啥要加个time_diff < 0呢？
        if(time_diff< 0 && abs(time_diff) < time_min){
            index = count;
            break;
        }
        else if(time_diff< 0){
            break;
        }
        else if(time_diff < time_min){
            index = count;
            time_min = time_diff;
        }
        count += 1;
    }

    //从点云中删除无效点
    /*
    首先定义一个pcl::PointCloud<pcl::PointXYZI>格式的点云帧，
    然后将获取的PointCloud2格式的点云转换为pcl::PointCloud<pcl::PointXYZI>格式的点云
    fromROSMsg就是将PointCloud2格式转换为pcl::PointCloud<T>格式的函数，
    再然后，调用removeNaNFromPointCloud，删除无效点
    removeNaNFromPointCloud函数有三个参数，分别为输入点云，输出点云及对应保留的索引
    这个函数有个bug，输出的size和原来一样，显式中可能会出现不会显式无效点，
    但不一定会真正去除，即indices里面的点和原来一样
    */
    pcl::PointCloud<pcl::PointXYZI>::Ptr laser_cloud(new pcl::PointCloud<pcl::PointXYZI>());
   
    pcl::fromROSMsg(*laser_scan, *laser_cloud);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*laser_cloud, *laser_cloud, indices);

    pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    //这里可以对点云进行插值或者其他的处理
    //由于车辆是静止的，所以只需要对点云进行对齐就足够了，这里采用的是NDT对齐方式
    if (point_cloud_lists_.size() > 2)
    {
        point_cloud_lists_.erase(point_cloud_lists_.begin());
    }
    else if (point_cloud_lists_.size() < 2)
    {
        point_cloud_lists_.push_back(*laser_cloud);
        return;
    }
    else
    {
        //这里开始进行插值处理，这里做了几个简化：1）没有考虑第一帧点云的对齐，2）没有进行图像时间校对，3）第二帧点云不一定在上一个图像时间序列中
        std::cout << "the image index is "<< index << std::endl;
        if(index != -1)
        {
            //取出index所对应的图像，由于这里采用的是std::list容器，取指定索引的指针方法不太一样，需要采用advance
            std::list<cv_bridge::CvImage>::iterator iter = images_.begin();
            advance(iter,index-1);
            std::cout << "the image time is " << iter->header.stamp << std::endl;
            std::cout << "the lidar time is " << time << std::endl;

            //这里定义几个点云变量，方面后面的使用
            pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZI>);
            pcl::PointCloud<pcl::PointXYZI>::Ptr fisrt_cloud (new pcl::PointCloud<pcl::PointXYZI>);
            pcl::PointCloud<pcl::PointXYZI>::Ptr second_cloud (new pcl::PointCloud<pcl::PointXYZI>);
            
            *fisrt_cloud = point_cloud_lists_[0];
            *second_cloud = point_cloud_lists_[1];

            /*
            类ApproximateVoxelGrid根据给定的点云形成三维体素栅格，并利用所有体素的中心，点近似体素中包含的点集，这样完成下采样得到滤波结果。
            该类比较适合对海量点云数据在处理前进行数据压缩，特别是在特征提取等处理中选择合适的体素大小等尺度相关参数，可以很好地提高算法的效率。
            void setLeafSize (float lx, float ly, float lz)设置体素栅格叶大小，lx、ly、lz分别设置体素在XYZ方向上的尺寸

            对input cloud进行过滤是为了缩短配准的计算时间。这里只对input cloud进行了滤波处理，减少其数据量到10%左右，而target cloud不需要滤波处理
            */
            pcl::ApproximateVoxelGrid<pcl::PointXYZI> approximate_voxel_filter;
            approximate_voxel_filter.setLeafSize (0.02, 0.02, 0.02);
            //设定输入点云为存储的第一帧点云，filter_cloud为滤波后的点云
            approximate_voxel_filter.setInputCloud (fisrt_cloud);
            approximate_voxel_filter.filter (*filtered_cloud);
            std::cout << "Filtered cloud contains " << filtered_cloud->size ()
                << " data points from cloud1_halfscan_a topic" << std::endl;

            // 初始化带默认参数的NDT算法对象
            // Initializing Normal Distributions Transform (NDT).
            pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;

            /*
            其中 ndt.setTransformationEpsilon() 即设置变换的 ϵ（两个连续变换之间允许的最大差值），这是判断优化过程是否已经收敛到最终解的阈值。
            ndt.setStepSize(0.01) 即设置牛顿法优化的最大步长。
            ndt.setResolution(1.0) 即设置网格化时立方体的边长，网格大小设置在NDT中非常重要，太大会导致精度不高，太小导致内存过高，并且只有两幅点云相差不大的情况才能匹配。
            ndt.setMaximumIterations(500) 即优化的迭代次数，我们这里设置为500次，即当迭代次数达到500次或者收敛到阈值时，停止优化,
            这个参数控制了优化过程的最大迭代次数。一般来说，在达到最大迭代次数之前程序就会先达到epsilon阈值而终止。
            添加最大迭代次数的限制能够增加程序鲁棒性，阻止了它在错误的方向运行过长时间
            */
            // Setting scale dependent NDT parameters
            // Setting minimum transformation difference for termination condition.
            ndt.setTransformationEpsilon (0.001);
            // Setting maximum step size for More-Thuente line search.
            ndt.setStepSize (0.01);
            //Setting Resolution of NDT grid structure (VoxelGridCovariance).
            ndt.setResolution (1.0);
            // Setting max number of registration iterations.
            ndt.setMaximumIterations (500);

            // Setting point cloud to be aligned.
            ndt.setInputSource (filtered_cloud);
            // Setting point cloud to be aligned to.
            ndt.setInputTarget (second_cloud);

            //这里主要计算点云配准变换的估计。尤其是当两块点云差异较大时，得到更好的结果，
            // Set initial alignment estimate found using robot odometry.
            Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity ();

            //最后进行点云配准，变换后的点云保存在output_cloud里，之后打印出配准分数。分数通过计算output_cloud与target cloud对应的最近点欧式距离的平方和得到，得分越小说明匹配效果越好。
            // Calculating required rigid transform to align the input cloud to the target cloud.
            ndt.align (*output_cloud, init_guess);
            std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged ()
                    << " score: " << ndt.getFitnessScore () << std::endl;
            std::cout << ndt.getFinalTransformation() << std::endl;

            // ////////////////////////align two point clouds//////////////////////////
            //执行变换，并将结果保存在新创建的‎‎ output_cloud ‎‎中,这个一般可以只要运行一次，其他直接转换即可，感兴趣自行试验
            // // Transforming unfiltered, input cloud using found transform.
            // pcl::transformPointCloud (second_cloud, *output_cloud, ndt.getFinalTransformation ());

        }
        
        std::cout << std::endl;
    }

    //************

    //对齐后进行校验，通过标定进行校验，这里本来想采用cv::solvePnP来进行验证的，这里没有做点云的特征检测和图像的特征检测，所以暂时放弃solvePnP方法，采用直接外参的形式进行比较
    //关于solvePnP的方法，且点云特征是下节课的重点，所以会在下次课结束后，会将代码更新进去
    if(index != -1)
    {
        std::list<cv_bridge::CvImage>::iterator iter = images_.begin();
        advance(iter,index-1);
        cv::Mat out_img = iter->image.clone();

        float max_val = 60.0;
        
        // for (int i = 0; i < laser_cloud->points.size(); ++i)
        // {
        //     if(laser_cloud->points[i].x <= 0){
        //         continue;
        //     }
        //     float point_in_cam_x = laser_cloud->points[i].x * laser_to_cam_(0, 0) + laser_cloud->points[i].y * laser_to_cam_(0, 1) + laser_cloud->points[i].z * laser_to_cam_(0, 2) + laser_to_cam_(0, 3);
        //     float point_in_cam_y = laser_cloud->points[i].x * laser_to_cam_(1, 0) + laser_cloud->points[i].y * laser_to_cam_(1, 1) + laser_cloud->points[i].z * laser_to_cam_(1, 2) + laser_to_cam_(1, 3);
        //     float point_in_cam_z = laser_cloud->points[i].x * laser_to_cam_(2, 0) + laser_cloud->points[i].y * laser_to_cam_(2, 1) + laser_cloud->points[i].z * laser_to_cam_(2, 2) + laser_to_cam_(2, 3);
        //     int x = (int)((camera_intrinsics_(0, 0) * point_in_cam_x + camera_intrinsics_(0, 1) * point_in_cam_y + camera_intrinsics_(0, 2) * point_in_cam_z) / point_in_cam_z);
        //     int y = (int)((camera_intrinsics_(1, 0) * point_in_cam_x + camera_intrinsics_(1, 1) * point_in_cam_y + camera_intrinsics_(1, 2) * point_in_cam_z) / point_in_cam_z);
        //     if(x > 0 && x < img_width_ && y > 0 && y < img_height_){
        //         // float dist = (float)(std::sqrt(point_in_cam_x * point_in_cam_x + point_in_cam_y * point_in_cam_y + point_in_cam_z * point_in_cam_z));
        //         int red = std::min(255, (int)(255 * abs((point_in_cam_z - max_val) / max_val)));
        //         int green = std::min(255, (int)(255 * (1 - abs((point_in_cam_z - max_val) / max_val))));
        //         cv::circle(out_img, cv::Point(x,y), 5, cv::Scalar(0, green, red), -1);
        //     }
        // }

        //将点云按照固定的外参投影到图像上
        for (int i = 0; i < output_cloud->points.size(); ++i)
        {
            if(output_cloud->points[i].x <= 0){
                continue;
            }
            float point_in_cam_x = output_cloud->points[i].x * laser_to_cam_(0, 0) + output_cloud->points[i].y * laser_to_cam_(0, 1) + output_cloud->points[i].z * laser_to_cam_(0, 2) + laser_to_cam_(0, 3);
            float point_in_cam_y = output_cloud->points[i].x * laser_to_cam_(1, 0) + output_cloud->points[i].y * laser_to_cam_(1, 1) + output_cloud->points[i].z * laser_to_cam_(1, 2) + laser_to_cam_(1, 3);
            float point_in_cam_z = output_cloud->points[i].x * laser_to_cam_(2, 0) + output_cloud->points[i].y * laser_to_cam_(2, 1) + output_cloud->points[i].z * laser_to_cam_(2, 2) + laser_to_cam_(2, 3);
            int x = (int)((camera_intrinsics_(0, 0) * point_in_cam_x + camera_intrinsics_(0, 1) * point_in_cam_y + camera_intrinsics_(0, 2) * point_in_cam_z) / point_in_cam_z);
            int y = (int)((camera_intrinsics_(1, 0) * point_in_cam_x + camera_intrinsics_(1, 1) * point_in_cam_y + camera_intrinsics_(1, 2) * point_in_cam_z) / point_in_cam_z);
            if(x > 0 && x < img_width_ && y > 0 && y < img_height_){
                // float dist = (float)(std::sqrt(point_in_cam_x * point_in_cam_x + point_in_cam_y * point_in_cam_y + point_in_cam_z * point_in_cam_z));
                int red = std::min(255, (int)(255 * abs((point_in_cam_z - max_val) / max_val)));
                int green = std::min(255, (int)(255 * (1 - abs((point_in_cam_z - max_val) / max_val))));
                cv::circle(out_img, cv::Point(x,y), 5, cv::Scalar(0, green, red), -1);
            }
        }
        
        std_msgs::Header header;
        header.stamp = laser_scan->header.stamp;
        header.frame_id = "/camera_init";
        pub_img_.publish(cv_bridge::CvImage(header, "bgr8", out_img).toImageMsg());
        cv::imwrite("result/raw_cloud/" + std::to_string(time)+".png",out_img); 
    }

}


