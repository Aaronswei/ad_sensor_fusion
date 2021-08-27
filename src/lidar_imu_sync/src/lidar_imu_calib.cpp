#include "lidar_imu_sync/lidar_imu_calib.hpp"
#include <omp.h>
#include <utility>
#include <tf/tf.h>
#include <pcl/io/pcd_io.h>

#define PRINT_LOG //std::cout << __FILE__ << ", " << __LINE__ << std::endl;

LidarIMUCalib::LidarIMUCalib()
{
    //析构函数
    //这里主要设定NDT算法的相关参数，这里采用的是ndt_omp库，是一个加了多线程的数据库
    imu_buffer_.clear();

    // init downsample object
    //设置体素栅格叶大小，向量参数leaf_ size 是体素栅格叶大小参数，每个元素分别表示体素在XYZ方向上的尺寸
    //属于VoxelGrid函数中
    downer_.setLeafSize(0.1, 0.1, 0.1);

    // init register object
    //NormalDistributionsTransform属于PCL的正态分布变换
    //初始化正态分布变量
    register_.reset(new pclomp::NormalDistributionsTransform<PointT, PointT>());
    //设置ndt网络的分辨率，该尺度与场景有关，默认为1.0
    register_->setResolution(1.0);
    //获取线程数并设定线程数（这属于omp中的，比较简单）
    int avalib_cpus = omp_get_max_threads();
    register_->setNumThreads(avalib_cpus);
    //设定邻域搜索方法，DIRECT7这个最快最好（其他的参数都是有特殊场景的用途）
    register_->setNeighborhoodSearchMethod(pclomp::DIRECT7);
}

LidarIMUCalib::~LidarIMUCalib()
{
}

void LidarIMUCalib::addLidarData(const LidarData &data)
{
    //将点云进行处理后，送入到标定列表中
    if (!data.cloud || data.cloud->size() == 0)
    {
        cout << "no cloud in lidar data !!!" << endl;
        return;
    }

    if (!register_)
    {
        cout << "register no initialize !!!" << endl;
        return;
    }

    //downsample lidar cloud for save align time
    //对点云进行降采样匹配
    CloudT::Ptr downed_cloud(new CloudT);
    //设定输入的点云
    downer_.setInputCloud(data.cloud);
    //对点云进行滤波，得到降采样之后的点云帧
    downer_.filter(*downed_cloud);

    //构建局部地图，如果点云地图不存在，那么就新建点云地图
    if (!local_map_)
    {
        local_map_.reset(new CloudT);
        //将第一帧点云作为地图的原始信息
        *local_map_ += *(data.cloud);

        LidarFrame frame;
        frame.stamp = data.stamp;
        //对地图点云帧进行初始化操作
        frame.T = Eigen::Matrix4d::Identity();
        frame.gT = Eigen::Matrix4d::Identity();
        frame.cloud = downed_cloud;
        //将点云数据追加到点云缓存区
        lidar_buffer_.push_back(move(frame));

        return;
    }

    // downsample local map for save align time
    //降采样建图，节省建图时间
    CloudT::Ptr downed_map(new CloudT);
    downer_.setInputCloud(local_map_);
    downer_.filter(*downed_map);
    local_map_ = downed_map;

    // get transform between frame and local map
    //获取输入的降采样点云与地图之间的旋转矩阵
    register_->setInputSource(downed_cloud);
    register_->setInputTarget(local_map_);
    CloudT::Ptr aligned(new CloudT);
    register_->align(*aligned);
    if (!register_->hasConverged())
    {
        cout << "register cant converge, please check initial value !!!" << endl;
        return;
    }
    //获取最终的旋转矩阵信息
    Eigen::Matrix4d T_l_m = (register_->getFinalTransformation()).cast<double>();

    //生成点云帧，更新点云地图缓存区
    // generate lidar frame
    LidarFrame frame;
    frame.stamp = data.stamp;
    frame.gT = T_l_m;
    Eigen::Matrix4d last_T_l_m = lidar_buffer_.back().gT;
    frame.T = last_T_l_m.inverse() * T_l_m;
    frame.cloud = downed_cloud;
    lidar_buffer_.push_back(move(frame));

    // update local map
    //更新点云地图
    *local_map_ += *aligned;

}

void LidarIMUCalib::addImuData(const ImuData &data)
{
    imu_buffer_.push_back(data);
}

Eigen::Vector3d LidarIMUCalib::calib(bool integration)
{
    //判断标定程序中，有无lidar数据和imu数据
    if (lidar_buffer_.size() == 0 || imu_buffer_.size() == 0)
    {
        cout << "no lidar data or imu data !!!" << endl;
        return init_R_;
    }
    cout << "total lidar buffer size " << lidar_buffer_.size() << ", imu buffer size " << imu_buffer_.size() << endl;
    // integration rotation of imu, when raw imu attitude has big error
    //imu预积分操作，主要针对原始imu姿态误差较大时
    if (integration)
    {
        imu_buffer_[0].rot = Eigen::Quaterniond::Identity();
        for (int i = 1; i < imu_buffer_.size(); i++)
        {
            Eigen::Vector3d bar_gyr = 0.5 * (imu_buffer_[i - 1].gyr + imu_buffer_[i].gyr);
            Eigen::Vector3d angle_inc = bar_gyr * (imu_buffer_[i].stamp - imu_buffer_[i - 1].stamp);
            Eigen::Quaterniond rot_inc = Eigen::Quaterniond(1.0, 0.5 * angle_inc[0], 0.5 * angle_inc[1], 0.5 * angle_inc[2]);
            imu_buffer_[i].rot = imu_buffer_[i - 1].rot * rot_inc;
        }
    }

    //对点云数据进行预处理，清除掉无效的点云数据，即第一帧imu帧到来之前的点云数据都认为时无效值
    //主要比较imu帧与点云帧的时间戳，凡是在第一帧imu到来前，所有的点云帧都认为是无效的
    auto invalid_lidar_it = lidar_buffer_.begin();
    for (; invalid_lidar_it != lidar_buffer_.end(); invalid_lidar_it++)
    {
        if (invalid_lidar_it->stamp >= imu_buffer_[0].stamp)
            break;
    }
    //清除无效帧，如果没有有效的点云帧，则直接返回0值
    if (invalid_lidar_it != lidar_buffer_.begin())
        lidar_buffer_.erase(lidar_buffer_.begin(), invalid_lidar_it);
    if (lidar_buffer_.size() == 0)
    {
        cout << "no valid lidar frame !!!" << endl;
        return move(Eigen::Vector3d(0.0, 0.0, 0.0));
    }

    // 获取时间对齐后的lidar-odo与IMU预积分后的rotation
    auto last_imu_it = imu_buffer_.begin();
    for (int i = 0; i < lidar_buffer_.size(); i++)
    {
        // 获取lidar信息
        const auto &lidar_frame = lidar_buffer_[i];

        // 获取当前lidar_frame信息最近的imu数据
        for (; last_imu_it != imu_buffer_.end(); last_imu_it++)
        {
            if (last_imu_it->stamp >= lidar_frame.stamp)
                break;
        }
        if (last_imu_it != imu_buffer_.begin())
            last_imu_it--;

        // 插值得到lidar时间戳处的imu，因为内插法，所以需要剔除imu队列中的最后一帧
        auto imu_it1 = last_imu_it;
        auto imu_it2 = last_imu_it + 1;
        if (imu_buffer_.end() == imu_it2)
            break;
        assert(imu_it2->stamp >= lidar_frame.stamp || imu_it1->stamp < imu_it2->stamp); 
        //获取两个时刻imu的rotation
        Eigen::Quaterniond q_b1_w = imu_it1->rot;
        Eigen::Quaterniond q_b2_w = imu_it2->rot;
        //获取时间比例，插值比例
        double scale = (lidar_frame.stamp - imu_it1->stamp) / (imu_it2->stamp - imu_it1->stamp);
        //获取按照时间比例插值后的角度
        Eigen::Quaterniond q_inter_w = getInterpolatedAttitude(q_b1_w, q_b2_w, scale);

        //将插值后的imu姿态和点云帧同时送入到对齐队列中
        aligned_lidar_imu_buffer_.push_back(move(pair<LidarFrame, Eigen::Quaterniond>(lidar_frame, q_inter_w)));
    }

    //计算点云帧与IMU帧的transform
    vector<pair<Eigen::Quaterniond, Eigen::Quaterniond>> corres(0);
    std::cout << aligned_lidar_imu_buffer_.size() << std::endl;
    for (int i = 1; i < aligned_lidar_imu_buffer_.size(); i++)
    {
        // 计算两个连续的对齐的pair（点云，imu）
        const auto &aligned1 = aligned_lidar_imu_buffer_[i - 1];
        const auto &aligned2 = aligned_lidar_imu_buffer_[i];

        //计算两帧连续点云的相对transform
        Eigen::Quaterniond q_l2_l1 = Eigen::Quaterniond(aligned_lidar_imu_buffer_[i].first.T.block<3, 3>(0, 0));

        //计算两帧插值后的imu的相对transform
        Eigen::Quaterniond q_b1_w = aligned1.second;
        Eigen::Quaterniond q_b2_w = aligned2.second;
        Eigen::Quaterniond q_b2_b1 = q_b1_w.inverse() * q_b2_w;

        //将计算出的两组transform存放在队列中
        corres.push_back(move(pair<Eigen::Quaterniond, Eigen::Quaterniond>(q_l2_l1, q_b2_b1)));
        corres1_ = corres;
    }
    //求解出相对的transform
    q_l_b_ = solve(corres);

    //利用得到的结果进行优化
    optimize();

    // 获取最后的结果
    tf::Matrix3x3 mat(tf::Quaternion(q_l_b_.x(), q_l_b_.y(), q_l_b_.z(), q_l_b_.w()));
    double roll, pitch, yaw;
    mat.getRPY(roll, pitch, yaw);

    return move(Eigen::Vector3d(roll, pitch, yaw));
}

//获取插值后的角度
Eigen::Quaterniond LidarIMUCalib::getInterpolatedAttitude(const Eigen::Quaterniond &q_s_w, const Eigen::Quaterniond &q_e_w, double scale)
{
    if (0 == scale || scale > 1)
        return move(Eigen::Quaterniond().Identity());

    //计算各个轴的角度差
    Eigen::Quaterniond q_e_s = q_s_w.inverse() * q_e_w;
    q_e_s.normalize();
    Eigen::AngleAxisd diff_angle_axis(q_e_s);

    // 按照时间比例进行状态插值
    double interpolated_angle = diff_angle_axis.angle() * scale;
    Eigen::Quaterniond q_ie_s(Eigen::AngleAxisd(interpolated_angle, diff_angle_axis.axis()).toRotationMatrix());
    Eigen::Quaterniond q_ie_w = q_s_w * q_ie_s;
    q_ie_w.normalize();

    return move(q_ie_w);
}

Eigen::Quaterniond LidarIMUCalib::solve(const vector<pair<Eigen::Quaterniond, Eigen::Quaterniond>> &corres)
{
    if (corres.size() == 0)
    {
        cout << "no constraint found !!!" << endl;
        return move(Eigen::Quaterniond().Identity());
    }

    cout << "constraints size " << corres.size() << endl;

    // 四元数到斜对称矩阵的变换
    auto toSkewSymmetric = [](const Eigen::Vector3d &q) -> Eigen::Matrix3d {
        Eigen::Matrix3d mat = Eigen::Matrix3d::Zero();
        mat(0, 1) = -q.z();
        mat(0, 2) = q.y();
        mat(1, 0) = q.z();
        mat(1, 2) = -q.x();
        mat(2, 0) = -q.y();
        mat(2, 1) = q.x();

        return move(mat);
    };

    // 齐次方程求解
    Eigen::MatrixXd A(corres.size() * 4, 4);
    for (int i = 0; i < corres.size(); i++)
    {
        // 相对的lidar和imu的transform
        const auto &q_l2_l1 = corres[i].first;
        const auto &q_b2_b1 = corres[i].second;

        // 左乘矩阵
        Eigen::Vector3d q_b2_b1_vec = q_b2_b1.vec();
        Eigen::Matrix4d left_Q_b2_b1 = Eigen::Matrix4d::Zero();
        left_Q_b2_b1.block<1, 3>(0, 1) = -q_b2_b1_vec.transpose();
        left_Q_b2_b1.block<3, 1>(1, 0) = q_b2_b1_vec;
        left_Q_b2_b1.block<3, 3>(1, 1) = toSkewSymmetric(q_b2_b1_vec);
        left_Q_b2_b1 += q_b2_b1.w() * Eigen::Matrix4d::Identity();

        // 右乘矩阵
        Eigen::Vector3d q_l2_l1_vec = q_l2_l1.vec();
        Eigen::Matrix4d right_Q_l2_l1 = Eigen::Matrix4d::Zero();
        right_Q_l2_l1.block<1, 3>(0, 1) = -q_l2_l1_vec.transpose();
        right_Q_l2_l1.block<3, 1>(1, 0) = q_l2_l1_vec;
        right_Q_l2_l1.block<3, 3>(1, 1) = -toSkewSymmetric(q_l2_l1_vec);
        right_Q_l2_l1 += q_l2_l1.w() * Eigen::Matrix4d::Identity();

        // 计算两者的距离
        double angle_distance = 180.0 / M_PI * q_b2_b1.angularDistance(q_l2_l1);
        double huber = angle_distance > 2.0 ? 2.0 / angle_distance : 1.0;

        A.block<4, 4>(i * 4, 0) = huber * (left_Q_b2_b1 - right_Q_l2_l1);
    }

    // SVD求解齐次方程
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Eigen::Quaterniond q_l_b(x(0), x(1), x(2), x(3));

    return move(q_l_b);
}

void LidarIMUCalib::optimize()
{
    if (aligned_lidar_imu_buffer_.size() == 0 || !register_)
    {
        cout << "no aligned data or register !!!" << endl;
        return;
    }

    // 地图的初始化
    if (local_map_)
        local_map_->clear();
    else
        local_map_.reset(new CloudT);
    *local_map_ += *(aligned_lidar_imu_buffer_[0].first.cloud);

    //estimated initial value to update lidar frame
    for (int i = 1; i < aligned_lidar_imu_buffer_.size(); i++)
    {
        // 获取前后两帧对齐的数据
        auto &aligned1 = aligned_lidar_imu_buffer_[i - 1];
        auto &aligned2 = aligned_lidar_imu_buffer_[i];

        // 下采样点云和地图
        CloudT::Ptr downed_map(new CloudT);
        downer_.setInputCloud(local_map_);
        downer_.filter(*downed_map);
        local_map_ = downed_map;

        // calculate estimated T_l_m
        Eigen::Matrix3d R_l1_m = aligned1.first.gT.block<3, 3>(0, 0);
        Eigen::Quaterniond q_b1_w = aligned1.second;
        Eigen::Quaterniond q_b2_w = aligned2.second;
        Eigen::Quaterniond est_q_b2_b1 = q_b1_w.inverse() * q_b2_w;
        Eigen::Matrix3d est_R_l2_l1 = Eigen::Matrix3d(q_l_b_.inverse() * est_q_b2_b1 * q_l_b_);
        Eigen::Matrix3d est_R_l2_m = R_l1_m * est_R_l2_l1;
        Eigen::Matrix4d est_T_l2_m = Eigen::Matrix4d::Identity();
        est_T_l2_m.block<3, 3>(0, 0) = est_R_l2_m;

        // 优化点云与地图之间的关系，使得register收敛
        register_->setInputSource(aligned2.first.cloud);
        register_->setInputTarget(local_map_);
        CloudT::Ptr aligned(new CloudT);
        register_->align(*aligned, est_T_l2_m.cast<float>());
        if (!register_->hasConverged())
        {
            cout << "register cant converge, please check initial value !!!" << endl;
            return;
        }
        //获取最终的transformation
        Eigen::Matrix4d T_l2_m = (register_->getFinalTransformation()).cast<double>();

        // 依据得到的transformation来更新点云和地图
        aligned2.first.gT = T_l2_m;
        Eigen::Matrix4d T_l1_m = aligned1.first.gT;
        aligned2.first.T = T_l1_m.inverse() * T_l2_m;

        // update local map
        *local_map_ += *aligned;
    }

    // 建立收敛约束
    vector<pair<Eigen::Quaterniond, Eigen::Quaterniond>> corres;
    for (int i = 1; i < aligned_lidar_imu_buffer_.size(); i++)
    {
        const auto &aligned1 = aligned_lidar_imu_buffer_[i - 1];
        const auto &aligned2 = aligned_lidar_imu_buffer_[i];

        // calculate relative transform between neighbor lidar
        Eigen::Quaterniond q_l2_l1 = Eigen::Quaterniond(aligned2.first.T.block<3, 3>(0, 0));

        // calculate relative transform between neighbor interpolated imu
        Eigen::Quaterniond q_b1_w = aligned1.second;
        Eigen::Quaterniond q_b2_w = aligned2.second;
        Eigen::Quaterniond q_b2_b1 = q_b1_w.inverse() * q_b2_w;

        corres.push_back(move(pair<Eigen::Quaterniond, Eigen::Quaterniond>(q_l2_l1, q_b2_b1)));
        corres2_ = corres;
    }

    Eigen::Quaterniond result = solve(corres);
    result.normalize();

    // 判断是否收敛
    double angle = fabs(q_l_b_.angularDistance(result));
    if (angle > 0.5236)
    {
       cout << "the difference between before and after optimze is " << angle << " which greater than given threshold 0.5236 !!!" << endl;
       return;
    }
    q_l_b_ = result;
}

