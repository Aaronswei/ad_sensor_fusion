#include "utility.h"
#include "lidar_imu_fusion/cloud_info.h"
struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

struct PandarPointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (PandarPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (double, time, time)
)


struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Subscriber subResetCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubExtractedCloudFront;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloudFront;
    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lidar_imu_fusion::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    double resetCloudTime = -1;
    std_msgs::Header cloudHeader;


public:
    ImageProjection():
    deskewFlag(0)
    {
        //订阅话题进入回调函数 imu数据、激光点云.
        //订阅imu里程计: 来自IMUPreintegration(IMUPreintegration.cpp中的类IMUPreintegration)发布的里程计话题（增量式） 
        subResetCloud = nh.subscribe<std_msgs::Float64>("/reset_cloud_time", 5, &ImageProjection::resetCloudHandler, this, ros::TransportHints().tcpNoDelay());
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        //发布去畸变的点云,1:queue_size
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> (PROJECT_NAME + "/lidar/deskew/cloud_deskewed", 1);
        pubExtractedCloudFront = nh.advertise<sensor_msgs::PointCloud2> (PROJECT_NAME + "/lidar/deskew/cloud_deskewed_front", 1);
        //发布激光点云信息 cloud_info的msg文件
        pubLaserCloudInfo = nh.advertise<lidar_imu_fusion::cloud_info> (PROJECT_NAME + "/lidar/deskew/cloud_info", 1);
        //分配内存
        allocateMemory();
        //重置部分参数
        resetParameters();
        //setVerbosityLevel用于设置控制台输出的信息。(pcl::console::L_ALWAYS)不会输出任何信息;L_DEBUG会输出DEBUG信息;L_ERROR会输出ERROR信息
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        //根据params.yaml中给出的N_SCAN Horizon_SCAN参数值分配内存
        //用智能指针的reset方法在构造函数里面进行初始化
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloudFront.reset(new pcl::PointCloud<PointType>());
        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        //cloudinfo是msg文件下自定义的cloud_info消息，对其中的变量进行赋值操作
        //(int size, int value):size-要分配的值数,value-要分配给向量名称的值
        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);
        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        extractedCloudFront->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){}

    /**
     * 订阅原始imu数据
     * 1、imu原始测量数据转换到lidar系，加速度、角速度
    */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        //imuConverter在头文件utility.h中，作用是把imu数据转换到lidar坐标系
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);
        // 上锁，添加数据的时候队列不可用
        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    /**
     * 订阅imu里程计，由imuPreintegration积分计算得到的每时刻imu位姿.(地图优化程序中发布的)
    */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void resetCloudHandler(const std_msgs::Float64ConstPtr& msg){
        resetCloudTime = msg->data;
        std::cout<<"get time reset "<<std::setprecision(19) << resetCloudTime << std::endl;
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        
        if(laserCloudMsg->header.stamp.toSec() < resetCloudTime){
            return;
        }
        //添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
        if (!cachePointCloud(laserCloudMsg))
            return;
        // 当前帧起止时刻对应的imu数据、imu里程计数据处理
        if (!deskewInfo())
            return;
        // 当前帧激光点云运动畸变校正
        // 1、检查激光点距离、扫描线是否合规
        // 2、激光运动畸变校正，保存激光点
        projectPointCloud();
        // 提取有效激光点，存extractedCloud
        cloudExtraction();

        publishClouds();

        resetParameters();
    }

    //点云缓存区
    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud
        currentCloudMsg = std::move(cloudQueue.front());
        // 取出激光点云队列中最早的一帧
        cloudQueue.pop_front();
        cloudHeader = currentCloudMsg.header;
        if (sensor == SensorType::VELODYNE)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
            for (size_t i = 0; i < laserCloudIn->points.size(); ++i)
            {
                auto &dst = laserCloudIn->points[i];
                Eigen::Vector3d pos(dst.x,dst.y,dst.z);
                pos = extRPYInv * pos + extTransLidar2Base;
                dst.x = pos[0];
                dst.y = pos[1];
                dst.z = pos[2];
            }
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else if (sensor == SensorType::Pandar40P){
            //这里是我们自己定义的禾赛40P的数据
            pcl::PointCloud<PandarPointXYZIRT>::Ptr tmp_cloud(new pcl::PointCloud<PandarPointXYZIRT>());
            // 转换成pcl点云格式 形参: (in,out)
            pcl::moveFromROSMsg(currentCloudMsg, *tmp_cloud);
            laserCloudIn->points.resize(tmp_cloud->size());
            laserCloudIn->is_dense = tmp_cloud->is_dense;
            // float start_ori = -atan2(tmp_cloud->points[0].y, tmp_cloud->points[0].x);
            // float end_ori = -atan2(tmp_cloud->points[tmp_cloud->size() - 1].y,tmp_cloud->points[tmp_cloud->size() - 1].x) + 2 * M_PI;
            // if (end_ori - start_ori > 3 * M_PI) 
            // {
            //     end_ori -= 2 * M_PI;
            // } 
            // else if (end_ori - start_ori < M_PI) 
            // {
            //     end_ori += 2 * M_PI;
            // }
            // float ori_diff = end_ori - start_ori;
            // bool half_passed = false;
            for (size_t i = 0; i < tmp_cloud->size(); i++)
            {
                
                auto &src = tmp_cloud->points[i];
                auto &dst = laserCloudIn->points[i];
                Eigen::Vector3d pos(src.x,src.y,src.z);
                pos = extRPYInv * pos + extTransLidar2Base;
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;

                dst.intensity = src.intensity;
                dst.ring = 39 - i % 40;
                
                // float vertical_angle = atan2(dst.z, sqrt(dst.x * dst.x + dst.y * dst.y)) * 180 / M_PI;
                // dst.ring = calculateRowIndex(vertical_angle + 25.0 + 0.1);
                // dst.time = (tmp_cloud->points[i].time - tmp_cloud->points[0].time);
                // float ori = -atan2(dst.y, dst.x);
                // if (!half_passed) {
      	        //     if (ori < start_ori - M_PI / 2) {
                //         ori += 2 * M_PI;
       	        //     }
                //     else if (ori > start_ori + M_PI * 3 / 2) {
                //         ori -= 2 * M_PI;
       	        //     }
 	 	        //     if (ori - start_ori > M_PI) {
                //         half_passed = true;
      	        //     }
                // } 
                // else 
                // {
                //     ori += 2 * M_PI;
		        //     if (ori < end_ori - M_PI * 3 / 2) {
                //     ori += 2 * M_PI;
                //     } 
                //     else if (ori > end_ori + M_PI / 2) {
                //         ori -= 2 * M_PI;
                //     }
                // }
	            // dst.time = (ori - start_ori) / ori_diff * 0.1;
                dst.time = (tmp_cloud->points[i].time - tmp_cloud->points[0].time);
                dst.x = pos[0];
                dst.y = pos[1];
                dst.z = pos[2];
                PointType p;
                p.x = dst.x;
                p.y = dst.y;
                p.z = dst.z;
                extractedCloudFront->push_back(p);
            }
            cloudHeader = currentCloudMsg.header;
            cloudHeader.stamp = ros::Time().fromSec(tmp_cloud->points[0].time);
        }        
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }
        // get timestamp
        //这一点的时间被记录下来，存入timeScanCur中，函数deskewPoint中会被加上laserCloudIn->points[i].time
        timeScanCur = cloudHeader.stamp.toSec();
        //可以看出lasercloudin中存储的time是一帧中距离起始点的相对时间,timeScanEnd是该帧点云的结尾时间
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;
        
        // check dense flag
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        //由于static关键字，只检查一次，检查ring这个field是否存在. veloodyne和ouster都有;
        //ring代表线数，0是最下面那条
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        // 检查是否存在time通道
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    size_t calculateRowIndex(const float & angle){
        if(angle <= 6.0){
            return size_t(angle/6.0);
        }
        else if(angle <= 11.0){ 
            return size_t(1 + (angle - 6.0)/5.0);
        }
        else if(angle <= 19.0){
            return size_t(2 + (angle - 11.0)/1.0);
        }
        else if(angle <= 27.0){
            return size_t(10 + (angle - 19.0)/0.33);
        }
        else if(angle <= 28.0){
            return size_t(34 + (angle -27.0)/1.0);
        }
        else if(angle <= 30.0){
            return size_t(35 + (angle - 28.0) / 2.0);
        }
        else if(angle <= 36.0){
            return size_t(36 + (angle - 30.0) / 3.0);
        }
        else{
            return size_t(38 + (angle - 36.0) / 4.0);
        }

    }
    

    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        // 要求imu数据时间上包含激光数据，否则不往下处理了
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        //imu去畸变参数计算（IMU数据转换到Lidar系下）
        // 当前帧对应imu数据处理
        // 1、遍历当前激光帧起止时刻之间的imu数据，初始时刻对应imu的姿态角RPY设为当前帧的初始姿态角
        // 2、用角速度、时间积分，计算每一时刻相对于初始时刻的旋转量，初始时刻旋转设为0
        imuDeskewInfo();

        //里程计去畸变参数计算（IMU数据转换到Lidar系下）
        // 当前帧对应imu里程计处理
        // 1、遍历当前激光帧起止时刻之间的imu里程计数据，初始时刻对应imu里程计设为当前帧的初始位姿
        // 2、用起始、终止时刻对应imu里程计，计算相对位姿变换，保存平移增量
        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        // 从imu队列中删除当前激光帧0.01s前面时刻的imu数据
        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        // 遍历当前激光帧起止时刻（前后扩展0.01s）之间的imu数据
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan
            // 提取imu姿态角RPY，作为当前lidar帧初始姿态角
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            // 超过当前激光帧结束时刻0.01s，结束
            if (currentImuTime > timeScanEnd + 0.01)
                break;

            // 第一帧imu旋转角初始化
            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            // 提取imu角速度
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;
        // 没有合规的imu数据
        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    //初始pose信息保存在cloudInfo里
    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;
        // 从imu里程计队列中删除当前激光帧0.01s前面时刻的imu数据
        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.06)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        float time = 0.06;
        float time_threshold = 0.06;
        float last_time = -1;
        int index = -1;
        for(int i = 0; i < odomQueue.size(); ++i){
            float time_diff = abs(timeScanCur - ROS_TIME(&odomQueue[i]));
            if(time_diff < time){
                time = time_diff;
                index = i;
                break;
            }
            else if(time < time_threshold || time_diff > last_time){
                break;
            }
            last_time = time_diff;
        }

        if(-1 == index){
            return;
        }

        // if (odomQueue.front().header.stamp.toSec() > timeScanCur)
        //     return;

        // get start odometry at the beinning of the scan
        //(地图优化程序中发布的)
        nav_msgs::Odometry startOdomMsg = odomQueue[index];
        odomQueue.erase(odomQueue.begin(), odomQueue.begin() + index);

        // for (int i = 0; i < (int)odomQueue.size(); ++i)
        // {
        //     startOdomMsg = odomQueue[i];

        //     if (ROS_TIME(&startOdomMsg) < timeScanCur)
        //         continue;
        //     else
        //         break;
        // }

        // 提取imu里程计姿态角
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        // 用当前激光帧起始时刻的imu里程计，初始化lidar位姿，后面用于mapOptmization
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;
        // 如果当前激光帧结束时刻之后没有imu里程计数据，返回
        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        time = 0.06;
        time_threshold = 0.06;
        last_time = -1;
        index = -1;
        for(int i = 0; i < odomQueue.size(); ++i){
            float time_diff = abs(timeScanCur - ROS_TIME(&odomQueue[i]));
            if(time_diff < time){
                time = time_diff;
                index = i;
                break;
            }
            else if(time < time_threshold || time_diff > last_time){
                break;
            }
            last_time = time_diff;
        }

        if(-1 == index){
            return;
        }

        nav_msgs::Odometry endOdomMsg = odomQueue[index];

        // for (int i = 0; i < (int)odomQueue.size(); ++i)
        // {
        //     endOdomMsg = odomQueue[i];

        //     if (ROS_TIME(&endOdomMsg) < timeScanEnd)
        //         continue;
        //     else
        //         break;
        // }

        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);
        // 起止时刻imu里程计的相对变换
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        // 给定的转换中，提取XYZ以及欧拉角,通过tranBt 获得增量值后续去畸变用到
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    /**
     * 在当前激光帧起止时间范围内，计算某一时刻的旋转（相对于起始时刻的旋转增量）
    */
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;
        // 查找当前时刻在imuTime下的索引
        //imuDeskewInfo中，对imuPointerCur进行计数(计数到超过当前激光帧结束时刻0.01s)
        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            //imuTime在imuDeskewInfo（deskewInfo中调用，deskewInfo在cloudHandler中调用）被赋值，从imuQueue中取值
            //pointTime为当前时刻，由此函数的函数形参传入,要找到imu积分列表里第一个大于当前时间的索引
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }
        // 设为离当前时刻最近的旋转增量
        //如果计数为0或该次imu时间戳小于了当前时间戳(异常退出)
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            //未找到大于当前时刻的imu积分索引
            //imuRotX等为之前积分出的内容.(imuDeskewInfo中)
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            // 前后时刻插值计算当前时刻的旋转增量
            //此时front的时间是大于当前pointTime时间，back=front-1刚好小于当前pointTime时间，前后时刻插值计算
            int imuPointerBack = imuPointerFront - 1;
            //算一下该点时间戳在本组imu中的位置
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            //这三项作为函数返回值，以形参指针的方式返回
            //按前后百分比赋予旋转量
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.
        // 如果传感器移动速度较慢，例如人行走的速度，那么可以认为激光在一帧时间范围内，平移量小到可以忽略不计
        if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
            return;

        float ratio = relTime / (timeScanEnd - timeScanCur);

        *posXCur = ratio * odomIncreX;
        *posYCur = ratio * odomIncreY;
        *posZCur = ratio * odomIncreZ;
    }

    /**
     * 激光运动畸变校正
     * 利用当前帧起止时刻之间的imu数据计算旋转增量，imu里程计数据计算平移增量，进而将每一时刻激光点位置变换到第一个激光点坐标系下，进行运动补偿
    */
    PointType deskewPoint(PointType *point, double relTime)
    {
        //这个来源于上文的时间戳通道和imu可用判断，没有或是不可用则返回点
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        //点的时间等于scan时间加relTime（后文的laserCloudIn->points[i].time）
        //lasercloudin中存储的time是一帧中距离起始点的相对时间
        // 在cloudHandler的cachePointCloud函数中，timeScanCur = cloudHeader.stamp.toSec();，即当前帧点云的初始时刻
        //二者相加即可得到当前点的准确时刻
        double pointTime = timeScanCur + relTime;

        //根据时间戳插值获取imu计算的旋转量与位置量（注意imu计算的相对于起始时刻的旋转增量）
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        //这里的firstPointFlag来源于resetParameters函数，而resetParameters函数每次ros调用cloudHandler都会启动 第一个点的位姿增量（0），求逆
        //只针对同一帧激光数据的第一次执行
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false; 
        }

        // transform points to start
        //获取当前的lidar世界坐标系下的变换矩阵
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        //根据lidar位姿变换 Tij，修正点云位置: Tij * Pj
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        //点云数据量 用于下面一个个点投影
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            //laserCloudIn就是原始的点云话题中的数据
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            //距离图像的行  与点云中ring对应,
            //rowIdn计算出该点激光雷达是水平方向上第几线的。从下往上计数，-15度记为初始线，第0线，一共16线(N_SCAN=16
            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;
            
            //水平角分辨率
            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            //Horizon_SCAN=1800,每格0.2度
            static float ang_res_x = 360.0/float(Horizon_SCAN);

            //horizonAngle为[-180,180],horizonAngle为[-270,90],-round为[-90,270], ang_res_x 为[-450,1350]
            //+Horizon_SCAN/2为[450,2250]
            // 即把horizonAngle从[-180,180]映射到[450,2250]
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
     
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;
            //去畸变  运动补偿 这里需要用到雷达信息中的time 这个field
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);
            rangeMat.at<float>(rowIdn, columnIdn) = range;
            // 转换成一维索引，保存校正之后的激光点
            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;             
        }
    }

    void cloudExtraction()
    {
        int count = 0;
        // 有效激光点数量
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {
            
            //记录每根扫描线起始第5个激光点在一维数组中的索引
            //提取特征的时候，每一行的前5个和最后5个不考虑
            cloudInfo.startRingIndex[i] = count - 1 + 5;
            
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    // 记录激光点对应的Horizon_SCAN方向上的索引，并保存
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j); //激光点距离
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]); //加入有效激光点
                    // size of extracted cloud
                    ++count; 
                }
            }
            // 记录每根扫描线倒数第5个激光点在一维数组中的索引
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);    
        publishCloud(&pubExtractedCloudFront, extractedCloudFront, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}
