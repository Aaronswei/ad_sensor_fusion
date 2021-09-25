#include "utility.h"
#include "lidar_imu_fusion/cloud_info.h"

struct smoothness_t{ 
    float value;
    size_t ind;
};

/**
 * 曲率比较函数，从小到大排序
*/
struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    ros::Subscriber subLaserCloudInfo;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    lidar_imu_fusion::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    //用来做曲率计算的中间变量
    float *cloudCurvature;
    // 特征提取标记，1表示遮挡、平行，或者已经进行特征提取的点，0表示还未进行特征提取处理
    int *cloudNeighborPicked;
    // 1表示角点，-1表示平面点
    int *cloudLabel;

    //构造函数，订阅点云数据，并将处理后的点云数据发布出去
    FeatureExtraction()
    {
        // 订阅当前激光帧运动畸变校正后的点云信息
        subLaserCloudInfo = nh.subscribe<lidar_imu_fusion::cloud_info>(PROJECT_NAME + "/lidar/deskew/cloud_info", 1, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        // 发布当前激光帧提取特征之后的点云信息、角点点云、面点点云
        pubLaserCloudInfo = nh.advertise<lidar_imu_fusion::cloud_info> (PROJECT_NAME + "/lidar/feature/cloud_info", 1);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_corner", 1);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_surface", 1);
        
        initializationValue();
    }

    //初始化变量
    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }
    //接收imageProjection.cpp中发布的去畸变的点云，实时处理的回调函数
    void laserCloudInfoHandler(const lidar_imu_fusion::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn; // new cloud info
        cloudHeader = msgIn->header; // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction

        //当前激光帧点云中每个点的曲率，为了判断点属于哪一类（边缘点或跳跃点）
        calculateSmoothness();

        //特征点属于遮挡or平行，不进行特征提取，主要是为了后续提特征点是的误操作
        markOccludedPoints();

        // 点云角点、平面点特征提取
        // 1、遍历扫描线，每根扫描线扫描一周的点云划分为6段，针对每段提取20个角点、不限数量的平面点，加入角点集合、平面点集合
        // 2、认为非角点的点都是平面点，加入平面点云集合，最后降采样
        extractFeatures();

        publishFeatureCloud();
    }

    //计算点的曲率，采用的方法为将当前点的距离与前后各5点计算误差累计和；显然误差累计和较大，则此点属于跳跃点，或边缘点；
    void calculateSmoothness()
    {
        // 遍历当前激光帧运动畸变校正后的有效点云
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {
            // 当前点的10倍与水平方向上前后5个点，距离差。差越大，表明当前点为1边缘点, 连续点云边缘的点最大
            // 用当前激光点前后5个点计算当前点的曲率
            //注意，这里把前后五个点共10个点加起来，还减去了10倍的当前点
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];            
            // 距离差值平方作为曲率
            //cloudNeighborPicked：0表示还未进行特征提取处理,1表示遮挡、平行，或者已经进行特征提取的点 
            //cloudLabel：1表示角点，-1表示平面点
            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;

            // 存储该点曲率值、激光点一维索引
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }
    /*
    特征点提取，防止提取同一位置附近的多个点或者边缘跳跃点，以及是否遮挡
    */
    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            /*
            遍历提取相邻的两个点的距离，并获取相邻两个点的在水平方向上的索引号差
            */
            // 当前点和下一个点的range值
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            
            // 计算两个激光点之间的一维索引差值，如果在一条扫描线上，那么值为1；
            // 如果两个点之间有一些无效点被剔除了，可能会比1大，但不会特别大
            // 如果恰好前一个点在扫描一周的结束时刻，下一个点是另一条扫描线的起始时刻，那么值会很大
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));

            // 如果水平索引在10个点内，将远处的边缘的5个点标记为1, 近处的边缘点则为0
            // 两个点在同一扫描线上，且距离相差大于0.3，认为存在遮挡关系（即不在同一个面上，如果在同一平面上，距离相差不会太大）
            if (columnDiff < 10){
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }

            // parallel beam
            // 用前后相邻点判断当前点所在平面是否与激光束方向平行
            // diff1和diff2是当前点距离前后两个点的距离
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));
            
            // 如果两个相邻点的距离超过本点的距离0.02倍时，也标记为1，即孤立的点标记为1
            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    /*
    特征点提取
    平面点及其角点提取时，均是对40线中其中一条激光线点云进行判断是否为连续点还是断开边缘点。因此垂直方向上40条线独立处理。
    由于每条激光线均是360度，并将其一圈的所有数据进行平均分成6等份分别进行特征点提取；取其中1份，将每个点按照平滑参数从小到大进行排序；
    */
    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();
            // 将一条扫描线扫描一周的点云数据，划分为6段，每段分开提取有限数量的特征，保证特征均匀分布
            for (int j = 0; j < 6; j++)
            {
                // 每段点云的起始、结束索引；startRingIndex为扫描线起始第5个激光点在一维数组中的索引
                // startRingIndex和 endRingIndex 在imageProjection.cpp中的 cloudExtraction函数里被填入
                // 所有的点云在这里都是以"一维数组"的形式保存
                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                // 按照曲率从小到大排序点云，可以看出之前的byvalue在这里被当成了判断函数来用
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                //寻找角点特征点
                /*
                角点特征点提取：按照曲率从大到小遍历，即如此可最先遍历到最大的跳跃点，即边缘特征点。
                其中边缘特征点条件：
                    不得是远处跳跃到近处的边缘点；
                    不得是地面上点；
                    并且曲率需大于一定值；
                    特征点相邻的5个点也不得是特征点，即需要设置屏蔽位
                */
                int largestPickedNum = 0;
                // 按照曲率从大到小遍历
                for (int k = ep; k >= sp; k--)
                {
                    // 激光点的索引
                    int ind = cloudSmoothness[k].ind;
                    // 当前激光点还未被处理，且曲率大于阈值，则认为是角点
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        // 统计满足条件的个数
                        // 每段只取20个角点,为稍微陡峭的点(记录最大的两个点，为最陡的两个点)，后面的不考虑
                        // 如果单条扫描线扫描一周是1800个点，则划分6段，每段300个点，从中提取20个角点
                        largestPickedNum++;
                        if (largestPickedNum <= 20){
                            // 标记为角点,加入角点点云
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }

                        // 此点近距离边缘端点，已经处理过并将其设置为1
                        cloudNeighborPicked[ind] = 1;

                        // 如果当前点在最后5个点内，不处理
                        // 同一条扫描线上后5个点标记一下，不再处理，避免特征聚集
                        for (int l = 1; l <= 5; l++)
                        {
                            // 如果相邻有效索引的两点，在水平上索超出10（即10个水平角度分辨率），便跳出，即不平坦
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            //大于10，说明距离远，则不作标记
                            if (columnDiff > 10)
                                break;
                            // 即与当前点相对平滑的5个点设置为1，
                            cloudNeighborPicked[ind + l] = 1;
                        }

                        // 若当前点在前5个点内，不处理
                        // 同一条扫描线上前5个点标记一下，不再处理，避免特征聚集
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                //寻找面特征点
                /*
                按照曲率从小到大遍历，即如此可最先遍历到最平滑点，连续平滑中间的点。
                其中平滑特征点条件：
                    不得是已标记；
                    必须是地面上点；
                    并且平滑性需小于一定值；
                    特征点相邻的5个点也不得是特征点，即需要设置屏蔽位
                */

                // 按照曲率从小到大遍历
                for (int k = sp; k <= ep; k++)
                {
                    // 激光点的索引
                    int ind = cloudSmoothness[k].ind;
                    // 平滑的点，且是地面上的点
                    // 当前激光点还未被处理，且曲率小于阈值，则认为是平面点
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {
                        // 标记平滑地面的点为-1
                        cloudLabel[ind] = -1;
                        // 标记已被处理
                        cloudNeighborPicked[ind] = 1;

                        // 同一条扫描线上后5个点标记一下，不再处理，避免特征聚集
                        for (int l = 1; l <= 5; l++) {
                            // 每选择一个点，则便使相邻的5个点，若是平滑的，则使其标记为1，
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            // 此操作可作为降采样功能，使特征点至少间隔5个点
                            cloudNeighborPicked[ind + l] = 1;
                        }

                        // 同一条扫描线上前5个点标记一下，不再处理，避免特征聚集
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                // 放入平滑面点云
                // 平面点和未被处理的点(<=0)，都认为是平面点，加入平面点云集合
                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }
            // 平面点云降采样
            surfaceCloudScanDS->clear();
            // 加入平面点云集合
            // 将surfaceCloudScan数据，放到downSizeFilter里，
            downSizeFilter.setInputCloud(surfaceCloudScan);
            //把结果输出到*surfaceCloudScanDS里，DS指的是DownSample
            downSizeFilter.filter(*surfaceCloudScanDS);

            //最后把DS装到surfaceCloud中
            //同样角点（边缘点）则没有这样的操作，直接就用cornerCloud来装点云。
            *surfaceCloud += *surfaceCloudScanDS;      
        }
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");

    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Feature Extraction Started.\033[0m");
   
    ros::spin();

    return 0;
}