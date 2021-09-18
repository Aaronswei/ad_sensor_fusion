
#include <sensor_processing_lib/sensor_fusion.h>

namespace sensor_processing{

/******************************************************************************/

SensorFusion::SensorFusion(ros::NodeHandle nh, ros::NodeHandle private_nh):
	nh_(nh),
	private_nh_(private_nh),
	pcl_in_(new VPointCloud),
	pcl_ground_plane_(new VPointCloud),
	pcl_ground_plane_inliers_(new VPointCloud),
	pcl_ground_plane_outliers_(new VPointCloud),
	pcl_ground_(new VPointCloud),
	pcl_elevated_(new VPointCloud),
	pcl_voxel_ground_(new VPointCloud),
	pcl_voxel_elevated_(new VPointCloud),
	pcl_semantic_(new VRGBPointCloud),
	pcl_sparse_semantic_(new VRGBPointCloud),
	cloud_sub_(nh, "/kitti/velo/pointcloud", 2),
 	image_sub_(nh,	"/kitti/camera_color_left/semantic", 2),
	sync_(MySyncPolicy(10), cloud_sub_, image_sub_){

	// Get data path
	std::string home_dir;
	if(ros::param::get("~home_dir", home_dir)){
		params_.home_dir = home_dir + "/kitti_data";
	}
	else{
		ROS_ERROR("Set dataset path as parameter");
	}

	// Get scenario parameter
	int scenario;
	if(ros::param::get("~scenario", scenario)){
		std::ostringstream scenario_stream;
		scenario_stream << std::setfill('0') << std::setw(4) << scenario;
		params_.scenario = scenario_stream.str();
	}
	else{
		ROS_ERROR("Failed to read scenario");
	}

	// Define lidar parameters
	private_nh_.param("lidar/height", params_.lidar_height,
		params_.lidar_height);
	private_nh_.param("lidar/z_min", params_.lidar_z_min,
		params_.lidar_z_min);
	params_.lidar_opening_angle = M_PI / 4;

	// Define grid parameters
	private_nh_.param("grid/range/min", params_.grid_range_min,
		params_.grid_range_min);
	private_nh_.param("grid/range/max", params_.grid_range_max,
		params_.grid_range_max);
	private_nh_.param("grid/cell/size", params_.grid_cell_size,
		params_.grid_cell_size);
	private_nh_.param("grid/cell/height", params_.grid_cell_height,
		params_.grid_cell_height);
	private_nh_.param("grid/segments", params_.grid_segments,
		params_.grid_segments);
	params_.grid_height = params_.grid_range_max / params_.grid_cell_size ;
	params_.grid_width = params_.grid_height * 2;
	params_.grid_bins = (params_.grid_range_max * std::sqrt(2)) /
		params_.grid_cell_size + 1;

	// Define semantic parameters
	private_nh_.param("semantic/edge_detection/perform", params_.sem_ed,
		params_.sem_ed);
	private_nh_.param("semantic/edge_detection/min", params_.sem_ed_min,
		params_.sem_ed_min);
	private_nh_.param("semantic/edge_detection/max", params_.sem_ed_max,
		params_.sem_ed_max);
	private_nh_.param("semantic/edge_detection/kernel", params_.sem_ed_kernel,
		params_.sem_ed_kernel);
	// Define ransac ground plane parameters
	private_nh_.param("ransac/tolerance", params_.ransac_tolerance,
		params_.ransac_tolerance);
	private_nh_.param("ransac/iterations", params_.ransac_iterations,
		params_.ransac_iterations);

	// Define static conversion values
	params_.inv_angular_res = 2 * params_.grid_segments / M_PI;
	params_.inv_radial_res = 1.0f / params_.grid_cell_size;

	// Print parameters
	ROS_INFO_STREAM("scenario " << params_.scenario);
	ROS_INFO_STREAM("lidar_height " << params_.lidar_height);
	ROS_INFO_STREAM("lidar_z_min " << params_.lidar_z_min);
	ROS_INFO_STREAM("grid_range_min " << params_.grid_range_min);
	ROS_INFO_STREAM("grid_range_max " << params_.grid_range_max);
	ROS_INFO_STREAM("grid_height " << params_.grid_height);
	ROS_INFO_STREAM("grid_width " << params_.grid_width);
	ROS_INFO_STREAM("grid_cell_size " << params_.grid_cell_size);
	ROS_INFO_STREAM("grid_cell_height " << params_.grid_cell_height);
	ROS_INFO_STREAM("grid_bins " << params_.grid_bins);
	ROS_INFO_STREAM("grid_segments " << params_.grid_segments);
	ROS_INFO_STREAM("ransac_tolerance " << params_.ransac_tolerance);
	ROS_INFO_STREAM("ransac_iterations " << params_.ransac_iterations);
	ROS_INFO_STREAM("inv_angular_res " << params_.inv_angular_res);
	ROS_INFO_STREAM("inv_radial_res " << params_.inv_radial_res);

	// Define polar grid
	polar_grid_ = std::vector< std::vector<PolarCell> >(params_.grid_segments,
		std::vector<PolarCell>(params_.grid_bins));

	// Define occupancy grid
	occ_grid_ = boost::make_shared<OccupancyGrid>();
	occ_grid_->data.resize(params_.grid_width * params_.grid_height);
	occ_grid_->info.width = uint32_t(params_.grid_width);
	occ_grid_->info.height = uint32_t(params_.grid_height);
	occ_grid_->info.resolution = float(params_.grid_cell_size);
	occ_grid_->info.origin.position.x = params_.grid_range_max;
	occ_grid_->info.origin.position.y = params_.grid_range_max;
	occ_grid_->info.origin.position.z = params_.lidar_height;
	occ_grid_->info.origin.orientation.w = 0;
	occ_grid_->info.origin.orientation.x = 0.707;
	occ_grid_->info.origin.orientation.y = -0.707;
	occ_grid_->info.origin.orientation.z = 0;

	// Init occupancy grid
	for(int j = 0; j < params_.grid_height; ++j){
		for(int i = 0; i < params_.grid_width; ++i){

			// Never reach this cells because of opening angle
			if(i < j || i >= params_.grid_width - j){
				occ_grid_->data[j * params_.grid_width + i] = -1;
			}			
		}
	}

	// Define Publisher 
	cloud_filtered_pub_ = nh_.advertise<PointCloud2>(
		"/sensor/cloud/filtered", 2);
	cloud_ground_plane_inliers_pub_ = nh_.advertise<PointCloud2>(
		"/sensor/cloud/groundplane/inliers", 2);
	cloud_ground_plane_outliers_pub_ = nh_.advertise<PointCloud2>(
		"/sensor/cloud/groundplane/outliers", 2);
	cloud_ground_pub_ = nh_.advertise<PointCloud2>(
		"/sensor/cloud/ground", 2);
	cloud_elevated_pub_ = nh_.advertise<PointCloud2>(
		"/sensor/cloud/elevated", 2);
	voxel_ground_pub_ = nh_.advertise<PointCloud2>(
		"/sensor/voxel/ground", 2);
	voxel_elevated_pub_ = nh_.advertise<PointCloud2>(
		"/sensor/voxel/elevated", 2);
	grid_occupancy_pub_ = nh_.advertise<OccupancyGrid>(
		"/sensor/grid/occupancy", 2);

	image_semantic_pub_ = nh_.advertise<Image>(
		"/sensor/image/semantic", 2);
	cloud_semantic_pub_ = nh_.advertise<PointCloud2>(
		"/sensor/cloud/semantic", 2);
	cloud_semantic_sparse_pub_ = nh_.advertise<PointCloud2>(
		"/sensor/cloud/semantic_sparse", 2);
	image_detection_grid_pub_ = nh_.advertise<Image>(
		"/sensor/image/detection_grid", 2);
	image_bev_semantic_grid_pub_ = nh_.advertise<Image>(
		"/sensor/image/bev_semantic_grid", 2);

	// Define Subscriber
	sync_.registerCallback(boost::bind(&SensorFusion::process, this, _1, _2));

	// Init counter for publishing
	time_frame_ = 0;
}

SensorFusion::~SensorFusion(){

}

void SensorFusion::process(
		const PointCloud2::ConstPtr & cloud,
		const Image::ConstPtr & image
	){

	// Preprocess point cloud
	//对点云进行处理
	processPointCloud(cloud);

	// Preprocess image
	//仅仅是来保存图像
	processImage(image);

	// Fuse sensors by mapping elevated point cloud into semantic segmentated image
	//将地面以上的点云映射到语义分割的图像中
	mapPointCloudIntoImage(pcl_elevated_, image);

	// Print sensor fusion
	ROS_INFO("Publishing Sensor Fusion [%d]: # PCL points [%d] # Ground [%d]"
		" # Elevated [%d] # Semantic [%d] # Sparse Semantic [%d]", time_frame_,
		int(pcl_in_->size()), int(pcl_ground_->size()),
		int(pcl_elevated_->size()), int(pcl_semantic_->size()),
		int(pcl_sparse_semantic_->size()));

	// Increment time frame
	time_frame_++;

}

void SensorFusion::processPointCloud(const PointCloud2::ConstPtr & cloud){

/******************************************************************************
  主要通过点云网格化来进行点云地面分割的，即读取每个cell内点云极值、均值、方差等，判断与邻域内其他cell的相关性，
  然后判断栅格的占有率

 * 1. Filter point cloud to only consider points in the front that can also be
 * found in image space.
   主要是在图像空间内将点云找出来，同时点云序列化
 */

	// Convert input cloud
	pcl::fromROSMsg(*cloud, *pcl_in_);

	//对点云、索引以及极坐标进行重新定义，主要涉及按点云索引提取点云子集
	// Define point_cloud_inliers and indices
	pcl::PointIndices::Ptr pcl_inliers(new pcl::PointIndices());
	pcl::ExtractIndices<VPoint> pcl_extractor;

	// Reset polar grid
	polar_grid_ = std::vector< std::vector<PolarCell> >(
		params_.grid_segments, std::vector<PolarCell>(params_.grid_bins));

	/*
	通过对点云角度、范围、高度等信息来对点云进行过滤，然后对点云进行重新编号，并将点云转换到极坐标系下
	可以理解为将点云的无序点云划分为有序点云，划分标准有segment，bins
	这里的segment可以理解为极坐标系下的扇区，bins可以理解为对Segment的细分

	*/
	// Loop through input point cloud
	for(int i = 0; i < pcl_in_->size(); ++i){

		// Read current point
		VPoint & point = pcl_in_->at(i);

		// Determine angle of lidar point and check
		float angle = std::abs( std::atan2(point.y, point.x) );
		if(angle < params_.lidar_opening_angle){

			// Determine range of lidar point and check
			float range = std::sqrt(point.x * point.x + point.y * point.y);
			if(range > params_.grid_range_min &&
				range < params_.grid_range_max){

				// Check height of lidar point
				if(point.z > params_.lidar_z_min){

					// Add index for filtered point cloud
					pcl_inliers->indices.push_back(i);

					// Buffer variables
					int seg, bin;

					// Get polar grid cell indices
					fromVeloCoordsToPolarCell(point.x, point.y, seg, bin);

					// Grab cell
					PolarCell & cell = polar_grid_[seg][bin];

					// Increase count
					cell.count++;

					// Update min max
					if(cell.count == 1){
						cell.x_min = point.x;
						cell.y_min = point.y;
						cell.z_min = point.z;
						cell.z_max = point.z;
					}
					else{
						if(point.z < cell.z_min){
							cell.x_min = point.x;
							cell.y_min = point.y;
							cell.z_min = point.z;
						}
						if(point.z > cell.z_max){
							cell.z_max = point.z;
						}
					}
				}
			}
		}
	}

	// Extract points from original point cloud
	pcl_extractor.setInputCloud(pcl_in_);
	pcl_extractor.setIndices(pcl_inliers);
	pcl_extractor.setNegative(false);
	pcl_extractor.filter(*pcl_in_);

	// Publish filtered cloud
	pcl_in_->header.frame_id = cloud->header.frame_id;
	pcl_in_->header.stamp = pcl_conversions::toPCL(cloud->header.stamp);
	cloud_filtered_pub_.publish(pcl_in_);

/******************************************************************************
 * 2. Ground plane estimation and dividing point cloud in elevated and ground
	通过地面估计，将地面点云和非地面点云划分出来，这是由于地面和非地面数据有明显的竖直维度的落差
 */
	// Clear ground plane points
	pcl_ground_plane_->points.clear();

	//依据极坐标单元中的相关属性，来构建地面，主要是进行最小z点的选取，来进行地面点的筛选
	// Loop over polar grid
	for(int i = 0; i < params_.grid_segments; ++i){
		for(int j = 0; j < params_.grid_bins; ++j){

			// Grab cell
			PolarCell & cell = polar_grid_[i][j];

			// Check if cell can be ground cell
			if(cell.count > 0 &&
				(cell.z_max - cell.z_min < params_.grid_cell_height)){

				// Push back cell attributes to ground plane cloud
				pcl_ground_plane_->points.push_back(
					VPoint(cell.x_min, cell.y_min, cell.z_min));
			}
		}
	}

	//下面就是使用PCL和RANSAC来估计地平面，主要涉及到点云分割模块SACSegmentation
	// Estimate the ground plane using PCL and RANSAC
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

	// Create the segmentation object
	pcl::SACSegmentation<VPoint> segmentation;
	segmentation.setOptimizeCoefficients(true);	 //该模型系数是否需要优化
	segmentation.setModelType(pcl::SACMODEL_PLANE); // 所提取目标模型的属性(平面、球面、圆柱) 平面的法线式：ax+by+cz=d，其中a^2+b^2+c^2=1，d>0，（a,b,c）为平面法矢，d为原点至平面的距离，这四个参数可以确定一个平面。
	segmentation.setMethodType(pcl::SAC_RANSAC); //设定采样算法（RANSAC，LMeds）
	segmentation.setDistanceThreshold(params_.ransac_tolerance); //设定查询点到目标模型的距离阈值（大于阈值，查询点不在该目标模型上，默认为0）
	segmentation.setMaxIterations(params_.ransac_iterations); //设定最大迭代次数
	segmentation.setInputCloud(pcl_ground_plane_->makeShared());
	segmentation.segment(*inliers, *coefficients);  //输出提取点的索引和目标模型的参数

	// Divide ground plane cloud in inlier cloud and outlier cloud
	//提取地面点云中的inlier cloud
	pcl_extractor.setInputCloud(pcl_ground_plane_);
	pcl_extractor.setIndices(inliers);
	pcl_extractor.setNegative(false);
	pcl_extractor.filter(*pcl_ground_plane_inliers_); //在下采样基础之上进行分割和ExtractInliers滤波的  结果就是一个平面模型映射的结果

	//提取地面点云中的outlier cloud
	pcl_extractor.setInputCloud(pcl_ground_plane_);
	pcl_extractor.setIndices(inliers);
	pcl_extractor.setNegative(true);
	pcl_extractor.filter(*pcl_ground_plane_outliers_); //在体素栅格下采样基础之上得到的点云数据和提取出的不符合条件的点云数据进行调换，进入下一轮循环

	// Sanity check
	if(inliers->indices.empty() || coefficients->values[3] > 2 || 
		coefficients->values[3] < 1.5){
		ROS_WARN("Bad ground plane estimation! # Ransac Inliers [%d] # Lidar "
			"height [%f]", int(inliers->indices.size()),
			coefficients->values[3]);
	}

	// Publish ground plane inliers and outliers point cloud
	pcl_ground_plane_inliers_->header.frame_id = cloud->header.frame_id;
	pcl_ground_plane_inliers_->header.stamp = 
		pcl_conversions::toPCL(cloud->header.stamp);
	cloud_ground_plane_inliers_pub_.publish(pcl_ground_plane_inliers_);

	pcl_ground_plane_outliers_->header.frame_id = cloud->header.frame_id;
	pcl_ground_plane_outliers_->header.stamp = 
		pcl_conversions::toPCL(cloud->header.stamp);
	cloud_ground_plane_outliers_pub_.publish(pcl_ground_plane_outliers_);

	// Print
	ROS_INFO("Ground plane estimation [%d] # Points [%d] # Inliers [%d] "
		" C [%f][%f][%f][%f]",	time_frame_, 
		int(pcl_ground_plane_->size()),	int(pcl_ground_plane_inliers_->size()), 
		coefficients->values[0], coefficients->values[1],
		coefficients->values[2], coefficients->values[3]);

/******************************************************************************
 * 3. Evaluate segments of polar grid to fill with unknown, free or occupied
   判断极坐标的cell是空闲的、占用的还是未知的，将点云从极坐标转换到Velo坐标系获取地面的高度信息，然后根据点云的高度判断cell的属性
 */
	
	// Loop over segments
	for(int s = 0; s < params_.grid_segments; s++){

		// Set hit to false
		bool hit = false;

		// Loop over bins
		for(int b = 0; b < params_.grid_bins; b++){
		
			// Grab cell
			PolarCell & cell = polar_grid_[s][b];

			// Buffer variables
			float x,y;

			// Get velodyne coodinates
			fromPolarCellToVeloCoords(s, b, x, y);

			// Get ground height
			cell.ground = (-coefficients->values[0] * x -
				coefficients->values[1] * y - coefficients->values[3]) /
				coefficients->values[2];

			// If cell is not filled
			if(cell.count == 0){

				// And has hit sth so far mark as unknown
				if(hit)
					cell.idx = PolarCell::UNKNOWN;

				// And has not hit sth so far mark as free
				else
					cell.idx = PolarCell::FREE;

				continue;
			}
			else{

				// Calculate cell height 
				cell.height = polar_grid_[s][b].z_max - cell.ground;

				// If cell height big enough fill cell as occupied
				if(cell.height > params_.grid_cell_height){

					cell.idx = PolarCell::OCCUPIED;

					// Mark segment as hit
					hit = true;
				}
				else{
					
					// And has hit sth so far mark as unknown
					if(hit)
						cell.idx = PolarCell::UNKNOWN;

					// And has not hit sth so far mark as free
					else
						cell.idx = PolarCell::FREE;
				}
			}
		}	
	}

	// Divide filtered point cloud in elevated and ground
	pcl_ground_->points.clear();
	pcl_elevated_->points.clear();

	for(int i = 0; i < pcl_in_->size(); ++i){

		// Read current point
		VPoint point = pcl_in_->at(i);

		// Buffer variables
		int seg, bin;

		// Get polar grid cell indices
		fromVeloCoordsToPolarCell(point.x, point.y, seg, bin);

		// Grab cell
		PolarCell & cell = polar_grid_[seg][bin];

		if(point.z > cell.ground && cell.height > params_.grid_cell_height){
			pcl_elevated_->points.push_back(point);
		}
		else{
			pcl_ground_->points.push_back(point);
		}
	}

	// Publish ground cloud
	pcl_ground_->header.frame_id = cloud->header.frame_id;
	pcl_ground_->header.stamp = pcl_conversions::toPCL(cloud->header.stamp);
	cloud_ground_pub_.publish(pcl_ground_);

	// Publish elevated cloud
	pcl_elevated_->header.frame_id = cloud->header.frame_id;
	pcl_elevated_->header.stamp = pcl_conversions::toPCL(cloud->header.stamp);
	cloud_elevated_pub_.publish(pcl_elevated_);	

/******************************************************************************
 * 4. Map polar grid back to cartesian occupancy grid
	将极坐标投影回笛卡尔坐标系，反映网格占有率
 */
	// Clear voxel pcls
	pcl_voxel_elevated_->points.clear();
	pcl_voxel_ground_->points.clear();

	// Init detection image and fill free space grid cells
	detection_grid_ = cv::Mat(params_.grid_height, params_.grid_width, CV_32FC3,
		cv::Scalar(-100.0, 0.0, 0.0));
	// Init detection image and fill free space grid cells
	bev_semantic_grid_ = cv::Mat(params_.grid_height, params_.grid_width, CV_8UC3,
		cv::Scalar(128, 128, 128));

	// Go through cartesian grid
	float x = params_.grid_range_max - params_.grid_cell_size / 2;
	for(int j = 0; j < params_.grid_height; ++j, x -= params_.grid_cell_size){
		float y = x;
		for(int i = j; i < params_.grid_width - j; ++i,
			y -= params_.grid_cell_size){

			// Buffer variables
			int seg, bin;

			// Get polar grid cell indices
			fromVeloCoordsToPolarCell(x, y, seg, bin);
			
			// Grab polar cell
			PolarCell & cell = polar_grid_[seg][bin];

			// Fill ground voxel cloud
			pcl_voxel_ground_->points.push_back( VPoint(x, y, cell.ground) );

			// Calculate occupancy grid cell index
			int cell_index = j * params_.grid_width + i;

			// If cell is free
			if(cell.idx == PolarCell::FREE){
				occ_grid_->data[cell_index] = 0;
				detection_grid_.at<cv::Vec3f>(j, i)[0] = -50.0;
				bev_semantic_grid_.at<cv::Vec3b>(j, i) = cv::Vec3b(255, 255, 255);
			}
			// If cell is unknown
			else if(cell.idx == PolarCell::UNKNOWN)
				occ_grid_->data[cell_index] = 50;

			// If cell is occupied
			else{
				occ_grid_->data[cell_index] = 100;

				// Fill elevated voxel cloud
				for(float v = cell.ground; v < cell.z_max; 
					v += params_.grid_cell_size){
						pcl_voxel_elevated_->points.push_back(
							VPoint(x, y, v));
				}
			}
		}
	}

	// Publish voxel ground
	pcl_voxel_ground_->header.frame_id = cloud->header.frame_id;
	pcl_voxel_ground_->header.stamp = 
		pcl_conversions::toPCL(cloud->header.stamp);
	voxel_ground_pub_.publish(pcl_voxel_ground_);

	// Publish voxel elevated
	pcl_voxel_elevated_->header.frame_id = cloud->header.frame_id;
	pcl_voxel_elevated_->header.stamp = 
		pcl_conversions::toPCL(cloud->header.stamp);
	voxel_elevated_pub_.publish(pcl_voxel_elevated_);

	// Publish occupancy grid
	occ_grid_->header.stamp = cloud->header.stamp;
	occ_grid_->header.frame_id = cloud->header.frame_id;
	occ_grid_->info.map_load_time = occ_grid_->header.stamp;
	grid_occupancy_pub_.publish(occ_grid_);
}

void SensorFusion::processImage(const Image::ConstPtr & msg){

/******************************************************************************
	读取分割后的图像信息，然后保存发布
 */
 	cv_bridge::CvImagePtr imagePtr = cv_bridge::toCvCopy(msg, "bgr8");
	sem_image_ = imagePtr->image.clone();

	// Sanity check if image is loaded correctly
	if(sem_image_.cols == 0 || sem_image_.rows == 0){
		ROS_WARN("Hardcode path in sensor_fusion.cpp processImage()!");
		return;
	}

	// Publish
	cv_bridge::CvImage cv_semantic_image;
	cv_semantic_image.image = sem_image_;
	cv_semantic_image.encoding = "bgr8";
	cv_semantic_image.header.stamp = msg->header.stamp;
	image_semantic_pub_.publish(cv_semantic_image.toImageMsg());
}

void SensorFusion::mapPointCloudIntoImage(const VPointCloud::Ptr cloud,
	const Image::ConstPtr & image){

/******************************************************************************
 * 1. Convert velodyne points into image space
	将点云投影到图像空间中，主要有：
	1）利用外参信息将点云投影到图像空间，判断点云的有效性，并与语义分割的信息进行融合，即得到RGBXYZ信息
	2）将笛卡尔grid cell中的语义属性进行拼接，以hash表的形式进行保存
	3）利用上一步的hash表，填充检测grid image和稀疏语义点云
 */

	// Define matrix to write velodyne points into
	MatrixXf matrix_velodyne_points = MatrixXf::Zero(4, cloud->size());
	for(int i = 0; i < cloud->size(); ++i){
		matrix_velodyne_points(0,i) = cloud->points[i].x;
		matrix_velodyne_points(1,i) = cloud->points[i].y;
		matrix_velodyne_points(2,i) = cloud->points[i].z;
		matrix_velodyne_points(3,i) = 1;
	}

	// Project the matrix from velodyne coordinates to the image plane
	//将定义的点云矩阵投影到图像平面,这里的外参都已经给出了
	MatrixXf matrix_image_points = 
		tools_.transformVeloToImage(matrix_velodyne_points);

	// Get image format
	int img_width = sem_image_.cols;
	int img_height = sem_image_.rows;

	// Clear semantic cloud
	pcl_semantic_->points.clear();

	// Loop over image points
	for(int i = 0; i < matrix_image_points.cols(); i++){

		// Check if image point is valid
		//判断点云图像的有效性
		const int & img_x = matrix_image_points(0, i);
		const int & img_y = matrix_image_points(1, i);
		const int & img_z = matrix_image_points(2, i);

		if( (img_x >= 0 && img_x < img_width) &&
			(img_y >= 0 && img_y < img_height) &&
			(img_z >= 0)){

			// Get R G B values of semantic image
			//获取分割图像的RGB信息
			uint8_t r = sem_image_.at<cv::Vec3b>(img_y,img_x)[2];
			uint8_t g = sem_image_.at<cv::Vec3b>(img_y,img_x)[1];
			uint8_t b = sem_image_.at<cv::Vec3b>(img_y,img_x)[0];

			// Create new point and fill it
			//RGBXYZ属性填充
			VRGBPoint point;
			point.x = cloud->points[i].x;
			point.y = cloud->points[i].y;
			point.z = cloud->points[i].z;
			point.r = r;
			point.g = g;
			point.b = b;

			// Push back point
			pcl_semantic_->points.push_back(point);
		}
	}

	// Sanity check
	if(pcl_semantic_->empty()){
		ROS_WARN("Empty semantic point cloud!");
		return;
	}

	// Publish semantic cloud
	pcl_semantic_->header.frame_id = cloud->header.frame_id;
	pcl_semantic_->header.stamp = cloud->header.stamp;
	cloud_semantic_pub_.publish(pcl_semantic_);

/******************************************************************************
 * 2. Gather in each cartesian grid cell the semantic labels
  收集每个笛卡尔grid cell中的语义标签，主要采用的是hash表的方法
 */	

	// Define hash table to remember points of semantic point cloud in each cell
	std::map<int, std::map<int,int> > cell_hash_table;

	// Loop through semantic point cloud
	for(int i = 0; i < pcl_semantic_->size(); ++i){

		// Read current point
		VRGBPoint & point = pcl_semantic_->at(i);

		// Buffer variables
		int grid_x, grid_y;

		// Get cartesian grid indices
		fromVeloCoordsToCartesianCell(point.x, point.y, grid_x, grid_y);
		int grid_occ = grid_y * params_.grid_width + grid_x;

		// Get semantic class
		int semantic_class = tools_.SEMANTIC_COLOR_TO_CLASS[
			point.r + point.g + point.b];

		// Increment hash table counter for this grid cell
		cell_hash_table[grid_occ][semantic_class]++;
	}

/******************************************************************************
 * 3. Fill detection grid image and sparse semantic point cloud
 */	

	// Init  sparse semantic point cloud
	pcl_sparse_semantic_->points.clear();

	// Loop over hash table to find most dominant semantic label
	std::map<int, std::map<int,int> >::iterator it;
	for(it = cell_hash_table.begin(); it != cell_hash_table.end(); it++ ){

		// Loop through all hits in cell and determine best semantic label
		std::map<int,int>::iterator it2;
		int max_semantic_counter = -1;
		int max_class;
		for(it2 = it->second.begin(); it2 != it->second.end(); it2++ ){

			if(it2->second > max_semantic_counter){
				max_semantic_counter = it2->second;
				max_class = it2->first;
			}
		}

		// Determine cartesian grid indices
		int grid_x = it->first % params_.grid_width;
		int grid_y = it->first / params_.grid_width;

		// Buffer variables
		float x, y;
		int seg, bin;

		// Calculate velodyne coordinates
		fromCartesianCellToVeloCoords(grid_x, grid_y, x, y);

		// Calculate polar grid indices to grab polar cell
		fromVeloCoordsToPolarCell(x, y, seg, bin);
		PolarCell & cell = polar_grid_[seg][bin];

		// Write point to sparse point cloud
		VRGBPoint point;
		point.x = x;
		point.y = y;
		point.z = cell.ground;
		point.r = tools_.SEMANTIC_CLASS_TO_COLOR(max_class,0);
		point.g = tools_.SEMANTIC_CLASS_TO_COLOR(max_class,1);
		point.b = tools_.SEMANTIC_CLASS_TO_COLOR(max_class,2);
		pcl_sparse_semantic_->points.push_back(point);

		// Fill detection grid with semantic class
		detection_grid_.at<cv::Vec3f>(grid_y, grid_x)[0] = max_class;
		detection_grid_.at<cv::Vec3f>(grid_y, grid_x)[1] = cell.ground;
		detection_grid_.at<cv::Vec3f>(grid_y, grid_x)[2] = cell.ground + cell.height;
		// BGR assignment
		bev_semantic_grid_.at<cv::Vec3b>(grid_y, grid_x)[0] = tools_.SEMANTIC_CLASS_TO_COLOR(max_class,2);
		bev_semantic_grid_.at<cv::Vec3b>(grid_y, grid_x)[1] = tools_.SEMANTIC_CLASS_TO_COLOR(max_class,1);
		bev_semantic_grid_.at<cv::Vec3b>(grid_y, grid_x)[2] = tools_.SEMANTIC_CLASS_TO_COLOR(max_class,0);
	}

	// Publish sparse semantic cloud
	pcl_sparse_semantic_->header.frame_id = cloud->header.frame_id;
	pcl_sparse_semantic_->header.stamp = cloud->header.stamp;
	cloud_semantic_sparse_pub_.publish(pcl_sparse_semantic_);

	// Publish detection grid
	cv_bridge::CvImage cv_detection_grid_image;
	cv_detection_grid_image.image = detection_grid_;
	cv_detection_grid_image.encoding = image_encodings::TYPE_32FC3;
	cv_detection_grid_image.header.stamp = image->header.stamp;
	image_detection_grid_pub_.publish(cv_detection_grid_image.toImageMsg());
	cv_bridge::CvImage cv_bev_semantic_grid_image;
	cv_bev_semantic_grid_image.image = bev_semantic_grid_;
	cv_bev_semantic_grid_image.encoding = image_encodings::TYPE_8UC3;
	cv_bev_semantic_grid_image.header.stamp = image->header.stamp;
	image_bev_semantic_grid_pub_.publish(cv_bev_semantic_grid_image.toImageMsg());
}
//将Velo坐标系转换到极坐标系
void SensorFusion::fromVeloCoordsToPolarCell(const float x, const float y,
		int & seg, int & bin){

	float mag = std::sqrt(x * x + y * y);
	float ang = -std::atan2(y, x);
	seg = int((ang + params_.lidar_opening_angle) * params_.inv_angular_res);
	bin = int(mag * params_.inv_radial_res);

	// For last segment
	if(x == -y)
		seg = params_.grid_segments - 1;
}

//将极坐标系转换到Velo坐标系
void SensorFusion::fromPolarCellToVeloCoords(const int seg, const int bin,
		float & x, float & y){

	float mag = bin / params_.inv_radial_res;
	float ang = seg / params_.inv_angular_res - params_.lidar_opening_angle;
	y = - std::sin(ang) * mag;
	x = std::cos(ang) * mag;
}
//将Velo坐标系转换到笛卡尔坐标系
void SensorFusion::fromVeloCoordsToCartesianCell(const float x, const float y,
		int & grid_x, int & grid_y){

	grid_y = params_.grid_height - x / params_.grid_cell_size;
	grid_x = - y / params_.grid_cell_size + params_.grid_height;
}
//从笛卡尔坐标系转换到Velo坐标系
void SensorFusion::fromCartesianCellToVeloCoords(const int grid_x,
	const int grid_y, float & x, float & y){

	x = (params_.grid_height - grid_y) * params_.grid_cell_size - 
		params_.grid_cell_size / 2;
	y = (params_.grid_height - grid_x) * params_.grid_cell_size -
		params_.grid_cell_size / 2;
}

} // namespace sensor_processing
