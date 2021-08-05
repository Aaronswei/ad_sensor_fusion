#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/features/don.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>
#include <velodyne_pointcloud/point_types.h>
#include <msg_box3d/Num.h>
#include <msg_box3d/NumArray.h>


// typedef pcl::PointCloud<velodyne_pointcloud::PointXYZIR> PointCloud1;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
// typedef pcl::PointCloud<pcl::PointXYZ> PointCloudRGB;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudRadar;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudCluster;
//typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
ros::Publisher pub;
ros::Publisher pub_cluster;
ros::Publisher pub_convex_hull;
ros::Publisher pub_radar_cloud;
ros::Publisher ind_pub;

///The smallest scale to use in the DoN filter.
double scale1 = 0.5;
///The largest scale to use in the DoN filter.
double scale2 = 5.0;
///The minimum DoN magnitude to threshold by
double threshold = 0.4;

///segment scene into clusters with given distance tolerance using euclidean clustering
double segradius = 10.0;

void callback(const PointCloud::ConstPtr& msg) 
{

    // Create a search tree, use KDTreee for non-organized data.
    pcl::search::Search<pcl::PointXYZ>::Ptr tree;
    if (msg->isOrganized()) {
        tree.reset(new pcl::search::OrganizedNeighbor<pcl::PointXYZ> ());
    } else {
        tree.reset(new pcl::search::KdTree<pcl::PointXYZ> (false));
    }

    // Set the input pointcloud for the search tree
    tree->setInputCloud(msg);
    if (scale1 >= scale2) {
        std::cerr << "Error: Large scale must be > small scale!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Compute normals using both small and large scales at each point
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::PointNormal> ne;
    ne.setInputCloud(msg);
    ne.setSearchMethod(tree);
    /**
     * NOTE: setting viewpoint is very important, so that we can ensure
     * normals are all pointed in the same direction!
     */
    ne.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

    // calculate normals with the small scale
    //    std::cout << "Calculating normals for scale..." << scale1 << std::endl;
    pcl::PointCloud<pcl::PointNormal>::Ptr normals_small_scale(new pcl::PointCloud<pcl::PointNormal>);

    ne.setRadiusSearch(scale1);
    ne.compute(*normals_small_scale);

    // calculate normals with the large scale
    //    std::cout << "Calculating normals for scale..." << scale2 << std::endl;
    pcl::PointCloud<pcl::PointNormal>::Ptr normals_large_scale(new pcl::PointCloud<pcl::PointNormal>);

    ne.setRadiusSearch(scale2);
    ne.compute(*normals_large_scale);

    // Create output cloud for DoN results
    pcl::PointCloud<pcl::PointNormal>::Ptr doncloud(new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud<pcl::PointXYZ, pcl::PointNormal>(*msg, *doncloud);
    //    std::cout << "Calculating DoN... " << std::endl;
    // Create DoN operator
    pcl::DifferenceOfNormalsEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::PointNormal> don;
    don.setInputCloud(msg);
    don.setNormalScaleLarge(normals_large_scale);
    don.setNormalScaleSmall(normals_small_scale);

    if (!don.initCompute()) {
        //        std::cerr << "Error: Could not intialize DoN feature operator" << std::endl;
        exit(EXIT_FAILURE);
    }
    // Compute DoN
    don.computeFeature(*doncloud);
    // Build the condition for filtering
    pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond( new pcl::ConditionOr<pcl::PointNormal> ()
            );
    range_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr( new pcl::FieldComparison<pcl::PointNormal> ("curvature", pcl::ComparisonOps::GT, threshold)) );
    // Build the filter
    pcl::ConditionalRemoval<pcl::PointNormal> condrem(range_cond);
    condrem.setInputCloud(doncloud);

    pcl::PointCloud<pcl::PointNormal>::Ptr doncloud_filtered(new pcl::PointCloud<pcl::PointNormal>);

    // Apply filter
    condrem.filter(*doncloud_filtered);

    doncloud = doncloud_filtered;

    // Save filtered output
    //    std::cout << "Filtered Pointcloud: " << doncloud->points.size() << " data points." << std::endl;

    //    // Create statistical outlayer removal the filtering object
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_filtered2(new pcl::PointCloud<pcl::PointNormal>);
    pcl::StatisticalOutlierRemoval<pcl::PointNormal> sor;
    sor.setInputCloud(doncloud);
    sor.setMeanK(20);
//    sor.setStddevMulThresh(1.0);
    sor.setStddevMulThresh(0.4);//desviacion estandar, los que esta fuera son eliminadoss
    sor.filter(*cloud_filtered2);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointNormal> seg;
    // Optional
    seg.setOptimizeCoefficients(true);
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
//    seg.setDistanceThreshold(0.12);
    seg.setDistanceThreshold(0.10);
    seg.setInputCloud(cloud_filtered2);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0) {
        PCL_ERROR("Could not estimate a planar model for the given dataset.");
        return;
    }

    double a = coefficients->values[0];
    double b = coefficients->values[1];
    double c = coefficients->values[2];
    double d = coefficients->values[3];

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud2(new pcl::PointCloud<pcl::PointNormal>), cloud_projected(new pcl::PointCloud<pcl::PointNormal>);

    // Project the model inliers
    pcl::ProjectInliers<pcl::PointNormal> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(cloud_filtered2);
    proj.setIndices(inliers);
    proj.setModelCoefficients(coefficients);
    proj.filter(*cloud_projected);

    // Create a Convex Hull representation of the projected inliers
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointNormal>);
    pcl::ConvexHull<pcl::PointNormal> chull;
    chull.setInputCloud(cloud_projected);
    chull.reconstruct(*cloud_hull);

    //    interseccion del plano con el eje y
    double x = 0;
    double z = 0;
//    double y = -(a*x + c*z + d) / b ;
    double y = -d / b ;
    int y_intercept = y;
    std::cerr << "interseccion del plano con el eje y "  << std::endl;    //    interseccion del plano con el eje y
	ROS_INFO_STREAM( "point y axis: " << x << " " << y << " " << z);
    double xplus1 = x;
    double yplus1 = y + 1;
    double zplus1 = z;
	ROS_INFO_STREAM( "point y+1 axis: " << xplus1 << " " << yplus1 << " " << zplus1);
    //    proyeccion de un punto del eje y (0,1,0) sobre el plano 
    double landa = -(a*xplus1 + b*yplus1 + c*zplus1 + d) / (pow(a,2) + pow(b,2) + pow(c,2));
    double Qx = xplus1 + a * landa;
    double Qy = yplus1 + b * landa;
    double Qz = zplus1 + c * landa;
	ROS_INFO_STREAM( "Q y axis: " << Qx << " " << Qy << " " << Qz << std::endl);

    double theta_z = acos( sqrt(pow(Qx - x, 2) + pow(Qy - y, 2) + pow(Qz - z, 2)) / 
    sqrt(pow(xplus1 - x, 2) + pow(yplus1 - y, 2) + pow(zplus1 - z, 2))    ); //angulo ente el plano del patron de calib y plano XY
//    double theta_z = acos(  ( ((Qx-x)*(xplus1-x)) + ((Qy-y)*(yplus1-y)) + ((Qz-z)*(zplus1-z)))  /  
//            sqrt(pow(Qx-x, 2) + pow(Qy-y, 2) + pow(Qz-z, 2)) * sqrt(pow(xplus1-x, 2) + pow( yplus1-y, 2) + pow( zplus1-z, 2))     ); //angulo ente el plano del patron de calib y plano XY
//    std::cerr << "theta_z: " << theta_z << std::endl;
//    interseccion del plano con el eje x
//    x = -(b*y + c*z + d) / a ;
    x = -d / a ;
    z = 0;
    y = 0;
    ROS_INFO_STREAM( "interseccion del plano con el eje x " );    //    interseccion del plano con el eje y
	ROS_INFO_STREAM( "point x axis: " << x << " " << y << " " << z );
    xplus1 = x + 1;
    yplus1 = y;
    zplus1 = z;
	ROS_INFO_STREAM( "point x+1 axis: " << xplus1 << " " << yplus1 << " " << zplus1 );
    //    proyeccion de un punto del eje x (1,0,0) sobre el plano 
    landa = -(a*xplus1 + b*yplus1 + c*zplus1 + d) / (pow(a,2) + pow(b,2) + pow(c,2));
    Qx = xplus1 + a * landa;
    Qy = yplus1 + b * landa;
    Qz = zplus1 + c * landa;
	ROS_INFO_STREAM( "Q x axis: " << Qx << " " << Qy << " " << Qz );

    double theta_y = acos(  sqrt(pow(Qx - x, 2) + pow(Qy - y, 2) + pow(Qz - z, 2))  /  
            sqrt(pow(xplus1 - x, 2) + pow(yplus1 - y, 2) + pow(zplus1 - z, 2))     ); //angulo ente el plano del patron de calib y plano XY

    ROS_INFO_STREAM( "theta y " << theta_y);
    ROS_INFO_STREAM( "theta z " << theta_z );

    cv::Matx33d roty(cos(theta_y), 0,  sin(theta_y),
                0,                 1,             0,
                -sin(theta_y),     0,  cos(theta_y));
      
    cv::Matx33d rotz(cos(theta_z),  -sin(theta_z),     0,
                     sin(theta_z),   cos(theta_z),     0,
                     0,              0,                1);     

    if ( y_intercept >= 0){
        rotz(0,0) = cos(-theta_z);   rotz(0,1) = -sin(-theta_z);   rotz(0,2) =  0;
        rotz(1,0) = sin(-theta_z);   rotz(1,1) = cos(-theta_z);    rotz(1,2) =  0;
        rotz(2,0) = 0;              rotz(2,1) = 0;               rotz(2,2) =  1;
    }

    cv::Matx31d T3;

    PointCloudRadar::Ptr msg_radar_to_stereo(new PointCloudRadar);
    pcl::PointCloud<pcl::PointNormal>::Ptr rect_cloud(new pcl::PointCloud<pcl::PointNormal>);
    rect_cloud->header = msg->header;
    msg_radar_to_stereo->header = msg->header;
    std::vector<cv::Point2f> poly;
    
    for (size_t i = 0; i < cloud_projected->points.size(); ++i) 
    {
        cv::Matx31d p2(cloud_projected->points[i].x - cloud_projected->points[0].x, cloud_projected->points[i].y - cloud_projected->points[0].y, cloud_projected->points[i].z - cloud_projected->points[0].z);
        T3 =    roty * 
                rotz * 
                p2;   
        ROS_INFO_STREAM( "rotated point   " << T3(0) << " "
                << T3(1) << " "
                << T3(2) );
        msg_radar_to_stereo->points.push_back(pcl::PointXYZ(T3(0), T3(1), T3(2))); ///////////////////stereo 
    	poly.push_back(cv::Point2f(T3(0), T3(1)));
    }

    cv::RotatedRect box = cv::minAreaRect(poly); // now it works!
    cv::Point2f vertices[4];
    box.points(vertices);

	for (int i = 0; i < 4; i++)
	{
	    ROS_INFO_STREAM( "rect x      "<< i << " " <<vertices[i] );
	    msg_radar_to_stereo->points.push_back(pcl::PointXYZ(vertices[i].x, vertices[i].y, 0.0f)); ///////////////////stereo 
	//    std::cout << "rect y      " <<box. << std::endl;
	//    std::cout << "rect width  " <<box.x << std::endl;
	//    std::cout << "rect height " <<box.x << std::endl;
	    cv::Matx31d p2(vertices[i].x /*- cloud_projected->points[0].x*/, vertices[i].y /*- cloud_projected->points[0].y*/, 0.0f /*- cloud_projected->points[0].z*/);
	    T3 = 
	        roty.inv() * 
	        rotz.inv() * 
	                p2;
	    ROS_INFO_STREAM( "rotated point   " << T3(0)+cloud_projected->points[0].x << " "
	            << T3(1)+cloud_projected->points[0].y << " "
	            << T3(2)+cloud_projected->points[0].z );
	    msg_radar_to_stereo->points.push_back(pcl::PointXYZ(T3(0)+cloud_projected->points[0].x, T3(1)+cloud_projected->points[0].y, T3(2)+cloud_projected->points[0].z)); ///////////////////stereo 
	    pcl::PointNormal p;
	    p.x = T3(0)+cloud_projected->points[0].x;
	    p.y = T3(1)+cloud_projected->points[0].y;
	    p.z = T3(2)+cloud_projected->points[0].z;
	    rect_cloud->points.push_back(pcl::PointNormal( p)); ///////////////////stereo 

	}
    pcl::PointCloud<pcl::PointNormal>::Ptr rect_cloud_proj(new pcl::PointCloud<pcl::PointNormal>);

    // Project the model inliers
    pcl::ProjectInliers<pcl::PointNormal> proj_rec;
    proj_rec.setModelType(pcl::SACMODEL_PLANE);
    proj_rec.setInputCloud(rect_cloud);
    proj_rec.setModelCoefficients(coefficients);
    proj_rec.filter(*rect_cloud_proj);
    doncloud->header = msg->header;
    pub_cluster.publish(cloud_filtered2);
    pub_convex_hull.publish(cloud_hull);
    pub_radar_cloud.publish(rect_cloud_proj);
    cloud_projected->clear(); 
    rect_cloud_proj->clear();
    msg_radar_to_stereo->clear();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "pub_pcl");
    ros::NodeHandle nh;
    pub_cluster = nh.advertise<PointCloudCluster> ("cluster", 1);
    pub_convex_hull = nh.advertise<PointCloudCluster> ("convex_hull", 1);
    pub_radar_cloud = nh.advertise<PointCloudRadar> ("pointsSquareCalib", 10);

    ros::Subscriber sub = nh.subscribe<PointCloud>("passthrough2", 1, callback); /////////////////////////////////////velodyne
    ros::spin();

}
