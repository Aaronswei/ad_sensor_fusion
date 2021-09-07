#include "camera_camera_sync/camera_camera_sync.hpp"
#include <dirent.h>
#include <unistd.h>
#include <ros/ros.h>
#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace std;

void CameraCameraSync::getFiles(string path, vector<string>& files)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    char* basePath = const_cast<char*>(path.c_str()); 


    if ((dir=opendir(basePath)) == NULL)
    {
        perror("Open dir error...");
        exit(1);

    }

    while ((ptr=readdir(dir)) != NULL)
    {
        // current dir 
        if(strcmp(ptr->d_name, ".")==0 || strcmp(ptr->d_name, "..")==0)
            continue;
        else if(ptr->d_type == 8) // file
            sprintf(base, "%s/%s", basePath, ptr->d_name);
        //puts(base);
        files.push_back(std::string(base));
    }
}

void CameraCameraSync::getImageTimeStamp(std::string oriDirName, std::string dstDirName)
{
    //采用该函数遍历获得得队列不是顺序的，正好适合采用时间距离最近法来匹配
    getFiles(oriDirName, oriImageLists_);
    getFiles(dstDirName, dstImageLists_);
    if(oriImageLists_.size() != dstImageLists_.size())
    {
        std::cout << "the two image lists not equal!" << std::endl;
        ROS_ERROR_STREAM("the two image lists not equal!");
        return;
    }
}

int CameraCameraSync::getImageNumber()
{
    if(oriImageLists_.size() != dstImageLists_.size())
    {
        std::cout << "the two image lists not equal!" << std::endl;
        ROS_ERROR_STREAM("the two image lists not equal!");
        return -1;
    }
    return oriImageLists_.size();
}

double CameraCameraSync::getbaseTime(std::string pngfilenames, std::string patt)
{
    size_t pattern = pngfilenames.find(patt);
    std::string baseFile = pngfilenames.substr(pattern-18, 17); // 仅针对该项目所提供的文件适用
    double baseImageTime = atof(baseFile.c_str());  
    return baseImageTime;
}

std::vector<std::pair<std::string, std::string> > CameraCameraSync::imageTimeStampSyncFuncion()
{
    std::vector<std::pair<std::string, std::string> > syncPairLists;

    double timeDifference;
    for(auto baseFileNames : oriImageLists_)
    {
        double maxSSIM = 0;
        std::string anchorFilenames;
        double baseImageTime = getbaseTime(baseFileNames, "png");

        for(auto candidateFileNames : dstImageLists_)
        {
            double candidateImageTime = getbaseTime(candidateFileNames, "png");
            timeDifference = std::abs(baseImageTime - candidateImageTime);
            if(timeDifference <= 0.1)
            {
                cv::Mat orgImage = cv::imread(baseFileNames, cv::IMREAD_GRAYSCALE);
                cv::Mat dstImage = cv::imread(candidateFileNames, cv::IMREAD_GRAYSCALE);
                if( !orgImage.data || !dstImage.data )
                { 
                    std::cout<< " --(!) Error reading images " << std::endl; 
                    break;
                }
                double ssim = evaluateImageTimeStampSync(orgImage, dstImage);
                if (ssim > maxSSIM)
                {
                    maxSSIM = ssim;
                    anchorFilenames = candidateFileNames;
                }
            }
        }
        std::pair<std::string, std::string> syncPair(std::make_pair(baseFileNames, anchorFilenames));
        syncPairLists.push_back(syncPair);
        std::cout << " Get the "<< baseFileNames << " time sync file is " << anchorFilenames << " and ssim is " << maxSSIM << std::endl;
    }

    return syncPairLists;
}


double CameraCameraSync::evaluateImageTimeStampSync(cv::Mat orgImage, cv::Mat dstImage)
{
    //这里采用SSIM结构相似性来作为图像相似性评判
    double C1 = 6.5025, C2 = 58.5225;
    int width = orgImage.cols;
    int height = orgImage.rows;
    
    int width2 = dstImage.cols;
    int height2 = dstImage.rows;

    double mean_x = 0;
    double mean_y = 0;
    double sigma_x = 0;
    double sigma_y = 0;
    double sigma_xy = 0;
    for (int v = 0; v < height; v++)
    {
        for (int u = 0; u < width; u++)
        {
            mean_x += orgImage.at<uchar>(v, u);
            mean_y += dstImage.at<uchar>(v, u);

        }
    }
    mean_x = mean_x / width / height;
    mean_y = mean_y / width / height;
    for (int v = 0; v < height; v++)
    {
        for (int u = 0; u < width; u++)
        {
            sigma_x += (orgImage.at<uchar>(v, u) - mean_x)* (orgImage.at<uchar>(v, u) - mean_x);
            sigma_y += (dstImage.at<uchar>(v, u) - mean_y)* (dstImage.at<uchar>(v, u) - mean_y);
            sigma_xy += std::abs((orgImage.at<uchar>(v, u) - mean_x)* (dstImage.at<uchar>(v, u) - mean_y));
        }
    }
    sigma_x = sigma_x / (width*height - 1);
    sigma_y = sigma_y / (width*height - 1);
    sigma_xy = sigma_xy / (width*height - 1);
    double molecule = (2 * mean_x*mean_y + C1) * (2 * sigma_xy + C2);
    double denominator = (mean_x*mean_x + mean_y * mean_y + C1) * (sigma_x + sigma_y + C2);
    double ssim = molecule / denominator;
    return ssim;
}

void CameraCameraSync::spatialSynchronization(cv::Mat srcImage1, cv::Mat srcImage2)
{
    // 提取特征点    
    //使用SURF算子检测关键点
	int minHessian = 400;//SURF算法中的hessian阈值
    std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;//vector模板类，存放任意类型的动态数组
    cv::Mat descriptors_object, descriptors_scene;
	cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SurfFeatureDetector::create(minHessian);
	
    cv::Ptr <cv::xfeatures2d::SURF> extractor = cv::xfeatures2d::SURF::create(minHessian);
    
	//调用detect函数检测出SURF特征关键点，保存在vector容器中
	detector->detect(srcImage1, keypoints_object);
	detector->detect(srcImage2, keypoints_scene);

    //特征点描述，为下边的特征点匹配做准备  
    cv::Mat matshow1, matshow2, kp1, kp2; 
    extractor->compute(srcImage1, keypoints_object, descriptors_object);
	extractor->compute(srcImage2, keypoints_scene, descriptors_scene);

    //使用FLANN匹配算子进行匹配
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matchePoints;
    matcher.match(descriptors_object, descriptors_scene, matchePoints);

    //最小距离和最大距离
    double max_dist = 0; 
    double min_dist = 100;

	//计算出关键点之间距离的最大值和最小值
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matchePoints[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf(">Max dist 最大距离 : %f \n", max_dist);
	printf(">Min dist 最小距离 : %f \n", min_dist);

	//匹配距离小于3*min_dist的点对
	std::vector< cv::DMatch > goodMatches;
 
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matchePoints[i].distance < 2.5 * min_dist)
		{
			goodMatches.push_back(matchePoints[i]);
		}
	}
    
	//绘制出匹配到的关键点
	cv::Mat imgMatches;
	cv::drawMatches(srcImage1, keypoints_object, srcImage2, keypoints_scene,
		goodMatches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1),
		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
 
	//定义两个局部变量
	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;
 
	//从匹配成功的匹配对中获取关键点
	for (unsigned int i = 0; i < goodMatches.size(); i++)
	{
		obj.push_back(keypoints_object[goodMatches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[goodMatches[i].trainIdx].pt);
	}
 
	cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC);//计算透视变换 
 
	//从待测图片中获取角点
	std::vector<cv::Point2f> obj_corners(4);
	obj_corners[0] = cv::Point(0, 0);
	obj_corners[1] = cv::Point(srcImage1.cols, 0);
	obj_corners[2] = cv::Point(srcImage1.cols, srcImage1.rows);
	obj_corners[3] = cv::Point(0, srcImage1.rows);
	std::vector<cv::Point2f> scene_corners(4);
 
	//进行透视变换
	cv::perspectiveTransform(obj_corners, scene_corners, H);
 
	// //显示最终结果
	// imshow("效果图", imgMatches);
    time_t timep;
    time(&timep);
    
    char name[1024];
    sprintf(name, "效果_%d.jpg", timep);
    
    cv::imwrite(name,imgMatches);

}

