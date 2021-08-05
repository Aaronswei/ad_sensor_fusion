#ifndef _CAMERA_CAMERA_SYNC_HPP_
#define _CAMERA_CAMERA_SYNC_HPP_

#include <string>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <time.h>

#define PRINTLOG std::cout << __LINE__ << " in " << __FILE__ << std::endl;

class CameraCameraSync
{
public:
    // 说明：获取图像的时间戳，这里的时间戳就是文件名，所以可以理解为直接获取文件名
    // 然后将获取的文件列表保存在两个队列中，方便后续找到对应的时间戳
    void getImageTimeStamp(std::string oriDirName, std::string dstDirName);

    int getImageNumber();

    // 说明：返回两个图像时间最接近的图像
    std::vector<std::pair<std::string, std::string> > imageTimeStampSyncFuncion();

    // 说明：评估两个图像的时间是否最接近的方法
    // 假设已经完成了时间硬件同步且两者曝光时间、帧率相同，内参一致，那么两个相机帧之间不会相差太多,
    // 如果完全同步，则两者的图像距离最接近，所以采用距离信息进行评价.
    // 假设 队列A中的元素n应该与队列B中的元素n距离最近，仅仅与B中的元素n-1，n+1进行比较，如果相差太多，那么认为时间硬件有问题
    double evaluateImageTimeStampSync(cv::Mat orgImage, cv::Mat dstImage);

    // 空间同步
    void spatialSynchronization(cv::Mat image1, cv::Mat image2);


private:
    std::vector<std::string> oriImageLists_;
    std::vector<std::string> dstImageLists_;

    float timeThreshold_;

    void getFiles(std::string path, std::vector<std::string>& files);
    double getbaseTime(std::string pngfilenames, std::string patt);
};

#endif
