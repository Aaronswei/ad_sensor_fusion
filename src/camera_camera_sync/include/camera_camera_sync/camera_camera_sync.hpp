#ifndef _CAMERA_CAMERA_SYNC_HPP_
#define _CAMERA_CAMERA_SYNC_HPP_

#include <string>
#include <iostream>
#include <vector>
#include <numeric>
#include <time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <cmath>
using namespace std;
using namespace cv;
#define PI 3.14159265
const double UTILS_MATCH_MIN_DIST = 25.0;
const double UTILS_FEATURE_MATCH_RATIO_THRESH = 0.5;

#define PRINTLOG std::cout << __LINE__ << " in " << __FILE__ << std::endl;
struct CamParams
{
    // left: cam1, right: cam1
    Mat K1, K2, D1, D2, R, T;
    Size imgSize;

    CamParams()
    {
        // camera1:intrinsic
        double arrK1[9] = {1.9457000000000000e+03, 0., 8.9667539999999997e+02, 0.,1.9433000000000000e+03, 5.0516239999999999e+02, 0., 0., 1.};
        // camera2:intrinsic
        double arrK2[9] = {1.9492000000000000e+03, 0., 9.2153930000000003e+02, 0.,1.9472000000000000e+03, 5.6912049999999999e+02, 0., 0., 1.};

        double arrR[9] = {1.0,0.,0.,0.,1.0,0.,0.,0.,1.0};
        double arrT[3] = {-1.1,0,0};
        // camera1:distortion coefficient
        double arrD1[5] = {5.7289999999999996e-01, 3.10500e-01,3.50001e-03, 6.8995000005e-04,-3.3500000000000002e-02};
        // camera2:distortion coefficient
        double arrD2[5] = {-5.8879999999999999e-01, 3.0020000000000002e-01,2.0999999999999999e-03, -2.2568999999999999e-04,2.1190000000000001e-01};

        K1 = Mat(3,3, CV_64F, arrK1).clone();
        K2 = Mat(3,3, CV_64F, arrK2).clone();

        R = Mat(3,3, CV_64F,arrR).clone();
        T = Mat(3,1, CV_64F,arrT).clone();

        D1 = Mat(1,5,CV_64F,arrD1).clone();
        D2 = Mat(1,5,CV_64F,arrD2).clone();

        imgSize.width = 1920;
        imgSize.height = 1080;
    }

    ~CamParams(){}

};

void findGoodMatch(std::vector<DMatch> matches, std::vector<DMatch> &good_matches);
void findMatchPoints(const Mat img_left, const Mat img_right, vector<Point2f>& pts1, vector<Point2f>& pts2);

template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum {InputsAtCompileTime = NX, ValuesAtCompileTime = NY};
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

    const int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs;}
    int values() const { return m_values;}
};

class MeanFunctor : public Functor<double> 
{
public:
    MeanFunctor(vector<vector<Point2f> > data):
        Functor<double>(2, data[0].size()), // 1:number of parameters to optimize
        _data(data),
        _cam(CamParams()) {}


    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
    {
        // pitch
        double arrDetlaRPitch[9] = {1., 0., 0.,
                                    0.,cos(x[0]),-sin(x[0]),
                                    0.,sin(x[0]),cos(x[0])};
        Mat detlaRPitch = Mat(3,3,CV_64F, arrDetlaRPitch);
        // roll
        double arrDetlaRRoll[9] = {cos(x[1]),-sin(x[1]), 0.,
                                    sin(x[1]),cos(x[1]),0.,
                                    0.,0.,1.};
        Mat detlaRRoll = Mat(3,3,CV_64F, arrDetlaRRoll);
        
        // add disturb
        double distPitch = 3.14159265/180 * 2;
        double arrDistPitch[9] = {1.,0.,0.,
                                    0.,cos(distPitch), -sin(distPitch),
                                    0.,sin(distPitch), cos(distPitch)};
        
        Mat distRPitch = Mat(3,3,CV_64F, arrDistPitch);

        // 
        Mat optimR = _cam.R * detlaRPitch * detlaRRoll * distRPitch;
        //stereo rectify
        Mat R1, R2, P1, P2, Q;
        stereoRectify(_cam.K1, _cam.D1, _cam.K2, _cam.D2, _cam.imgSize, optimR, _cam.T, R1, R2, P1, P2, Q);

        //  points rectify
        vector<Point2f> rectpts1, rectpts2;
        undistortPoints(_data[0], rectpts1, _cam.K1, _cam.D1, R1, P1);
        undistortPoints(_data[1], rectpts2, _cam.K2, _cam.D2, R2, P2);

        // cost :L1/L2
        for(size_t i =0; i< rectpts1.size(); i++)
        {
            fvec[i] = abs(rectpts1[i].y - rectpts2[i].y);
        }

        return 0;
    }

private:
    vector<vector<Point2f> > _data;
    CamParams _cam;
};


class CameraCameraSync
{
public:
    CameraCameraSync():timeThreshold_(0.1)//100ms
    {

    }

    ~CameraCameraSync() = default;
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
    // pitch and roll between two cameras
    bool synchronizePitchRoll(cv::Mat img_left, cv::Mat img_right);
private:
    void getFiles(std::string path, std::vector<std::string>& files);
    double getbaseTime(std::string pngfilenames, std::string patt);
    
    std::vector<std::string> oriImageLists_;
    std::vector<std::string> dstImageLists_;

    float timeThreshold_;
    std::vector<double> pitch_cache_;
    std::vector<double> roll_cache_;

};

#endif
