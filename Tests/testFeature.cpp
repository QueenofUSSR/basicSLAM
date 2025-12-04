// simple_orb_vo.cpp
// 一个极简的单目视觉里程计演示，使用 ORB 特征与匹配。
// - 从帧中提取 ORB 特征
// - 使用汉明距离与比值测试匹配相邻帧
// - 使用 findEssentialMat + recoverPose 估计相对位姿
// - 累积位姿（存在尺度歧义）并显示二维俯视轨迹

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace cv;

#include "VisualOdometry.hpp"

static Point worldToPixel(const Point2d &p, const Size &mapSize, double meters_per_pixel)
{
    Point2d origin(mapSize.width/2.0, mapSize.height/2.0);
    int x = int(origin.x + p.x / meters_per_pixel);
    int y = int(origin.y - p.y / meters_per_pixel); // 翻转 y 轴
    return Point(x,y);
}

int main(int argc, char** argv)
{
    if(argc < 1){
        cout << "Usage: " << argv[0] << " [image_dir] [scale_m_per_unit=1.0]" << endl;
        return 0;
    }

    // string img_dir = "../../datasets/MH01/mav0/cam0/data";
    // string img_dir = "../../datasets/iphone/2025-11-05_170303";
    string img_dir = "../../datasets/vivo/room";
    if(argc >= 2) img_dir = argv[1];

    double scale_m = 0.02;
    if(argc >= 3) scale_m = atof(argv[2]);

    cv::Ptr<cv::Feature2D> feature = cv::ORB::create(2000);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    cv::vo::VisualOdometry vo(feature, matcher);
    return vo.run(img_dir, scale_m);
}
