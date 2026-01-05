// Thin wrapper around OpenCV's cv::vo::VisualOdometry API to validate
// the locally installed opencv_contrib slam module.

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/slam.hpp>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if(argc < 1){
        std::cout << "Usage: " << argv[0] << " [image_dir] [scale_m=0.02]" << std::endl;
        return 0;
    }

    // std::string imgDir = "../../datasets/iphone/2025-11-05_170303";
    std::string imgDir = "../../datasets/EuRoC/MH01/mav0/cam0/data";
    // std::string imgDir = "../../datasets/vivo/room2";
    if(argc >= 2) imgDir = argv[1];
    double scale_m = (argc >= 3) ? std::atof(argv[2]) : 0.01;

    cv::Ptr<cv::Feature2D> feature = cv::ORB::create(2000);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    cv::vo::VisualOdometry vo(feature, matcher);
    cv::vo::VisualOdometryOptions options;
    // 可以根据需要设置 options.* 字段，这里使用默认参数

    std::cout << "Running OpenCV VisualOdometry on " << imgDir << std::endl;
    int ret = vo.run(imgDir, scale_m, options);
    std::cout << "VisualOdometry finished with code " << ret << std::endl;
    return ret;
}
