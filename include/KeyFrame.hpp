#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

struct KeyFrame {
    int id = -1;
    cv::Mat image; // optional
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    // pose: rotation and translation to map coordinates
    // X_world = R * X_cam + t
    cv::Mat R_w = cv::Mat::eye(3,3,CV_64F);
    cv::Mat t_w = cv::Mat::zeros(3,1,CV_64F);
};
