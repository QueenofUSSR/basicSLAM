#pragma once
#include <opencv2/core.hpp>
#include <vector>

class PoseEstimator {
public:
    PoseEstimator() = default;
    // Estimate relative pose from matched normalized image points. Returns true if pose recovered.
    bool estimate(const std::vector<cv::Point2f> &pts1,
                  const std::vector<cv::Point2f> &pts2,
                  double fx, double fy, double cx, double cy,
                  cv::Mat &R, cv::Mat &t, cv::Mat &mask, int &inliers);
};
