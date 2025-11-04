#pragma once
#include <opencv2/core.hpp>
#include <string>
#include "FeatureExtractor.hpp"
#include "Matcher.hpp"
#include "PoseEstimator.hpp"

class Tracker {
public:
    Tracker();
    // Process a gray image, returns true if a pose was estimated. imgOut contains visualization (matches or keypoints)
    bool processFrame(const cv::Mat &gray, const std::string &imagePath, cv::Mat &imgOut, cv::Mat &R_out, cv::Mat &t_out, std::string &info);
private:
    FeatureExtractor feat_;
    Matcher matcher_;
    PoseEstimator poseEst_;

    cv::Mat prevGray_, prevDesc_;
    std::vector<cv::KeyPoint> prevKp_;
    int frame_id_;
};
