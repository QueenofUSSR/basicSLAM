#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

class FeatureExtractor {
public:
    explicit FeatureExtractor(int nfeatures = 2000);
    // detectAndCompute: detect keypoints and compute descriptors.
    // If previous-frame data (prevGray, prevKp) is provided, a flow-aware grid allocation
    // will be used (score = response * (1 + flow_lambda * normalized_flow)). Otherwise a
    // simpler ANMS selection is used. The prev arguments have defaults so this function
    // replaces the two previous overloads.
    void detectAndCompute(const cv::Mat &image, std::vector<cv::KeyPoint> &kps, cv::Mat &desc,
                          const cv::Mat &prevGray = cv::Mat(), const std::vector<cv::KeyPoint> &prevKp = std::vector<cv::KeyPoint>(),
                          double flow_lambda = 5.0);
private:
    cv::Ptr<cv::ORB> orb_;
    int nfeatures_;
};
