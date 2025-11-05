#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

class FeatureExtractor {
public:
    explicit FeatureExtractor(int nfeatures = 2000);
    // detectAndCompute: detect keypoints, apply ANMS to pick up to nfeatures, then compute descriptors
    void detectAndCompute(const cv::Mat &image, std::vector<cv::KeyPoint> &kps, cv::Mat &desc);
private:
    cv::Ptr<cv::ORB> orb_;
    int nfeatures_;
};
