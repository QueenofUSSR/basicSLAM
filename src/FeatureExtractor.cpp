#include "FeatureExtractor.hpp"

FeatureExtractor::FeatureExtractor(int nfeatures)
{
    orb_ = cv::ORB::create(nfeatures);
}

void FeatureExtractor::detectAndCompute(const cv::Mat &image, std::vector<cv::KeyPoint> &kps, cv::Mat &desc)
{
    if(image.empty()) return;
    orb_->detectAndCompute(image, cv::noArray(), kps, desc);
}
