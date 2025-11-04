#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

class Matcher {
public:
    explicit Matcher(float ratio = 0.75f);
    // Match descriptors from prev -> curr, return good matches (queryIdx refers to prev, trainIdx to curr)
    void knnMatch(const cv::Mat &desc1, const cv::Mat &desc2, std::vector<cv::DMatch> &goodMatches);
private:
    float ratio_;
    cv::BFMatcher bf_;
};
