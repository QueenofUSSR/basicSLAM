#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

class Matcher {
public:
    explicit Matcher(float ratio = 0.75f);
    // Match descriptors from prev -> curr, return good matches (queryIdx refers to prev, trainIdx to curr)
    // Basic knn ratio test (backwards-compatible)
    void knnMatch(const cv::Mat &desc1, const cv::Mat &desc2, std::vector<cv::DMatch> &goodMatches);

    // Stronger match: knn + ratio test + optional mutual cross-check + optional spatial bucketing.
    // desc1/desc2 are descriptors for prev/curr frames.
    // kps1/kps2 are the corresponding keypoints (needed for spatial bucketing).
    // imgCols/imgRows are used to size the bucketing grid. Defaults provide conservative values.
    // maxTotal: if >0, final matches will be truncated to this count (keep smallest distances).
    // enableMutual: enable/disable mutual cross-check. enableBucketing: enable/disable grid bucketing.
    void match(const cv::Mat &desc1, const cv::Mat &desc2,
               const std::vector<cv::KeyPoint> &kps1, const std::vector<cv::KeyPoint> &kps2,
               std::vector<cv::DMatch> &goodMatches,
               int imgCols = 640, int imgRows = 480,
               int bucketRows = 8, int bucketCols = 8, int topKPerBucket = 4,
               int maxTotal = 0, bool enableMutual = true, bool enableBucketing = true);
private:
    float ratio_;
    cv::BFMatcher bf_;
};
