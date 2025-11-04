#include "Matcher.hpp"

Matcher::Matcher(float ratio)
    : ratio_(ratio), bf_(cv::NORM_HAMMING)
{
}

void Matcher::knnMatch(const cv::Mat &desc1, const cv::Mat &desc2, std::vector<cv::DMatch> &goodMatches)
{
    goodMatches.clear();
    if(desc1.empty() || desc2.empty()) return;
    std::vector<std::vector<cv::DMatch>> knn;
    bf_.knnMatch(desc1, desc2, knn, 2);
    for(size_t i=0;i<knn.size();++i){
        if(knn[i].size() < 2) continue;
        const cv::DMatch &m1 = knn[i][0];
        const cv::DMatch &m2 = knn[i][1];
        if(m1.distance < ratio_ * m2.distance) goodMatches.push_back(m1);
    }
}
