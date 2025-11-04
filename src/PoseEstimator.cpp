#include "PoseEstimator.hpp"
#include <opencv2/calib3d.hpp>

bool PoseEstimator::estimate(const std::vector<cv::Point2f> &pts1,
                             const std::vector<cv::Point2f> &pts2,
                             double fx, double fy, double cx, double cy,
                             cv::Mat &R, cv::Mat &t, cv::Mat &mask, int &inliers)
{
    if(pts1.size() < 8 || pts2.size() < 8) { inliers = 0; return false; }
    double focal = (fx + fy) * 0.5;
    cv::Point2d pp(cx, cy);
    if(pp.x <= 2.0 && pp.y <= 2.0 && !pts1.empty()){
        // fallback to image center using first point's image size is unknown here; leave as is
    }
    mask.release();
    cv::Mat E = cv::findEssentialMat(pts1, pts2, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
    if(E.empty()) { inliers = 0; return false; }
    inliers = cv::recoverPose(E, pts1, pts2, R, t, focal, pp, mask);
    return true;
}
