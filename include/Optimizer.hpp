#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "KeyFrame.hpp"
#include "MapPoint.hpp"

// Bundle Adjustment Optimizer using OpenCV-based Levenberg-Marquardt
// Note: For production, should use g2o or Ceres for better performance
class Optimizer {
public:
    Optimizer();
    
    // Local Bundle Adjustment
    // Optimizes a window of recent keyframes and all observed map points
    // fixedKFs: indices of keyframes to keep fixed during optimization
    static void localBundleAdjustment(
        std::vector<KeyFrame> &keyframes,
        std::vector<MapPoint> &mappoints,
        const std::vector<int> &localKfIndices,
        const std::vector<int> &fixedKfIndices,
        double fx, double fy, double cx, double cy,
        int iterations = 10);

#ifdef USE_OPENCV_SFM
    // Alternative BA using OpenCV SFM module when available
    static void localBundleAdjustmentSFM(
        std::vector<KeyFrame> &keyframes,
        std::vector<MapPoint> &mappoints,
        const std::vector<int> &localKfIndices,
        const std::vector<int> &fixedKfIndices,
        double fx, double fy, double cx, double cy,
        int iterations = 10);
#endif
    
    // Pose-only optimization (optimize camera pose given fixed 3D points)
    static bool optimizePose(
        KeyFrame &kf,
        const std::vector<MapPoint> &mappoints,
        const std::vector<int> &matchedMpIndices,
        double fx, double fy, double cx, double cy,
        std::vector<bool> &inliers,
        int iterations = 10);
    
    // Global Bundle Adjustment (expensive, use after loop closure)
    static void globalBundleAdjustment(
        std::vector<KeyFrame> &keyframes,
        std::vector<MapPoint> &mappoints,
        double fx, double fy, double cx, double cy,
        int iterations = 20);
    
private:
    // Compute reprojection error and Jacobian
    static double computeReprojectionError(
        const cv::Point3d &point3D,
        const cv::Mat &R, const cv::Mat &t,
        const cv::Point2f &observed,
        double fx, double fy, double cx, double cy,
        cv::Mat &jacobianPose,
        cv::Mat &jacobianPoint);
    
    // Project 3D point to image
    static cv::Point2f project(
        const cv::Point3d &point3D,
        const cv::Mat &R, const cv::Mat &t,
        double fx, double fy, double cx, double cy);
};
