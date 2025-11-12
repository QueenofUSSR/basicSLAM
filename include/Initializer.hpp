#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include "KeyFrame.hpp"
#include "MapPoint.hpp"

class Initializer {
public:
    Initializer();
    
    // Attempt initialization with two frames
    // Returns true if initialization successful
    bool initialize(const std::vector<cv::KeyPoint> &kps1,
                   const std::vector<cv::KeyPoint> &kps2,
                   const std::vector<cv::DMatch> &matches,
                   double fx, double fy, double cx, double cy,
                   cv::Mat &R, cv::Mat &t,
                   std::vector<cv::Point3d> &points3D,
                   std::vector<bool> &isTriangulated);
    
    // Check if frames have sufficient parallax for initialization
    static bool checkParallax(const std::vector<cv::KeyPoint> &kps1,
                             const std::vector<cv::KeyPoint> &kps2,
                             const std::vector<cv::DMatch> &matches,
                             double minMedianParallax = 15.0);
    
private:
    // Reconstruct from Homography
    bool reconstructH(const std::vector<cv::Point2f> &pts1,
                     const std::vector<cv::Point2f> &pts2,
                     const cv::Mat &H,
                     double fx, double fy, double cx, double cy,
                     cv::Mat &R, cv::Mat &t,
                     std::vector<cv::Point3d> &points3D,
                     std::vector<bool> &isTriangulated,
                     float &parallax);
    
    // Reconstruct from Fundamental/Essential
    bool reconstructF(const std::vector<cv::Point2f> &pts1,
                     const std::vector<cv::Point2f> &pts2,
                     const cv::Mat &F,
                     double fx, double fy, double cx, double cy,
                     cv::Mat &R, cv::Mat &t,
                     std::vector<cv::Point3d> &points3D,
                     std::vector<bool> &isTriangulated,
                     float &parallax);
    
    // Check reconstructed points quality
    int checkRT(const cv::Mat &R, const cv::Mat &t,
               const std::vector<cv::Point2f> &pts1,
               const std::vector<cv::Point2f> &pts2,
               const std::vector<cv::Point3d> &points3D,
               std::vector<bool> &isGood,
               double fx, double fy, double cx, double cy,
               float &parallax);
    
    // Triangulate points
    void triangulate(const cv::Mat &P1, const cv::Mat &P2,
                    const std::vector<cv::Point2f> &pts1,
                    const std::vector<cv::Point2f> &pts2,
                    std::vector<cv::Point3d> &points3D);
    
    // Decompose Homography
    void decomposeH(const cv::Mat &H, std::vector<cv::Mat> &Rs,
                   std::vector<cv::Mat> &ts, std::vector<cv::Mat> &normals);
    
    // Compute homography score
    float computeScore(const cv::Mat &H21, const cv::Mat &H12,
                      const std::vector<cv::Point2f> &pts1,
                      const std::vector<cv::Point2f> &pts2,
                      std::vector<bool> &inliersH,
                      float sigma = 1.0);
    
    // Compute fundamental score
    float computeScoreF(const cv::Mat &F21,
                       const std::vector<cv::Point2f> &pts1,
                       const std::vector<cv::Point2f> &pts2,
                       std::vector<bool> &inliersF,
                       float sigma = 1.0);
};
