#include "Initializer.hpp"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <algorithm>

Initializer::Initializer() {}

bool Initializer::initialize(const std::vector<cv::KeyPoint> &kps1,
                             const std::vector<cv::KeyPoint> &kps2,
                             const std::vector<cv::DMatch> &matches,
                             double fx, double fy, double cx, double cy,
                             cv::Mat &R, cv::Mat &t,
                             std::vector<cv::Point3d> &points3D,
                             std::vector<bool> &isTriangulated) {
    
    if(matches.size() < 50) {
        std::cout << "Initializer: too few matches (" << matches.size() << ")" << std::endl;
        return false;
    }
    
    // Extract matched points
    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(matches.size());
    pts2.reserve(matches.size());
    
    for(const auto &m : matches) {
        pts1.push_back(kps1[m.queryIdx].pt);
        pts2.push_back(kps2[m.trainIdx].pt);
    }
    
    // Estimate both Homography and Fundamental
    std::vector<uchar> inliersH, inliersF;
    cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, inliersH, 2000, 0.999);
    cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, 0.999, inliersF);
    
    if(H.empty() || F.empty()) {
        std::cout << "Initializer: failed to compute H or F" << std::endl;
        return false;
    }
    
    // Compute scores
    std::vector<bool> inlH, inlF;
    float scoreH = computeScore(H, H.inv(), pts1, pts2, inlH, 1.0);
    float scoreF = computeScoreF(F, pts1, pts2, inlF, 1.0);
    
    float ratio = scoreH / (scoreH + scoreF);
    
    std::cout << "Initializer: H score=" << scoreH << " F score=" << scoreF 
              << " ratio=" << ratio << std::endl;
    
    // Decide between H and F
    // If ratio > 0.45, scene is likely planar, use H
    // Otherwise use F for general scenes
    bool useH = (ratio > 0.45);
    
    cv::Mat R_out, t_out;
    std::vector<cv::Point3d> pts3D;
    std::vector<bool> isTri;
    float parallax = 0.0f;
    
    bool success = false;
    if(useH) {
        std::cout << "Initializer: using Homography" << std::endl;
        success = reconstructH(pts1, pts2, H, fx, fy, cx, cy, R_out, t_out, pts3D, isTri, parallax);
    } else {
        std::cout << "Initializer: using Fundamental" << std::endl;
        success = reconstructF(pts1, pts2, F, fx, fy, cx, cy, R_out, t_out, pts3D, isTri, parallax);
    }
    
    if(!success) {
        std::cout << "Initializer: reconstruction failed" << std::endl;
        return false;
    }
    
    // Count good triangulated points
    int goodCount = 0;
    for(bool b : isTri) if(b) goodCount++;
    
    std::cout << "Initializer: triangulated " << goodCount << "/" << pts3D.size() 
              << " points, parallax=" << parallax << std::endl;
    
    // Check quality: need enough good points
    if(goodCount < 50) {
        std::cout << "Initializer: too few good points (" << goodCount << ")" << std::endl;
        return false;
    }
    
    // Check parallax
    if(parallax < 1.0f) {
        std::cout << "Initializer: insufficient parallax (" << parallax << ")" << std::endl;
        return false;
    }
    
    // Success
    R = R_out;
    t = t_out;
    points3D = pts3D;
    isTriangulated = isTri;
    
    std::cout << "Initializer: SUCCESS!" << std::endl;
    return true;
}

bool Initializer::checkParallax(const std::vector<cv::KeyPoint> &kps1,
                                const std::vector<cv::KeyPoint> &kps2,
                                const std::vector<cv::DMatch> &matches,
                                double minMedianParallax) {
    if(matches.empty()) return false;
    
    std::vector<double> parallaxes;
    parallaxes.reserve(matches.size());
    
    for(const auto &m : matches) {
        cv::Point2f p1 = kps1[m.queryIdx].pt;
        cv::Point2f p2 = kps2[m.trainIdx].pt;
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        parallaxes.push_back(std::sqrt(dx*dx + dy*dy));
    }
    
    std::sort(parallaxes.begin(), parallaxes.end());
    double median = parallaxes[parallaxes.size()/2];
    
    return median >= minMedianParallax;
}

bool Initializer::reconstructF(const std::vector<cv::Point2f> &pts1,
                               const std::vector<cv::Point2f> &pts2,
                               const cv::Mat &F,
                               double fx, double fy, double cx, double cy,
                               cv::Mat &R, cv::Mat &t,
                               std::vector<cv::Point3d> &points3D,
                               std::vector<bool> &isTriangulated,
                               float &parallax) {
    
    // Compute Essential matrix from F
    cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat E = K.t() * F * K;
    
    // Recover pose from E
    cv::Mat mask;
    int inliers = cv::recoverPose(E, pts1, pts2, K, R, t, mask);
    
    if(inliers < 30) return false;
    
    // Triangulate
    points3D.resize(pts1.size());
    isTriangulated.resize(pts1.size(), false);
    
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat P2(3, 4, CV_64F);
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) P2.at<double>(i,j) = R.at<double>(i,j);
        P2.at<double>(i, 3) = t.at<double>(i, 0);
    }
    
    triangulate(P1, P2, pts1, pts2, points3D);
    
    // Check quality
    std::vector<bool> isGood;
    int nGood = checkRT(R, t, pts1, pts2, points3D, isGood, fx, fy, cx, cy, parallax);
    
    isTriangulated = isGood;
    return nGood >= 30;
}

bool Initializer::reconstructH(const std::vector<cv::Point2f> &pts1,
                               const std::vector<cv::Point2f> &pts2,
                               const cv::Mat &H,
                               double fx, double fy, double cx, double cy,
                               cv::Mat &R, cv::Mat &t,
                               std::vector<cv::Point3d> &points3D,
                               std::vector<bool> &isTriangulated,
                               float &parallax) {
    
    // Decompose H to get multiple solutions
    std::vector<cv::Mat> Rs, ts, normals;
    decomposeH(H, Rs, ts, normals);
    
    // Try all solutions and pick best
    int bestGood = 0;
    int bestIdx = -1;
    std::vector<std::vector<cv::Point3d>> allPoints(Rs.size());
    std::vector<std::vector<bool>> allGood(Rs.size());
    std::vector<float> allParallax(Rs.size());
    
    for(size_t i = 0; i < Rs.size(); ++i) {
        std::vector<cv::Point3d> pts3D;
        std::vector<bool> isGood;
        float par = 0.0f;
        
        cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
        cv::Mat P2(3, 4, CV_64F);
        for(int r = 0; r < 3; ++r) {
            for(int c = 0; c < 3; ++c) P2.at<double>(r,c) = Rs[i].at<double>(r,c);
            P2.at<double>(r, 3) = ts[i].at<double>(r, 0);
        }
        
        triangulate(P1, P2, pts1, pts2, pts3D);
        int nGood = checkRT(Rs[i], ts[i], pts1, pts2, pts3D, isGood, fx, fy, cx, cy, par);
        
        allPoints[i] = pts3D;
        allGood[i] = isGood;
        allParallax[i] = par;
        
        if(nGood > bestGood) {
            bestGood = nGood;
            bestIdx = i;
        }
    }
    
    if(bestIdx < 0 || bestGood < 30) return false;
    
    R = Rs[bestIdx];
    t = ts[bestIdx];
    points3D = allPoints[bestIdx];
    isTriangulated = allGood[bestIdx];
    parallax = allParallax[bestIdx];
    
    return true;
}

int Initializer::checkRT(const cv::Mat &R, const cv::Mat &t,
                         const std::vector<cv::Point2f> &pts1,
                         const std::vector<cv::Point2f> &pts2,
                         const std::vector<cv::Point3d> &points3D,
                         std::vector<bool> &isGood,
                         double fx, double fy, double cx, double cy,
                         float &parallax) {
    
    isGood.resize(points3D.size(), false);
    
    cv::Mat R1 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t1 = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat R2 = R;
    cv::Mat t2 = t;
    
    int nGood = 0;
    std::vector<float> cosParallaxes;
    cosParallaxes.reserve(points3D.size());
    
    for(size_t i = 0; i < points3D.size(); ++i) {
        const cv::Point3d &p3d = points3D[i];
        
        // Check depth in first camera
        cv::Mat p3dMat = (cv::Mat_<double>(3,1) << p3d.x, p3d.y, p3d.z);
        cv::Mat p1 = R1 * p3dMat + t1;
        double z1 = p1.at<double>(2, 0);
        
        if(z1 <= 0) continue;
        
        // Check depth in second camera
        cv::Mat p2 = R2 * p3dMat + t2;
        double z2 = p2.at<double>(2, 0);
        
        if(z2 <= 0) continue;
        
        // Check reprojection error in first camera
        double u1 = fx * p1.at<double>(0,0)/z1 + cx;
        double v1 = fy * p1.at<double>(1,0)/z1 + cy;
        double err1 = std::sqrt(std::pow(u1 - pts1[i].x, 2) + std::pow(v1 - pts1[i].y, 2));
        
        if(err1 > 4.0) continue;
        
        // Check reprojection error in second camera
        double u2 = fx * p2.at<double>(0,0)/z2 + cx;
        double v2 = fy * p2.at<double>(1,0)/z2 + cy;
        double err2 = std::sqrt(std::pow(u2 - pts2[i].x, 2) + std::pow(v2 - pts2[i].y, 2));
        
        if(err2 > 4.0) continue;
        
        // Check parallax
        cv::Mat normal1 = p3dMat / cv::norm(p3dMat);
        cv::Mat normal2 = (p3dMat - t2) / cv::norm(p3dMat - t2);
        double cosParallax = normal1.dot(normal2);
        
        cosParallaxes.push_back(cosParallax);
        
        isGood[i] = true;
        nGood++;
    }
    
    if(!cosParallaxes.empty()) {
        std::sort(cosParallaxes.begin(), cosParallaxes.end());
        float medianCos = cosParallaxes[cosParallaxes.size()/2];
        parallax = std::acos(medianCos) * 180.0 / CV_PI;
    }
    
    return nGood;
}

void Initializer::triangulate(const cv::Mat &P1, const cv::Mat &P2,
                              const std::vector<cv::Point2f> &pts1,
                              const std::vector<cv::Point2f> &pts2,
                              std::vector<cv::Point3d> &points3D) {
    
    points3D.resize(pts1.size());
    
    cv::Mat pts4D;
    cv::triangulatePoints(P1, P2, pts1, pts2, pts4D);
    
    for(int i = 0; i < pts4D.cols; ++i) {
        cv::Mat x = pts4D.col(i);
        x /= x.at<float>(3, 0);
        points3D[i] = cv::Point3d(x.at<float>(0,0), x.at<float>(1,0), x.at<float>(2,0));
    }
}

void Initializer::decomposeH(const cv::Mat &H, std::vector<cv::Mat> &Rs,
                             std::vector<cv::Mat> &ts, std::vector<cv::Mat> &normals) {
    
    cv::Mat H_normalized = H / H.at<double>(2,2);
    
    std::vector<cv::Mat> rotations, translations, normalsOut;
    int solutions = cv::decomposeHomographyMat(H_normalized, cv::Mat::eye(3,3,CV_64F),
                                               rotations, translations, normalsOut);
    
    Rs = rotations;
    ts = translations;
    normals = normalsOut;
}

float Initializer::computeScore(const cv::Mat &H21, const cv::Mat &H12,
                               const std::vector<cv::Point2f> &pts1,
                               const std::vector<cv::Point2f> &pts2,
                               std::vector<bool> &inliersH,
                               float sigma) {
    
    const float th = 5.991 * sigma;
    inliersH.resize(pts1.size(), false);
    
    float score = 0.0f;
    const float thSq = th * th;
    
    for(size_t i = 0; i < pts1.size(); ++i) {
        // Forward error
        cv::Mat p1 = (cv::Mat_<double>(3,1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat p2pred = H21 * p1;
        p2pred /= p2pred.at<double>(2,0);
        
        float dx = pts2[i].x - p2pred.at<double>(0,0);
        float dy = pts2[i].y - p2pred.at<double>(1,0);
        float errSq = dx*dx + dy*dy;
        
        if(errSq < thSq) {
            score += thSq - errSq;
        }
        
        // Backward error
        cv::Mat p2 = (cv::Mat_<double>(3,1) << pts2[i].x, pts2[i].y, 1.0);
        cv::Mat p1pred = H12 * p2;
        p1pred /= p1pred.at<double>(2,0);
        
        dx = pts1[i].x - p1pred.at<double>(0,0);
        dy = pts1[i].y - p1pred.at<double>(1,0);
        errSq = dx*dx + dy*dy;
        
        if(errSq < thSq) {
            score += thSq - errSq;
            inliersH[i] = true;
        }
    }
    
    return score;
}

float Initializer::computeScoreF(const cv::Mat &F21,
                                 const std::vector<cv::Point2f> &pts1,
                                 const std::vector<cv::Point2f> &pts2,
                                 std::vector<bool> &inliersF,
                                 float sigma) {
    
    const float th = 3.841 * sigma;
    const float thSq = th * th;
    
    inliersF.resize(pts1.size(), false);
    float score = 0.0f;
    
    for(size_t i = 0; i < pts1.size(); ++i) {
        cv::Mat p1 = (cv::Mat_<double>(3,1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat p2 = (cv::Mat_<double>(3,1) << pts2[i].x, pts2[i].y, 1.0);
        
        // Epipolar line in second image
        cv::Mat l2 = F21 * p1;
        float a2 = l2.at<double>(0,0);
        float b2 = l2.at<double>(1,0);
        float c2 = l2.at<double>(2,0);
        
        float num2 = a2*pts2[i].x + b2*pts2[i].y + c2;
        float den2 = a2*a2 + b2*b2;
        float distSq2 = (num2*num2) / den2;
        
        if(distSq2 < thSq) {
            score += thSq - distSq2;
            inliersF[i] = true;
        }
    }
    
    return score;
}
