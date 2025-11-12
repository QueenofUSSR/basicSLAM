#pragma once
#include <opencv2/core.hpp>
#include <vector>

struct MapPoint {
    int id = -1; // unique id for id-based lookups
    cv::Point3d p; // 3D position in world frame
    std::vector<std::pair<int,int>> observations; // pairs of (keyframe id, keypoint idx)
    
    // Quality management fields
    cv::Mat descriptor;           // Representative descriptor for matching
    int nObs = 0;                 // Number of observations
    bool isBad = false;           // Flag for bad points to be culled
    double minDistance = 0.0;     // Min viewing distance
    double maxDistance = 0.0;     // Max viewing distance
    
    // Statistics
    int nFound = 0;               // Number of times found in tracking
    int nVisible = 0;             // Number of times visible
    
    // Constructor
    MapPoint() = default;
    MapPoint(const cv::Point3d& pos) : p(pos) {}
    
    // Helper: compute found ratio
    float getFoundRatio() const {
        return nVisible > 0 ? static_cast<float>(nFound) / nVisible : 0.0f;
    }
};
