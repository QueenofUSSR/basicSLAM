#pragma once
#include "KeyFrame.hpp"
#include "MapPoint.hpp"
#include <vector>
#include <unordered_map>
#include <opencv2/core.hpp>

class MapManager {
public:
    MapManager();

    void addKeyFrame(const KeyFrame &kf);
    void addMapPoints(const std::vector<MapPoint> &pts);

    const std::vector<KeyFrame>& keyframes() const { return keyframes_; }
    const std::vector<MapPoint>& mappoints() const { return mappoints_; }

    // Find candidate mappoint indices visible from pose (lastR, lastT) and inside image
    std::vector<int> findVisibleCandidates(const cv::Mat &lastR, const cv::Mat &lastT,
                                           double fx, double fy, double cx, double cy,
                                           int imgW, int imgH) const;

    // Triangulate given normalized coordinates between last keyframe and current
    // Returns newly created map points (appended to internal list).
    std::vector<MapPoint> triangulateBetweenLastTwo(const std::vector<cv::Point2f> &pts1n,
                                                    const std::vector<cv::Point2f> &pts2n,
                                                    const std::vector<int> &pts1_kp_idx,
                                                    const std::vector<int> &pts2_kp_idx,
                                                    const KeyFrame &lastKf, const KeyFrame &curKf,
                                                    double fx, double fy, double cx, double cy);

    // Lookup keyframe index by id (-1 if not found)
    int keyframeIndex(int id) const;
    // Lookup mappoint index by id (-1 if not found)
    int mapPointIndex(int id) const;
    // Mutable access if caller legitimately needs to modify keyframes/mappoints.
    // Prefer using the applyOptimized... APIs for controlled updates.
    std::vector<KeyFrame>& keyframesMutable() { return keyframes_; }
    std::vector<MapPoint>& mappointsMutable() { return mappoints_; }

    // Apply optimized pose/point by id (safe writeback APIs for backend)
    void applyOptimizedKeyframePose(int keyframeId, const cv::Mat &R, const cv::Mat &t);
    void applyOptimizedMapPoint(int mappointId, const cv::Point3d &p);

    // Quality management
    void cullBadMapPoints();
    double computeReprojError(const MapPoint &mp, const KeyFrame &kf, 
                             double fx, double fy, double cx, double cy) const;
    void updateMapPointDescriptor(MapPoint &mp);
    
    // Statistics
    int countGoodMapPoints() const;

private:
    std::vector<KeyFrame> keyframes_;
    std::vector<MapPoint> mappoints_;
    std::unordered_map<int,int> id2idx_;
    std::unordered_map<int,int> mpid2idx_;
    int next_mappoint_id_ = 1;
};
