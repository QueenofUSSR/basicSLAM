#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>

namespace cv {
namespace vo {

struct CameraIntrinsics {
    cv::Mat K;      // 3x3 camera matrix
    cv::Mat dist;   // distortion coefficients (optional)
};

struct VOParams {
    int minInitInliers = 80;
    double ransacProb = 0.999;
    double ransacThresh = 1.5; // pixels
    double minParallaxDeg = 1.0;
    double reprojErrThresh = 3.0;
    int pnpMinInliers = 30;
    double pnpReprojErr = 3.0;
    int keyframeInterval = 10; // frames
    double keyframeMinBaseline = 0.05; // relative scale
    double keyframeMinRotationDeg = 5.0;
    int localWindowSize = 7; // for BA
    // Backend parameters
    bool enableBackend = false;
    int baMaxIters = 50;
    // Backend type: 0=g2o (default), 1=opencv_sfm
    int backendType = 0;
};

class VisualOdometry {
public:
    VisualOdometry(const cv::Ptr<cv::Feature2D>& feature,
                   const cv::Ptr<cv::DescriptorMatcher>& matcher,
                   const CameraIntrinsics& intrinsics,
                   const VOParams& params = VOParams());

    // Process a single frame. Returns true if tracking succeeded.
    bool processFrame(const cv::Mat& img, double timestamp);

    // Current camera pose (R|t) as 4x4 SE3 matrix.
    cv::Mat getCurrentPose() const;

    // Full trajectory as vector of 4x4 matrices.
    std::vector<cv::Mat> getTrajectory() const;

    // Backend controls
    void setEnableBackend(bool on);
    void setWindowSize(int N);
    void setBAParams(double reprojThresh, int maxIters);
    void setBackendType(int type);

    // Loop closure & localization stubs (Milestone 4)
    enum Mode { MODE_SLAM, MODE_LOCALIZATION };
    void setMode(Mode m);
    bool saveMap(const std::string& path) const;
    bool loadMap(const std::string& path);
    void enableLoopClosure(bool on);
    void setPlaceRecognizer(const cv::Ptr<cv::Feature2D>& vocabFeature);

private:
    // internal helpers
    bool initializeTwoView(const cv::Mat& img0, const cv::Mat& img1);
    bool trackWithPnP(const cv::Mat& img);
    bool shouldInsertKeyframe() const;
    void triangulateWithLastKeyframe();
    void runLocalBAIfEnabled();

    // state
    cv::Ptr<cv::Feature2D> feature_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    CameraIntrinsics K_;
    VOParams params_;

    bool initialized_ = false;
    bool backendEnabled_ = false;
    int frameCount_ = 0;
    Mode mode_ = MODE_SLAM;
    bool loopClosureEnabled_ = false;

    // cached
    cv::Mat lastImg_;
    cv::Mat currentPoseSE3_; // 4x4
    std::vector<cv::Mat> trajectory_;

    // minimal containers (placeholder, to be integrated with MapManager/KeyFrame)
    std::vector<cv::Point3d> mapPoints_; // simple storage for demo

    // keyframe state (latest KF only)
    std::vector<cv::KeyPoint> kf_keypoints_;
    cv::Mat kf_descriptors_;
    std::vector<int> kf_kp_to_map_; // size == kf_keypoints_.size(), -1 if not mapped
    cv::Mat kfPoseSE3_; // KF pose 4x4

    // current frame cached features
    std::vector<cv::KeyPoint> cur_keypoints_;
    cv::Mat cur_descriptors_;
    int lastPnPInliers_ = 0;
    std::vector<std::pair<int,int>> curFeat_to_map_inliers_; // (cur_kp_idx, map_idx)
    double unmatchedRatio_ = 0.0; // fraction of matched keyframe features without map points (for KF decision)

    // simple history for sliding window
    std::vector<cv::Mat> keyframePoses_; // poses of historical KFs
    std::vector<std::vector<cv::KeyPoint>> keyframeKeypoints_;
    std::vector<cv::Mat> keyframeDescriptors_;
    std::vector<std::vector<int>> keyframeKpToMap_; // per-KF mapping from kp idx to map point idx
};

} // namespace vo
} // namespace cv