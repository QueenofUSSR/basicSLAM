#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

// ControllerOptions holds tunable thresholds used by the controller
struct ControllerOptions {
    int min_matches = 15;       // minimum descriptor matches to attempt pose
    int min_inliers = 4;        // absolute inlier guard
    double min_inlier_ratio = 0.1; // relative inlier ratio threshold
    double diff_zero_thresh = 2.0; // mean gray diff to consider frames identical
    double flow_zero_thresh = 0.3; // median flow to consider frames identical (px)
    double min_translation_norm = 1e-4; // small translation magnitude
    double min_rotation_rad = 0.5 * CV_PI / 180.0; // 0.5 degree in radians
    int max_matches_keep = 500; // cap after flow-weighted selection
    double flow_weight_lambda = 5.0; // weight applied to normalized flow in scoring
};

class Controller {
public:
    Controller();
    // 运行完整流水线：imageDir 为图像目录，scale_m 为 recoverPose 单位到米的缩放
    // options: tunable thresholds (optional)
    int run(const std::string &imageDir, double scale_m = 1.0, const ControllerOptions &options = ControllerOptions());
};