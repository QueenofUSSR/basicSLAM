#pragma once
#include "MapManager.hpp"
#include "DataLoader.hpp"
#include <opencv2/core.hpp>
#include <vector>

class Localizer {
public:
    Localizer(float ratio = 0.7f);

    // Try to solve PnP against map points. Returns true if solved and fills R_out/t_out and inliers.
    // Provide camera intrinsics and image dimensions explicitly (DataLoader doesn't expose width/height).
    // Enhanced tryPnP with optional diagnostics output (frame id, image and output directory)
    // out_preMatches and out_postMatches can be nullptr. If outDir provided, diagnostics images/logs will be saved there.

    bool tryPnP(const MapManager &map, const cv::Mat &desc, const std::vector<cv::KeyPoint> &kps,
                double fx, double fy, double cx, double cy, int imgW, int imgH,
                int min_inliers,
                cv::Mat &R_out, cv::Mat &t_out, int &inliers_out,
                int frame_id = -1, const cv::Mat *image = nullptr, const std::string &outDir = "",
                int *out_preMatches = nullptr, int *out_postMatches = nullptr, double *out_meanReproj = nullptr) const;

    float ratio() const { return ratio_; }
private:
    float ratio_ = 0.7f;
};
