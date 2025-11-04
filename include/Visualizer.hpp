#pragma once
#include <opencv2/core.hpp>
#include <vector>

class Visualizer {
public:
    Visualizer(int W=1000, int H=800, double meters_per_pixel=0.02);
    // 更新轨迹（传入 x,z 坐标）
    void addPose(double x, double z);
    // 返回帧绘制（matches 或 keypoints）到窗口
    void showFrame(const cv::Mat &frame);
    // 返回并显示俯视图
    void showTopdown();
    // 保存最终轨迹图像到文件
    bool saveTrajectory(const std::string &path);
private:
    int W_, H_;
    double meters_per_pixel_;
    cv::Size mapSize_;
    cv::Mat map_;
    std::vector<cv::Point2d> traj_; // 存储 (x,z)
    cv::Point worldToPixel(const cv::Point2d &p) const;
};
