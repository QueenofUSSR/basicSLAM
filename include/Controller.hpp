#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
// ...

class Controller {
public:
    Controller();
    // 运行完整流水线：imageDir 为图像目录，scale_m 为 recoverPose 单位到米的缩放
    int run(const std::string &imageDir, double scale_m = 1.0);
};