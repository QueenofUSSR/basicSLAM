#pragma once
#include <opencv2/core.hpp>

class DataLoader {
public:
    DataLoader(const std::string &imageDir);
    bool getNextImage(cv::Mat &image, std::string &imagePath);
private:
    std::vector<std::string> imageFiles;
    size_t currentIndex;
};