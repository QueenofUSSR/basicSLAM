#include "DataLoader.hpp"

DataLoader::DataLoader(const std::string &imageDir)
    : currentIndex(0)
{
    cv::glob(imageDir + "/*", imageFiles, false);
}