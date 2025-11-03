#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

// Function to detect and compute features in an image
inline void detectAndComputeFeatures(const cv::Mat &image,
                                     std::vector<cv::KeyPoint> &keypoints,
                                     cv::Mat &descriptors) {
    // Create ORB detector and descriptor
    auto orb = cv::ORB::create();
    // Detect keypoints
    orb->detect(image, keypoints);
    // Compute descriptors
    orb->compute(image, keypoints, descriptors);
}