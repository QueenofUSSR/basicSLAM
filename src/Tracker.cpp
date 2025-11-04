#include "Tracker.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

Tracker::Tracker()
    : feat_(), matcher_(), poseEst_(), frame_id_(0)
{
}

bool Tracker::processFrame(const cv::Mat &gray, const std::string &imagePath, cv::Mat &imgOut, cv::Mat &R_out, cv::Mat &t_out, std::string &info)
{
    if(gray.empty()) return false;
    // detect
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    feat_.detectAndCompute(gray, kps, desc);

    if(!prevGray_.empty() && !prevDesc_.empty() && !desc.empty()){
        // match
        std::vector<cv::DMatch> goodMatches;
        matcher_.knnMatch(prevDesc_, desc, goodMatches);

        // draw matches for visualization
        cv::drawMatches(prevGray_, prevKp_, gray, kps, goodMatches, imgOut,
                        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // prepare points
        std::vector<cv::Point2f> pts1, pts2;
        for(const auto &m: goodMatches){
            pts1.push_back(prevKp_[m.queryIdx].pt);
            pts2.push_back(kps[m.trainIdx].pt);
        }

        if(pts1.size() >= 8){
            cv::Mat R, t, mask;
            int inliers = 0;
            // Note: we don't have intrinsics here; caller should provide via global or arguments. For now, caller will use PoseEstimator directly if needed.
            // We'll estimate using default focal/pp later (caller will adapt). Return false for now so caller can invoke PoseEstimator separately.
            // But to keep compatibility, leave R_out/t_out empty and set info.
            info = "matches=" + std::to_string(goodMatches.size()) + ", inliers=" + std::to_string(inliers);
            // update prev buffers below
        } else {
            info = "matches=" + std::to_string(goodMatches.size()) + ", inliers=0";
        }
    } else {
        // first frame: draw keypoints
        cv::drawKeypoints(gray, kps, imgOut, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        info = "first_frame";
    }

    // update prev
    prevGray_ = gray.clone();
    prevKp_ = kps;
    prevDesc_ = desc.clone();
    frame_id_++;
    return true;
}
