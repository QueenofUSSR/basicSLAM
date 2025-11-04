#include "Controller.hpp"

#include "DataLoader.hpp"
#include "FeatureExtractor.hpp"
#include "Matcher.hpp"
#include "PoseEstimator.hpp"
#include "Visualizer.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

Controller::Controller() {
    // empty
}

int Controller::run(const std::string &imageDir, double scale_m){
    DataLoader loader(imageDir);
    if(loader.size() == 0){
        std::cerr << "Controller: no images found in " << imageDir << std::endl;
        return -1;
    }

    FeatureExtractor feat(2000);
    Matcher matcher(0.75f);
    PoseEstimator poseEst;
    Visualizer vis;

    cv::Mat R_g = cv::Mat::eye(3,3,CV_64F);
    cv::Mat t_g = cv::Mat::zeros(3,1,CV_64F);

    cv::Mat frame;
    std::string imgPath;
    int frame_id = 0;
    while(loader.getNextImage(frame, imgPath)){
        cv::Mat gray = frame;
        if(gray.channels() > 1) cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        feat.detectAndCompute(gray, kps, desc);

        static std::vector<cv::KeyPoint> prevKp;
        static cv::Mat prevGray, prevDesc;

        if(!prevGray.empty() && !prevDesc.empty() && !desc.empty()){
            std::vector<cv::DMatch> goodMatches;
            matcher.knnMatch(prevDesc, desc, goodMatches);

            cv::Mat imgMatches;
            cv::drawMatches(prevGray, prevKp, gray, kps, goodMatches, imgMatches,
                            cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            std::vector<cv::Point2f> pts1, pts2;
            pts1.reserve(goodMatches.size()); pts2.reserve(goodMatches.size());
            for(const auto &m: goodMatches){
                pts1.push_back(prevKp[m.queryIdx].pt);
                pts2.push_back(kps[m.trainIdx].pt);
            }

            if(pts1.size() >= 8){
                cv::Mat R, t, mask; int inliers = 0;
                if(poseEst.estimate(pts1, pts2, loader.fx(), loader.fy(), loader.cx(), loader.cy(), R, t, mask, inliers)){
                    cv::Mat t_d; t.convertTo(t_d, CV_64F);
                    cv::Mat t_scaled = t_d * scale_m;
                    cv::Mat R_d; R.convertTo(R_d, CV_64F);
                    t_g = t_g + R_g * t_scaled;
                    R_g = R_g * R_d;
                    double x = t_g.at<double>(0);
                    double z = t_g.at<double>(2);
                    vis.addPose(x,z);
                    // annotate and show
                    std::string info = std::string("Frame ")+std::to_string(frame_id)+" " + imgPath + " matches=" + std::to_string(goodMatches.size()) + " inliers=" + std::to_string(inliers);
                    cv::putText(imgMatches, info, cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
                    vis.showFrame(imgMatches);
                } else {
                    vis.showFrame(gray);
                }
            } else {
                vis.showFrame(gray);
            }
        } else {
            cv::Mat visFrame; cv::drawKeypoints(gray, kps, visFrame, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            vis.showFrame(visFrame);
        }

        vis.showTopdown();
        // update prev
        prevGray = gray.clone(); prevKp = kps; prevDesc = desc.clone();
        frame_id++;
        char key = (char)cv::waitKey(1);
        if(key == 27) break;
    }

    vis.saveTrajectory("trajectory.png");
    return 0;
}