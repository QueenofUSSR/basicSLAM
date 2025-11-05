#include "Controller.hpp"

#include "DataLoader.hpp"
#include "FeatureExtractor.hpp"
#include "Matcher.hpp"
#include "PoseEstimator.hpp"
#include "Visualizer.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>

Controller::Controller() {
    // empty
}

int Controller::run(const std::string &imageDir, double scale_m){
    DataLoader loader(imageDir);
    std::cout << "Controller: loaded " << loader.size() << " images from " << imageDir << std::endl;
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
            // use stronger matching: ratio + mutual cross-check + spatial bucketing
            matcher.match(prevDesc, desc, prevKp, kps, goodMatches, gray.cols, gray.rows, 8, 8, 4);

            cv::Mat imgMatches;
            cv::drawMatches(prevGray, prevKp, gray, kps, goodMatches, imgMatches,
                            cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            std::vector<cv::Point2f> pts1, pts2;
            pts1.reserve(goodMatches.size()); pts2.reserve(goodMatches.size());
            for(const auto &m: goodMatches){
                pts1.push_back(prevKp[m.queryIdx].pt);
                pts2.push_back(kps[m.trainIdx].pt);
            }

            // quick frame-diff to detect near-static frames
            double meanDiff = 0.0;
            if(!prevGray.empty()){
                cv::Mat diff; cv::absdiff(gray, prevGray, diff);
                meanDiff = cv::mean(diff)[0];
            }

            // compute median displacement (flow) between matched keypoints
            double median_flow = 0.0;
            if(!pts1.empty()){
                std::vector<double> dists; dists.reserve(pts1.size());
                for(size_t i=0;i<pts1.size();++i){
                    double dx = pts2[i].x - pts1[i].x;
                    double dy = pts2[i].y - pts1[i].y;
                    dists.push_back(std::sqrt(dx*dx + dy*dy));
                }
                size_t mid = dists.size()/2;
                std::nth_element(dists.begin(), dists.begin()+mid, dists.end());
                median_flow = dists[mid];
            }

            if(pts1.size() >= 8){
                cv::Mat R, t, mask; int inliers = 0;
                bool ok = poseEst.estimate(pts1, pts2, loader.fx(), loader.fy(), loader.cx(), loader.cy(), R, t, mask, inliers);

                int matchCount = static_cast<int>(goodMatches.size());
                double inlierRatio = matchCount > 0 ? double(inliers) / double(matchCount) : 0.0;

                // thresholds (tunable) -- relaxed and add absolute inlier guard
                const int MIN_MATCHES = 15;           // require at least this many matches (relative)
                const int MIN_INLIERS = 4;             // OR accept if at least this many absolute inliers
                double t_norm = 0.0, rot_angle = 0.0;
                if(ok){
                    cv::Mat t_d; t.convertTo(t_d, CV_64F);
                    t_norm = cv::norm(t_d);
                    cv::Mat R_d; R.convertTo(R_d, CV_64F);
                    double trace = R_d.at<double>(0,0) + R_d.at<double>(1,1) + R_d.at<double>(2,2);
                    double cos_angle = std::min(1.0, std::max(-1.0, (trace - 1.0) * 0.5));
                    rot_angle = std::acos(cos_angle);
                }

                // Print per-frame diagnostics
                std::cout << "F" << frame_id << " diff=" << meanDiff << " median_flow=" << median_flow
                          << " matches=" << matchCount << " inliers=" << inliers << " inlierRatio=" << inlierRatio
                          << " t_norm=" << t_norm << " rot_rad=" << rot_angle << std::endl;

                // decide whether to integrate
                // Prefer geometry-based decision (absolute inliers OR matchCount + ratio). Use image-diff/flow
                // only to skip when geometry is weak or motion truly negligible.
                bool integrate = true;
                if(!ok){
                    integrate = false;
                    std::cout << "  -> pose estimation failed, skipping integration." << std::endl;
                } else if(inliers < MIN_INLIERS || matchCount < MIN_MATCHES){
                    // Not enough geometric support -> skip (unless absolute inliers pass)
                    integrate = false;
                    std::cout << "  -> insufficient matches/inliers (by both absolute and relative metrics), skipping integration." << std::endl;
                } else {
                    // We have sufficient geometric support. Only skip if motion is truly negligible
                    // (both translation and rotation tiny) AND the image/flow indicate near-identical frames.
                    const double MIN_TRANSLATION_NORM = 1e-4;
                    const double MIN_ROTATION_RAD = (0.5 * CV_PI / 180.0); // 0.5 degree
                    const double DIFF_ZERO_THRESH = 2.0;   // nearly identical image
                    const double FLOW_ZERO_THRESH = 0.3;   // nearly zero flow in pixels

                    if(t_norm < MIN_TRANSLATION_NORM && std::abs(rot_angle) < MIN_ROTATION_RAD
                       && meanDiff < DIFF_ZERO_THRESH && median_flow < FLOW_ZERO_THRESH){
                        integrate = false; // truly static
                        std::cout << "  -> negligible motion and near-identical frames, skipping integration." << std::endl;
                    }
                }
                if (inliers >= MIN_INLIERS || (inliers >= 2 && matchCount > 50 && median_flow > 2.0)) {
                    integrate = true;
                }

                // integrate transform if allowed
                if(integrate){
                    cv::Mat t_d; t.convertTo(t_d, CV_64F);
                    cv::Mat t_scaled = t_d * scale_m;
                    cv::Mat R_d; R.convertTo(R_d, CV_64F);
                    t_g = t_g + R_g * t_scaled;
                    R_g = R_g * R_d;
                    double x = t_g.at<double>(0);
                    double z = t_g.at<double>(2);
                    vis.addPose(x,-z);
                }

                // Always show a single image; if we have matches, draw small boxes around matched keypoints
                cv::Mat visImg;
                if(frame.channels() > 1) visImg = frame.clone();
                else cv::cvtColor(gray, visImg, cv::COLOR_GRAY2BGR);
                std::string info = std::string("Frame ") + std::to_string(frame_id) + " matches=" + std::to_string(matchCount) + " inliers=" + std::to_string(inliers);

                if(!goodMatches.empty()){
                    for(size_t mi=0; mi<goodMatches.size(); ++mi){
                        // prefer refined pts2 if available, otherwise use keypoint location
                        cv::Point2f p2;
                        if(mi < pts2.size()) p2 = pts2[mi];
                        else p2 = kps[goodMatches[mi].trainIdx].pt;

                        // determine inlier status from mask (robust to mask shape)
                        bool isInlier = false;
                        if(!mask.empty()){
                            if(mask.rows == static_cast<int>(goodMatches.size())){
                                isInlier = mask.at<uchar>(static_cast<int>(mi), 0) != 0;
                            } else if(mask.cols == static_cast<int>(goodMatches.size())){
                                isInlier = mask.at<uchar>(0, static_cast<int>(mi)) != 0;
                            }
                        }
                        cv::Scalar col = isInlier ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255);
                        cv::Point ip(cvRound(p2.x), cvRound(p2.y));
                        cv::Rect r(ip - cv::Point(4,4), cv::Size(8,8));
                        cv::rectangle(visImg, r, col, 2, cv::LINE_AA);
                    }
                }
                cv::putText(visImg, info, cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
                vis.showFrame(visImg);

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

    // save trajectory with timestamp into result/ folder
    try{
        std::filesystem::path outDir("../../result");
        if(!std::filesystem::exists(outDir)) std::filesystem::create_directories(outDir);
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&t);
        std::ostringstream ss;
        ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        std::string fname = std::string("trajectory_") + ss.str() + std::string(".png");
        std::filesystem::path outPath = outDir / fname;
        if(vis.saveTrajectory(outPath.string())){
            std::cout << "Saved trajectory to " << outPath.string() << std::endl;
        } else {
            std::cerr << "Failed to save trajectory to " << outPath.string() << std::endl;
        }
    } catch(const std::exception &e){
        std::cerr << "Error saving trajectory: " << e.what() << std::endl;
    }
    return 0;
}