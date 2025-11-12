#include "Controller.hpp"

#include "DataLoader.hpp"
#include "FeatureExtractor.hpp"
#include "Matcher.hpp"
#include "PoseEstimator.hpp"
#include "Visualizer.hpp"
#include "KeyFrame.hpp"
#include "MapPoint.hpp"
#include "MapManager.hpp"
#include "Localizer.hpp"
#include "Optimizer.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <fstream>
#include <algorithm>

Controller::Controller() {
    // empty
}

int Controller::run(const std::string &imageDir, double scale_m, const ControllerOptions &options){
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
    MapManager map;
    // configure Localizer with a slightly stricter Lowe ratio (0.7)
    Localizer localizer(0.7f);

    // prepare per-run CSV diagnostics
    std::string runTimestamp;
    {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&t);
        std::ostringstream ss; ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        runTimestamp = ss.str();
    }
    std::filesystem::path resultDir("../../result");
    if(!std::filesystem::exists(resultDir)) std::filesystem::create_directories(resultDir);
    // create a per-run folder under result/ named by timestamp
    std::filesystem::path runDir = resultDir / runTimestamp;
    if(!std::filesystem::exists(runDir)) std::filesystem::create_directories(runDir);
    std::filesystem::path csvPath = runDir / std::string("run.csv");
    std::ofstream csv(csvPath);
    if(csv){
        csv << "frame_id,mean_diff,median_flow,pre_matches,post_matches,inliers,inlier_ratio,integrated\n";
        csv.flush();
        std::cout << "Writing diagnostics to " << csvPath.string() << std::endl;
    } else {
        std::cerr << "Failed to open diagnostics CSV " << csvPath.string() << std::endl;
    }

    cv::Mat R_g = cv::Mat::eye(3,3,CV_64F);
    cv::Mat t_g = cv::Mat::zeros(3,1,CV_64F);

    // simple map structures
    std::vector<KeyFrame> keyframes;
    std::vector<MapPoint> mappoints;
    std::unordered_map<int,int> keyframeIdToIndex;

    // Backend (BA) thread primitives
    std::mutex mapMutex; // protects map and keyframe modifications and writeback
    std::condition_variable backendCv;
    std::atomic<bool> backendStop(false);
    std::atomic<int> backendRequests(0);
    const int LOCAL_BA_WINDOW = 5; // window size for local BA (adjustable)

    // Start backend thread: waits for notifications and runs BA on a snapshot
    std::thread backendThread([&]() {
        while(!backendStop.load()){
            std::unique_lock<std::mutex> lk(mapMutex);
            backendCv.wait(lk, [&]{ return backendStop.load() || backendRequests.load() > 0; });
            if(backendStop.load()) break;
            // snapshot map and keyframes
            auto kfs_snapshot = map.keyframes();
            auto mps_snapshot = map.mappoints();
            // reset requests
            backendRequests.store(0);
            lk.unlock();

            // determine local window
            int K = static_cast<int>(kfs_snapshot.size());
            if(K <= 0) continue;
            int start = std::max(0, K - LOCAL_BA_WINDOW);
            std::vector<int> localKfIndices;
            for(int ii = start; ii < K; ++ii) localKfIndices.push_back(ii);
            std::vector<int> fixedKfIndices;
            if(start > 0) fixedKfIndices.push_back(0);

            // Run BA on snapshot (may take time) - uses Optimizer which will use g2o if enabled
            Optimizer::localBundleAdjustment(kfs_snapshot, mps_snapshot, localKfIndices, fixedKfIndices,
                                            loader.fx(), loader.fy(), loader.cx(), loader.cy(), 10);

            // write back optimized poses/points into main map under lock using id-based lookup
            std::lock_guard<std::mutex> lk2(mapMutex);
            auto &kfs_ref = const_cast<std::vector<KeyFrame>&>(map.keyframes());
            auto &mps_ref = const_cast<std::vector<MapPoint>&>(map.mappoints());
            // copy back poses by id to ensure we update the authoritative containers
            for(const auto &kf_opt : kfs_snapshot){
                int idx = map.keyframeIndex(kf_opt.id);
                if(idx >= 0 && idx < static_cast<int>(kfs_ref.size())){
                    kfs_ref[idx].R_w = kf_opt.R_w.clone();
                    kfs_ref[idx].t_w = kf_opt.t_w.clone();
                }
            }
            // copy back mappoint positions by id
            for(const auto &mp_opt : mps_snapshot){
                if(mp_opt.id <= 0) continue;
                int idx = map.mapPointIndex(mp_opt.id);
                if(idx >= 0 && idx < static_cast<int>(mps_ref.size())){
                    mps_ref[idx].p = mp_opt.p;
                }
            }
        }
    });

    cv::Mat frame;
    std::string imgPath;
    int frame_id = 0;

    // persistent previous-frame storage (declare outside loop so detectAndCompute can use them)
    static std::vector<cv::KeyPoint> prevKp;
    static cv::Mat prevGray, prevDesc;
    while(loader.getNextImage(frame, imgPath)){
        cv::Mat gray = frame;
        if(gray.channels() > 1) cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        // use flow-aware detect when previous frame exists
        if(!prevGray.empty() && !prevKp.empty()){
            feat.detectAndCompute(gray, kps, desc, prevGray, prevKp, options.flow_weight_lambda);
        } else {
            feat.detectAndCompute(gray, kps, desc);
        }

    // (previous-frame storage declared outside loop)

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

            // compute per-match flow magnitudes and median flow
            std::vector<double> flows; flows.reserve(pts1.size());
            double median_flow = 0.0;
            for(size_t i=0;i<pts1.size();++i){
                double dx = pts2[i].x - pts1[i].x;
                double dy = pts2[i].y - pts1[i].y;
                flows.push_back(std::sqrt(dx*dx + dy*dy));
            }
            if(!flows.empty()){
                std::vector<double> tmp = flows;
                size_t mid = tmp.size()/2;
                std::nth_element(tmp.begin(), tmp.begin()+mid, tmp.end());
                median_flow = tmp[mid];
            }

            int pre_matches = static_cast<int>(goodMatches.size());
            int post_matches = pre_matches;

            // Try PnP against map points first (via Localizer)
            bool solvedByPnP = false;
            cv::Mat R_pnp, t_pnp; int inliers_pnp = 0;
            int preMatches_pnp = 0, postMatches_pnp = 0; double meanReproj_pnp = 0.0;
            if(localizer.tryPnP(map, desc, kps, loader.fx(), loader.fy(), loader.cx(), loader.cy(), gray.cols, gray.rows,
                                options.min_inliers, R_pnp, t_pnp, inliers_pnp, frame_id, &frame, runDir.string(),
                                &preMatches_pnp, &postMatches_pnp, &meanReproj_pnp)){
                solvedByPnP = true;
                std::cout << "PnP solved: preMatches="<<preMatches_pnp<<" post="<<postMatches_pnp<<" inliers="<<inliers_pnp<<" meanReproj="<<meanReproj_pnp<<std::endl;
            }

            if(pts1.size() >= 8 && !solvedByPnP){
                cv::Mat R, t, mask; int inliers = 0;
                bool ok = poseEst.estimate(pts1, pts2, loader.fx(), loader.fy(), loader.cx(), loader.cy(), R, t, mask, inliers);

                int matchCount = post_matches;
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
                // std::cout << "F" << frame_id << " diff=" << meanDiff << " median_flow=" << median_flow
                //           << " pre_matches=" << pre_matches << " post_matches=" << matchCount << " inliers=" << inliers << " inlierRatio=" << inlierRatio
                //           << " t_norm=" << t_norm << " rot_rad=" << rot_angle << std::endl;

                // decide whether to integrate
                // Prefer geometry-based decision (absolute inliers OR matchCount + ratio). Use image-diff/flow
                // only to skip when geometry is weak or motion truly negligible.
                bool integrate = true;
                if(!ok){
                    integrate = false;
                    // std::cout << "  -> pose estimation failed, skipping integration." << std::endl;
                } else if(inliers < MIN_INLIERS || matchCount < MIN_MATCHES){
                    // Not enough geometric support -> skip (unless absolute inliers pass)
                    integrate = false;
                    // std::cout << "  -> insufficient matches/inliers (by both absolute and relative metrics), skipping integration." << std::endl;
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
                        // std::cout << "  -> negligible motion and near-identical frames, skipping integration." << std::endl;
                    }
                }
                if (inliers >= options.min_inliers || (inliers >= 2 && matchCount > 50 && median_flow > 2.0)) {
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

                    // if we integrated, create a keyframe and optionally triangulate new map points
                if(integrate){
                    KeyFrame kf;
                    kf.id = frame_id;
                    kf.image = frame.clone();
                    kf.kps = kps;
                    kf.desc = desc.clone();
                    kf.R_w = R_g.clone(); kf.t_w = t_g.clone();

                    bool didTriangulate = false;
                    if(!map.keyframes().empty() && map.keyframes().back().id == frame_id - 1){
                        // triangulate between last keyframe and this frame using normalized coordinates
                        const KeyFrame &last = map.keyframes().back();
                        std::vector<cv::Point2f> pts1n, pts2n; pts1n.reserve(pts1.size()); pts2n.reserve(pts2.size());
                        double fx = loader.fx(), fy = loader.fy(), cx = loader.cx(), cy = loader.cy();
                        for(size_t i=0;i<pts1.size();++i){
                            pts1n.emplace_back(float((pts1[i].x - cx)/fx), float((pts1[i].y - cy)/fy));
                            pts2n.emplace_back(float((pts2[i].x - cx)/fx), float((pts2[i].y - cy)/fy));
                        }
                        // build kp index lists (matching goodMatches order)
                        std::vector<int> pts1_kp_idx; pts1_kp_idx.reserve(goodMatches.size());
                        std::vector<int> pts2_kp_idx; pts2_kp_idx.reserve(goodMatches.size());
                        for(const auto &m: goodMatches){ pts1_kp_idx.push_back(m.queryIdx); pts2_kp_idx.push_back(m.trainIdx); }
                        auto newPts = map.triangulateBetweenLastTwo(pts1n, pts2n, pts1_kp_idx, pts2_kp_idx, last, keyframes.empty() ? kf : keyframes.back(), fx, fy, cx, cy);
                        if(!newPts.empty()){
                            didTriangulate = true;
                            // already appended inside MapManager
                        }
                    }

                    {
                        // insert keyframe and map points under lock to keep consistent state
                        std::lock_guard<std::mutex> lk(mapMutex);
                        keyframes.push_back(std::move(kf));
                        map.addKeyFrame(keyframes.back());
                    }
                    if(didTriangulate){
                        std::cout << "Created keyframe " << frame_id << " and triangulated new map points (total=" << map.mappoints().size() << ")" << std::endl;
                    } else {
                        std::cout << "Created keyframe " << frame_id << " (no triangulation)" << std::endl;
                    }
                    // Notify backend thread to run local BA asynchronously
                    backendRequests.fetch_add(1);
                    backendCv.notify_one();
                }

                // write CSV line
                if(csv){
                    csv << frame_id << "," << meanDiff << "," << median_flow << "," << pre_matches << "," << post_matches << "," << inliers << "," << inlierRatio << "," << (integrate?1:0) << "\n";
                    csv.flush();
                }

                // Always show a single image; if we have matches, draw small boxes around matched keypoints
                cv::Mat visImg;
                if(frame.channels() > 1) visImg = frame.clone();
                else cv::cvtColor(gray, visImg, cv::COLOR_GRAY2BGR);
                std::string info = std::string("Frame ") + std::to_string(frame_id) + " matches=" + std::to_string(matchCount) + " inliers=" + std::to_string(inliers);
                if(!goodMatches.empty()){
                    for(size_t mi=0; mi<goodMatches.size(); ++mi){
                        cv::Point2f p2 = (mi < pts2.size()) ? pts2[mi] : kps[goodMatches[mi].trainIdx].pt;

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
        // save trajectory into the per-run folder using a simple filename (no timestamp)
        std::filesystem::path outDir = resultDir / runTimestamp;
        if(!std::filesystem::exists(outDir)) std::filesystem::create_directories(outDir);
        std::filesystem::path outPath = outDir / std::string("trajectory.png");
        if(vis.saveTrajectory(outPath.string())){
            std::cout << "Saved trajectory to " << outPath.string() << std::endl;
        } else {
            std::cerr << "Failed to save trajectory to " << outPath.string() << std::endl;
        }
    } catch(const std::exception &e){
        std::cerr << "Error saving trajectory: " << e.what() << std::endl;
    }

    // Shutdown backend thread gracefully
    backendStop.store(true);
    backendCv.notify_one();
    if(backendThread.joinable()) backendThread.join();

    return 0;
}