#include "Localizer.hpp"
#include "Matcher.hpp"
#include <opencv2/calib3d.hpp>
#include <filesystem>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

Localizer::Localizer(float ratio) : ratio_(ratio) {}

bool Localizer::tryPnP(const MapManager &map, const cv::Mat &desc, const std::vector<cv::KeyPoint> &kps,
                       double fx, double fy, double cx, double cy, int imgW, int imgH,
                       int min_inliers,
                       cv::Mat &R_out, cv::Mat &t_out, int &inliers_out,
                       int frame_id, const cv::Mat *image, const std::string &outDir,
                       int *out_preMatches, int *out_postMatches, double *out_meanReproj) const {
    inliers_out = 0; R_out.release(); t_out.release();
    const auto &mappoints = map.mappoints();
    const auto &keyframes = map.keyframes();
    if(mappoints.empty() || keyframes.empty() || desc.empty()) return false;

    // Use last keyframe as prior
    const KeyFrame &last = keyframes.back();
    cv::Mat lastR = last.R_w, lastT = last.t_w;

    // select visible candidates
    std::vector<int> candidates = map.findVisibleCandidates(lastR, lastT, fx, fy, cx, cy, imgW, imgH);
    if(candidates.empty()) return false;

    // gather descriptors from map (prefer mp.descriptor if available)
    cv::Mat trainDesc;
    std::vector<cv::Point3f> objPts; objPts.reserve(candidates.size());
    std::vector<int> trainMpIds; trainMpIds.reserve(candidates.size());
    for(int idx: candidates){
        const auto &mp = mappoints[idx];
        if(mp.observations.empty()) continue;
        // prefer representative descriptor on MapPoint
        if(!mp.descriptor.empty()){
            trainDesc.push_back(mp.descriptor.row(0));
        } else {
            auto ob = mp.observations.front();
            int kfid = ob.first; int kpidx = ob.second;
            int kfIdx = map.keyframeIndex(kfid);
            if(kfIdx < 0) continue;
            const KeyFrame &kf = keyframes[kfIdx];
            if(kf.desc.empty() || kpidx < 0 || kpidx >= kf.desc.rows) continue;
            trainDesc.push_back(kf.desc.row(kpidx));
        }
        objPts.emplace_back((float)mp.p.x, (float)mp.p.y, (float)mp.p.z);
        trainMpIds.push_back(mp.id);
    }
    if(trainDesc.empty()) return false;

    // Forward and backward knn to enable mutual cross-check
    cv::BFMatcher bf(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnF, knnB;
    bf.knnMatch(desc, trainDesc, knnF, 2);
    bf.knnMatch(trainDesc, desc, knnB, 1);

    int preMatches = static_cast<int>(knnF.size());
    if(out_preMatches) *out_preMatches = preMatches;

    // Ratio + mutual
    const float RATIO = ratio_;
    std::vector<cv::DMatch> goodMatches;
    goodMatches.reserve(knnF.size());
    for(size_t q=0;q<knnF.size();++q){
        if(knnF[q].empty()) continue;
        const cv::DMatch &m0 = knnF[q][0];
        if(knnF[q].size() >= 2){
            const cv::DMatch &m1 = knnF[q][1];
            if(m0.distance > RATIO * m1.distance) continue;
        }
        int trainIdx = m0.trainIdx;
        // mutual check: ensure best match of trainIdx points back to this query
        if(trainIdx < 0 || trainIdx >= static_cast<int>(knnB.size())) continue;
        if(knnB[trainIdx].empty()) continue;
        int backIdx = knnB[trainIdx][0].trainIdx; // index in desc
        if(backIdx != static_cast<int>(q)) continue;
        // passed ratio and mutual
        goodMatches.push_back(cv::DMatch(static_cast<int>(q), trainIdx, m0.distance));
    }

    if(out_postMatches) *out_postMatches = static_cast<int>(goodMatches.size());

    if(goodMatches.size() < static_cast<size_t>(std::max(6, min_inliers))) return false;

    // build PnP inputs
    std::vector<cv::Point3f> obj; std::vector<cv::Point2f> imgpts; obj.reserve(goodMatches.size()); imgpts.reserve(goodMatches.size());
    std::vector<int> matchedMpIds; matchedMpIds.reserve(goodMatches.size());
    for(const auto &m: goodMatches){
        int q = m.queryIdx; int t = m.trainIdx;
        if(t < 0 || t >= static_cast<int>(objPts.size()) || q < 0 || q >= static_cast<int>(kps.size())) continue;
        obj.push_back(objPts[t]);
        imgpts.push_back(kps[q].pt);
        matchedMpIds.push_back(trainMpIds[t]);
    }

    if(obj.size() < static_cast<size_t>(std::max(6, min_inliers))) return false;

    cv::Mat camera = (cv::Mat_<double>(3,3) << fx,0,cx, 0,fy,cy, 0,0,1);
    cv::Mat dist = cv::Mat::zeros(4,1,CV_64F);
    std::vector<int> inliersIdx;
    bool ok = cv::solvePnPRansac(obj, imgpts, camera, dist, R_out, t_out, false,
                                 100, 8.0, 0.99, inliersIdx, cv::SOLVEPNP_ITERATIVE);
    if(!ok) return false;
    inliers_out = static_cast<int>(inliersIdx.size());

    // compute mean reprojection error on inliers
    double meanReproj = 0.0;
    if(!inliersIdx.empty()){
        std::vector<cv::Point2f> proj;
        cv::projectPoints(obj, R_out, t_out, camera, dist, proj);
        double sum = 0.0;
        for(int idx: inliersIdx){
            double e = std::hypot(proj[idx].x - imgpts[idx].x, proj[idx].y - imgpts[idx].y);
            sum += e;
        }
        meanReproj = sum / inliersIdx.size();
    }
    if(out_meanReproj) *out_meanReproj = meanReproj;

    // Diagnostics: draw matches and inliers if requested
    if(!outDir.empty() && image){
        try{
            std::filesystem::create_directories(outDir);
            cv::Mat vis;
            if(image->channels() == 1) cv::cvtColor(*image, vis, cv::COLOR_GRAY2BGR);
            else vis = image->clone();
            // draw all good matches as small circles; inliers green
            for(size_t i=0;i<goodMatches.size();++i){
                cv::Point2f p = imgpts[i];
                bool isInlier = std::find(inliersIdx.begin(), inliersIdx.end(), static_cast<int>(i)) != inliersIdx.end();
                cv::Scalar col = isInlier ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255);
                cv::circle(vis, p, 3, col, 2, cv::LINE_AA);
            }
            std::ostringstream name; name << outDir << "/pnp_frame_" << frame_id << ".png";
            cv::putText(vis, "pre=" + std::to_string(preMatches) + " post=" + std::to_string(goodMatches.size()) + " inliers=" + std::to_string(inliers_out) + " mean_px=" + std::to_string(meanReproj),
                        cv::Point(10,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
            cv::imwrite(name.str(), vis);
            // append a small CSV-like log
            std::ofstream logf((outDir + "/pnp_stats.csv"), std::ios::app);
            if(logf){
                logf << frame_id << "," << preMatches << "," << goodMatches.size() << "," << inliers_out << "," << meanReproj << "\n";
                logf.close();
            }
        } catch(...) { /* don't crash on diagnostics */ }
    }

    return inliers_out >= min_inliers;
}
