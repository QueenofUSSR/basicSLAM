#include "VisualOdometry.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "Optimizer.hpp"
#include <numeric>

namespace cv {
namespace vo {

static cv::Mat makeSE3(const cv::Mat& R, const cv::Mat& t) {
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T(cv::Rect(0,0,3,3)));
    t.copyTo(T(cv::Rect(3,0,1,3)));
    return T;
}

VisualOdometry::VisualOdometry(const cv::Ptr<cv::Feature2D>& feature,
                               const cv::Ptr<cv::DescriptorMatcher>& matcher,
                               const CameraIntrinsics& intrinsics,
                               const VOParams& params)
    : feature_(feature), matcher_(matcher), K_(intrinsics), params_(params) {
    currentPoseSE3_ = cv::Mat::eye(4,4,CV_64F);
}

bool VisualOdometry::processFrame(const cv::Mat& img, double /*timestamp*/) {
    frameCount_++;
    if (!initialized_) {
        if (lastImg_.empty()) {
            lastImg_ = img.clone();
            return false;
        }
        initialized_ = initializeTwoView(lastImg_, img);
        lastImg_ = img.clone();
        if (initialized_) trajectory_.push_back(currentPoseSE3_.clone());
        return initialized_;
    }

    bool ok = trackWithPnP(img);
    lastImg_ = img.clone();
    if (ok) {
        trajectory_.push_back(currentPoseSE3_.clone());
        if (shouldInsertKeyframe()) {
            // promote current frame to new keyframe using current features and PnP inliers
            kfPoseSE3_ = currentPoseSE3_.clone();
            kf_keypoints_ = cur_keypoints_;
            kf_descriptors_ = cur_descriptors_.clone();
            kf_kp_to_map_.assign((int)kf_keypoints_.size(), -1);
                keyframePoses_.push_back(kfPoseSE3_.clone());
                keyframeKeypoints_.push_back(kf_keypoints_);
                keyframeDescriptors_.push_back(kf_descriptors_.clone());
            // seed mapping for tracked correspondences
            for (auto &pr : curFeat_to_map_inliers_) {
                int curIdx = pr.first; int mpIdx = pr.second;
                if (curIdx >=0 && curIdx < (int)kf_kp_to_map_.size()) kf_kp_to_map_[curIdx] = mpIdx;
            }
                // push kp->map history for BA
                keyframeKpToMap_.push_back(kf_kp_to_map_);
            triangulateWithLastKeyframe();
            runLocalBAIfEnabled();
        }
    }
    return ok;
}

cv::Mat VisualOdometry::getCurrentPose() const { return currentPoseSE3_.clone(); }
std::vector<cv::Mat> VisualOdometry::getTrajectory() const { return trajectory_; }

void VisualOdometry::setEnableBackend(bool on) { backendEnabled_ = on; }
void VisualOdometry::setWindowSize(int N) { params_.localWindowSize = N; }
void VisualOdometry::setBAParams(double reprojThresh, int maxIters) {
    params_.reprojErrThresh = reprojThresh;
    params_.baMaxIters = std::max(1, maxIters);
}
void VisualOdometry::setBackendType(int type) { params_.backendType = type; }
void VisualOdometry::setMode(Mode m) { mode_ = m; }
bool VisualOdometry::saveMap(const std::string& /*path*/) const { return false; }
bool VisualOdometry::loadMap(const std::string& /*path*/) { return false; }
void VisualOdometry::enableLoopClosure(bool on) { loopClosureEnabled_ = on; }
void VisualOdometry::setPlaceRecognizer(const cv::Ptr<cv::Feature2D>& /*vocabFeature*/) {}

bool VisualOdometry::initializeTwoView(const cv::Mat& img0, const cv::Mat& img1) {
    std::vector<cv::KeyPoint> k0, k1; cv::Mat d0, d1;
    feature_->detectAndCompute(img0, cv::noArray(), k0, d0);
    feature_->detectAndCompute(img1, cv::noArray(), k1, d1);
    if (k0.size() < 50 || k1.size() < 50) return false;

    std::vector<std::vector<cv::DMatch>> knn01; matcher_->knnMatch(d0, d1, knn01, 2);
    std::vector<cv::DMatch> good;
    for (auto& v : knn01) if (v.size()>=2 && v[0].distance < 0.75 * v[1].distance) good.push_back(v[0]);
    if (good.size() < 80) return false;

    std::vector<cv::Point2f> p0, p1;
    p0.reserve(good.size()); p1.reserve(good.size());
    for (auto& m : good) { p0.push_back(k0[m.queryIdx].pt); p1.push_back(k1[m.trainIdx].pt); }

    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(p0, p1, K_.K, cv::RANSAC, params_.ransacProb, params_.ransacThresh, mask);
    if (E.empty()) return false;

    cv::Mat R, t;
    int inliers = cv::recoverPose(E, p0, p1, K_.K, R, t, mask);
    if (inliers < params_.minInitInliers) return false;

    // Set first pose = Identity, second = (R,t)
    currentPoseSE3_ = makeSE3(R, t);

    // Minimal triangulation demo (no storage of per-point observations yet)
    cv::Mat P0 = K_.K * cv::Mat::eye(3,4,CV_64F);
    cv::Mat P1(3,4,CV_64F); R.copyTo(P1(cv::Rect(0,0,3,3))); t.copyTo(P1(cv::Rect(3,0,1,3))); P1 = K_.K * P1;

    std::vector<cv::Point2f> tp0, tp1; tp0.reserve(inliers); tp1.reserve(inliers);
    std::vector<int> inlier_trainIdx; inlier_trainIdx.reserve(inliers);
    for (size_t i=0;i<good.size();++i) if (mask.at<uchar>(int(i))) {
        tp0.push_back(p0[i]); tp1.push_back(p1[i]);
        inlier_trainIdx.push_back(good[i].trainIdx);
    }
    if (tp0.size() >= 20) {
        cv::Mat X4;
        cv::triangulatePoints(P0, P1, tp0, tp1, X4);
        // initialize KF as second frame
        kf_keypoints_ = k1;
        kf_descriptors_ = d1.clone();
        kf_kp_to_map_.assign((int)kf_keypoints_.size(), -1);
        kfPoseSE3_ = currentPoseSE3_.clone();
        for (int i=0;i<X4.cols;i++) {
            cv::Vec4d h = X4.col(i);
            if (std::abs(h[3]) < 1e-8) continue;
            cv::Point3d X(h[0]/h[3], h[1]/h[3], h[2]/h[3]);
            if (X.z > 0) {
                int mpIndex = (int)mapPoints_.size();
                mapPoints_.push_back(X);
                int trainIdx = inlier_trainIdx[i];
                if (trainIdx >=0 && trainIdx < (int)kf_kp_to_map_.size())
                    kf_kp_to_map_[trainIdx] = mpIndex;
            }
        }
    }
    return true;
}

bool VisualOdometry::trackWithPnP(const cv::Mat& img) {
    cur_keypoints_.clear();
    feature_->detectAndCompute(img, cv::noArray(), cur_keypoints_, cur_descriptors_);
    if (kf_descriptors_.empty() || cur_descriptors_.empty()) return false;

    // Match KF -> current
    std::vector<std::vector<cv::DMatch>> knn;
    matcher_->knnMatch(kf_descriptors_, cur_descriptors_, knn, 2);
    std::vector<cv::DMatch> good;
    good.reserve(knn.size());
    for (auto &v : knn) if (v.size()>=2 && v[0].distance < 0.75*v[1].distance) good.push_back(v[0]);

    std::vector<cv::Point3f> objPts; objPts.reserve(good.size());
    std::vector<cv::Point2f> imgPts; imgPts.reserve(good.size());
    std::vector<int> curIdxOfCorr; curIdxOfCorr.reserve(good.size());
    std::vector<int> mapIdxOfCorr; mapIdxOfCorr.reserve(good.size());

    int unmatchedCount = 0;
    for (auto &m : good) {
        int kf_idx = m.queryIdx;
        int mp_idx = (kf_idx >=0 && kf_idx < (int)kf_kp_to_map_.size()) ? kf_kp_to_map_[kf_idx] : -1;
        if (mp_idx >= 0 && mp_idx < (int)mapPoints_.size()) {
            const cv::Point3d &Xd = mapPoints_[mp_idx];
            objPts.emplace_back((float)Xd.x,(float)Xd.y,(float)Xd.z);
            imgPts.push_back(cur_keypoints_[m.trainIdx].pt);
            curIdxOfCorr.push_back(m.trainIdx);
            mapIdxOfCorr.push_back(mp_idx);
        } else {
            unmatchedCount++;
        }
    }
    unmatchedRatio_ = good.empty() ? 0.0 : (double)unmatchedCount / (double)good.size();
    if (objPts.size() < 4) return false;

    cv::Mat rvec, tvec, R;
    std::vector<int> inliers;
    bool ok = cv::solvePnPRansac(objPts, imgPts, K_.K, K_.dist, rvec, tvec, false,
                                 100, params_.pnpReprojErr, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
    if (!ok || (int)inliers.size() < std::max(4, params_.pnpMinInliers)) return false;

    cv::Rodrigues(rvec, R);
    currentPoseSE3_ = makeSE3(R, tvec);
    lastPnPInliers_ = (int)inliers.size();

    // keep inlier correspondences for KF mapping when inserting a new KF
    curFeat_to_map_inliers_.clear();
    curFeat_to_map_inliers_.reserve(inliers.size());
    for (int idx : inliers) {
        int curIdx = curIdxOfCorr[idx];
        int mpIdx = mapIdxOfCorr[idx];
        curFeat_to_map_inliers_.emplace_back(curIdx, mpIdx);
    }
    return true;
}

bool VisualOdometry::shouldInsertKeyframe() const {
    // simple: periodic or tracking weakening
    bool periodic = (frameCount_ % params_.keyframeInterval) == 0;
    bool weakTracking = (lastPnPInliers_ > 0 && lastPnPInliers_ < params_.pnpMinInliers + 10);

    // baseline & rotation criteria
    auto extractRt = [](const cv::Mat& T, cv::Mat& R, cv::Mat& t){ R = T(cv::Rect(0,0,3,3)).clone(); t = T(cv::Rect(3,0,1,3)).clone(); };
    cv::Mat R_prev, t_prev, R_cur, t_cur; extractRt(kfPoseSE3_, R_prev, t_prev); extractRt(currentPoseSE3_, R_cur, t_cur);
    double baseline = cv::norm(t_cur - t_prev);
    cv::Mat R_delta = R_prev.t() * R_cur;
    double traceVal = R_delta.at<double>(0,0) + R_delta.at<double>(1,1) + R_delta.at<double>(2,2);
    double rotAngle = std::acos(std::min(1.0, std::max(-1.0, (traceVal - 1.0) / 2.0))) * 180.0 / CV_PI;
    bool baselineEnough = baseline > params_.keyframeMinBaseline;
    bool rotationEnough = rotAngle > params_.keyframeMinRotationDeg;

    // mapping expansion need: many unmatched candidate features
    bool needExpansion = unmatchedRatio_ > 0.5; // heuristic

    return periodic || weakTracking || baselineEnough || rotationEnough || needExpansion;
}

void VisualOdometry::triangulateWithLastKeyframe() {
    if (kf_descriptors_.empty() || cur_descriptors_.empty()) return;

    // Match KF -> current again to find candidates without map points
    std::vector<std::vector<cv::DMatch>> knn;
    matcher_->knnMatch(kf_descriptors_, cur_descriptors_, knn, 2);
    std::vector<cv::Point2f> kfPts, curPts;
    std::vector<int> kfIdxs;
    for (auto &v : knn) {
        if (v.size()>=2 && v[0].distance < 0.75*v[1].distance) {
            int qi = v[0].queryIdx; int ti = v[0].trainIdx;
            if (qi >=0 && qi < (int)kf_kp_to_map_.size() && kf_kp_to_map_[qi] == -1) {
                kfPts.push_back(kf_keypoints_[qi].pt);
                curPts.push_back(cur_keypoints_[ti].pt);
                kfIdxs.push_back(qi);
            }
        }
    }
    if (kfPts.size() < 15) return;

    auto extractRt = [](const cv::Mat& T, cv::Mat& R, cv::Mat& t){
        R = T(cv::Rect(0,0,3,3)).clone();
        t = T(cv::Rect(3,0,1,3)).clone();
    };
    cv::Mat R1,t1,R2,t2; extractRt(kfPoseSE3_, R1,t1); extractRt(currentPoseSE3_, R2,t2);

    cv::Mat P1(3,4,CV_64F), P2(3,4,CV_64F);
    R1.copyTo(P1(cv::Rect(0,0,3,3))); t1.copyTo(P1(cv::Rect(3,0,1,3)));
    R2.copyTo(P2(cv::Rect(0,0,3,3))); t2.copyTo(P2(cv::Rect(3,0,1,3)));
    P1 = K_.K * P1; P2 = K_.K * P2;

    cv::Mat X4; cv::triangulatePoints(P1, P2, kfPts, curPts, X4);

    auto reprojErr = [&](const cv::Point3d& X, const cv::Mat& R, const cv::Mat& t, const cv::Point2f& uv){
        cv::Mat Xv = (cv::Mat_<double>(3,1) << X.x,X.y,X.z);
        cv::Mat x = K_.K * (R*Xv + t);
        double u = x.at<double>(0)/x.at<double>(2);
        double v = x.at<double>(1)/x.at<double>(2);
        double du = u - uv.x; double dv = v - uv.y; return std::sqrt(du*du+dv*dv);
    };

    for (int i=0;i<X4.cols;i++) {
        double hx = X4.at<double>(0,i);
        double hy = X4.at<double>(1,i);
        double hz = X4.at<double>(2,i);
        double hw = X4.at<double>(3,i);
        if (std::abs(hw) < 1e-8) continue;
        cv::Point3d X(hx/hw, hy/hw, hz/hw);
        // positive depth check (in both views)
        cv::Mat R1_,t1_,R2_,t2_;
        R1_ = R1; t1_ = t1; R2_ = R2; t2_ = t2;
        cv::Mat Xv = (cv::Mat_<double>(3,1) << X.x,X.y,X.z);
        cv::Mat Y1 = R1_*Xv + t1_;
        cv::Mat Y2 = R2_*Xv + t2_;
        double z1 = Y1.ptr<double>(2)[0];
        double z2 = Y2.ptr<double>(2)[0];
        if (z1 <= 0 || z2 <= 0) continue;
        double e1 = reprojErr(X, R1, t1, kfPts[i]);
        double e2 = reprojErr(X, R2, t2, curPts[i]);
        if (e1 > params_.reprojErrThresh || e2 > params_.reprojErrThresh) continue;
        int mpIndex = (int)mapPoints_.size();
        mapPoints_.push_back(X);
        int kf_kp = kfIdxs[i];
        if (kf_kp >=0 && kf_kp < (int)kf_kp_to_map_.size()) kf_kp_to_map_[kf_kp] = mpIndex;
    }
}

void VisualOdometry::runLocalBAIfEnabled() {
    if (!backendEnabled_) return;
    // Collect recent keyframes and build observations for map points
    int N = std::max(2, params_.localWindowSize);
    int totalKFs = (int)keyframePoses_.size();
    if (totalKFs < 2) return;
    int startIdx = std::max(0, totalKFs - N);

    // Build KeyFrame vector for optimizer
    std::vector<KeyFrame> kfs;
    kfs.reserve(totalKFs - startIdx);
    for (int i = startIdx; i < totalKFs; ++i) {
        KeyFrame kf;
        kf.id = i; // use history index as id
        kf.kps = keyframeKeypoints_[i];
        kf.desc = keyframeDescriptors_[i];
        // extract R,t from SE3 4x4
        const cv::Mat &T = keyframePoses_[i];
        cv::Mat R = T(cv::Rect(0,0,3,3)).clone();
        cv::Mat t = T(cv::Rect(3,0,1,3)).clone();
        kf.R_w = R; kf.t_w = t;
        kfs.push_back(kf);
    }

    // Build MapPoint vector with observations from selected window
    std::vector<MapPoint> mps;
    mps.reserve(mapPoints_.size());
    // Track which points are observed within the window
    std::vector<char> pointObserved(mapPoints_.size(), 0);
    for (int i = startIdx; i < totalKFs; ++i) {
        const auto &kp2mp = keyframeKpToMap_[i];
        for (int kpIdx = 0; kpIdx < (int)kp2mp.size(); ++kpIdx) {
            int mpIdx = kp2mp[kpIdx];
            if (mpIdx >= 0 && mpIdx < (int)mapPoints_.size()) {
                pointObserved[mpIdx] = 1;
            }
        }
    }
    // Create mps with observations
    for (size_t mpIdx = 0; mpIdx < mapPoints_.size(); ++mpIdx) {
        if (!pointObserved[mpIdx]) continue; // only optimize observed points
        MapPoint mp;
        mp.id = (int)mpIdx;
        mp.p = mapPoints_[mpIdx];
        for (int i = startIdx; i < totalKFs; ++i) {
            const auto &kp2mp = keyframeKpToMap_[i];
            if ((int)kp2mp.size() == 0) continue;
            for (int kpIdx = 0; kpIdx < (int)kp2mp.size(); ++kpIdx) {
                if (kp2mp[kpIdx] == (int)mpIdx) {
                    mp.observations.emplace_back(i, kpIdx);
                }
            }
        }
        if (!mp.observations.empty()) mps.push_back(mp);
    }

    if (kfs.size() < 2 || mps.empty()) return;

    // Local/fixed KF indices relative to full history ids
    std::vector<int> localKfIndices;
    localKfIndices.reserve(kfs.size());
    for (const auto &kf : kfs) localKfIndices.push_back(kf.id);
    // Fix the oldest KF in the window to anchor optimization
    std::vector<int> fixedKfIndices = { startIdx };

    // Select backend
#ifdef USE_OPENCV_SFM
    if (params_.backendType == 1) {
        Optimizer::localBundleAdjustmentSFM(
            kfs, mps, localKfIndices, fixedKfIndices,
            K_.K.at<double>(0,0), K_.K.at<double>(1,1), K_.K.at<double>(0,2), K_.K.at<double>(1,2),
            params_.baMaxIters);
    } else
#endif
    {
        Optimizer::localBundleAdjustment(
            kfs, mps, localKfIndices, fixedKfIndices,
            K_.K.at<double>(0,0), K_.K.at<double>(1,1), K_.K.at<double>(0,2), K_.K.at<double>(1,2),
            params_.baMaxIters);
    }

    // Write back optimized poses and points
    for (const auto &kf : kfs) {
        int i = kf.id;
        cv::Mat T = cv::Mat::eye(4,4,CV_64F);
        kf.R_w.copyTo(T(cv::Rect(0,0,3,3)));
        kf.t_w.copyTo(T(cv::Rect(3,0,1,3)));
        if (i >= 0 && i < totalKFs) keyframePoses_[i] = T.clone();
        if (i == totalKFs - 1) { // also update current KF pose cache
            kfPoseSE3_ = T.clone();
            currentPoseSE3_ = T.clone();
        }
    }
    for (const auto &mp : mps) {
        if (mp.id >= 0 && mp.id < (int)mapPoints_.size()) {
            mapPoints_[mp.id] = mp.p;
        }
    }
}

} // namespace vo
} // namespace cv