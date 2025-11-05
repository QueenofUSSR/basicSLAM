#include "FeatureExtractor.hpp"

#include <limits>
#include <cmath>

FeatureExtractor::FeatureExtractor(int nfeatures)
    : nfeatures_(nfeatures)
{
    orb_ = cv::ORB::create(nfeatures_);
}

// Adaptive Non-Maximal Suppression (ANMS)
static void anms(const std::vector<cv::KeyPoint> &in, std::vector<cv::KeyPoint> &out, int maxFeatures)
{
    out.clear();
    if(in.empty()) return;
    int N = (int)in.size();
    if(maxFeatures <= 0 || N <= maxFeatures){ out = in; return; }

    // For each keypoint, find distance to the nearest keypoint with strictly higher response
    std::vector<float> radius(N, std::numeric_limits<float>::infinity());
    for(int i=0;i<N;++i){
        for(int j=0;j<N;++j){
            if(in[j].response > in[i].response){
                float dx = in[i].pt.x - in[j].pt.x;
                float dy = in[i].pt.y - in[j].pt.y;
                float d2 = dx*dx + dy*dy;
                if(d2 < radius[i]) radius[i] = d2;
            }
        }
        // if no stronger keypoint exists, radius[i] stays INF
    }

    // Now pick top maxFeatures by radius (larger radius preferred). If radius==INF, treat as large.
    std::vector<int> idx(N);
    for(int i=0;i<N;++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){
        float ra = radius[a]; float rb = radius[b];
        if(std::isinf(ra) && std::isinf(rb)) return in[a].response > in[b].response; // tie-break by response
        if(std::isinf(ra)) return true;
        if(std::isinf(rb)) return false;
        if(ra == rb) return in[a].response > in[b].response;
        return ra > rb;
    });

    int take = std::min(maxFeatures, N);
    out.reserve(take);
    for(int i=0;i<take;++i) out.push_back(in[idx[i]]);
}

void FeatureExtractor::detectAndCompute(const cv::Mat &image, std::vector<cv::KeyPoint> &kps, cv::Mat &desc)
{
    kps.clear(); desc.release();
    if(image.empty()) return;

    // 1) detect candidate keypoints with ORB (so we have responses)
    std::vector<cv::KeyPoint> candidates;
    orb_->detect(image, candidates);

    // 2) apply ANMS to select up to nfeatures_
    std::vector<cv::KeyPoint> selected;
    anms(candidates, selected, nfeatures_);

    // 3) compute descriptors for selected keypoints
    if(selected.empty()) return;
    orb_->compute(image, selected, desc);

    // return selected keypoints
    kps = std::move(selected);
}
