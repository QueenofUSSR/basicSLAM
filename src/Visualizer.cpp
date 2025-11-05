#include "Visualizer.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

Visualizer::Visualizer(int W, int H, double meters_per_pixel)
    : W_(W), H_(H), meters_per_pixel_(meters_per_pixel), mapSize_(W,H)
{
    map_ = cv::Mat::zeros(mapSize_, CV_8UC3);
}

cv::Point Visualizer::worldToPixel(const cv::Point2d &p) const {
    cv::Point2d origin(mapSize_.width/2.0, mapSize_.height/2.0);
    int x = int(origin.x + p.x / meters_per_pixel_);
    int y = int(origin.y - p.y / meters_per_pixel_);
    return cv::Point(x,y);
}

void Visualizer::addPose(double x, double z){
    traj_.emplace_back(x,z);
}

void Visualizer::showFrame(const cv::Mat &frame){
    if(frame.empty()) return;
    // Do not draw heading overlay on video frames; only show raw frame.
    cv::imshow("frame", frame);
}

void Visualizer::showTopdown(){
    map_ = cv::Mat::zeros(mapSize_, CV_8UC3);
    for (int gx = 0; gx < mapSize_.width; gx += 50) cv::line(map_, cv::Point(gx,0), cv::Point(gx,mapSize_.height), cv::Scalar(30,30,30), 1);
    for (int gy = 0; gy < mapSize_.height; gy += 50) cv::line(map_, cv::Point(0,gy), cv::Point(mapSize_.width,gy), cv::Scalar(30,30,30), 1);
    for(size_t i=1;i<traj_.size();++i){
        cv::Point p1 = worldToPixel(traj_[i-1]);
        cv::Point p2 = worldToPixel(traj_[i]);
        cv::line(map_, p1, p2, cv::Scalar(0,255,0), 2);
    }
    if(!traj_.empty()){
        cv::Point p = worldToPixel(traj_.back());
        // draw heading arrow on topdown map based on recent trajectory
        if(traj_.size() >= 2){
            int K = std::min<size_t>(5, traj_.size()-1);
            double dx = 0.0, dz = 0.0;
            for(int i=0;i<K;i++){
                auto a = traj_[traj_.size()-1 - i];
                auto b = traj_[traj_.size()-2 - i];
                dx += (a.x - b.x);
                dz += (a.y - b.y);
            }
            dx /= K; dz /= K;
            double norm = std::hypot(dx, dz);
            if(norm > 1e-6){
                dx /= norm; dz /= norm;
                // arrow length in world meters
                double arrow_m = 0.5; // 0.5 meters
                // tail is behind the current position by arrow_m, head (tip) at current position
                cv::Point2d tail_world(traj_.back().x - dx * arrow_m, traj_.back().y - dz * arrow_m);
                cv::Point tail_px = worldToPixel(tail_world);
                cv::arrowedLine(map_, tail_px, p, cv::Scalar(0,0,255), 2, cv::LINE_AA, 0, 0.3);
            }
        }
        // label near current position
        cv::putText(map_, "Robot", p + cv::Point(10,-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
    }
    cv::imshow("topdown", map_);
}

bool Visualizer::saveTrajectory(const std::string &path){
    if(map_.empty()) showTopdown();
    return cv::imwrite(path, map_);
}
