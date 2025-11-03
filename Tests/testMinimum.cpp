#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    // Map settings
    const int W = 1000, H = 800;
    const double meters_per_pixel = 0.02; // 1 pixel = 2 cm
    const cv::Point2d origin_px(W/2, H/2); // world (0,0) -> center of image

    cv::Mat map = cv::Mat::zeros(H, W, CV_8UC3);

    // Example trajectory in world meters (x, y)
    std::vector<cv::Point2d> traj = {{0,0}, {0.5,0.2}, {1.0,0.3}, {1.5, 0.6}};

    auto worldToPixel = [&](const cv::Point2d &p)->cv::Point {
        int x = int(origin_px.x + p.x / meters_per_pixel);
        int y = int(origin_px.y - p.y / meters_per_pixel); // flip y
        return {x, y};
    };

    // draw grid
    for (int gx = 0; gx < W; gx += 50) cv::line(map, {gx,0}, {gx,H}, {30,30,30}, 1);
    for (int gy = 0; gy < H; gy += 50) cv::line(map, {0,gy}, {W,gy}, {30,30,30}, 1);

    // draw trajectory
    for (size_t i=1;i<traj.size();++i) {
        cv::line(map, worldToPixel(traj[i-1]), worldToPixel(traj[i]), {0,255,0}, 2);
    }

    // draw robot at last pose
    cv::Point p = worldToPixel(traj.back());
    cv::circle(map, p, 6, {0,0,255}, -1);
    cv::putText(map, "Robot", p + cv::Point(10,-10), cv::FONT_HERSHEY_SIMPLEX, 0.5, {255,255,255}, 1);

    cv::imshow("topdown", map);
    cv::waitKey(0);
    return 0;
}