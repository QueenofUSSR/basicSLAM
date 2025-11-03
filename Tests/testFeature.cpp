// simple_orb_vo.cpp
// A tiny monocular visual odometry style demo using ORB features + matching.
// - Extract ORB from frames
// - Match consecutive frames using Hamming + ratio test
// - Estimate relative pose with findEssentialMat + recoverPose
// - Accumulate poses (up-to-scale) and display a 2D top-down trajectory

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;
using namespace cv;

static Point worldToPixel(const Point2d &p, const Size &mapSize, double meters_per_pixel)
{
    Point2d origin(mapSize.width/2.0, mapSize.height/2.0);
    int x = int(origin.x + p.x / meters_per_pixel);
    int y = int(origin.y - p.y / meters_per_pixel); // flip y
    return Point(x,y);
}

int main(int argc, char** argv)
{
    if(argc < 1){
        cout << "Usage: " << argv[0] << " [image_dir] [scale_m_per_unit=1.0]" << endl;
        return 0;
    }

    // image directory (defaults to dataset path requested)
    string img_dir = "datasets/MH01/mav0/cam0/data";
    if(argc >= 2) img_dir = argv[1];

    double scale_m = 1.0; // meters per recovered translation unit (user-supplied)
    if(argc >= 3) scale_m = atof(argv[2]);

    // Try to glob files in the directory
    vector<String> img_files;
    cv::glob(img_dir + "/*", img_files, false);
    if(img_files.empty()){
        cerr << "ERROR: no images found in " << img_dir << endl;
        return -1;
    }

    // Try to load intrinsics from sensor.yaml next to the images
    string sensor_yaml = "datasets/MH01/mav0/cam0/sensor.yaml";
    double fx = 700.0, fy = 700.0, cx = 0.5, cy = 0.5; // fallbacks
    auto loadIntrinsics = [&](const string &path)->bool{
        ifstream ifs(path);
        if(!ifs.is_open()) return false;
        string line;
        while(getline(ifs, line)){
            // find "intrinsics:" line
            auto pos = line.find("intrinsics:");
            if(pos != string::npos){
                // extract numbers inside brackets
                size_t lb = line.find('[', pos);
                size_t rb = line.find(']', pos);
                string nums;
                if(lb != string::npos && rb != string::npos && rb > lb){
                    nums = line.substr(lb+1, rb-lb-1);
                } else {
                    // maybe numbers are on following lines; read subsequent lines until ']' found
                    string rest;
                    while(getline(ifs, rest)){
                        nums += rest + " ";
                        if(rest.find(']') != string::npos) break;
                    }
                }
                // normalize separators
                for(char &c: nums) if(c == ',' || c == '[' || c == ']') c = ' ';
                stringstream ss(nums);
                vector<double> vals;
                double v;
                while(ss >> v) vals.push_back(v);
                if(vals.size() >= 4){
                    fx = vals[0]; fy = vals[1]; cx = vals[2]; cy = vals[3];
                    return true;
                }
            }
        }
        return false;
    };
    loadIntrinsics(sensor_yaml); // best-effort; use fallbacks if fails

    // Map settings
    const int W = 1000, H = 800;
    const double meters_per_pixel = 0.02; // 1 pixel = 2 cm in display (for visualization only)
    const Size mapSize(W,H);
    Mat map = Mat::zeros(mapSize, CV_8UC3);

    // ORB + matcher
    Ptr<ORB> orb = ORB::create(2000);
    BFMatcher matcher(NORM_HAMMING);

    Mat prevGray, prevDesc;
    vector<KeyPoint> prevKp;

    // Global pose (R, t) of current frame in world coordinates
    Mat R_g = Mat::eye(3,3,CV_64F);
    Mat t_g = Mat::zeros(3,1,CV_64F);

    vector<Point2d> trajectory; // store (x,z) pairs
    trajectory.emplace_back(0.0, 0.0);

    int frame_id = 0;
    Mat frame;
    namedWindow("frame", WINDOW_NORMAL);
    namedWindow("topdown", WINDOW_NORMAL);

    size_t file_idx = 0;
    while(true){
        if(file_idx >= img_files.size()) break;
        frame = imread(img_files[file_idx], IMREAD_GRAYSCALE);
        if(frame.empty()){
            cerr << "WARN: couldn't read " << img_files[file_idx] << ", skipping\n";
            file_idx++;
            continue;
        }
        Mat gray = frame; // already grayscale

        // detect + compute
        vector<KeyPoint> kps;
        Mat desc;
        orb->detectAndCompute(gray, noArray(), kps, desc);

        if(!prevGray.empty() && !prevDesc.empty() && !desc.empty()){
            // match
            vector<vector<DMatch>> knnMatches;
            matcher.knnMatch(prevDesc, desc, knnMatches, 2);

            const float ratio_thresh = 0.75f;
            vector<DMatch> goodMatches;
            for(size_t i=0;i<knnMatches.size();++i){
                if(knnMatches[i].size() < 2) continue;
                const DMatch &m1 = knnMatches[i][0];
                const DMatch &m2 = knnMatches[i][1];
                if(m1.distance < ratio_thresh * m2.distance)
                    goodMatches.push_back(m1);
            }

            // Draw matches (for visualization)
            Mat imgMatches;
            drawMatches(prevGray, prevKp, gray, kps, goodMatches, imgMatches,
                        Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            // Prepare points for Essential matrix
            vector<Point2f> pts1, pts2;
            pts1.reserve(goodMatches.size()); pts2.reserve(goodMatches.size());
            for(const DMatch &m : goodMatches){
                pts1.push_back(prevKp[m.queryIdx].pt);
                pts2.push_back(kps[m.trainIdx].pt);
            }

            if(pts1.size() >= 8){
                // Use intrinsics loaded from sensor.yaml (or fallbacks)
                double focal = (fx + fy) * 0.5;
                Point2d pp(cx, cy);
                // If principal point looks normalized or zero, fallback to image center
                if(pp.x <= 2.0 && pp.y <= 2.0){
                    pp = Point2d((double)gray.cols/2.0, (double)gray.rows/2.0);
                }

                Mat mask;
                Mat E = findEssentialMat(pts1, pts2, focal, pp, RANSAC, 0.999, 1.0, mask);
                if(!E.empty()){
                    Mat R, t;
                    int inliers = recoverPose(E, pts1, pts2, R, t, focal, pp, mask);

                    // Convert t to double and scale by user-provided scale (meters per unit)
                    Mat t_d; t.convertTo(t_d, CV_64F);
                    Mat t_scaled = t_d * scale_m; // scale factor to convert to meters

                    // Accumulate pose: new_pose = prev_pose * [R|t]
                    Mat R_prev = R_g.clone();
                    Mat t_prev = t_g.clone();

                    // Update translation: t_g = t_prev + R_prev * t_scaled
                    t_g = t_prev + R_prev * t_scaled;
                    // Update rotation: R_g = R_prev * R
                    Mat R_d;
                    R.convertTo(R_d, CV_64F);
                    R_g = R_prev * R_d;

                    // Save position (use x and z -> forward is z)
                    double x = t_g.at<double>(0);
                    double z = t_g.at<double>(2);
                    trajectory.emplace_back(x, z);

                    // Display some info
                    string info = format("Frame %d (%s): matches=%zu, inliers=%d", frame_id, img_files[file_idx].c_str(), goodMatches.size(), inliers);
                    putText(imgMatches, info, Point(10,20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 2);

                    imshow("frame", imgMatches);
                } else {
                    imshow("frame", gray);
                }
            } else {
                imshow("frame", gray);
            }
        } else {
            // first frame: show keypoints
            Mat vis;
            drawKeypoints(gray, kps, vis, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            imshow("frame", vis);
        }

        // Draw top-down map
        map = Mat::zeros(mapSize, CV_8UC3);
        // grid
        for (int gx = 0; gx < mapSize.width; gx += 50) line(map, Point(gx,0), Point(gx,mapSize.height), Scalar(30,30,30), 1);
        for (int gy = 0; gy < mapSize.height; gy += 50) line(map, Point(0,gy), Point(mapSize.width,gy), Scalar(30,30,30), 1);

        // draw trajectory
        for(size_t i=1;i<trajectory.size();++i){
            Point p1 = worldToPixel(trajectory[i-1], mapSize, meters_per_pixel);
            Point p2 = worldToPixel(trajectory[i], mapSize, meters_per_pixel);
            line(map, p1, p2, Scalar(0,255,0), 2);
        }
        // robot marker
        Point p = worldToPixel(trajectory.back(), mapSize, meters_per_pixel);
        circle(map, p, 6, Scalar(0,0,255), -1);
        putText(map, "Robot", p + Point(10,-10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1);

        imshow("topdown", map);

        // shift buffers: keep current as previous
        prevGray = gray.clone();
        prevKp = kps;
        prevDesc = desc.clone();

        frame_id++;
        char key = (char)waitKey(1);
        if(key == 27) break; // ESC
        file_idx++;
    }

    // Save final trajectory image
    imwrite("trajectory.png", map);
    cout << "Finished. Trajectory saved to trajectory.png" << endl;
    return 0;
}
