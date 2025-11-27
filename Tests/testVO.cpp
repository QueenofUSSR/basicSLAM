#include "VisualOdometry.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " --dataset <path_to_images> --calib <calib.yml>" << std::endl;
        return 1;
    }

    std::string datasetDir, calibFile;
    bool enableBackend = false;
    int windowSize = -1;
    int backendType = -1; // 0=g2o, 1=opencv_sfm
    for (int i=1;i<argc;i++) {
        std::string a = argv[i];
        if (a == "--dataset" && i+1<argc) datasetDir = argv[++i];
        else if (a == "--calib" && i+1<argc) calibFile = argv[++i];
        else if (a == "--enable-backend") enableBackend = true;
        else if (a == "--window-size" && i+1<argc) windowSize = std::stoi(argv[++i]);
        else if (a == "--backend-type" && i+1<argc) backendType = std::stoi(argv[++i]);
    }
    if (datasetDir.empty() || calibFile.empty()) {
        std::cerr << "Missing required arguments." << std::endl;
        return 1;
    }

    // Validate paths
    if (!std::filesystem::exists(datasetDir)) {
        std::cerr << "Dataset path not found: " << datasetDir << std::endl;
        return 1;
    }
    if (!std::filesystem::exists(calibFile)) {
        std::cerr << "Calib file not found: " << calibFile << std::endl;
        return 1;
    }

    // Robust calibration loading supporting common schemas
    cv::Mat K, dist;
    auto tryLoadOpenCVFS = [&](cv::Mat &Kout, cv::Mat &dout)->bool{
        try {
            cv::FileStorage fs(calibFile, cv::FileStorage::READ);
            if (!fs.isOpened()) return false;
            if (!fs["K"].empty()) fs["K"] >> Kout; else if (!fs["camera_matrix"].empty()) fs["camera_matrix"] >> Kout;
            if (!fs["dist"].empty()) fs["dist"] >> dout; else if (!fs["distortion_coefficients"].empty()) fs["distortion_coefficients"] >> dout;
            fs.release();
            return !Kout.empty();
        } catch(...) { return false; }
    };
    auto tryLoadIntrinsicsArray = [&](cv::Mat &Kout)->bool{
        // Expect a line like: intrinsics: [fu, fv, cu, cv]
        std::ifstream ifs(calibFile);
        if (!ifs.is_open()) return false;
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.find("intrinsics") != std::string::npos) {
                auto lb = line.find('['); auto rb = line.find(']');
                if (lb == std::string::npos || rb == std::string::npos || rb <= lb) continue;
                std::string arr = line.substr(lb+1, rb-lb-1);
                std::vector<double> vals; vals.reserve(4);
                std::stringstream ss(arr);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    // trim spaces
                    tok.erase(0, tok.find_first_not_of(" \t"));
                    tok.erase(tok.find_last_not_of(" \t")+1);
                    try { vals.push_back(std::stod(tok)); } catch(...) {}
                }
                if (vals.size() == 4) {
                    double fu=vals[0], fv=vals[1], cu=vals[2], cvv=vals[3];
                    Kout = (cv::Mat_<double>(3,3) << fu, 0, cu,
                                                     0, fv, cvv,
                                                     0,  0,   1);
                    return true;
                }
            }
        }
        return false;
    };
    auto tryLoadFxFyCxCy = [&](cv::Mat &Kout)->bool{
        // Parse simple YAML/TOML-like with lines containing fx, fy, cx, cy
        std::ifstream ifs(calibFile);
        if (!ifs.is_open()) return false;
        double fx=-1, fy=-1, cx=-1, cy=-1;
        std::string line;
        while (std::getline(ifs, line)) {
            // remove spaces
            for (auto &ch : line) { if (ch=='\t') ch=' '; }
            // lowercase copy
            std::string l = line;
            // simple extraction
            auto grab = [&](const std::string &key)->double{
                auto pos = l.find(key);
                if (pos == std::string::npos) return -1;
                auto eq = l.find(':', pos);
                if (eq == std::string::npos) eq = l.find('=', pos);
                if (eq == std::string::npos) return -1;
                std::string val = l.substr(eq+1);
                // trim
                val.erase(0, val.find_first_not_of(" \""));
                val.erase(val.find_last_not_of(" \"", val.size()-1)+1);
                try { return std::stod(val); } catch(...) { return -1; }
            };
            if (line.find("fx") != std::string::npos) { double v = grab("fx"); if (v>0) fx=v; }
            if (line.find("fy") != std::string::npos) { double v = grab("fy"); if (v>0) fy=v; }
            if (line.find("cx") != std::string::npos) { double v = grab("cx"); if (v>=0) cx=v; }
            if (line.find("cy") != std::string::npos) { double v = grab("cy"); if (v>=0) cy=v; }
        }
        if (fx>0 && fy>0 && cx>=0 && cy>=0) {
            Kout = (cv::Mat_<double>(3,3) << fx, 0, cx,
                                             0, fy, cy,
                                             0,  0,  1);
            return true;
        }
        return false;
    };
    bool okK = tryLoadOpenCVFS(K, dist) || tryLoadIntrinsicsArray(K) || tryLoadFxFyCxCy(K);
    if (!okK || K.empty() || K.rows!=3 || K.cols!=3) {
        std::cerr << "Failed to parse calibration. Expected OpenCV FileStorage (K/camera_matrix) or fx/fy/cx/cy keys: " << calibFile << std::endl;
        return 1;
    }
    std::cout << "Loaded K:\n" << K << std::endl;
    cv::vo::CameraIntrinsics ci{K, dist};

    auto orb = cv::ORB::create(2000);
    // Use crossCheck=false when using knnMatch
    auto bf = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    cv::vo::VOParams params;
    cv::vo::VisualOdometry vo(orb, bf, ci, params);
    if (enableBackend) vo.setEnableBackend(true);
    if (windowSize > 0) vo.setWindowSize(windowSize);
    if (backendType >= 0) vo.setBackendType(backendType);

    std::vector<std::string> images;
    for (auto& p : std::filesystem::directory_iterator(datasetDir)) {
        if (!p.is_regular_file()) continue;
        auto ext = p.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext==".png" || ext==".jpg" || ext==".jpeg" || ext==".bmp" || ext==".tiff") {
            images.push_back(p.path().string());
        }
    }
    std::sort(images.begin(), images.end());
    if (images.size() < 2) {
        std::cerr << "Dataset must have at least 2 images. Found " << images.size() << std::endl;
        return 1;
    }
    std::cout << "Found " << images.size() << " images in " << datasetDir << std::endl;

    int okCount = 0;
    std::vector<cv::Mat> traj;
    for (size_t i=0;i<images.size();++i) {
        cv::Mat img = cv::imread(images[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) { std::cerr << "Failed to read: " << images[i] << std::endl; continue; }
        bool ok = vo.processFrame(img, static_cast<double>(i));
        if (ok) okCount++;
        cv::Mat T = vo.getCurrentPose();
        traj.push_back(T);
        std::cout << "Frame " << i << " pose t = " << T.at<double>(0,3) << ", " << T.at<double>(1,3) << ", " << T.at<double>(2,3) << std::endl;
    }

    std::cout << "Tracking OK frames: " << okCount << " / " << images.size() << std::endl;

    // Save trajectory CSV: frame, tx, ty, tz
    std::string outCsv = "trajectory.csv";
    std::ofstream ofs(outCsv);
    if (ofs) {
        ofs << "frame,tx,ty,tz\n";
        for (size_t i=0;i<traj.size();++i) {
            const cv::Mat &T = traj[i];
            ofs << i << "," << T.at<double>(0,3) << "," << T.at<double>(1,3) << "," << T.at<double>(2,3) << "\n";
        }
        ofs.close();
        std::cout << "Saved trajectory to " << outCsv << std::endl;
    } else {
        std::cerr << "Failed to write trajectory.csv" << std::endl;
    }
    return 0;
}
