#include <opencv2/opencv.hpp>
#include <iostream>
#include "Initializer.hpp"
#include "MapManager.hpp"
#include "Optimizer.hpp"
#include "KeyFrame.hpp"
#include "MapPoint.hpp"

int main() {
    std::cout << "=== Testing P0 Features ===" << std::endl;
    
    // Test 1: MapPoint quality management
    std::cout << "\n--- Test 1: MapPoint Quality Management ---" << std::endl;
    {
        MapManager map;
        
        // Create some test map points
        for(int i = 0; i < 10; ++i) {
            MapPoint mp(cv::Point3d(i, i*0.5, i*2));
            mp.nObs = (i % 3); // Vary observation count
            mp.observations.push_back({0, i});
            if(i % 3 == 0) {
                mp.observations.push_back({1, i});
            }
            map.addMapPoints({mp});
        }
        
        std::cout << "Created " << map.mappoints().size() << " map points" << std::endl;
        std::cout << "Good map points before culling: " << map.countGoodMapPoints() << std::endl;
        
        // Cull bad points
        map.cullBadMapPoints();
        
        std::cout << "Total map points after culling: " << map.mappoints().size() << std::endl;
        std::cout << "Good map points after culling: " << map.countGoodMapPoints() << std::endl;
    }
    
    // Test 2: Initializer
    std::cout << "\n--- Test 2: Initializer ---" << std::endl;
    {
        // Create synthetic matched points with known transformation
        std::vector<cv::KeyPoint> kps1, kps2;
        std::vector<cv::DMatch> matches;
        
        // Generate points on a virtual plane with some depth variation
        cv::Mat R_true = (cv::Mat_<double>(3,3) << 
            0.9, -0.1, 0.1,
            0.1,  0.95, -0.05,
            -0.1, 0.05, 0.98);
        cv::Mat t_true = (cv::Mat_<double>(3,1) << 0.1, 0.05, 0.2);
        
        double fx = 500, fy = 500, cx = 320, cy = 240;
        
        for(int i = 0; i < 100; ++i) {
            // Random 3D point
            cv::Point3d pt3d(
                (rand() % 200 - 100) / 100.0,
                (rand() % 200 - 100) / 100.0,
                2.0 + (rand() % 100) / 100.0
            );
            
            // Project to first camera
            double u1 = fx * pt3d.x / pt3d.z + cx;
            double v1 = fy * pt3d.y / pt3d.z + cy;
            kps1.push_back(cv::KeyPoint(u1, v1, 7));
            
            // Transform and project to second camera
            cv::Mat p3d = (cv::Mat_<double>(3,1) << pt3d.x, pt3d.y, pt3d.z);
            cv::Mat p3d2 = R_true * p3d + t_true;
            double u2 = fx * p3d2.at<double>(0,0) / p3d2.at<double>(2,0) + cx;
            double v2 = fy * p3d2.at<double>(1,0) / p3d2.at<double>(2,0) + cy;
            kps2.push_back(cv::KeyPoint(u2, v2, 7));
            
            matches.push_back(cv::DMatch(i, i, 0));
        }
        
        std::cout << "Created " << matches.size() << " synthetic matches" << std::endl;
        
        // Check parallax
        bool hasParallax = Initializer::checkParallax(kps1, kps2, matches, 10.0);
        std::cout << "Has sufficient parallax: " << (hasParallax ? "YES" : "NO") << std::endl;
        
        // Attempt initialization
        Initializer initializer;
        cv::Mat R, t;
        std::vector<cv::Point3d> points3D;
        std::vector<bool> isTriangulated;
        
        bool success = initializer.initialize(kps1, kps2, matches, fx, fy, cx, cy,
                                             R, t, points3D, isTriangulated);
        
        if(success) {
            int nTriangulated = 0;
            for(bool b : isTriangulated) if(b) nTriangulated++;
            
            std::cout << "Initialization SUCCESS!" << std::endl;
            std::cout << "Triangulated " << nTriangulated << " points" << std::endl;
            std::cout << "Estimated rotation:\n" << R << std::endl;
            std::cout << "Estimated translation:\n" << t << std::endl;
            std::cout << "Ground truth rotation:\n" << R_true << std::endl;
            std::cout << "Ground truth translation:\n" << t_true << std::endl;
        } else {
            std::cout << "Initialization FAILED" << std::endl;
        }
    }
    
    // Test 3: Optimizer (basic test)
    std::cout << "\n--- Test 3: Optimizer ---" << std::endl;
    {
        MapManager map;
        
        // Create a simple map with 2 keyframes and some map points
        KeyFrame kf1, kf2;
        kf1.id = 0;
        kf1.R_w = cv::Mat::eye(3, 3, CV_64F);
        kf1.t_w = cv::Mat::zeros(3, 1, CV_64F);
        
        kf2.id = 1;
        kf2.R_w = cv::Mat::eye(3, 3, CV_64F);
        kf2.t_w = (cv::Mat_<double>(3,1) << 0.1, 0, 0);
        
        // Add some keypoints
        for(int i = 0; i < 10; ++i) {
            kf1.kps.push_back(cv::KeyPoint(320 + i*10, 240 + i*5, 7));
            kf2.kps.push_back(cv::KeyPoint(325 + i*10, 242 + i*5, 7));
        }
        
        map.addKeyFrame(kf1);
        map.addKeyFrame(kf2);
        
        // Add map points
        for(int i = 0; i < 10; ++i) {
            MapPoint mp(cv::Point3d(i*0.1, i*0.05, 2.0));
            mp.observations.push_back({0, i});
            mp.observations.push_back({1, i});
            map.addMapPoints({mp});
        }
        
        std::cout << "Created test map with " << map.keyframes().size() 
                  << " keyframes and " << map.mappoints().size() << " map points" << std::endl;
        
        // Run local BA
        std::vector<int> localKfIndices = {1};
        std::vector<int> fixedKfIndices = {0};
        
        auto kfs = map.keyframes();
        auto mps = map.mappoints();
        
        Optimizer::localBundleAdjustment(
            const_cast<std::vector<KeyFrame>&>(kfs),
            const_cast<std::vector<MapPoint>&>(mps),
            localKfIndices, fixedKfIndices,
            500, 500, 320, 240,
            5  // iterations
        );
        
        std::cout << "Local BA completed" << std::endl;
    }
    
    std::cout << "\n=== All P0 tests completed ===" << std::endl;
    std::cout << "\nNext steps:" << std::endl;
    std::cout << "1. Integrate Initializer into Controller" << std::endl;
    std::cout << "2. Call cullBadMapPoints() periodically" << std::endl;
    std::cout << "3. Trigger Local BA after keyframe insertion" << std::endl;
    std::cout << "4. For production: integrate g2o for better BA performance" << std::endl;
    
    return 0;
}
