#include <iostream>
#include "matplotlibcpp.h"
#include <fstream>
#include <opencv2/opencv.hpp>

namespace plt = matplotlibcpp;

/*
// Kalman Filter 'memory' variables
double x_hat = 0;   // predicted state estimate
double P = 1;       // predicted error covariance
//Tunable Parameters
double Q = 0.001;   // process noise covariance
double R = 10;      // measurement noise covariance
*/


int main()
{
    //init KF
    cv::KalmanFilter kf(2,2,0);
    // Define transition matrix, measurement matrix, process noise, and measurement noise
    kf.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1); // Identity matrix
    kf.measurementMatrix = (cv::Mat_<float>(2, 2) << 1, 0, 0, 1); // Identity matrix
    kf.processNoiseCov = (cv::Mat_<float>(2, 2) << 0.001, 0, 0, 0.001); // Process noise covariance
    kf.measurementNoiseCov = (cv::Mat_<float>(2, 2) << 10, 0, 0, 10); // Measurement noise covariance


    // Read noisy input signal from CSV file
    std::ifstream file("../xs_kyle.csv");
    std::vector<cv::Point2f> noisySignal;
    std::string line;
    float x = 0.0;
    while (std::getline(file, line)) {
        cv::Point2f pt;
        float rnum = rand()&10;
        rnum /=10;
        auto sig = (float)std::stod(line);
        pt = cv::Point2f(x,sig+rnum);
        noisySignal.push_back(pt);
        x += 1;
    }

    //Filter:
    std::vector<cv::Point2f> filtered_signal;
    for(const cv::Point2f& measurement:noisySignal){
        cv::Mat prediction  = kf.predict();
        cv::Mat corrected   = kf.correct(cv::Mat(measurement));
        filtered_signal.emplace_back(corrected.at<float>(0),corrected.at<float>(1));
    }
    // Plot the noisy and filtered signals
    std::vector<float> noisy_x, noisy_y, filt_x,filt_y;
    for(const cv::Point2f& point : noisySignal){
        noisy_x.push_back(point.x);
        noisy_y.push_back(point.y);
    }
    for(const cv::Point2f& point : filtered_signal){
        filt_x.push_back(point.x);
        filt_y.push_back(point.y);
    }
    plt::plot(noisy_x,noisy_y);
    plt::plot(filt_x,filt_y);
    plt::xlabel("Time (ms)");
    plt::ylabel("Roll Value (deg)");
    plt::title("Kalman Filter Example");
    plt::show();

    return 0;
}
