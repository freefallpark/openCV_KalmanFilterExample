#include <iostream>
#include "matplotlibcpp.h"
#include <fstream>
#include <opencv2/opencv.hpp>

namespace plt = matplotlibcpp;


int main(){
    //init KF
    cv::KalmanFilter kf(4,2,0);
    // Because we're only Measureing x and y positins, and not x and y velocities, 2, 2 is ok
    kf.transitionMatrix = (cv::Mat_<float>(4,4) <<  1, 0, 0, 0,
                                                    0, 1, 0, 0,
                                                    0, 0, 1, 0,
                                                    0, 0, 0, 1);

    kf.measurementMatrix = (cv::Mat_<float>(2,4) << 1, 0, 0, 0,
                                                    0, 1, 0, 0);

    //Tuneable Parameters:
    float processCovariance = 0.001;
    float measurementCovariance = 10;
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(processCovariance));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(measurementCovariance));


    // Import Fake y-axis data, create artificial noise and just step x axis in increments of 1
    std::ifstream file("../fakeData.csv");
    std::vector<cv::Point2f> noisySignal;
    std::string line;
    float x = 0.0;
    while (std::getline(file, line)) {
        //Create Fake noise (data wasn't 'noisy' enough):
        float rnum = rand()&10;
        rnum /=10;

        //load Data in to cv::Point2f
        auto sig = (float)std::stod(line);
        cv::Point2f pt = cv::Point2f(x,sig+rnum);

        //Push to Vector
        noisySignal.push_back(pt);

        //Step x axis
        x += 1;
    }

    //Filter Noisy Data:
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
