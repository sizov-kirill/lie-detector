// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine gaze_estimation_demo application
* \file gaze_estimation_demo/main.cpp
* \example gaze_estimation_demo/main.cpp
*/
#define _USE_MATH_DEFINES
#define GAZE_ESTIMATION_MODEL "../data/gaze-estimation-adas-0002.xml"
#define FACE_DETECTION_MODEL "../data/face-detection-retail-0004.xml"
#define HEAD_POSE_MODEL "../data/head-pose-estimation-adas-0001.xml"
#define FACIAL_LANDMARKS_MODEL "../data/facial-landmarks-35-adas-0002.xml"
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>
#include <sstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>

#include <inference_engine.hpp>
#include "face_inference_results.hpp"
#include "face_detector.hpp"
#include "base_estimator.hpp"
#include "head_pose_estimator.hpp"
#include "landmarks_estimator.hpp"
#include "gaze_estimator.hpp"
#include "results_marker.hpp"
#include "exponential_averager.hpp"
#include "utils.hpp"
//#include "samples/presenter.h"
#include <ie_iextension.h>

using namespace InferenceEngine;
using namespace gaze_estimation;

int main(int argc, char *argv[]) {
    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;
        cv::VideoCapture cap;
        std::string inputCapName;
        if (argc > 1)
            inputCapName = argv[1];
        else 
            inputCapName = "cam";
        if (!(inputCapName == "cam" ? cap.open(0) : cap.open(inputCapName))) {
            throw std::logic_error("Cannot open input file or camera: " + inputCapName);
        }
        cv::Mat frame;
        if (!cap.read(frame)) {
            throw std::logic_error("Failed to get frame from cv::VideoCapture");
        }

        ResultsMarker resultsMarker(false, false, false, true);

        std::vector<std::pair<std::string, std::string>> options = {
                    {"CPU", GAZE_ESTIMATION_MODEL}, 
                    {"CPU", FACE_DETECTION_MODEL},
                    {"CPU", HEAD_POSE_MODEL}, 
                    {"CPU", FACIAL_LANDMARKS_MODEL}
                };

        InferenceEngine::Core ie;
        initializeIEObject(ie, options);

        FaceDetector faceDetector(ie, FACE_DETECTION_MODEL, "CPU", 0.5, false);
        HeadPoseEstimator headPoseEstimator(ie, HEAD_POSE_MODEL, "CPU");
        LandmarksEstimator landmarksEstimator(ie, FACIAL_LANDMARKS_MODEL, "CPU");
        GazeEstimator gazeEstimator(ie, GAZE_ESTIMATION_MODEL, "CPU");

        BaseEstimator* estimators[] = {&headPoseEstimator, &landmarksEstimator, &gazeEstimator};
        std::vector<FaceInferenceResults> inferenceResults;
        double smoothingFactor = 0.1;
        ExponentialAverager overallTimeAverager(smoothingFactor, 30.);
        ExponentialAverager inferenceTimeAverager(smoothingFactor, 30.);
        bool flipImage = false;

        std::ofstream of;
        of.open("results.csv");

        int clrR[3] = {0, 0, 0};
        int clrBR[3] = {0, 0, 0};
        int clrTR[3] = {0, 0, 0};
        int clrL[3] = {0, 0, 0};
        int clrBL[3] = {0, 0, 0};
        int clrTL[3] = {0, 0, 0};
        
        do {
            if (flipImage) {
                cv::flip(frame, frame, 1);
            }

            auto tInferenceBegins = cv::getTickCount();
            auto inferenceResults = faceDetector.detect(frame);
            for (auto& inferenceResult : inferenceResults) {
                for (auto estimator : estimators) {
                    estimator->estimate(frame, inferenceResult);
                }
            }
            auto tInferenceEnds = cv::getTickCount();

            for (auto& inferenceResult : inferenceResults) {
                cv::Point2f gazeAngles;
                cv::Point3f qqq;
                float yaw = inferenceResult.headPoseAngles.x;
                qqq.x = inferenceResult.gazeVector.x * cos(yaw*M_PI/180.0) + inferenceResult.gazeVector.z * sin(yaw*M_PI/180.0);
                qqq.y = inferenceResult.gazeVector.y;
                qqq.z = -inferenceResult.gazeVector.x * sin(yaw*M_PI/180.0) + inferenceResult.gazeVector.z * cos(yaw*M_PI/180.0);
                gazeVectorToGazeAngles(qqq, gazeAngles);
                of << gazeAngles.x << ";" << gazeAngles.y << std::endl;
                std::cout << "yaw = " << yaw << std::endl;

                if (gazeAngles.x > 20 && gazeAngles.y < -15) {
                    clrBL[0] = 255;
                    clrBL[1] = 170;
                } else {
                    clrBL[0] = cv::max(clrBL[0] - 5, 0);
                    clrBL[1] = cv::max(clrBL[1] - 5, 0);
                }

                if (gazeAngles.x > 20 && gazeAngles.y > 15) {
                    clrTL[0] = 150;
                    clrTL[2] = 150;
                } else {
                    clrTL[0] = cv::max(clrTL[0] - 5, 0);
                    clrTL[2] = cv::max(clrTL[2] - 5, 0);
                }

                if (gazeAngles.x > 20 && gazeAngles.y < 15 && gazeAngles.y > -15) { 
                    clrL[0] = 255;
                } else {
                    clrL[0] = cv::max(clrL[0] - 5, 0);
                }

                if (gazeAngles.x < -20 && gazeAngles.y < -15) {
                    clrBR[1] = 255;
                    clrBR[2] = 255;
                } else {
                    clrBR[1] = cv::max(clrBR[1] - 5, 0);
                    clrBR[2] = cv::max(clrBR[2] - 5, 0);
                }

                if (gazeAngles.x < -20 && gazeAngles.y > 15) {
                    clrTR[1] = 255;
                    clrTR[2] = 150;
                } else {
                    clrTR[1] = cv::max(clrTR[1] - 5, 0);
                    clrTR[2] = cv::max(clrTR[2] - 5, 0);
                }

                if (gazeAngles.x < -20 && gazeAngles.y > -15 && gazeAngles.y < 15) {
                    clrR[2] = 255;
                } else {
                    clrR[2] = cv::max(clrR[2] - 5, 0);
                }

            }

            // Display the results

            for (auto const& inferenceResult : inferenceResults) {
                resultsMarker.mark(frame, inferenceResult);
            }
            putTimingInfoOnFrame(frame, overallTimeAverager.getAveragedValue(),
                                 inferenceTimeAverager.getAveragedValue());

            int height = frame.size().height;
            int width = frame.size().width;
            cv::Rect gazeSubframe(width - width/4, height - height/3, width/4, height/3);
            cv::Mat roiSubframe = frame(gazeSubframe);
            //roiSubframe.setTo(cv::Scalar(120, 120, 120));
            if (clrL[0] != 0) {
                //cv::putText(frame, "remebered", cv::Point(20*width/28, 6*height/14), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255));
                cv::ellipse(frame, cv::Point(7*width/8, 4*height/6), cv::Size(height/6, width/8), 0.0, 0.0, 30.0, cv::Scalar(clrL[0], clrL[1], clrL[2]), -1);
                cv::ellipse(frame, cv::Point(7*width/8, 4*height/6), cv::Size(height/6, width/8), 0.0, 330.0, 360.0, cv::Scalar(clrL[0], clrL[1], clrL[2]), -1);
            } else { 
                cv::ellipse(frame, cv::Point(7*width/8, 4*height/6), cv::Size(height/6, width/8), 0.0, 0.0, 30.0, cv::Scalar(clrL[0], clrL[1], clrL[2]));
                cv::ellipse(frame, cv::Point(7*width/8, 4*height/6), cv::Size(height/6, width/8), 0.0, 330.0, 360.0, cv::Scalar(clrL[0], clrL[1], clrL[2]));
            }

            if (clrBL[0] != 0) {
                //cv::putText(frame, "internal dialogue", cv::Point(20*width/28, 6*height/14), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255));
                cv::ellipse(frame, cv::Point(7*width/8, 4*height/6), cv::Size(height/6, width/8), 0.0, 30.0, 90.0, cv::Scalar(clrBL[0], clrBL[1], clrBL[2]), -1);
            } else {
                cv::ellipse(frame, cv::Point(7*width/8, 4*height/6), cv::Size(height/6, width/8), 0.0, 30.0, 90.0, cv::Scalar(clrBL[0], clrBL[1], clrBL[2]));
            }

            if (clrBR[1] != 0) {
                //cv::putText(frame, "feelings", cv::Point(20*width/28, 6*height/14), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255));
                cv::ellipse(frame, cv::Point(7*width/8, 4*height/6), cv::Size(height/6, width/8), 0.0, 90.0, 150.0, cv::Scalar(clrBR[0], clrBR[1], clrBR[2]), -1);
            } else { 
                cv::ellipse(frame, cv::Point(7*width/8, 4*height/6), cv::Size(height/6, width/8), 0.0, 90.0, 150.0, cv::Scalar(clrBR[0], clrBR[1], clrBR[2]));
            }

            if (clrR[2] != 0) {
                //cv::putText(frame, "construct", cv::Point(20*width/28, 6*height/14), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255));
                cv::ellipse(frame, cv::Point(7*width/8, 4*height/6), cv::Size(height/6, width/8), 0.0, 150.0, 210.0, cv::Scalar(clrR[0], clrR[1], clrR[2]), -1);
            } else { 
                cv::ellipse(frame, cv::Point(7*width/8, 4*height/6), cv::Size(height/6, width/8), 0.0, 150.0, 210.0, cv::Scalar(clrR[0], clrR[1], clrR[2]));
            }

            if (clrTR[1] != 0) {
                //cv::putText(frame, "construct", cv::Point(20*width/28, 6*height/14), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255));
                cv::ellipse(frame, cv::Point(7*width/8, 4*height/6), cv::Size(height/6, width/8), 0.0, 210.0, 270.0, cv::Scalar(clrTR[0], clrTR[1], clrTR[2]), -1);
            } else { 
                cv::ellipse(frame, cv::Point(7*width/8, 4*height/6), cv::Size(height/6, width/8), 0.0, 210.0, 270.0, cv::Scalar(clrTR[0], clrTR[1], clrTR[2]));
            }

            if (clrTL[0] != 0) {
                //cv::putText(frame, "remebered", cv::Point(20*width/28, 6*height/14), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255));
                cv::ellipse(frame, cv::Point(7*width/8, 4*height/6), cv::Size(height/6, width/8), 0.0, 270.0, 330.0, cv::Scalar(clrTL[0], clrTL[1], clrTL[2]), -1);
            } else { 
                cv::ellipse(frame, cv::Point(7*width/8, 4*height/6), cv::Size(height/6, width/8), 0.0, 270.0, 330.0, cv::Scalar(clrTL[0], clrTL[1], clrTL[2]));
            }
            
            
            char key = static_cast<char>(cv::waitKey(5));
            if (key == 27)
                break;
            else if (key == 'f') {
                int count = 0;
                while (count != 20) {
                    cap.read(frame);
                    count++;
                }
                //flipImage = !flipImage;
            }
            imshow("Gaze stimator", frame);
        } while (cap.read(frame));   
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }
    return 0;
}
