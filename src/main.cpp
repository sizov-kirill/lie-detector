// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine gaze_estimation_demo application
* \file gaze_estimation_demo/main.cpp
* \example gaze_estimation_demo/main.cpp
*/
#define GAZE_ESTIMATION_MODEL "../data/gaze-estimation-adas-0002.xml"
#define FACE_DETECTION_MODEL "../data/face-detection-retail-0004.xml"
#define HEAD_POSE_MODEL "../data/head-pose-estimation-adas-0001.xml"
#define FACIAL_LANDMARKS_MODEL "../data/facial-landmarks-35-adas-0002.xml"
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
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
        int countR = 0, countL = 0;
        int countTR = 0, countTL = 0;
        int countBR = 0, countBL = 0;
        int scale = 255;
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
                gazeVectorToGazeAngles(inferenceResult.gazeVector, gazeAngles);
                of << inferenceResult.gazeVector.x << "; " << inferenceResult.gazeVector.y << std::endl;

                if (inferenceResult.gazeVector.y < -0.3 && inferenceResult.gazeVector.x < -0.3) countBR = 255;// cv::min(countBR + 17, 255);
                else countBR = cv::max(countBR - 5, 0);

                if (inferenceResult.gazeVector.y > 0.3 && inferenceResult.gazeVector.x < -0.3) countTR = 255;//cv::min(countTR + 17, 255);
                else countTR = cv::max(countTR - 5, 0);

                if (inferenceResult.gazeVector.y > -0.3 && inferenceResult.gazeVector.y < 0.3 && inferenceResult.gazeVector.x < -0.3) countR = 255;// cv::min(countR + 17, 255);
                else countR = cv::max(countR - 5, 0);

                if (inferenceResult.gazeVector.y < -0.3 && inferenceResult.gazeVector.x > 0.3) countBL = 255;// cv::min(countBL + 17, 255);
                else countBL = cv::max(countBL - 5, 0);

                if (inferenceResult.gazeVector.y > 0.3 && inferenceResult.gazeVector.x > 0.3) countTL = 255;// cv::min(countTL + 17, 255);
                else countTL = cv::max(countTL - 5, 0);

                if (inferenceResult.gazeVector.y > -0.3 && inferenceResult.gazeVector.y < 0.3 && inferenceResult.gazeVector.x > 0.3) countL = 255;// cv::min(countL + 17, 255);
                else countL = cv::max(countL - 5, 0);

            }

            if (countR > 10) std::cout << "right" << std::endl;
            if (countL > 10) std::cout << "left" << std::endl;
            if (countTR > 10) std::cout << "top right" << std::endl;
            if (countBR > 10) std::cout << "bottom right" << std::endl;
            if (countTL > 10) std::cout << "top left" << std::endl;
            if (countBL > 10) std::cout << "bottom left" << std::endl;

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
            if (countR != 0) {
                cv::ellipse(frame, cv::Point(7*width/8, 5*height/6), cv::Size(width/8, height/6), 0.0, 0.0, 30.0, cv::Scalar(countR, 0, 0), -1);
                cv::ellipse(frame, cv::Point(7*width/8, 5*height/6), cv::Size(width/8, height/6), 0.0, 330.0, 360.0, cv::Scalar(countR, 0, 0), -1);
            } else { 
                cv::ellipse(frame, cv::Point(7*width/8, 5*height/6), cv::Size(width/8, height/6), 0.0, 0.0, 30.0, cv::Scalar(countR, 0, 0));
                cv::ellipse(frame, cv::Point(7*width/8, 5*height/6), cv::Size(width/8, height/6), 0.0, 330.0, 360.0, cv::Scalar(countR, 0, 0));
            }

            if (countBR != 0)
                cv::ellipse(frame, cv::Point(7*width/8, 5*height/6), cv::Size(width/8, height/6), 0.0, 30.0, 90.0, cv::Scalar(0, countBR, 0), -1);
            else
                cv::ellipse(frame, cv::Point(7*width/8, 5*height/6), cv::Size(width/8, height/6), 0.0, 30.0, 90.0, cv::Scalar(0, countBR, 0));

            if (countBL != 0)
                cv::ellipse(frame, cv::Point(7*width/8, 5*height/6), cv::Size(width/8, height/6), 0.0, 90.0, 150.0, cv::Scalar(0, 0, countBL), -1);
            else 
                cv::ellipse(frame, cv::Point(7*width/8, 5*height/6), cv::Size(width/8, height/6), 0.0, 90.0, 150.0, cv::Scalar(0, 0, countBL));

            if (countL != 0)
                cv::ellipse(frame, cv::Point(7*width/8, 5*height/6), cv::Size(width/8, height/6), 0.0, 150.0, 210.0, cv::Scalar(0, countL, countL), -1);
            else 
                cv::ellipse(frame, cv::Point(7*width/8, 5*height/6), cv::Size(width/8, height/6), 0.0, 150.0, 210.0, cv::Scalar(0, countL, countL));

            if (countTL != 0)
                cv::ellipse(frame, cv::Point(7*width/8, 5*height/6), cv::Size(width/8, height/6), 0.0, 210.0, 270.0, cv::Scalar(countTL, 0, countTL), -1);
            else 
                cv::ellipse(frame, cv::Point(7*width/8, 5*height/6), cv::Size(width/8, height/6), 0.0, 210.0, 270.0, cv::Scalar(0, countTL, countTL));

            if (countTR != 0)
                cv::ellipse(frame, cv::Point(7*width/8, 5*height/6), cv::Size(width/8, height/6), 0.0, 270.0, 330.0, cv::Scalar(countTR, countTR, 0), -1);
            else 
                cv::ellipse(frame, cv::Point(7*width/8, 5*height/6), cv::Size(width/8, height/6), 0.0, 270.0, 330.0, cv::Scalar(countTR, countTR, 0));
            
            
            
            
            
            char key = static_cast<char>(cv::waitKey(30));
            if (scale == 1) scale = 255;
            if (key == 27)
                break;
            else if (key == 'f') {
                flipImage = !flipImage;
                scale -= 1;
            }
            imshow("frame", frame);
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
