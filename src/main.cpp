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
        std::string inputCapName = argv[1];
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
                of << gazeAngles.x << "; " << gazeAngles.y << std::endl;

                if (gazeAngles.y < -10 && gazeAngles.x < -20) countBR++;
                else countBR = 0;

                if (gazeAngles.y > 10 && gazeAngles.x < -20) countTR++;
                else countTR = 0;

                if (gazeAngles.y > -10 && gazeAngles.y < 10 && gazeAngles.x < -20) countR++;
                else countR = 0;

                if (gazeAngles.y < -10 && gazeAngles.x > 20) countBL++;
                else countBL = 0;

                if (gazeAngles.y > 10 && gazeAngles.x > 20) countTL++;
                else countTL = 0;

                if (gazeAngles.y > -10 && gazeAngles.y < 10 && gazeAngles.x > 20) countL++;
                else countL = 0;

            }

            if (countR == 10) std::cout << "right" << std::endl;
            else if (countL == 10) std::cout << "left" << std::endl;
            else if (countTR == 10) std::cout << "top right" << std::endl;
            else if (countBR == 10) std::cout << "bottom right" << std::endl;
            else if (countTL == 10) std::cout << "top left" << std::endl;
            else if (countBL == 10) std::cout << "bottom left" << std::endl;

            // Display the results
            for (auto const& inferenceResult : inferenceResults) {
                resultsMarker.mark(frame, inferenceResult);
            }
            putTimingInfoOnFrame(frame, overallTimeAverager.getAveragedValue(),
                                 inferenceTimeAverager.getAveragedValue());

            char key = static_cast<char>(cv::waitKey(30));
            if (key == 27)
                break;
            else if (key == 'f')
                flipImage = !flipImage;
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
