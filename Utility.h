//
//  Utility.h
//  Ride
//
//  Created by William Henderson on 3/1/17.
//  Copyright © 2017 Knock Softwae, Inc. All rights reserved.
//

#ifndef Utility_h
#define Utility_h
#include <opencv2/core/core.hpp>
#include <vector>
#include "json/json.h"

using namespace std;

struct RFConfiguration {
    int sampleSize;
    float samplingRateHz;

    string modelSha256;
    string dataSha256;
};
bool loadConfigurationFromString(RFConfiguration* config, const char* jsonString);
bool loadConfigurationFromJsonFile(RFConfiguration* config, const char* pathToJson);

bool interpolateSplineRegular(float* inputX, float* inputY, int inputLength, float* outputY, int outputLength, float newSpacing, float initialOffset);
float max(cv::Mat mat);
double maxMean(cv::Mat mat, int windowSize);
double skewness(cv::Mat mat);
double kurtosis(cv::Mat mat);
float trapezoidArea(vector<float>::iterator start, vector<float>::iterator end);
float percentile(float *input, int length, float percentile);
void loadJson(char* pathToJson);

#endif /* Utility_h */
