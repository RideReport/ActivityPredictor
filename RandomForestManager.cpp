//
//  RandomForestManager
//  Ride
//
//  Created by William Henderson on 12/4/15.
//  Copyright © 2015 Knock Softwae, Inc. All rights reserved.
//

#include "RandomForestManager.h"
#include "FFTManager.h"

#ifdef __ANDROID__
#include <android/log.h>
#define DEBUG(str) __android_log_print(ANDROID_LOG_VERBOSE, "RandomForestManager", (str))
#else
#include <iostream>
#define DEBUG(str) (cerr << "RandomForestManager DEBUG: " << str << endl)
#endif

#include <algorithm>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>
#include <cmath>

#include "Utility.h"

#ifdef __APPLE__
#define FFT_TYPE_NUMBER 0
#else
#define FFT_TYPE_NUMBER 1
#endif

using namespace cv;
using namespace std;

struct RandomForestManager {
    string modelPath;
    string modelSha256;
    string dataSha256;

    int sampleSize;
    float samplingRateHz;

    int fftIndex_below2_5hz;
    int fftIndex_above2hz;
    int fftIndex_above3_5hz;

    FFTManager *fftManager;


    vector<float> differences;

    cv::Ptr<cv::ml::RTrees> model;
};

RandomForestManager *createRandomForestManagerFromConfiguration(RFConfiguration* config, const char* pathToModelFile) {
    RandomForestManager *r = new RandomForestManager;
    r->sampleSize = config->sampleSize;
    r->samplingRateHz = config->samplingRateHz;
    r->modelSha256 = config->modelSha256;
    r->dataSha256 = config->dataSha256;

    assert(fmod(log2(r->sampleSize), 1.0) == 0.0); // sampleSize must be a power of 2

    r->fftManager = createFFTManager(r->sampleSize);
    if (pathToModelFile != NULL) {
        r->modelPath = string(pathToModelFile);
    }

    float sampleSpacing = 1. / r->samplingRateHz;
    r->fftIndex_below2_5hz = floorf(sampleSpacing * r->sampleSize * 2.5);
    r->fftIndex_above2hz = ceilf(sampleSpacing * r->sampleSize * 2.0);
    r->fftIndex_above3_5hz = ceilf(sampleSpacing * r->sampleSize * 3.5);

    r->differences = vector<float>(r->sampleSize);

    return r;
}

RandomForestManager *createRandomForestManagerFromJsonString(const char* jsonString) {
    RFConfiguration config;
    if (loadConfigurationFromString(&config, jsonString)) {
        return createRandomForestManagerFromConfiguration(&config, NULL);
    }
    else {
        return NULL;
    }
}

RandomForestManager *createRandomForestManagerFromFiles(const char* pathToJson, const char* pathToModelFile)
{
    RFConfiguration config;
    if (loadConfigurationFromJsonFile(&config, pathToJson)) {
        return createRandomForestManagerFromConfiguration(&config, pathToModelFile);
    }
    else {
        return NULL;
    }
}

/**
 * Maximum spacing between accelerometer readings, in seconds
 */
float randomForestGetDesiredSpacing(RandomForestManager *r) {
    return 1. / r->samplingRateHz;
}

/**
 * Minimum length of continuous readings, in seconds
 */
float randomForestGetDesiredDuration(RandomForestManager *r) {
    // Desired duration is the difference between the time of the first
    // and the time of the last
    return (r->sampleSize - 1) / r->samplingRateHz;
}

const char* randomForestGetModelHash(RandomForestManager *r) {
    return r->modelSha256.c_str();
}

const char* randomForestGetDataHash(RandomForestManager *r) {
    return r->dataSha256.c_str();
}

bool randomForestLoadModel(RandomForestManager *r) {
    if (r->modelPath.empty()) {
        return false;
    }
    r->model = cv::ml::RTrees::load<cv::ml::RTrees>(r->modelPath);
    return true;
}

bool randomForestManagerCanPredict(RandomForestManager *r) {
    if (r->model.empty()) {
        DEBUG("model is not loaded");
        return false;
    }

    if (r->model->getVarCount() != RANDOM_FOREST_VECTOR_SIZE) {
        DEBUG("var count does not match");
        return false;
    }
    return true;
}

void deleteRandomForestManager(RandomForestManager *r)
{
    if (r != NULL) {
        deleteFFTManager(r->fftManager);

        if (!r->model.empty()) {
            r->model.release();
        }
        free(r);
    }
}

void calculateFeaturesFromNorms(RandomForestManager *randomForestManager, float* features, float* accelerometerVector) {
    LOCAL_TIMING_START();

    cv::Mat mags = cv::Mat(randomForestManager->sampleSize, 1, CV_32F, accelerometerVector);

    cv::Scalar meanMag,stddevMag;
    meanStdDev(mags,meanMag,stddevMag);

    float *fftOutput = new float[randomForestManager->sampleSize];

    fft(randomForestManager->fftManager, accelerometerVector, randomForestManager->sampleSize, fftOutput);
    float maxPower = dominantPower(fftOutput, randomForestManager->sampleSize);

    int spectrumLength = randomForestManager->sampleSize / 2; // exclude nyquist frequency
    vector<float> spectrum (fftOutput, fftOutput + spectrumLength);
    float fftIntegral = trapezoidArea(spectrum.begin() + 1, spectrum.end()); // exclude DC / 0Hz power

    float fftIntegralBelow2_5hz = trapezoidArea(
        spectrum.begin() + 1, // exclude DC
        spectrum.begin() + randomForestManager->fftIndex_below2_5hz + 1); // include 2.5Hz component

    features[0] = max(mags);
    features[1] = (float)meanMag.val[0];
    features[2] = maxMean(mags, 5);
    features[3] = (float)stddevMag.val[0];
    features[4] = (float)skewness(mags);
    features[5] = (float)kurtosis(mags);
    features[6] = maxPower;
    features[7] = fftIntegral;
    features[8] = fftIntegralBelow2_5hz;
    features[9] = percentile(accelerometerVector, randomForestManager->sampleSize, 0.25);
    features[10] = percentile(accelerometerVector, randomForestManager->sampleSize, 0.5);
    features[11] = percentile(accelerometerVector, randomForestManager->sampleSize, 0.75);
    features[12] = percentile(accelerometerVector, randomForestManager->sampleSize, 0.9);

    LOCAL_TIMING_FINISH("calculateFeaturesFromNorms");
}

void randomForestClassifyFeatures(RandomForestManager *randomForestManager, float* features, float* confidences, int n_classes) {
    cv::Mat featuresMat = cv::Mat(1, RANDOM_FOREST_VECTOR_SIZE, CV_32F, (void*) features);
    if (randomForestManager->model.empty()) {
        return;
    }

    cv::Mat results;

    LOCAL_TIMING_START();
    randomForestManager->model->predictProb(featuresMat, results, cv::ml::DTrees::PREDICT_CONFIDENCE);
    LOCAL_TIMING_FINISH("predictProb");

    for (int i = 0; i < n_classes; ++i) {
        confidences[i] = results.at<float>(i);
    }
}

bool readingIsLess(AccelerometerReading a, AccelerometerReading b) {
    return ((a.t) < (b.t));
}

bool prepareNormsAndSeconds(AccelerometerReading* readings, float* norms, float* seconds, int readingCount) {
    LOCAL_TIMING_START();

    auto readingVector = std::vector<AccelerometerReading>(readings, readings + readingCount);
    std::sort(readings, readings + readingCount, readingIsLess);

    if (readingCount < 4) {
        return false;
    }

    float* norm;
    float* second;
    AccelerometerReading* reading;
    int i;
    double firstReadingT = readings[0].t;
    for (second = seconds, norm = norms, reading = readings, i = 0; i < readingCount; ++i, ++reading, ++norm, ++second) {
        *norm = sqrt(
            (reading->x * reading->x) +
            (reading->y * reading->y) +
            (reading->z * reading->z));
        *second = (float)(reading->t - firstReadingT);
    }

    LOCAL_TIMING_FINISH("prepareNormsAndSeconds");
    return true;
}

bool randomForestClassifyAccelerometerSignal(RandomForestManager *randomForestManager, AccelerometerReading* readings, int readingCount, float* confidences, int n_classes) {
    float* features = new float[RANDOM_FOREST_VECTOR_SIZE];
    bool successful = randomForestPrepareFeaturesFromAccelerometerSignal(randomForestManager, readings, readingCount, features, RANDOM_FOREST_VECTOR_SIZE, 0.f);
    if (successful) {
        randomForestClassifyFeatures(randomForestManager, features, confidences, n_classes);
    }

    delete[] features;
    return successful;
}

bool randomForestPrepareFeaturesFromAccelerometerSignal(RandomForestManager *randomForestManager,
        AccelerometerReading* readings, int readingCount,
        float* features, int feature_count, float offsetSeconds) {
    if (feature_count < RANDOM_FOREST_VECTOR_SIZE) {
        return false;
    }

    float* norms = new float[readingCount];
    float* seconds = new float[readingCount];
    bool successful = false;

    if (prepareNormsAndSeconds(readings, norms, seconds, readingCount)) {
        float* resampledNorms = new float[randomForestManager->sampleSize];
        float newSpacing = 1.f / ((float)randomForestManager->samplingRateHz);

        successful = interpolateSplineRegular(seconds, norms, readingCount, resampledNorms, randomForestManager->sampleSize, newSpacing, offsetSeconds);
        if (successful) {
            calculateFeaturesFromNorms(randomForestManager, features, resampledNorms);
        }

        delete[] resampledNorms;
    }
    delete[] norms;
    delete[] seconds;

    return successful;
}

int randomForestGetClassLabels(RandomForestManager *randomForestManager, int *labels, int n_classes) {
    if (randomForestManager->model.empty()) {
        return 0;
    }

    Mat labelsMat = randomForestManager->model->getClassLabels();
    for (int i = 0; i < n_classes && i < labelsMat.rows; ++i) {
        labels[i] = labelsMat.at<int>(i);
    }
    return labelsMat.rows;
}

int randomForestGetClassCount(RandomForestManager *randomForestManager) {
    if (randomForestManager->model.empty()) {
        return 0;
    }

    Mat labelsMat = randomForestManager->model->getClassLabels();
    return labelsMat.rows;
}
