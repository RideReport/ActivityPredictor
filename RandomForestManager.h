//
//  RandomForestManager.h
//  Ride
//
//  Created by William Henderson on 12/4/15.
//  Copyright © 2015 Knock Softwae, Inc. All rights reserved.
//

#define RANDOM_FOREST_VECTOR_SIZE (23)
#define RANDOM_FOREST_VECTOR_SIZE_ACCELEROMETER_ONLY (15)
#define RANDOM_FOREST_SAMPLING_RATE_HZ 20f
#ifdef __cplusplus
extern "C" {
#endif
    typedef struct RandomForestManager RandomForestManager;
    RandomForestManager *createRandomForestManager(int sampleSize, int samplingRateHz, const char* pathToModelFile, bool isAcclereomterOnlyVersion);
    void deleteRandomForestManager(RandomForestManager *r);
    float dominantPowerOfFFT(RandomForestManager *randomForestManager, float * input, int inputSize, int managerType);
    float percentile(float *input, int length, float percentile);
    void prepFeatureVector(RandomForestManager *randomForestManager, float* features, float* accelerometerVector, float* gyroscopeVector);
	void prepFeatureVectorAccelerometerOnly(RandomForestManager *randomForestManager, float* features, float* accelerometerVector);
    void randomForestClassificationConfidences(RandomForestManager *randomForestManager, float* accelerometerVector, float* gyroscopeVector, float *confidences, int classCount);
	void randomForestClassificationConfidencesAccelerometerOnly(RandomForestManager *randomForestManager, float* accelerometerVector, float *confidences, int n_classes);
    int randomForestGetClassCount(RandomForestManager *randomForestManager);
    int randomForestGetClassLabels(RandomForestManager *randomForestManager, int *labels, int classCount);
#ifdef __cplusplus
}
#endif
