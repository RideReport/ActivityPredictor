//
//  RandomForestManager.h
//  Ride
//
//  Created by William Henderson on 12/4/15.
//  Copyright © 2015 Knock Softwae, Inc. All rights reserved.
//

#define RANDOM_FOREST_VECTOR_SIZE (13)
#define RANDOM_FOREST_SAMPLING_RATE_HZ 20f
#ifdef __cplusplus
extern "C" {
#endif
    typedef struct RandomForestManager RandomForestManager;
    RandomForestManager *createRandomForestManager(int sampleSize, int samplingRateHz, const char* pathToModelFile);
    bool randomForestManagerCanPredict(RandomForestManager *r);
    void deleteRandomForestManager(RandomForestManager *r);
	void prepFeatureVector(RandomForestManager *randomForestManager, float* features, float* accelerometerVector);
    void randomForestClassifyFeatures(RandomForestManager *randomForestManager, float* features, float* confidences, int n_classes);
	void randomForestClassifyMagnitudeVector(RandomForestManager *randomForestManager, float* accelerometerVector, float *confidences, int n_classes);
    int randomForestGetClassCount(RandomForestManager *randomForestManager);
    int randomForestGetClassLabels(RandomForestManager *randomForestManager, int *labels, int classCount);
#ifdef __cplusplus
}
#endif
