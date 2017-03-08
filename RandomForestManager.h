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

    struct AccelerometerReading {
        float x;
        float y;
        float z;
        double t; // seconds
    };
    typedef struct AccelerometerReading AccelerometerReading;

    RandomForestManager *createRandomForestManagerFromJsonString(const char* jsonString);
    RandomForestManager *createRandomForestManagerFromFiles(const char* pathToJson);
    bool randomForestLoadModel(RandomForestManager *r, const char* pathToModelFile);
    
    float randomForestGetDesiredDuration(RandomForestManager *r);
    float randomForestGetDesiredSpacing(RandomForestManager *r);
    const char* randomForestGetModelUniqueIdentifier(RandomForestManager *r);

    bool randomForestManagerCanPredict(RandomForestManager *r);
    void deleteRandomForestManager(RandomForestManager *r);
    void randomForestClassifyFeatures(RandomForestManager *randomForestManager, float* features, float* confidences, int n_classes);
    bool randomForestClassifyAccelerometerSignal(RandomForestManager *randomForestManager, AccelerometerReading* signal, int readingCount, float* confidences, int n_classes);
    bool randomForestPrepareFeaturesFromAccelerometerSignal(RandomForestManager *randomForestManager, AccelerometerReading* readings, int readingCount, float* features, int feature_count, float offsetSeconds);
    int randomForestGetClassCount(RandomForestManager *randomForestManager);
    int randomForestGetClassLabels(RandomForestManager *randomForestManager, int *labels, int classCount);
#ifdef __cplusplus
}
#endif
