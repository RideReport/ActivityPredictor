//
//  FFTManager.c
//  Ride
//
//  Created by William Henderson on 3/7/16.
//  Copyright © 2016 Knock Softwae, Inc. All rights reserved.
//

#include "FFTManager.h"
#include<stdio.h>
#include <Accelerate/Accelerate.h>
#include <math.h>

struct FFTManager {
    FFTSetup fftWeights;
    float* multipliers;
};

void setupHammingWindow(float *values, int N) {
    // The vdsp hamming window has the wrong divisor in the cosine argument, so
    // we have to make our own hamming window.
    for (int i = 0; i < N; ++i) {
        values[i] = 0.54f - 0.46f * cosf(2.f*M_PI*i/(N-1));
    }
}

FFTManager *createFFTManager(int sampleSize)
{
//    assert(fmod(log2(sampleSize), 1.0) == 0.0); // sampleSize must be a power of 2
    
    struct FFTManager *f;
    f = (struct FFTManager*) malloc(sizeof(struct FFTManager));
    f->fftWeights = vDSP_create_fftsetup(vDSP_Length(log2f(sampleSize)), FFT_RADIX2);
    f->multipliers = (float*) malloc(sizeof(float) * sampleSize);
    setupHammingWindow(f->multipliers, sampleSize);
    
    return f;
}

void deleteFFTManager(FFTManager *fftManager)
{
    vDSP_destroy_fftsetup(fftManager->fftWeights);
    free(fftManager);
}

void fft(FFTManager *manager, float * input, int inputSize, float *output)
{
    float *hammedInput = new float[inputSize]();
    vDSP_vmul(input, 1, manager->multipliers, 1, hammedInput, 1, inputSize);
    
    // pack the input samples in preparation for FFT
    float *zeroArray = new float[inputSize]();
    DSPSplitComplex splitComplex = {.realp = hammedInput, .imagp =  zeroArray};
    
    // run the FFT and get the magnitude components (vDSP_zvmags returns squared components)
    vDSP_fft_zip(manager->fftWeights, &splitComplex, 1, log2f(inputSize), FFT_FORWARD);
    vDSP_zvmags(&splitComplex, 1, output, 1, inputSize);
    
    delete[] zeroArray;
    delete[] hammedInput;
}

void autocorrelation(float *input, int inputSize, float *output)
{
    int lenSignal = 2 * inputSize - 1;
    float *signal = new float[lenSignal];
    
    for (int i = 0; i < inputSize; i++) {
        if (i < inputSize) {
            signal[i] = input[i];
        } else {
            signal[i] = 0;
        }
    }
    
//    float *result = new float[inputSize];
//    vDSP_conv(signal, 1, &input[inputSize - 1], -1, result, 1, inputSize, inputSize);
    vDSP_conv(signal, 1, &input[inputSize - 1], -1, output, 1, inputSize, inputSize);
    
    delete[](signal);
//    free(result);

//    return 0.0;
}

float dominantPower(float *input, int inputSize)
{
    float dominantPower = 0;
    for (int i=1; i<=inputSize/2; i+=1) {
        float value = input[i];
        if (value > dominantPower) {
            dominantPower = value;
        }
    }
    
    return dominantPower;
}
