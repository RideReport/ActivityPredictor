#include <malloc.h>
#include "FFTManager_opencv.h"
#include <math.h>

struct FFTManager {
    unsigned int N;
    float* multipliers;
};

void setupHammingWindow(float *values, int N) {
    for (int i = 0; i < N; ++i) {
        values[i] = 0.54 - 0.46 * cos(2*M_PI*i/(N-1));
    }
}

FFTManager* createFFTManager(int sampleSize) {
    FFTManager* _fft = (struct FFTManager*) malloc(sizeof(struct FFTManager));
    _fft->multipliers = (float*) malloc(sizeof(float) * sampleSize);
    setupHammingWindow(_fft->multipliers, sampleSize);

    return _fft;
}

void fft(FFTManager *_fft, float * input, int inputSize, float *output) {
    // FIXME
    for (int i = 0; i <= _fft->N/2; ++i) {
        output[i] = 0.f;
    }
}

void deleteFFTManager(FFTManager *_fft) {
    free(_fft->multipliers);
    free(_fft);
}

float dominantPower(float *output, int inputSize) {
    float max = 0.0;
    for (int i = 1; i <= inputSize/2; ++i) {
        if (output[i] > max) {
            max = output[i];
        }
    }

    return max;
}
