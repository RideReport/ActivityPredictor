#include <stdio.h>

/**
 * Interpolate a 1-dimensional input function along a new spacing
 *
 * Returns true if successful
 */
bool interpolateRegular(float* inputX, float* inputY, int inputLength, float* outputY, int outputLength, float newSpacing)
{
    float newX = inputX[0];
    float slope;
    int nextInputIndex, outputIndex;
    for (outputIndex = 0, nextInputIndex = 1; outputIndex < outputLength; ++outputIndex) {
        newX = inputX[0] + newSpacing * outputIndex;
        while (newX > inputX[nextInputIndex]) {
            nextInputIndex += 1;
            if (nextInputIndex >= inputLength) {
                break;
            }
        }
        slope = (inputY[nextInputIndex] - inputY[nextInputIndex-1]) / (inputX[nextInputIndex] - inputX[nextInputIndex-1]);
        outputY[outputIndex] = slope * (newX - inputX[nextInputIndex-1]) + inputY[nextInputIndex-1];
    }
    return outputIndex == outputLength;
}
