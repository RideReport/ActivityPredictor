#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <vector>

#include "spline.h"

using namespace std;

/**
 * Interpolate a 1-dimensional input function along a new spacing
 *
 * Returns true if successful
 */
bool interpolateLinearRegular(float* inputX, float* inputY, int inputLength, float* outputY, int outputLength, float newSpacing, float initialOffset)
{
    float newX;
    float slope;
    int nextInputIndex, outputIndex;
    for (outputIndex = 0, nextInputIndex = 1; outputIndex < outputLength; ++outputIndex) {
        newX = inputX[0] + initialOffset + newSpacing * outputIndex;
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

bool interpolateSplineRegular(float* inputX, float* inputY, int inputLength, float* outputY, int outputLength, float newSpacing, float initialOffset) {

    vector<double> X(inputX, inputX + inputLength);
    vector<double> Y(inputY, inputY + inputLength);

    tk::spline s;
    s.set_points(X, Y, true);

    int outputIndex;
    for (outputIndex = 0; outputIndex < outputLength; ++outputIndex) {
        outputY[outputIndex] = s(inputX[0] + initialOffset + outputIndex * newSpacing);
    }

    return outputIndex == outputLength;
}

float max(cv::Mat mat)
{
    float max = 0;
    for (int i=0;i<mat.rows;i++)
    {
        float elem = mat.at<float>(i,0);
        if (elem > max) {
            max = elem;
        }
    }

    return max;
}

double maxMean(cv::Mat mat, int windowSize)
{
    if (windowSize>mat.rows) {
        return 0;
    }

    cv::Mat rollingMeans = cv::Mat::zeros(mat.rows - windowSize, 1, CV_32F);

    for (int i=0;i<=(mat.rows - windowSize);i++)
    {
        float sum = 0;
        for (int j=0;j<windowSize;j++) {
            sum += mat.at<float>(i+j,0);
        }
        rollingMeans.at<float>(i,0) = sum/windowSize;
    }

    double min, max;
    cv::minMaxLoc(rollingMeans, &min, &max);

    return max;
}

double skewness(cv::Mat mat)
{
    cv::Scalar skewness,mean,stddev;
    skewness.val[0]=0;
    skewness.val[1]=0;
    skewness.val[2]=0;
    meanStdDev(mat,mean,stddev,cv::Mat());
    int sum0, sum1, sum2;
    float den0=0,den1=0,den2=0;
    int N=mat.rows*mat.cols;

    for (int i=0;i<mat.rows;i++)
    {
        for (int j=0;j<mat.cols;j++)
        {
            sum0=mat.ptr<uchar>(i)[3*j]-mean.val[0];
            sum1=mat.ptr<uchar>(i)[3*j+1]-mean.val[1];
            sum2=mat.ptr<uchar>(i)[3*j+2]-mean.val[2];

            skewness.val[0]+=sum0*sum0*sum0;
            skewness.val[1]+=sum1*sum1*sum1;
            skewness.val[2]+=sum2*sum2*sum2;
            den0+=sum0*sum0;
            den1+=sum1*sum1;
            den2+=sum2*sum2;
        }
    }

    skewness.val[0]=skewness.val[0]*sqrt(N)/(den0*sqrt(den0));
    skewness.val[1]=skewness.val[1]*sqrt(N)/(den1*sqrt(den1));
    skewness.val[2]=skewness.val[2]*sqrt(N)/(den2*sqrt(den2));

    return skewness.val[0];
}

double kurtosis(cv::Mat mat)
{
    cv::Scalar kurt,mean,stddev;
    kurt.val[0]=0;
    kurt.val[1]=0;
    kurt.val[2]=0;
    meanStdDev(mat,mean,stddev,cv::Mat());
    int sum0, sum1, sum2;
    int N=mat.rows*mat.cols;
    float den0=0,den1=0,den2=0;

    for (int i=0;i<mat.rows;i++)
    {
        for (int j=0;j<mat.cols;j++)
        {
            sum0=mat.ptr<uchar>(i)[3*j]-mean.val[0];
            sum1=mat.ptr<uchar>(i)[3*j+1]-mean.val[1];
            sum2=mat.ptr<uchar>(i)[3*j+2]-mean.val[2];

            kurt.val[0]+=sum0*sum0*sum0*sum0;
            kurt.val[1]+=sum1*sum1*sum1*sum1;
            kurt.val[2]+=sum2*sum2*sum2*sum2;
            den0+=sum0*sum0;
            den1+=sum1*sum1;
            den2+=sum2*sum2;
        }
    }

    kurt.val[0]= (kurt.val[0]*N*(N+1)*(N-1)/(den0*den0*(N-2)*(N-3)))-(3*(N-1)*(N-1)/((N-2)*(N-3)));
    kurt.val[1]= (kurt.val[1]*N/(den1*den1))-3;
    kurt.val[2]= (kurt.val[2]*N/(den2*den2))-3;

    return kurt.val[0];
}

/**
 * Compute area under the curve for an evenly spaced vector `y` of length `length`
 *
 * We assume unit steps on the X-axis. Multiply the return value by a scaling
 * factor to convert to real-world measurements.
 */
float trapezoidArea(vector<float>::iterator start, vector<float>::iterator end)
{
    float area = 0.0;
    if (start != end) {
        for (auto it = start + 1; it != end; it++) {
            area += (*it + *(it - 1)) / 2.;
        }
    }
    return area;
}

float percentile(float *input, int length, float percentile)
{
    std::vector<float> sortedInput(length);

    // using default comparison (operator <):
    std::partial_sort_copy (input, input+length, sortedInput.begin(), sortedInput.end());

    return sortedInput[cvFloor(length*percentile)-1];
}
