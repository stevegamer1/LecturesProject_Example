#ifndef CONVNET_H
#define CONVNET_H

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "Preprocessing.h"

class ConvNet {
public:
    ConvNet(int inputWidth, int inputHeight, int kernelWidth, int kernelHeight, int numKernels, int fcOutputSize);
    ~ConvNet();

    void loadWeights(const std::string& kernelFile, const std::string& fcWeightFile, const std::string& fcBiasFile);
    void predict(const std::vector<float>& input, std::vector<float>& output);

    std::vector<float> preprocessImage(const cv::Mat& img);

private:
    int inputWidth, inputHeight;
    int kernelWidth, kernelHeight;
    int outputWidth, outputHeight;
    int numKernels;
    int fcInputSize, fcOutputSize;

    Preprocessor processor;

    std::vector<float> kernels, fcWeights, fcBias;
    float *d_input, *d_kernels, *d_convOutput, *d_fcWeights, *d_fcBias, *d_finalOutput;

    void readKernelWeights(const std::string& filename, std::vector<float>& kernels, int kernelSize);
    void readFullyConnectedWeights(const std::string& weightFile, const std::string& biasFile,
                                   std::vector<float>& fcWeights, std::vector<float>& fcBias, 
                                   int fcWeightSize, int fcBiasSize);
    void convNet(float* input, float* kernels, float* convOutput, float* fcWeights, float* fcBias, float* finalOutput);
};

#endif // CONVNET_H
