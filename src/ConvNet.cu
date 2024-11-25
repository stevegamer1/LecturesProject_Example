#include "ConvNet.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

// CUDA kernels
__global__ void conv2d(float* input, float* kernels, float* output, int inputWidth, int inputHeight, 
                       int kernelWidth, int kernelHeight, int outputWidth, int outputHeight, int numKernels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.z;

    if (row < outputHeight && col < outputWidth && channel < numKernels) {
        float sum = 0.0f;

        // Apply convolution for the specific kernel (output channel)
        for (int i = 0; i < kernelHeight; ++i) {
            for (int j = 0; j < kernelWidth; ++j) {
                int inputRow = row + i;
                int inputCol = col + j;
                sum += input[inputRow * inputWidth + inputCol] * kernels[channel * kernelWidth * kernelHeight + i * kernelWidth + j];
            }
        }
        // Store the result in the output tensor (each channel has its own slice in output)
        output[channel * outputHeight * outputWidth + row * outputWidth + col] = sum;
    }
}

__global__ void reluActivation(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (data[idx] < 0) data[idx] = 0;
    }
}

__global__ void linearLayer(float* input, float* weights, float* bias, float* output, int inputSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize) {
        float sum = bias[idx];  // Start with bias
        for (int i = 0; i < inputSize; ++i) {
            sum += input[i] * weights[idx * inputSize + i];
        }
        output[idx] = sum;
    }
}

// Constructor
ConvNet::ConvNet(int inputWidth, int inputHeight, int kernelWidth, int kernelHeight, int numKernels, int fcOutputSize)
    : inputWidth(inputWidth), inputHeight(inputHeight), kernelWidth(kernelWidth), kernelHeight(kernelHeight),
      numKernels(numKernels), fcOutputSize(fcOutputSize) {
    outputWidth = inputWidth - kernelWidth + 1;
    outputHeight = inputHeight - kernelHeight + 1;
    fcInputSize = outputWidth * outputHeight * numKernels;

    cudaMalloc(&d_input, inputWidth * inputHeight * sizeof(float));
    cudaMalloc(&d_kernels, kernelWidth * kernelHeight * numKernels * sizeof(float));
    cudaMalloc(&d_convOutput, outputWidth * outputHeight * numKernels * sizeof(float));
    cudaMalloc(&d_fcWeights, fcOutputSize * fcInputSize * sizeof(float));
    cudaMalloc(&d_fcBias, fcOutputSize * sizeof(float));
    cudaMalloc(&d_finalOutput, fcOutputSize * sizeof(float));
}

// Destructor
ConvNet::~ConvNet() {
    cudaFree(d_input);
    cudaFree(d_kernels);
    cudaFree(d_convOutput);
    cudaFree(d_fcWeights);
    cudaFree(d_fcBias);
    cudaFree(d_finalOutput);
}

// Load weights
void ConvNet::loadWeights(const std::string& kernelFile, const std::string& fcWeightFile, const std::string& fcBiasFile) {
    readKernelWeights(kernelFile, kernels, kernelWidth * kernelHeight * numKernels);
    readFullyConnectedWeights(fcWeightFile, fcBiasFile, fcWeights, fcBias, fcOutputSize * fcInputSize, fcOutputSize);

    cudaMemcpy(d_kernels, kernels.data(), kernels.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fcWeights, fcWeights.data(), fcWeights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fcBias, fcBias.data(), fcBias.size() * sizeof(float), cudaMemcpyHostToDevice);
}

// Prediction
void ConvNet::predict(const std::vector<float>& input, std::vector<float>& output) {
    cudaDeviceSynchronize();

    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Call Net" << std::endl;
    convNet(d_input, d_kernels, d_convOutput, d_fcWeights, d_fcBias, d_finalOutput);


    std::cout << "Summurzing" << std::endl;
    output.resize(fcOutputSize);
    cudaMemcpy(output.data(), d_finalOutput, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float maxVal = *std::max_element(output.begin(), output.end());
    float sum = 0.0f;
    for (auto& val : output) {
        val = std::exp(val - maxVal);
        sum += val;
    }
    for (auto& val : output) {
        val /= sum;
    }
}

// Private helper functions
void ConvNet::readKernelWeights(const std::string& filename, std::vector<float>& kernels, int kernelSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file: " << filename << std::endl;
        return;
    }

    float value;
    int count = 0;
    while (file >> value && count < kernelSize) {
        kernels.push_back(value);
        count++;
    }

    if (count != kernelSize) {
        std::cerr << "Warning: Expected in Kernel " << kernelSize << " values, but found " << count << std::endl;
    }
    file.close();
}

void ConvNet::readFullyConnectedWeights(const std::string& weightFile, const std::string& biasFile,
                                        std::vector<float>& fcWeights, std::vector<float>& fcBias,
                                        int fcWeightSize, int fcBiasSize) {
    // Read fully connected layer weights
    std::ifstream weightStream(weightFile);
    if (!weightStream.is_open()) {
        std::cerr << "Error: Could not open the weight file: " << weightFile << std::endl;
        return;
    }

    float value;
    int count = 0;
    while (weightStream >> value && count < fcWeightSize) {
        fcWeights.push_back(value);
        count++;
    }

    if (count != fcWeightSize) {
        std::cerr << "Warning: Expected in FC Weights " << fcWeightSize << " weights, but found " << count << std::endl;
    }
    weightStream.close();

    // Read fully connected layer biases
    std::ifstream biasStream(biasFile);
    if (!biasStream.is_open()) {
        std::cerr << "Error: Could not open the bias file: " << biasFile << std::endl;
        return;
    }

    count = 0;
    while (biasStream >> value && count < fcBiasSize) {
        fcBias.push_back(value);
        count++;
    }

    if (count != fcBiasSize) {
        std::cerr << "Warning: Expected in FC Bias" << fcBiasSize << " biases, but found " << count << std::endl;
    }
    biasStream.close();
}

void ConvNet::convNet(float* input, float* kernels, float* convOutput, float* fcWeights, float* fcBias, float* finalOutput) {
    // Convolution kernel with multiple output channels
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((outputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (outputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y, numKernels);
    conv2d<<<blocksPerGrid, threadsPerBlock>>>(input, kernels, convOutput, inputWidth, inputHeight, 
                                               kernelWidth, kernelHeight, outputWidth, outputHeight, numKernels);

    // ReLU activation
    int convOutputSize = outputWidth * outputHeight * numKernels;
    int threadsPerBlockReLU = 256;
    int blocksPerGridReLU = (convOutputSize + threadsPerBlockReLU - 1) / threadsPerBlockReLU;
    reluActivation<<<blocksPerGridReLU, threadsPerBlockReLU>>>(convOutput, convOutputSize);

    // Fully connected layer kernel
    int threadsPerBlockFC = 256;
    int blocksPerGridFC = (fcOutputSize + threadsPerBlockFC - 1) / threadsPerBlockFC;
    linearLayer<<<blocksPerGridFC, threadsPerBlockFC>>>(convOutput, fcWeights, fcBias, finalOutput, fcInputSize, fcOutputSize);
}


std::vector<float> ConvNet::preprocessImage(const cv::Mat& img) {
    cv::Mat gray, resized;

    // Convert to grayscale
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }

    // Resize to target dimensions
    cv::resize(gray, resized, cv::Size(inputWidth, inputHeight));

    // Normalize to [0, 1]
    resized.convertTo(resized, CV_32F, 1.0 / 255);

    // Flatten into a vector
    std::vector<float> input(inputWidth * inputHeight);
    std::memcpy(input.data(), resized.data, input.size() * sizeof(float));

    return input;
}