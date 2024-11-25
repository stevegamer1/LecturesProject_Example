#include "ConvNet.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    int inputWidth = 28, inputHeight = 28;
    int kernelWidth = 5, kernelHeight = 5;
    int numKernels = 6;
    int fcOutputSize = 10;

    ConvNet network(inputWidth, inputHeight, kernelWidth, kernelHeight, numKernels, fcOutputSize);

    network.loadWeights("/home/mzingerenko/Desktop/LecturesProject_Example/data/conv_kernels.txt",
                        "/home/mzingerenko/Desktop/LecturesProject_Example/data/fc_weights.txt",
                        "/home/mzingerenko/Desktop/LecturesProject_Example/data/fc_biases.txt");

    cv::Mat img = cv::imread("/home/mzingerenko/Desktop/LecturesProject_Example/data/mnist_img3.png", cv::IMREAD_GRAYSCALE);

    std::vector<float> input = network.preprocessImage(img);
    std::vector<float> output;
    network.predict(input, output);

    std::cout << "Predictions (probabilities):" << std::endl;
    for (int i = 0; i < output.size(); ++i) {
        std::cout << "Class " << i << ": " << output[i] << std::endl;
    }

    return 0;
}
