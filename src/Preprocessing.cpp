#include "Preprocessing.h"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <iostream>
#include <cmath>

// Convert image to grayscale
cv::Mat Preprocessor::toGrayscale(const cv::Mat& input) {
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    return gray;
}

// Normalize pixel values to range [0, 1]
cv::Mat Preprocessor::normalize(const cv::Mat& input) {
    cv::Mat normalized = input.clone();
    normalized.convertTo(normalized, CV_32F);  // Ensure target type is float

    tbb::parallel_for(tbb::blocked_range2d<int>(0, input.rows, 0, input.cols),
        [&](const tbb::blocked_range2d<int>& range) {
            for (int i = range.rows().begin(); i < range.rows().end(); ++i) {
                for (int j = range.cols().begin(); j < range.cols().end(); ++j) {
                    normalized.at<float>(i, j) = input.at<uchar>(i, j) / 255.0f;
                }
            }
        });

    return normalized;
}

// Apply Sobel edge detection
cv::Mat Preprocessor::applySobel(const cv::Mat& input) {
    cv::Mat gradX, gradY, grad;

    // Compute Sobel gradients
    cv::Sobel(input, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(input, gradY, CV_32F, 0, 1, 3);

    grad.create(input.size(), CV_32F);

    // Combine gradients in parallel
    tbb::parallel_for(tbb::blocked_range2d<int>(0, input.rows, 0, input.cols),
        [&](const tbb::blocked_range2d<int>& range) {
            for (int i = range.rows().begin(); i < range.rows().end(); ++i) {
                for (int j = range.cols().begin(); j < range.cols().end(); ++j) {
                    float gx = gradX.at<float>(i, j);
                    float gy = gradY.at<float>(i, j);
                    grad.at<float>(i, j) = std::sqrt(gx * gx + gy * gy);
                }
            }
        });

    return grad;
}
