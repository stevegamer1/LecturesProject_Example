#include <gtest/gtest.h>
#include "Preprocessing.h"

// Test grayscale conversion
TEST(PreprocessingTest, GrayscaleConversion) {
    cv::Mat colorImg(100, 100, CV_8UC3, cv::Scalar(255, 255, 255)); // White image
    cv::Mat grayImg = Preprocessor::toGrayscale(colorImg);

    ASSERT_EQ(grayImg.channels(), 1);
    ASSERT_EQ(grayImg.rows, colorImg.rows);
    ASSERT_EQ(grayImg.cols, colorImg.cols);
    ASSERT_EQ(grayImg.at<uchar>(0, 0), 255); // Check pixel value
}

// Test normalization
TEST(PreprocessingTest, Normalization) {
    cv::Mat grayImg = cv::Mat::ones(100, 100, CV_8UC1) * 128; // Mid-gray image
    cv::Mat normalizedImg = Preprocessor::normalize(grayImg);

    ASSERT_EQ(normalizedImg.rows, grayImg.rows);
    ASSERT_EQ(normalizedImg.cols, grayImg.cols);
    ASSERT_NEAR(normalizedImg.at<float>(0, 0), 128.0 / 255.0, 1e-5);
}

// Test Sobel filter
TEST(PreprocessingTest, SobelFilter) {
    cv::Mat grayImg = cv::Mat::zeros(100, 100, CV_8UC1);
    grayImg.at<uchar>(50, 50) = 255;       // Bright pixel
    grayImg.at<uchar>(50, 51) = 255;       // Neighboring bright pixel for gradient
    cv::Mat normalized = Preprocessor::normalize(grayImg);
    cv::Mat gradImg = Preprocessor::applySobel(normalized);

    ASSERT_EQ(gradImg.rows, grayImg.rows);
    ASSERT_EQ(gradImg.cols, grayImg.cols);
    ASSERT_GT(gradImg.at<float>(50, 50), 0); // Check that Sobel detects the edge
}
