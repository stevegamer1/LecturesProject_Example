#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <opencv2/opencv.hpp>

class Preprocessor {
public:
    static cv::Mat toGrayscale(const cv::Mat& input);
    static cv::Mat normalize(const cv::Mat& input);
    static cv::Mat applySobel(const cv::Mat& input);
};

#endif // PREPROCESSING_H
