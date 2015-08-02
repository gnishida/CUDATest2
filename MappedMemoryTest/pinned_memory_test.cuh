#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

void cudaMain(cv::Mat_<float>& a, cv::Mat_<float>& b, cv::Mat_<uchar>& mask);

