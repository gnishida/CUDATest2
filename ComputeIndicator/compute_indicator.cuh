#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

const int GRID_SIZE = 300;

struct Literal {
	char name;
	float value1;
	float value2;
};

void cudaMain(const cv::Mat& target, const cv::Mat& mask);

