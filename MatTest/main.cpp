/**
 * opencvのMatの計算
 */

#include <stdio.h>
#include "mattest.cuh"
#include <opencv/cv.h>
#include <opencv/highgui.h>

int main() {
	cv::Mat_<float> a(100, 100);
	cv::Mat_<float> b(100, 100);
	cv::Mat_<uchar> c(100, 100);

	cv::randu(a, 0, 1.0);
	cv::randu(b, 0, 1.0);
	cv::threshold(a, a, 0.0, 1.0, cv::THRESH_BINARY);
	cv::threshold(b, b, 0.0, 1.0, cv::THRESH_BINARY);
	cv::circle(c, cv::Point(50, 50), 20, cv::Scalar(1), -1);

	cudaMain(a, b, c);

	return 0;
}