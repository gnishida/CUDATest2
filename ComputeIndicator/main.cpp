/**
 * compute indicator
 *
 * GPUでのMCTS実装に向けた、性能測定。
 */

#include <stdio.h>
#include "compute_indicator.cuh"
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;

int main() {
	cv::Mat_<float> target(GRID_SIZE, GRID_SIZE);
	cv::Mat_<uchar> mask = cv::Mat_<uchar>::zeros(GRID_SIZE, GRID_SIZE);

	cv::randu(target, 0, 1.0);
	cv::threshold(target, target, 0.5, 1.0, cv::THRESH_BINARY);
	cv::circle(mask, cv::Point(GRID_SIZE * 0.5, GRID_SIZE * 0.5), GRID_SIZE * 0.2, cv::Scalar(1), -1);

	// CUDA版
	cudaMain(target, mask);

	return 0;
}