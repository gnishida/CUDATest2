﻿/**
 * opencvのMatを使った処理
 * CPU版とGPU版を実装したが、残念ながら、CPU版の方が速い。
 * 主な原因は、MatデータをCPUからGPUへ転送する時間だ。
 */

#include <stdio.h>
#include "mattest.cuh"
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;

const int N = 3000;
const int GRID_SIZE = 300;

int main() {
	cv::Mat_<float> a(GRID_SIZE, GRID_SIZE);
	cv::Mat_<float> b(GRID_SIZE, GRID_SIZE);
	cv::Mat_<uchar> c = cv::Mat_<uchar>::zeros(GRID_SIZE, GRID_SIZE);

	cv::randu(a, 0, 1.0);
	cv::randu(b, 0, 1.0);
	cv::threshold(a, a, 0.5, 1.0, cv::THRESH_BINARY);
	cv::threshold(b, b, 0.5, 1.0, cv::THRESH_BINARY);
	cv::circle(c, cv::Point(GRID_SIZE * 0.5, GRID_SIZE * 0.5), GRID_SIZE * 0.2, cv::Scalar(1), -1);

	/*
	cout << a << endl;
	cout << b << endl;
	cout << c << endl;
	*/

	// CPU版
	time_t start = clock();
	for (int i = 0; i < N; ++i) {
		cv::Mat result;
		cv::subtract(a, b, result, c);
		float count = cv::countNonZero(result);
		cv::subtract(a, cv::Mat::zeros(a.size(), a.type()), result, c);
		float total = cv::countNonZero(result);
		//cout << count << ", " << total << endl;
	}
	time_t end = clock();
	cout << "Time1: " << (double)(end - start) << endl;

	// CUDA版
	for (int i = 0; i < N; ++i) {
		cudaMain(a, b, c);
	}

	return 0;
}