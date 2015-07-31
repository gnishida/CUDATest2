#include <stdio.h>
#include "mattest.cuh"

using namespace std;

const int NUM_THREADS = 256;

__global__ 
void mattest(float* matA, float* matB, unsigned char* mask, int* size, float *result)  {
	int id = threadIdx.x + blockIdx.x * NUM_THREADS;

	__shared__ float count;
	__shared__ float total;
	
	if (id == 0) {
		count = 0.0;
		total = 0.0;
	}
	int s = size[0];

	__syncthreads();

	int iterations = ceil((float)s / NUM_THREADS);

	for (int iter = 0; iter < iterations; ++iter) {
		int index = iter * NUM_THREADS + id;
		if (index < s) {
			if (mask[index] > 0) {
				float a = matA[index];
				float b = matB[index];

				if (a > 0.5) {
					if (b < 0.5) {
						atomicAdd(&count, 1);
					}
					atomicAdd(&total, 1);
				} else {
					if (b > 0.5) {
						atomicAdd(&count, 1);
					}
				}
			}
		}
	}

	__syncthreads();

	if (id == 0) {
		result[0] = count;
		result[1] = total;
	}
}
 
/**
 * 2つの行列の差分の数を計算する。
 */
void cudaMain(cv::Mat_<float>& a, cv::Mat_<float>& b, cv::Mat_<uchar>& mask) {
	float* hMatA = (float*)malloc(sizeof(float) * a.rows * a.cols);
	float* hMatB = (float*)malloc(sizeof(float) * b.rows * b.cols);
	unsigned char* hMask = (unsigned char*)malloc(sizeof(unsigned char) * mask.rows * mask.cols);
	int hSize = a.rows * a.cols;
	float* dMatA;
	float* dMatB;
	unsigned char* dMask;
	int* dSize;
	float* hResult = (float*)malloc(sizeof(float) * 2);
	float* dResult;

	memcpy(hMatA, a.data, sizeof(float) * a.rows * a.cols);
	cudaMalloc((void**)&dMatA, sizeof(float) * a.rows * a.cols); 
	cudaMemcpy(dMatA, hMatA, sizeof(float) * a.rows * a.cols, cudaMemcpyHostToDevice);

	memcpy(hMatB, b.data, sizeof(float) * b.rows * b.cols);
	cudaMalloc((void**)&dMatB, sizeof(float) * b.rows * b.cols); 
	cudaMemcpy(dMatB, hMatB, sizeof(float) * b.rows * b.cols, cudaMemcpyHostToDevice);

	memcpy(hMask, mask.data, sizeof(unsigned char) * mask.rows * mask.cols);
	cudaMalloc((void**)&dMask, sizeof(unsigned char) * mask.rows * mask.cols); 
	cudaMemcpy(dMask, hMask, sizeof(unsigned char) * mask.rows * mask.cols, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dSize, sizeof(int)); 
	cudaMemcpy(dSize, &hSize, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dResult, sizeof(float) * 2);
	
	mattest<<<1, NUM_THREADS>>>(dMatA, dMatB, dMask, dSize, dResult);

	cudaMemcpy(hResult, dResult, sizeof(float) * 2, cudaMemcpyDeviceToHost);
	cudaFree(dResult);

	//cout << "Result: " << hResult[0] << ", " << hResult[1] << endl;
	free(hResult);
}