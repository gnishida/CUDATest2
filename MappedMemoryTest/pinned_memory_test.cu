#include <stdio.h>
#include "pinned_memory_test.cuh"

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
	// ホスト側でメモリ確保
	float* hMatA;// = (float*)malloc(sizeof(float) * a.rows * a.cols);
	cudaHostAlloc((void**)&hMatA, sizeof(float) * a.rows * a.cols, cudaHostAllocDefault);
	float* hMatB;// = (float*)malloc(sizeof(float) * b.rows * b.cols);
	cudaHostAlloc((void**)&hMatB, sizeof(float) * b.rows * b.cols, cudaHostAllocDefault);
	unsigned char* hMask;// = (unsigned char*)malloc(sizeof(unsigned char) * mask.rows * mask.cols);
	cudaHostAlloc((void**)&hMask, sizeof(unsigned char) * mask.rows * mask.cols, cudaHostAllocDefault);
	//int hSize = a.rows * a.cols;
	int* hSize;
	cudaHostAlloc((void**)&hSize, sizeof(int), cudaHostAllocDefault);
	hSize[0] = a.rows * a.cols;
	float* hResult;// = (float*)malloc(sizeof(float) * 2);
	cudaHostAlloc((void**)&hResult, sizeof(float) * 2, cudaHostAllocDefault);

	// デバイス側の変数
	float* dMatA;
	float* dMatB;
	unsigned char* dMask;
	int* dSize;
	float* dResult;

	memcpy(hMatA, a.data, sizeof(float) * a.rows * a.cols);
	memcpy(hMatB, b.data, sizeof(float) * b.rows * b.cols);
	memcpy(hMask, mask.data, sizeof(unsigned char) * mask.rows * mask.cols);

	cudaMalloc((void**)&dMatA, sizeof(float) * a.rows * a.cols); 
	cudaMemcpy(dMatA, hMatA, sizeof(float) * a.rows * a.cols, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dMatB, sizeof(float) * b.rows * b.cols); 
	cudaMemcpy(dMatB, hMatB, sizeof(float) * b.rows * b.cols, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dMask, sizeof(unsigned char) * mask.rows * mask.cols); 
	cudaMemcpy(dMask, hMask, sizeof(unsigned char) * mask.rows * mask.cols, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dSize, sizeof(int)); 
	cudaMemcpy(dSize, hSize, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dResult, sizeof(float) * 2);

	mattest<<<1, NUM_THREADS>>>(dMatA, dMatB, dMask, dSize, dResult);

	cudaMemcpy(hResult, dResult, sizeof(float) * 2, cudaMemcpyDeviceToHost);

	cudaFree(dMatA);
	cudaFree(dMatB);
	cudaFree(dMask);
	cudaFree(dSize);
	cudaFree(dResult);

	//cout << "Result: " << hResult[0] << ", " << hResult[1] << endl;
	//cout << "Time2: " << t1 << "," << t2 << "," << t3 << "," << t4 << "," << t5 << endl;

	//free(hMatA);
	//free(hMatB);
	//free(hMask);
	//free(hSize);
	//free(hResult);

	cudaFreeHost(hMatA);
	cudaFreeHost(hMatB);
	cudaFreeHost(hMask);
	cudaFreeHost(hSize);
	cudaFreeHost(hResult);
}