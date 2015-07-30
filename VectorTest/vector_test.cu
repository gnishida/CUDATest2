#include <stdio.h>
#include <curand_kernel.h>
#include "vector_test.cuh"
#include "vector.cuh"

const int NUM_THREADS = 16;
const int N = 10;

__device__
void test() {
	vector<float> test;
	test.push_back(0.3);
	test.push_back(1.3);

	vector<float> test2;
	test2.push_back(3.3);

	test.insert(test.end(), test2.begin(), test2.end());
}

__global__ 
void hello(float *result)  {
	int id = threadIdx.x;// + blockIdx.x * 64;
	curandState state;
	curand_init(1234, id, 0, &state);

	vector<float> values;

	values.push_back(0.3);
	values.push_back(1.3);
	values.push_back(5.3);
	values.push_back(2.1);
	values.push_back(2.8);

	vector<float> test;
	test.push_back(3.0);
	test.push_back(3.1);
	test.push_back(3.2);
	test.push_back(3.3);
	test.push_back(3.4);

	values.insert(values.end(), test.begin(), test.end());

	for (int i = 0; i < N; ++i) {
		//values[i] = curand_uniform(&state);
	}

	for (int i = 0; i < N; ++i) {
		result[id * N + i] = values[i];
	}
}
 
void cudaMain() {
	int size = sizeof(float) * NUM_THREADS * N;

	float* hResult = (float*)malloc(size);
	float* dResult;

	cudaMalloc((void**)&dResult, size); 
	cudaMemcpy(dResult, hResult, size, cudaMemcpyHostToDevice);
	
	hello<<<1, NUM_THREADS>>>(dResult);
	cudaMemcpy(hResult, dResult, size, cudaMemcpyDeviceToHost);
	cudaFree(dResult);

	for (int i = 0; i < NUM_THREADS; ++i) {
		for (int j = 0; j < N; ++j) {
			printf("%lf, ", hResult[i * N + j]);
		}
		printf("\n");
	}
}