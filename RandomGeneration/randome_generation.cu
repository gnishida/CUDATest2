#include <stdio.h>
#include <curand_kernel.h>
#include "random_generation.cuh"

const int NUM_THREADS = 16;
const int N = 10;

__global__ 
void hello(float *result)  {
	int id = threadIdx.x;// + blockIdx.x * 64;
	curandState state;
	curand_init(1234, id, 0, &state);

	for (int i = 0; i < N; ++i) {
		result[id * N + i] = curand_uniform(&state);
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