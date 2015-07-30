#include <stdio.h>
#include <curand_kernel.h>
#include "simple_derivation.cuh"
#include "lsystem.cuh"

const int NUM_THREADS = 16;
const int N = 10;

__global__ 
void hello(float *result)  {
	int id = threadIdx.x;// + blockIdx.x * 64;
	curandState state;
	curand_init(1234, id, 0, &state);

	Literal l(0);

	/*String model = Literal("X", 0, 124.0f, 0.0f);
	vector<Action> actions = getActions(model);
	int index = curand_uniform(&state) * actions.size();
	model = actions[index].apply(model);*/


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