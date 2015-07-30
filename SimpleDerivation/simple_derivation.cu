#include <stdio.h>
#include <curand_kernel.h>
#include "simple_derivation.cuh"
#include "lsystem.cuh"

const int NUM_THREADS = 16;
const int N = 100;

__global__ 
void hello(char *result)  {
	int id = threadIdx.x;// + blockIdx.x * 64;
	curandState state;
	curand_init(1234, id, 0, &state);

	String model = Literal("X", 0, 124.0f, 0.0f);

	vector<Action> actions = getActions(model);
	int index = curand_uniform(&state) * actions.size();
	model = actions[index].apply(model);

	for (int i = 0; i < model.length(); ++i) {
		printf("%s ", model[i].name);
	}

}
 
void cudaMain() {
	int size = sizeof(char) * NUM_THREADS * N;

	char* hResult = (char*)malloc(size);
	char* dResult;

	cudaMalloc((void**)&dResult, size); 
	cudaMemcpy(dResult, hResult, size, cudaMemcpyHostToDevice);
	
	hello<<<1, NUM_THREADS>>>(dResult);
	cudaMemcpy(hResult, dResult, size, cudaMemcpyDeviceToHost);
	cudaFree(dResult);

	for (int i = 0; i < NUM_THREADS; ++i) {
		for (int j = 0; j < N; ++j) {
			if (hResult[i * N + j] == 0) break;
			printf("%c, ", hResult[i * N + j]);
		}
		printf("\n");
	}
}