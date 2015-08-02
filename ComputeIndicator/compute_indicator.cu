#include <stdio.h>
#include "compute_indicator.cuh"

using namespace std;

const int NUM_THREADS = 256;

__device__
void select() {
}

__device__
void expand() {
}

__device__
void simulation(Literal* model, float* indicator) {
}

__device__
float computeScore(float* target, unsigned char* mask, float* indicator) {
	int count = 0;
	int total = 0;
	for (int i = 0; i < GRID_SIZE * GRID_SIZE; ++i) {
		if (mask[i] > 0) {
			if (target[i] > 0.5) {
				if (indicator[i] < 0.5) count++;
				total++;
			} else {
				if (indicator[i] > 0.5) count++;
			}
		}
	}

	if (total > 0) {
		return (float)count / total;
	} else {
		if (count > 0) {
			return -1;
		} else {
			return 1;
		}
	}
}

__device__
void backpropagate(int score) {
}

__global__
void doNothing() {
}

__global__ 
void UCT(Literal* model, int* modelLength, float* target, unsigned char* mask, float* indicator, int* result)  {
	__shared__ float score[NUM_THREADS];

	int id = threadIdx.x;

	for (int iter = 0; iter < 10; ++iter) {//500; ++iter) {
		if (id == 0) {
			// UCT select
			select();

			// UCT expansion
			expand();
		}

		__syncthreads();

		// UCT simulation
		simulation(model, indicator);

		// compute score
		score[id] = computeScore(target, mask, indicator);

		__syncthreads();

		// merge score
		int offset = NUM_THREADS / 2;
		while (offset > 0) {
			if (id < offset) {
				score[id] += score[id + offset];
			}
			offset /= 2;
		}

		// UCT backpropagation
		if (id == 0) {
			backpropagate(score[0]);
		}
	}
}

/**
 * indicatorを計算する
 */
void cudaMain(const cv::Mat& target, const cv::Mat& mask) {
	double t1 = 0.0;
	double t2 = 0.0;
	double t3 = 0.0;
	double t4 = 0.0;
	double t5 = 0.0;
	double t6 = 0.0;
	double t7 = 0.0;

	// ダミーのKenelを起動し、オーバーヘッドをなくす
	time_t start = clock();
	doNothing<<<1, 1>>>();
	time_t end = clock();
	t1 += end - start;

	// Hostメモリの確保
	start = clock();
	int* hResult = (int*)malloc(sizeof(int));
	end = clock();
	t2 += end - start;

	// Deviceメモリの確保
	start = clock();
	float* dIndicator;
	cudaMalloc((void**)&dIndicator, sizeof(float) * target.rows * target.cols); 

	float* dTarget;
	cudaMalloc((void**)&dTarget, sizeof(float) * target.rows * target.cols); 
	cudaMemcpy(dTarget, target.ptr(), sizeof(float) * target.rows * target.cols, cudaMemcpyHostToDevice);

	unsigned char* dMask;
	cudaMalloc((void**)&dMask, sizeof(unsigned char) * mask.rows * mask.cols); 

	Literal* dModel;
	cudaMalloc((void**)&dModel, sizeof(Literal) * 100);

	int* dModelLength;
	cudaMalloc((void**)&dModelLength, sizeof(int));

	int* dResult;
	cudaMalloc((void**)&dResult, sizeof(int));
	end = clock();
	t3 += end - start;

	// モデルを設定
	Literal* hModel = (Literal*)malloc(sizeof(Literal) * 100);
	hModel[0].name = 'F';
	hModel[0].value1 = 6.0;
	hModel[0].value2 = 0.0;
	hModel[1].name = '+';
	hModel[1].value1 = 60.0;
	hModel[2].name = 'F';
	hModel[2].value1 = 12.0;
	hModel[2].value2 = 6.0;

	int hModelLength = 3;

	for (int i = 0; i < 500; ++i) {
		// maskをdeviceへ転送する
		start = clock();
		cudaMemcpy(dMask, mask.ptr(), sizeof(unsigned char) * mask.rows * mask.cols, cudaMemcpyHostToDevice);
		end = clock();
		t4 += end - start;

		start = clock();
		// モデルをdeviceへ転送する
		cudaMemcpy(dModel, hModel, sizeof(Literal) * 100, cudaMemcpyHostToDevice);
		cudaMemcpy(dModelLength, &hModelLength, sizeof(int), cudaMemcpyHostToDevice);
		end = clock();
		t5 += end - start;

		// UCTアルゴリズム
		start = clock();
		UCT<<<1, NUM_THREADS>>>(dModel, dModelLength, dTarget, dMask, dIndicator, dResult);
		cudaDeviceSynchronize();
		end = clock();
		t6 += end - start;

		// ベストactionを転送
		start = clock();
		cudaMemcpy(hResult, dResult, sizeof(int), cudaMemcpyDeviceToHost);
		end = clock();
		t7 += end - start;

		// modelにベストactionをapply

	}
	cout << "Time1: " << t1 << endl;
	cout << "Time2: " << t2 << endl;
	cout << "Time3: " << t3 << endl;
	cout << "Time4: " << t4 << endl;
	cout << "Time5: " << t5 << endl;
	cout << "Time6: " << t6 << endl;
	cout << "Time7: " << t7 << endl;

	// Deviceメモリ解放
	cudaFree(dIndicator);
	cudaFree(dTarget);
	cudaFree(dMask);
	cudaFree(dModel);
	cudaFree(dModelLength);
	cudaFree(dResult);

	cout << "Result: " << hResult[0] << endl;

	// Hostメモリ解放
	free(hModel);
	free(hResult);
}