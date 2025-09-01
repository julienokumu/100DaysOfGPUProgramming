#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void softmaxKernel(float* input, float* output, int N) {
	extern __shared__ float sharedMem[];

	int tid = threadIdx.x;

	sharedMem[tid] = (tid < N) ? input[tid] : -FLT_MAX;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>=1) {
		if (tid < s) {
			sharedMem[tid] = max(sharedMem[tid], sharedMem[tid + s]);
		}
		__syncthreads();
	}

	float maxVal = sharedMem[0];
	__syncthreads();

	float expVal = (tid < N) ? expf(input[tid] - maxVal) : 0.0f;
	sharedMem[tid] = expVal;
	__syncthreads();

	for (int s = blockDim.x / 2; s > 0; s >>=1) {
		if (tid < s) {
			sharedMem[tid] += sharedMem[tid + s];
		}
		__syncthreads();
	}

	float expSum = sharedMem[0];
	__syncthreads();

	if (tid < N) {
		output[tid] = expVal / expSum;
	}
}

int main() {
	const int N = 512;
	size_t bytes = N * sizeof(float);

	float* h_input = (float*)malloc(bytes);
	float* h_output = (float*)malloc(bytes);

	srand(42);
	for (int i = 0; i < N; i++) {
		h_input[i] = (rand() / (float)RAND_MAX) * 10.0f - 5.0f;
	}

	float* d_input, *d_output;
	cudaMalloc((void **)&d_input, bytes);
	cudaMalloc((void **)&d_output, bytes);

	cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

	int threadsPerBlock = N;
	int blocksPerGrid = 1;

	softmaxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
	cudaDeviceSynchronize();

	cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

	float sum = 0.0f;
	for (int i = 0; i < N; i++) {
		sum += h_output[i];
	}
	printf("Softmax Probability: %f\n", sum);

	cudaFree(d_input);
	cudaFree(d_output);
	free(h_input);
	free(h_output);

	return 0;
}
