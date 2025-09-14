#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void dotProductKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
	extern __shared__ float sMem[];

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tx = threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	float sum = 0.0f;
	if (idx < N) {
		sum += A[idx] * B[idx];
		idx += stride;
	}
	sMem[tx] = sum;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tx < s) {
			sMem[tx] += sMem[tx + s];
		}
		__syncthreads();
	}

	if (tx == 0) {
		atomicAdd(C, sMem[0]);
	}
}

void dotProduct(float* A, float* B, int N) {
	for (int i = 0; i < N; i++) {
		A[i] = (float)(i % 10) + 1.0f;
		B[i] = (float)(10 - (i % 10));
	}
}

int main() {
	const int N = 1024;
	size_t bytes = N * sizeof(float);
	const int threadsPerBlock = 512;
	size_t sMemBytes = threadsPerBlock * sizeof(float);
	const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	float* h_A = (float*)malloc(bytes);
	float* h_B = (float*)malloc(bytes);
	float* h_C = (float*)malloc(sizeof(float));

	dotProduct(h_A, h_B, N);
	*h_C = 0.0f;

	float* d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, bytes);
	cudaMalloc((void **)&d_B, bytes);
	cudaMalloc((void **)&d_C, sizeof(float));

	cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

	dotProductKernel<<<blocksPerGrid, threadsPerBlock, sMemBytes>>>(d_A, d_B, d_C, N);
	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost);

	printf("Dot Product: %f/n", *h_C);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}

