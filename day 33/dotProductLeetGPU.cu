#include <cuda_runtime.h>

__global__ void dotProduct(const float* A, const float* B, float* result, int N) {
	extern __shared__ float sMem[];

	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	float acc = 0.0f;
	while (idx + 3 * stride < N) {
		acc += A[idx] * B[idx];
		acc += A[idx + stride] * B[idx + stride];
		acc += A[idx + 2 * stride] * B[idx + 2 * stride];
		acc += A[idx + 3 * stride] * B[idx + 3 * stride];
		idx += 4 * stride;
	}

	while(idx < N) {
		acc += A[idx] * B[idx];
		idx += stride;
	}
	sMem[tid] = acc;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
		if (tid < s) {
			sMem[tid] += sMem[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		atomicAdd(result, sMem[0]);
	}
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	size_t sMemBytes = threadsPerBlock * sizeof(float);

	dotProduct<<<blocksPerGrid, threadsPerBlock, sMemBytes>>>(A, B, result, N);
	cudaDeviceSynchronize();
}
