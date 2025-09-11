#include <cuda_runtime.h>

__global__ void dotProduct(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
	__shared__ float sMem[];

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	float sum = 0.0f;
	if (idx < N) {
		sum += A[idx] * B[idx];
		idx += stride;
	}
	sMem[tid] = sum;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sMem[tid] += sMem[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		atomicAdd(C, sMem[0]);
	}
}


