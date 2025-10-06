#include <cuda_runtime.h>

__global__ void dotProduct(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
	extern __shared__  float dotMem[];

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + tid;
	int stride = blockDim.x * gridDim.x;

	float tmp = 0.0f;

	#pragma unroll
	for (unsigned int i = idx; i < N; i += stride) {
		tmp += A[idx] * B[idx];
	}
	dotMem[tid] = tmp;
	__syncthreads();

	#pragma unroll
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			dotMem[tid] += dotMem[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		atomicAdd(C, dotMem[0]);
	}
}
