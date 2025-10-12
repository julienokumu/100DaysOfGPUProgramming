#include <cuda_runtime.h>

__global__ void dotProduct(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ result, int N) {
	extern __shared__ float dotMem[];

	int tidx = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + tidx;
	int stride = blockDim.x * gridDim.x;

	float tmp  = 0.0f;

	#pragma unroll
	for (unsigned int i = idx; i < N; i += stride) {
		tmp += A[i] * B[i];
	}
	dotMem[tidx] = tmp;
	__syncthreads();

	#pragma unroll
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tidx < s) {
			dotMem[tidx] += dotMem[tidx + s];
		}
		__syncthreads();
	}

	if (tidx == 0) {
		atomicAdd(result, dotMem[0]);
	}
}
