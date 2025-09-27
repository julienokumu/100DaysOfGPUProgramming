#include <cuda_runtime.h>

__global__ void dotProduct(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ result, int N) {
	extern __shared__ float dotMem[];

	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;
	unsigned int stride = blockDim.x * gridDim.x;

	float sum = 0.0f;

	#pragma unroll
	for (unsigned int i = idx; i < N; i += stride) {
		sum += A[idx] * B[idx];
	}
	dotMem[tid] = sum;
	__syncthreads();
	
	#pragma unroll
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			dotMem[tid] += dotMem[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		atomicAdd(result, dotMem[0]);
	}
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
	int threadsPerBlock = 1024;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	size_t dotMemBytes = threadsPerBlock * sizeof(float);

	dotProduct<<<blocksPerGrid, threadsPerBlock, dotMemBytes>>>(A, B, result, N);
	cudaDeviceSynchronize();
}
