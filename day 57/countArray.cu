#include <cuda_runtime.h>

__global__ void countarray_kernel(const int* input, int* output, int N, int K) {
	extern __shared__ int countMem[];

	int tx = threadIdx.x;
	int bdimx = blockDim.x;
	int tid = blockIdx.x * bdimx + tx;

	countMem[tx] = 0;

	if (tid < N) {
		if (input[tid] == K) {
			countMem[tx] = 1;
		}
	}
	__syncthreads();

	for (int s = bdimx / 2; s > 0; s /= 2) {
		if (tx < s && (tx + s) < bdimx) {
			countMem[tx] += countMem[tx + s];
		}
		__syncthreads();
	}

	if (tx == 0) {
		atomicAdd(output, countMem[0]);
	}
}

extern "C" void solve(const int* input, int* output, int N, int K) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	size_t countMemBytes = threadsPerBlock * sizeof(int);

	cudaMemset(output, 0, sizeof(int));

	countarray_kernel<<<blocksPerGrid, threadsPerBlock, countMemBytes>>>(input, output, N, K);
	cudaDeviceSynchronize();
}
