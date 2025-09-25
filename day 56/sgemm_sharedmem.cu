#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 32

__global__ void sgemm_sharedmem(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
	__shared__ half As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE];

	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int row = blockIdx.y * BLOCK_SIZE + ty;
	int col = blockIdx.x * BLOCK_SIZE + tx;

	float tmp = 0.0f;

	for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
		if (row < M && (t * BLOCK_SIZE + tx) < K) {
			As[ty][tx] = A[row * K + t * BLOCK_SIZE + tx];
		} else {
			As[ty][tx] = 0.0f;
		}

		if (col < N && (t * BLOCK_SIZE + ty) < K) {
			Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];
		} else {
			Bs[ty][tx] = 0.0f;
		}
		__syncthreads();

		#pragma unroll 4
		for (int k = 0; k < BLOCK_SIZE; k++) {
			tmp += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
		}
		__syncthreads();
	}

	if (row < M && col < N) {
		C[row * N + col] = alpha * tmp + beta * __half2float(C[row * N + col]);
	}
}

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
										 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

	sgemm_sharedmem<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
}
