#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 32

__global__ void sgemm_half(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
	int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	if (row < M && col < N) {
		float tmp = 0.0f;

		#pragma unroll 4
		for (int k = 0; k < K; k++) {
			tmp += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
		}
		C[row * N + col] = alpha * tmp + beta * __half2float(C[row * N + col]);
	}
}

extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
										 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

	sgemm_half<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, alpha, beta);
	cudaDeviceSynchronize();
}
