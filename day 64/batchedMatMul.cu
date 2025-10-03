#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define TILE_WIDTH 4

__global__ void batched_matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	int b = blockIdx.z;
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int row = blockIdx.y * BLOCK_SIZE + ty;
	int col = blockIdx.x * BLOCK_SIZE * TILE_WIDTH + tx;

	float tmp[TILE_WIDTH] = {0.0f};

	for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
		if (row < M && (t * BLOCK_SIZE + tx) < K) {
			As[ty][tx] = A[b * M * K + row * K + t * BLOCK_SIZE + tx];
		} else{
			As[ty][tx] = 0.0f;
		}

		#pragma unroll
		for (int i = 0; i < TILE_WIDTH; i++) {
			int col_idx = col + i * BLOCK_SIZE;
			if (col_idx < N && (t * BLOCK_SIZE + ty) < K) {
				Bs[ty][tx] = B[b * N * K + (t * BLOCK_SIZE + ty) * N + col_idx];
			} else {
				Bs[ty][tx] = 0.0f;
			}
			__syncthreads();

			#pragma unroll
			for (int k = 0; k < BLOCK_SIZE; k++) {
				tmp[i] += As[ty][k] * Bs[k][tx];
			}
			__syncthreads();
		}
	}
	#pragma unroll
	for (int i = 0; i < TILE_WIDTH; i++) {
		int col_idx = col + i * BLOCK_SIZE;
		if (row < M && col_idx < N) {
			C[b * M * N + row * N + col_idx] = tmp[i];
		}
	}
}

extern "C" void solve(const float* A, const float* B, float* C, int b, int M, int N, int K) {
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid((N + BLOCK_SIZE * TILE_WIDTH - 1) / (BLOCK_SIZE * TILE_WIDTH),
										 (M + BLOCK_SIZE - 1) / BLOCK_SIZE, b);

	batched_matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
}
