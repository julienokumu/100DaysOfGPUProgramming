#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void BMM(const float* A, const float* B, float* C, int M, int N, int K) {
	__shared__ float tileA[TILE_SIZE][TILE_SIZE];
	__shared__ float tileB[TILE_SIZE][TILE_SIZE];

	int batch = blockIdx.z;
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int row = blockIdx.y * TILE_SIZE + ty;
	int col = blockIdx.x * TILE_SIZE + tx;

	float sum = 0.0f;

	for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
		if (row < M && (t * TILE_SIZE + tx) < K) {
			tileA[ty][tx] = A[batch * M * K + row * K + t * TILE_SIZE + tx];
		} else {
			tileA[ty][tx] = 0.0f;
		}

		if (col < N && (t * TILE_SIZE + ty) < K) {
			tileB[ty][tx] = B[batch * N * K + (t * TILE_SIZE + ty) * N + col];
		} else {
			tileB[ty][tx] = 0.0f;
		}
		__syncthreads();

		if (row < M && col < N) {
			for (int k = 0; k < TILE_SIZE; k++) {
				sum += tileA[ty][k] * tileB[k][tx];
			}
		}
		__syncthreads();
	}

	if (row < M && col < N){
		C[batch * M * N + row * N + col] = sum;
	}
}

extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
	dim3 blocksPerGrid((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

	BMM<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
	cudaDeviceSynchronize();
}
