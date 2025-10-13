#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void transposeMatrix(const float* __restrict__ input, float* __restrict__ output, int rows,int cols) {
	__shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

	int ty = threadIdx.y;
	int tx = threadIdx.x;
	
	int row = blockIdx.y * TILE_SIZE + ty;
	int col = blockIdx.x * TILE_SIZE + tx;

	#pragma unroll
	for (unsigned int i = 0; i < TILE_SIZE; i += blockDim.y) {
		if (row < rows && col < cols) {
			tile[ty + i][tx] = input[(row + i) * cols + col];
		}
	}
	__syncthreads();

	#pragma unroll
	for (unsigned int i = 0; i < TILE_SIZE; i += blockDim.y) {
		int t_row = blockIdx.y * TILE_SIZE + tx;
		int t_col = blockIdx.x * TILE_SIZE + ty + i;

		if (t_row < rows && t_col < cols) {
			output[t_col * rows + t_row] = tile[tx][ty + i];
		}
	}
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
	dim3 blocksPerGrid((cols + TILE_SIZE - 1) / TILE_SIZE,
										 (rows + TILE_SIZE - 1) / TILE_SIZE);

	transposeMatrix<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
}
