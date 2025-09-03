#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrixTranspose(const float* input, float* output, int rows, int cols) {
	__shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;

	if (row < rows && col < cols) {
		tile[threadIdx.y][threadIdx.x] = input[row * cols + col];
	}
	__syncthreads();

	int t_row = blockIdx.y * TILE_SIZE + threadIdx.x;
	int t_col = blockIdx.x * TILE_SIZE + threadIdx.y;

	if (t_row < rows && t_col < cols) {
		output[t_col * rows + t_row] = tile[threadIdx.x][threadIdx.y];
	}
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
	dim3 blocksPerGrid((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

	matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
	cudaDeviceSynchronize();
}
