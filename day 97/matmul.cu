#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrixMul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float tmp = 0.0f;

    // Loop over tiles of A and B
    #pragma unroll
    for (unsigned int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // Load tile of A into shared memory (A is M x N, row-major)
        if (row < M && (tile * BLOCK_SIZE + tx) < N) {
            As[ty][tx] = A[row * N + tile * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f; // Pad with zero if out of bounds
        }

        // Load tile of B into shared memory (B is N x K, row-major)
        if ((tile * BLOCK_SIZE + ty) < N && col < K) {
            Bs[ty][tx] = B[(tile * BLOCK_SIZE + ty) * K + col];
        } else {
            Bs[ty][tx] = 0.0f; // Pad with zero if out of bounds
        }

        // Synchronize to ensure all threads have loaded the tile
        __syncthreads();

        #pragma unroll
        // Compute dot product for this tile
        for (unsigned int k = 0; k < BLOCK_SIZE; k++) {
            tmp += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write result to C (C is M x K, row-major)
    if (row < M && col < K) {
        C[row * K + col] = tmp;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
}
