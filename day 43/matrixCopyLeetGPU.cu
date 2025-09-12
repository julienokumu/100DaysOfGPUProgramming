#include <cuda_runtime.h>

__global__ void matrixCopy(const float* A, float* B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;

    if (idx < total) {
        int i = idx / N;
        int j = idx % N;
        B[idx] = A[idx];
    }
}

extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

    matrixCopy<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}