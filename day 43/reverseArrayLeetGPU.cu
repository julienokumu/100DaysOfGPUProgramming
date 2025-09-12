#include <cuda_runtime.h>

__global__ void reverseArray(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = N / 2;

    if (idx < half) {
        float temp = input[idx];
        input[idx] = input[N - 1 - idx];
        input[N - 1 - idx] = temp;
    }
}

extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 261888;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverseArray<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}