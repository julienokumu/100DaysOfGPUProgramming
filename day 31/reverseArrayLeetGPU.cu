#include <cuda_runtime.h>

__global__ void reverseArray(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    for (int i = idx; i < N / 2; i += totalThreads) {
        float temp = input[i];
        input[i] = input[N - 1 - i];
        input[N - 1 - i] = temp;
    }
}

extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverseArray<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
