#include <cuda_runtime.h>

__global__ void ReLU(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }
    output[idx] = max(0.0f, input[idx]);
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    ReLU<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}