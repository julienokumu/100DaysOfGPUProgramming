#include <cuda_runtime.h>

__global__ void bitonic_sort(float* data, int N) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            unsigned int ixj = idx ^ j;
            if (ixj > idx && ixj < N) {
                if ((idx & k) == 0) {
                    if (data[idx] > data[ixj]) {
                        float tmp = data[idx];
                        data[idx] = data[ixj];
                        data[ixj] = tmp;
                    }
                } else {
                    if (data[idx] < data[ixj]) {
                        float tmp = data[idx];
                        data[idx] = data[ixj];
                        data[ixj] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }
}

extern "C" void solve(float* data, int N) {
    int ThreadsPerBlock = 256;
    int BlocksPerGrid = (ThreadsPerBlock + N - 1)/ ThreadsPerBlock;
    bitonic_sort<<<BlocksPerGrid, ThreadsPerBlock>>>(data, N);
}
