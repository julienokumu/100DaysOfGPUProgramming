#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = width * height;
    if(idx < N) {
        image[idx * 4 + 0] = 255 - image[idx * 4 + 0];
        image[idx * 4 + 1] = 255 - image[idx * 4 + 1];
        image[idx * 4 + 2] = 255 - image[idx * 4 + 2];
    }
}

extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
}
