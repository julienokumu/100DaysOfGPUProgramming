#include <cuda_runtime.h>

__global__ void dotProduct(const float* A, const float* B, float* result, int N) {
    extern __shared__ float sharedMem[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (idx < N) {
        sum += A[idx] * B[idx];
        idx += blockDim.x * gridDim.x;
    }
    sharedMem[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
        if (tid < s) {
            sharedMem[tid] += sharedMem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sharedMem[0]);
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    float *d_A, *d_B, *d_result;
    size_t bytes = N * sizeof(float);
    cudaMemset(result, 0, sizeof(float));

    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_result, sizeof(float));

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    dotProduct<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_A, d_B, d_result, N);
    cudaDeviceSynchronize();

    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);
}
