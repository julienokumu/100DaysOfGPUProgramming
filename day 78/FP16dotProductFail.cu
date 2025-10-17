//Failed Attempt

#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void dotProduct(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ result, int N) {
    extern __shared__ half dotMem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float tmp = 0.0f;

    #pragma unroll
    for (unsigned int i = idx; i < N; i += stride) {
        tmp += __half2float(A[i]) * __half2float(B[i]);
    }
    dotMem[tid] = tmp;
    __syncthreads();

    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            dotMem[tid] += dotMem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, dotMem[0]);
    }
}

extern "C" void solve(const half* A, const half* B, half* result, int N) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t dotMemBytes = threadsPerBlock * sizeof(half);

    dotProduct<<<blocksPerGrid, threadsPerBlock, dotMemBytes>>>(A, B, result, N);
}