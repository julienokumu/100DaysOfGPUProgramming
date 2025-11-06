#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define STRIDE 32

__device__ __forceinline__ half warpReduction(half v) {
    unsigned m = __activemask();

    v += __shfl_down_sync(m, v, 16);
    v += __shfl_down_sync(m, v, 8);
    v += __shfl_down_sync(m, v, 4);
    v += __shfl_down_sync(m, v, 2);
    v += __shfl_down_sync(m, v, 1);
    return v;
}

__global__ void dotFP16(const half* A, const half* B, half* result, int N) {
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int bdmx = blockDim.x;

    int idx = bidx * bdmx * STRIDE + tidx;
    int lane = tidx & 31;
    int warp = tidx >> 5;
    int warpsBlock = (bdmx + 31) >> 5;

    half tmp = 0.0f;
    
    for (int i = 0; i < STRIDE; i++) {
        int s_idx = idx + i * bdmx;
        if (s_idx < N) {
            tmp = __hadd(tmp, (__hmul(A[s_idx], B[s_idx])));
        }
    }

    __shared__ half warpMem[32];
    half fullsum = warpReduction(tmp);
    if (lane == 0) {
        warpMem[warp] = fullsum;
    }
    __syncthreads();

    if (warp == 0 && tidx < 32) {
        half val = (tidx < warpsBlock) ? warpMem[tidx] : __float2half(0.0f);
        val = warpReduction(val);
        if (tidx == 0) {
            atomicAdd(result, val);
        }
    }
}

extern "C" void solve(const half* A, const half* B, half* result, int N) {
    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock * STRIDE - 1) / (threadsPerBlock * STRIDE);

    dotFP16<<<blocksPerGrid, threadsPerBlock>>>(A, B, result, N);
}
