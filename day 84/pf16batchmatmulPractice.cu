#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define BLOCK_SIZE 32
#define WARP_SIZE 32
#define TILE_WIDTH 4
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_batched(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        int a_row = warpM * WMMA_M;
        int a_col = k;
        
        int b_col = warpN * WMMA_N;
        int b_row = k;

        if (a_row < M && a_col < K) {
            wmma::load_matrix_sync(a_frag, A + b * M * K + a_row * K + a_col, K);
        }

        if (b_col < N && b_row < K) {
            wmma::load_matrix_sync(b_frag, B + b * N * K + b_row * N + b_col, N);
        }
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    int c_row = warpM * WMMA_M;
    int c_col = warpN * WMMA_N;

    if (c_row < M && c_col < N) {
        wmma::load_matrix_sync(c_frag, C + b * M * N + c_row * N + c_col, N, wmma::mem_row_major);

        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = __float2half(acc_frag.x[i]);
        }
        wmma::store_matrix_sync(C + b * M * N + c_row * N + c_col, c_frag, N, wmma::mem_row_major);
    }
}

__global__ void batched_matmul(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, int M, int N, int K) {
    __shared__ half As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int b = blockIdx.z;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE * TILE_WIDTH + tx;

    float tmp[TILE_WIDTH] = {0.0f};

    #pragma unroll
    for (unsigned int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        if (row < M && (t * BLOCK_SIZE + tx) < K) {
            As[ty][tx] = A[b * M * K + row * K + t * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        #pragma unroll
        for (unsigned int i = 0; i < TILE_WIDTH; i++) {
            int col_idx = col + i * BLOCK_SIZE;
            if (col_idx < N && (t * BLOCK_SIZE + ty) < K) {
                Bs[ty][tx] = B[b * N * K + (t * BLOCK_SIZE + ty) * N  + col_idx];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            __syncthreads();

            #pragma unroll
            for (unsigned int k = 0; k < BLOCK_SIZE; k++) {
                tmp[i] += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (unsigned int i = 0; i < TILE_WIDTH; i++) {
        int col_idx = col + i * BLOCK_SIZE;
        if (row < M && col_idx < N) {
            C[b * M * N + row * N + col_idx] = tmp[i];
        }
    }
}

extern "C" void solve(const half* A, const half* B, half* C, int b, int M, int N, int K) {
    bool use_wmma = (M % WMMA_M == 0) && (N % WMMA_N == 0) && (K % WMMA_K == 0);

    if (use_wmma) {
        dim3 threadsPerBlock(WARP_SIZE, 4);
        dim3 blocksPerGrid((M + (WMMA_M * threadsPerBlock.x/ WARP_SIZE) - 1) / (WMMA_M * threadsPerBlock.x / WARP_SIZE),
                           (N + (WMMA_N * threadsPerBlock.y) - 1) / (WMMA_N * threadsPerBlock.y), b);
        
        wmma_batched<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    } else {
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocksPerGrid((N + BLOCK_SIZE * TILE_WIDTH - 1) / (BLOCK_SIZE * TILE_WIDTH),
                           (M + BLOCK_SIZE - 1) / BLOCK_SIZE, b);
        
        batched_matmul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    }
}
