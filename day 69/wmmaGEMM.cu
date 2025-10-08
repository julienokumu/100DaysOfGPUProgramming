#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
	int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
	int warpN = blockIdx.y * blockDim.y + threadIdx.y;

	if (warpN * WMMA_N >= N || warpM * WMMA_M >= M) return;

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
			wmma::load_matrix_sync(a_frag, A + a_row * K + a_col, K);
		}
		if (b_row < K && b_col < N) {
			wmma::load_matrix_sync(b_frag, B + b_row * N + b_col, N);
		}
		wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	}

	int c_row = warpM * WMMA_M;
	int c_col = warpN * WMMA_N;

	if (c_row < M && c_col < N) {
		wmma::load_matrix_sync(c_frag, C + c_row * N + c_col, N, wmma::mem_row_major);

		for (int i = 0; i < c_frag.num_elements; i++) {
			c_frag.x[i] = alpha * acc_frag.x[i] + beta * __half2float(c_frag.x[i]);
		}
		wmma::store_matrix_sync(C + c_row * N + c_col, c_frag, N, wmma::mem_row_major);
	}
}
