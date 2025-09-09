#include <cuda_runtime.h>
#include <math.h>

__global__ void silu(const float* input, float* output, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N) {
		float x = input[idx];
		float sigmoid = 1.0f / (1.0f + __expf(-x));
		output[idx] = x * sigmoid;
	}
}

extern "C" void solve(const float* input, float* output, int N) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	silu<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
	cudaDeviceSynchronize();
}
