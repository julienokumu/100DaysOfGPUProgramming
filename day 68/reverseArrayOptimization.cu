#include <cuda_runtime.h>

__global__ void reverse_array(float* input, float* output, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int half = N / 2;

	if (idx < half) {
		float tmp = input[idx];
		input[idx] = input[N - 1 - idx];
		input[N - 1 - idx] = tmp;
	}
}

extern "C" void solve(float* input, float* output, int N) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
}
