#include <cuda_runtime.h>

__global__ void leakyReLU(const float* input, float* output, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N) {
		if (input[idx] > 0) {
			output[idx] = input[idx];
		} else {
			output[idx] = 0.01f * input[idx];
		}
	}
}

extern "C" void solve(const float* input, float* output, int N) {
	float* d_input, *d_output;
	size_t bytes = N * sizeof(float);

	cudaMalloc((void **)&d_input, bytes);
	cudaMalloc((void **)&d_output, bytes);

	cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;\
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	leakyReLU<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);\
	cudaDeviceSynchronize();
}
