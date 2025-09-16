#include <cuda_runtime.h>

__global__ void conv_1d(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
	extern __shared__ float kMem[];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int output_size = input_size - kernel_size + 1;

	for (int j = threadIdx.x; j < kernel_size; j += blockDim.x) {
		kMem[j] = kernel[j];
	}
	__syncthreads();

	if (i >= output_size) return;
	float sum = 0.0f;
	for (int j = 0; j < kernel_size; j++) {
		sum += input[i + j] * kMem[j];
	}
	output[i] = sum;
}

extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
	int output_size = input_size - kernel_size + 1;
	int threadsPerBlock = 1024;
	int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
	size_t kMemBytes = kernel_size * sizeof(float);

	conv_1d<<<blocksPerGrid, threadsPerBlock, kMemBytes>>>(input, kernel, output, input_size, kernel_size);
	cudaDeviceSynchronize();
}
