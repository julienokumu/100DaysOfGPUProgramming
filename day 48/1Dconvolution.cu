#include <cuda_runtime.h>

__global__ void conv_1d(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
	extern __shared__ float kernelMem[];

	int tx = threadIdx.x;
	int bDimx = blockDim.x;
	int i = blockIdx.x * bDimx + tx;

	for (int j = tx; j < kernel_size; j += bDimx) {
		kernelMem[j] = kernel[j];
	}
	__syncthreads();

	if (i >= output_size) return;
	float sum = 0.0f;

	for (int j = 0; j < kernel_size; j++) {
		sum += input[i + j] * kernelMem[j];
	}
	output[i] = sum;
}

extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
	int output_size = input_size - kernel_size + 1;
	int threadsPerBlock = 1024;
	int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
	size_t kernelMemBytes = kernel_size * sizeof(float);

	conv_1d<<<blocksPerGrid, threadsPerBlock, kernelMemBytes>>>(input, kernel, output, input_size, kernel_size, output_size);
	cudaDeviceSynchronize();
}
