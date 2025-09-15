#include <cuda_runtime.h>
#include <stdio.h>

__global__ void reverseArray(float* input, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int half  = N / 2;

	if (idx < half) {
		float tmp = input[idx];
		input[idx] = input[N - 1 - idx];
		input[N - 1 - idx] = tmp;
	}
}

int main() {
	float h_input[] = {1.0, 2.0, 3.0, 4.0};
	int N = sizeof(h_input) / sizeof(h_input[0]);
	size_t bytes = N * sizeof(float);

	float* h_output = (float*)malloc(bytes);

	float* d_input;
	cudaMalloc((void **)&d_input, bytes);

	cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	reverseArray<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
	cudaDeviceSynchronize();

	cudaMemcpy(h_output, d_input, bytes, cudaMemcpyDeviceToHost);

	printf("Original Array: ");
	for (int i = 0; i < N; i++) {
		printf("%.1f ", h_input[i]);
	}

	printf("Reversed Array: ");
	for (int i = 0; i < N; i++) {
		printf("%.1f ", h_output[i]);
	}
	printf("/n");

	cudaFree(d_input);
	free(h_output);
	return 0;
}
