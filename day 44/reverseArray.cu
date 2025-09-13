#include cuda_runtime.h>

__global__ void reverseArray(float* input, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int half = N / 2;

	if (idx < half) {
		float tmp = input[idx];
		input[idx] = input[N - 1 - idx];
		input[N - 1 - idx] = tmp;
	}
}
