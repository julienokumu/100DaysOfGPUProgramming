#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void conv_2d_kernel(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float tmp = 0.0f;

    if (row < input_rows - kernel_rows + 1 && col < input_cols - kernel_cols + 1) {
        for (int i = 0; i < kernel_rows; i++) {
            #pragma unroll 4
            for (int j = 0; j < kernel_cols; j++) {
                tmp += input[(row + i) * input_cols + (col + j)] * kernel[i * kernel_cols + j];
            }
        }
        output[row * (input_cols - kernel_cols + 1) + col] = tmp;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((output_cols + TILE_SIZE - 1) / TILE_SIZE,
                       (output_rows + TILE_SIZE - 1) / TILE_SIZE);

    conv_2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
