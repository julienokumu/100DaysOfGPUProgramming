#include <cuda_runtime.h>

__global__ void gausBlur(const float* input, const float* kernel, float* output, int input_rows, int input_cols, int kernel_rows, int kernel_cols){
    int j = threadIdx.x + blockIdx.x * blockDim.x; // column
    int i = threadIdx.y + blockIdx.y * blockDim.y; // row
    int start_height=kernel_rows/2;
    int start_length=kernel_cols/2;
    float sum=0.0f;
    if(i < input_rows && j < input_cols){
        for(int m=-start_height;m<=start_height;m++){
            for(int n=-start_length;n<=start_length;n++){
                int row=i+m;
                int col=j+n;
                float val=0.0f;
                if(row >= 0 && row < input_rows && col >= 0 && col < input_cols){
                    val=input[(row)*input_cols+(col)];
                    sum+=val * kernel[(m+start_height)*kernel_cols+(n+start_length)];
                }
            }
        }
    output[i*input_cols + j] = sum;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((input_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (input_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
  

    gausBlur<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_rows, input_cols,kernel_rows,kernel_cols);
}
