import cutlasss
import cutlass.cute as cute

@cute.kernel
def relu_kernel(input: cute.Tensor, output: cute.Tensor, N: cute.Uint64):
    bidx,_,_ = cute.arch.block_idx()
    bdmx,_,_ = cute.arch.block_dim()
    tidx,_,_ = cute.arch.thread_idx()

    idx = bidx * bdmx + tidx

    if idx < N:
        output[idx] = max(0.0, input[idx])

@cute.jit
def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Uint64):
    threadsPerBlock = 256

    kernel = relu_kernel(input, output, N)

    kernel.launch(grid=((N // threadsPerBlock) + 1, 1, 1),
                  block=(threadsPerBlock, 1, 1))
