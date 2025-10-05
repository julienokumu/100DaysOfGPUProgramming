import cutlass
import cutlass.cute as cute

@cute.kernel
def vectorAdd(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint64):
    bidx,_,_ = cute.arch.block_idx()
    bdmx,_,_ = cute.arch.block_dim()
    tidx,_,_ = cute.arch.thread_idx()

    idx = bidx * bdmx + tidx

    if idx < N:
        C[idx] = A[idx] + B[idx]

@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint64):
    threadsPerBlock = 256

    kernel = vectorAdd(A, B, C, N)

    kernel.launch(grid=(N // threadsPerBlock) + 1, 1,
                  block=(threadsPerBlock, 1, 1))

