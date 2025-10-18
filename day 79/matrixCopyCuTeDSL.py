import cutlass
import cutlass.cute as cute


@cute.kernel
def copy_kernel(A:cute.Tensor, B: cute.Tensor, N: cute.Int32):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdmy, _ = cute.arch.block_dim()

    idx_x = bidx * bdimx + tidx
    idx_y = bidy * bdimx + tidy

    if idx_x < N and idx_y < N:
        B[idx_x, idx_y] = A[idx_x, idx_y]
    
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, N: cute.Int32):
    threads = 16
    blocks = cute.ceil_div(N, threads)

    kernel = copy_kernel(A, B, N)
    kernel.launch(grid=(blocks, blocks, 1), block=(threads, threads, 1))