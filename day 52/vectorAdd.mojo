from gpu.host import DeviceContext
from gpu.id import block_dim, grid_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn vectorAdd(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], N: Int32):
    var idx = Int32(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int32(block_dim.x * grid_dim.x)

    if idx < N:
        C[idx] = A[idx] + B[idx]
        idx += stride

@export
def solve(A: UnsafePointer[Float32], B: UnsafePointer[Float32], C: UnsafePointer[Float32], N: Int32):
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()
    var blocksPerGrid = ceildiv(N, threadsPerBlock)

    ctx.enqueue_function[vectorAdd](
            A, B, C, N,
            grid_dim = blocksPerGrid,
            block_dim = threadsPerBlock
            )
    ctx.synchronize()
    
