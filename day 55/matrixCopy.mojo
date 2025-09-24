from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn matrixCopy(A: UnsafePointer[Float32], B: UnsafePointer[Float32], N: Int32):
    var idx: Int32 = block_idx.x * block_dim.x + thread_idx.x
    
    if idx < N * N:
        B[idx] = A[idx]

@export
def solve(A: UnsafePointer[Float32], B: UnsafePointer[Float32], N: Int32):
    var total: Int32 = N * N
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()
    var blocksPerGrid = ceildiv(total, threadsPerBlock)

    ctx.enqueue_function[matrixCopy](
            A, B, N,
            grid_dim = blocksPerGrid,
            block_dim = threadsPerBlock
            )
    ctx.synchronize()

