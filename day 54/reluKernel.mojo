from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn relu_kernel(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    var idx: Int32 = block_idx.x * block_dim.x + thread_idx.x

    if idx < N:
        output[idx] = max(0, input[idx])

@export
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()
    var blocksPerGrid = ceildiv(N, threadsPerBlock)

    ctx.enqueue_function[relu_kernel](
            input, output, N,
            grid_dim = blocksPerGrid,
            block_dim = threadsPerBlock
            )
    ctx.synchronize()

