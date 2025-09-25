from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn leaky_relu(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    var idx: Int32 = block_idx.x * block_dim.x + thread_idx.x

    if idx < N:
        var val = input[idx]
        output[idx] = val if val >= 0 else 0.01 * val

@export
def solve(input: UnsafePointer[Float32], output: UnsafePointer[Float32], N: Int32):
    var threadsPerBlock: Int32 = 256
    var ctx = DeviceContext()
    var blocksPerGrid = ceildiv(N, blocksPerGrid)

    ctx.enqueue_function[leaky_relu](
            input, output, N,
            grid_dim = blocksPerGrid,
            block_dim = threadsPerBlock
            )
    ctx.synchronize()
