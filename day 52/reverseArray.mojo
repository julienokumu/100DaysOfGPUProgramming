from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

fn reverseArray(input: UnsafePointer[Float32], N: Int32):
    var idx = Int32(block_idx.x * block_dim.x + thread_idx.x)

    if idx < N // 2:
        var tmp: Float32 = input[idx]
        input[idx] = input[N - 1 - idx]
        input[N - 1 - idx] = tmp

@export
def solve(input: UnsafePointer[Float32], N: Int32):
    var threadsPerBlock: Int32 = 261888
    var ctx = DeviceContext()
    var blocksPerGrid = ceildiv(N, threadsPerBlock)

    ctx.enqueue_function[reverseArray](
            input, N,
            grid_dim = blocksPerGrid,
            block_dim = threadsPerBlock
            )
    ctx.synchronize()
