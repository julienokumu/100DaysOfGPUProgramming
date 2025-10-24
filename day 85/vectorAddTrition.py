import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(a_ptr + offsets, mask=mask)
    y = tl.load(b_ptr + offsets, mask=mask)
    output = x + y
    tl.store(c_ptr + offsets, output, mask=mask)
   
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
