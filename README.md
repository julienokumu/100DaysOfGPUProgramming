**Day 0**: Hello GPU World Kernel

**Resource**: Read Chapter 1 of Programming Massively Parallel Processors

**What I learnt**:
- CUDA kernels are functions executed on the GPU by multiple threads in parallel
- The 'threadIdx.x' variable gives each thread a unique ID within the block
- 'cudaDeviceSynchronize()' ensures the GPU finishes before the program exits
- Learnt about different GPU's like NVIDIA Geforce RTX 4090 and AMD Radeon RX 7900 XTX
- Learnt about how kernels utilize GPU's
- Learnt about the GPU design and how it emphasizes on throughtput
- Learnt some error handling in CUDA
- Learnt about Amdahl's Law

**Challenges Faced**:
- Had to install the CUDA toolkit on my laptop to gain experience
- Due to no personal access to a GPU, I had to compile my code in Google Colab
- Using a nvccjupyter notebook didn't work so I had to manually compile the code
- Mistook 'cudaGetErrorString' for 'cudaGetLastErrorString'

**Performance Observations**:
- Kernel ran instantly, probably because it just prints messages

**Thoughts**:
- What happens when I launch more blocks?
-------------------------------------------------------------------------------------------------------------------------------------

**Day 1**: Vector Addition Kernel

**Resource**: Read chapter 2.1-2.3 of PMPP

**What I learnt**:
- Learnt about Data and Task Parallelism and how they differ
- Learnt about how C pointers work and how we use them in writing CUDA kernels

**Challenges Faced**:
- Understanding the maths in grid_stride loop were quite difficult, the core computation as well
- Took some time to gain intuition on C pointers

**Performance Observations**:
- Kernel ran instantly as well

**Thoughts**:
- I wonder how matrix multiplications work
-------------------------------------------------------------------------------------------------------------------------------------

**Day 2**: Vector Addition Kernel with Timing

**Resources**: Read Chapter 2.4 of PMPP, Read chapter 1-2 of the Best Cuda Practices Guide

**What I learnt**:
- Learnt about the APOD design cycle
- Learnt about CUDA API functions for managing device global memory
- Learnt that CUDA's malloc is a lot like C's
- Leant how to structure my args in cudaMemcpy
- Learnt how the pointer variable should be cast in cudaMalloc
- Learnt aboutr Gustafson's Law
- Solved a vector addition LeetGPU

**Challenges Faced**:
 - Struggled to understand the abstract math behind calculating the global idx and the grid-stride loop
 - Took some time solving a LeetGPU, had to figure out how to write a test program to follow the implementation details
 
**Performance Observations**:
- Compared sequential and parallel addition of vectors on CPU and GPU respectively and achieved a 29x speedup in the computation, which means the sequential method of the CPU must have been really slow
 
**Thoughts**:
- Wondering how many lines of code the largest kernels is
-------------------------------------------------------------------------------------------------------------------------------------

**Day 3**: Array Multiplication Kernel with Timing

**Resources**: Finished Chapter 2 of PMPP

**What I Learnt**:
- Learnt about kernel functions and threading
- Learnt about the built in variables like blockDim.x and threadIdx.x and how many dimensions they handle
- Learnt the difference between SPMD and SIMD
- Learnt about qualifier keywords

**Challenges Faced**:
- NO challenges faced today, getting the hang of the cuda syntax

**Performance Observations**:
- Tested GPU execution time and found that, the larger the number of threads the less the number of blocks and the faster the execution time and viceversa

**Thoughts**:
- How to write 2D kernels

-------------------------------------------------------------------------------------------------------------------------------------

**Day 4**: Array Multiplication Kernel, GPU vs CPU

**Resources**: Read Chapter 3 of PMPP

**What I Learnt**:
- Learnt about Multi-dimensional Grids
- Learnt how to write 2D and 3D arrays

**Challenges Faced**:
- Had a little trouble with some syntax errors but quickly fixed them

**Performance Observations**:
- A GPU achieves a max speedup of 2.82x
- The larger the block size, the faster the kernel and the more the speedup
- CPU execution time was faster on the 64 block size

**Thoughts**:
- Why do CPU's perform better with smaller threads?
-------------------------------------------------------------------------------------------------------------------------------------

**Day 5**: 2D Matrix Addition Kernel

**Resources**: Reread Chapter 3 of PMPP

**What I learnt**:
- Learnt how to handle 2D thread and block indexing
- Learnt how to implement strides
- Created a mental model to help me gain intuition on the dimesions of a grid and block

**Challenges Faced**:
- Took me a bit of time to understand the purpose of stride_x and stride_y

**Thoughts**:
- I wonder how CPU's perform of 2D computations
-------------------------------------------------------------------------------------------------------------------------------------

**Day 6**: 2D Matrix Addition Kernel GPU vs CPU

**Resources**: Chapter 3.4 of PMPP

**What I Learnt**:

- Read about matrix multiplication tiling

**Challenges Faced**:
- Debugging some syntax errors in the code gave me hell, but getting better at debugging cuda code

**Thoughts**:
- Same thought as yesterday, will find an answer
------------------------------------------------------------------------------------------------------------------------------------

**Day 7**: 2D Tiled Matrix Multiplication Kernel with Shared Memory

**Resources**: Chapter 4 of PMPP

**What I learnt**:
- Learnt about tiling and shared memory and how they are key optimization techniques

**Challenges Faced**:
- It was challenging implementing the tiling and understanding the matrix computations

**Thoughts**:
- I wonder how CPUs perform compared to GPU in matrix multiplications
------------------------------------------------------------------------------------------------------------------------------------

**Day 8**: Optimized a previous Array Multiplication Kernel I wrote
- Very busy today but i managed to write this kernel.

**Performance Observations**:
- Removed the grid-stride loop and moved the cudaMemcpy out of the timing loop; doing this significantly improved the gpu execution time, from 0.199ms on 155 blocks to 0.098ms on 157 blocks

---------------------------------------------------------------------------------------------------------------------------

**Day 9**: Reduction Sum Kernel

**Resources**: Chapter 4 of PMPP

**What I learnt**:
- Learnt about the architecture of a modern GPU and block scheduling

**Challenges Faced**:
- Took a bit of time understanding the maths behind the partial sum

**Performance Observations**:
- The faster the block size, the faster the execution time

**Thoughts**:
- What happens if we dont do a partial sun?
----------------------------------------------------------------------------------------------------------------------------

**Day 10**: Metal Flash Attention Kernel

**Resources**: Tinygrad's Generic Metal Flash Attention Kernel

**What I learnt**: 
- Learnt how to write metal kernels, noted the similarities with cuda
- Learnt how to write attention computation

**Challenges Faced**:
- It was difficult translating the code from a cuda-minded perspective

**Thoughts**:
- I wonder how a cuda flash attention kernel is written
------------------------------------------------------------------------------------------------------------------------------

**Day 11**: Reduction Sum Kernel, GPU vs CPU

- wrote a reduction sum kernel and compared performance between gpu and cpu, faced a challenge when rounding off the gpu and cpu sum as they weren't matching but managed to fix it
--------------------------------------------------------------------------------------------------------------------------

**Day 12**: Dot Product Kernel

- wrote a dot product kernel with shared memory, reduction strategy and an atomic operations
- learnt how to profile my kernel with nvprof instead of manually testing for gpu execution time
- noticed that cudaMalloc takes up most of the gpu's execution time
------------------------------------------------------------------------------------------------------------------------------------

**Day 13**: LeetGPU Dot Product Kernel

- solved a medium level leetgpu, implemented a CUDA program that computes the dot product of two vectors containing 32-bit floating point numbers
-----------------------------------------------------------------------------------------------------------------------------

**Day 14**: Matrix Transpose Kernel
- wrote a matrix transpose kernel with shared memory for tiling, memory coelescing and 2D thread indexing
-----------------------------------------------------------------------------------------------------------------------------------

**Day 15**: Hadamard Product Kernel

- wrote a hadamard product kernel coalesced memory access, no shared memory as each element is accessed once and zero __synchthreads(); to avoid synchronization overhead
-------------------------------------------------------------------------------------------------------------------------------------

**Day 16**: LeetGPU Matrix Transpose Example 1

- solved a leetgpu problem: Write a program that transposes a matrix of 32-bit floating point numbers on a GPU. The transpose of a matrix switches its rows and columns. Given a matrix A of dimensions rows x cols, the transpose A^T will have dimensions cols x rows. All matrices are stored in row-major format.
----------------------------------------------------------------------------------------------------------------------------

**Day 17**: LeetGPU Matrix Transpose Example 2

- solved a leetgpu problem: Write a program that transposes a matrix of 32-bit floating point numbers on a GPU. The transpose of a matrix switches its rows and columns. Given a matrix A of dimensions rows x cols, the transpose A^T will have dimensions cols x rows. All matrices are stored in row-major format.
----------------------------------------------------------------------------------------------------------------------------

**Day 18**: GEMM Kernels

- practised solving 6 leetgpu
- wrote a 2 GEMM kernels, first test is on perfomance on different A & B matrix shapes and large m, n, k integers, second test is on performance with same A & B matrix shapes and small m, n, k integers
-------------------------------------------------------------------------------------------------------------------------

**Day 19**: LeetGPU FP16 Gemm Kernel

- solved another medium level leetgpu problem by writing a FP16 gemm...thats 4/42 solved

  ---------------------------------------------------------------------------------------------------------------------------
**Day 20**: Optimized GEMM kernel
- practised solving 3 leetgpu problems, wrote an optimized GEMM kernel by use tiled shared memory

---------------------------------------------------------------------------------------------------------------------------

**Day 21**: L2 Norm Kernel
- wrote a L2 Norm vector kernel with shared memory fusing the normalization computation and division, hence reducing global memory access
- will be practicing leetgpu for the rest of the day

-------------------------------------------------------------------------------------------------------------------------------

**Day 22**: Vector Subtraction Kernel
- wrote a vector subtraction kernel and tested it's performance on 1 million elements
  -----------------------------------------------------------------------------------------------------------------------

**Day 23**: Metal Vector Addition Kernel
- feeling unwell but managed to rewrite on of my cuda vector add kernel to a metal implemention.
- learning how the syntax and attributes differ...metal seems much simpler
--------------------------------------------------------------------------------------------------------------------------

**Day 24**: Metal MatMul Kernel
- turns out adding a grid-stride loop to my vectorAdd kernel increases perf from 19.64GFLOPS to 19.81GFLOPS, that's 0.866% faster
- implemented a naive metal matmul kernel, noticing the differences between cuda and metal
-----------------------------------------------------------------------------------------------------------------------------

**Day 25**: Sigmoid Activation Kernel

- implemented a sigmoid activation kernel
-----------------------------------------------------------------------------------------------------------------------

**Day 26**: ReLU Activation Kernel
- implemented a ReLU activation kernel
- solved a new leetgpu problem(ReLU), thats 5/44 solved now
- will be practicing leetgpu for the rest of the day
------------------------------------------------------------------------------------------------------------------------------------

**Day 27**: LeetGPU's
- practised solving 4 leetgpu's
----------------------------------------------------------------------------------------------------------------------------------

**Day 28**: Max Reduction Kernel
- wrote a single-block max reduction kernel with shared memory
-------------------------------------------------------------------------------------------------------------------------------------

**Day 29**: LeetGPU Matrix Multiplication Kernel
- solved a leetgpu matrix multiplication problem, that's 6/47 now..will stay leetgpu-maxxing for the rest of the day
-------------------------------------------------------------------------------------------------------------------------------------

**Day 30**: LeetGPU Leaky ReLU Kernel
- wrote a leaky relu kernel and solved the problem on leetgpu, thats 7/47 now..stayed leetgpu maxxing for the rest of the day
-----------------------------------------------------------------------------------------------------------------------------------

**Day 31**: LeetGPU Reverse Array Kernel
- solved a reverse array leetgpu problem, thats 8/47 solved now...tried solving a reduction problem but getting issues with memory allocation, will figure it out
-----------------------------------------------------------------------------------------------------------------------------------
**day 32**: Single Block Softmax Kernel
- learnt about a softmax kernel, started off simple by writing a single block softmax kernel avoid any custom atomic operations but used shared memory and parallel reduction and the log-sum-exp trick to avoid overflow
- solved 2 new leepgpu problems, matrix copy and softmax
----------------------------------------------------------------------------------------------------------------------------------

**Day 33**: LeetGPU Dot Product Optimization

- optimized my dot product kernel on leetgpu by replacing the accumulation loop with an unrolling loop by a factor of 4 to enable the kernel process four mulitply-add operation at once, this reduced loop overhead hence becoming currently the fastest kernel on a H200, B200, H100 and second place on an A100-80GB
---------------------------------------------------------------------------------------------------------------------------- 

**Day 34**: LeetGPU Matrix Transpose Optimization

- optimized my matrix transpose kernel on leetgpu by adding tile shared memory, achieving a 10.10x runtime speed up(from 25.79119 to 2.55282)
----------------------------------------------------------------------------------------------------------------------------------

**Day 35**: LeetGPU Matrix Transpose Further Optimization
- further optimized my matrix transpose kernel on leetgpu by using #pragma unroll to reduce branch overhead, __restric__ pointer aliasing to prevent pointing to overlapping memory and experimented with different tile sizes.
- on tile size 8 and tesla t4, my kernel got 1.0198x faster(2.55282ms to 2.50317ms)
-------------------------------------------------------------------------------------------------------------------------------------
**Day 36**: Dot Product LeetGPU Practise
- practiced writing a dot product kernel on leetgpu
------------------------------------------------------------------------------------------------------------------------------------
**Day 37**: Batched Matrix Multiplication Kernel LeetGPU

- learnt about and solved a batched matrix multipliccation leetgpu problem, the fp16 GEMM kernel was kind of difficult to grasp but this BMM has helped me gain deeper intuition, that's 12/50 solved now.
- will be practicing writing fp16 GEMM and BMM kernels for the rest of the day.
---------------------------------------------------------------------------------------------------------------------------------

**Day 38**: Optimized Batched Matrix Multiplication Kernel LeetGPU

- optimized my batched matrix multiplication kernel on leetgpu by adding shared memory, making my kernel 1.20x faster(3.53275s to 2.94001 ms)
----------------------------------------------------------------------------------------------------------------------------

**Day 39**: LeetGPU Practice
- practiced solving 6 leetgpu problems
-----------------------------------------------------------------------------------------------------------------------------------

**Day 40**: Sigmoid Linear Unit Kernel LeetGPU

- learnt about and solved a sigmoid linear unit kernel problem on leetgpu solving 13/51 problems now
- i'll stay leetgpu-maxxing for the rest of the day
---------------------------------------------------------------------------------------------------------------------------

**Day 41**: Matrix Transpose Kernel practice
- practiced writing a matrix transpose kernel
