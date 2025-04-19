# Day 4 - CUDA Learning

File: [matrixMultiply.cu](https://github.com/yosephbernandus/cuda-100-days/blob/main/day004/matrixMultiply.cu)

## Book Coverage
Chapter 3 - Multidimensional Grids and Data (Image blur: a more complex kernel)

## Diagram Explain
This diagram shows a 4×4 result matrix P with BLOCK_WIDTH = 2. The matrix is divided into 4 tiles (blocks):

- Block(0,0): Calculates elements P₀,₀, P₀,₁, P₁,₀, P₁,₁
- Block(0,1): Calculates elements P₀,₂, P₀,₃, P₁,₂, P₁,₃
- Block(1,0): Calculates elements P₂,₀, P₂,₁, P₃,₀, P₃,₁
- Block(1,1): Calculates elements P₂,₂, P₂,₃, P₃,₂, P₃,₃

Within each block, we have a 2×2 grid of threads:

- Thread(0,0) in Block(0,0) calculates P₀,₀
- Thread(0,1) in Block(0,0) calculates P₀,₁
- Thread(1,0) in Block(0,0) calculates P₁,₀
- Thread(1,1) in Block(0,0) calculates P₁,₁

And so on for other blocks.## Code Implemented

## Key Insights
- Thread Indices Map to Matrix Indices
The text explains how row and col are calculated:
``` 
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
```
For example, in a system with BLOCK_WIDTH = 2:
Thread(1,0) in Block(0,0) has:

  - blockIdx.y = 0, blockDim.y = 2, threadIdx.y = 1
  - blockIdx.x = 0, blockDim.x = 2, threadIdx.x = 0
  - So row = 02 + 1 = 1, col = 02 + 0 = 0
  - This thread calculates P₁,₀

- Step By Step Element Calculation
Figure 3.13 shows what happens when Thread(0,0) in Block(0,0) calculates element P₀,₀:

1. The thread computes the dot product between:

  - The 0th row of matrix M: [M₀,₀, M₀,₁, M₀,₂, M₀,₃]
  - The 0th column of matrix N: [N₀,₀, N₁,₀, N₂,₀, N₃,₀]

2. The thread's for-loop iterations access:

  - Iteration 0 (k=0): M₀,₀ and N₀,₀
  - Iteration 1 (k=1): M₀,₁ and N₁,₀
  - Iteration 2 (k=2): M₀,₂ and N₂,₀
  - Iteration 3 (k=3): M₀,₃ and N₃,₀

3. The formula for memory access is:

  - M element: rowWidth + k = 04 + k
  - N element: kWidth + col = k4 + 0

4. Finally, the result P₀,₀ = M₀,₀×N₀,₀ + M₀,₁×N₁,₀ + M₀,₂×N₂,₀ + M₀,₃×N₃,₀ is stored at P[row*Width+col]

- Thread-Element Mapping: Each CUDA thread calculates exactly one element in the result matrix.

- Memory Access Pattern:
  - Each thread accesses an entire row of M and an entire column of N
  - The linear memory accesses are calculated using offsets: row*Width + k for M and k*Width + col for N


- Block Organization:
  - The output matrix is divided into blocks
  - Each block contains a BLOCK_WIDTH × BLOCK_WIDTH grid of threads
  - This division allows CUDA to efficiently manage the parallel execution

- Scalability:
  - This approach has limitations based on the maximum number of blocks per grid and threads per block
  - For very large matrices, you would need to either: 
  a) Divide the matrix into submatrices and process them separately
  b) Modify the kernel so each thread calculates multiple elements

