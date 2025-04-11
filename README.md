# Cuda on 100 days learning

Tools to generate directory:
- ./create_dir.sh --days 5

## Resource That I Used
- Book Programming Massively Prallel Processors

# Day 1 - CUDA Learning

File: [vec-add.cu](https://github.com/yosephbernandus/cuda-100-days/blob/main/day001/vec-add.cu)

## Book Coverage
Chapter 1 - Introduction
Chapter 2 - Heterogeneous data parallel

## Concepts Learned
- Cuda program initiates parallel execution by calling kernel function -> Runtime Mechanism launch grid of threads
- `cudaMalloc` Allocated object in device global memory, 2 params address of a pointer and size
- `cudaFree` frees object from device global memory
- `cudaMemcpy` memory data transfer, 4 params pointer to destination, pointer to source, number of bytes copied, type/direction transfer
- `cudaError_t` error checking and handling in cuda
- Each grid is organized as an array of thread blocks
- ALl blocks of a grid are the same size, each block can contain up to 1024 threads
- `blockDim` is a struct with three unsigned integer fields (x, y, and z) can be 3 dimensional array, for one dimensional only use x
- `threadIdx`
- `blockIdx`
- cuda code compiled using nvcc (NVIDIA C Compiler) there is 2 condition Host Code to host C preprocessor compiler and linker and PTX (Device Code) for doing kernel function in device


## Code Implemented
- Create cuda kernel for adding two vector
- cudaMalloc and cudaFree
```float *A_d
int size=n*sizeof(float);
cudaMalloc((void **)&A_d, size);
cudaFree(A_d);
```
- cudaMemcpy
```
cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
```
- cudaError_t
```
cudaError_t
err 5 cudaMalloc((void ** ) &A_d, size);
if (error!=cudaSuccess)
{
    printf(“%s in %s at line %d\n”, cudaGetErrorString(err),  __FILE__, __LINE__);
    exit(EXIT_FAILURE);
}
```

## Key Insights
- GPU focus to throughput oriented, cpu to latency oriented
- Execution speed of many applications limited by memory access latency, and or throughput. This consider to doing some techniques for optimize memory access
- Writing basic cuda flow process from allocated until copied to cuda device

## Challenges
- Need to understand basic C and C++

## Notes
- GPU Have dram also cache (L1, L2, L3) like on CPU
- Also the PC (Program counter) and SP (Stack Pointer) and the ALU (Arithmetic Logic Unit)


# Day 2

## Book Coverage
Chapter 3 - Multidimensional Grids and Data

## Concepts Learned
- CUDA thread hierarchy (grids, blocks, threads)
- Variable scope (blockIdx, threadIdx, blockDim, gridDim)
- Thread-to-data mapping formulas
- Memory linearization techniques
- Row-major and column-major memory layouts

## Code Implemented
- Basic image processing example for converting color to grayscale

## Key Insights
- CUDA uses a hierarchical thread organization that maps naturally to multidimensional data like images
- The formula blockIdx.x * blockDim.x + threadIdx.x converts 3D thread coordinates to a global position
- Thread blocks are limited to 1024 threads total, which can be arranged in any 3D configuration that doesn't exceed this limit
- Memory in computers is fundamentally linear, requiring techniques to map multidimensional data to 1D memory

## Challenges
- Understanding how thread coordinates map to pixel positions
- Grasping the difference between block indices and thread indices
- Visualizing the thread hierarchy across multiple dimensions
- Converting 2D/3D array indices to 1D memory locations

## Notes
1. **CUDA Thread Hierarchy**:
   - All threads in a block share the same `blockIdx` value
   - Each thread has its own unique `threadIdx` value
   - `gridDim` and `blockDim` define the dimensions of the grid and block

2. **Thread Coordinate Formulas**:
```
Vertical (row) coordinate = blockIdx.y * blockDim.y + threadIdx.y
Horizontal (column) coordinate = blockIdx.x * blockDim.x + threadIdx.x
```

3. **Grid and Block Dimensions**:
- Defined using the `dim3` data type: vector of three integers (x, y, z)
- Example: `dim3 dimGrid(ceil(n/256.0), 1, 1); dim3 dimBlock(256, 1, 1);`
- Range limits: 
  - gridDim.x: 1 to 2³¹-1
  - gridDim.y/z: 1 to 2¹⁶-1
  - Total threads per block: maximum 1024

4. **Memory Linearization**:
- Row-major layout: elements of same row in consecutive locations
- 2D to 1D conversion: `index = row * width + column`
- Color pixel example: `index = row * width * 3 + column` (×3 for RGB)
- Example: For matrix M, element at row 2, column 1 has 1D index: 2×4+1 = 9

5. **Handling Image Boundaries**:
- Threads must check if their position is within the image bounds
- For a 62×76 image with 16×16 blocks, we need 5×4=20 blocks
- Some threads will be outside the image bounds and must be skipped

