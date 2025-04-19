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


# Day 2 - CUDA Learning
File: [colorToGrayscaleConversion.cu](https://github.com/yosephbernandus/cuda-100-days/blob/main/day002/colorToGrayscaleConversion.cu)

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

# Day 3 - CUDA Learning

File: [cudaBlurImage.cu](https://github.com/yosephbernandus/cuda-100-days/blob/main/day003/cudaBlurImage.cu)

## Book Coverage
Chapter 3 - Multidimensional Grids and Data (Image blur: a more complex kernel)

## Concepts Learned
- Blur operation as a stencil computation that averages surrounding pixels
- The concept of a pixel patch (neighborhood) for blur operations
- How BLUR_SIZE determines the radius and dimensions of the blur patch
- Boundary checking for image processing kernels
- Managing thread-to-pixel mapping for operations that use surrounding pixels

## Code Implemented
- Image blur kernel that processes each pixel based on a surrounding patch of pixels
- Boundary condition handling to prevent out-of-bounds memory access
- Channel-by-channel processing for RGB images

## Key Insights
- The BLUR_SIZE parameter defines the radius of the blur patch, making the total size (2×BLUR_SIZE+1)
- For a 3×3 blur patch, BLUR_SIZE=1; for a 7×7 patch, BLUR_SIZE=3
- When processing pixel (25,50) with a 3×3 patch (BLUR_SIZE=1), we need pixels from:
  - Row 24: (24,49), (24,50), (24,51)
  - Row 25: (25,49), (25,50), (25,51)
  - Row 26: (26,49), (26,50), (26,51)
- Boundary checking is critical to prevent accessing invalid memory locations
- For pixels at the image boundaries, we compute averages using fewer surrounding pixels

## Challenges
- Understanding how to calculate the exact pixels needed for each blur operation
- Visualizing the mapping between CUDA threads and the blur patch for each pixel
- Handling image boundaries correctly to avoid memory access errors
- Efficiently implementing the blur algorithm to maximize parallel processing

## Notes
```
The value of BLUR_SIZE is set such that BLUR_SIZE gives the number of pixels on
each side (radius) of the patch and 2*BLUR_SIZE+1 gives the total number of pixels
across one dimension of the patch. For example, for a 3 x 3 patch, BLUR_SIZE is
set to 1, whereas for a 7 x 7 patch, BLUR_SIZE is set to 3. The outer loop iterates
through the rows of the patch. For each row, the inner loop iterates through the
columns of the patch.

In our 3 x 3 patch example, the BLUR_SIZE is 1. For the thread that calculates
output pixel (25, 50), during the first iteration of the outer loop, the curRow vari-
able is row-BLUR_SIZE 5 (25 - 1) = 24

Thus during the first iteration of the outer
loop, the inner loop iterates through the patch pixels in row 24. The inner loop
iterates from column col-BLUR_SIZE = 50-1 = 49 to col+BLUR_SIZE = 51 using
the curCol variable. Therefore the pixels that are processed in the first iteration of
the outer loop are (24, 49), (24, 50), and (24, 51).

The reader should verify that in
the second iteration of the outer loop, the inner loop iterates through pixels (25,
49), (25, 50), and (25, 51). Finally, in the third iteration of the outer loop, the
inner loop iterates through pixels (26, 49), (26, 50), and (26, 51).

During the execution of the nested loop, the curRow and curCol
values for the nine iterations are (21, 2 1), (21,0), (21,1), (0, 2 1), (0,0), (0,1),
(1, 2 1), (1,0), and (1,1). Note that for the five pixels that are outside the image,
at least one of the values is less than 0. The curRow , 0 and curCol , 0 conditions
of the if-statement catch these values and skip the execution of lines 16 and 17.
As a result, only the values of the four valid pixels are accumulated into the run-
ning sum variable.
```



