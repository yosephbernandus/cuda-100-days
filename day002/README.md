# Day 2 - CUDA Learning

## Book Coverage
Chapter 3 - Multidimensional Grids and Data (Mapping threads to multidimensional data)

## Concepts Learned
- CUDA thread hierarchy (grids, blocks, threads)
- Variable scope (blockIdx, threadIdx, blockDim, gridDim)
- Thread-to-data mapping formulas
- Memory linearization techniques
- Row-major and column-major memory layouts
- Thread-to-output-data

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

