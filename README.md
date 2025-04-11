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

