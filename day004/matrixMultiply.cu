#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Cuda kernel for matrix multiplication
__global__ void MatrixMulKernel(float *M, float *N, float *P, int Width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((row < Width) && (col < Width)) {
    float Pvalue = 0;
    for (int k = 0; k < Width; ++k) {
      Pvalue += M[row * Width + k] * N[k * Width + col];
    }
    P[row * Width + col] = Pvalue;
  }
}

// function to initialize a matrix with values
void initializeMatrix(float *matrix, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      matrix[i * size + j] = (float)(rand() % 10);
    }
  }
}

// function to print a matrix
void printMatrix(float *matrix, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      printf("%f ", matrix[i * size + j]);
    }
    printf("\n");
  }
  printf("\n");
}

// CPU matrix multiplication
void matrixMultiply(float *M, float *N, float *P, int Width) {
  for (int row = 0; row < Width; row++) {
    for (int col = 0; col < Width; col++) {
      P[row * Width + col] = 0;
      for (int k = 0; k < Width; k++) {
        P[row * Width + col] += M[row * Width + k] * N[k * Width + col];
      }
    }
  }
}

// Function to compare matrices
int compareMatrices(float *A, float *B, int size, float tolerance) {
  for (int i = 0; i < size * size; i++) {
    if (fabs(A[i] - B[i]) > tolerance) {
      printf("Mismatch at element %d: GPU = %f, CPU = %f\n", i, A[i], B[i]);
      return 0;
    }
  }
  return 1;
}

int main() {
  // Define matrix size
  const int Width = 16;
  const int size = Width * Width;
  const int mem_size = size * sizeof(float);

  // Allocate host memory
  float *h_M = (float *)malloc(mem_size);
  float *h_N = (float *)malloc(mem_size);
  float *h_P_GPU = (float *)malloc(mem_size);
  float *h_P_CPU = (float *)malloc(mem_size);

  // Initialize matrices M and N
  srand(42); // for reproducible results
  initializeMatrix(h_M, Width);
  initializeMatrix(h_N, Width);

  // Print input matrices
  printf("Matrix M:\n");
  printMatrix(h_M, Width);
  printf("Matrix N:\n");
  printMatrix(h_N, Width);

  // Allocate device memory
  float *d_M, *d_N, *d_P;
  cudaMalloc((void **)&d_M, mem_size);
  cudaMalloc((void **)&d_N, mem_size);
  cudaMalloc((void **)&d_P, mem_size);

  // Copy host memory to device
  cudaMemcpy(d_M, h_M, mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, h_N, mem_size, cudaMemcpyHostToDevice);

  // Set execution configuration
  dim3 dimBlock(16, 16);
  dim3 dimGrid((Width + dimBlock.x - 1) / dimBlock.x,
               (Width + dimBlock.y - 1) / dimBlock.y);

  // Launch kernel
  MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  // Copy result from device to host
  cudaMemcpy(h_P_GPU, d_P, mem_size, cudaMemcpyDeviceToHost);

  // Print the GPU result matrix
  printf("Matrix P (GPU Result):\n");
  printMatrix(h_P_GPU, Width);

  // CPU matrix multiplication for verification
  printf("Performing CPU matrix multiplication for verification...\n");
  matrixMultiply(h_M, h_N, h_P_CPU, Width);

  // Print the CPU result matrix
  printf("Matrix P (CPU Result):\n");
  printMatrix(h_P_CPU, Width);

  // Compare GPU and CPU results
  printf("Comparing GPU and CPU results...\n");
  if (compareMatrices(h_P_GPU, h_P_CPU, Width, 1e-5)) {
    printf("Results match! GPU and CPU computed the same answer.\n");
  } else {
    printf("ERROR: Results don't match!\n");
  }

  // Free device memory
  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);

  // Free host memory
  free(h_M);
  free(h_N);
  free(h_P_GPU);
  free(h_P_CPU);

  return 0;
}
