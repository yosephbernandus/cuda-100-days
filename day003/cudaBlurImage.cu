#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHANNELS 3  // RGB
#define BLUR_SIZE 1 // For a 3x3 blur patch

// Original blurKernel function with only a printf added for thread mapping
__global__ void blurKernel(unsigned char *in, unsigned char *out, int width,
                           int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // Print thread and pixel mapping for select threads
  if (blockIdx.x < 2 && blockIdx.y < 2 && threadIdx.x == 0 &&
      threadIdx.y == 0) {
    printf("Block(%d,%d), Thread(%d,%d) processes Pixel[%d][%d]\n", blockIdx.x,
           blockIdx.y, threadIdx.x, threadIdx.y, row, col);
  }

  if (col < width && row < height) {
    int pixVal = 0;
    int pixels = 0;
    // get average of the surrounding BLUR_SIZE x BLUR_SIZE box
    for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
      for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
        int curRow = row + blurRow;
        int curCol = col + blurCol;
        // verify we have a valid image pixel
        if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
          pixVal += in[curRow * width + curCol];
          ++pixels;
        }
      }
    }
    // write our new pixel value out
    out[row * width + col] = (unsigned char)(pixVal / pixels);
  }
}

// CPU implementation of blur for comparison
void blurCPU(unsigned char *in, unsigned char *out, int width, int height) {
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      int pixVal = 0;
      int pixels = 0;

      // Get average of the surrounding BLUR_SIZE x BLUR_SIZE box
      for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
        for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
          int curRow = row + blurRow;
          int curCol = col + blurCol;

          // Verify we have a valid image pixel
          if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
            pixVal += in[curRow * width + curCol];
            ++pixels;
          }
        }
      }

      // Write our new pixel value out
      out[row * width + col] = (unsigned char)(pixVal / pixels);
    }
  }
}

int main(int argc, char **argv) {
  // Default values
  int width = 512;
  int height = 512;

  // Generate a simple test image (grayscale gradient)
  unsigned char *image = (unsigned char *)malloc(width * height);
  if (!image) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    return 1;
  }

  // Fill with a gradient pattern
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      image[y * width + x] = (x + y) % 256;
    }
  }

  printf("Test image created: %dx%d pixels\n", width, height);

  // Allocate memory for the blurred images
  unsigned char *blurGPU = (unsigned char *)malloc(width * height);
  unsigned char *blurCPU = (unsigned char *)malloc(width * height);

  if (!blurGPU || !blurCPU) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    free(image);
    if (blurGPU)
      free(blurGPU);
    if (blurCPU)
      free(blurCPU);
    return 1;
  }

  // Print sample pixels from the input image
  printf("\nSample pixels from input image:\n");
  for (int y = 0; y < 5 && y < height; y++) {
    for (int x = 0; x < 5 && x < width; x++) {
      printf("Pixel[%d][%d] = %d\n", y, x, image[y * width + x]);
    }
  }

  // ========== Thread to Pixel Mapping Examples ==========
  printf("\nThread to Pixel Mapping Examples:\n");
  int blockSize = 16; // Standard block size we'll use

  for (int by = 0; by < 2; by++) {
    for (int bx = 0; bx < 2; bx++) {
      int baseRow = by * blockSize;
      int baseCol = bx * blockSize;
      printf("Block(%d,%d):\n", bx, by);

      // Show example of first few threads in the block
      for (int ty = 0; ty < 3; ty++) {
        for (int tx = 0; tx < 3; tx++) {
          int pixelRow = baseRow + ty;
          int pixelCol = baseCol + tx;
          if (pixelRow < height && pixelCol < width) {
            printf("  Thread(%d,%d) → Pixel[%d][%d]\n", tx, ty, pixelRow,
                   pixelCol);
          }
        }
      }
    }
  }

  // ========== CPU Implementation ==========
  printf("\n--- Running CPU Implementation ---\n");

  clock_t cpu_start = clock();
  // Apply blur on CPU
  blurCPU(image, blurCPU, width, height);
  clock_t cpu_end = clock();
  double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;

  printf("CPU blur operation: %.4f seconds\n", cpu_time);

  // ========== GPU Implementation ==========
  printf("\n--- Running GPU Implementation ---\n");

  // Allocate device memory
  unsigned char *d_input, *d_output;
  cudaMalloc((void **)&d_input, width * height);
  cudaMalloc((void **)&d_output, width * height);

  // Set up grid and block dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
               (height + blockDim.y - 1) / blockDim.y);

  printf("CUDA grid: %dx%d blocks of %dx%d threads\n", gridDim.x, gridDim.y,
         blockDim.x, blockDim.y);

  clock_t gpu_copy_start = clock();
  // Copy image data to device
  cudaMemcpy(d_input, image, width * height, cudaMemcpyHostToDevice);
  clock_t gpu_copy_end = clock();
  double gpu_copy_time =
      ((double)(gpu_copy_end - gpu_copy_start)) / CLOCKS_PER_SEC;

  clock_t gpu_blur_start = clock();
  // Apply blur on GPU
  blurKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
  cudaDeviceSynchronize();
  clock_t gpu_blur_end = clock();
  double gpu_blur_time =
      ((double)(gpu_blur_end - gpu_blur_start)) / CLOCKS_PER_SEC;

  clock_t gpu_result_start = clock();
  // Copy result back to host
  cudaMemcpy(blurGPU, d_output, width * height, cudaMemcpyDeviceToHost);
  clock_t gpu_result_end = clock();
  double gpu_result_time =
      ((double)(gpu_result_end - gpu_result_start)) / CLOCKS_PER_SEC;

  printf("GPU memory copy to device: %.4f seconds\n", gpu_copy_time);
  printf("GPU blur operation: %.4f seconds\n", gpu_blur_time);
  printf("GPU memory copy to host: %.4f seconds\n", gpu_result_time);
  printf("GPU total time: %.4f seconds\n",
         gpu_copy_time + gpu_blur_time + gpu_result_time);

  // Calculate and print speedups
  printf("\n--- Performance Comparison ---\n");
  printf("Blur operation speedup: %.2fx\n", cpu_time / gpu_blur_time);
  printf("Total speedup (including memory transfers): %.2fx\n",
         cpu_time / (gpu_copy_time + gpu_blur_time + gpu_result_time));

  // Print sample pixels from the GPU blurred result
  printf("\nSample pixels from GPU blurred result:\n");
  for (int y = 0; y < 5 && y < height; y++) {
    for (int x = 0; x < 5 && x < width; x++) {
      printf("Pixel[%d][%d] = %d\n", y, x, blurGPU[y * width + x]);
    }
  }

  // Verify that CPU and GPU results match
  int mismatch_count = 0;
  for (int i = 0; i < width * height; i++) {
    if (blurCPU[i] != blurGPU[i]) {
      mismatch_count++;
      if (mismatch_count <= 5) {
        printf("Mismatch at pixel %d: CPU=%d, GPU=%d\n", i, blurCPU[i],
               blurGPU[i]);
      }
    }
  }

  if (mismatch_count > 0) {
    printf("Total mismatches: %d (%.2f%%)\n", mismatch_count,
           (float)mismatch_count / (width * height) * 100.0f);
  } else {
    printf("CPU and GPU blur results match perfectly!\n");
  }

  // Clean up
  cudaFree(d_input);
  cudaFree(d_output);
  free(image);
  free(blurGPU);
  free(blurCPU);

  return 0;
}
