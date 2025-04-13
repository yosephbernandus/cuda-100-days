#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHANNELS 3  // RGB
#define BLUR_SIZE 1 // For a 3x3 blur patch

// Original blur kernel
__global__ void blurKernel(unsigned char *in, unsigned char *out, int width,
                           int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

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

// Original CPU implementation
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

// Simple BMP loading function - supports 24-bit BMPs without compression
unsigned char *loadBMP(const char *filename, int *width, int *height) {
  FILE *file = fopen(filename, "rb");
  if (!file) {
    fprintf(stderr, "Error: Could not open file %s\n", filename);
    return NULL;
  }

  // Read BMP header
  unsigned char header[54];
  if (fread(header, 1, 54, file) != 54) {
    fprintf(stderr, "Error: Invalid BMP file (header)\n");
    fclose(file);
    return NULL;
  }

  // Check if it's a BMP file
  if (header[0] != 'B' || header[1] != 'M') {
    fprintf(stderr, "Error: Not a BMP file\n");
    fclose(file);
    return NULL;
  }

  // Extract image dimensions
  *width = *(int *)&header[18];
  *height = *(int *)&header[22];
  int bitsPerPixel = *(short *)&header[28];

  if (bitsPerPixel != 24) {
    fprintf(stderr, "Error: Only 24-bit BMP files are supported\n");
    fclose(file);
    return NULL;
  }

  // Calculate row padding (rows must be a multiple of 4 bytes in BMP)
  int rowSize = ((*width) * 3 + 3) & ~3;
  int imageSize = rowSize * (*height);

  // Allocate memory for the image data
  unsigned char *data = (unsigned char *)malloc((*width) * (*height) * 3);
  if (!data) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    fclose(file);
    return NULL;
  }

  // Read image data
  // BMP stores image bottom-to-top, so we need to flip it
  for (int i = 0; i < *height; i++) {
    // Seek to the start of the row (bottom-to-top)
    fseek(file, 54 + rowSize * ((*height) - 1 - i), SEEK_SET);

    // Read one row
    unsigned char rowBuffer[rowSize];
    if (fread(rowBuffer, 1, rowSize, file) != rowSize) {
      fprintf(stderr, "Error: Failed to read image data\n");
      free(data);
      fclose(file);
      return NULL;
    }

    // Copy the row data (excluding padding)
    for (int j = 0; j < *width; j++) {
      // BMP stores in BGR order, we convert to RGB
      data[(i * (*width) + j) * 3 + 0] = rowBuffer[j * 3 + 2]; // R
      data[(i * (*width) + j) * 3 + 1] = rowBuffer[j * 3 + 1]; // G
      data[(i * (*width) + j) * 3 + 2] = rowBuffer[j * 3 + 0]; // B
    }
  }

  fclose(file);
  return data;
}

// Save a BMP file
void saveBMP(const char *filename, unsigned char *data, int width, int height) {
  FILE *file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr, "Error: Could not create file %s\n", filename);
    return;
  }

  // Calculate row padding (rows must be a multiple of 4 bytes in BMP)
  int rowSize = (width * 3 + 3) & ~3;
  int imageSize = rowSize * height;

  // File header (14 bytes)
  unsigned char fileHeader[14] = {
      'B', 'M',       // Signature
      0,   0,   0, 0, // File size (filled below)
      0,   0,   0, 0, // Reserved
      54,  0,   0, 0  // Offset to pixel data
  };

  // Info header (40 bytes)
  unsigned char infoHeader[40] = {
      40, 0, 0, 0, // Info header size
      0,  0, 0, 0, // Width (filled below)
      0,  0, 0, 0, // Height (filled below)
      1,  0,       // Planes
      24, 0,       // Bits per pixel (24 for color)
      0,  0, 0, 0, // Compression (none)
      0,  0, 0, 0, // Image size (filled below)
      0,  0, 0, 0, // X pixels per meter
      0,  0, 0, 0, // Y pixels per meter
      0,  0, 0, 0, // Colors in color table
      0,  0, 0, 0  // Important colors
  };

  // Fill headers with proper values
  int fileSize = 54 + imageSize; // 54 header + image data
  fileHeader[2] = (unsigned char)(fileSize);
  fileHeader[3] = (unsigned char)(fileSize >> 8);
  fileHeader[4] = (unsigned char)(fileSize >> 16);
  fileHeader[5] = (unsigned char)(fileSize >> 24);

  infoHeader[4] = (unsigned char)(width);
  infoHeader[5] = (unsigned char)(width >> 8);
  infoHeader[6] = (unsigned char)(width >> 16);
  infoHeader[7] = (unsigned char)(width >> 24);

  infoHeader[8] = (unsigned char)(height);
  infoHeader[9] = (unsigned char)(height >> 8);
  infoHeader[10] = (unsigned char)(height >> 16);
  infoHeader[11] = (unsigned char)(height >> 24);

  infoHeader[20] = (unsigned char)(imageSize);
  infoHeader[21] = (unsigned char)(imageSize >> 8);
  infoHeader[22] = (unsigned char)(imageSize >> 16);
  infoHeader[23] = (unsigned char)(imageSize >> 24);

  // Write headers
  fwrite(fileHeader, 1, 14, file);
  fwrite(infoHeader, 1, 40, file);

  // Write image data (bottom-to-top for BMP)
  unsigned char *rowBuffer = (unsigned char *)malloc(rowSize);
  memset(rowBuffer, 0, rowSize); // Initialize with zeros (for padding)

  for (int i = height - 1; i >= 0; i--) {
    // Copy row data to buffer (converting RGB to BGR)
    for (int j = 0; j < width; j++) {
      rowBuffer[j * 3 + 0] = data[(i * width + j) * 3 + 2]; // B
      rowBuffer[j * 3 + 1] = data[(i * width + j) * 3 + 1]; // G
      rowBuffer[j * 3 + 2] = data[(i * width + j) * 3 + 0]; // R
    }

    // Write row
    fwrite(rowBuffer, 1, rowSize, file);
  }

  free(rowBuffer);
  fclose(file);
}

int main(int argc, char **argv) {
  // Default image path if none provided
  const char *inputPath = (argc > 1) ? argv[1] : "input.bmp";
  const char *outputCPUPath = "output_blur_cpu.bmp";
  const char *outputGPUPath = "output_blur_gpu.bmp";

  int width, height;

  // Load the color image
  unsigned char *image = loadBMP(inputPath, &width, &height);
  if (!image) {
    return 1;
  }

  printf("Image loaded: %dx%d pixels\n", width, height);

  // Allocate memory for the output images - renamed variables to avoid conflict
  unsigned char *blur_CPU_result =
      (unsigned char *)malloc(width * height * CHANNELS);
  unsigned char *blur_GPU_result =
      (unsigned char *)malloc(width * height * CHANNELS);

  if (!blur_CPU_result || !blur_GPU_result) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    free(image);
    if (blur_CPU_result)
      free(blur_CPU_result);
    if (blur_GPU_result)
      free(blur_GPU_result);
    return 1;
  }

  // Print sample pixels from the input image
  printf("\nSample pixels from input image:\n");
  for (int y = 0; y < 5 && y < height; y++) {
    for (int x = 0; x < 5 && x < width; x++) {
      int offset = (y * width + x) * CHANNELS;
      printf("Pixel[%d][%d] = RGB(%d, %d, %d)\n", y, x, image[offset],
             image[offset + 1], image[offset + 2]);
    }
  }

  // ========== Thread to Pixel Mapping Examples ==========
  printf("\nThread to Pixel Mapping Examples:\n");
  int blockSize = 16;

  for (int by = 0; by < 2; by++) {
    for (int bx = 0; bx < 2; bx++) {
      int baseRow = by * blockSize;
      int baseCol = bx * blockSize;
      printf("Block(%d,%d):\n", bx, by);

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
  // Apply blur on CPU (one channel at a time)
  for (int c = 0; c < CHANNELS; c++) {
    // Extract channel
    unsigned char *channel_in = (unsigned char *)malloc(width * height);
    unsigned char *channel_out = (unsigned char *)malloc(width * height);

    for (int i = 0; i < width * height; i++) {
      channel_in[i] = image[i * CHANNELS + c];
    }

    // Blur this channel
    blurCPU(channel_in, channel_out, width, height);

    // Put back into color image
    for (int i = 0; i < width * height; i++) {
      blur_CPU_result[i * CHANNELS + c] = channel_out[i];
    }

    free(channel_in);
    free(channel_out);
  }
  clock_t cpu_end = clock();
  double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;

  printf("CPU blur operation: %.4f seconds\n", cpu_time);

  // ========== GPU Implementation ==========
  printf("\n--- Running GPU Implementation ---\n");

  // Set up grid and block dimensions
  dim3 blockDim(16, 16);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
               (height + blockDim.y - 1) / blockDim.y);

  printf("CUDA grid: %dx%d blocks of %dx%d threads\n", gridDim.x, gridDim.y,
         blockDim.x, blockDim.y);

  clock_t gpu_start = clock();

  // Blur each channel separately on GPU
  for (int c = 0; c < CHANNELS; c++) {
    // Extract channel
    unsigned char *channel_in = (unsigned char *)malloc(width * height);
    unsigned char *channel_out = (unsigned char *)malloc(width * height);

    for (int i = 0; i < width * height; i++) {
      channel_in[i] = image[i * CHANNELS + c];
    }

    // Allocate device memory
    unsigned char *d_in, *d_out;
    cudaMalloc((void **)&d_in, width * height);
    cudaMalloc((void **)&d_out, width * height);

    // Copy to device
    cudaMemcpy(d_in, channel_in, width * height, cudaMemcpyHostToDevice);

    // Blur this channel
    blurKernel<<<gridDim, blockDim>>>(d_in, d_out, width, height);
    cudaDeviceSynchronize();

    // Copy back result
    cudaMemcpy(channel_out, d_out, width * height, cudaMemcpyDeviceToHost);

    // Put back into color image
    for (int i = 0; i < width * height; i++) {
      blur_GPU_result[i * CHANNELS + c] = channel_out[i];
    }

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(channel_in);
    free(channel_out);
  }

  clock_t gpu_end = clock();
  double gpu_time = ((double)(gpu_end - gpu_start)) / CLOCKS_PER_SEC;

  printf("GPU blur operation: %.4f seconds\n", gpu_time);
  printf("Speedup: %.2fx\n", cpu_time / gpu_time);

  // Print sample pixels from the blurred results
  printf("\nSample pixels from GPU blurred result:\n");
  for (int y = 0; y < 5 && y < height; y++) {
    for (int x = 0; x < 5 && x < width; x++) {
      int offset = (y * width + x) * CHANNELS;
      printf("Pixel[%d][%d] = RGB(%d, %d, %d)\n", y, x, blur_GPU_result[offset],
             blur_GPU_result[offset + 1], blur_GPU_result[offset + 2]);
    }
  }

  // Save the blurred images
  saveBMP(outputCPUPath, blur_CPU_result, width, height);
  saveBMP(outputGPUPath, blur_GPU_result, width, height);

  printf("\nCPU blurred image saved to %s\n", outputCPUPath);
  printf("GPU blurred image saved to %s\n", outputGPUPath);

  // Clean up
  free(image);
  free(blur_CPU_result);
  free(blur_GPU_result);

  return 0;
}
