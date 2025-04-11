#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHANNELS 3 // RGB

// CUDA kernel for grayscale conversion
__global__ void colortoGrayscaleConvertion(unsigned char *Pout,
                                           unsigned char *Pin, int width,
                                           int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    // Get 1D offset for the grayscale image
    int grayOffset = row * width + col;
    // One can think of the RGB image having channel
    // times more columns than the gray scale image
    int rgbOffset = grayOffset * CHANNELS;
    unsigned char r = Pin[rgbOffset];     // Red value
    unsigned char g = Pin[rgbOffset + 1]; // Green value
    unsigned char b = Pin[rgbOffset + 2]; // Blue value
    // Perform the rescaling and store it
    // We multiply by floating point constants
    Pout[grayOffset] = (0.21f * r + 0.71f * g + 0.07f * b);
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

// Save a grayscale image as BMP
void saveGrayscaleBMP(const char *filename, unsigned char *data, int width,
                      int height) {
  FILE *file = fopen(filename, "wb");
  if (!file) {
    fprintf(stderr, "Error: Could not create file %s\n", filename);
    return;
  }

  // Calculate row padding (rows must be a multiple of 4 bytes in BMP)
  int rowSize = (width + 3) & ~3;
  int imageSize = rowSize * height;

  // File header (14 bytes)
  unsigned char fileHeader[14] = {
      'B', 'M',       // Signature
      0,   0,   0, 0, // File size (filled below)
      0,   0,   0, 0, // Reserved
      54,  4,   0, 0  // Offset to pixel data (54 + 256*4 = 1078)
  };

  // Info header (40 bytes)
  unsigned char infoHeader[40] = {
      40, 0, 0, 0, // Info header size
      0,  0, 0, 0, // Width (filled below)
      0,  0, 0, 0, // Height (filled below)
      1,  0,       // Planes
      8,  0,       // Bits per pixel (8 for grayscale)
      0,  0, 0, 0, // Compression (none)
      0,  0, 0, 0, // Image size (filled below)
      0,  0, 0, 0, // X pixels per meter
      0,  0, 0, 0, // Y pixels per meter
      0,  1, 0, 0, // Colors in color table (256)
      0,  1, 0, 0  // Important colors (256)
  };

  // Fill headers with proper values
  int fileSize =
      54 + 1024 + imageSize; // 54 header + 1024 color table + image data
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

  // Write color table (required for 8-bit grayscale)
  for (int i = 0; i < 256; i++) {
    unsigned char color[4] = {(unsigned char)i, (unsigned char)i,
                              (unsigned char)i, 0};
    fwrite(color, 1, 4, file);
  }

  // Write image data (bottom-to-top for BMP)
  unsigned char *rowBuffer = (unsigned char *)malloc(rowSize);
  memset(rowBuffer, 0, rowSize); // Initialize with zeros (for padding)

  for (int i = height - 1; i >= 0; i--) {
    // Copy row data to buffer
    for (int j = 0; j < width; j++) {
      rowBuffer[j] = data[i * width + j];
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
  const char *outputPath = (argc > 2) ? argv[2] : "output_gray.bmp";

  int width, height;

  // Load the image
  unsigned char *image = loadBMP(inputPath, &width, &height);
  if (!image) {
    return 1;
  }

  printf("Image loaded: %dx%d pixels\n", width, height);

  // Print sample pixels in a 2D format (5x5 grid from top-left)
  printf("\nSample pixels from loaded image:\n");
  for (int y = 0; y < 5 && y < height; y++) {
    for (int x = 0; x < 5 && x < width; x++) {
      // Calculate the 1D offset for this pixel
      int offset = (y * width + x) * CHANNELS;
      
      // Get RGB values
      unsigned char r = image[offset];
      unsigned char g = image[offset + 1];
      unsigned char b = image[offset + 2];
      
      printf("Pixel[%d][%d] = RGB(%d, %d, %d)\n", 
             y, x, r, g, b);
    }
  }

  // Allocate memory for the grayscale image
  unsigned char *gray = (unsigned char *)malloc(width * height);
  if (!gray) {
    fprintf(stderr, "Error: Memory allocation failed for grayscale image\n");
    free(image);
    return 1;
  }

  // Allocate device memory
  unsigned char *d_input, *d_output;
  cudaMalloc((void **)&d_input, width * height * CHANNELS);
  cudaMalloc((void **)&d_output, width * height);

  // Copy image data to device
  cudaMemcpy(d_input, image, width * height * CHANNELS, cudaMemcpyHostToDevice);

  // Set up grid and block dimensions
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  printf("\nCUDA grid: %dx%d blocks of %dx%d threads\n", 
         gridSize.x, gridSize.y, blockSize.x, blockSize.y);

  // Launch the kernel
  colortoGrayscaleConvertion<<<gridSize, blockSize>>>(d_output, d_input, width,
                                                      height);

  // Check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    cudaFree(d_input);
    cudaFree(d_output);
    free(image);
    free(gray);
    return 1;
  }

  // Copy result back to host
  cudaMemcpy(gray, d_output, width * height, cudaMemcpyDeviceToHost);

  // Print sample pixels from the grayscale result
  printf("\nSample pixels from grayscale result:\n");
  for (int y = 0; y < 5 && y < height; y++) {
    for (int x = 0; x < 5 && x < width; x++) {
      // Calculate the 1D offset for this pixel
      int offset = y * width + x;
      
      printf("Pixel[%d][%d] = %d\n", 
             y, x, gray[offset]);
    }
  }

  // Save the grayscale image
  saveGrayscaleBMP(outputPath, gray, width, height);
  printf("\nGrayscale image saved to %s\n", outputPath);

  // Clean up
  cudaFree(d_input);
  cudaFree(d_output);
  free(image);
  free(gray);

  return 0;
}
