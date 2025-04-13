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
