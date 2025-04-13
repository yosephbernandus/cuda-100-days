# TODO: Replace SOURCE with your actual CUDA source file
NVCC = nvcc
NVCCFLAGS = -std=c++11 -O3 -arch=sm_70 -diag-suppress=1650
TARGET = cudaBlurImage # TODO: Replace with your target executable name
SOURCE = cudaBlurImage.cu # TODO: Replace with your source file

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run:
	./$(TARGET)

.PHONY: all clean run

