# TODO: Replace SOURCE with your actual CUDA source file
NVCC = nvcc
NVCCFLAGS = -std=c++11 -O3
TARGET = colorToGrayscaleConversion # TODO: Replace with your target executable name
SOURCE = colorToGrayscaleConversion.cu # TODO: Replace with your source file

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run:
	./$(TARGET)

.PHONY: all clean run
