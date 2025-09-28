# Makefile for CUDA Hello World

# Compiler and flags
NVCC = nvcc
# NVCCFLAGS = -std=c++17 -O3

# To get the best performance, you can specify the architecture of your GPU.
# Find your architecture's "Compute Capability" number (e.g., 7.5 for Turing, 8.6 for Ampere).
# You can find this by running `nvidia-smi`.
# Example for an Ampere GPU (RTX 30-series):
# NVCCFLAGS += -gencode arch=compute_86,code=sm_86

# Target executable name
TARGET = outputs/exec

# Phony targets (targets that are not files)
.PHONY: all run clean

# Default target: build the executable
all: $(TARGET)

$(TARGET): src/main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(TARGET)
	mkdir outputs
	./$(TARGET)

clean:
	rm -f $(TARGET)

