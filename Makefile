# Makefile for CUDA K-Means

# --- Configuration ---
NVCC = nvcc
EXEC = bin/kmeans

# All source files are listed here.
# Add any new .cu files to this list.
SRCS = src/main.cu \
       src/argparse.cu \
       src/io.cu \
       src/dataset.cu \
       src/kmeans.cu \
       src/sequential_kmeans.cu \
       src/cuda_kmeans.cu \
       src/thrust_kmeans.cu

# Compiler flags
NVCCFLAGS = -std=c++17 -O3
INC_DIR = -Isrc

# --- Rules ---
.PHONY: all clean

all: $(EXEC)

$(EXEC): $(SRCS)
	@mkdir -p $(dir $(EXEC))
	@echo "Compiling all sources..."
	$(NVCC) $(NVCCFLAGS) $(INC_DIR) $(SRCS) -o $(EXEC)

clean:
	@echo "Cleaning up..."
	rm -f $(EXEC)
