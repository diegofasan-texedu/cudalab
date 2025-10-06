# Makefile for CUDA K-Means (Simplified)

# --- Configuration ---
NVCC = nvcc
EXEC_DOUBLE = bin/kmeans
EXEC_FLOAT = bin/kmeans_float

# Source files
SRCS_DOUBLE = src/*.cu
SRCS_FLOAT = src_float/*.cu

# Compiler flags
NVCCFLAGS = -std=c++17 -O3 -Isrc -arch=sm_60 --extended-lambda
NVCCFLAGS_FLOAT = -std=c++17 -O3 -Isrc_float -arch=sm_60 --extended-lambda

# --- Rules ---
.PHONY: all clean

all: double float

double: $(EXEC_DOUBLE)

float: $(EXEC_FLOAT)

$(EXEC_DOUBLE): $(SRCS_DOUBLE)
	@mkdir -p $(dir $(EXEC_DOUBLE))
	@echo "Compiling and linking all sources for DOUBLE..."
	$(NVCC) $(NVCCFLAGS) $(SRCS_DOUBLE) -o $(EXEC_DOUBLE)

$(EXEC_FLOAT): $(SRCS_FLOAT)
	@mkdir -p $(dir $(EXEC_FLOAT))
	@echo "Compiling and linking all sources for FLOAT..."
	$(NVCC) $(NVCCFLAGS_FLOAT) $(SRCS_FLOAT) -o $(EXEC_FLOAT)

clean:
	@echo "Cleaning up..."
	rm -f $(EXEC_DOUBLE) $(EXEC_FLOAT)
