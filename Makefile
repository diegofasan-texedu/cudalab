# Makefile for CUDA K-Means (Simplified)

# --- Configuration ---
NVCC = nvcc
EXEC = bin/kmeans

# Source files
# nvcc will use the host compiler (like g++) for .cpp files automatically.
SRCS = src/*.cu

# Compiler flags
NVCCFLAGS = -std=c++17 -O3 -Isrc -arch=sm_60

# --- Rules ---
.PHONY: all clean

all: $(EXEC)

$(EXEC): $(SRCS)
	@mkdir -p $(dir $(EXEC))
	@echo "Compiling and linking all sources..."
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(EXEC)

clean:
	@echo "Cleaning up..."
	rm -f $(EXEC)
