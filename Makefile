# Makefile for CUDA K-Means

# Compiler for CUDA
NVCC = nvcc

# Source files
SRCS = $(wildcard src/*.cu)

# Include directory
INC = ./src/

# Compiler flags
NVCCFLAGS = -std=c++17 -Wall -O3

# Target executable name
EXEC = bin/kmeans

.PHONY: all compile clean

all: clean compile

compile:
	mkdir -p $(dir $(EXEC))
	$(NVCC) $(SRCS) $(NVCCFLAGS) -I$(INC) -o $(EXEC)

clean:
	rm -f $(EXEC)
