# Makefile for CUDA K-Means

# Compiler for CUDA
NVCC = nvcc

# Source files
SRCS = $(wildcard src/*.cu)

# Include directory
INC = ./src/

# Compiler flags
# -Xcompiler is used to pass flags like -Wall and -Werror to the host compiler (g++)
NVCCFLAGS = -std=c++17 -Xcompiler -Wall -Xcompiler -Werror -O3

# Target executable name
EXEC = bin/kmeans

.PHONY: all compile clean

all: clean compile

compile:
	mkdir -p $(dir $(EXEC))
	$(NVCC) $(SRCS) $(NVCCFLAGS) -I$(INC) -o $(EXEC)

clean:
	rm -f $(EXEC)
