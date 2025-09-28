#include "kmeans.cuh"

#include <iostream>
#include <stdio.h>

/**
 * A CUDA kernel that is executed on the GPU.
 * The __global__ specifier indicates that this function runs on the device
 * and can be called from host code.
 */
__global__ void hello_from_gpu() {
    // printf is executed on the GPU and prints to the console.
    printf("Hello World from the GPU!\n");
}

void kmeans(int num_cluster, int dims, const char* inputfilename, int max_num_iter, double threshold, bool output_centroids_flag, int seed) {
    // This part of the code runs on the CPU (the host).
    std::cout << "Hello World from the CPU!" << std::endl;

    // Launch the kernel on the GPU.
    // 10 blocks of threads, with 1 thread in each block.
    hello_from_gpu<<<10, 1>>>();

    // Wait for the GPU to finish all its work before the CPU continues.
    // This is crucial to see the output from the GPU before the program exits.
    cudaDeviceSynchronize();

    // In the future, you can add more logic here, for example:
    // 1. Read data from inputfilename
    // 2. Allocate memory on host and device
    // 3. Implement the k-means iteration loop
    // 4. Clean up resources
}