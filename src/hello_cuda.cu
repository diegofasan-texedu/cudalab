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

int main() {
    // This part of the code runs on the CPU (the host).
    std::cout << "Hello World from the CPU!" << std::endl;

    // Launch the kernel on the GPU.
    // The <<<1, 1>>> syntax specifies the execution configuration:
    // 1 block of threads, with 1 thread in that block.
    hello_from_gpu<<<1, 1>>>();

    // Wait for the GPU to finish all its work before the CPU continues.
    // This is crucial to see the output from the GPU before the program exits.
    cudaDeviceSynchronize();

    return 0;
}