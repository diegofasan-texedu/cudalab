#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> // For getopt()


/**
 * A CUDA kernel that is executed on the GPU.
 * The __global__ specifier indicates that this function runs on the device
 * and can be called from host code.
 */
__global__ void hello_from_gpu() {
    // printf is executed on the GPU and prints to the console.
    printf("Hello World from the GPU!\n");
}

int main(int argc, char* argv[]) {
    // Default values for arguments
    int num_cluster = 0;
    int dims = 0;
    const char* inputfilename = nullptr;
    int max_num_iter = 100;
    double threshold = 0.001;
    bool output_centroids_flag = false;
    int seed = 0;

    // Argument parsing with getopt()
    int opt;
    while ((opt = getopt(argc, argv, "k:d:i:m:t:cs:")) != -1) {
        switch (opt) {
            case 'k':
                num_cluster = atoi(optarg);
                break;
            case 'd':
                dims = atoi(optarg);
                break;
            case 'i':
                inputfilename = optarg;
                break;
            case 'm':
                max_num_iter = atoi(optarg);
                break;
            case 't':
                threshold = atof(optarg);
                break;
            case 'c':
                output_centroids_flag = true;
                break;
            case 's':
                seed = atoi(optarg);
                break;
            default: /* '?' */
                fprintf(stderr, "Usage: %s -k num_cluster -d dims -i inputfile ...\n", argv[0]);
                return 1;
        }
    }

    // Print parsed arguments to verify
    std::cout << "--- Parsed Arguments ---" << std::endl;
    std::cout << "Num Clusters (-k): " << num_cluster << std::endl;
    std::cout << "Dimensions (-d): " << dims << std::endl;
    std::cout << "Input Filename (-i): " << (inputfilename ? inputfilename : "Not specified") << std::endl;
    std::cout << "Max Iterations (-m): " << max_num_iter << std::endl;
    std::cout << "Threshold (-t): " << threshold << std::endl;
    std::cout << "Output Centroids Flag (-c): " << (output_centroids_flag ? "true" : "false") << std::endl;
    std::cout << "Seed (-s): " << seed << std::endl;
    std::cout << "------------------------" << std::endl;


    // This part of the code runs on the CPU (the host).
    std::cout << "Hello World from the CPU!" << std::endl;

    // Launch the kernel on the GPU.
    // 10 blocks of threads, with 1 thread in each block.
    hello_from_gpu<<<10, 1>>>();

    // Wait for the GPU to finish all its work before the CPU continues.
    // This is crucial to see the output from the GPU before the program exits.
    cudaDeviceSynchronize();

    return 0;
}