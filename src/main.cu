#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // For getopt()

#include "kmeans.cuh"

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

    // Call the main k-means logic
    kmeans(num_cluster, dims, inputfilename, max_num_iter, threshold, output_centroids_flag, seed);

    return 0;
}