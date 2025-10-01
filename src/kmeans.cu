#include "kmeans.cuh"

// Include headers for the different implementations
#include "sequential_kmeans.cuh"
#include "cuda_kmeans.cuh"
#include "thrust_kmeans.cuh"

#include <iostream>
#include <stdio.h>

void kmeans(int num_cluster, KmeansData& data, int max_num_iter, double threshold, bool output_centroids_flag, int seed, bool verbose, ExecutionMethod method) {
    // Use a switch to dispatch to the correct k-means implementation
    // based on the selected method.
    switch (method) {
        case SEQ:
            sequential_kmeans(num_cluster, data, max_num_iter, threshold, output_centroids_flag, seed, verbose);
            break;
        case CUDA:
            cuda_kmeans(num_cluster, data, max_num_iter, threshold, output_centroids_flag, seed, verbose);
            break;
        case THRUST:
            thrust_kmeans(num_cluster, data, max_num_iter, threshold, output_centroids_flag, seed, verbose);
            break;
        case UNSPECIFIED:
            // This case should ideally not be reached due to argument parsing validation.
            fprintf(stderr, "Error: Execution method is unspecified.\n");
            break;
    }
}