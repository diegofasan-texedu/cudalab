#include "cuda_kmeans.cuh"
#include <iostream>

void cuda_kmeans(int num_cluster, KmeansData& data, int max_num_iter, double threshold, bool output_centroids_flag, int seed, bool verbose) {
    if (verbose) {
        std::cout << "Executing CUDA K-Means..." << std::endl;
    }
    
    // start cuda method
    

}