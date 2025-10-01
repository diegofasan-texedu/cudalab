#ifndef CUDA_KMEANS_CUH
#define CUDA_KMEANS_CUH

#include "dataset.cuh"

void cuda_kmeans(int num_cluster, DataSet& data, int max_num_iter, double threshold, bool output_centroids_flag, int seed, bool verbose);

#endif // CUDA_KMEANS_CUH