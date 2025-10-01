#ifndef SEQUENTIAL_KMEANS_CUH
#define SEQUENTIAL_KMEANS_CUH

#include "dataset.cuh"

void sequential_kmeans(int num_cluster, DataSet& data, int max_num_iter, double threshold, bool output_centroids_flag, int seed, bool verbose);

#endif // SEQUENTIAL_KMEANS_CUH