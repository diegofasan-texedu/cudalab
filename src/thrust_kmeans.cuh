#ifndef THRUST_KMEANS_CUH
#define THRUST_KMEANS_CUH

#include "dataset.cuh"

void thrust_kmeans(int num_cluster, KmeansData& data, int max_num_iter, double threshold, bool output_centroids_flag, int seed, bool verbose);

#endif // THRUST_KMEANS_CUH