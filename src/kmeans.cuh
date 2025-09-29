#ifndef KMEANS_CUH
#define KMEANS_CUH

#include "dataset.cuh"

/**
 * @brief Executes the main k-means algorithm logic.
 * @param num_cluster Number of clusters.
 * @param data The dataset containing points and dimensions.
 * @param max_num_iter Maximum number of iterations for the algorithm.
 * @param threshold Convergence threshold.
 * @param output_centroids_flag Flag to indicate if final centroids should be output.
 * @param seed Seed for random number generation.
 * @param verbose Flag to enable verbose output.
 */
void kmeans(int num_cluster, DataSet& data, int max_num_iter, double threshold, bool output_centroids_flag, int seed, bool verbose);

#endif // KMEANS_CUH