#ifndef KMEANS_CUH
#define KMEANS_CUH

/**
 * @brief Executes the main k-means algorithm logic.
 * @param num_cluster Number of clusters.
 * @param dims Number of dimensions for each data point.
 * @param inputfilename Path to the input data file.
 * @param max_num_iter Maximum number of iterations for the algorithm.
 * @param threshold Convergence threshold.
 * @param output_centroids_flag Flag to indicate if final centroids should be output.
 * @param seed Seed for random number generation.
 */
void kmeans(int num_cluster, int dims, const char* inputfilename, int max_num_iter, double threshold, bool output_centroids_flag, int seed);

#endif // KMEANS_CUH