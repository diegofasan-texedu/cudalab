#ifndef KMEANS_KERNEL_CUH
#define KMEANS_KERNEL_CUH

#include <cuda_runtime.h>

/**
 * @brief Assigns each point to the nearest cluster centroid.
 *
 * Each thread processes one point. It calculates the squared Euclidean distance
 * to every centroid and finds the index of the centroid with the minimum distance.
 *
 * @param points Device pointer to the input data points.
 * @param centroids Device pointer to the current cluster centroids.
 * @param cluster_assignments Device pointer to store the assigned cluster index for each point.
 * @param num_points Total number of points.
 * @param num_clusters Total number of clusters.
 * @param dims The dimensionality of each point.
 */
__global__ void assign_clusters_kernel(const float* points, const float* centroids, int* cluster_assignments, int num_points, int num_clusters, int dims);

/**
 * @brief Sums the coordinates of all points belonging to each cluster.
 *
 * This is the first stage of a two-pass reduction. Each thread block calculates
 * its own partial sum and count for each cluster, writing the result to a
 * block-specific location in global memory. This avoids using global atomics.
 *
 * @param points Device pointer to the input data points.
 * @param cluster_assignments Device pointer to the assigned cluster for each point.
 * @param partial_centroid_sums Device pointer to store the partial sum of points for each cluster from each block.
 * @param partial_cluster_counts Device pointer to store the partial count of points in each cluster from each block.
 * @param num_points Total number of points.
 * @param num_clusters Total number of clusters.
 * @param dims The dimensionality of each point.
 */
__global__ void sum_points_for_clusters_kernel(const float* points, const int* cluster_assignments, float* partial_centroid_sums, int* partial_cluster_counts, int num_points, int num_clusters, int dims);

/**
 * @brief Reduces the partial sums from all blocks into a final sum.
 *
 * This is the second stage of the reduction. Each thread is responsible for
 * reducing the partial sums for one dimension of one cluster.
 * 
 * @param grid_size The number of blocks used in the first kernel (sum_points_for_clusters_kernel).
 */
__global__ void reduce_partial_sums_kernel(const float* partial_centroid_sums, const int* partial_cluster_counts, float* final_centroid_sums, int* final_cluster_counts, int num_clusters, int dims, int grid_size);

/**
 * @brief Calculates the new centroids by averaging the sums.
 * 
 * Each thread processes one cluster. It divides the sum of points for that cluster
 * by the number of points in it to get the new centroid. If a cluster is empty,
 * its centroid is not updated.
 */
__global__ void average_clusters_kernel(double* centroids, const double* centroid_sums, const int* cluster_counts, int num_clusters, int dims);

/**
 * @brief Checks if the centroids have moved less than a given threshold.
 *
 * Each thread processes one cluster. It calculates the squared Euclidean distance
 * between the old and new centroid positions. If this distance exceeds the
 * squared threshold for any centroid, a convergence flag is set to 0 (false).
 *
 * @param threshold_sq The squared convergence threshold.
 */
__global__ void check_convergence_kernel(const float* old_centroids, const float* new_centroids, int* converged_flag, int num_clusters, int dims, float threshold_sq);


#endif // KMEANS_KERNEL_CUH