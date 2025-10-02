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
__global__ void assign_clusters_kernel(const double* points, const double* centroids, int* cluster_assignments, int num_points, int num_clusters, int dims);

/**
 * @brief Sums the coordinates of all points belonging to each cluster.
 *
 * Each thread processes one point. It reads the point's assigned cluster and
 * atomically adds its coordinates to the corresponding sum in `centroid_sums`.
 * It also atomically increments the count for that cluster.
 *
 * @param points Device pointer to the input data points.
 * @param cluster_assignments Device pointer to the assigned cluster for each point.
 * @param centroid_sums Device pointer to store the sum of points for each cluster.
 * @param cluster_counts Device pointer to store the number of points in each cluster.
 * @param num_points Total number of points.
 * @param dims The dimensionality of each point.
 */
__global__ void sum_points_for_clusters_kernel(const double* points, const int* cluster_assignments, double* centroid_sums, int* cluster_counts, int num_points, int dims);

/**
 * @brief Calculates the new centroids by averaging the sums.
 *
 * Each thread processes one cluster. It divides the sum of points for that cluster
 * by the number of points in it to get the new centroid. If a cluster is empty,
 * its centroid is not updated.
 */
__global__ void average_clusters_kernel(double* centroids, const double* centroid_sums, const int* cluster_counts, int num_clusters, int dims);

#endif // KMEANS_KERNEL_CUH