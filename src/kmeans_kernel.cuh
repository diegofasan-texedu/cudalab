#ifndef KMEANS_KERNEL_CUH
#define KMEANS_KERNEL_CUH

/**
 * @brief CUDA kernel to assign each point to the nearest centroid.
 * Each thread is responsible for one point.
 */
__global__ void assign_clusters_kernel(const double* points, const double* centroids, int* cluster_assignments, const int num_points, const int num_clusters, const int dims);

/**
 * @brief CUDA kernel to sum the coordinates and count points for each cluster.
 * Each thread processes one point and uses atomic operations to update the
 * sums and counts for its assigned cluster.
 */
__global__ void sum_points_for_clusters_kernel(const double* points, const int* cluster_assignments, double* centroid_sums, int* cluster_counts, const int num_points, const int dims);

/**
 * @brief CUDA kernel to calculate the new centroids by averaging the sums.
 * Each thread is responsible for one cluster. It calculates the new centroid
 * position by dividing the sum of coordinates by the number of points.
 */
__global__ void average_clusters_kernel(double* new_centroids, const double* centroid_sums, const int* cluster_counts, const int num_clusters, const int dims);


#endif // KMEANS_KERNEL_CUH