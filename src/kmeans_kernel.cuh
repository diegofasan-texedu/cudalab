#ifndef KMEANS_KERNEL_CUH
#define KMEANS_KERNEL_CUH

/**
 * @brief CUDA kernel to assign each point to the nearest centroid.
 * Each thread is responsible for one point.
 */
__global__ void assign_clusters_kernel(const double* points, const double* centroids, int* cluster_assignments, const int num_points, const int num_clusters, const int dims);

/**
 * @brief CUDA kernel to calculate new centroids using shared memory reduction.
 * Each block is responsible for one cluster. It calculates the new centroid
 * position by summing points in shared memory and then averaging.
 */
__global__ void update_centroids_kernel(const double* points, const int* cluster_assignments, double* new_centroids, const int num_points, const int num_clusters, const int dims);


#endif // KMEANS_KERNEL_CUH