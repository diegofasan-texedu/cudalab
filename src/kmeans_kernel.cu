#include "kmeans_kernel.cuh"
#include <cuda_runtime.h>
#include <limits>

__global__ void assign_clusters_kernel(const double* points, const double* centroids, int* cluster_assignments, const int num_points, const int num_clusters, const int dims) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < num_points) {
        double min_dist_sq = std::numeric_limits<double>::max();
        int best_cluster = -1;

        for (int c = 0; c < num_clusters; ++c) {
            double current_dist_sq = 0.0;
            for (int d = 0; d < dims; ++d) {
                double diff = points[point_idx * dims + d] - centroids[c * dims + d];
                current_dist_sq += diff * diff;
            }

            if (current_dist_sq < min_dist_sq) {
                min_dist_sq = current_dist_sq;
                best_cluster = c;
            }
        }
        cluster_assignments[point_idx] = best_cluster;
    }
}
    }
}

/**
 * @brief CUDA kernel to calculate new centroids. Each block is responsible for one cluster.
 * It iterates through all points, and if a point belongs to its cluster, it contributes
 * to a sum stored in shared memory. A final reduction in shared memory produces the
 * new centroid, which is then written to global memory.
 */
__global__ void update_centroids_kernel(const double* points, const int* cluster_assignments, double* new_centroids, const int num_points, const int num_clusters, const int dims) {
    // Each block processes one cluster.
    int cluster_idx = blockIdx.x;

    // Dynamically allocate shared memory. One double for each dimension plus one int for the count.
    extern __shared__ char s_data[];
    double* s_centroid_sum = (double*)s_data;
    int* s_point_count = (int*)&s_centroid_sum[dims];

    // Initialize shared memory for this block's cluster sum and count.
    if (threadIdx.x < dims) {
        s_centroid_sum[threadIdx.x] = 0.0;
    }
    if (threadIdx.x == 0) {
        *s_point_count = 0;
    }
    __syncthreads();

    // Each thread in the block iterates over a stride of points.
    for (int point_idx = threadIdx.x; point_idx < num_points; point_idx += blockDim.x) {
        if (cluster_assignments[point_idx] == cluster_idx) {
            // This point belongs to our cluster. Add its coordinates to the shared sum.
            // Use atomics on SHARED memory, which is much faster than global.
            atomicAdd(s_point_count, 1);
            for (int d = 0; d < dims; ++d) {
                atomicAdd(&s_centroid_sum[d], points[point_idx * dims + d]);
            }
        }
    }
    __syncthreads();

    // Thread 0 of each block calculates the new centroid and writes it to global memory.
    if (threadIdx.x == 0 && *s_point_count > 0) {
        for (int d = 0; d < dims; ++d) {
            new_centroids[cluster_idx * dims + d] = s_centroid_sum[d] / *s_point_count;
        }
    }
    // If count is 0, the centroid remains unchanged from the previous iteration.
}


