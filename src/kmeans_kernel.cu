#include "kmeans_kernel.cuh"
#include <cuda_runtime.h>
#include <cfloat> // For DBL_MAX

__global__ void assign_clusters_kernel(const double* points, const double* centroids, int* cluster_assignments, const int num_points, const int num_clusters, const int dims) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < num_points) {
        double min_dist_sq = DBL_MAX;
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

/**
 * @brief CUDA kernel to calculate new centroids. Each block is responsible for one cluster.
 * It iterates through all points, and if a point belongs to its cluster, it contributes
 * to a sum stored in shared memory. A final reduction in shared memory produces the
 * new centroid, which is then written to global memory.
 */
__global__ void update_centroids_kernel(const double* points, const int* cluster_assignments, double* new_centroids, const int num_points, const int num_clusters, const int dims) {
    // Each block processes one cluster.
    int cluster_idx = blockIdx.x;

    // Dynamically allocate shared memory for the reduction.
    // We need space for each thread in the block to store a partial sum (dims doubles) and a partial count (1 int).
    extern __shared__ char s_data[];
    double* s_sums = (double*)s_data; // Shape: [blockDim.x][dims]
    int* s_counts = (int*)&s_sums[blockDim.x * dims]; // Shape: [blockDim.x]

    // --- 1. Per-Thread Accumulation ---
    // Each thread calculates its own partial sum and count.
    double my_sum[32] = {0.0}; // Max dims supported = 32. Use VLA if supported or malloc for more.
    int my_count = 0;

    // Each thread in the block iterates over a stride of points.
    for (int point_idx = threadIdx.x; point_idx < num_points; point_idx += blockDim.x) {
        if (cluster_assignments[point_idx] == cluster_idx) {
            my_count++;
            for (int d = 0; d < dims; ++d) {
                my_sum[d] += points[point_idx * dims + d];
            }
        }
    }

    // --- 2. Store Partial Results in Shared Memory ---
    // Each thread writes its partial sum and count to its designated slot in shared memory.
    for (int d = 0; d < dims; ++d) {
        s_sums[threadIdx.x * dims + d] = my_sum[d];
    }
    s_counts[threadIdx.x] = my_count;
    __syncthreads();

    // --- 3. Parallel Reduction in Shared Memory ---
    // Iteratively combine results. At the end, thread 0 will hold the total sum.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_counts[threadIdx.x] += s_counts[threadIdx.x + s];
            for (int d = 0; d < dims; d++) {
                s_sums[threadIdx.x * dims + d] += s_sums[(threadIdx.x + s) * dims + d];
            }
        }
        __syncthreads();
    }

    // --- 4. Write Final Result ---
    // Thread 0, which now holds the total sum for the block, calculates the new centroid.
    if (threadIdx.x == 0 && s_counts[0] > 0) {
        double inv_count = 1.0 / s_counts[0];
        for (int d = 0; d < dims; ++d) {
            new_centroids[cluster_idx * dims + d] = s_sums[d] * inv_count;
        }
    }
    // If count is 0, the centroid remains unchanged from the previous iteration.
}
