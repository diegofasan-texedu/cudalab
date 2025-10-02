#include "kmeans_kernel.cuh"
#include <float.h> // For FLT_MAX

__global__ void assign_clusters_kernel(const float* points,
                                       const float* centroids,
                                       int* cluster_assignments,
                                       int num_points,
                                       int num_clusters,
                                       int dims) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < num_points) {
        float min_dist = FLT_MAX;
        int best_cluster = -1;

        // Find the closest centroid for the current point
        for (int cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
            float current_dist = 0.0f;
            // Calculate squared Euclidean distance
            for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
                float diff = points[point_idx * dims + dim_idx] - centroids[cluster_idx * dims + dim_idx];
                current_dist += diff * diff;
            }

            if (current_dist < min_dist) {
                min_dist = current_dist;
                best_cluster = cluster_idx;
            }
        }
        cluster_assignments[point_idx] = best_cluster;
    }
}

__global__ void sum_points_for_clusters_kernel(const float* points,
                                               const int* cluster_assignments,
                                               float* partial_centroid_sums,
                                               int* partial_cluster_counts,
                                               int num_points,
                                               int num_clusters,
                                               int dims) {
    // Using shared memory to store per-block sums to avoid global atomics.
    // This assumes `num_clusters * dims` is small enough for shared memory.
    // For larger `k` or `dims`, this would need a different strategy.
    extern __shared__ float s_sums[];
    int* s_counts = (int*)(&s_sums[num_clusters * dims]);
    
    // Initialize shared memory
    for (int i = threadIdx.x; i < num_clusters * dims; i += blockDim.x) {
        s_sums[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        s_counts[i] = 0;
    }
    __syncthreads();
    
    // Each thread is responsible for one point.
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < num_points) {
        int assigned_cluster = cluster_assignments[point_idx];
        if (assigned_cluster != -1) {
            // Atomically update the counts and sums in shared memory for this block
            atomicAdd(&s_counts[assigned_cluster], 1);
            for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
                atomicAdd(&s_sums[assigned_cluster * dims + dim_idx], points[point_idx * dims + dim_idx]);
            }
        }
    }
    __syncthreads();

    // Each thread writes a portion of the shared memory results to global memory.
    // This is a parallel write, not just thread 0.
    for (int i = threadIdx.x; i < num_clusters * dims; i += blockDim.x) {
        partial_centroid_sums[blockIdx.x * num_clusters * dims + i] = s_sums[i];
    }
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        partial_cluster_counts[blockIdx.x * num_clusters + i] = s_counts[i];
    }
}

__global__ void reduce_partial_sums_kernel(const float* partial_centroid_sums,
                                           const int* partial_cluster_counts,
                                           float* final_centroid_sums,
                                           int* final_cluster_counts,
                                           int num_clusters, int dims, int grid_size) {
    // Each thread is responsible for one element of the final sums/counts array.
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // This thread handles one dimension of one cluster's sum
    if (global_idx < num_clusters * dims) {
        float total_sum = 0.0f;
        for (int block_id = 0; block_id < grid_size; ++block_id) {
            total_sum += partial_centroid_sums[block_id * num_clusters * dims + global_idx];
        }
        final_centroid_sums[global_idx] = total_sum;
    }

    // This thread handles one cluster's count.
    // We use a separate index `cluster_idx` to avoid mixing with the sums logic.
    int cluster_idx = global_idx;
    if (cluster_idx < num_clusters) {
        int total_count = 0;
        for (int block_id = 0; block_id < grid_size; ++block_id) {
            total_count += partial_cluster_counts[block_id * num_clusters + cluster_idx];
        }
        final_cluster_counts[cluster_idx] = total_count;
    }
}

__global__ void average_clusters_kernel(float* centroids,
                                        const float* centroid_sums,
                                        const int* cluster_counts,
                                        int num_clusters,
                                        int dims) {
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cluster_idx < num_clusters) {
        int count = cluster_counts[cluster_idx];

        // Avoid division by zero for empty clusters
        if (count > 0) {
            for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
                // Calculate the new centroid by averaging
                centroids[cluster_idx * dims + dim_idx] = centroid_sums[cluster_idx * dims + dim_idx] / count;
            }
        }
    }
}

__global__ void check_convergence_kernel(const float* old_centroids,
                                         const float* new_centroids,
                                         int* converged_flag,
                                         int num_clusters,
                                         int dims,
                                         float threshold_sq) {
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cluster_idx < num_clusters) {
        float dist_sq = 0.0f;
        // Calculate the squared Euclidean distance between the old and new centroid
        for (int d = 0; d < dims; ++d) {
            float diff = old_centroids[cluster_idx * dims + d] - new_centroids[cluster_idx * dims + d];
            dist_sq += diff * diff;
        }

        // If any centroid moved more than the threshold, set the flag to 0 (not converged)
        if (dist_sq > threshold_sq) {
            // Use an atomic operation to ensure only one thread writes at a time, preventing race conditions.
            atomicMin(converged_flag, 0);
        }
    }
}