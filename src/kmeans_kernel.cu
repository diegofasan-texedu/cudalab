#include "kmeans_kernel.cuh"
#include <float.h> // For DBL_MAX

__global__ void assign_clusters_kernel(const double* points,
                                       const double* centroids,
                                       int* cluster_assignments,
                                       int num_points,
                                       int num_clusters,
                                       int dims) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < num_points) {
        double min_dist = DBL_MAX;
        int best_cluster = -1;

        // Find the closest centroid for the current point
        for (int cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
            double current_dist = 0.0;
            // Calculate squared Euclidean distance
            for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
                double diff = points[point_idx * dims + dim_idx] - centroids[cluster_idx * dims + dim_idx];
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

__global__ void sum_points_for_clusters_kernel(const double* points,
                                               const int* cluster_assignments,
                                               double* partial_centroid_sums,
                                               int* partial_cluster_counts,
                                               int num_points,
                                               int num_clusters,
                                               int dims) {
    // Using shared memory to store per-block sums to avoid global atomics.
    // This assumes `num_clusters * dims` is small enough for shared memory.
    // For larger `k` or `dims`, this would need a different strategy.
    extern __shared__ double s_sums[];
    int* s_counts = (int*)&s_sums[num_clusters * dims];

    // Each thread initializes a portion of the shared memory.
    for (int i = threadIdx.x; i < num_clusters * dims; i += blockDim.x) {
        s_sums[i] = 0.0;
    }
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        s_counts[i] = 0;
    }
    __syncthreads();

    // Each thread processes a subset of points and accumulates in shared memory.
    int point_idx_start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int point_idx = point_idx_start; point_idx < num_points; point_idx += stride) {
        int assigned_cluster = cluster_assignments[point_idx];
        if (assigned_cluster != -1) {
            atomicAdd(&s_counts[assigned_cluster], 1);
            for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
                atomicAdd(&s_sums[assigned_cluster * dims + dim_idx], points[point_idx * dims + dim_idx]);
            }
        }
    }
    __syncthreads();

    // One thread per cluster dimension writes the partial result to global memory.
    for (int i = threadIdx.x; i < num_clusters * dims; i += blockDim.x) {
        partial_centroid_sums[blockIdx.x * num_clusters * dims + i] = s_sums[i];
    }
    // One thread per cluster writes the partial count to global memory.
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        partial_cluster_counts[blockIdx.x * num_clusters + i] = s_counts[i];
    }
}

__global__ void reduce_partial_sums_kernel(const double* partial_centroid_sums,
                                           const int* partial_cluster_counts,
                                           double* final_centroid_sums,
                                           int* final_cluster_counts,
                                           int num_clusters, int dims, int grid_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_clusters * dims) {
        double total_sum = 0;
        for (int block_id = 0; block_id < grid_size; ++block_id) {
            total_sum += partial_centroid_sums[block_id * num_clusters * dims + i];
        }
        final_centroid_sums[i] = total_sum;
    }

    if (i < num_clusters) {
        int total_count = 0;
        for (int block_id = 0; block_id < grid_size; ++block_id) {
            total_count += partial_cluster_counts[block_id * num_clusters + i];
        }
        final_cluster_counts[i] = total_count;
    }
}

__global__ void average_clusters_kernel(double* centroids,
                                        const double* centroid_sums,
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