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
    int* s_counts = (int*)(&s_sums[num_clusters * dims]);

    // --- Stage 1: Each thread accumulates sums into private memory (registers) ---
    // This requires enough registers to hold sums for all clusters.
    // This is only feasible for a small number of clusters.
    double p_sums[128]; // Max K=128, D=1 for this to work. A more robust solution would use local memory.
    int p_counts[128];  // Assuming max K=128

    for (int i = 0; i < num_clusters; ++i) {
        for (int d = 0; d < dims; ++d) p_sums[i * dims + d] = 0.0;
        p_counts[i] = 0;
    }

    // Each thread processes a subset of points using a grid-stride loop.
    int point_idx_start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int point_idx = point_idx_start; point_idx < num_points; point_idx += stride) {
        int assigned_cluster = cluster_assignments[point_idx];
        if (assigned_cluster != -1) {
            p_counts[assigned_cluster]++;
            for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
                p_sums[assigned_cluster * dims + dim_idx] += points[point_idx * dims + dim_idx];
            }
        }
    }

    // --- Stage 2: Fully parallel reduction of private sums into shared memory (NO ATOMICS) ---
    // Each thread is responsible for reducing one element of the final sum arrays.
    // This loop iterates over all elements that this thread is responsible for.
    for (int i = threadIdx.x; i < num_clusters * dims; i += blockDim.x) {
        // Step 1: Each thread writes its private contribution for this element to shared memory.
        // We use shared memory as temporary storage for the reduction.
        s_sums[i] = p_sums[i];
    }
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        s_counts[i] = p_counts[i];
    }
    __syncthreads();

    // Step 2: Perform a tree-based reduction in shared memory.
    // At each step, half the threads add a value from the other half.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            // Reduce the sums array
            for (int i = threadIdx.x; i < num_clusters * dims; i += s) s_sums[i] += s_sums[i + s];
            // Reduce the counts array
            for (int i = threadIdx.x; i < num_clusters; i += s) s_counts[i] += s_counts[i + s];
        }
        __syncthreads();
    }

    // Step 3: After reduction, thread 0 holds the final sum for each element.
    // It writes the block's total partial sum to global memory.
    if (threadIdx.x == 0) {
        for (int i = 0; i < num_clusters * dims; ++i) {
            partial_centroid_sums[blockIdx.x * num_clusters * dims + i] = s_sums[i];
        }
        for (int i = 0; i < num_clusters; ++i) {
            partial_cluster_counts[blockIdx.x * num_clusters + i] = s_counts[i];
        }
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

__global__ void check_convergence_kernel(const double* old_centroids,
                                         const double* new_centroids,
                                         int* converged_flag,
                                         int num_clusters,
                                         int dims,
                                         double threshold_sq) {
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cluster_idx < num_clusters) {
        double dist_sq = 0.0;
        // Calculate the squared Euclidean distance between the old and new centroid
        for (int d = 0; d < dims; ++d) {
            double diff = old_centroids[cluster_idx * dims + d] - new_centroids[cluster_idx * dims + d];
            dist_sq += diff * diff;
        }

        // If any centroid moved more than the threshold, set the flag to 0 (not converged)
        if (dist_sq > threshold_sq) {
            // Use an atomic operation to ensure only one thread writes at a time, preventing race conditions.
            atomicMin(converged_flag, 0);
        }
    }
}