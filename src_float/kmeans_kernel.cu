#include "kmeans_kernel.cuh"
#include <float.h> // For FLT_MAX

__global__ void assign_clusters_kernel(const float* points, const float* centroids, int* cluster_assignments, int num_points, int num_clusters, int dims) {
    // Get the unique index for this thread, which corresponds to a data point.
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: ensure the thread is not accessing beyond the number of points.
    if (point_idx >= num_points) {
        return;
    }

    float min_dist_sq = FLT_MAX;
    int best_cluster = 0; // Default to the first cluster

    // Find the closest centroid for the current point.
    for (int cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
        float current_dist_sq = 0.0f;
        
        // Calculate squared Euclidean distance to save on sqrt() operations.
        for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
            float diff = points[point_idx * dims + dim_idx] - centroids[cluster_idx * dims + dim_idx];
            current_dist_sq += diff * diff;
        }
        if (current_dist_sq < min_dist_sq) {
            min_dist_sq = current_dist_sq;
            best_cluster = cluster_idx;
        }
    }
    // Assign the point to the closest cluster.
    cluster_assignments[point_idx] = best_cluster;
}

__global__ void assign_clusters_smem_kernel(const float* points, const float* centroids, int* cluster_assignments, int num_points, int num_clusters, int dims) {
    // Dynamically allocate shared memory for centroids. The size is passed during kernel launch.
    extern __shared__ float smem_centroids[];

    // Cooperatively load all centroids from global memory into shared memory.
    // Each thread loads one or more values.
    int num_centroid_values = num_clusters * dims;
    for (int i = threadIdx.x; i < num_centroid_values; i += blockDim.x) {
        smem_centroids[i] = centroids[i];
    }
    
    // Synchronize all threads in the block to ensure centroids are fully loaded
    // before any thread starts using them.
    __syncthreads();

    // Get the unique index for this thread, which corresponds to a data point.
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (point_idx >= num_points) {
        return;
    }

    float min_dist_sq = FLT_MAX;
    int best_cluster = 0;

    // Find the closest centroid for the current point.
    // This loop now reads from the much faster shared memory.
    for (int cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
        float current_dist_sq = 0.0f;
        
        // Calculate squared Euclidean distance.
        for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
            float diff = points[point_idx * dims + dim_idx] - smem_centroids[cluster_idx * dims + dim_idx];
            current_dist_sq += diff * diff;
        }
        if (current_dist_sq < min_dist_sq) {
            min_dist_sq = current_dist_sq;
            best_cluster = cluster_idx;
        }
    }
    cluster_assignments[point_idx] = best_cluster;
}

__global__ void update_centroids_sum_kernel(const float* d_points, const int* d_cluster_assignments, float* d_new_centroids_sum, int* d_cluster_counts, int num_points, int dims) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx >= num_points) {
        return;
    }

    // Get the cluster this point is assigned to.
    int assigned_cluster = d_cluster_assignments[point_idx];

    // Atomically increment the count for this cluster.
    atomicAdd(&d_cluster_counts[assigned_cluster], 1);

    // Atomically add this point's coordinates to the sum for its cluster.
    for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
        float point_coord = d_points[point_idx * dims + dim_idx];
        atomicAdd(&d_new_centroids_sum[assigned_cluster * dims + dim_idx], point_coord);
    }
}

__global__ void update_centroids_sum_smem_kernel(const float* d_points, const int* d_cluster_assignments, float* d_new_centroids_sum, int* d_cluster_counts, int num_points, int num_clusters, int dims) {
    // Dynamically allocate shared memory for partial sums and counts.
    // The layout is: [sums for all clusters/dims][counts for all clusters]
    extern __shared__ char smem_buf[];
    float* smem_sums = (float*)smem_buf;
    int* smem_counts = (int*)(smem_buf + num_clusters * dims * sizeof(float));

    // Initialize shared memory in parallel.
    int smem_sum_size = num_clusters * dims;
    for (int i = threadIdx.x; i < smem_sum_size; i += blockDim.x) {
        smem_sums[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        smem_counts[i] = 0;
    }
    __syncthreads();

    // Each thread processes a subset of points using a grid-stride loop.
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int p = point_idx; p < num_points; p += stride) {
        // Get the cluster this point is assigned to.
        int assigned_cluster = d_cluster_assignments[p];

        // Atomically update the partial count and sums in SHARED memory.
        // Atomics on shared memory are much faster than on global memory.
        atomicAdd(&smem_counts[assigned_cluster], 1);
        for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
            float point_coord = d_points[p * dims + dim_idx];
            atomicAdd(&smem_sums[assigned_cluster * dims + dim_idx], point_coord);
        }
    }

    // Synchronize to ensure all threads in the block have finished their partial sums.
    __syncthreads();

    // Now, have the block cooperatively write its final partial results to global memory.
    // This reduces global atomic operations from one-per-point to one-per-cluster-per-block.
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        // Only perform the global atomic if there's something to add.
        if (smem_counts[i] > 0) {
            atomicAdd(&d_cluster_counts[i], smem_counts[i]);
            for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
                int index = i * dims + dim_idx;
                if (smem_sums[index] != 0.0f) {
                    atomicAdd(&d_new_centroids_sum[index], smem_sums[index]);
                }
            }
        }
    }
}

__global__ void calculate_and_reset_smem_kernel(float* d_centroids, float* d_new_centroids_sum, int* d_cluster_counts, int num_clusters, int dims) {
    // One block per cluster.
    int cluster_idx = blockIdx.x;
    if (cluster_idx >= num_clusters) {
        return;
    }

    // Use shared memory to store the count for the current cluster.
    __shared__ int s_count;

    // The first thread in the block loads the count from global memory.
    if (threadIdx.x == 0) {
        s_count = d_cluster_counts[cluster_idx];
    }
    // Synchronize to ensure s_count is visible to all threads in the block.
    __syncthreads();

    // Only proceed if the cluster is not empty.
    if (s_count > 0) {
        // Parallelize the dimension loop across threads in the block.
        for (int dim_idx = threadIdx.x; dim_idx < dims; dim_idx += blockDim.x) {
            int index = cluster_idx * dims + dim_idx;
            d_centroids[index] = d_new_centroids_sum[index] / s_count;
        }
    }

    // Reset buffers for the next iteration in parallel.
    for (int i = threadIdx.x; i < dims; i += blockDim.x) {
        d_new_centroids_sum[cluster_idx * dims + i] = 0.0f;
    }
    if (threadIdx.x == 0) {
        d_cluster_counts[cluster_idx] = 0;
    }
}

__global__ void calculate_and_reset_kernel(float* d_centroids, float* d_new_centroids_sum, int* d_cluster_counts, int num_clusters, int dims) {
    // Use a grid-stride loop to have each thread handle multiple dimensions.
    // This allows for a more flexible and efficient grid/block configuration.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // This loop processes all centroid dimensions.
    for (int index = i; index < num_clusters * dims; index += stride) {
        int cluster_idx = index / dims;
        int count = d_cluster_counts[cluster_idx];

        // Calculate the new centroid dimension if the cluster is not empty.
        if (count > 0) {
            d_centroids[index] = d_new_centroids_sum[index] / count;
        }
        // **Optimization**: Reset the sum buffer for the next iteration.
        d_new_centroids_sum[index] = 0.0f;
    }

    // Have the first few threads reset the cluster counts array for the next iteration.
    for (int cluster_idx = i; cluster_idx < num_clusters; cluster_idx += stride) {
        d_cluster_counts[cluster_idx] = 0;
    }
}

__global__ void check_convergence_kernel(const float* d_centroids, const float* d_old_centroids, int* d_converged, int num_clusters, int dims, float threshold_sq) {
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cluster_idx >= num_clusters) {
        return;
    }

    // Calculate squared distance between old and new centroid
    float dist_sq = 0.0f;
    for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
        float diff = d_centroids[cluster_idx * dims + dim_idx] - d_old_centroids[cluster_idx * dims + dim_idx];
        dist_sq += diff * diff;
    }

    // If any centroid moved more than the threshold, we have not converged.
    // This write is a benign race condition: if multiple threads write 0, the result is still 0.
    if (dist_sq > threshold_sq) {
        *d_converged = 0; // Set flag to "not converged"
    }
}