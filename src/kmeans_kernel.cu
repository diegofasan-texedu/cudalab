#include "kmeans_kernel.cuh"
#include <float.h> // For DBL_MAX
#include <stdio.h> // For printf

__global__ void assign_clusters_kernel(const double* points, const double* centroids, int* cluster_assignments, int num_points, int num_clusters, int dims) {
    // Get the unique index for this thread, which corresponds to a data point.
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: ensure the thread is not accessing beyond the number of points.
    if (point_idx >= num_points) {
        return;
    }

    double min_dist_sq = DBL_MAX;
    int best_cluster = 0; // Default to the first cluster

    // Find the closest centroid for the current point.
    for (int cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
        double current_dist_sq = 0.0;
        
        // Calculate squared Euclidean distance to save on sqrt() operations.
        for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
            double diff = points[point_idx * dims + dim_idx] - centroids[cluster_idx * dims + dim_idx];
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

__global__ void reset_update_buffers_kernel(double* d_new_centroids_sum, int* d_cluster_counts, int num_clusters, int dims) {
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cluster_idx >= num_clusters) {
        return;
    }

    // Reset the count for this cluster to zero.
    d_cluster_counts[cluster_idx] = 0;

    // Reset the sum of coordinates for this cluster to zero.
    for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
        d_new_centroids_sum[cluster_idx * dims + dim_idx] = 0.0;
    }
}

__global__ void update_centroids_sum_kernel(const double* d_points, const int* d_cluster_assignments, double* d_new_centroids_sum, int* d_cluster_counts, int num_points, int dims) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx >= num_points) {
        return;
    }

    // Get the cluster this point is assigned to.
    int assigned_cluster = d_cluster_assignments[point_idx];

    // Atomically increment the count for this cluster.
    atomicAdd(&d_cluster_counts[assigned_cluster], 1);

    // Atomically add this point's coordinates to the sum for its cluster.
    // Note: atomicAdd for double requires a device with compute capability 6.x or higher.
    for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
        double point_coord = d_points[point_idx * dims + dim_idx];
        atomicAdd(&d_new_centroids_sum[assigned_cluster * dims + dim_idx], point_coord);
    }
}

__global__ void calculate_new_centroids_kernel(double* d_centroids, const double* d_new_centroids_sum, const int* d_cluster_counts, int num_clusters, int dims) {
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cluster_idx >= num_clusters) {
        return;
    }

    int count = d_cluster_counts[cluster_idx];

    // Avoid division by zero if a cluster becomes empty.
    // In this case, the old centroid position is kept.
    if (count > 0) {
        for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
            // Calculate the new centroid position (mean).
            d_centroids[cluster_idx * dims + dim_idx] = d_new_centroids_sum[cluster_idx * dims + dim_idx] / count;
        }
    }
    // If count is 0, we do nothing, leaving the centroid at its previous position.
    // A more advanced implementation might re-initialize empty clusters.
}
