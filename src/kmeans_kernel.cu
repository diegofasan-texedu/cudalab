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
                                               double* centroid_sums,
                                               int* cluster_counts,
                                               int num_points,
                                               int dims) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < num_points) {
        int assigned_cluster = cluster_assignments[point_idx];

        if (assigned_cluster != -1) {
            // Atomically increment the count for the assigned cluster
            atomicAdd(&cluster_counts[assigned_cluster], 1);

            // Atomically add the point's coordinates to the sum for that cluster
            for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
                atomicAdd(&centroid_sums[assigned_cluster * dims + dim_idx], points[point_idx * dims + dim_idx]);
            }
        }
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