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

__global__ void sum_points_for_clusters_kernel(const double* points, const int* cluster_assignments, double* centroid_sums, int* cluster_counts, const int num_points, const int dims) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < num_points) {
        int cluster_idx = cluster_assignments[point_idx];
        atomicAdd(&cluster_counts[cluster_idx], 1);
        for (int d = 0; d < dims; ++d) {
            atomicAdd(&centroid_sums[cluster_idx * dims + d], points[point_idx * dims + d]);
        }
    }
}

__global__ void average_clusters_kernel(double* new_centroids, const double* centroid_sums, const int* cluster_counts, const int num_clusters, const int dims) {
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cluster_idx < num_clusters) {
        int count = cluster_counts[cluster_idx];
        if (count > 0) {
            for (int d = 0; d < dims; ++d) {
                new_centroids[cluster_idx * dims + d] = centroid_sums[cluster_idx * dims + d] / count;
            }
        }
    }
}
