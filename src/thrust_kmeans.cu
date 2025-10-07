#include "dataset.cuh"
#include "thrust_kmeans.cuh"
#include "error.cuh"
#include <float.h> // For DBL_MAX
#include <iostream>
#include <chrono>

// Functor to find the nearest centroid for a given point.
struct assign_cluster_functor {
    const double* points;
    const double* centroids;
    int num_clusters;
    int dims;

    assign_cluster_functor(const double* _points, const double* _centroids, int _num_clusters, int _dims)
        : points(_points), centroids(_centroids), num_clusters(_num_clusters), dims(_dims) {}

    __host__ __device__ int operator()(int point_idx) const {
        double min_dist_sq = DBL_MAX;
        int best_cluster = 0;
        
        for (int cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
            double current_dist_sq = 0.0;
            for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
                double diff = points[point_idx * dims + dim_idx] - centroids[cluster_idx * dims + dim_idx];
                current_dist_sq += diff * diff;
            }
            
            if (current_dist_sq < min_dist_sq) {
                min_dist_sq = current_dist_sq;
                best_cluster = cluster_idx;
            }
        }
        return best_cluster;
    }
};

__global__ void update_centroids_kernel(
    const double* __restrict__ d_points,
    const int* __restrict__ d_cluster_assignments,
    double* __restrict__ d_sums,
    int* __restrict__ d_counts,
    int num_points,
    int dims
) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < num_points) {
        int cluster_idx = d_cluster_assignments[point_idx];

        atomicAdd(&d_counts[cluster_idx], 1);

        for (int d = 0; d < dims; ++d) {
            double point_coord = d_points[point_idx * dims + d];
            int sum_idx = cluster_idx * dims + d;
            atomicAdd(&d_sums[sum_idx], point_coord);
        }
    }
};

// --- FUNCTOR: Division (New Centroid = Sum / Count) ---
struct divide_by_count_functor {
    const int* counts;
    int dims;

    divide_by_count_functor(const int* _counts, int _dims) : counts(_counts), dims(_dims) {}

    __host__ __device__
    double operator()(const thrust::tuple<double, int>& t) const {
        const double sum = thrust::get<0>(t);
        const int i = thrust::get<1>(t);
        const int count = counts[i / dims];
        return (count > 0) ? sum / count : 0.0;
    }
};

// Functor to calculate the squared distance a centroid has moved.
struct centroid_distance_functor {
    const double* new_centroids;
    const double* old_centroids;
    int dims;

    centroid_distance_functor(const double* _new, const double* _old, int _dims)
        : new_centroids(_new), old_centroids(_old), dims(_dims) {}

    __host__ __device__
    double operator()(int cluster_idx) const {
        double dist_sq = 0.0;
        for (int d = 0; d < dims; ++d) {
            int i = cluster_idx * dims + d;
            double diff = new_centroids[i] - old_centroids[i];
            dist_sq += diff * diff;
        }
        return dist_sq;
    }
};

void thrust_kmeans(int num_cluster, KmeansData& data, int max_num_iter, double threshold, bool output_centroids_flag, int seed, bool verbose) {
    if (verbose) {
        std::cout << "Executing Thrust K-Means (Full-Batch)..." << std::endl;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    const int num_points = data.num_points;
    const int dims = data.dims;
    const int threads_per_block = 256;
    const int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;

    thrust::device_vector<double> d_points(data.h_points, data.h_points + num_points * dims);
    thrust::device_vector<double> d_centroids(data.h_centroids, data.h_centroids + num_cluster * dims);
    thrust::device_vector<int> d_cluster_assignments(num_points);

    thrust::device_vector<double> d_old_centroids(num_cluster * dims);
    thrust::device_vector<double> d_sums(num_cluster * dims);
    thrust::device_vector<int> d_counts(num_cluster);
    thrust::device_vector<double> d_centroid_dist_sq(num_cluster);

    int iter_to_converge = 0;

    // K-Means Iteration Loop
    for (int iter = 0; iter < max_num_iter; ++iter) {
        iter_to_converge = iter + 1;

        thrust::copy(d_centroids.begin(), d_centroids.end(), d_old_centroids.begin());

        // == A. Assignment Step ==
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(num_points),
            d_cluster_assignments.begin(),
            assign_cluster_functor(
                thrust::raw_pointer_cast(d_points.data()),
                thrust::raw_pointer_cast(d_centroids.data()),
                num_cluster,
                dims
            )
        );

        // == B. Update Step ==
        thrust::fill(d_sums.begin(), d_sums.end(), 0.0);
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        update_centroids_kernel<<<num_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(d_cluster_assignments.data()),
            thrust::raw_pointer_cast(d_sums.data()),
            thrust::raw_pointer_cast(d_counts.data()),
            num_points,
            dims
        );
        HANDLE_CUDA_ERROR(cudaGetLastError());

        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(
                d_sums.begin(),
                thrust::counting_iterator<int>(0)
            )),
            thrust::make_zip_iterator(thrust::make_tuple(
                d_sums.end(),
                thrust::counting_iterator<int>(num_cluster * dims)
            )),
            d_centroids.begin(),
            divide_by_count_functor(thrust::raw_pointer_cast(d_counts.data()), dims)
        );

        // == C. Convergence Check ==
        const double threshold_sq = threshold * threshold;
        const double* raw_new_centroids = thrust::raw_pointer_cast(d_centroids.data());
        const double* raw_old_centroids = thrust::raw_pointer_cast(d_old_centroids.data());

        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_cluster),
            d_centroid_dist_sq.begin(),
            centroid_distance_functor(raw_new_centroids, raw_old_centroids, dims)
        );

        double max_dist_sq = *thrust::max_element(d_centroid_dist_sq.begin(), d_centroid_dist_sq.end());

        if (max_dist_sq < threshold_sq) {
            if (verbose) {
                std::cout << "Convergence reached after " << iter_to_converge << " iterations." << std::endl;
            }
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_milliseconds = end_time - start_time;

    printf("%d,%lf\n", iter_to_converge, total_milliseconds.count() / iter_to_converge);

    thrust::copy(d_centroids.begin(), d_centroids.end(), data.h_centroids);
    thrust::copy(d_cluster_assignments.begin(), d_cluster_assignments.end(), data.h_cluster_assignments);
}