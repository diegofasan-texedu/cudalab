#include "dataset.cuh"
#include "thrust_kmeans.cuh"
#include "error.cuh"
#include <iostream>
#include <chrono>
#include <float.h>

// Functor to find the nearest centroid for a given point.
struct assign_cluster_functor {
    const float* points;
    const float* centroids;
    int num_clusters;
    int dims;

    assign_cluster_functor(const float* _points, const float* _centroids, int _num_clusters, int _dims)
        : points(_points), centroids(_centroids), num_clusters(_num_clusters), dims(_dims) {}

    __host__ __device__ int operator()(int point_idx) const {
        float min_dist_sq = FLT_MAX;
        int best_cluster = 0;
        
        for (int cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
            float current_dist_sq = 0.0f;
            for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
                float diff = points[point_idx * dims + dim_idx] - centroids[cluster_idx * dims + dim_idx];
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

// Kernel to sum points and counts for each cluster using atomics.
__global__ void update_centroids_kernel(
    const float* __restrict__ d_points,
    const int* __restrict__ d_cluster_assignments,
    float* __restrict__ d_sums,
    int* __restrict__ d_counts,
    int num_points,
    int dims
) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < num_points) {
        int cluster_idx = d_cluster_assignments[point_idx];

        atomicAdd(&d_counts[cluster_idx], 1);

        for (int d = 0; d < dims; ++d) {
            float point_coord = d_points[point_idx * dims + d];
            int sum_idx = cluster_idx * dims + d;
            atomicAdd(&d_sums[sum_idx], point_coord);
        }
    }
};

// Functor to calculate a new centroid by dividing the sum of points by the count.
struct divide_by_count_functor {
    const int* counts;
    int dims;

    divide_by_count_functor(const int* _counts, int _dims) : counts(_counts), dims(_dims) {}

    __host__ __device__
    float operator()(const thrust::tuple<float, int>& t) const {
        const float sum = thrust::get<0>(t);
        const int i = thrust::get<1>(t);
        const int count = counts[i / dims];
        return (count > 0) ? sum / count : 0.0f;
    }
};

// Functor to calculate the squared distance a centroid has moved.
struct centroid_distance_functor {
    const float* new_centroids;
    const float* old_centroids;
    int dims;

    centroid_distance_functor(const float* _new, const float* _old, int _dims)
        : new_centroids(_new), old_centroids(_old), dims(_dims) {}

    __host__ __device__
    float operator()(int cluster_idx) const {
        float dist_sq = 0.0f;
        for (int d = 0; d < dims; ++d) {
            int i = cluster_idx * dims + d;
            float diff = new_centroids[i] - old_centroids[i];
            dist_sq += diff * diff;
        }
        return dist_sq;
    }
};

void thrust_kmeans(int num_cluster, KmeansData& data, int max_num_iter, float threshold, bool output_centroids_flag, int seed, bool verbose) {
    if (verbose) {
        std::cout << "Executing Thrust K-Means (Full-Batch, float)..." << std::endl;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    const int num_points = data.num_points;
    const int dims = data.dims;
    const int threads_per_block = 256;
    const int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;

    thrust::device_vector<float> d_points(data.h_points, data.h_points + num_points * dims);
    thrust::device_vector<float> d_centroids(data.h_centroids, data.h_centroids + num_cluster * dims);
    thrust::device_vector<int> d_cluster_assignments(num_points);

    thrust::device_vector<float> d_old_centroids(num_cluster * dims);
    thrust::device_vector<float> d_sums(num_cluster * dims);
    thrust::device_vector<int> d_counts(num_cluster);
    thrust::device_vector<float> d_centroid_dist_sq(num_cluster);

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
        thrust::fill(d_sums.begin(), d_sums.end(), 0.0f);
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
        const float threshold_sq = threshold * threshold;
        const float* raw_new_centroids = thrust::raw_pointer_cast(d_centroids.data());
        const float* raw_old_centroids = thrust::raw_pointer_cast(d_old_centroids.data());

        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_cluster),
            d_centroid_dist_sq.begin(),
            centroid_distance_functor(raw_new_centroids, raw_old_centroids, dims)
        );

        float max_dist_sq = *thrust::max_element(d_centroid_dist_sq.begin(), d_centroid_dist_sq.end());

        if (max_dist_sq < threshold_sq) {
            if (verbose) {
                std::cout << "Convergence reached after " << iter_to_converge << " iterations." << std::endl;
            }
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_milliseconds = end_time - start_time;

    printf("%d,%f\n", iter_to_converge, total_milliseconds.count() / iter_to_converge);

    thrust::copy(d_centroids.begin(), d_centroids.end(), data.h_centroids);
    thrust::copy(d_cluster_assignments.begin(), d_cluster_assignments.end(), data.h_cluster_assignments);
}