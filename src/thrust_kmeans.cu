#include "dataset.cuh"
#include "thrust_kmeans.cuh"
#include "error.cuh"
#include <float.h> // For DBL_MAX
#include <iostream>
#include <chrono>

// --- Functor for the assignment step (Device-side distance calculation) ---
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
        
        // Loop through all clusters
        for (int cluster_idx = 0; cluster_idx < num_clusters; ++cluster_idx) {
            double current_dist_sq = 0.0;
            
            // Loop through all dimensions
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

// --- Configuration ---
#define MINI_BATCH_SIZE 1024
#define THREADS_PER_BLOCK 256
// ---------------------

// A simple device-side LCG random number generator.
__device__ unsigned int lcg_rand(unsigned int *state) {
    *state = (*state * 1103515245 + 12345);
    return (*state / 65536) % 32768;
}

// --- KERNEL: Random Sampling (Selects M points from N) ---
__global__ void sample_indices_kernel(int* d_sample_indices, int num_points, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        // Each thread gets a unique seed. clock64() is a fast device-side clock.
        unsigned int seed = clock64() + idx;
        d_sample_indices[idx] = lcg_rand(&seed) % num_points;
    }
}

// --- KERNEL: Gather Mini-Batch Points (Replaces Thrust Gather) ---
__global__ void gather_points_kernel(
    const double* __restrict__ d_points,
    const int* __restrict__ d_sample_indices,
    double* __restrict__ d_minibatch_points,
    int M, // Mini-Batch Size
    int dims
) {
    int idx_in_batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_in_batch < M) {
        int original_point_idx = d_sample_indices[idx_in_batch];
        for (int d = 0; d < dims; ++d) {
            d_minibatch_points[idx_in_batch * dims + d] = d_points[original_point_idx * dims + d];
        }
    }
}

// --- KERNEL: Mini-Batch Centroid Update (O(M*D) with Atomics) ---
__global__ void update_minibatch_centroids_kernel(
    const double* __restrict__ d_points,
    const int* __restrict__ d_cluster_assignments,
    double* __restrict__ d_sums,
    int* __restrict__ d_counts,
    int M, // Mini-Batch Size
    int dims
) {
    // One thread per point in the mini-batch (M)
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_idx < M) {
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
        // Only update if count > 0, otherwise the sum (which is now in d_centroids) is reset to 0.0
        return (count > 0) ? sum / count : 0.0;
    }
};

// Functor to calculate the squared distance between old and new centroids.
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
        std::cout << "Executing Mini-Batch K-Means for fastest wall-clock time..." << std::endl;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    const int num_points = data.num_points;
    const int dims = data.dims;
    const int M = MINI_BATCH_SIZE;

    // Ensure mini-batch size is not larger than the dataset
    const int actual_M = (num_points < M) ? num_points : M;
    const int num_blocks_M = (actual_M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // --- 1. Allocate GPU Memory using device_vectors ---
    thrust::device_vector<double> d_points(data.h_points, data.h_points + num_points * dims);
    thrust::device_vector<double> d_centroids(data.h_centroids, data.h_centroids + num_cluster * dims);
    thrust::device_vector<int> d_minibatch_assignments(actual_M);
    thrust::device_vector<int> d_sample_indices(actual_M);
    thrust::device_vector<double> d_minibatch_points(actual_M * dims);

    // --- 2. Allocate temporary buffers for intermediate steps ---
    thrust::device_vector<double> d_old_centroids(num_cluster * dims);
    thrust::device_vector<double> d_sums(num_cluster * dims);
    thrust::device_vector<int> d_counts(num_cluster);
    thrust::device_vector<double> d_centroid_dist_sq(num_cluster);

    int iter_to_converge = 0;

    // --- 3. K-Means Iteration Loop ---
    for (int iter = 0; iter < max_num_iter; ++iter) {
        iter_to_converge = iter + 1;

        thrust::copy(d_centroids.begin(), d_centroids.end(), d_old_centroids.begin());

        // == A. Sampling Step ==
        sample_indices_kernel<<<num_blocks_M, THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_sample_indices.data()), num_points, actual_M);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // A.2: Gather the mini-batch points (O(M*D))
        gather_points_kernel<<<num_blocks_M, THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(d_sample_indices.data()),
            thrust::raw_pointer_cast(d_minibatch_points.data()),
            actual_M,
            dims
        );
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // == B. Mini-Batch Assignment Step (O(M*K*D)) ==
        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(actual_M),
            d_minibatch_assignments.begin(),
            assign_cluster_functor(
                thrust::raw_pointer_cast(d_minibatch_points.data()),
                thrust::raw_pointer_cast(d_centroids.data()),
                num_cluster,
                dims
            )
        );

        // == C. Mini-Batch Update Step (O(M*D)) ==
        thrust::fill(d_sums.begin(), d_sums.end(), 0.0);
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        update_minibatch_centroids_kernel<<<num_blocks_M, THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_minibatch_points.data()),
            thrust::raw_pointer_cast(d_minibatch_assignments.data()),
            thrust::raw_pointer_cast(d_sums.data()),
            thrust::raw_pointer_cast(d_counts.data()),
            actual_M,
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

        // == D. Convergence Check ==
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
                std::cout << "Mini-Batch Convergence reached after " << iter_to_converge << " iterations." << std::endl;
            }
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_milliseconds = end_time - start_time;

    // Print results in the format: iter_count,avg_iter_ms
    printf("%d,%lf\n", iter_to_converge, total_milliseconds.count() / iter_to_converge);

    // --- 4. Final Assignment and Copy Results ---
    // After convergence, run one final assignment step on the full dataset
    // to get the correct assignments for all points.
    thrust::device_vector<int> d_final_assignments(num_points);
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(num_points),
        d_final_assignments.begin(),
        assign_cluster_functor(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(d_centroids.data()),
            num_cluster, dims)
    );

    thrust::copy(d_centroids.begin(), d_centroids.end(), data.h_centroids);
    thrust::copy(d_final_assignments.begin(), d_final_assignments.end(), data.h_cluster_assignments);
}