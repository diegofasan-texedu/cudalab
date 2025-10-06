#include "dataset.cuh"
#include "thrust_kmeans.cuh"
#include "error.cuh"
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
        double min_dist_sq = 1.7976931348623158e+308; // DBL_MAX
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

// --- Functors for the update and convergence steps ---

// Functor to calculate the new centroid by dividing sum by count.
struct divide_by_count_functor {
    const int* counts;
    int dims;

    divide_by_count_functor(const int* _counts, int _dims) : counts(_counts), dims(_dims) {}

    // This functor is used in a zip_iterator, so it operates on a tuple.
    __host__ __device__
    double operator()(const thrust::tuple<double, int>& t) const {
        const double sum = thrust::get<0>(t);
        const int i = thrust::get<1>(t); // This is the global index of the sum element
        const int count = counts[i / dims];
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


/**
 * @brief A custom kernel to perform the multi-dimensional summation using atomicAdd.
 * This is often more straightforward than a full Thrust-based sort/reduce_by_key for
 * the multi-dimensional reduction.
 */
__global__ void update_centroids_sum_kernel(const double* d_points, const int* d_cluster_assignments, double* d_new_centroids_sum, int num_points, int dims) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (point_idx >= num_points) {
        return;
    }
 
    int assigned_cluster = d_cluster_assignments[point_idx];
 
    for (int dim_idx = 0; dim_idx < dims; ++dim_idx) {
        double point_coord = d_points[point_idx * dims + dim_idx];
        atomicAdd(&d_new_centroids_sum[assigned_cluster * dims + dim_idx], point_coord);
    }
}

void thrust_kmeans(int num_cluster, KmeansData& data, int max_num_iter, double threshold, bool output_centroids_flag, int seed, bool verbose) {
    if (verbose) {
        std::cout << "Executing Thrust K-Means..." << std::endl;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    const int num_points = data.num_points;
    const int dims = data.dims;

    // --- 1. Allocate GPU Memory using device_vectors ---
    thrust::device_vector<double> d_points(data.h_points, data.h_points + num_points * dims);
    thrust::device_vector<double> d_centroids(data.h_centroids, data.h_centroids + num_cluster * dims);
    thrust::device_vector<int> d_cluster_assignments(num_points);

    // --- 2. Allocate temporary buffers for intermediate steps ---
    thrust::device_vector<double> d_old_centroids(num_cluster * dims);
    
    // Buffers for the update step
    thrust::device_vector<double> d_scattered_sums(num_cluster * dims);
    thrust::device_vector<int> d_scattered_counts(num_cluster);

    // Buffer for convergence check
    thrust::device_vector<double> d_centroid_dist_sq(num_cluster);

    int iter_to_converge = 0;

    // --- 3. K-Means Iteration Loop ---
    for (int iter = 0; iter < max_num_iter; ++iter) {
        iter_to_converge = iter + 1;

        // Store current centroids to check for convergence later
        thrust::copy(d_centroids.begin(), d_centroids.end(), d_old_centroids.begin());

        // == A. Assignment Step ==
        // Use a transform with a custom functor to find the best cluster for each point.
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
        // This step is a segmented reduction. We will do it in three parts.

        // B.1. Calculate cluster counts.
        // First, create a temporary sorted copy of assignments to count occurrences.
        thrust::device_vector<int> d_sorted_assignments = d_cluster_assignments;
        thrust::sort(d_sorted_assignments.begin(), d_sorted_assignments.end());

        // Use reduce_by_key to count unique assignments.
        // The keys are the sorted assignments, the values are all 1s.
        thrust::device_vector<int> d_unique_keys(num_cluster);
        thrust::device_vector<int> d_compact_counts(num_cluster);
        auto end_unique = thrust::reduce_by_key(
            d_sorted_assignments.begin(),
            d_sorted_assignments.end(),
            thrust::constant_iterator<int>(1),
            d_unique_keys.begin(),
            d_compact_counts.begin()
        );
        // The result is a "compact" list. We need to scatter it back to a full-size array.
        thrust::fill(d_scattered_counts.begin(), d_scattered_counts.end(), 0);
        thrust::scatter(
            d_compact_counts.begin(),
            d_compact_counts.begin() + (end_unique.first - d_unique_keys.begin()), // Calculate end iterator for values
            d_unique_keys.begin(),
            d_scattered_counts.begin()
        );

        // B.2. Sum point coordinates for each cluster.
        // Using a custom kernel with atomicAdd is often simpler for multi-dimensional sums.
        thrust::fill(d_scattered_sums.begin(), d_scattered_sums.end(), 0.0);
        int threads_per_block = 256;
        int point_blocks = (num_points + threads_per_block - 1) / threads_per_block;
        update_centroids_sum_kernel<<<point_blocks, threads_per_block>>>(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(d_cluster_assignments.data()),
            thrust::raw_pointer_cast(d_scattered_sums.data()),
            num_points,
            dims
        );
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // B.3. Divide sums by counts to get the new centroids.
        // We use a transform operation to update d_centroids in place.
        thrust::transform(
            // Zip together the sums and an index to look up the correct count
            thrust::make_zip_iterator(thrust::make_tuple(
                d_scattered_sums.begin(),
                thrust::counting_iterator<int>(0)
            )),
            thrust::make_zip_iterator(thrust::make_tuple(
                d_scattered_sums.end(),
                thrust::counting_iterator<int>(num_cluster * dims)
            )),
            d_centroids.begin(),
            divide_by_count_functor(thrust::raw_pointer_cast(d_scattered_counts.data()), dims)
        );

        // == C. Convergence Check ==
        const double threshold_sq = threshold * threshold;
        const double* raw_new_centroids = thrust::raw_pointer_cast(d_centroids.data());
        const double* raw_old_centroids = thrust::raw_pointer_cast(d_old_centroids.data());

        // Calculate the squared distance moved for each centroid
        thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_cluster),
            d_centroid_dist_sq.begin(),
            centroid_distance_functor(raw_new_centroids, raw_old_centroids, dims)
        );

        // Find the maximum distance any centroid has moved.
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

    // Print results in the format: iter_count,avg_iter_ms
    printf("%d,%lf\n", iter_to_converge, total_milliseconds.count() / iter_to_converge);

    // --- 4. Copy Final Results from Device to Host ---
    thrust::copy(d_centroids.begin(), d_centroids.end(), data.h_centroids);
    thrust::copy(d_cluster_assignments.begin(), d_cluster_assignments.end(), data.h_cluster_assignments);
}