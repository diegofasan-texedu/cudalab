// Include headers for k-means implementations
#include "thrust_kmeans.cuh"

// Include headers for program setup
#include "argparse.cuh"
#include "io.cuh"
#include "dataset.cuh"
#include "kmeans_kernel.cuh" // For CUDA kernels
#include "error.cuh"         // For HANDLE_CUDA_ERROR
#include <iomanip> // Required for setprecision and fixed

// For log10, ceil, and std::min
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>

#include <stdio.h>
#include <iostream>

void sequential_kmeans(int num_cluster, KmeansData& data, int max_num_iter, double threshold, bool output_centroids_flag, int seed, bool verbose) {
    if (verbose) {
        std::cout << "Executing Sequential K-Means..." << std::endl;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    const int num_points = data.num_points;
    const int dims = data.dims;
    const double threshold_sq = threshold * threshold;

    // --- 1. Allocate Memory for Intermediate Calculations ---
    std::vector<double> old_centroids(num_cluster * dims);
    std::vector<double> new_centroids_sum(num_cluster * dims, 0.0);
    std::vector<int> cluster_counts(num_cluster, 0);

    int iter_to_converge = 0;
    bool converged = false;

    // --- 2. K-Means Iteration Loop ---
    for (int iter = 0; iter < max_num_iter; ++iter) {
        iter_to_converge = iter + 1;

        // Store current centroids to check for convergence later
        memcpy(old_centroids.data(), data.h_centroids, (size_t)num_cluster * dims * sizeof(double));

        // == Assignment Step ==
        // For each point, find the nearest centroid
        for (int p_idx = 0; p_idx < num_points; ++p_idx) {
            double min_dist_sq = -1.0;
            int best_cluster = 0;

            for (int c_idx = 0; c_idx < num_cluster; ++c_idx) {
                double current_dist_sq = 0.0;
                // Calculate squared Euclidean distance
                for (int d_idx = 0; d_idx < dims; ++d_idx) {
                    double diff = data.h_points[p_idx * dims + d_idx] - data.h_centroids[c_idx * dims + d_idx];
                    current_dist_sq += diff * diff;
                }

                if (min_dist_sq < 0 || current_dist_sq < min_dist_sq) {
                    min_dist_sq = current_dist_sq;
                    best_cluster = c_idx;
                }
            }
            data.h_cluster_assignments[p_idx] = best_cluster;
        }

        // == Update Step ==
        // 1. Reset sum and count buffers
        std::fill(new_centroids_sum.begin(), new_centroids_sum.end(), 0.0);
        std::fill(cluster_counts.begin(), cluster_counts.end(), 0);

        // 2. Sum up points for each cluster
        for (int p_idx = 0; p_idx < num_points; ++p_idx) {
            int cluster_id = data.h_cluster_assignments[p_idx];
            cluster_counts[cluster_id]++;
            for (int d_idx = 0; d_idx < dims; ++d_idx) {
                new_centroids_sum[cluster_id * dims + d_idx] += data.h_points[p_idx * dims + d_idx];
            }
        }

        // 3. Calculate new centroids (mean of points)
        for (int c_idx = 0; c_idx < num_cluster; ++c_idx) {
            if (cluster_counts[c_idx] > 0) {
                for (int d_idx = 0; d_idx < dims; ++d_idx) {
                    data.h_centroids[c_idx * dims + d_idx] = new_centroids_sum[c_idx * dims + d_idx] / cluster_counts[c_idx];
                }
            }
        }

        // == Convergence Check ==
        converged = true;
        for (int c_idx = 0; c_idx < num_cluster; ++c_idx) {
            double dist_sq = 0.0;
            for (int d_idx = 0; d_idx < dims; ++d_idx) {
                double diff = data.h_centroids[c_idx * dims + d_idx] - old_centroids[c_idx * dims + d_idx];
                dist_sq += diff * diff;
            }
            if (dist_sq > threshold_sq) {
                converged = false;
                break; // A centroid moved too much, no need to check others
            }
        }

        if (converged) {
            if (verbose) std::cout << "Convergence reached after " << iter_to_converge << " iterations." << std::endl;
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_milliseconds = end_time - start_time;

    if (iter_to_converge == max_num_iter && !converged) {
        if (verbose) std::cout << "Max iterations (" << max_num_iter << ") reached without convergence." << std::endl;
    }

    // Print results in the format: iter_count,avg_iter_ms
    printf("%d,%lf\n", iter_to_converge, total_milliseconds.count() / iter_to_converge);
}

void smem_cuda_kmeans(int num_cluster, KmeansData& data, int max_num_iter, double threshold, bool output_centroids_flag, bool verbose) {
    if (verbose) {
        std::cout << "Executing CUDA K-Means with Shared Memory..." << std::endl;
    }

    cudaEvent_t start, stop;
    HANDLE_CUDA_ERROR(cudaEventCreate(&start));
    HANDLE_CUDA_ERROR(cudaEventCreate(&stop));
    HANDLE_CUDA_ERROR(cudaEventRecord(start));
    
    const int num_points = data.num_points;
    const int dims = data.dims;

    size_t points_size = (size_t)num_points * dims * sizeof(double);
    size_t centroids_size = (size_t)num_cluster * dims * sizeof(double);
    size_t assignments_size = (size_t)num_points * sizeof(int);

    double* d_new_centroids_sum;
    int* d_cluster_counts;
    int* d_cluster_assignments;
    double* d_old_centroids;
    int* d_converged;
    int h_converged = 0;
    const double threshold_sq = threshold * threshold;

    HANDLE_CUDA_ERROR(cudaMalloc(&data.d_points, points_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&data.d_centroids, centroids_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_cluster_assignments, assignments_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_new_centroids_sum, centroids_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_cluster_counts, (size_t)num_cluster * sizeof(int)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_old_centroids, centroids_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_converged, sizeof(int)));

    HANDLE_CUDA_ERROR(cudaMemcpy(data.d_points, data.h_points, points_size, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(data.d_centroids, data.h_centroids, centroids_size, cudaMemcpyHostToDevice));

    // Optimization: Use cudaMemsetAsync for initial zeroing of buffers.
    HANDLE_CUDA_ERROR(cudaMemsetAsync(d_new_centroids_sum, 0, centroids_size));
    HANDLE_CUDA_ERROR(cudaMemsetAsync(d_cluster_counts, 0, (size_t)num_cluster * sizeof(int)));

    int threads_per_block = 256;
    int point_blocks = (num_points + threads_per_block - 1) / threads_per_block;
    int cluster_blocks = (num_cluster + threads_per_block - 1) / threads_per_block;
    size_t smem_size = centroids_size; // Shared memory needed for all centroids

    // Calculate shared memory size for the update kernel.
    size_t update_smem_size = (num_cluster * dims * sizeof(double)) + (num_cluster * sizeof(int));
    // It's good practice to check if this exceeds device limits, though for typical
    // k-means problems it should be fine.

    int iter_to_converge = 0;
    for (int iter = 0; iter < max_num_iter; ++iter) {
        iter_to_converge = iter + 1;

        HANDLE_CUDA_ERROR(cudaMemcpy(d_old_centroids, data.d_centroids, centroids_size, cudaMemcpyDeviceToDevice));

        // == Assignment Step (Shared Memory) ==
        assign_clusters_smem_kernel<<<point_blocks, threads_per_block, smem_size>>>(
            data.d_points, data.d_centroids, d_cluster_assignments, num_points, num_cluster, dims);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // == Update Step (Shared Memory Reduction) ==
        update_centroids_sum_smem_kernel<<<point_blocks, threads_per_block, update_smem_size>>>(
            data.d_points, d_cluster_assignments, d_new_centroids_sum, d_cluster_counts, num_points, num_cluster, dims);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // Use the shared memory version of the calculate/reset kernel.
        // Launch one block per cluster.
        calculate_and_reset_smem_kernel<<<num_cluster, threads_per_block>>>(data.d_centroids, d_new_centroids_sum, d_cluster_counts, num_cluster, dims);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        h_converged = 1;
        HANDLE_CUDA_ERROR(cudaMemcpy(d_converged, &h_converged, sizeof(int), cudaMemcpyHostToDevice));

        // Reverted to the non-shared memory version of the convergence check.
        check_convergence_kernel<<<cluster_blocks, threads_per_block>>>(data.d_centroids, d_old_centroids, d_converged, num_cluster, dims, threshold_sq);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        HANDLE_CUDA_ERROR(cudaMemcpy(&h_converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_converged) break;
    }

    HANDLE_CUDA_ERROR(cudaEventRecord(stop));
    HANDLE_CUDA_ERROR(cudaEventSynchronize(stop));
    float total_milliseconds = 0;
    HANDLE_CUDA_ERROR(cudaEventElapsedTime(&total_milliseconds, start, stop));

    printf("%d,%lf\n", iter_to_converge, total_milliseconds / iter_to_converge);

    HANDLE_CUDA_ERROR(cudaMemcpy(data.h_centroids, data.d_centroids, centroids_size, cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaMemcpy(data.h_cluster_assignments, d_cluster_assignments, assignments_size, cudaMemcpyDeviceToHost));

    HANDLE_CUDA_ERROR(cudaFree(data.d_points));
    HANDLE_CUDA_ERROR(cudaFree(data.d_centroids));
    HANDLE_CUDA_ERROR(cudaFree(d_cluster_assignments));
    HANDLE_CUDA_ERROR(cudaFree(d_old_centroids));
    HANDLE_CUDA_ERROR(cudaFree(d_new_centroids_sum));
    HANDLE_CUDA_ERROR(cudaFree(d_cluster_counts));
    HANDLE_CUDA_ERROR(cudaFree(d_converged));
    HANDLE_CUDA_ERROR(cudaEventDestroy(start));
    HANDLE_CUDA_ERROR(cudaEventDestroy(stop));
    data.d_points = nullptr;
    data.d_centroids = nullptr;
}

void cuda_kmeans(int num_cluster, KmeansData& data, int max_num_iter, double threshold, bool output_centroids_flag, bool verbose) {
    if (verbose) {
        std::cout << "Executing CUDA K-Means..." << std::endl;
    }


    // --- CUDA Events for Timing ---
    cudaEvent_t start, stop;
    HANDLE_CUDA_ERROR(cudaEventCreate(&start));
    HANDLE_CUDA_ERROR(cudaEventCreate(&stop));
    HANDLE_CUDA_ERROR(cudaEventRecord(start)); // Start timer before the loop
    
    // Extract data dimensions for clarity
    const int num_points = data.num_points;
    const int dims = data.dims;

    // --- 1. Allocate GPU Memory ---
    size_t points_size = (size_t)num_points * dims * sizeof(double);
    size_t centroids_size = (size_t)num_cluster * dims * sizeof(double);
    size_t assignments_size = (size_t)num_points * sizeof(int);

    // Memory for the update step
    double* d_new_centroids_sum;
    int* d_cluster_counts;
    
    // Device pointers
    int* d_cluster_assignments;
    double* d_old_centroids;
    int* d_converged;
    int h_converged = 0; // Host-side convergence flag
    const double threshold_sq = threshold * threshold;

    // Allocate memory for points, centroids, and assignments
    HANDLE_CUDA_ERROR(cudaMalloc(&data.d_points, points_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&data.d_centroids, centroids_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_cluster_assignments, assignments_size));
    
    // Allocate memory for update step buffers
    HANDLE_CUDA_ERROR(cudaMalloc(&d_new_centroids_sum, centroids_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_cluster_counts, (size_t)num_cluster * sizeof(int)));

    // Allocate memory for convergence check
    HANDLE_CUDA_ERROR(cudaMalloc(&d_old_centroids, centroids_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_converged, sizeof(int)));

    // --- 2. Copy Initial Data from Host to Device ---
    HANDLE_CUDA_ERROR(cudaMemcpy(data.d_points, data.h_points, points_size, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(data.d_centroids, data.h_centroids, centroids_size, cudaMemcpyHostToDevice));

    // Optimization: Use cudaMemsetAsync for initial zeroing of buffers.
    HANDLE_CUDA_ERROR(cudaMemsetAsync(d_new_centroids_sum, 0, centroids_size));
    HANDLE_CUDA_ERROR(cudaMemsetAsync(d_cluster_counts, 0, (size_t)num_cluster * sizeof(int)));

    // --- 3. K-Means Iteration Loop ---
    int threads_per_block = 256;
    int point_blocks = (num_points + threads_per_block - 1) / threads_per_block;
    int cluster_blocks = (num_cluster + threads_per_block - 1) / threads_per_block;

    if (verbose) std::cout << "Starting K-Means iterations..." << std::endl;

    int iter_to_converge = 0;
    for (int iter = 0; iter < max_num_iter; ++iter) {
        iter_to_converge = iter + 1;

        // Store current centroids in d_old_centroids to check for convergence
        HANDLE_CUDA_ERROR(cudaMemcpy(d_old_centroids, data.d_centroids, centroids_size, cudaMemcpyDeviceToDevice));

        // == Assignment Step ==
        // For each point, find the nearest centroid (launches one thread per point)
        assign_clusters_kernel<<<point_blocks, threads_per_block>>>(
            data.d_points, data.d_centroids, d_cluster_assignments, num_points, num_cluster, dims);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // == Update Step (Optimized) ==
        // 1. Sum up all the points for each cluster.
        update_centroids_sum_kernel<<<point_blocks, threads_per_block>>>(
            data.d_points, d_cluster_assignments, d_new_centroids_sum, d_cluster_counts, num_points, dims);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // 2. Divide sums by counts to get new centroids AND reset buffers for the next iteration.
        int calc_reset_blocks = (num_cluster * dims + threads_per_block - 1) / threads_per_block;
        calculate_and_reset_kernel<<<calc_reset_blocks, threads_per_block>>>(
            data.d_centroids, d_new_centroids_sum, d_cluster_counts, num_cluster, dims);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // == Convergence Check ==
        // 1. Initialize convergence flag on device to 1 (true)
        h_converged = 1;
        HANDLE_CUDA_ERROR(cudaMemcpy(d_converged, &h_converged, sizeof(int), cudaMemcpyHostToDevice));

        // 2. Launch kernel to check if any centroid moved more than the threshold
        check_convergence_kernel<<<cluster_blocks, threads_per_block>>>(
            data.d_centroids, d_old_centroids, d_converged, num_cluster, dims, threshold_sq);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // 3. Copy the flag back to the host
        HANDLE_CUDA_ERROR(cudaMemcpy(&h_converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost));

        // 4. If converged, break the loop
        if (h_converged) {
            if (verbose) std::cout << "Convergence reached after " << iter_to_converge << " iterations." << std::endl;
            break;
        }
    }

    // --- Stop Timer and Report ---
    HANDLE_CUDA_ERROR(cudaEventRecord(stop));
    HANDLE_CUDA_ERROR(cudaEventSynchronize(stop)); // Wait for all GPU work to finish
    float total_milliseconds = 0;
    HANDLE_CUDA_ERROR(cudaEventElapsedTime(&total_milliseconds, start, stop));

    if (iter_to_converge == max_num_iter && !h_converged) {
        if (verbose) std::cout << "Iters (" << max_num_iter << ") reached without convergence." << std::endl;
    }

    // Print the results in the requested format
    printf("%d,%lf\n", iter_to_converge, total_milliseconds / iter_to_converge);

    // --- 4. Copy Final Centroids from Device to Host ---
    HANDLE_CUDA_ERROR(cudaMemcpy(data.h_centroids, data.d_centroids, centroids_size, cudaMemcpyDeviceToHost));

    // Copy the final cluster assignments from the device back to the host
    HANDLE_CUDA_ERROR(cudaMemcpy(data.h_cluster_assignments, d_cluster_assignments, assignments_size, cudaMemcpyDeviceToHost));

    // --- 5. Free GPU Memory ---
    HANDLE_CUDA_ERROR(cudaFree(data.d_points));
    HANDLE_CUDA_ERROR(cudaFree(data.d_centroids));
    HANDLE_CUDA_ERROR(cudaFree(d_cluster_assignments));
    HANDLE_CUDA_ERROR(cudaFree(d_old_centroids));
    HANDLE_CUDA_ERROR(cudaFree(d_new_centroids_sum));
    HANDLE_CUDA_ERROR(cudaFree(d_cluster_counts));
    HANDLE_CUDA_ERROR(cudaFree(d_converged));
    HANDLE_CUDA_ERROR(cudaEventDestroy(start));
    HANDLE_CUDA_ERROR(cudaEventDestroy(stop));
    data.d_points = nullptr;
    data.d_centroids = nullptr;
}

void kmeans(int num_cluster, KmeansData& data, int max_num_iter, double threshold, bool output_centroids_flag, int seed, bool verbose, ExecutionMethod method) {
    // Use a switch to dispatch to the correct k-means implementation
    // based on the selected method.
    switch (method) {
        case SEQ:
            sequential_kmeans(num_cluster, data, max_num_iter, threshold, output_centroids_flag, seed, verbose);
            break;
        case CUDA:
           cuda_kmeans(num_cluster, data, max_num_iter, threshold, output_centroids_flag, verbose);
            break;
        case THRUST:
            thrust_kmeans(num_cluster, data, max_num_iter, threshold, output_centroids_flag, seed, verbose);
            break;
        case SMEMCUDA:
            smem_cuda_kmeans(num_cluster, data, max_num_iter, threshold, output_centroids_flag, verbose);
            break;
        case UNSPECIFIED:
            // This case should ideally not be reached due to argument parsing validation.
            fprintf(stderr, "Error: Execution method is unspecified.\n");
            break;
    }
}

int main(int argc, char* argv[]) {
    KMeansParams params;

    if (!parse_args(argc, argv, params)) {
        return 1; // Exit if argument parsing fails or help is requested
    }

    KmeansData data;
    data.dims = params.dims;

    // Read data from input file into host memory
    if (!read_points(params.inputfilename, data, params.verbose)) {
        return 1; // Exit if data reading fails
    }

    // Allocate host memory for final cluster assignments
    data.h_cluster_assignments = new int[data.num_points];

    // Initialize the centroids by randomly selecting points from the dataset
    initialize_centroids(data, params.num_cluster, params.seed);

    // Call the main k-means logic
    kmeans(params.num_cluster,
           data,
           params.max_num_iter,
           params.threshold,
           params.output_centroids_flag,
           params.seed,
           params.verbose,
           params.method);
    
    if (params.output_centroids_flag) {
        // If -c is specified, print the final centroids in the requested format
        for (int i = 0; i < data.num_centroids; ++i) {
            // Print cluster ID, followed by a space
            printf("%d ", i);
            for (int d = 0; d < data.dims; ++d) {
                // Print each coordinate, followed by a space.
                // Access pattern is row-major: h_centroids[cluster_id * dims + dim_index]
                printf("%lf ", data.h_centroids[i * data.dims + d]);
            }
            printf("\n");
        }
    } else {
        // If -c is NOT specified, print the cluster assignments for each point
        printf("clusters:");
        for (int i = 0; i < data.num_points; ++i) {
            printf(" %d", data.h_cluster_assignments[i]);
        }
        printf("\n");
    }

    // Free all host-side memory.
    delete[] data.h_points;
    delete[] data.h_centroids;
    delete[] data.h_cluster_assignments;

    return 0;
}