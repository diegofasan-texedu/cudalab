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

#include <stdio.h>
#include <iostream>

void sequential_kmeans(int num_cluster, KmeansData& data, int max_num_iter, double threshold, bool output_centroids_flag, int seed, bool verbose) {
    if (verbose) {
        std::cout << "Executing Sequential K-Means..." << std::endl;
    }

    // The implementation for the sequential k-means algorithm will go here.
    // For now, it's a placeholder.
    std::cout << "Sequential implementation is not yet complete." << std::endl;
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

        // == Update Step ==
        // 1. Reset the summation and count buffers to zero.
        reset_update_buffers_kernel<<<cluster_blocks, threads_per_block>>>(
            d_new_centroids_sum, d_cluster_counts, num_cluster, dims);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // 2. Sum up all the points for each cluster.
        update_centroids_sum_kernel<<<point_blocks, threads_per_block>>>(
            data.d_points, d_cluster_assignments, d_new_centroids_sum, d_cluster_counts, num_points, dims);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // 3. Divide sums by counts to get the new centroids.
        calculate_new_centroids_kernel<<<cluster_blocks, threads_per_block>>>(
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
        if (verbose) std::cout << "Max iterations (" << max_num_iter << ") reached without convergence." << std::endl;
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

    // Set device pointers to null to avoid double free issues
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

        // Calculate precision from threshold. e.g., 0.001 -> 3, 0.000001 -> 6
        // A double has about 15-17 decimal digits of precision, so we cap it.
        int precision = 8; // Default precision
        if (params.threshold > 0) {
            precision = std::min(15, static_cast<int>(ceil(-log10(params.threshold))));
        }

        for (int i = 0; i < data.num_centroids; ++i) {
            printf("%d", i);
            for (int d = 0; d < data.dims; ++d) {
                // Use dynamic precision in the format string, e.g., " %.6lf"
                printf(" %.*lf", precision, data.h_centroids[i * data.dims + d]);
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