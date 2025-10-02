// Include headers for k-means implementations
#include "thrust_kmeans.cuh"

// Include headers for program setup
#include "argparse.cuh"
#include "io.cuh"
#include "dataset.cuh"
#include "kmeans_kernel.cuh" // For CUDA kernels
#include "error.cuh"         // For HANDLE_CUDA_ERROR


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

// void thrust_kmeans(int num_cluster, KmeansData& data, int max_num_iter, double threshold, bool output_centroids_flag, int seed, bool verbose) {
//     if (verbose) {
//         std::cout << "Executing Thrust K-Means..." << std::endl;
//     }

//     // The implementation for the Thrust k-means algorithm will go here.
//     // For now, it's a placeholder.
//     std::cout << "Thrust implementation is not yet complete." << std::endl;
// }

void cuda_kmeans(int num_cluster, KmeansData& data, int max_num_iter, float threshold, bool output_centroids_flag, bool verbose) {
    if (verbose) {
        std::cout << "Executing CUDA K-Means..." << std::endl;
    }

    // Extract data dimensions from the KmeansData struct for clarity
    const int num_points = data.num_points;
    const int dims = data.dims;

    // --- 1. Allocate GPU Memory ---
    size_t points_size = (size_t)num_points * dims * sizeof(float);
    size_t centroids_size = (size_t)num_cluster * dims * sizeof(float);
    size_t assignments_size = (size_t)num_points * sizeof(int);
    size_t counts_size = (size_t)num_cluster * sizeof(int);

    // Device pointers
    float* d_centroid_sums;
    int* d_cluster_assignments;
    int* d_cluster_counts;

    // --- Additional Memory for Two-Pass Reduction (to avoid atomics) ---
    int point_blocks = (num_points + 256 - 1) / 256;
    float* d_partial_centroid_sums;
    int* d_partial_cluster_counts;

    // --- Additional Memory for Convergence Check ---
    float* d_old_centroids;
    int* d_converged;
    int h_converged = 0;
    const double threshold_sq = threshold * threshold;

    // Allocate memory
    HANDLE_CUDA_ERROR(cudaMalloc(&data.d_points, points_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&data.d_centroids, centroids_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_cluster_assignments, assignments_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_centroid_sums, centroids_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_cluster_counts, counts_size));

    HANDLE_CUDA_ERROR(cudaMalloc(&d_partial_centroid_sums, centroids_size * point_blocks));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_partial_cluster_counts, counts_size * point_blocks));

    HANDLE_CUDA_ERROR(cudaMalloc(&d_old_centroids, centroids_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_converged, sizeof(int)));

    // --- 2. Copy Initial Data from Host to Device ---
    HANDLE_CUDA_ERROR(cudaMemcpy(data.d_points, data.h_points, points_size, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(data.d_centroids, data.h_centroids, centroids_size, cudaMemcpyHostToDevice));

    // --- 3. K-Means Iteration Loop ---
    int threads_per_block = 256;
    int cluster_blocks = (num_cluster + threads_per_block - 1) / threads_per_block;
    int last_iter = 0;

    for (int iter = 0; iter < max_num_iter; ++iter) {
        last_iter = iter;
        h_converged = 1; // Assume convergence until proven otherwise

        // Store current centroids to check for convergence later
        HANDLE_CUDA_ERROR(cudaMemcpy(d_old_centroids, data.d_centroids, centroids_size, cudaMemcpyDeviceToDevice));

        // -- Assignment Step --
        assign_clusters_kernel<<<point_blocks, threads_per_block>>>(data.d_points, data.d_centroids, d_cluster_assignments, num_points, num_cluster, dims);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // -- Update Step --
        HANDLE_CUDA_ERROR(cudaMemset(d_centroid_sums, 0, centroids_size));
        HANDLE_CUDA_ERROR(cudaMemset(d_cluster_counts, 0, counts_size));

        // -- Update Step (Two-Pass Reduction) --
        // Pass 1: Each block computes partial sums into its own output slot.
        size_t shared_mem_size = (num_cluster * dims * sizeof(float)) + (num_cluster * sizeof(int));
        sum_points_for_clusters_kernel<<<point_blocks, threads_per_block, shared_mem_size>>>(
            data.d_points, d_cluster_assignments, d_partial_centroid_sums, d_partial_cluster_counts, num_points, num_cluster, dims);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // Pass 2: Reduce all partial sums into the final sum arrays.
        int reduce_blocks = (num_cluster * dims + threads_per_block - 1) / threads_per_block;
        reduce_partial_sums_kernel<<<reduce_blocks, threads_per_block>>>(
            d_partial_centroid_sums, d_partial_cluster_counts, d_centroid_sums, d_cluster_counts, num_cluster, dims, point_blocks);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        average_clusters_kernel<<<cluster_blocks, threads_per_block>>>(data.d_centroids, d_centroid_sums, d_cluster_counts, num_cluster, dims);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // -- Convergence Check --
        // Initialize convergence flag on device to 1 (true)
        HANDLE_CUDA_ERROR(cudaMemcpy(d_converged, &h_converged, sizeof(int), cudaMemcpyHostToDevice));

        // Launch kernel to check if any centroid moved more than the threshold
        check_convergence_kernel<<<cluster_blocks, threads_per_block>>>(
            d_old_centroids, data.d_centroids, d_converged, num_cluster, dims, threshold_sq);
        HANDLE_CUDA_ERROR(cudaGetLastError());

        // Copy the convergence flag back to the host
        HANDLE_CUDA_ERROR(cudaMemcpy(&h_converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

        if (h_converged) {
            if (verbose) std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }
    if (!h_converged && verbose) std::cout << "Reached max iterations (" << max_num_iter << ") without converging." << std::endl;

    // --- 4. Copy Final Centroids from Device to Host ---
    HANDLE_CUDA_ERROR(cudaMemcpy(data.h_centroids, data.d_centroids, centroids_size, cudaMemcpyDeviceToHost));

    // --- 5. Free GPU Memory ---
    HANDLE_CUDA_ERROR(cudaFree(data.d_points));
    HANDLE_CUDA_ERROR(cudaFree(data.d_centroids));
    HANDLE_CUDA_ERROR(cudaFree(d_cluster_assignments));
    HANDLE_CUDA_ERROR(cudaFree(d_centroid_sums));
    HANDLE_CUDA_ERROR(cudaFree(d_cluster_counts));
    HANDLE_CUDA_ERROR(cudaFree(d_partial_centroid_sums));
    HANDLE_CUDA_ERROR(cudaFree(d_partial_cluster_counts));
    HANDLE_CUDA_ERROR(cudaFree(d_old_centroids));
    HANDLE_CUDA_ERROR(cudaFree(d_converged));

    // Set device pointers to null to avoid double free issues
    data.d_points = nullptr;
    data.d_centroids = nullptr;
}

void kmeans(int num_cluster, KmeansData& data, int max_num_iter, float threshold, bool output_centroids_flag, int seed, bool verbose, ExecutionMethod method) {
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

    // Print a sample of the loaded data if in verbose mode
    if (params.verbose) data.print_points();

    // Initialize the centroids by randomly selecting points from the dataset
    initialize_centroids(data, params.num_cluster, params.seed);

    // Print a sample of the initial centroids if in verbose mode
    if (params.verbose) data.print_centroids();

    // Call the main k-means logic
    kmeans(params.num_cluster,
           data,               // Pass the KmeansData object
           params.max_num_iter,
           params.threshold,
           params.output_centroids_flag,
           params.seed,
           params.verbose,
           params.method);

    // Free all host-side memory.
    delete[] data.h_points;
    delete[] data.h_centroids;

    return 0;
}