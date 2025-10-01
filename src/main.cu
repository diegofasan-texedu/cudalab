#include <iostream>

#include "kmeans.cuh"
#include "argparse.cuh"
#include "io.cuh"
#include "dataset.cuh"

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

    // --- Manual Cleanup ---
    // Free all host-side memory.
    delete[] data.h_points;
    delete[] data.h_centroids;
    // cudaFree(data.d_points); // Device memory not yet allocated.

    return 0;
}