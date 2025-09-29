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

    DataSet data;
    data.dims = params.dims;

    // Read data from input file into host memory
    if (!read_points(params.inputfilename, data, params.verbose)) {
        return 1; // Exit if data reading fails
    }

    // Call the main k-means logic
    kmeans(params.num_cluster, // Pass num_cluster
           data,               // Pass the DataSet object
           params.max_num_iter,
           params.threshold,
           params.output_centroids_flag,
           params.seed,
           params.verbose);

    // --- Manual Cleanup of DataSet ---
    delete[] data.h_points;
    data.h_points = nullptr;
    cudaFree(data.d_points); // Safe to call on nullptr
    data.d_points = nullptr;

    return 0;
}