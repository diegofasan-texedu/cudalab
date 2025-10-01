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

    // Print a sample of the loaded data if in verbose mode
    if (params.verbose) data.print();

    // Call the main k-means logic
    kmeans(params.num_cluster, // Pass num_cluster
           data,               // Pass the DataSet object
           params.max_num_iter,
           params.threshold,
           params.output_centroids_flag,
           params.seed,
           params.verbose);

    // --- Manual Cleanup ---
    // Since the DataSet destructor is removed, we must free memory here.
    delete[] data.h_points;
    // cudaFree(data.d_points); // Device memory not yet allocated.

    return 0;
}