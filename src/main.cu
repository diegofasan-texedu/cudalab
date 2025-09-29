#include <iostream>

#include "kmeans.cuh"
#include "argparse.cuh"

int main(int argc, char* argv[]) {
    KMeansParams params;

    if (!parse_args(argc, argv, params)) {
        return 1; // Exit if argument parsing fails
    }

    // Call the main k-means logic
    kmeans(params.num_cluster,
           params.dims,
           params.inputfilename,
           params.max_num_iter,
           params.threshold,
           params.output_centroids_flag,
           params.seed);

    return 0;
}