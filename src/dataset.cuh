#ifndef DATASET_CUH
#define DATASET_CUH

#include <cuda_runtime.h>

/**
 * @brief A struct to hold all data related to the k-means algorithm.
 */
struct KmeansData {
    // Point data
    int num_points = 0;
    int dims = 0;
    double* h_points = nullptr; // on the host (CPU)
    double* d_points = nullptr; // on the device (GPU)

    // Centroid data
    int num_centroids = 0;
    double* h_centroids = nullptr; // on the host (CPU)
    double* d_centroids = nullptr; // on the device (GPU)

    // Print a sample of the points
    void print_points() const;

    // Print a sample of the centroids
    void print_centroids(int precision = 12) const;
};

/**
 * @param data The KmeansData struct to be initialized.
 * @param num_centroids The number of centroids to initialize (k).
 * @param seed The seed for the random number generator.
 */
void initialize_centroids(KmeansData& data, int num_centroids, unsigned int seed);

#endif // DATASET_CUH
