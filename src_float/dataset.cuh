#ifndef DATASET_CUH
#define DATASET_CUH

#include <cuda_runtime.h>

struct KmeansData {
    // Point data
    int num_points = 0;
    int dims = 0;
    float* h_points = nullptr; // on the host (CPU)
    float* d_points = nullptr; // on the device (GPU)

    // Centroid data
    int num_centroids = 0;
    float* h_centroids = nullptr; // on the host (CPU)
    float* d_centroids = nullptr; // on the device (GPU)

    // Cluster assignments for each point
    int* h_cluster_assignments = nullptr; // on the host (CPU)

    void print_points() const;
    void print_centroids(int precision = 6) const;
};

void initialize_centroids(KmeansData& data, int num_centroids, unsigned int seed);

#endif // DATASET_CUH