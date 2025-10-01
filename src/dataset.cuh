#ifndef DATASET_CUH
#define DATASET_CUH

#include <cuda_runtime.h>

/**
 * @brief A struct to hold the dataset for k-means.
 *
 * This struct encapsulates the data points, number of points, and dimensions.
 * It manages the memory for the points array, deallocating it upon destruction.
 */
struct DataSet {
    int num_points = 0;
    int dims = 0;

    // Pointers to the matrix data (num_points x dims)
    double* h_points = nullptr; // on the host (CPU)
    double* d_points = nullptr; // on the device (GPU)

    // Print a sample of the dataset
    void print() const;
};

#endif // DATASET_CUH
