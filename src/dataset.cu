#include "dataset.cuh"

#include <iostream>
#include <algorithm> // For std::min
#include <string.h>  // For memcpy

// --- Simple LCG Random Number Generator ---
static unsigned long int next_rand = 1;
static unsigned long kmeans_rmax = 32767;

int kmeans_rand() {
    next_rand = next_rand * 1103515245 + 12345;
    return (unsigned int)(next_rand / 65536) % (kmeans_rmax + 1);
}

void kmeans_srand(unsigned int seed) {
    next_rand = seed;
}
// -----------------------------------------

void KmeansData::print_points() const {
    std::cout << "--- DataSet Sample ---" << std::endl;
    std::cout << "Points: " << num_points << ", Dimensions: " << dims << std::endl;

    if (h_points == nullptr) {
        std::cout << "Host data is not allocated." << std::endl;
    } else {
        int points_to_print = std::min(5, num_points);
        for (int i = 0; i < points_to_print; ++i) {
            std::cout << "Point " << i << ": ";
            for (int j = 0; j < dims; ++j) {
                std::cout << h_points[i * dims + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "----------------------" << std::endl;
}

void KmeansData::print_centroids() const {
    std::cout << "--- Centroids Sample ---" << std::endl;
    std::cout << "Centroids: " << num_centroids << ", Dimensions: " << dims << std::endl;

    if (h_centroids == nullptr) {
        std::cout << "Host data is not allocated." << std::endl;
    } else {
        int centroids_to_print = std::min(5, num_centroids);
        for (int i = 0; i < centroids_to_print; ++i) {
            std::cout << "Centroid " << i << ": ";
            for (int j = 0; j < dims; ++j) {
                std::cout << h_centroids[i * dims + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "------------------------" << std::endl;
}

void initialize_centroids(KmeansData& data, int num_centroids, unsigned int seed) {
    // Set centroid properties
    data.num_centroids = num_centroids;

    // Allocate memory for host-side centroids
    size_t centroids_size = (size_t)num_centroids * data.dims * sizeof(double);
    data.h_centroids = new double[num_centroids * data.dims];

    // Seed the random number generator
    kmeans_srand(seed);

    // Randomly select k points from the dataset to be the initial centroids
    for (int i = 0; i < num_centroids; i++) {
        // Generate a random index in the range [0, num_points-1]
        int point_index = kmeans_rand() % data.num_points;

        // Use a pointer to refer to the destination centroid's location
        double* dest_centroid = &data.h_centroids[i * data.dims];

        // Use a pointer to refer to the source point's location
        double* src_point = &data.h_points[point_index * data.dims];

        // Copy the point data to the centroid location
        memcpy(dest_centroid, src_point, data.dims * sizeof(double));
    }
}