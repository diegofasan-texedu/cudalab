#include "dataset.cuh"

#include <iostream>
#include <vector>
#include <random>
#include <algorithm> // For std::min, std::shuffle
#include <iomanip> // For std::fixed and std::setprecision

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
    std::cout << "Displaying a sample of points (" << num_points << " total points, " << dims << " dimensions):\n";

    if (h_points == nullptr) {
        std::cout << "Host data is not allocated." << std::endl;
        return;
    }

    int limit = std::min(5, num_points);
    for (int i = 0; i < limit; ++i) {
        std::cout << "  Point " << i << ": [";
        for (int d = 0; d < dims; ++d) {
            std::cout << std::fixed << std::setprecision(4) << h_points[i * dims + d] << (d == dims - 1 ? "" : ", ");
        }
        std::cout << "]\n";
    }
    if (num_points > limit) {
        std::cout << "  ...\n";
    }
}

void KmeansData::print_centroids() const {
    std::cout << "Displaying all " << num_centroids << " centroids (" << dims << " dimensions):\n";

    if (h_centroids == nullptr) {
        std::cout << "Host data is not allocated." << std::endl;
        return;
    }

    for (int i = 0; i < num_centroids; ++i) {
        std::cout << "  Centroid " << i << ": [";
        for (int d = 0; d < dims; ++d) {
            // Accessing the centroid data: h_centroids[centroid_index * num_dimensions + dimension_index]
            std::cout << std::fixed << std::setprecision(4) << h_centroids[i * dims + d] << (d == dims - 1 ? "" : ", ");
        }
        std::cout << "]\n";
    }
}

void initialize_centroids(KmeansData& data, int num_centroids, unsigned int seed) {
    // Set centroid properties
    data.num_centroids = num_centroids;

    // Allocate memory for host-side centroids
    // size_t centroids_size = (size_t)num_centroids * data.dims * sizeof(double);
    data.h_centroids = new float[num_centroids * data.dims];

    // Seed the random number generator
    kmeans_srand(seed);
    std::vector<int> indices(data.num_points);
    // std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...

    // Randomly select k points from the dataset to be the initial centroids
    for (int i = 0; i < num_centroids; i++) {
        // Generate a random index in the range [0, num_points-1]
        int point_index = kmeans_rand() % data.num_points;
    // Shuffle indices to select unique random points
    // std::mt19937 gen(seed);
    // std::shuffle(indices.begin(), indices.end(), gen);

        // Use a pointer to refer to the destination centroid's location
        float* dest_centroid = &data.h_centroids[i * data.dims];

        // Use a pointer to refer to the source point's location
        float* src_point = &data.h_points[point_index * data.dims];

        // Copy the point data to the centroid location
        memcpy(dest_centroid, src_point, data.dims * sizeof(float)); // Requires <string.h>
    // Copy the first 'num_centroids' random points to be the initial centroids
    // for (int i = 0; i < num_centroids; ++i) {
    //     int point_idx = indices[i];
    //     for (int d = 0; d < data.dims; ++d) {
    //         data.h_centroids[i * data.dims + d] = data.h_points[point_idx * data.dims + d];
    //     }
    // }
    }
}