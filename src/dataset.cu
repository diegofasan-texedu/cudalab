#include "dataset.cuh"

#include <iostream>
#include <vector>
#include <random>
#include <algorithm> // For std::min, std::shuffle
#include <iomanip> // For std::fixed and std::setprecision

// --- Simple LCG Random Number Generator ---
static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;
int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}
void kmeans_srand(unsigned int seed) {
    next = seed;
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
            std::cout << std::fixed << std::setprecision(12) << h_points[i * dims + d] << (d == dims - 1 ? "" : ", ");
        }
        std::cout << "]\n";
    }
    if (num_points > limit) {
        std::cout << "  ...\n";
    }
}

void KmeansData::print_centroids(int precision) const {
    std::cout << "Displaying all " << num_centroids << " centroids (" << dims << " dimensions):\n";

    if (h_centroids == nullptr) {
        std::cout << "Host data is not allocated." << std::endl;
        return;
    }
    
    for (int i = 0; i < num_centroids; ++i) {
        std::cout << "  Centroid " << i << ": [";
        for (int d = 0; d < dims; ++d) {
            // Accessing the centroid data: h_centroids[centroid_index * num_dimensions + dimension_index]
            std::cout << std::fixed << std::setprecision(precision) << h_centroids[i * dims + d];
            if (d < dims - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

void initialize_centroids(KmeansData& data, int num_centroids, unsigned int seed) {
    // Set centroid properties
    data.num_centroids = num_centroids;

    // Allocate memory for host-side centroids
    data.h_centroids = new double[num_centroids * data.dims];

    // Seed the random number generator
    kmeans_srand(seed);

    // For each centroid, pick a random point from the dataset and copy its data.
    // This method does not guarantee that the selected points will be unique.
    for (int i = 0; i < num_centroids; i++) {
        int point_index = kmeans_rand() % data.num_points;

        // Copy the point data to the centroid location
        memcpy(&data.h_centroids[i * data.dims],
               &data.h_points[point_index * data.dims],
               data.dims * sizeof(double));
    }
}
