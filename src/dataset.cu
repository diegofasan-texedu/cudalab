#include "dataset.cuh"

#include <iostream>
#include <algorithm> // For std::min

/**
 * @brief Prints a sample of the dataset for verification.
 *
 * This function prints the number of points, dimensions, and the coordinates
 * of the first few points (up to 5) to allow for a quick check of the data.
 */
void DataSet::print() const {
    std::cout << "--- DataSet Sample ---" << std::endl;
    std::cout << "Points: " << num_points << ", Dimensions: " << dims << std::endl;

    if (h_points == nullptr) {
        std::cout << "Host data is not allocated." << std::endl;
        return;
    }

    // Print up to the first 5 points
    int points_to_print = std::min(5, num_points);
    for (int i = 0; i < points_to_print; ++i) {
        std::cout << "Point " << i << ": ";
        for (int j = 0; j < dims; ++j) {
            std::cout << h_points[i * dims + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "----------------------" << std::endl;
}