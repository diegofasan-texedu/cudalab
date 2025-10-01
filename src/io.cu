#include "io.cuh"

#include <iostream>
#include <fstream>
#include <vector>

bool read_points(const char* filename, KmeansData& data, bool verbose) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open input file " << filename << std::endl;
        return false;
    }

    // Read the number of points from the first line
    infile >> data.num_points;
    if (verbose) {
        std::cout << "Reading " << data.num_points << " points with " << data.dims << " dimensions." << std::endl;
    }

    // Allocate memory for the points
    // Data is stored in a 1D array (row-major order)
    data.h_points = new double[data.num_points * data.dims];

    int point_id;
    for (int i = 0; i < data.num_points; ++i) {
        // Read and discard the point ID
        infile >> point_id;

        for (int j = 0; j < data.dims; ++j) {
            infile >> data.h_points[i * data.dims + j];
        }

        // Handle potential read errors or end-of-file
        if (infile.fail()) {
            std::cerr << "Error: Failed to read data for point " << i + 1 << std::endl;
            infile.close();
            return false;
        }
    }

    infile.close();

    std::cout << "Successfully read " << data.num_points << " points from " << filename << std::endl;
    return true;
}