#ifndef IO_CUH
#define IO_CUH

#include "dataset.cuh"

/**
 * @brief Reads data points from a file into host memory.
 *
 * The file is expected to have the number of points on the first line.
 * Each subsequent line should start with a point ID (which is ignored),
 * followed by the coordinates for that point.
 *
 * @param filename The path to the input file.
 * @param data A reference to a DataSet struct to be populated.
 * @param verbose A flag to enable verbose output during the reading process.
 * @return true if the file was read successfully, false otherwise.
 */
bool read_points(const char* filename, DataSet& data, bool verbose);

#endif // IO_CUH