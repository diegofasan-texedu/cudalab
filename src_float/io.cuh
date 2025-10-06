#ifndef IO_CUH
#define IO_CUH

#include "dataset.cuh"

/**
 * @param data A reference to a KmeansData struct to be populated.
 * @param verbose A flag to enable verbose output during the reading process.
 * @return true if the file was read successfully, false otherwise.
 */
bool read_points(const char* filename, KmeansData& data, bool verbose);

#endif // IO_CUH