#ifndef THRUST_KMEANS_CUH
#define THRUST_KMEANS_CUH

// --- Thrust Includes ---
// Core library
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
// Algorithms
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/extrema.h> // For max_element
#include <thrust/tuple.h>

#include "dataset.cuh"

void thrust_kmeans(int num_cluster, KmeansData& data, int max_num_iter, double threshold, bool output_centroids_flag, int seed, bool verbose);

#endif // THRUST_KMEANS_CUH