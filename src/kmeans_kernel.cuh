/**
 * @brief Assigns each data point to the nearest cluster centroid.
 *
 * This kernel is launched with one thread per data point. Each thread calculates
 * the distance from its assigned point to every centroid and writes the index
 * of the closest centroid to the `cluster_assignments` array.
 *
 * @param points Device pointer to the input data points.
 * @param centroids Device pointer to the current centroids.
 * @param cluster_assignments Device pointer to an array where the assigned cluster for each point will be stored.
 * @param num_points The total number of data points.
 * @param num_clusters The number of clusters (k).
 * @param dims The number of dimensions for each point and centroid.
 */
__global__ void assign_clusters_kernel(const double* points, const double* centroids, int* cluster_assignments, int num_points, int num_clusters, int dims);

/**
 * @brief Resets the centroid sum and count buffers to zero.
 *
 * This kernel is launched with one thread per centroid. It initializes the
 * summation buffer and the cluster count buffer before the main summation happens.
 *
 * @param d_new_centroids_sum Device pointer to the buffer storing the sum of coordinates for each cluster.
 * @param d_cluster_counts Device pointer to the buffer storing the number of points in each cluster.
 * @param num_clusters The number of clusters (k).
 * @param dims The number of dimensions for each point.
 */
__global__ void reset_update_buffers_kernel(double* d_new_centroids_sum, int* d_cluster_counts, int num_clusters, int dims);

/**
 * @brief Atomically adds each point's coordinates to its assigned cluster's sum.
 *
 * This kernel is launched with one thread per data point. Each thread reads its
 * assigned cluster and adds its coordinate data to the corresponding location
 * in `d_new_centroids_sum`. It also increments the count for that cluster.
 *
 * @param d_points Device pointer to the input data points.
 * @param d_cluster_assignments Device pointer to the cluster assignment for each point.
 * @param d_new_centroids_sum Device pointer to the buffer for summing coordinates.
 * @param d_cluster_counts Device pointer to the buffer for counting points per cluster.
 * @param num_points The total number of data points.
 * @param dims The number of dimensions.
 */
__global__ void update_centroids_sum_kernel(const double* d_points, const int* d_cluster_assignments, double* d_new_centroids_sum, int* d_cluster_counts, int num_points, int dims);

/**
 * @brief Calculates the new centroids by dividing the sums by the counts.
 *
 * This kernel is launched with one thread per centroid. It computes the average
 * for each cluster and writes the result to the main `d_centroids` array.
 */
__global__ void calculate_new_centroids_kernel(double* d_centroids, const double* d_new_centroids_sum, const int* d_cluster_counts, int num_clusters, int dims);
