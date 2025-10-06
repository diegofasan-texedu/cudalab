// Cuda Method kernels
__global__ void assign_clusters_kernel(const float* points, const float* centroids, int* cluster_assignments, int num_points, int num_clusters, int dims);

__global__ void calculate_and_reset_kernel(float* d_centroids, float* d_new_centroids_sum, int* d_cluster_counts, int num_clusters, int dims);

__global__ void check_convergence_kernel(const float* d_centroids, const float* d_old_centroids, int* d_converged, int num_clusters, int dims, float threshold_sq);

__global__ void update_centroids_sum_kernel(const float* d_points, const int* d_cluster_assignments, float* d_new_centroids_sum, int* d_cluster_counts, int num_points, int dims);

// shared memory cuda kernels
__global__ void assign_clusters_smem_kernel(const float* points, const float* centroids, int* cluster_assignments, int num_points, int num_clusters, int dims);

__global__ void update_centroids_sum_smem_kernel(const float* d_points, const int* d_cluster_assignments, float* d_new_centroids_sum, int* d_cluster_counts, int num_points, int num_clusters, int dims);

__global__ void calculate_and_reset_smem_kernel(float* d_centroids, float* d_new_centroids_sum, int* d_cluster_counts, int num_clusters, int dims);