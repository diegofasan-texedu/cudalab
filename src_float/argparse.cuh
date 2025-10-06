#ifndef ARGPARSE_CUH
#define ARGPARSE_CUH

/**
 * @brief Enum to specify the k-means implementation method.
 */
enum ExecutionMethod {
    UNSPECIFIED, SEQ, CUDA, THRUST, SMEMCUDA
};

/**
 * @brief A struct to hold the parameters for the k-means algorithm.
 */
struct KMeansParams {
    int num_cluster = 0;
    int dims = 0;
    const char* inputfilename = nullptr;
    int max_num_iter = 100;
    float threshold = 0.0001f; 
    bool output_centroids_flag = false;
    int seed = 0;
    bool verbose = false;
    ExecutionMethod method = UNSPECIFIED;
};

bool parse_args(int argc, char* argv[], KMeansParams& params);

#endif // ARGPARSE_CUH