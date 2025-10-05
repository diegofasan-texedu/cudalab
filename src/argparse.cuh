#ifndef ARGPARSE_CUH
#define ARGPARSE_CUH

/**
 * @brief Enum to specify the k-means implementation method.
 */
enum ExecutionMethod {
    UNSPECIFIED, SEQ, CUDA, THRUST
};

/**
 * @brief A struct to hold the parameters for the k-means algorithm.
 */
struct KMeansParams {
    int num_cluster = 0;
    int dims = 0;
    const char* inputfilename = nullptr;
    int max_num_iter = 100;
    double threshold = 0.00000001;
    bool output_centroids_flag = false;
    int seed = 0;
    bool verbose = false;
    ExecutionMethod method = UNSPECIFIED;
};

/**
 * @brief Parses command-line arguments for the k-means algorithm.
 *
 * @param argc The argument count from main().
 * @param argv The argument vector from main().
 * @param params A reference to a KMeansParams struct to be populated.
 * @return true if parsing was successful and the program should continue.
 * @return false if there was an error or help was requested, and the program should exit.
 */
bool parse_args(int argc, char* argv[], KMeansParams& params);


#endif // ARGPARSE_CUH