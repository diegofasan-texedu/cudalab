#include "argparse.cuh"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // For getopt()

bool parse_args(int argc, char* argv[], KMeansParams& params) {
    // If no arguments are provided, print usage and exit.
    if (argc == 1) {
        fprintf(stderr, "No arguments provided. Please specify the required parameters.\n");
        fprintf(stderr, "Usage: %s -k num_cluster -d dims -i inputfile [-m max_iter] [-t threshold] [-c] [-s seed] [-v]\n", argv[0]);
        return false;
    }

    int opt;
    while ((opt = getopt(argc, argv, "k:d:i:m:t:cs:v")) != -1) {
        switch (opt) {
            case 'k':
                params.num_cluster = atoi(optarg);
                break;
            case 'd':
                params.dims = atoi(optarg);
                break;
            case 'i':
                params.inputfilename = optarg;
                break;
            case 'm':
                params.max_num_iter = atoi(optarg);
                break;
            case 't':
                params.threshold = atof(optarg);
                break;
            case 'c':
                params.output_centroids_flag = true;
                break;
            case 's':
                params.seed = atoi(optarg);
                break;
            case 'v':
                params.verbose = true;
                break;
            default: /* '?' */
                fprintf(stderr, "Usage: %s -k num_cluster -d dims -i inputfile [-m max_iter] [-t threshold] [-c] [-s seed] [-v]\n", argv[0]);
                return false;
        }
    }

    // If verbose flag is set, print parsed arguments to verify
    if (params.verbose) {
        std::cout << "--- Parsed Arguments ---" << std::endl;
        std::cout << "Num Clusters (-k): " << params.num_cluster << std::endl;
        std::cout << "Dimensions (-d): " << params.dims << std::endl;
        std::cout << "Input Filename (-i): " << (params.inputfilename ? params.inputfilename : "Not specified") << std::endl;
        std::cout << "Max Iterations (-m): " << params.max_num_iter << std::endl;
        std::cout << "Threshold (-t): " << params.threshold << std::endl;
        std::cout << "Output Centroids Flag (-c): " << (params.output_centroids_flag ? "true" : "false") << std::endl;
        std::cout << "Seed (-s): " << params.seed << std::endl;
        std::cout << "------------------------" << std::endl;
    }

    return true;
}