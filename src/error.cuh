#ifndef ERROR_CUH
#define ERROR_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro to wrap CUDA calls for error checking
#define HANDLE_CUDA_ERROR(err) (HandleCudaError(err, __FILE__, __LINE__))

inline void HandleCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#endif // ERROR_CUH