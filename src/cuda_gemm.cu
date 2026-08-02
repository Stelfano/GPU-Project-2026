// cuda_gemm.cu
#include "../include/cuda_gemm.cuh"
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "Errore CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                            \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)
