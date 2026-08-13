// cuda_gemm.cuh
#pragma once
#include <cuda_bf16.h>
#include <cstdio>
#include "common.cuh"

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "Errore CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                            \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)


// Esegue C = A * B su GPU con un kernel naive (un thread per elemento di C,
// nessuna shared memory, nessun register blocking — è deliberatamente il
// punto di partenza più semplice possibile). Misura il tempo di solo kernel
// con CUDA events, mediato su n_reps ripetizioni dopo un run di warm-up non
// cronometrato. A, B, C sono puntatori host; la funzione gestisce da sola
// malloc/copy/free su device. Ritorna i millisecondi medi per iterazione.

template <typename T>
__global__ void gemm_naive_kernel(const T* __restrict__ A,
                                  const T* __restrict__ B,
                                  T* __restrict__ C, 
                                  int M, int N, int K, int Bsize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < M && col < N && batch < Bsize) {
        T acc = static_cast<T>(0.0f);
        
        for (int k = 0; k < K; k++) {
            acc += A[static_cast<size_t>(row) * K + k + batch*(M*K)] * B[static_cast<size_t>(k) * N + col + batch*(K*N)];
        }

        C[static_cast<size_t>(row) * N + col + batch*(M*N)] = acc;
    }
}


template <typename T>
__global__ void naiveReLU(T *C, int M, int N, int Bsize){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < M && col < N && batch < Bsize) {
        int pos = static_cast<size_t>(row) * N + col + batch*(M*N);
        C[pos] = ReLU(C[pos]);
    }
}

template <typename T>
double gemm_cuda_timed(const T* h_A, const T* h_B, T* h_C,
                       int M, int N, int K, int Bsize, int n_reps) {
    
    // Dimensione automatica basata sul tipo T passata alla funzione
    size_t bytesA = static_cast<size_t>(M) * K * Bsize * sizeof(T);
    size_t bytesB = static_cast<size_t>(K) * N * Bsize * sizeof(T);
    size_t bytesC = static_cast<size_t>(M) * N * Bsize * sizeof(T);

    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16, 1);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y,
                 (Bsize + blockDim.z - 1) / blockDim.z);

    // Warm-up: specifichiamo <T> al kernel
    gemm_naive_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < n_reps; ++r) {
        gemm_naive_kernel<T><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
        //naiveReLU<T><<<gridDim, blockDim>>>(d_C, M, N, Bsize);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
    double ms_avg = static_cast<double>(ms_total) / n_reps;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return ms_avg;
}

