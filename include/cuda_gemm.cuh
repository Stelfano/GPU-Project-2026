// cuda_gemm.cuh
#pragma once
#include <cuda_bf16.h>
#include <cstdio>
#include "common.cuh"
#include <mma.h>

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
        if(C[pos] < (T)0)
            C[pos] = 0; 
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
        naiveReLU<T><<<gridDim, blockDim>>>(d_C, M, N, Bsize);
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

template <typename T, int BM, int BN, int BK, int TM>
__global__ void gemm_tiled_kernel(const T* __restrict__ A,
                                   const T* __restrict__ B,
                                   T* __restrict__ C,
                                   int M, int N, int K, int Bsize) {
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int batch     = blockIdx.z;

    __shared__ T As[BM * BK];
    __shared__ T Bs[BK * BN];

    // Ogni thread possiede una colonna fissa (thread_col) e TM righe
    // consecutive a partire da thread_row*TM: e' qui che si decide il
    // register blocking.
    const int thread_col = threadIdx.x % BN;
    const int thread_row = threadIdx.x / BN;   // 0 .. (BM/TM - 1)

    const T* A_batch = A + static_cast<size_t>(batch) * M * K;
    const T* B_batch = B + static_cast<size_t>(batch) * K * N;
    T*       C_batch = C + static_cast<size_t>(batch) * M * N;

    const T* A_tile = A_batch + static_cast<size_t>(block_row) * K;
    const T* B_tile = B_batch + block_col;
    T*       C_tile = C_batch + static_cast<size_t>(block_row) * N + block_col;

    // Caricamento cooperativo: ogni thread porta in shared memory
    // esattamente un elemento di As e uno di Bs per iterazione
    // (BM*BK e BK*BN sono entrambi multipli del numero di thread/blocco).
    const int inner_row_a = threadIdx.x / BK;
    const int inner_col_a = threadIdx.x % BK;
    const int inner_row_b = threadIdx.x / BN;
    const int inner_col_b = threadIdx.x % BN;

    T acc[TM];
    #pragma unroll
    for (int i = 0; i < TM; ++i) acc[i] = static_cast<T>(0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        As[inner_row_a * BK + inner_col_a] = A_tile[static_cast<size_t>(inner_row_a) * K + inner_col_a];
        Bs[inner_row_b * BN + inner_col_b] = B_tile[static_cast<size_t>(inner_row_b) * N + inner_col_b];
        __syncthreads();

        A_tile += BK;
        B_tile += static_cast<size_t>(BK) * N;

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            T b_val = Bs[k * BN + thread_col];   // un solo accesso a shared...
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                T a_val = As[(thread_row * TM + i) * BK + k];
                acc[i] += a_val * b_val;          // ...riusato per TM MAC
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        C_tile[static_cast<size_t>(thread_row * TM + i) * N + thread_col] = acc[i];
    }
}

template <typename T>
double gemm_tiled_timed(const T* h_A, const T* h_B, T* h_C,
                         int M, int N, int K, int Bsize, int n_reps = 10) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;

    // Versione semplice: nessuna gestione dei bordi. M/N/K devono essere
    // multipli di BM/BN/BK -- tutte le shape della consegna attuale lo sono.
    if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
        fprintf(stderr, "gemm_tiled_timed: M/N/K devono essere multipli di %d/%d/%d\n", BM, BN, BK);
        exit(EXIT_FAILURE);
    }

    size_t bytesA = static_cast<size_t>(M) * K * Bsize * sizeof(T);
    size_t bytesB = static_cast<size_t>(K) * N * Bsize * sizeof(T);
    size_t bytesC = static_cast<size_t>(M) * N * Bsize * sizeof(T);

    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 blockDim((BM * BN) / TM);          // 512 thread, 1D
    dim3 gridDim(N / BN, M / BM, Bsize);

    gemm_tiled_kernel<T, BM, BN, BK, TM><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < n_reps; ++r) {
        gemm_tiled_kernel<T, BM, BN, BK, TM><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
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


template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_tiled_kernel_2D(const T* __restrict__ A,
                                   const T* __restrict__ B,
                                   T* __restrict__ C,
                                   int M, int N, int K, int Bsize) {
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int batch     = blockIdx.z;

    __shared__ T As[BM * BK];
    __shared__ T Bs[BK * BN];

    // Ogni thread possiede una colonna fissa (thread_col) e TM righe
    // consecutive a partire da thread_row*TM: e' qui che si decide il
    // register blocking.
    const int thread_col = threadIdx.x % BN;
    const int thread_row = threadIdx.x / BN;   // 0 .. (BM/TM - 1)

    const T* A_batch = A + static_cast<size_t>(batch) * M * K;
    const T* B_batch = B + static_cast<size_t>(batch) * K * N;
    T*       C_batch = C + static_cast<size_t>(batch) * M * N;

    const T* A_tile = A_batch + static_cast<size_t>(block_row) * K;
    const T* B_tile = B_batch + block_col;
    T*       C_tile = C_batch + static_cast<size_t>(block_row) * N + block_col;

    // Caricamento cooperativo: ogni thread porta in shared memory
    // esattamente un elemento di As e uno di Bs per iterazione
    // (BM*BK e BK*BN sono entrambi multipli del numero di thread/blocco).
    const int inner_row_a = threadIdx.x / BK;
    const int inner_col_a = threadIdx.x % BK;
    const int inner_row_b = threadIdx.x / BN;
    const int inner_col_b = threadIdx.x % BN;

    T acc[TM];
    #pragma unroll
    for (int i = 0; i < TM; ++i) acc[i] = static_cast<T>(0.0f);

    T a_val[TM] = {0.0};
    T b_val[TN] = {0.0};

    for (int k0 = 0; k0 < K; k0 += BK) {
        As[inner_row_a * BK + inner_col_a] = A_tile[static_cast<size_t>(inner_row_a) * K + inner_col_a];
        Bs[inner_row_b * BN + inner_col_b] = B_tile[static_cast<size_t>(inner_row_b) * N + inner_col_b];
        __syncthreads();

        A_tile += BK;
        B_tile += static_cast<size_t>(BK) * N;

        T thread_results[TM * TN] = {0.0};

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_val[i] = As[(thread_row * TM + i) * BK + k];
            }
            for(int i=0;i<TN;++i){
                b_val[i] = Bs[k * BN + thread_col * TN + i];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        for(int j = 0; j < TN; j++)
            C_tile[static_cast<size_t>(i * TM + j) * N + thread_col] = a_val[i] * b_val[j];
    }
}


template <typename T>
double gemm_tiled_timed_2D(const T* h_A, const T* h_B, T* h_C,
                         int M, int N, int K, int Bsize, int n_reps = 10) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8;

    // Versione semplice: nessuna gestione dei bordi. M/N/K devono essere
    // multipli di BM/BN/BK -- tutte le shape della consegna attuale lo sono.
    if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
        fprintf(stderr, "gemm_tiled_timed: M/N/K devono essere multipli di %d/%d/%d\n", BM, BN, BK);
        exit(EXIT_FAILURE);
    }

    size_t bytesA = static_cast<size_t>(M) * K * Bsize * sizeof(T);
    size_t bytesB = static_cast<size_t>(K) * N * Bsize * sizeof(T);
    size_t bytesC = static_cast<size_t>(M) * N * Bsize * sizeof(T);

    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 blockDim((BM * BN) / TM);          // 512 thread, 1D
    dim3 gridDim(N / BN, M / BM, Bsize);

    gemm_tiled_kernel<T, BM, BN, BK, TM><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < n_reps; ++r) {
        gemm_tiled_kernel<T, BM, BN, BK, TM><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
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