// cuda_gemm.cuh
#pragma once
#include <cuda_bf16.h>
#include <cstdio>
#include "common.cuh"
#include <mma.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            fprintf(stderr, "Errore CUDA %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                            \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define WARP_SIZE 32

template <typename T, typename Acc, bool Fusion, bool Epl>
__global__ void gemm_naive_kernel(const T* __restrict__ A,const T* __restrict__ B, Acc* __restrict__ C, int M, int N, int K, int Bsize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < M && col < N && batch < Bsize) {
        Acc acc = 0.0f;
        
        for (int k = 0; k < K; k++) {
            acc += (float)A[static_cast<size_t>(row) * K + k + batch*(M*K)] * (float)B[static_cast<size_t>(k) * N + col + batch*(K*N)];
        }

        if constexpr (Fusion){
             C[static_cast<size_t>(row) * N + col + batch*(M*N)] = max(0.0f, acc);
        }else{
            C[static_cast<size_t>(row) * N + col + batch*(M*N)] = acc;
        }
    }
}


template <typename Acc>
__global__ void naiveReLU(Acc *C, int M, int N, int Bsize){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < M && col < N && batch < Bsize) {
        int pos = static_cast<size_t>(row) * N + col + batch*(M*N);
        if(C[pos] < (Acc)0)
            C[pos] = 0; 
    }
}

template <typename T, typename Acc, bool Fusion, bool Epl>
double gemm_cuda_timed(const T* h_A, const T* h_B, Acc* h_C, int M, int N, int K, int Bsize, int n_reps) {
    
    size_t bytesA = static_cast<size_t>(M) * K * Bsize * sizeof(T);
    size_t bytesB = static_cast<size_t>(K) * N * Bsize * sizeof(T);
    size_t bytesC = static_cast<size_t>(M) * N * Bsize * sizeof(Acc);

    T *d_A = nullptr, *d_B = nullptr;
    Acc *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16, 1);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y,
                 (Bsize + blockDim.z - 1) / blockDim.z);

    
    if constexpr (Epl){
        gemm_naive_kernel<T, Acc, Fusion, false><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
        naiveReLU<Acc><<<gridDim, blockDim>>>(d_C, M, N, Bsize);
    }else{
        gemm_naive_kernel<T, Acc, Fusion, false><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < n_reps; ++r) {
        if constexpr (Epl){
            gemm_naive_kernel<T, Acc, Fusion, false><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
            naiveReLU<T><<<gridDim, blockDim>>>(d_C, M, N, Bsize);
        }else{
            gemm_naive_kernel<T, Acc, Fusion, false><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
        }
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
__global__ void gemm_tiled_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, int M, int N, int K, int Bsize) {
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int batch     = blockIdx.z;

    __shared__ T As[BM * BK];
    __shared__ T Bs[BK * BN];

    const int thread_col = threadIdx.x % BN;
    const int thread_row = threadIdx.x / BN;  

    const T* A_batch = A + static_cast<size_t>(batch) * M * K;
    const T* B_batch = B + static_cast<size_t>(batch) * K * N;
    T*       C_batch = C + static_cast<size_t>(batch) * M * N;

    const T* A_tile = A_batch + static_cast<size_t>(block_row) * K;
    const T* B_tile = B_batch + block_col;
    T*       C_tile = C_batch + static_cast<size_t>(block_row) * N + block_col;

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
            T b_val = Bs[k * BN + thread_col];   
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                T a_val = As[(thread_row * TM + i) * BK + k];
                acc[i] += a_val * b_val;          
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

    size_t bytesA = static_cast<size_t>(M) * K * Bsize * sizeof(T);
    size_t bytesB = static_cast<size_t>(K) * N * Bsize * sizeof(T);
    size_t bytesC = static_cast<size_t>(M) * N * Bsize * sizeof(T);

    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 blockDim((BM * BN) / TM);          
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

    const int thread_col = threadIdx.x % (BN / TN);
    const int thread_row = threadIdx.x / (BN / TN);

    const T* A_batch = A + static_cast<size_t>(batch) * M * K;
    const T* B_batch = B + static_cast<size_t>(batch) * K * N;
    T*       C_batch = C + static_cast<size_t>(batch) * M * N;

    const T* A_tile = A_batch + static_cast<size_t>(block_row) * K;
    const T* B_tile = B_batch + block_col;
    T*       C_tile = C_batch + static_cast<size_t>(block_row) * N + block_col;
    int numThreads = (BM/TM) * (BN/TN);

    const int strideA = numThreads / BK;
    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;

    const int strideB = numThreads / BN;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    T a_val[TM] = {0.0};
    T b_val[TN] = {0.0};
    T thread_results[TM * TN] = {0.0};


    for (int k0 = 0; k0 < K; k0 += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A_tile[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B_tile[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        A_tile += BK;
        B_tile += static_cast<size_t>(BK) * N;


        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            T a_val[TM] = {0.0};
            T b_val[TN] = {0.0};
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_val[i] = As[(thread_row * TM + i) * BK + k];
            }
            for(int i = 0;i < TN; ++i){
                b_val[i] = Bs[k * BN + thread_col * TN + i];
            }

            for(int i = 0;i < TM;i++){
                for(int j = 0;j < TN; j++){
                    thread_results[i * TN + j] += a_val[i] * b_val[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        for(int j = 0; j < TN; j++)
            C_tile[static_cast<size_t>(thread_row * TM + i) * N + (thread_col * TN + j)] = thread_results[i * TN + j]; 
    }
}



template <typename T>
double gemm_tiled_timed_2D(const T* h_A, const T* h_B, T* h_C,
                         int M, int N, int K, int Bsize, int n_reps = 10) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 8, TN = 8;

    size_t bytesA = static_cast<size_t>(M) * K * Bsize * sizeof(T);
    size_t bytesB = static_cast<size_t>(K) * N * Bsize * sizeof(T);
    size_t bytesC = static_cast<size_t>(M) * N * Bsize * sizeof(T);

    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 blockDim((BM / TM) * (BN / TN));          // 512 thread, 1D
    dim3 gridDim(N / BN, M / BM, Bsize);

    gemm_tiled_kernel_2D<T, BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < n_reps; ++r) {
        gemm_tiled_kernel_2D<T, BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
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

template <typename T, int BM, int BN, int BK, int TM, int TN, int WN, int WM>
__global__ void gemm_warptiled_kernel(const T* __restrict__ A,
                                   const T* __restrict__ B,
                                   T* __restrict__ C,
                                   int M, int N, int K, int Bsize) {
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int batch     = blockIdx.z;
    const int WNITER    = 2;

    constexpr int WARPSIZE = 32;
    const int warpIdx = threadIdx.x / WARPSIZE;
    const int warpCol = warpIdx % (BN / WN);
    const int warpRow = warpIdx / (BN / WN);

    constexpr int WMITER = WN * WM / (TN * TM * WARPSIZE * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    //Thread in the subtile
    const int threadIdxWarp = threadIdx.x % WARPSIZE;
    const int threadColWarp = threadIdxWarp % (WSUBN / TN);
    const int threadRowWarp = threadIdxWarp / (WSUBN / TN);


    __shared__ T As[BM * BK];
    __shared__ T Bs[BK * BN];

    const T* A_batch = A + static_cast<size_t>(batch) * M * K;
    const T* B_batch = B + static_cast<size_t>(batch) * K * N;
    T*       C_batch = C + static_cast<size_t>(batch) * M * N;

    const T* A_tile = A_batch + static_cast<size_t>(block_row) * K;
    const T* B_tile = B_batch + block_col;
    T*       C_tile = C_batch + static_cast<size_t>(block_row) * N + block_col;
    int numThreads = (BM/WM)*(BN/WN)*WARPSIZE;

    const int strideA = numThreads / BK;
    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;

    const int strideB = numThreads / BN;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;

    T a_val[WMITER * TM] = {0.0};
    T b_val[WNITER * TN] = {0.0};
    T thread_results[WMITER * TM * TN * WNITER] = {0.0};

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A_tile[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B_tile[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        A_tile += BK;
        B_tile += static_cast<size_t>(BK) * N;

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < WMITER; ++i) {
                for(int j = 0; j < TM; ++j)
                    a_val[i * TM + j] = As[(warpRow * WM + i * WSUBM + threadRowWarp * TM + j) * BK + k];
            }

            for (int i = 0; i < WNITER; ++i) {
                for(int j = 0; j < TN; ++j)
                    b_val[i * TN + j] = Bs[k * BN + (warpCol * WN + i * WSUBN + threadColWarp * TN + j)];
            }

            for(int subRow = 0;subRow < WMITER; ++subRow){
                for(int subCol = 0;subCol < WNITER; ++subCol){
                    for(int i = 0; i < TM; ++i){
                        for(int j = 0; j < TN; j++){
                            thread_results[(subRow * TM + i) * (WNITER * TN) + (subCol * TN) + j] +=
                            a_val[subRow * TM + i] * b_val[subCol * TN + j];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    for(int subRow = 0;subRow < WMITER; ++subRow){
        for(int subCol = 0;subCol < WNITER; ++subCol){
            for(int i = 0; i < TM; ++i){
                for(int j = 0; j < TN; j++){
                    C_tile[(warpRow * WM + subRow * WSUBM + threadRowWarp * TM + i) * N
                       + (warpCol * WN + subCol * WSUBN + threadColWarp * TN + j)]
                        = thread_results[(subRow * TM + i) * (WNITER * TN) + (subCol * TN) + j];
                }
            }
        }
    }
}

template <typename T>
double gemm_warptiled_timed(const T* h_A, const T* h_B, T* h_C,
                         int M, int N, int K, int Bsize, int n_reps = 10) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4, WN = 32, WM = 64;

    size_t bytesA = static_cast<size_t>(M) * K * Bsize * sizeof(T);
    size_t bytesB = static_cast<size_t>(K) * N * Bsize * sizeof(T);
    size_t bytesC = static_cast<size_t>(M) * N * Bsize * sizeof(T);

    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    const int WARPSIZE = 32;
    const int NUM_WARPS = (BM/WM) * (BN/WN);
    const int NUM_THREADS = NUM_WARPS * WARPSIZE;
    dim3 blockDim(NUM_THREADS); 
    dim3 gridDim(N / BN, M / BM, Bsize);

    gemm_warptiled_kernel<T, BM, BN, BK, TM, TN, WN, WM><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < n_reps; ++r) {
        gemm_warptiled_kernel<T, BM, BN, BK, TM, TN, WN, WM><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
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


template <typename T, typename Acc>
__global__ void mma_kernel(T *a, T *b, Acc *c, int M, int N, int K, int Bsize) {
   // The only dimensions currently supported by WMMA
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
 
    // Leading dimensions. Packed with no transpositions.
    int lda = K;
    int ldb = N;
    int ldc = N;
     
    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Acc> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Acc> c_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

        // Loop over the K-dimension
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
    
            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

        // Load in current value of c, scale by beta, and add to result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_row_major);
        
        //Add ReLU here
        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
        }

            // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_row_major);
    }
}


template <typename T, typename Acc>
__global__ void batched_mma_kernel(T *a, T *b, Acc *c, int M, int N, int K, int Bsize) {
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
 
    int lda = K;
    int ldb = N;
    int ldc = N;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, Acc> acc_frag;

        // Loop over the K-dimension
    for(int batch = 0; batch < Bsize; batch++){
        wmma::fill_fragment(acc_frag, 0.0f);
        int cBatch = batch * M * N;
        int cRow = warpM * WMMA_M;
        int cCol = warpN * WMMA_N;

        for (int i = 0; i < K; i += WMMA_K) {
            int aRow = warpM * WMMA_M;
            int aCol = i;
            int bRow = i;
            int bCol = warpN * WMMA_N;
        
            int aBatch = batch * M * K;
            int bBatch = batch * K * N;
            int aOffset = aBatch + aRow * lda + aCol;
            int bOffset = bBatch + bRow * ldb + bCol; 

            if (aRow < M && aCol < K && bRow < K && bCol < N) {
                wmma::load_matrix_sync(a_frag, a + aOffset, lda);
                wmma::load_matrix_sync(b_frag, b + bOffset, ldb);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }

        if (cRow < M && cCol < N) {
            wmma::store_matrix_sync(c + cBatch + cRow * ldc + cCol, acc_frag, ldc, wmma::mem_row_major);
        }
    }
}


template <typename T, typename Acc>
double gemm_tensor_timed(const T* h_A, const T* h_B, Acc* h_C,
                       int M, int N, int K, int Bsize, int n_reps) {
    
    // Dimensione automatica basata sul tipo T passata alla funzione
    size_t bytesA = static_cast<size_t>(M) * K * Bsize * sizeof(T);
    size_t bytesB = static_cast<size_t>(K) * N * Bsize * sizeof(T);
    size_t bytesC = static_cast<size_t>(M) * N * Bsize * sizeof(Acc);

    T *d_A = nullptr, *d_B = nullptr;
    Acc *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    int BLKSIZE = 16;
    dim3 blockDim(128, 4);
    dim3 gridDim;
    gridDim.x = (M + (BLKSIZE * blockDim.x / 32 - 1)) / (BLKSIZE * blockDim.x / 32);
    gridDim.y = (N + (BLKSIZE * blockDim.y) - 1) / (BLKSIZE * blockDim.y);
    gridDim.z = 1;

    //warm-up
    batched_mma_kernel<T, float><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < n_reps; ++r) {
        batched_mma_kernel<T, float><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
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


template <typename T, typename Acc>
__global__ void tiled_mma_kernel(T *A, T *B, Acc *C, int M, int N, int K, int Bsize) {
    const int BLKSIZE = 128;
    const int SMEM_PAD = 8;
    int numThreads = 256;

    const int strideA = numThreads / BLKSIZE;
    const int innerRowA = threadIdx.x / BLKSIZE;
    const int innerColA = threadIdx.x % BLKSIZE;

    const int strideB = numThreads / BLKSIZE;
    const int innerRowB = threadIdx.x / BLKSIZE;
    const int innerColB = threadIdx.x % BLKSIZE;

    int warp_id = threadIdx.x / WARP_SIZE;
    const int warp_row = warp_id / 4; 
    const int warp_col = warp_id % 4; 

    const int block_row = blockIdx.y * BLKSIZE;
    const int block_col = blockIdx.x * BLKSIZE;
    const int batch     = blockIdx.z;

    const int warp_m_offset = warp_row * 64;
    const int warp_n_offset = warp_col * 32;

    const T* A_batch = A + static_cast<size_t>(batch) * M * K;
    const T* B_batch = B + static_cast<size_t>(batch) * K * N;
    Acc*       C_batch = C + static_cast<size_t>(batch) * M * N;

    const T* A_tile = A_batch + static_cast<size_t>(block_row) * K;
    const T* B_tile = B_batch + block_col;
    Acc*   C_tile = C_batch + static_cast<size_t>(block_row) * N + block_col;

    __shared__ T As[BLKSIZE*(BLKSIZE+SMEM_PAD)];
    __shared__ T Bs[BLKSIZE*(BLKSIZE+SMEM_PAD)];
    const int SMEM_LD = BLKSIZE + SMEM_PAD;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4][2];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    for (int k0 = 0; k0 < K; k0 += BLKSIZE) {
        #pragma unroll
        for (uint loadOffset = 0; loadOffset < BLKSIZE; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * SMEM_LD + innerColA] =
                A_tile[(innerRowA + loadOffset) * K + innerColA];
        }
        #pragma unroll
        for (uint loadOffset = 0; loadOffset < BLKSIZE; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * SMEM_LD + innerColB] =
                B_tile[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        A_tile += BLKSIZE;
        B_tile += static_cast<size_t>(BLKSIZE) * N;

        for (int k_step = 0; k_step < BLKSIZE; k_step += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag[2];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int row_a = warp_m_offset + (i * 16);
                wmma::load_matrix_sync(a_frag[i], &As[row_a * SMEM_LD + k_step], SMEM_LD);
            }

            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int col_b = warp_n_offset + (j * 16);
                wmma::load_matrix_sync(b_frag[j], &Bs[k_step * SMEM_LD + col_b], SMEM_LD);
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int global_r = block_row + warp_m_offset + (i * 16);
            int global_c = block_col + warp_n_offset + (j * 16);

            if (global_r < M && global_c < N) {
                wmma::store_matrix_sync(
                    &C_batch[global_r * N + global_c],
                    c_frag[i][j],
                    N,
                    wmma::mem_row_major
                );
            }
        }
    }
}



template <typename T, typename Acc>
double gemm_tensor_staged_timed(const T* h_A, const T* h_B, Acc* h_C,
                       int M, int N, int K, int Bsize, int n_reps) {
    size_t bytesA = static_cast<size_t>(M) * K * Bsize * sizeof(T);
    size_t bytesB = static_cast<size_t>(K) * N * Bsize * sizeof(T);
    size_t bytesC = static_cast<size_t>(M) * N * Bsize * sizeof(Acc);

    int BLKSIZE =  128;
    T *d_A = nullptr, *d_B = nullptr;
    Acc *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 blockDim(256, 1, 1); 
    dim3 gridDim((N + BLKSIZE - 1) / BLKSIZE, (M + BLKSIZE - 1) / BLKSIZE, Bsize);

    cudaFuncSetAttribute(
    tiled_mma_kernel<__half, float>, 
    cudaFuncAttributeMaxDynamicSharedMemorySize, 
    64 * 1024
    );
    tiled_mma_kernel<__half, float><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < n_reps; ++r) {
        tiled_mma_kernel<__half, float><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, Bsize);
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



