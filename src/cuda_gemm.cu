// cuda_gemm.cu
#include "../include/cuda_gemm.cuh"
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

// Kernel naive: un thread calcola un elemento di C leggendo A e B
// direttamente da global memory ad ogni iterazione di k. Nessun tiling,
// nessun register blocking: è la baseline da battere con le versioni
// successive (shared memory, register blocking, bassa precisione, WMMA).
__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[static_cast<size_t>(row) * K + k] * B[static_cast<size_t>(k) * N + col];
        }
        C[static_cast<size_t>(row) * N + col] = acc;
    }
}

double gemm_cuda_timed(const float* h_A, const float* h_B, float* h_C,
                        int M, int N, int K, int n_reps) {
    size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Warm-up non cronometrato: la prima esecuzione paga costi una tantum
    // (inizializzazione del context, clock della GPU non ancora a regime)
    // che falserebbero la media se inclusi nella misura vera e propria.
    gemm_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < n_reps; ++r) {
        gemm_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
