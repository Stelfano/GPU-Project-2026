// cpu_gemm.cpp
#include "cpu_gemm.h"
#include <chrono>

void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[static_cast<size_t>(m) * K + k] * B[static_cast<size_t>(k) * N + n];
            }
            C[static_cast<size_t>(m) * N + n] = acc;
        }
    }
}

double gemm_cpu_timed(const float* A, const float* B, float* C, int M, int N, int K) {
    auto t0 = std::chrono::high_resolution_clock::now();
    gemm_cpu(A, B, C, M, N, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
