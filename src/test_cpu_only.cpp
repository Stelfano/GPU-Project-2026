// test_cpu_only.cpp
// Verifica solo-CPU della logica condivisa (generazione matrici, GEMM di
// riferimento, confronto) prima di compilare la parte CUDA sulla A100.
// Non fa parte del benchmark finale: serve solo come sanity check.
#include <cstdio>
#include <vector>

#include "../include/common.cuh"
#include "../include/cpu_gemm.h"

int main() {
    // Test 1: generate_matrix + compare_matrices, matrice contro se stessa.
    std::vector<float> M1;
    generate_matrix(M1, 64, 64, 42, 2.0f);
    double max_abs_err, mean_rel_err;
    compare_matrices(M1.data(), M1.data(), M1.size(), max_abs_err, mean_rel_err);
    printf("Test self-compare: max_abs_err=%.6f mean_rel_err=%.6f (attesi entrambi 0)\n",
           max_abs_err, mean_rel_err);

    // Test 2: timing della GEMM naive su una shape piccola.
    std::vector<float> A2, B2, C2;
    generate_matrix(A2, 256, 256, 1, 1.0f);
    generate_matrix(B2, 256, 256, 2, 1.0f);
    C2.resize(256 * 256);
    double ms = gemm_cpu_timed(A2.data(), B2.data(), C2.data(), 256, 256, 256);
    printf("Test timing: GEMM 256x256x256 in %.3f ms -> %.3f GFLOP/s\n",
           ms, gflops(256, 256, 256, ms));

    return 0;
}
