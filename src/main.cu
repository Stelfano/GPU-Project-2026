// main.cu
// Driver principale: per ogni shape genera A e B, esegue il riferimento CPU
// (dove previsto), esegue il kernel CUDA naive, confronta i risultati e
// stampa una tabella riassuntiva con tempi e GFLOP/s.
#include <cstdio>
#include <vector>
#include <typeinfo>

#include "../include/common.cuh"
#include "../include/cpu_gemm.h"
#include "../include/cuda_gemm.cuh"

int main() {
    auto shapes = default_shapes();

    printf("%-38s %6s %6s %6s %12s %12s %14s %12s\n",
           "shape", "M", "N", "K", "CPU(ms)", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<float> A, B;
        generate_matrix(A, s.M, s.K, /*seed=*/1234, 1.0f);
        generate_matrix(B, s.K, s.N, /*seed=*/5678, 1.0f);

        std::vector<float> C_gpu(static_cast<size_t>(s.M) * s.N);
        double gpu_ms = gemm_cuda_timed(A.data(), B.data(), C_gpu.data(), s.M, s.N, s.K, /*n_reps=*/10);
        double gpu_gflops = gflops(s.M, s.N, s.K, gpu_ms);

        char cpu_ms_str[32];
        char rel_err_str[32];

        if (s.verify_cpu) {
            std::vector<float> C_cpu(static_cast<size_t>(s.M) * s.N);
            double cpu_ms = gemm_cpu_timed(A.data(), B.data(), C_cpu.data(), s.M, s.N, s.K);
            double max_abs_err, mean_rel_err;
            compare_matrices(C_cpu.data(), C_gpu.data(), C_cpu.size(), max_abs_err, mean_rel_err);
            snprintf(cpu_ms_str, sizeof(cpu_ms_str), "%.3f", cpu_ms);
            snprintf(rel_err_str, sizeof(rel_err_str), "%.2e", mean_rel_err);
        } else {
            snprintf(cpu_ms_str, sizeof(cpu_ms_str), "skipped");
            snprintf(rel_err_str, sizeof(rel_err_str), "n/a");
        }

        printf("%-38s %6d %6d %6d %12s %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K,
               cpu_ms_str, gpu_ms, gpu_gflops, rel_err_str);
    }


    
    printf("------------------BFLOAT 16-----------------\n");
    printf("%-38s %6s %6s %6s %12s %12s %14s %12s\n",
           "shape", "M", "N", "K", "CPU(ms)", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<__nv_bfloat16> A, B;
        generate_matrix_bf16(A, s.M, s.K, /*seed=*/1234, 1.0f);
        generate_matrix_bf16(B, s.K, s.N, /*seed=*/5678, 1.0f);

        std::vector<__nv_bfloat16> C_gpu(static_cast<size_t>(s.M) * s.N);


        double gpu_ms = gemm_cuda_timed_bf16(A.data(),
                                             B.data(),
                                             C_gpu.data(), s.M, s.N, s.K, /*n_reps=*/10);
        double gpu_gflops = gflops(s.M, s.N, s.K, gpu_ms);

        char cpu_ms_str[32];
        char rel_err_str[32];
        snprintf(cpu_ms_str, sizeof(cpu_ms_str), "skipped");
        snprintf(rel_err_str, sizeof(rel_err_str), "n/a");

        printf("%-38s %6d %6d %6d %12s %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K,
               cpu_ms_str, gpu_ms, gpu_gflops, rel_err_str);
    }

    return 0;
}
