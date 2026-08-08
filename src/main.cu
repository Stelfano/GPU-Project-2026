// main.cu
// Driver principale: per ogni shape genera A e B, esegue il riferimento CPU
// (dove previsto), esegue il kernel CUDA naive, confronta i risultati e
// stampa una tabella riassuntiva con tempi e GFLOP/s.
#include <cstdio>
#include <vector>
#include <typeinfo>
#include "cublas_v2.h"

#include "../include/common.cuh"
#include "../include/cpu_gemm.h"
#include "../include/cuda_gemm.cuh"

int main() {
    auto shapes = default_shapes();

    printf("%-38s %6s %6s %6s %6s %12s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "CPU(ms)", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<float> A, B;
        generate_matrix<float>(A, s.M, s.K, s.Bsize,/*seed=*/1234, 1.0f);
        generate_matrix<float>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<float> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);
        double gpu_ms = gemm_cuda_timed<float>(A.data(), B.data(), C_gpu.data(), s.M, s.N, s.K, s.Bsize, /*n_reps=*/10);
        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);

        char cpu_ms_str[32];
        char rel_err_str[32];

        if (s.verify_cpu) {
            std::vector<float> C_cpu(static_cast<size_t>(s.M) * s.N * s.Bsize);
            double cpu_ms = gemm_cpu_timed(A.data(), B.data(), C_cpu.data(), s.M, s.N, s.K);
            double max_abs_err, mean_rel_err;
            compare_matrices(C_cpu.data(), C_gpu.data(), C_cpu.size(), max_abs_err, mean_rel_err);
            snprintf(cpu_ms_str, sizeof(cpu_ms_str), "%.3f", cpu_ms);
            snprintf(rel_err_str, sizeof(rel_err_str), "%.2e", mean_rel_err);
        } else {
            snprintf(cpu_ms_str, sizeof(cpu_ms_str), "skipped");
            snprintf(rel_err_str, sizeof(rel_err_str), "n/a");
        }

        printf("%-38s %6d %6d %6d %6d %12s %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               cpu_ms_str, gpu_ms, gpu_gflops, rel_err_str);
    }


    
    printf("------------------BFLOAT 16-----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "CPU(ms)", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<__nv_bfloat16> A, B;
        generate_matrix<__nv_bfloat16>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<__nv_bfloat16>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<__nv_bfloat16> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);


        double gpu_ms = gemm_cuda_timed<__nv_bfloat16>(A.data(),
                                             B.data(),
                                             C_gpu.data(), s.M, s.N, s.K, s.Bsize, /*n_reps=*/10);
        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);

        char cpu_ms_str[32];
        char rel_err_str[32];
        snprintf(cpu_ms_str, sizeof(cpu_ms_str), "skipped");
        snprintf(rel_err_str, sizeof(rel_err_str), "n/a");

        printf("%-38s %6d %6d %6d %6d %12s %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               cpu_ms_str, gpu_ms, gpu_gflops, rel_err_str);
    }


        printf("------------------Float 16-----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "CPU(ms)", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<__half> A, B;
        generate_matrix<__half>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<__half>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<__half> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);


        double gpu_ms = gemm_cuda_timed<__half>(A.data(),
                                             B.data(),
                                             C_gpu.data(), s.M, s.N, s.K, s.Bsize, /*n_reps=*/10);
        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);

        char cpu_ms_str[32];
        char rel_err_str[32];
        snprintf(cpu_ms_str, sizeof(cpu_ms_str), "skipped");
        snprintf(rel_err_str, sizeof(rel_err_str), "n/a");

        printf("%-38s %6d %6d %6d %6d %12s %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               cpu_ms_str, gpu_ms, gpu_gflops, rel_err_str);
    }
    
    printf("------------------  CuBlas FP16  -----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "CPU(ms)", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<__half> A, B;
        generate_matrix<__half>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<__half>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<__half> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);

        size_t bytesA = static_cast<size_t>(s.M) * s.K * s.Bsize * sizeof(__half);
        size_t bytesB = static_cast<size_t>(s.K) * s.N * s.Bsize * sizeof(__half);
        size_t bytesC = static_cast<size_t>(s.M) * s.N * s.Bsize * sizeof(__half);

        __half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

        cudaMalloc(&d_A, bytesA);
        cudaMalloc(&d_B, bytesB);
        cudaMalloc(&d_C, bytesC);

        cublasHandle_t handle;

        cublasStatus_t status = cublasCreate(&handle);

        if(status != CUBLAS_STATUS_SUCCESS){
            fprintf(stderr, "cuBLAS FP16 initialization error\n");
            return EXIT_FAILURE;
        }

        cublasSetMatrix(s.M, s.K, sizeof(__half), A.data(), s.M, d_A, s.M);
        cublasSetMatrix(s.K, s.N, sizeof(__half), B.data(), s.K, d_B, s.K);
        cublasSetMatrix(s.M, s.N, sizeof(__half), C_gpu.data(), s.M, d_C, s.N);
        __half alpha = 1, beta = 1;
        cublasStatus_t stat;

        //Per ora senza warm-up per vedere se funziona
        //INoltre una sola moltiplicazione senza ripetizione (anche questo solo per vedere se funziona)
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float gpu_ms = 0;

        if(s.Bsize == 1){
            cudaEventRecord(start);

            stat = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, s.M, s.N, s.K, &alpha, d_A, s.K, d_B, s.K, &beta, d_C, s.M);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            if(stat != CUBLAS_STATUS_SUCCESS){
                fprintf(stderr, "cuBLAS FP16 non-batched gemm failure\n");
                return EXIT_FAILURE;
            }
            
            gpu_ms = 0.0f;
            cudaEventElapsedTime(&gpu_ms, start, stop);
        }else{
            cudaEventRecord(start);

            stat = cublasHgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                                            s.M, s.N, s.K, 
                                            &alpha, d_A, s.K, s.M*s.K*sizeof(__half),
                                            d_B, s.K, s.K*s.N*sizeof(__half),
                                            &beta, d_C, s.M, s.M*s.N*sizeof(__half), s.Bsize);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            if(stat != CUBLAS_STATUS_SUCCESS){
                fprintf(stderr, "cuBLAS FP16 non-batched gemm failure\n");
                return EXIT_FAILURE;
            }
            
            gpu_ms = 0.0f;
            cudaEventElapsedTime(&gpu_ms, start, stop);
        }


        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);

        char cpu_ms_str[32];
        char rel_err_str[32];
        snprintf(cpu_ms_str, sizeof(cpu_ms_str), "skipped");
        snprintf(rel_err_str, sizeof(rel_err_str), "n/a");

        printf("%-38s %6d %6d %6d %6d %12s %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               cpu_ms_str, gpu_ms, gpu_gflops, rel_err_str);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
    }

    printf("------------------  CuBlas BF16  -----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "CPU(ms)", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<__nv_bfloat16> A, B;
        generate_matrix<__nv_bfloat16>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<__nv_bfloat16>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<__half> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);

        size_t bytesA = static_cast<size_t>(s.M) * s.K * s.Bsize * sizeof(__nv_bfloat16);
        size_t bytesB = static_cast<size_t>(s.K) * s.N * s.Bsize * sizeof(__nv_bfloat16);
        size_t bytesC = static_cast<size_t>(s.M) * s.N * s.Bsize * sizeof(__nv_bfloat16);

        __half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

        cudaMalloc(&d_A, bytesA);
        cudaMalloc(&d_B, bytesB);
        cudaMalloc(&d_C, bytesC);

        cublasHandle_t handle;

        cublasStatus_t status = cublasCreate(&handle);

        if(status != CUBLAS_STATUS_SUCCESS){
            fprintf(stderr, "cuBLAS BF16 initialization error\n");
            return EXIT_FAILURE;
        }

        cublasSetMatrix(s.M, s.K, sizeof(__nv_bfloat16), A.data(), s.M, d_A, s.M);
        cublasSetMatrix(s.K, s.N, sizeof(__nv_bfloat16), B.data(), s.K, d_B, s.K);
        cublasSetMatrix(s.M, s.N, sizeof(__nv_bfloat16), C_gpu.data(), s.M, d_C, s.N);
        __half alpha = 1, beta = 1;
        cublasStatus_t stat;

        //Per ora senza warm-up per vedere se funziona
        //INoltre una sola moltiplicazione senza ripetizione (anche questo solo per vedere se funziona)
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float gpu_ms = 0;

        if(s.Bsize == 1){
            cudaEventRecord(start);

            stat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, s.M, s.N, s.K,
                                &alpha, d_A, CUDA_R_16BF, s.K, d_B, CUDA_R_16BF, s.K, 
                                &beta, d_C, CUDA_R_16BF, s.M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            if(stat != CUBLAS_STATUS_SUCCESS){
                fprintf(stderr, "cuBLAS BF16 non-batched gemm failure\n");
                return EXIT_FAILURE;
            }
            
            gpu_ms = 0.0f;
            cudaEventElapsedTime(&gpu_ms, start, stop);
        }else{
            cudaEventRecord(start);

            stat = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, s.M, s.N, s.K,
                                &alpha, d_A, CUDA_R_16BF, s.K, s.M*s.K*sizeof(__nv_bfloat16),
                                d_B, CUDA_R_16BF, s.K, s.M*s.K*sizeof(__nv_bfloat16),
                                &beta, d_C, CUDA_R_16BF, s.M, s.M*s.N*sizeof(__nv_bfloat16), s.Bsize,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            if(stat != CUBLAS_STATUS_SUCCESS){
                fprintf(stderr, "cuBLAS BF16 non-batched gemm failure\n");
                return EXIT_FAILURE;
            }
            
            gpu_ms = 0.0f;
            cudaEventElapsedTime(&gpu_ms, start, stop);
        }


        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);

        char cpu_ms_str[32];
        char rel_err_str[32];
        snprintf(cpu_ms_str, sizeof(cpu_ms_str), "skipped");
        snprintf(rel_err_str, sizeof(rel_err_str), "n/a");

        printf("%-38s %6d %6d %6d %6d %12s %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               cpu_ms_str, gpu_ms, gpu_gflops, rel_err_str);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
    }


    printf("------------------  CuBlas FP32  -----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "CPU(ms)", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<float> A, B;
        generate_matrix<float>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<float>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<float> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);

        size_t bytesA = static_cast<size_t>(s.M) * s.K * s.Bsize * sizeof(float);
        size_t bytesB = static_cast<size_t>(s.K) * s.N * s.Bsize * sizeof(float);
        size_t bytesC = static_cast<size_t>(s.M) * s.N * s.Bsize * sizeof(float);

        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

        cudaMalloc(&d_A, bytesA);
        cudaMalloc(&d_B, bytesB);
        cudaMalloc(&d_C, bytesC);

        cublasHandle_t handle;

        cublasStatus_t status = cublasCreate(&handle);

        if(status != CUBLAS_STATUS_SUCCESS){
            fprintf(stderr, "cuBLAS FP32 initialization error\n");
            return EXIT_FAILURE;
        }

        cublasSetMatrix(s.M, s.K, sizeof(float), A.data(), s.M, d_A, s.M);
        cublasSetMatrix(s.K, s.N, sizeof(float), B.data(), s.K, d_B, s.K);
        cublasSetMatrix(s.M, s.N, sizeof(float), C_gpu.data(), s.M, d_C, s.N);
        __half alpha = 1, beta = 1;
        cublasStatus_t stat;

        //Per ora senza warm-up per vedere se funziona
        //INoltre una sola moltiplicazione senza ripetizione (anche questo solo per vedere se funziona)
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float gpu_ms = 0;

        if(s.Bsize == 1){
            cudaEventRecord(start);

            stat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, s.M, s.N, s.K,
                                &alpha, d_A, CUDA_R_32F, s.K, d_B, CUDA_R_32F, s.K, 
                                &beta, d_C, CUDA_R_32F, s.M, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            if(stat != CUBLAS_STATUS_SUCCESS){
                fprintf(stderr, "cuBLAS FP32 non-batched gemm failure\n");
                return EXIT_FAILURE;
            }
            
            gpu_ms = 0.0f;
            cudaEventElapsedTime(&gpu_ms, start, stop);
        }else{
            cudaEventRecord(start);

            stat = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, s.M, s.N, s.K,
                                &alpha, d_A, CUDA_R_32F, s.K, s.M*s.K*sizeof(float),
                                d_B, CUDA_R_32F, s.K, s.M*s.K*sizeof(float),
                                &beta, d_C, CUDA_R_32F, s.M, s.M*s.N*sizeof(float), s.Bsize,
                                CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            if(stat != CUBLAS_STATUS_SUCCESS){
                fprintf(stderr, "cuBLAS FP32 non-batched gemm failure\n");
                return EXIT_FAILURE;
            }
            
            gpu_ms = 0.0f;
            cudaEventElapsedTime(&gpu_ms, start, stop);
        }


        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);

        char cpu_ms_str[32];
        char rel_err_str[32];
        snprintf(cpu_ms_str, sizeof(cpu_ms_str), "skipped");
        snprintf(rel_err_str, sizeof(rel_err_str), "n/a");

        printf("%-38s %6d %6d %6d %6d %12s %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               cpu_ms_str, gpu_ms, gpu_gflops, rel_err_str);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
    }

    return 0;
}
