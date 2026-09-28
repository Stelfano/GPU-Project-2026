// main.cu
#include <cstdio>
#include <vector>
#include <typeinfo>
#include "cublas_v2.h"

#include "../include/common.cuh"
#include "../include/cpu_gemm.h"
#include "../include/cuda_gemm.cuh"

int main() {
    auto shapes = reduced_shapes();
    const bool Epl = false;
    const bool Fusion = false;

    printf("------------------  CuBlas FP32  -----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (auto& s : shapes) {
        std::vector<float> A, B;
        generate_matrix<float>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<float>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);
        std::vector<float> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);
        size_t bytesC = static_cast<size_t>(s.M) * s.N * s.Bsize * sizeof(float);

        double mean_rel_err = 0;
        double max_abs_err = 0;
        char rel_err_str[32];

        double time = cuBlasFunc<float, float, Epl, Fusion>(A, B, C_gpu, s.M, s.N, s.K, s.Bsize, 30);

        if(s.reference == nullptr){
            float *ref = new float[s.M * s.N * s.Bsize];
            memcpy(ref, C_gpu.data(), bytesC);
            s.reference = ref;
        }

        if(s.verify_cpu){
            compare_matrices(s.reference, C_gpu.data(), C_gpu.size(), max_abs_err, mean_rel_err);
            snprintf(rel_err_str, sizeof(rel_err_str), "%.2e", mean_rel_err);
        }else{
            snprintf(rel_err_str, sizeof(rel_err_str), "n/a");
        }
            
        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, time);

        printf("%-38s %6d %6d %6d %6d %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               time, gpu_gflops, rel_err_str);
   }
    

    printf("------------------FLOAT 32-----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<float> A, B;
        generate_matrix<float>(A, s.M, s.K, s.Bsize,/*seed=*/1234, 1.0f);
        generate_matrix<float>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<float> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);
        double gpu_ms = gemm_cuda_timed<float, float, Fusion, Epl>(A.data(), B.data(), C_gpu.data(), s.M, s.N, s.K, s.Bsize, /*n_reps=*/30);
        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);

        double max_abs_err;
        double mean_rel_err;

        char rel_err_str[32];

        if(s.verify_cpu){
            compare_matrices_frob(s.reference, C_gpu.data(), C_gpu.size(), max_abs_err, mean_rel_err);
            snprintf(rel_err_str, sizeof(rel_err_str), "%.2e", mean_rel_err);
        }else{
            snprintf(rel_err_str, sizeof(rel_err_str), "n/a");
        }

        printf("%-38s %6d %6d %6d %6d %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
                gpu_ms, gpu_gflops, rel_err_str);
    }


    
    printf("------------------BFLOAT 16-----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<__nv_bfloat16> A, B;
        generate_matrix<__nv_bfloat16>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<__nv_bfloat16>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<float> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);

        double gpu_ms = gemm_cuda_timed<__nv_bfloat16, float, Fusion, Epl>(A.data(), B.data(), C_gpu.data(), s.M, s.N, s.K, s.Bsize, /*n_reps=*/30);
        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);

        double max_abs_err;
        double mean_rel_err;

        char rel_err_str[32];

        if(s.verify_cpu){
            compare_matrices_frob<float>(s.reference, C_gpu.data(), C_gpu.size(), max_abs_err, mean_rel_err);
            snprintf(rel_err_str, sizeof(rel_err_str), "%.2e", mean_rel_err);
        }else{
            snprintf(rel_err_str, sizeof(rel_err_str), "n/a");
        }


        printf("%-38s %6d %6d %6d %6d %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               gpu_ms, gpu_gflops, rel_err_str);
    }


    printf("------------------Float 16-----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<__half> A, B;
        generate_matrix<__half>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<__half>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<float> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);


        double gpu_ms = gemm_cuda_timed<__half, float, Fusion, Epl>(A.data(), B.data(), C_gpu.data(), s.M, s.N, s.K, s.Bsize, /*n_reps=*/30);
        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);

       
        double max_abs_err;
        double mean_rel_err;

        char rel_err_str[32];

        if(s.verify_cpu){
            compare_matrices_frob<float>(s.reference, C_gpu.data(), C_gpu.size(), max_abs_err, mean_rel_err);
            snprintf(rel_err_str, sizeof(rel_err_str), "%.2e", mean_rel_err);
        }else{
            snprintf(rel_err_str, sizeof(rel_err_str), "n/a");
        }

        printf("%-38s %6d %6d %6d %6d %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               gpu_ms, gpu_gflops, rel_err_str);
    }
    
    printf("------------------  CuBlas FP16  -----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<__half> A, B;
        generate_matrix<__half>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<__half>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

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

        cudaMemcpy(d_A, A.data(), bytesA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), bytesB, cudaMemcpyHostToDevice);
        float alpha = 1, beta = 0;
        cublasStatus_t stat;
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float gpu_ms = 0;

        stat = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            s.N, s.M, s.K,
                            &alpha,
                            d_B, CUDA_R_16F, s.N, s.M*s.K,
                            d_A, CUDA_R_16F, s.K, s.N*s.K,
                            &beta,
                            d_C, CUDA_R_16F, s.N, s.M*s.N, s.Bsize,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        cudaEventRecord(start);

        for(int r = 0; r < 10; r++){
            stat = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                s.N, s.M, s.K,
                                &alpha,
                                d_B, CUDA_R_16F, s.N, s.M*s.K,
                                d_A, CUDA_R_16F, s.K, s.N*s.K,
                                &beta,
                                d_C, CUDA_R_16F, s.N, s.M*s.N, s.Bsize,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            if constexpr (Fusion || Epl){
                dim3 blockDim(16, 16, 1);
                dim3 gridDim((s.N + blockDim.x - 1) / blockDim.x,
                    (s.M + blockDim.y - 1) / blockDim.y,
                    (s.Bsize + blockDim.z - 1) / blockDim.z);
                naiveReLU<__half><<<gridDim, blockDim>>>(d_C, s.M, s.N, s.Bsize);
            }
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        if(stat != CUBLAS_STATUS_SUCCESS){
            fprintf(stderr, "cuBLAS FP16 GEMM failure\n");
            return EXIT_FAILURE;
        }
           
        cudaEventElapsedTime(&gpu_ms, start, stop);


        gpu_ms = gpu_ms / 10;
        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);
        double max_abs_err;
        double mean_rel_err;

        char rel_err_str[32];

        if(s.verify_cpu){
            std::vector<__half> C_gpu(s.M * s.N * s.Bsize);
            cudaMemcpy(C_gpu.data(), d_C, bytesC, cudaMemcpyDeviceToHost);
            compare_matrices_frob(s.reference, C_gpu.data(), C_gpu.size(), max_abs_err, mean_rel_err);
            snprintf(rel_err_str, sizeof(rel_err_str), "%.2e", mean_rel_err);
        }else{
            snprintf(rel_err_str, sizeof(rel_err_str), "n/a");
        }


        printf("%-38s %6d %6d %6d %6d %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               gpu_ms, gpu_gflops, rel_err_str);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
    }

    printf("------------------  CuBlas BF16  -----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<__nv_bfloat16> A, B;
        generate_matrix<__nv_bfloat16>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<__nv_bfloat16>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<__nv_bfloat16> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);

        size_t bytesA = static_cast<size_t>(s.M) * s.K * s.Bsize * sizeof(__nv_bfloat16);
        size_t bytesB = static_cast<size_t>(s.K) * s.N * s.Bsize * sizeof(__nv_bfloat16);
        size_t bytesC = static_cast<size_t>(s.M) * s.N * s.Bsize * sizeof(__nv_bfloat16);

        __nv_bfloat16 *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

        cudaMalloc(&d_A, bytesA);
        cudaMalloc(&d_B, bytesB);
        cudaMalloc(&d_C, bytesC);

        cublasHandle_t handle;
        float gpu_ms;

        cublasStatus_t status = cublasCreate(&handle);

        if(status != CUBLAS_STATUS_SUCCESS){
            fprintf(stderr, "cuBLAS BF16 initialization error\n");
            return EXIT_FAILURE;
        }

        cudaMemcpy(d_A, A.data(), bytesA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), bytesB, cudaMemcpyHostToDevice);
        float alpha = 1, beta = 0;
        cublasStatus_t stat;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        //warmup
        stat = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    s.N, s.M, s.K,
                                    &alpha,
                                    d_B, CUDA_R_16BF, s.N, s.M*s.K,
                                    d_A, CUDA_R_16BF, s.K, s.N*s.K,
                                    &beta,
                                    d_C, CUDA_R_16BF, s.N, s.M*s.N, s.Bsize,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            

        cudaEventRecord(start);

        for(int r=0;r < 10; r++){
            stat = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                s.N, s.M, s.K,
                                &alpha,
                                d_B, CUDA_R_16BF, s.N, s.M*s.K,
                                d_A, CUDA_R_16BF, s.K, s.N*s.K,
                                &beta,
                                d_C, CUDA_R_16BF, s.N, s.M*s.N, s.Bsize,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            if constexpr (Fusion || Epl){
                dim3 blockDim(16, 16, 1);
                dim3 gridDim((s.N + blockDim.x - 1) / blockDim.x,
                    (s.M + blockDim.y - 1) / blockDim.y,
                    (s.Bsize + blockDim.z - 1) / blockDim.z);
                naiveReLU<__nv_bfloat16><<<gridDim, blockDim>>>(d_C, s.M, s.N, s.Bsize);
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        if(stat != CUBLAS_STATUS_SUCCESS){
            fprintf(stderr, "cuBLAS BF16 non-batched gemm failure\n");
            return EXIT_FAILURE;
        }
            
        cudaEventElapsedTime(&gpu_ms, start, stop);


        gpu_ms = gpu_ms/10;
        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);
        double max_abs_err;
        double mean_rel_err;

        char rel_err_str[32];

        if(s.verify_cpu){
            std::vector<__nv_bfloat16> C_gpu(s.M * s.N * s.Bsize);
            cudaMemcpy(C_gpu.data(), d_C, bytesC, cudaMemcpyDeviceToHost);
            compare_matrices_frob(s.reference, C_gpu.data(), C_gpu.size(), max_abs_err, mean_rel_err);
            snprintf(rel_err_str, sizeof(rel_err_str), "%.2e", mean_rel_err);
        }else{
            snprintf(rel_err_str, sizeof(rel_err_str), "n/a");
        }


        printf("%-38s %6d %6d %6d %6d %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               gpu_ms, gpu_gflops, rel_err_str);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
    }

        
    printf("------------------BFLOAT 16 -- SharedMem+RegisterBlock 1D-----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<__nv_bfloat16> A, B;
        generate_matrix<__nv_bfloat16>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<__nv_bfloat16>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<float> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);


        double gpu_ms = gemm_tiled_timed<__nv_bfloat16, float, Fusion, Epl>(A.data(),
                                             B.data(),
                                             C_gpu.data(), s.M, s.N, s.K, s.Bsize, /*n_reps=*/30);
        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);

        double max_abs_err;
        double mean_rel_err;

        char rel_err_str[32];

        if(s.verify_cpu){
            compare_matrices_frob<float>(s.reference, C_gpu.data(), C_gpu.size(), max_abs_err, mean_rel_err);
            snprintf(rel_err_str, sizeof(rel_err_str), "%.2e", mean_rel_err);
        }else{
            snprintf(rel_err_str, sizeof(rel_err_str), "n/a");
        }


        printf("%-38s %6d %6d %6d %6d %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               gpu_ms, gpu_gflops, rel_err_str);
    }


    printf("------------------BFLOAT 16 -- SharedMem+RegisterBlock 2D -----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<__nv_bfloat16> A, B;
        generate_matrix<__nv_bfloat16>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<__nv_bfloat16>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<float> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);


        double gpu_ms = gemm_tiled_timed_2D<__nv_bfloat16, float, Fusion, Epl>(A.data(),
                                             B.data(),
                                             C_gpu.data(), s.M, s.N, s.K, s.Bsize, /*n_reps=*/30);
        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);

        double max_abs_err;
        double mean_rel_err;

        char rel_err_str[32];

        if(s.verify_cpu){
            compare_matrices_frob<float>(s.reference, C_gpu.data(), C_gpu.size(), max_abs_err, mean_rel_err);
            snprintf(rel_err_str, sizeof(rel_err_str), "%.2e", mean_rel_err);
        }else{
            snprintf(rel_err_str, sizeof(rel_err_str), "n/a");
        }


        printf("%-38s %6d %6d %6d %6d %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               gpu_ms, gpu_gflops, rel_err_str);
    }

     printf("------------------FLOAT 16 -- Warptile kernel -----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B",  "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<__half> A, B;
        generate_matrix<__half>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<__half>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<float> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);


        double gpu_ms = gemm_warptiled_timed<__half, float, Fusion, Epl>(A.data(),
                                             B.data(),
                                             C_gpu.data(), s.M, s.N, s.K, s.Bsize, /*n_reps=*/30);
        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);

        double max_abs_err;
        double mean_rel_err;

        char rel_err_str[32];

        if(s.verify_cpu){
            compare_matrices_frob(s.reference, C_gpu.data(), C_gpu.size(), max_abs_err, mean_rel_err);
            snprintf(rel_err_str, sizeof(rel_err_str), "%.2e", mean_rel_err);
        }else{
            snprintf(rel_err_str, sizeof(rel_err_str), "n/a");
        }


        printf("%-38s %6d %6d %6d %6d %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               gpu_ms, gpu_gflops, rel_err_str);
    }


 printf("------------------FLOAT 16 / F32 Accumulation -- Tensor Core Basic -----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<__half> A, B;
        generate_matrix<__half>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<__half>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<float> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);


        double gpu_ms = gemm_tensor_timed<__half, float, Fusion, Epl>(A.data(),
                                             B.data(),
                                             C_gpu.data(), s.M, s.N, s.K, s.Bsize, /*n_reps=*/30);
        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);

        double max_abs_err;
        double mean_rel_err;

        char rel_err_str[32];

        if(s.verify_cpu){
            compare_matrices_frob(s.reference, C_gpu.data(), C_gpu.size(), max_abs_err, mean_rel_err);
            snprintf(rel_err_str, sizeof(rel_err_str), "%.2e", mean_rel_err);
        }else{
            snprintf(rel_err_str, sizeof(rel_err_str), "n/a");
        }


        printf("%-38s %6d %6d %6d %6d %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               gpu_ms, gpu_gflops, rel_err_str);
    }


    printf("------------------FLOAT 16 / F32 Accumulation -- Tensor Core Advanced -----------------\n");
    printf("%-38s %6s %6s %6s %6s %12s %14s %12s\n",
           "shape", "M", "N", "K", "B", "GPU(ms)", "GPU GFLOP/s", "err.rel");
    printf("--------------------------------------------------------------------------------------------------------\n");

    for (const auto& s : shapes) {
        std::vector<__half> A, B;
        generate_matrix<__half>(A, s.M, s.K, s.Bsize, /*seed=*/1234, 1.0f);
        generate_matrix<__half>(B, s.K, s.N, s.Bsize,/*seed=*/5678, 1.0f);

        std::vector<float> C_gpu(static_cast<size_t>(s.M) * s.N * s.Bsize);


        double gpu_ms = gemm_tensor_staged_timed<__half, float, Fusion, Epl>(A.data(),
                                             B.data(),
                                             C_gpu.data(), s.M, s.N, s.K, s.Bsize, /*n_reps=*/30);
        double gpu_gflops = gflops(s.M, s.N, s.K, s.Bsize, gpu_ms);

        double max_abs_err;
        double mean_rel_err;

        char rel_err_str[32];

        if(s.verify_cpu){
            compare_matrices_frob(s.reference, C_gpu.data(), C_gpu.size(), max_abs_err, mean_rel_err);
            snprintf(rel_err_str, sizeof(rel_err_str), "%.2e", mean_rel_err);
        }else{
            snprintf(rel_err_str, sizeof(rel_err_str), "n/a");
        }


        printf("%-38s %6d %6d %6d %6d %12.3f %14.4f %12s\n",
               s.label.c_str(), s.M, s.N, s.K, s.Bsize,
               gpu_ms, gpu_gflops, rel_err_str);
    }


    for(auto &s : shapes){
        delete[] s.reference;
    }
    return 0;
}
