// cuda_gemm.cu
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include "../include/cuda_gemm.cuh"


__global__ void tiled_mma_kernel_tf32(float *A, float *B, float *C, int M, int N, int K, int Bsize, bool Fusion) {
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

    const float* A_batch = A + static_cast<size_t>(batch) * M * K;
    const float* B_batch = B + static_cast<size_t>(batch) * K * N;
    float*       C_batch = C + static_cast<size_t>(batch) * M * N;

    const float* A_tile = A_batch + static_cast<size_t>(block_row) * K;
    const float* B_tile = B_batch + block_col;
    float*       C_tile = C_batch + static_cast<size_t>(block_row) * N + block_col;

    extern __shared__ float smem[];
    float* As = smem;
    float* Bs = smem + BLKSIZE * (BLKSIZE + SMEM_PAD);
    const int SMEM_LD = BLKSIZE + SMEM_PAD;

    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag[4][2];
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
                wmma::__float_to_tf32(A_tile[(innerRowA + loadOffset) * K + innerColA]);
        }
        #pragma unroll
        for (uint loadOffset = 0; loadOffset < BLKSIZE; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * SMEM_LD + innerColB] =
                wmma::__float_to_tf32(B_tile[(innerRowB + loadOffset) * N + innerColB]);
        }
        __syncthreads();

        A_tile += BLKSIZE;
        B_tile += static_cast<size_t>(BLKSIZE) * N;

        for (int k_step = 0; k_step < BLKSIZE; k_step += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag[2];

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

            if (Fusion){
                    for(int h = 0; h < c_frag[i][j].num_elements; h++){
                        c_frag[i][j].x[h] = max((float)0, c_frag[i][j].x[h]);
                }
            }


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

double gemm_tensor_tf32(const float* h_A, const float* h_B, float* h_C, int M, int N, int K, int Bsize, int n_reps, bool Fusion, bool Epl) {
    size_t bytesA = static_cast<size_t>(M) * K * Bsize * sizeof(float);
    size_t bytesB = static_cast<size_t>(K) * N * Bsize * sizeof(float);
    size_t bytesC = static_cast<size_t>(M) * N * Bsize * sizeof(float);

    int BLKSIZE =  128;
    float *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr;
    double times[n_reps] = {0.0f};
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    dim3 blockDim(256, 1, 1); 
    dim3 gridDim((N + BLKSIZE - 1) / BLKSIZE, (M + BLKSIZE - 1) / BLKSIZE, Bsize);

    size_t smem_bytes = 2ull * 128 * (128 + 8) * sizeof(float);

    cudaFuncSetAttribute(
        tiled_mma_kernel_tf32, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes
    );

    tiled_mma_kernel_tf32<<<gridDim, blockDim, smem_bytes>>>(d_A, d_B, d_C, M, N, K, Bsize, Fusion);

    if (Epl){
        dim3 blockDim(16, 16, 1);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
            (M + blockDim.y - 1) / blockDim.y,
            (Bsize + blockDim.z - 1) / blockDim.z);
        naiveReLU<float><<<gridDim, blockDim>>>(d_C, M, N, Bsize);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int r = 0; r < n_reps; ++r) {
        CUDA_CHECK(cudaEventRecord(start));

        dim3 blockDim(256, 1, 1); 
        dim3 gridDim((N + BLKSIZE - 1) / BLKSIZE, (M + BLKSIZE - 1) / BLKSIZE, Bsize);
        tiled_mma_kernel_tf32<<<gridDim, blockDim, smem_bytes>>>(d_A, d_B, d_C, M, N, K, Bsize, Fusion);

        if (Epl){
            dim3 blockDim(16, 16, 1);
            dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                (M + blockDim.y - 1) / blockDim.y,
                (Bsize + blockDim.z - 1) / blockDim.z);
            naiveReLU<float><<<gridDim, blockDim>>>(d_C, M, N, Bsize);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms_total = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
        times[r] = ms_total;
    }

    std::sort(times, times + n_reps);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return times[n_reps/2];
}