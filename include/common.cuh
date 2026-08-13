// common.h
// Utilità condivise tra la versione CPU e quella CUDA: generazione matrici,
// confronto risultati, e definizione delle shape M,N,K da testare.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <cuda_bf16.h>

#define ReLU(x) max((float)0, (float)x) 

template <typename T>
inline void generate_matrix(std::vector<T>& mat, int rows, int cols, int Bsize,
                            unsigned seed, T scale=1.0f) {
    mat.resize((static_cast<size_t>(rows) * static_cast<size_t>(cols)) * static_cast<size_t>(Bsize));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for(auto& v : mat) v = static_cast<T>(dist(gen));
}

// Confronta due matrici (stessa dimensione n, stesso layout) e calcola
// errore assoluto massimo ed errore relativo medio. Serve già ora per
// validare GPU-naive contro CPU, e servirà tale e quale più avanti per
// confrontare i kernel a bassa precisione contro il riferimento FP32.
template<typename T>
inline void compare_matrices(const float* ref, const T* test, size_t n,
                              double& max_abs_err, double& mean_rel_err) {
    max_abs_err = 0.0;
    double sum_rel_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double r = static_cast<float>(ref[i]);
        double t = static_cast<float>(test[i]);
        double diff = std::fabs(r - t);
        max_abs_err = std::max(max_abs_err, diff);
        sum_rel_err += diff / (std::fabs(r) + 1e-8);
    }
    mean_rel_err = sum_rel_err / static_cast<float>(n);
}

// Una singola configurazione di test: C (MxN) = A (MxK) * B (KxN).
// verify_cpu: per le shape grandi il triplo loop naive su CPU diventa
// lento (minuti), quindi qui lo disattiviamo — quelle shape andranno
// validate più avanti a campione, o contro cuBLAS invece che contro
// questo riferimento scalare.
struct GemmShape {
    int M, N, K, Bsize;
    bool verify_cpu;
    std::string label;
    float *reference;
};

inline std::vector<GemmShape> default_shapes() {
    return {
        {256,  256,  256, 1,  true,  "square-small", nullptr},
        {1024, 1024, 1024, 1, true,  "square-medium", nullptr},
        {4096, 4096, 4096, 1, true, "square-large", nullptr},
        {4096, 1024, 1024, 1, true,  "tall-skinny (batch*seq x hidden)", nullptr},
        {4096, 4096, 1024, 1, true, "FFN up-projection (hidden -> 4*hidden)", nullptr},
        {4096, 1024, 4096, 1, true, "FFN down-projection (4*hidden -> hidden)", nullptr},
        {256,  256,  256, 2,  true,  "square-small-B2", nullptr},
    };
}

inline std::vector<GemmShape> reduced_shapes(){
    return{
        {256,  256,  256, 1,  true,  "square-small", nullptr},
        {1024, 256, 1024, 1,  true, "FFN down-projection (4*hidden -> hidden)", nullptr},
    };
}

inline double gflops(long long M, long long N, long long K, long long Bsize, double ms) {
    double flop = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) * static_cast<double>(Bsize);
    return flop / (ms / 1000.0) / 1e9;
}