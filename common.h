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

// Genera una matrice rows x cols in row-major, valori uniformi in [-scale, scale].
// Il seed è esplicito così A e B (o due run diversi) sono riproducibili.
inline void generate_matrix(std::vector<float>& mat, int rows, int cols,
                             unsigned seed, float scale = 1.0f) {
    mat.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& v : mat) v = dist(gen);
}

// Confronta due matrici (stessa dimensione n, stesso layout) e calcola
// errore assoluto massimo ed errore relativo medio. Serve già ora per
// validare GPU-naive contro CPU, e servirà tale e quale più avanti per
// confrontare i kernel a bassa precisione contro il riferimento FP32.
inline void compare_matrices(const float* ref, const float* test, size_t n,
                              double& max_abs_err, double& mean_rel_err) {
    max_abs_err = 0.0;
    double sum_rel_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double r = static_cast<double>(ref[i]);
        double t = static_cast<double>(test[i]);
        double diff = std::fabs(r - t);
        max_abs_err = std::max(max_abs_err, diff);
        sum_rel_err += diff / (std::fabs(r) + 1e-8);
    }
    mean_rel_err = sum_rel_err / static_cast<double>(n);
}

// Una singola configurazione di test: C (MxN) = A (MxK) * B (KxN).
// verify_cpu: per le shape grandi il triplo loop naive su CPU diventa
// lento (minuti), quindi qui lo disattiviamo — quelle shape andranno
// validate più avanti a campione, o contro cuBLAS invece che contro
// questo riferimento scalare.
struct GemmShape {
    int M, N, K;
    bool verify_cpu;
    std::string label;
};

inline std::vector<GemmShape> default_shapes() {
    return {
        {256,  256,  256,  true,  "square-small"},
        {1024, 1024, 1024, true,  "square-medium"},
        {4096, 4096, 4096, false, "square-large"},
        {4096, 1024, 1024, true,  "tall-skinny (batch*seq x hidden)"},
        {4096, 4096, 1024, false, "FFN up-projection (hidden -> 4*hidden)"},
        {4096, 1024, 4096, false, "FFN down-projection (4*hidden -> hidden)"},
    };
}

inline double gflops(long long M, long long N, long long K, double ms) {
    double flop = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    return flop / (ms / 1000.0) / 1e9;
}
