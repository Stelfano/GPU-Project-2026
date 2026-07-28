// cpu_gemm.h
#pragma once

// GEMM naive su CPU (triplo loop scalare): C (MxN) = A (MxK) * B (KxN),
// tutte row-major. È il riferimento di correttezza ad alta precisione per
// tutte le versioni successive (GPU naive, tiled, low-precision, Tensor Core).
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K);

// Esegue gemm_cpu misurando il tempo con std::chrono. Ritorna i millisecondi.
double gemm_cpu_timed(const float* A, const float* B, float* C, int M, int N, int K);
