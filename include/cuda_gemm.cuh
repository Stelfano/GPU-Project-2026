// cuda_gemm.cuh
#pragma once
#include <cuda_bf16.h>

// Esegue C = A * B su GPU con un kernel naive (un thread per elemento di C,
// nessuna shared memory, nessun register blocking — è deliberatamente il
// punto di partenza più semplice possibile). Misura il tempo di solo kernel
// con CUDA events, mediato su n_reps ripetizioni dopo un run di warm-up non
// cronometrato. A, B, C sono puntatori host; la funzione gestisce da sola
// malloc/copy/free su device. Ritorna i millisecondi medi per iterazione.
double gemm_cuda_timed(const float* h_A, const float* h_B, float* h_C,
                        int M, int N, int K, int n_reps = 10);

double gemm_cuda_timed_bf16(const __nv_bfloat16 *h_A, const __nv_bfloat16 *h_B, __nv_bfloat16 *h_C,
                            int M, int N, int K, int n_reps = 10);
