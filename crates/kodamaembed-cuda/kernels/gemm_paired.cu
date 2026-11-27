// The GEMMs over pairs of columns as gemm_bias_geglu_f16_<NAME> and gemm_bias_turned_f16_<NAME>,
// in each of gemm.cuh's r<rows>c<columns> tilings and writing identical bits in all of them. They
// are a module of their own: only an encoder that launches them pays to load it. Grid, block and
// dynamic shared: as gemm.cu's r tilings. The gated product's out is [M, N / 2]. A turned GEMM's
// low is as gemm.cu's, its positions hold M values, its turns each position's head / 2 pairs, and
// turned_columns is a multiple of head and at most N.
#include "gemm.cuh"

#define PAIRED(NAME) DEFINE_PAIRED(NAME, (gemm::NAME::THREADS))
// A one-block minimum on the tiles where gemm.cu's kernels take one to keep state in registers.
#define PAIRED_MIN_ONE(NAME) DEFINE_PAIRED(NAME, (gemm::NAME::THREADS, 1))
#define DEFINE_PAIRED(NAME, BOUNDS)                                                               \
    extern "C" __global__ void __launch_bounds__ BOUNDS                                           \
        gemm_bias_geglu_f16_##NAME(const __half* a, const __half* w, const float* bias,           \
                                   __half* out, int M, int N, int K) {                            \
        extern __shared__ __align__(128) __half shared[];                                         \
        gemm::run<gemm::NAME, gemm::BIAS_GEGLU_F16>(a, w, bias, nullptr, out, nullptr, 0, M, N,   \
                                                    K, shared);                                   \
    }                                                                                             \
    extern "C" __global__ void __launch_bounds__ BOUNDS                                           \
        gemm_bias_turned_f16_##NAME(const __half* a, const __half* w, const float* bias,          \
                                    __half* out, __half* low, int low_columns,                    \
                                    const int* positions, const float2* turns, int head,          \
                                    int turned_columns, int M, int N, int K) {                    \
        extern __shared__ __align__(128) __half shared[];                                         \
        gemm::run<gemm::NAME, gemm::BIAS_TURNED_SPLIT_F16>(                                       \
            a, w, bias, nullptr, out, low, low_columns, M, N, K, shared,                          \
            gemm::Turns{positions, turns, head, turned_columns});                                 \
    }

PAIRED_MIN_ONE(r16c32)
PAIRED(r16c64)
PAIRED(r32c32)
PAIRED(r48c32)
PAIRED_MIN_ONE(r48c64)
PAIRED(r48c128)
PAIRED(r80c64)
PAIRED(r80c128)
PAIRED(r96c128)
PAIRED_MIN_ONE(r64c64s3)
PAIRED(r64c64s4)
PAIRED(r64c128)
PAIRED(r128c64)
PAIRED(r128c128)
PAIRED(r128c128w8)
PAIRED(r256c128)
PAIRED(r64c128k64)
PAIRED(r128c128w8k64)
PAIRED(r128c88)
PAIRED(r128c176)
