// FP16 output for the next GEMM, an optional FP32 copy for the residual stream.
#include "norm.cuh"

// One warp a row and one row a block, so short inputs spread over every multiprocessor.

// x: [rows, WIDTH]. out16: [rows, WIDTH]. out32: [rows, WIDTH] or null.
// Block: 32 threads. Grid: rows.
#define LAYERNORM(WIDTH)                                                                          \
    extern "C" __global__ void __launch_bounds__(32)                                              \
        layernorm_##WIDTH(const float* x, const float* gamma, const float* beta, __half* out16,   \
                          float* out32, float eps) {                                              \
        layernorm_kernel::layernorm<WIDTH>(x, gamma, beta, out16, out32, eps);                    \
    }

// hidden = LayerNorm((word[id] + token_type[type]) + position[pos]) per token.
// Block: 32 threads. Grid: tokens.
#define EMBED_LAYERNORM(WIDTH)                                                                    \
    extern "C" __global__ void __launch_bounds__(32) embed_layernorm_##WIDTH(                     \
        const int* ids, const int* types, const int* positions, const float* word,                \
        const float* position, const float* token_type, const float* gamma, const float* beta,    \
        __half* out16, float* out32, float eps) {                                                 \
        layernorm_kernel::embed_layernorm<WIDTH>(ids, types, positions, word, position,           \
                                                 token_type, gamma, beta, out16, out32, eps);     \
    }

LAYERNORM(384)
LAYERNORM(768)
LAYERNORM(1024)
EMBED_LAYERNORM(384)
EMBED_LAYERNORM(768)
EMBED_LAYERNORM(1024)
