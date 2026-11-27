// FP16 Q, K and V with Q and K's rounding residuals; see attention.cuh. Each kernel serves the
// sequences of its kind: below `corrected_below` tokens the corrected ones, from there the plain ones.
#include "attention.cuh"

// Grid: (ceil(max_len / 64), HEADS, B). Block: 128 threads. Dynamic shared:
// 6 * 64 * D halfs (48 KiB at D = 64, 24 KiB at D = 32).
#define ATTENTION(NAME, HEADS, D, CORRECTED)                                                       \
    extern "C" __global__ void __launch_bounds__(attention_kernel::THREADS)                     \
        NAME(const __half* qkv, const __half* low, const int* cu_seqlens, __half* out,            \
             float scale_log2e, int corrected_below) {                                            \
        extern __shared__ __align__(128) __half shared[];                                         \
        attention_kernel::attention<HEADS, D, false, CORRECTED>(                                  \
            qkv, low, cu_seqlens, out, scale_log2e, nullptr, 0, corrected_below, shared);         \
    }

ATTENTION(attention_varlen_h12_d64, 12, 64, true)
ATTENTION(attention_varlen_plain_h12_d64, 12, 64, false)
ATTENTION(attention_varlen_h12_d32, 12, 32, true)
ATTENTION(attention_varlen_plain_h12_d32, 12, 32, false)

// The same, plus MPNet's relative-position bias for sequences of at most `longest` tokens.
#define ATTENTION_BIAS(NAME, HEADS, D, CORRECTED)                                                  \
    extern "C" __global__ void __launch_bounds__(attention_kernel::THREADS)                     \
        NAME(const __half* qkv, const __half* low, const int* cu_seqlens, __half* out,            \
             float scale_log2e, const float* bias, int longest, int corrected_below) {            \
        extern __shared__ __align__(128) __half shared[];                                         \
        attention_kernel::attention<HEADS, D, true, CORRECTED>(                                   \
            qkv, low, cu_seqlens, out, scale_log2e, bias, longest, corrected_below, shared);      \
    }

ATTENTION_BIAS(attention_varlen_bias_h16_d64, 16, 64, true)
ATTENTION_BIAS(attention_varlen_bias_plain_h16_d64, 16, 64, false)
ATTENTION_BIAS(attention_varlen_bias_h12_d64, 12, 64, true)
ATTENTION_BIAS(attention_varlen_bias_plain_h12_d64, 12, 64, false)
ATTENTION_BIAS(attention_varlen_bias_h12_d32, 12, 32, true)
ATTENTION_BIAS(attention_varlen_bias_plain_h12_d32, 12, 32, false)

// The 16 heads of 64 without a bias, in blocks of ROWS query rows. Grid: (ceil(max_len / ROWS),
// HEADS, B). Block: 2 * ROWS threads. Dynamic shared: 6 * 64 * D halfs. Blocks of 32 rows serve
// only batches too short for the plain kind.
#define PIPELINED(NAME, HEADS, D, ROWS, CORRECTED)                                                 \
    extern "C" __global__ void __launch_bounds__(ROWS * 2)                                        \
        NAME(const __half* qkv, const __half* low, const int* cu_seqlens, __half* out,            \
             float scale_log2e, int corrected_below) {                                            \
        extern __shared__ __align__(128) __half shared[];                                         \
        attention_kernel::pipelined<HEADS, D, ROWS / 16, CORRECTED>(                              \
            qkv, low, cu_seqlens, out, scale_log2e, corrected_below, shared);                     \
    }

PIPELINED(attention_pipelined_h16_d64_rows32, 16, 64, 32, true)
PIPELINED(attention_pipelined_h16_d64_rows64, 16, 64, 64, true)
PIPELINED(attention_pipelined_plain_h16_d64_rows64, 16, 64, 64, false)

// Attention over the keys within `reach` rows of each query, corrected at every length: a
// window's work does not grow with the sequence. Grid, block and shared as ATTENTION's.
#define ATTENTION_LOCAL(HEADS, D)                                                                  \
    extern "C" __global__ void __launch_bounds__(attention_kernel::THREADS)                     \
        attention_varlen_local_h##HEADS##_d##D(const __half* qkv, const __half* low,              \
                                               const int* cu_seqlens, __half* out,                \
                                               float scale_log2e, int reach) {                    \
        extern __shared__ __align__(128) __half shared[];                                         \
        attention_kernel::attention<HEADS, D, false, true, true>(                                 \
            qkv, low, cu_seqlens, out, scale_log2e, nullptr, 0, 0, shared, reach);                \
    }

ATTENTION_LOCAL(12, 64)
ATTENTION_LOCAL(12, 32)
ATTENTION_LOCAL(16, 64)
