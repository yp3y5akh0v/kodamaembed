// Sentence vectors (means of every row or of the first, summed in FP64, L2
// normalized) and reranking scores (BERT's classification head in FP32).
#include "common.cuh"

namespace pool_kernel {

__device__ __forceinline__ double warp_sum_double(double value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Block: WIDTH / 4 threads holding four columns each of one sequence's pooled row, which they
// store divided by its L2 norm.
template <int WIDTH>
__device__ __forceinline__ void normalize_store(const double (&pooled)[4], float* __restrict__ out) {
    constexpr int THREADS = WIDTH / 4;
    static_assert(THREADS % 32 == 0, "whole warps");
    __shared__ double warp_totals[THREADS / 32];
    double squares = 0.0;
#pragma unroll
    for (int i = 0; i < 4; ++i) squares += pooled[i] * pooled[i];
    squares = warp_sum_double(squares);
    if ((threadIdx.x & 31) == 0) warp_totals[threadIdx.x >> 5] = squares;
    __syncthreads();
    double total = 0.0;
#pragma unroll
    for (int w = 0; w < THREADS / 32; ++w) total += warp_totals[w];
    const double scale = 1.0 / fmax(sqrt(total), 1e-12);
    const float4 result = make_float4(
        static_cast<float>(pooled[0] * scale), static_cast<float>(pooled[1] * scale),
        static_cast<float>(pooled[2] * scale), static_cast<float>(pooled[3] * scale));
    *reinterpret_cast<float4*>(out + static_cast<size_t>(blockIdx.x) * WIDTH + threadIdx.x * 4) =
        result;
}

// One column a thread, summed in token order, so a short batch's columns spread over every
// multiprocessor. Grid: (WIDTH / 32, sequences). Block: 32 threads.
template <int WIDTH>
__device__ __forceinline__ void column_means(const float* __restrict__ hidden,
                                             const int* __restrict__ cu_seqlens,
                                             double* __restrict__ means) {
    const int sequence = blockIdx.y, column = blockIdx.x * 32 + threadIdx.x;
    const int first = cu_seqlens[sequence], end = cu_seqlens[sequence + 1];
    double sum = 0.0;
    for (int token = first; token < end; ++token) {
        sum += hidden[static_cast<size_t>(token) * WIDTH + column];
    }
    const double count = end - first;
    means[static_cast<size_t>(sequence) * WIDTH + column] = sum / count;
}

// Grid: one block per sequence. Block: WIDTH / 4 threads.
template <int WIDTH>
__device__ __forceinline__ void normalize_means(const double* __restrict__ means,
                                                float* __restrict__ out) {
    const double* row = means + static_cast<size_t>(blockIdx.x) * WIDTH + threadIdx.x * 4;
    const double pooled[4] = {row[0], row[1], row[2], row[3]};
    normalize_store<WIDTH>(pooled, out);
}

// Grid: one block per sequence. Block: WIDTH / 4 threads.
template <int WIDTH>
__device__ __forceinline__ void cls_pool_normalize(const float* __restrict__ hidden,
                                                   const int* __restrict__ cu_seqlens,
                                                   float* __restrict__ out) {
    const float4 v = *reinterpret_cast<const float4*>(
        hidden + static_cast<size_t>(cu_seqlens[blockIdx.x]) * WIDTH + threadIdx.x * 4);
    // A one-row mean summed from zero as column_means sums, so a -0.0 feature pools to +0.0 alike.
    const double pooled[4] = {0.0 + v.x, 0.0 + v.y, 0.0 + v.z, 0.0 + v.w};
    normalize_store<WIDTH>(pooled, out);
}

// Block: WIDTH threads, one pooled feature each. Grid: one block per sequence.
template <int WIDTH>
__device__ __forceinline__ void rerank_head(const float* __restrict__ hidden,
                                            const int* __restrict__ cu_seqlens,
                                            const float* __restrict__ pooler_w,
                                            const float* __restrict__ pooler_b,
                                            const float* __restrict__ classifier_w,
                                            const float* __restrict__ classifier_b,
                                            float* __restrict__ scores) {
    static_assert(WIDTH % 32 == 0 && WIDTH <= 1024, "whole warps in one block");
    __shared__ float first_row[WIDTH];
    __shared__ float warp_totals[WIDTH / 32];
    const int sequence = blockIdx.x, feature = threadIdx.x;
    first_row[feature] = hidden[static_cast<size_t>(cu_seqlens[sequence]) * WIDTH + feature];
    __syncthreads();
    const float* weights = pooler_w + static_cast<size_t>(feature) * WIDTH;
    float acc = pooler_b[feature];
    for (int k = 0; k < WIDTH; ++k) acc = fmaf(weights[k], first_row[k], acc);
    // tanh(Inf) is finite, so overflow would pass as a saturated score.
    const float activation = isfinite(acc) ? tanhf(acc) : __int_as_float(0x7fffffff);
    const float contribution = warp_sum(activation * classifier_w[feature]);
    if ((feature & 31) == 0) warp_totals[feature >> 5] = contribution;
    __syncthreads();
    if (feature == 0) {
        float total = classifier_b[0];
        for (int w = 0; w < WIDTH / 32; ++w) total += warp_totals[w];
        scores[sequence] = total;
    }
}

}  // namespace pool_kernel

// Mean pooling in two launches: column_means_<WIDTH> writes the FP64 means, normalize_means_<WIDTH>
// the normalized vectors.
#define POOL(WIDTH)                                                                               \
    extern "C" __global__ void __launch_bounds__(32)                                              \
        column_means_##WIDTH(const float* hidden, const int* cu_seqlens, double* means) {         \
        pool_kernel::column_means<WIDTH>(hidden, cu_seqlens, means);                              \
    }                                                                                             \
    extern "C" __global__ void __launch_bounds__(WIDTH / 4)                                       \
        normalize_means_##WIDTH(const double* means, float* out) {                                \
        pool_kernel::normalize_means<WIDTH>(means, out);                                          \
    }                                                                                             \
    extern "C" __global__ void __launch_bounds__(WIDTH / 4)                                       \
        cls_pool_normalize_##WIDTH(const float* hidden, const int* cu_seqlens, float* out) {      \
        pool_kernel::cls_pool_normalize<WIDTH>(hidden, cu_seqlens, out);                          \
    }

#define RERANK_HEAD(WIDTH)                                                                        \
    extern "C" __global__ void __launch_bounds__(WIDTH) rerank_head_##WIDTH(                      \
        const float* hidden, const int* cu_seqlens, const float* pooler_w, const float* pooler_b, \
        const float* classifier_w, const float* classifier_b, float* scores) {                    \
        pool_kernel::rerank_head<WIDTH>(hidden, cu_seqlens, pooler_w, pooler_b, classifier_w,    \
                                        classifier_b, scores);                                    \
    }

POOL(384)
POOL(768)
POOL(1024)
RERANK_HEAD(384)
RERANK_HEAD(768)
RERANK_HEAD(1024)
