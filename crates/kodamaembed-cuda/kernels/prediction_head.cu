// A cross-encoder's prediction head in FP32 as prediction_head_cls_<WIDTH> and
// prediction_head_mean_<WIDTH>. They are a module of their own: only an encoder that scores
// through this head pays to load it.
#include "common.cuh"

namespace prediction_kernel {

// The sum of `value` over the block's WIDTH threads, the same in each of them.
template <int WIDTH>
__device__ __forceinline__ float block_sum(float value, float* __restrict__ warp_totals) {
    const float warp = warp_sum(value);
    if ((threadIdx.x & 31) == 0) warp_totals[threadIdx.x >> 5] = warp;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < WIDTH / 32; ++w) total += warp_totals[w];
    // The next sum writes the totals again.
    __syncthreads();
    return total;
}

// classifier . norm(act(dense . pooled)) + classifier_bias for the block's sequence: `pooled` is
// this thread's feature of the sequence's pooled row, the activation SiLU when `silu` is set and
// the exact GELU otherwise. Block: WIDTH threads, one feature each. Grid: one block per sequence.
template <int WIDTH>
__device__ __forceinline__ void prediction_head(float pooled, const float* __restrict__ dense_w,
                                                const float* __restrict__ norm_g,
                                                const float* __restrict__ classifier_w,
                                                const float* __restrict__ classifier_b, float eps,
                                                int silu, float* __restrict__ scores) {
    static_assert(WIDTH % 32 == 0 && WIDTH <= 1024, "whole warps in one block");
    __shared__ float row[WIDTH];
    __shared__ float warp_totals[WIDTH / 32];
    const int feature = threadIdx.x;
    row[feature] = pooled;
    __syncthreads();
    const float* weights = dense_w + static_cast<size_t>(feature) * WIDTH;
    float acc = 0.0f;
    for (int k = 0; k < WIDTH; ++k) acc = fmaf(weights[k], row[k], acc);
    const float activated = silu != 0 ? acc / (1.0f + expf(-acc)) : gelu(acc);
    const float mean = block_sum<WIDTH>(activated, warp_totals) / WIDTH;
    const float centered = activated - mean;
    const float variance_eps = block_sum<WIDTH>(centered * centered, warp_totals) / WIDTH + eps;
    // rsqrt(Inf) is zero and would hide an overflowing sum or variance as a finite score.
    const float rstd = isfinite(variance_eps) ? rsqrtf(variance_eps) : __int_as_float(0x7fffffff);
    const float total = block_sum<WIDTH>(centered * rstd * norm_g[feature] * classifier_w[feature],
                                         warp_totals);
    if (feature == 0) scores[blockIdx.x] = classifier_b[0] + total;
}

}  // namespace prediction_kernel

// The head from a sequence's first row, or from the FP64 means pool.cu's column_means_<WIDTH> wrote.
#define PREDICTION_HEAD(WIDTH)                                                                    \
    extern "C" __global__ void __launch_bounds__(WIDTH) prediction_head_cls_##WIDTH(              \
        const float* hidden, const int* cu_seqlens, const float* dense_w, const float* norm_g,    \
        const float* classifier_w, const float* classifier_b, float eps, int silu,                \
        float* scores) {                                                                          \
        const float pooled =                                                                      \
            hidden[static_cast<size_t>(cu_seqlens[blockIdx.x]) * WIDTH + threadIdx.x];            \
        prediction_kernel::prediction_head<WIDTH>(pooled, dense_w, norm_g, classifier_w,          \
                                                  classifier_b, eps, silu, scores);               \
    }                                                                                             \
    extern "C" __global__ void __launch_bounds__(WIDTH) prediction_head_mean_##WIDTH(             \
        const double* means, const float* dense_w, const float* norm_g,                           \
        const float* classifier_w, const float* classifier_b, float eps, int silu,                \
        float* scores) {                                                                          \
        const float pooled =                                                                      \
            static_cast<float>(means[static_cast<size_t>(blockIdx.x) * WIDTH + threadIdx.x]);     \
        prediction_kernel::prediction_head<WIDTH>(pooled, dense_w, norm_g, classifier_w,          \
                                                  classifier_b, eps, silu, scores);               \
    }

PREDICTION_HEAD(384)
PREDICTION_HEAD(768)
PREDICTION_HEAD(1024)
