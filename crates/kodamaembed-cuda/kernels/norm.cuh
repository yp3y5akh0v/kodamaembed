// LayerNorm with FP32 statistics, a warp a row; lane l holds float4s l + 32 i.
#pragma once

#include <cuda_fp16.h>

#include "common.cuh"

namespace layernorm_kernel {

// Normalizes the row in v and passes each float4 of the result to
// emit(group, column, y), with column = lane + 32 * group its float4 index.
template <int WIDTH, class Emit>
__device__ __forceinline__ void normalize_row(const float4 (&v)[WIDTH / 128],
                                              const float* __restrict__ gamma,
                                              const float* __restrict__ beta, float eps, Emit&& emit) {
    constexpr int GROUPS = WIDTH / 128;
    const int lane = threadIdx.x & 31;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < GROUPS; ++i) sum += (v[i].x + v[i].y) + (v[i].z + v[i].w);
    const float mean = warp_sum(sum) / WIDTH;
    float squares = 0.0f;
#pragma unroll
    for (int i = 0; i < GROUPS; ++i) {
        const float a = v[i].x - mean, b = v[i].y - mean;
        const float c = v[i].z - mean, d = v[i].w - mean;
        squares += (a * a + b * b) + (c * c + d * d);
    }
    const float variance_eps = warp_sum(squares) / WIDTH + eps;
    // rsqrt(Inf) is zero and could hide overflow as a finite all-zero row.
    // A NaN survives later layers and reaches the encoder's output guard.
    const float rstd = isfinite(mean) && isfinite(variance_eps)
                            ? rsqrtf(variance_eps)
                            : __int_as_float(0x7fffffff);
    const float4* scale = reinterpret_cast<const float4*>(gamma);
    const float4* shift = reinterpret_cast<const float4*>(beta);
#pragma unroll
    for (int i = 0; i < GROUPS; ++i) {
        const int column = lane + 32 * i;
        const float4 s = scale[column], t = shift[column];
        float4 y;
        y.x = (v[i].x - mean) * rstd * s.x + t.x;
        y.y = (v[i].y - mean) * rstd * s.y + t.y;
        y.z = (v[i].z - mean) * rstd * s.z + t.z;
        y.w = (v[i].w - mean) * rstd * s.w + t.w;
        emit(i, column, y);
    }
}

__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template <int WIDTH>
__device__ __forceinline__ void load_row(const float* __restrict__ x, int row, float4 (&v)[WIDTH / 128]) {
    const int lane = threadIdx.x & 31;
    const float4* source = reinterpret_cast<const float4*>(x + static_cast<size_t>(row) * WIDTH);
#pragma unroll
    for (int i = 0; i < WIDTH / 128; ++i) v[i] = source[lane + 32 * i];
}

// (word[id] + token_type[type]) + position[pos] of `token`.
template <int WIDTH>
__device__ __forceinline__ void load_embedding(const int* __restrict__ ids, const int* __restrict__ types,
                                               const int* __restrict__ positions,
                                               const float* __restrict__ word,
                                               const float* __restrict__ position,
                                               const float* __restrict__ token_type, int token,
                                               float4 (&v)[WIDTH / 128]) {
    const int lane = threadIdx.x & 31;
    const float4* w = reinterpret_cast<const float4*>(word + static_cast<size_t>(ids[token]) * WIDTH);
    const float4* p =
        reinterpret_cast<const float4*>(position + static_cast<size_t>(positions[token]) * WIDTH);
    const float4* t =
        reinterpret_cast<const float4*>(token_type + static_cast<size_t>(types[token]) * WIDTH);
#pragma unroll
    for (int i = 0; i < WIDTH / 128; ++i) {
        const int c = lane + 32 * i;
        v[i] = add4(add4(w[c], t[c]), p[c]);
    }
}

// FP16 output for the next GEMM and an optional FP32 copy for the residual.
template <int WIDTH>
__device__ __forceinline__ void normalize_store(const float4 (&v)[WIDTH / 128],
                                                const float* __restrict__ gamma,
                                                const float* __restrict__ beta,
                                                __half* __restrict__ out16,
                                                float* __restrict__ out32, float eps) {
    normalize_row<WIDTH>(v, gamma, beta, eps, [&](int, int column, float4 y) {
        reinterpret_cast<uint2*>(out16)[column] = pack_half4(y);
        if (out32 != nullptr) reinterpret_cast<float4*>(out32)[column] = y;
    });
}

template <int WIDTH>
__device__ __forceinline__ void layernorm(const float* __restrict__ x, const float* __restrict__ gamma,
                                          const float* __restrict__ beta, __half* __restrict__ out16,
                                          float* __restrict__ out32, float eps) {
    const int row = blockIdx.x;
    float4 v[WIDTH / 128];
    load_row<WIDTH>(x, row, v);
    const size_t offset = static_cast<size_t>(row) * WIDTH;
    normalize_store<WIDTH>(v, gamma, beta, out16 + offset,
                           out32 != nullptr ? out32 + offset : nullptr, eps);
}

template <int WIDTH>
__device__ __forceinline__ void embed_layernorm(
    const int* __restrict__ ids, const int* __restrict__ types, const int* __restrict__ positions,
    const float* __restrict__ word, const float* __restrict__ position,
    const float* __restrict__ token_type, const float* __restrict__ gamma,
    const float* __restrict__ beta, __half* __restrict__ out16, float* __restrict__ out32,
    float eps) {
    const int token = blockIdx.x;
    float4 v[WIDTH / 128];
    load_embedding<WIDTH>(ids, types, positions, word, position, token_type, token, v);
    const size_t offset = static_cast<size_t>(token) * WIDTH;
    normalize_store<WIDTH>(v, gamma, beta, out16 + offset,
                           out32 != nullptr ? out32 + offset : nullptr, eps);
}

}  // namespace layernorm_kernel
