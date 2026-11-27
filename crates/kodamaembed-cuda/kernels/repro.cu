// The CPU encoder's centered INT8 operations in its order, for the same bits. Rounding
// intrinsics stop nvcc reordering; work splits only over exact sums and maxima.

#include <cuda_fp16.h>
#include <stdint.h>

#include <type_traits>

#include "int8.cuh"

// Float constants, the CPU's exact values, uploaded by the host.
#define FC_RECIP_SQRT_BLOCK 0
#define FC_GELU_A 1  // Abramowitz and Stegun 7.1.28's a1 .. a6
#define FC_GELU_CAP 7
#define FC_FRAC_1_SQRT_2 8

#define BLOCK 128
// The inputs of one promoted dot: half a rotated block, which takes a step of its own.
#define HALF 64

// A product's steps lie across, steps[part * rows + row]: the rows one load takes are
// neighbours at any depth.
template <class Step>
__device__ __forceinline__ Step* repro_step_at(Step* steps, int rows, int part, int row) {
    return steps + (size_t)part * rows + row;
}

// An immediate ID: from a register, ptxas reserves all sixteen barriers. Groups split
// the block, so these barriers must not assert CTA-aligned execution.
template <int ID>
__device__ __forceinline__ void repro_bar_sync(int count) {
    asm volatile("barrier.sync %0, %1;" ::"n"(ID), "r"(count) : "memory");
}
template <int ID>
__device__ __forceinline__ void repro_bar_arrive(int count) {
    asm volatile("barrier.arrive %0, %1;" ::"n"(ID), "r"(count) : "memory");
}

// The whole block, or THREADS threads on named barrier ID.
struct ReproBlockSync {
    __device__ __forceinline__ void operator()() const { __syncthreads(); }
};
template <int ID, int THREADS>
struct ReproNamedSync {
    __device__ __forceinline__ void operator()() const { repro_bar_sync<ID>(THREADS); }
};
// Group INDEX's four warps on named barrier 1 + INDEX.
template <int INDEX>
using GroupSync = ReproNamedSync<1 + INDEX, 128>;

// Thread 0 orders the block's writes before its count, as a grid barrier does;
// a waiter's thread 0 polls with acquire semantics, then its block meets.
template <class Sync>
__device__ __forceinline__ void repro_signal(unsigned* count, Sync sync) {
    sync();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(count, 1u);
    }
}
template <class Sync>
__device__ __forceinline__ void repro_wait(const unsigned* count, unsigned target, Sync sync) {
    if (threadIdx.x == 0) {
        unsigned seen;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(seen) : "l"(count) : "memory");
        } while (seen < target);
    }
    sync();
}

// The whole block, or group `index` of G taking every G-th k tile.
struct ReproWholeBlock {
    static constexpr int step = 1;
    __device__ __forceinline__ int thread(int) const { return (int)threadIdx.x; }
    __device__ __forceinline__ void sync() const { __syncthreads(); }
};
template <int G>
struct ReproLoopGroup {
    static constexpr int step = G;
    int index;
    __device__ __forceinline__ int thread(int threads) const {
        return (int)threadIdx.x - threads * index;
    }
    __device__ __forceinline__ void sync() const { sync_from<0>(); }
    template <int I>
    __device__ __forceinline__ void sync_from() const {
        if constexpr (I < G) {
            if (index == I)
                GroupSync<I>{}();
            else
                sync_from<I + 1>();
        }
    }
};

// cp_async_16 for single, 4-byte aligned values.
__device__ __forceinline__ void cp_async_4(unsigned target, const void* source) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(target), "l"(source));
}

// The same for 8 bytes, 8-byte aligned.
__device__ __forceinline__ void cp_async_8(unsigned target, const void* source) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(target), "l"(source));
}

// _mm256_max_ps(a, b) and _mm256_min_ps(a, b): b when the comparison fails.
__device__ __forceinline__ float avx_max(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ float avx_min(float a, float b) { return a < b ? a : b; }

__device__ __forceinline__ float abs_bits(float x) {
    return __int_as_float(__float_as_int(x) & 0x7fffffff);
}

// An accumulator's 32 bits: an INT32 dot as it is, a promoted FP32 sum's bits.
__device__ __forceinline__ int repro_bits(int x) { return x; }
__device__ __forceinline__ int repro_bits(float x) { return __float_as_int(x); }

// The low bytes of four ints, a's first.
__device__ __forceinline__ unsigned repro_low_bytes(int a, int b, int c, int d) {
    return __byte_perm(__byte_perm(a, b, 0x0040), __byte_perm(c, d, 0x0040), 0x5410);
}

// The rounding must not depend on the device's approximate reciprocal estimate.
__device__ __forceinline__ float repro_reciprocal(float d) {
    float r;
    asm("rcp.rn.f32 %0, %1;" : "=f"(r) : "f"(d));
    return r;
}

// Counts repro_reciprocal's mismatches with __fdiv_rn(1, d); an atomic a warp.
extern "C" __global__ void repro_reciprocal_check(unsigned first, int n,
                                                  unsigned long long* mismatches) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float d = __uint_as_float(first + (unsigned)i);
    bool differs =
        i < n && __float_as_uint(repro_reciprocal(d)) != __float_as_uint(__fdiv_rn(1.0f, d));
    unsigned count = __popc(__ballot_sync(0xffffffffu, differs));
    if ((threadIdx.x & 31) == 0 && count) atomicAdd(mismatches, (unsigned long long)count);
}

// q = a y with one FMA correction, normal throughout for integer b in [1, 2^21) and a = 0
// or |a| in [2^-60, 2^60]; repro_quotient_check's [1, 2) x [2^20, 2^21) scales to it.
__device__ __forceinline__ float repro_quotient(float a, float b, float y) {
    float q = __fmul_rn(a, y);
    return __fmaf_rn(__fmaf_rn(-b, q, a), y, q);
}

// repro_reciprocal rounds as the division for every normal step (all below
// 2^126); a subnormal step gets 0 and divides.
__device__ __forceinline__ float repro_step_reciprocal(float step) {
    return abs_bits(step) >= 0x1p-126f ? repro_reciprocal(step) : 0.0f;
}

// math::on_step.
__device__ __forceinline__ float repro_on_step(float x, float step, float reciprocal) {
    return reciprocal != 0.0f ? __fmul_rn(x, reciprocal) : __fdiv_rn(x, step);
}

// One step test for all N values: tested value by value, each would sit in its
// own branch beside the division's slow path, serializing the values.
template <int N>
__device__ __forceinline__ void repro_on_steps(const float* x, float step, float reciprocal,
                                               int* q) {
    if (reciprocal != 0.0f) {
#pragma unroll
        for (int i = 0; i < N; ++i) q[i] = __float2int_rn(__fmul_rn(x[i], reciprocal));
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) q[i] = __float2int_rn(__fdiv_rn(x[i], step));
    }
}

// Counts repro_quotient's mismatches with __fdiv_rn; an atomic a warp.
extern "C" __global__ void repro_quotient_check(int b0, int nb, unsigned long long* mismatches) {
    float a = __uint_as_float(0x3f800000u + blockIdx.x * blockDim.x + threadIdx.x);
    unsigned count = 0;
    for (int j = 0; j < nb; ++j) {
        float b = __int2float_rn(b0 + j);
        bool differs = __float_as_uint(repro_quotient(a, b, repro_reciprocal(b))) !=
                       __float_as_uint(__fdiv_rn(a, b));
        count += __popc(__ballot_sync(0xffffffffu, differs));
    }
    if ((threadIdx.x & 31) == 0 && count) atomicAdd(mismatches, (unsigned long long)count);
}

// Loaded once for a run of values.
struct ReproGelu {
    float frac_1_sqrt_2, a[6], cap;
};
__device__ __forceinline__ ReproGelu repro_gelu_constants(const float* fc) {
    ReproGelu g;
    g.frac_1_sqrt_2 = fc[FC_FRAC_1_SQRT_2];
#pragma unroll
    for (int i = 0; i < 6; ++i) g.a[i] = fc[FC_GELU_A + i];
    g.cap = fc[FC_GELU_CAP];
    return g;
}

// math::avx2::gelu, each step over every value before the next: each
// reciprocal has its own slow-path branch, so chains overlap only between steps.
template <int R, int V>
__device__ __forceinline__ void repro_gelus(float (&x)[R][V], const ReproGelu& g) {
    float d[R][V];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < V; ++j) {
            const float z = abs_bits(__fmul_rn(x[r][j], g.frac_1_sqrt_2));
            float p = g.a[5];
#pragma unroll
            for (int i = 4; i >= 0; --i) p = __fmaf_rn(p, z, g.a[i]);
            p = __fmaf_rn(p, z, 1.0f);
            const float p2 = __fmul_rn(p, p), p4 = __fmul_rn(p2, p2), p8 = __fmul_rn(p4, p4);
            d[r][j] = avx_min(__fmul_rn(p8, p8), g.cap);
        }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < V; ++j) d[r][j] = repro_reciprocal(d[r][j]);
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < V; ++j)
            x[r][j] = __fmul_rn(__fmul_rn(0.5f, x[r][j]), x[r][j] >= 0.0f ? __fsub_rn(2.0f, d[r][j]) : d[r][j]);
}
__device__ __forceinline__ float repro_gelu(float x, const ReproGelu& g) {
    float v[1][1] = {{x}};
    repro_gelus(v, g);
    return v[0][0];
}
__device__ float repro_gelu(float x, const float* fc) { return repro_gelu(x, repro_gelu_constants(fc)); }

// out[i] = GELU of the float whose bits are first + i.
extern "C" __global__ void repro_gelus(unsigned first, int n, const float* fc, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = repro_gelu(__uint_as_float(first + (unsigned)i), fc);
}

// FP32 rows round once to FP16; FP16 rows already are.
__device__ __forceinline__ float fp16_value(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float fp16_value(__half x) { return __half2float(x); }

// The integer pipe avoids GA10x's FP64 conversion; values are integers below
// 2^40 in 2^-24 units, so sums of up to 2^13 stay exact in any order.
__device__ __forceinline__ long long fp16_units(__half x) {
    const unsigned bits = __half_as_ushort(x), e = (bits >> 10) & 0x1fu, m = bits & 0x3ffu;
    const long long units = e ? (long long)(1024u + m) << (e - 1) : (long long)m;
    return (bits & 0x8000u) ? -units : units;
}

// Counts fp16_units' mismatches with the product by 2^24; an atomic a warp.
extern "C" __global__ void repro_units_check(unsigned first, int n, unsigned long long* mismatches) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned short bits = (unsigned short)(first + (unsigned)i);
    __half h = __ushort_as_half(bits);
    bool finite = (bits & 0x7c00u) != 0x7c00u;
    bool differs = i < n && finite &&
                   fp16_units(h) != __float2ll_rn(__fmul_rn(__half2float(h), 16777216.0f));
    unsigned count = __popc(__ballot_sync(0xffffffffu, differs));
    if ((threadIdx.x & 31) == 0 && count) atomicAdd(mismatches, (unsigned long long)count);
}

// Norm::apply's statistics: per-lane sums halved by xor shuffles in
// encoder.rs lane_sum's order, so every lane holds the CPU's total. `value(i)`
// is the lane's i-th value of the row, read once a pass.
struct RowMoments {
    float mean, rstd;
    bool finite;
};
template <class Value>
__device__ __forceinline__ RowMoments repro_row_moments_of(const Value& value, float eps, int h) {
    const float n = (float)h;
    auto halved = [](float part) {
#pragma unroll
        for (int w = 16; w > 0; w >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xffffffffu, part, w));
        return part;
    };
    float part = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i)
        if (32 * i < h) part = __fadd_rn(part, value(i));
    const float mean = __fdiv_rn(halved(part), n);
    part = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i)
        if (32 * i < h) {
            const float d = __fsub_rn(value(i), mean);
            part = __fadd_rn(part, __fmul_rn(d, d));
        }
    const float variance = __fdiv_rn(halved(part), n);
    const float denominator = __fadd_rn(variance, eps);
    const bool finite = isfinite(mean) && isfinite(denominator);
    return {mean, finite ? __fdiv_rn(1.0f, __fsqrt_rn(denominator)) : 0.0f, finite};
}
__device__ __forceinline__ RowMoments repro_row_moments(const float (&x)[32], float eps, int h) {
    return repro_row_moments_of([&](int i) { return x[i]; }, eps, h);
}

// Norm::apply's value for x (NaN on a row whose statistics are not finite).
__device__ __forceinline__ float repro_normalized(float x, const RowMoments& m, float gamma,
                                                  float beta) {
    return m.finite ? __fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(x, m.mean), m.rstd), gamma), beta)
                    : __int_as_float(0x7fc00000);
}

// encoder.rs Norm::apply_rows_keeping: mean and rstd, both NaN when not finite.
__device__ __forceinline__ float2 repro_deferred_moments(const RowMoments& m) {
    const float nan = __int_as_float(0x7fc00000);
    return m.finite ? make_float2(m.mean, m.rstd) : make_float2(nan, nan);
}

// The row test is the warp's 32 i < h: a lane's own test would put each of the
// 32 stores in its own branch.
template <class Stage>
__device__ void repro_layernorm_lanes_on(const float (&x)[32], const RowMoments& m, __half* out,
                                         const Stage& stage, const float* gamma, const float* beta,
                                         int h) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < 32; ++i)
        if (32 * i < h) {
            const int c = lane + 32 * i;
            const __half value = __float2half_rn(repro_normalized(x[i], m, gamma[c], beta[c]));
            out[c] = value;
            stage(c, value);
        }
}

struct ReproStageHalves {
    __half* row;
    __device__ __forceinline__ void operator()(int c, __half value) const { row[c] = value; }
};

// Four rows spread a short input over more multiprocessors; a batch's blocks
// take eight, staging the parameters for twice the rows.
#define NORM_ROWS 4
#define NORM_BATCH_ROWS 8

// repro.rs norm_bytes mirrors this layout.
struct NormShared {
    __half* rows;
    float* params;
};
template <int ROWS>
__device__ __forceinline__ NormShared norm_shared(void* base, int h) {
    __half* rows = static_cast<__half*>(base);
    return {rows, reinterpret_cast<float*>(rows + ROWS * h)};
}

// Staged, gamma and beta need no loads the compiler cannot hoist above aliasing FP16
// stores. The caller waits for these copies before its barrier.
__device__ __forceinline__ void repro_stage_norm(float* params, int h, const float* gamma,
                                                 const float* beta) {
    for (int i = threadIdx.x; i < h / 4; i += blockDim.x) {
        cp_async_16(shared_address(params + 4 * i), gamma + 4 * i, 16);
        cp_async_16(shared_address(params + h + 4 * i), beta + 4 * i, 16);
    }
    cp_async_commit();
}

// The input staging clears this for the first layer, a second LayerNorm for
// the next, before QKV takes maxima and flags into it.
struct AttentionClear {
    unsigned* value_magnitudes;
    int* failed;
    int sequences, heads;
};

__device__ __forceinline__ void repro_attention_clear(const AttentionClear& clear, int hidden,
                                                      int at, int threads) {
    const int count = clear.sequences * hidden;
    for (int i = at; i < count; i += threads) {
        const int s = i / hidden, c = i % hidden;
        clear.value_magnitudes[i] = 0;
        if (c < clear.heads) clear.failed[(size_t)s * clear.heads + c] = 0;
    }
}

// Declared before the template that calls it: nvcc on Linux applies C++ two-phase
// lookup to that non-dependent call.
__device__ __forceinline__ void repro_stage_row32(const __half* source, const float* gamma,
                                                  const float4 (&raw_center)[8], const float4 (&raw_signs)[8],
                                                  bool quantizes, int r, int h, int rows, const float* fc,
                                                  signed char* quantized, float* steps, int digits);

// LayerNorm and the next input stage in one launch; the sign mask forms after the norm,
// so every load is in flight before any value is used.
template <int ROWS>
__device__ __forceinline__ void repro_layernorm_quantize_rows(
    void* shared, const __half* rows, __half* out, const float* gamma, const float* beta, float eps,
    int h, int tokens, const int* row_codes, const float* centers, const float* signs,
    const float* fc, signed char* quantized, float* steps, float2* moments, int digits) {
    const NormShared at = norm_shared<ROWS>(shared, h);
    const int first = blockIdx.x * ROWS, count = min(ROWS, tokens - first);
    repro_stage_norm(at.params, h, gamma, beta);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c0 = 32 * lane, r = first + warp;
    const bool live = warp < count, quantizes = live && c0 < h;
    const __half* row = rows + (size_t)r * h;
    float x[32];
    if (live) {
#pragma unroll
        for (int i = 0; i < 32; ++i)
            if (32 * i < h) x[i] = __half2float(row[lane + 32 * i]);
    }
    float4 raw_center[8], raw_signs[8];
    if (quantizes) {
        const float* row = centers + (size_t)(-row_codes[r] - 1) * h + c0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            raw_center[i] = *reinterpret_cast<const float4*>(row + 4 * i);
            raw_signs[i] = *reinterpret_cast<const float4*>(signs + c0 + 4 * i);
        }
    }
    cp_async_wait<0>();
    __syncthreads();
    if (!live) return;
    __half* staged = at.rows + warp * h;
    const RowMoments m = repro_row_moments(x, eps, h);
    repro_layernorm_lanes_on(x, m, out + (size_t)r * h, ReproStageHalves{staged}, at.params, at.params + h, h);
    if (moments && lane == 0) moments[r] = repro_deferred_moments(m);
    // A deferred stage takes the raw values times gamma.
    repro_stage_row32(moments ? row : staged, moments ? at.params : nullptr, raw_center, raw_signs, quantizes, r,
                      h, tokens, fc, quantized, steps, digits);
}

extern "C" __global__ void __launch_bounds__(NORM_BATCH_ROWS * 32)
    repro_layernorm_quantize_batch(const __half* rows, __half* out, const float* gamma,
                                   const float* beta, float eps, int h, int tokens,
                                   const int* row_codes, const float* centers, const float* signs,
                                   const float* fc, signed char* quantized, float* steps,
                                   float2* moments, unsigned* clear_magnitudes,
                                   int* clear_failed, int clear_sequences, int heads, int digits) {
    extern __shared__ float4 shared[];
    repro_attention_clear(AttentionClear{clear_magnitudes, clear_failed, clear_sequences, heads},
                          h, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x);
    repro_layernorm_quantize_rows<NORM_BATCH_ROWS>(shared, rows, out, gamma, beta, eps, h, tokens,
                                                   row_codes, centers, signs, fc, quantized, steps,
                                                   moments, digits);
}

// Each encode starts its sums from zero; readers then clear what they read.
extern "C" __global__ void repro_clear(long long* values, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        values[i] = 0;
}

#define READ_AHEAD_WARPS 2
#define READ_AHEAD_DEPTH 4

// Reading one byte brings the whole 32-byte sector into L2. Limit outstanding loads
// so prefetch traffic leaves bandwidth for the running layer.
__device__ __forceinline__ unsigned repro_read_digits(const signed char* digits, int bytes) {
    const int readers = gridDim.x * READ_AHEAD_WARPS;
    const int reader = blockIdx.x * READ_AHEAD_WARPS + (threadIdx.x >> 5);
    const size_t piece = 32 * 32, stride = readers * piece;
    const unsigned char* first = reinterpret_cast<const unsigned char*>(digits);
    const unsigned char* at = first + reader * piece + (threadIdx.x & 31) * 32;
    const unsigned char* end = first + bytes;
    unsigned sum = 0;
    for (; at + (READ_AHEAD_DEPTH - 1) * stride < end; at += READ_AHEAD_DEPTH * stride) {
        unsigned round = 0;
#pragma unroll
        for (int d = 0; d < READ_AHEAD_DEPTH; ++d) round += __ldcg(at + d * stride);
        sum += round;
    }
    for (; at < end; at += stride) sum += __ldcg(at);
    return sum;
}

// Store the checksum so the compiler keeps the prefetch loads and tests can verify their coverage.
extern "C" __global__ void repro_read_ahead(const signed char* qkv, int qkv_bytes, const signed char* out,
                                            int out_bytes, const signed char* up, int up_bytes,
                                            const signed char* down, int down_bytes, unsigned* sums) {
    unsigned sum = repro_read_digits(qkv, qkv_bytes);
    sum += repro_read_digits(out, out_bytes);
    sum += repro_read_digits(up, up_bytes);
    sum += repro_read_digits(down, down_bytes);
    sums[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

// Staged by the encode's first launch, so no copy engine runs ahead of it.
__device__ __forceinline__ void repro_stage_batch(const int4* __restrict__ host, int4* __restrict__ device,
                                                  int vectors, const AttentionClear& clear, int hidden, int at,
                                                  int threads) {
    for (int i = at; i < vectors; i += threads) device[i] = host[i];
    repro_attention_clear(clear, hidden, at, threads);
}

// repro_stage_batch alone, for the checks that run no embedding.
extern "C" __global__ void repro_stage_inputs(const int4* __restrict__ host, int4* __restrict__ device,
                                              int vectors, unsigned* clear_magnitudes, int* clear_failed,
                                              int clear_sequences, int hidden, int heads) {
    repro_stage_batch(host, device, vectors, AttentionClear{clear_magnitudes, clear_failed, clear_sequences, heads},
                      hidden, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x);
}

// The lower keeps u + w, the upper u - w.
__device__ __forceinline__ void butterfly(float& lower, float& upper) {
    float u = lower, w = upper;
    lower = __fadd_rn(u, w);
    upper = __fsub_rn(u, w);
}

// H128 stages 1-16 in the lane and 32, 64 across a block's four lanes, in the
// CPU's pair order, so its bits; every lane takes part in the shuffles. Returns the
// largest magnitude of the lane's half block, whose step it sets.
__device__ __forceinline__ float repro_rotate_x32(float (&x)[32], int part, float rotate) {
#pragma unroll
    for (int bit = 1; bit < 32; bit <<= 1)
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (i & bit) continue;
            butterfly(x[i], x[i | bit]);
        }
#pragma unroll
    for (int delta = 1; delta <= 2; delta <<= 1) {
        const bool upper = part & delta;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const float other = __shfl_xor_sync(0xffffffffu, x[i], delta);
            x[i] = upper ? __fsub_rn(other, x[i]) : __fadd_rn(x[i], other);
        }
    }
    float most = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        x[i] = __fmul_rn(x[i], rotate);
        most = fmaxf(most, abs_bits(x[i]));
    }
    return fmaxf(most, __shfl_xor_sync(0xffffffffu, most, 1));
}

// The step of an operand's second digits: 127 of them span the half step a first digit leaves.
__device__ __forceinline__ float repro_second_step(float step) { return __fdiv_rn(step, 254.0f); }

// prepared/avx2.rs quantize_half_twice: what N digits leave of their values, one FMA each,
// in digits on the second step.
template <int N>
__device__ __forceinline__ void repro_second_digits(const float* x, const int* first, float step, int* q) {
    const float second = repro_second_step(step);
    float left[N];
#pragma unroll
    for (int i = 0; i < N; ++i) left[i] = __fmaf_rn(-__int2float_rn(first[i]), step, x[i]);
    repro_on_steps<N>(left, second, repro_step_reciprocal(second), q);
}

// V bytes in stores of up to 16.
template <int V>
__device__ __forceinline__ void repro_store_digits(const int (&b)[V], signed char* q) {
    unsigned word[V / 4];
#pragma unroll
    for (int w = 0; w < V / 4; ++w) word[w] = repro_low_bytes(b[4 * w], b[4 * w + 1], b[4 * w + 2], b[4 * w + 3]);
    if constexpr (V == 4) {
        *reinterpret_cast<unsigned*>(q) = word[0];
    } else if constexpr (V == 8) {
        *reinterpret_cast<uint2*>(q) = make_uint2(word[0], word[1]);
    } else {
#pragma unroll
        for (int w = 0; w < V / 4; w += 4)
            *reinterpret_cast<uint4*>(q + 4 * w) = make_uint4(word[w], word[w + 1], word[w + 2], word[w + 3]);
    }
}

// A lane's V values in digits on their half block's step; `behind` takes their second digits.
template <int V>
__device__ __forceinline__ void repro_store_block(const float (&x)[V], float scale, float reciprocal,
                                                  signed char* q, signed char* behind = nullptr) {
    int b[V];
    repro_on_steps<V>(x, scale, reciprocal, b);
    repro_store_digits<V>(b, q);
    if (behind) {
        int rest[V];
        repro_second_digits<V>(x, b, scale, rest);
        repro_store_digits<V>(rest, behind);
    }
}
template <int V>
__device__ __forceinline__ void repro_store_block(const float (&x)[V], float scale, signed char* q,
                                                  signed char* behind = nullptr) {
    repro_store_block<V>(x, scale, repro_step_reciprocal(scale), q, behind);
}

// Lanes past h / 32 take part in the shuffles and store nothing. A row of `digits` digits
// holds its second digits behind its first, and their steps behind the first's.
__device__ __forceinline__ void repro_stage_row32(const __half* source, const float* gamma,
                                                  const float4 (&raw_center)[8], const float4 (&raw_signs)[8],
                                                  bool quantizes, int r, int h, int rows, const float* fc,
                                                  signed char* quantized, float* steps, int digits) {
    const int lane = threadIdx.x & 31, c0 = 32 * lane;
    float center[32];
    unsigned negative = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 m = raw_center[i], sg = raw_signs[i];
        const int at_i = 4 * i;
        center[at_i] = m.x, center[at_i + 1] = m.y, center[at_i + 2] = m.z, center[at_i + 3] = m.w;
        negative |= (sg.x < 0.0f ? 1u : 0u) << at_i | (sg.y < 0.0f ? 1u : 0u) << (at_i + 1) |
                    (sg.z < 0.0f ? 1u : 0u) << (at_i + 2) | (sg.w < 0.0f ? 1u : 0u) << (at_i + 3);
    }
    __syncwarp();
    float v[32];
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
        const uint4 words = quantizes ? *reinterpret_cast<const uint4*>(source + c0 + i)
                                      : make_uint4(0, 0, 0, 0);
        const unsigned w[4] = {words.x, words.y, words.z, words.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            float2 pair = __half22float2(*reinterpret_cast<const __half2*>(&w[e]));
            const int at_i = i + 2 * e;
            if (gamma && quantizes) {
                pair.x = __fmul_rn(pair.x, gamma[c0 + at_i]);
                pair.y = __fmul_rn(pair.y, gamma[c0 + at_i + 1]);
            }
            v[at_i] = __fmul_rn(__fsub_rn(pair.x, quantizes ? center[at_i] : 0.0f),
                                (negative >> at_i) & 1 ? -1.0f : 1.0f);
            v[at_i + 1] = __fmul_rn(__fsub_rn(pair.y, quantizes ? center[at_i + 1] : 0.0f),
                                    (negative >> (at_i + 1)) & 1 ? -1.0f : 1.0f);
        }
    }
    const float most = repro_rotate_x32(v, lane % 4, fc[FC_RECIP_SQRT_BLOCK]);
    const float scale = most > 0.0f ? __fdiv_rn(most, 127.0f) : 1.0f;
    if (!quantizes) return;
    signed char* q = quantized + (size_t)r * digits * h + c0;
    repro_store_block<32>(v, scale, q, digits == 2 ? q + h : nullptr);
    if (lane % 2 == 0) {
        *repro_step_at(steps, rows, lane / 2, r) = scale;
        if (digits == 2) *repro_step_at(steps, rows, h / HALF + lane / 2, r) = repro_second_step(scale);
    }
}

// The first launch also stages the batch, and no block's copy is ordered before the
// embedding, so it reads its tokens from the page-locked inputs.
extern "C" __global__ void __launch_bounds__(NORM_ROWS * 32)
    repro_embed(const int4* __restrict__ host, int4* __restrict__ device, int vectors,
                unsigned* clear_magnitudes, int* clear_failed, int clear_sequences, int heads,
                const int* ids, const int* types, const int* positions, const float* word,
                const float* position, const float* token_type, const float* gamma,
                const float* beta, float eps, int h, int tokens, __half* out, const int* row_codes,
                const float* centers, const float* signs, const float* fc, signed char* quantized,
                float* steps) {
    extern __shared__ float4 shared[];
    const NormShared at = norm_shared<NORM_ROWS>(shared, h);
    const int first = blockIdx.x * NORM_ROWS, rows = min(NORM_ROWS, tokens - first);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c0 = 32 * lane;
    const int token = first + warp;
    const bool live = warp < rows, quantizes = live && c0 < h;
    int id = 0, type = 0, place = 0, code = 0;
    if (live) id = ids[token], type = types[token], place = positions[token], code = row_codes[token];
    repro_stage_batch(host, device, vectors, AttentionClear{clear_magnitudes, clear_failed, clear_sequences, heads},
                      h, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x);
    repro_stage_norm(at.params, h, gamma, beta);
    float x[32];
    if (live) {
        const float* w = word + (size_t)id * h;
        const float* s = token_type + (size_t)type * h;
        const float* p = position + (size_t)place * h;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int c = lane + 32 * i;
            if (32 * i < h) x[i] = __fadd_rn(__fadd_rn(w[c], s[c]), p[c]);
        }
    }
    float4 raw_center[8], raw_signs[8];
    if (quantizes) {
        const float* row = centers + (size_t)(-code - 1) * h + c0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            raw_center[i] = *reinterpret_cast<const float4*>(row + 4 * i);
            raw_signs[i] = *reinterpret_cast<const float4*>(signs + c0 + 4 * i);
        }
    }
    cp_async_wait<0>();
    __syncthreads();
    if (!live) return;
    __half* staged = at.rows + warp * h;
    repro_layernorm_lanes_on(x, repro_row_moments(x, eps, h), out + (size_t)token * h, ReproStageHalves{staged},
                             at.params, at.params + h, h);
    repro_stage_row32(staged, nullptr, raw_center, raw_signs, quantizes, token, h, tokens, fc, quantized, steps, 1);
}

// repro_rotate_x32's pairs in its order, each stage over every row first so
// the rows' shuffles overlap; own * -1 is exact, so the FMA rounds once.
template <int V, int R>
__device__ __forceinline__ void repro_rotate_rows(float (&x)[R][V], int part, float rotate, float (&most)[R]) {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int bit = 1; bit < V; bit <<= 1)
#pragma unroll
            for (int i = 0; i < V; ++i) {
                if (i & bit) continue;
                butterfly(x[r][i], x[r][i | bit]);
            }
#pragma unroll
    for (int delta = 1; delta < BLOCK / V; delta <<= 1) {
        const float sign = part & delta ? -1.0f : 1.0f;
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int i = 0; i < V; ++i)
                x[r][i] = __fmaf_rn(x[r][i], sign, __shfl_xor_sync(0xffffffffu, x[r][i], delta));
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        most[r] = 0.0f;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            x[r][i] = __fmul_rn(x[r][i], rotate);
            most[r] = fmaxf(most[r], abs_bits(x[r][i]));
        }
    }
    // The largest magnitude of the lane's half block, whose step it sets.
#pragma unroll
    for (int delta = 1; delta < HALF / V; delta <<= 1)
#pragma unroll
        for (int r = 0; r < R; ++r) most[r] = fmaxf(most[r], __shfl_xor_sync(0xffffffffu, most[r], delta));
}

template <int V>
__device__ __forceinline__ float repro_rotate_block(float (&x)[V], int part, float rotate) {
    float row[1][V], most[1];
#pragma unroll
    for (int i = 0; i < V; ++i) row[0][i] = x[i];
    repro_rotate_rows<V, 1>(row, part, rotate, most);
#pragma unroll
    for (int i = 0; i < V; ++i) x[i] = row[0][i];
    return most[0];
}

// repro.rs NORM_SPLIT_WARPS.
#define NORM_SPLIT_WARPS 8

// Values load before the block's barrier and form after it.
struct ReproHalfRow {
    static constexpr bool STAGED = false;
    const __half* row;
    using Raw = uint2;
    // A block's copy of its row in shared memory.
    __device__ __forceinline__ static __half* copy() {
        __shared__ __align__(16) __half copied[32 * 32];
        return copied;
    }
    __device__ __forceinline__ float value(int c) const { return __half2float(row[c]); }
    __device__ __forceinline__ Raw load4(int c) const { return *reinterpret_cast<const uint2*>(row + c); }
    __device__ __forceinline__ void form4(const Raw& raw, float (&in)[4]) const {
        const float2 low = __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
        const float2 high = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
        in[0] = low.x, in[1] = low.y, in[2] = high.x, in[3] = high.y;
    }
};
struct ReproHalfRows {
    const __half* rows;
    __device__ __forceinline__ ReproHalfRow row(int r, int h) const { return {rows + (size_t)r * h}; }
};

// Segments join in order, ((s0 + s1) + s2) + ..., as the residual epilogue
// would; staging spreads the seven loads a value across warps.
template <int SEGS>
struct ReproJoinedRow {
    static constexpr bool STAGED = true;
    const float* sums;
    size_t plane;
    const float* scales;
    const float* shared;
    const __half* residual;
    struct Raw {
        float4 sums[SEGS], scales, shared;
        uint2 residual;
    };
    __device__ __forceinline__ float joined(float total, float scale, float shared_value, float residual_value) const {
        return __half2float(
            __float2half_rn(__fadd_rn(__fadd_rn(__fmul_rn(total, scale), shared_value), residual_value)));
    }
    __device__ __forceinline__ Raw load4(int c) const {
        Raw raw;
#pragma unroll
        for (int s = 0; s < SEGS; ++s) raw.sums[s] = *reinterpret_cast<const float4*>(sums + s * plane + c);
        raw.scales = *reinterpret_cast<const float4*>(scales + c);
        raw.shared = *reinterpret_cast<const float4*>(shared + c);
        raw.residual = *reinterpret_cast<const uint2*>(residual + c);
        return raw;
    }
    __device__ __forceinline__ void form4(const Raw& raw, float (&in)[4]) const {
        float total[4] = {raw.sums[0].x, raw.sums[0].y, raw.sums[0].z, raw.sums[0].w};
#pragma unroll
        for (int s = 1; s < SEGS; ++s) {
            total[0] = __fadd_rn(total[0], raw.sums[s].x);
            total[1] = __fadd_rn(total[1], raw.sums[s].y);
            total[2] = __fadd_rn(total[2], raw.sums[s].z);
            total[3] = __fadd_rn(total[3], raw.sums[s].w);
        }
        const float2 low = __half22float2(*reinterpret_cast<const __half2*>(&raw.residual.x));
        const float2 high = __half22float2(*reinterpret_cast<const __half2*>(&raw.residual.y));
        in[0] = joined(total[0], raw.scales.x, raw.shared.x, low.x);
        in[1] = joined(total[1], raw.scales.y, raw.shared.y, low.y);
        in[2] = joined(total[2], raw.scales.z, raw.shared.z, high.x);
        in[3] = joined(total[3], raw.scales.w, raw.shared.w, high.y);
    }
};
template <int SEGS>
struct ReproJoinedRows {
    const float* partials;
    int rows;
    const float* col_scales;
    const float* statics;
    const int* row_codes;
    const __half* residual;
    __device__ __forceinline__ ReproJoinedRow<SEGS> row(int r, int h) const {
        return {partials + (size_t)r * h, (size_t)rows * h, col_scales, statics + (size_t)(-row_codes[r] - 1) * h,
                residual + (size_t)r * h};
    }
};

// repro_layernorm_quantize_batch's values with a row split over blocks; warp 0's
// statistics meet every warp's 128-block in shared memory. The grid is a row wide.
template <class Rows, int WARPS = NORM_SPLIT_WARPS>
__device__ __forceinline__ void repro_layernorm_quantize_from(
    const Rows& rows, __half* out, const float* gamma, const float* beta, float eps, int h, const int* row_codes,
    const float* centers, const float* signs, const float* fc, signed char* quantized, float* steps,
    float2* moments, unsigned* clear_magnitudes, int* clear_failed, int clear_sequences, int heads, int digits) {
    __shared__ RowMoments row_moments;
    const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = WARPS * blockIdx.y + warp, c = BLOCK * b + 4 * lane;
    const bool live = b < h / BLOCK;
    const auto row = rows.row(r, h);
    constexpr bool STAGED = decltype(row)::STAGED;
    // A plain row is copied into shared memory and read from there: held in the first warp's
    // registers through the statistics, it would cut the blocks a multiprocessor seats.
    if constexpr (!STAGED) {
        if (warp == 0) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int at = 256 * j + 8 * lane;
                cp_async_16(shared_address(row.copy() + at), row.row + at, at < h ? 16 : 0);
            }
            cp_async_commit();
        }
    }
    [[maybe_unused]] float x[32];
    [[maybe_unused]] typename decltype(row)::Raw raw = {};
    float4 g = {}, be = {}, m = {}, sg = {};
    if (live) {
        if constexpr (STAGED) raw = row.load4(c);
        g = *reinterpret_cast<const float4*>(gamma + c);
        be = *reinterpret_cast<const float4*>(beta + c);
        m = *reinterpret_cast<const float4*>(centers + (size_t)(-row_codes[r] - 1) * h + c);
        sg = *reinterpret_cast<const float4*>(signs + c);
    }
    repro_attention_clear(AttentionClear{clear_magnitudes, clear_failed, clear_sequences, heads},
                          h, (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x,
                          gridDim.x * gridDim.y * blockDim.x);
    const float rotate = fc[FC_RECIP_SQRT_BLOCK];
    float in[4];
    if constexpr (STAGED) {
        __shared__ __align__(16) float formed[WARPS * BLOCK];
        if (live) {
            row.form4(raw, in);
            *reinterpret_cast<float4*>(formed + c) = make_float4(in[0], in[1], in[2], in[3]);
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int i = 0; i < 32; ++i)
                if (32 * i < h) x[i] = formed[lane + 32 * i];
        }
    }
    if (warp == 0) {
        RowMoments rm;
        if constexpr (STAGED) {
            rm = repro_row_moments(x, eps, h);
        } else {
            cp_async_wait<0>();
            // Volatile, so each pass reads the copy: kept from the first pass, the row
            // would hold 32 registers through the second.
            const volatile __half* copied = row.copy();
            rm = repro_row_moments_of(
                [&](int i) {
                    const __half value = copied[lane + 32 * i];
                    return __half2float(value);
                },
                eps, h);
        }
        if (lane == 0) {
            row_moments = rm;
            if (moments && blockIdx.y == 0) moments[r] = repro_deferred_moments(rm);
        }
    }
    __syncthreads();
    if (!live) return;
    const RowMoments rm = row_moments;
    if constexpr (!STAGED) row.form4(*reinterpret_cast<const uint2*>(row.copy() + c), in);
    const float gs[4] = {g.x, g.y, g.z, g.w}, bs[4] = {be.x, be.y, be.z, be.w};
    const float ms[4] = {m.x, m.y, m.z, m.w}, ss[4] = {sg.x, sg.y, sg.z, sg.w};
    __half value[4];
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        value[i] = __float2half_rn(repro_normalized(in[i], rm, gs[i], bs[i]));
        // The deferred stage's raw value times gamma, else the normalized one.
        const float x = moments ? __fmul_rn(in[i], gs[i]) : __half2float(value[i]);
        v[i] = __fmul_rn(__fsub_rn(x, ms[i]), ss[i] < 0.0f ? -1.0f : 1.0f);
    }
    const __half2 first = __halves2half2(value[0], value[1]), second = __halves2half2(value[2], value[3]);
    *reinterpret_cast<uint2*>(out + (size_t)r * h + c) =
        make_uint2(*reinterpret_cast<const unsigned*>(&first), *reinterpret_cast<const unsigned*>(&second));
    const float most = repro_rotate_block<4>(v, lane, rotate);
    const float scale = most > 0.0f ? __fdiv_rn(most, 127.0f) : 1.0f;
    signed char* q = quantized + (size_t)r * digits * h + c;
    repro_store_block<4>(v, scale, q, digits == 2 ? q + h : nullptr);
    if (lane % 16 == 0) {
        const int part = 2 * b + lane / 16, tokens = gridDim.x;
        *repro_step_at(steps, tokens, part, r) = scale;
        if (digits == 2) *repro_step_at(steps, tokens, h / HALF + part, r) = repro_second_step(scale);
    }
}

extern "C" __global__ void __launch_bounds__(NORM_SPLIT_WARPS * 32)
    repro_layernorm_quantize(const __half* rows, __half* out, const float* gamma,
                             const float* beta, float eps, int h, const int* row_codes,
                             const float* centers, const float* signs, const float* fc,
                             signed char* quantized, float* steps, float2* moments,
                             unsigned* clear_magnitudes, int* clear_failed, int clear_sequences,
                             int heads, int digits) {
    repro_layernorm_quantize_from(ReproHalfRows{rows}, out, gamma, beta, eps, h, row_codes, centers, signs, fc,
                                  quantized, steps, moments, clear_magnitudes, clear_failed, clear_sequences,
                                  heads, digits);
}

// Written over the residual in place: a row stays in one block, so every read
// precedes its barrier and every write follows it.
#define REPRO_LAYERNORM_QUANTIZE_JOINED(WARPS)                                                                   \
    extern "C" __global__ void __launch_bounds__(WARPS * 32) repro_layernorm_quantize_joined_##WARPS(          \
        const float* partials, int rows, const float* col_scales, const float* statics, __half* stream,         \
        const float* gamma, const float* beta, float eps, int h, const int* row_codes, const float* centers,    \
        const float* signs, const float* fc, signed char* quantized, float* steps, float2* moments,             \
        unsigned* clear_magnitudes, int* clear_failed, int clear_sequences, int heads, int digits) {            \
        if (gridDim.y != 1 || (int)gridDim.x != rows) __trap();                                                  \
        repro_layernorm_quantize_from<ReproJoinedRows<4>, WARPS>(                                                \
            ReproJoinedRows<4>{partials, rows, col_scales, statics, row_codes, stream}, stream, gamma, beta, eps, \
            h, row_codes, centers, signs, fc, quantized, steps, moments, clear_magnitudes, clear_failed,         \
            clear_sequences, heads, digits);                                                                     \
    }
REPRO_LAYERNORM_QUANTIZE_JOINED(8)
#undef REPRO_LAYERNORM_QUANTIZE_JOINED

// Each sign is exactly +1 or -1, so a bit mask holds it losslessly.
__device__ __forceinline__ unsigned repro_sign_mask32(const float* signs) {
    unsigned negative = 0;
#pragma unroll
    for (int i = 0; i < 32; i += 4) {
        const float4 s = *reinterpret_cast<const float4*>(signs + i);
        negative |= (s.x < 0.0f ? 1u : 0u) << i | (s.y < 0.0f ? 1u : 0u) << (i + 1) |
                    (s.z < 0.0f ? 1u : 0u) << (i + 2) | (s.w < 0.0f ? 1u : 0u) << (i + 3);
    }
    return negative;
}

__device__ __forceinline__ void repro_centered_x32(const uint4 (&raw)[4], const float (&mean)[32],
                                                   unsigned negative, float (&x)[32]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned words[4] = {raw[i].x, raw[i].y, raw[i].z, raw[i].w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 pair = __half22float2(*reinterpret_cast<const __half2*>(&words[e]));
            const int v = 8 * i + 2 * e;
            x[v] = __fmul_rn(__fsub_rn(pair.x, mean[v]), (negative >> v) & 1 ? -1.0f : 1.0f);
            x[v + 1] = __fmul_rn(__fsub_rn(pair.y, mean[v + 1]), (negative >> (v + 1)) & 1 ? -1.0f : 1.0f);
        }
    }
}

// prepared/avx2.rs prepare with two shuffles a value.
extern "C" __global__ void repro_quantize_half_blocks(const __half* __restrict__ input,
                                                      const int* __restrict__ classes,
                                                      const float* __restrict__ centers,
                                                      const int* __restrict__ group_of_row,
                                                      const float* __restrict__ signs,
                                                      const float* __restrict__ fc, int k, int rows,
                                                      signed char* __restrict__ q,
                                                      float* __restrict__ steps) {
    const int lanes = k / 32, slots = blockDim.x / lanes;
    const int t = threadIdx.x, slot = t / lanes, j = t % lanes;
    const int run = (rows + gridDim.x - 1) / gridDim.x;
    const int r0 = blockIdx.x * run, end = min(rows, r0 + run);
    if (r0 >= end) return;
    const int passes = (end - r0 + slots - 1) / slots;
    const int c0 = 32 * j;
    const unsigned negative = repro_sign_mask32(signs + c0);
    const float rotate = fc[FC_RECIP_SQRT_BLOCK];
    auto load_row = [&](int r, uint4 (&raw)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            raw[i] = *reinterpret_cast<const uint4*>(input + (size_t)r * k + c0 + 8 * i);
    };
    float mean[32] = {};
    int group = -1;
    auto load_mean = [&](int g) {
        group = g;
        const float* row = centers + (size_t)(classes[g] - 1) * k;
#pragma unroll
        for (int i = 0; i < 32; i += 4) {
            const float4 m = *reinterpret_cast<const float4*>(row + c0 + i);
            mean[i] = m.x, mean[i + 1] = m.y, mean[i + 2] = m.z, mean[i + 3] = m.w;
        }
    };
    int r = r0 + slot;
    uint4 raw[4] = {};
    if (slot < slots && r < end) {
        load_row(r, raw);
        load_mean(group_of_row[r]);
    }
    for (int pass = 0; pass < passes; ++pass, r += slots) {
        const bool live = slot < slots && r < end;
        float x[32];
        repro_centered_x32(raw, mean, negative, x);
        const int next = r + slots;
        if (live && next < end) {
            load_row(next, raw);
            const int g = group_of_row[next];
            if (g != group) load_mean(g);
        }
        const float most = repro_rotate_x32(x, j % 4, rotate);
        // largest / 127 (1 for zeros), as the CPU divides it.
        const float scale = most > 0.0f ? __fdiv_rn(most, 127.0f) : 1.0f;
        if (!live) continue;
        repro_store_block<32>(x, scale, q + (size_t)r * k + c0);
        if (j % 2 == 0) *repro_step_at(steps, rows, j / 2, r) = scale;
    }
}

// The out projection's input stage in the wide attention's tail.
struct ContextTail {
    unsigned* counts;
    const int* classes;
    const int* group_of_row;
    const float* centers;
    const float* signs;
    const float* fc;
    signed char* quantized;
    float* steps;
    // The batch's rows, which the steps' table lies across.
    int rows;

    // Ordinary launches cannot wait for other blocks to become resident.
    // The last head in each 128-column group consumes its published context.
    template <int D>
    __device__ __forceinline__ void finish(const __half* context, int first, int count, int hidden,
                                          int head, int row_group) const {
        static_assert(D == 64, "two whole heads partition each quantization block");
        constexpr int HEADS_PER_GROUP = 128 / D;
        const int column_group = head / HEADS_PER_GROUP, part = threadIdx.x & 3;
        const int c0 = 128 * column_group + 32 * part;
        const unsigned mask = ((1u << HEADS_PER_GROUP) - 1) << (column_group * HEADS_PER_GROUP);
        unsigned* group = counts + 2 * row_group;
        __shared__ unsigned complete;
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned before;
            const unsigned bit = 1u << head;
            asm volatile("atom.acq_rel.gpu.global.or.b32 %0, [%1], %2;"
                         : "=r"(before) : "l"(group), "r"(bit) : "memory");
            complete = ((before | bit) & mask) == mask;
        }
        __syncthreads();
        if (!complete) return;
        const int rows_per_pass = blockDim.x / 4;
        const unsigned negative = repro_sign_mask32(signs + c0);
        // Inactive rows still participate in the rotation's warp shuffles.
        for (int pass = 0; pass < (count + rows_per_pass - 1) / rows_per_pass; ++pass) {
            const int at = pass * rows_per_pass + threadIdx.x / 4, r = first + at;
            const bool live = at < count;
            uint4 raw[4] = {};
            float mean[32] = {};
            if (live) {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    raw[i] = *reinterpret_cast<const uint4*>(context + (size_t)r * hidden + c0 + 8 * i);
                const float* center = centers + (size_t)(classes[group_of_row[r]] - 1) * hidden + c0;
#pragma unroll
                for (int i = 0; i < 32; i += 4) {
                    const float4 m = *reinterpret_cast<const float4*>(center + i);
                    mean[i] = m.x, mean[i + 1] = m.y, mean[i + 2] = m.z, mean[i + 3] = m.w;
                }
            }
            float x[32];
            repro_centered_x32(raw, mean, negative, x);
            const float most = repro_rotate_x32(x, part, fc[FC_RECIP_SQRT_BLOCK]);
            const float scale = most > 0.0f ? __fdiv_rn(most, 127.0f) : 1.0f;
            if (live) {
                repro_store_block<32>(x, scale, quantized + (size_t)r * hidden + c0);
                if (part % 2 == 0) *repro_step_at(steps, rows, 2 * column_group + part / 2, r) = scale;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned finished;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                         : "=r"(finished) : "l"(group + 1) : "memory");
            if (finished == (unsigned)(hidden / 128 - 1)) {
                group[0] = 0;
                group[1] = 0;
            }
        }
    }
};

// Column j's digits are B row 2j and a zero row 2j + 1, so a fragment holds one column
// for both of a thread's rows; exact INT32 sums give the same integers over any k split.
template <class T, class AfterTile = int8proj::NoAfterTile, class Group = ReproWholeBlock>
__device__ __forceinline__ void repro_shared_mainloop(const signed char* __restrict__ a_global,
                                                      const signed char* __restrict__ q, int M,
                                                      int K, signed char* shared,
                                                      int (&acc)[T::MT][T::NT][4], int k_first = 0,
                                                      int k_tiles = -1, AfterTile after_tile = {},
                                                      Group group = {}) {
    using int8proj::tile_offset;
    constexpr int BM = T::BM, BN = T::BN, BK = T::BK, WM = T::WM, WN = T::WN;
    constexpr int STAGES = T::STAGES, THREADS = T::THREADS, CHUNKS = T::CHUNKS;
    constexpr int MT = T::MT, NT = T::NT, A_STAGE = T::A_STAGE, B_STAGE = T::B_STAGE;
    // A narrow tile has fewer A chunks than threads; the first ones copy.
    constexpr int A_CHUNKS = (BM * CHUNKS + THREADS - 1) / THREADS;
    constexpr bool EVERY_THREAD = BM * CHUNKS % THREADS == 0;
    constexpr int W_CHUNKS = BN * CHUNKS / THREADS;
    static_assert(BN % 2 == 0, "a tile's B rows pair up from an even one");
    static_assert(BN * CHUNKS % THREADS == 0, "every thread copies the same B chunks");
    signed char* w_shared = shared + STAGES * A_STAGE;
    const int thread = group.thread(THREADS), lane = thread & 31, warp = thread >> 5;
    const int warp_m = warp / T::WARPS_N, warp_n = warp % T::WARPS_N;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    if (k_tiles < 0) k_tiles = K / BK;

    // Addresses form once; a k tile adds its offset and a stage its base.
    const bool a_copies = EVERY_THREAD || thread < BM * CHUNKS;
    const signed char* a_from[A_CHUNKS];
    int a_bytes[A_CHUNKS];
    unsigned a_to[A_CHUNKS];
#pragma unroll
    for (int j = 0; j < A_CHUNKS; ++j) {
        const int i = min(thread + j * THREADS, BM * CHUNKS - 1);
        const int row = i / CHUNKS, chunk = i % CHUNKS, global_row = m0 + row;
        a_to[j] = tile_offset(row, chunk);
        a_from[j] = a_global + static_cast<size_t>(min(global_row, M - 1)) * K + chunk * 16;
        a_bytes[j] = global_row < M ? 16 : 0;
    }
    const signed char* w_from[W_CHUNKS];
    unsigned w_to[W_CHUNKS];
    bool zero[W_CHUNKS];
#pragma unroll
    for (int j = 0; j < W_CHUNKS; ++j) {
        const int i = thread + j * THREADS;
        const int row = i / CHUNKS, chunk = i % CHUNKS, column = (n0 + row) >> 1;
        zero[j] = row & 1;
        w_to[j] = tile_offset(row, chunk);
        if (!zero[j]) w_from[j] = q + static_cast<size_t>(column) * K + chunk * 16;
    }
    // The odd rows stay zero, written once before the first barrier hands a
    // stage to the products.
#pragma unroll
    for (int stage = 0; stage < STAGES; ++stage)
#pragma unroll
        for (int j = 0; j < W_CHUNKS; ++j)
            if (zero[j])
                *reinterpret_cast<uint4*>(w_shared + stage * B_STAGE + w_to[j]) =
                    make_uint4(0u, 0u, 0u, 0u);
    const unsigned a_base = shared_address(shared), w_base = shared_address(w_shared);
    const unsigned long long weights = l2_evict_first();
    auto load_stage = [&](int stage, int k_tile) {
        const int k0 = (k_first + k_tile * group.step) * BK;
        const unsigned a = a_base + stage * A_STAGE;
        if (a_copies)
#pragma unroll
            for (int j = 0; j < A_CHUNKS; ++j) cp_async_16(a + a_to[j], a_from[j] + k0, a_bytes[j]);
#pragma unroll
        for (int j = 0; j < W_CHUNKS; ++j)
            if (!zero[j]) cp_async_16_policy(w_base + stage * B_STAGE + w_to[j], w_from[j] + k0, weights);
    };
    unsigned a_at[MT][BK / 32], w_at[NT / 2][BK / 32];
#pragma unroll
    for (int kk = 0; kk < BK / 32; ++kk) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            a_at[mt][kk] = tile_offset(warp_m * WM + mt * 16 + (lane & 15), kk * 2 + (lane >> 4));
#pragma unroll
        for (int np = 0; np < NT / 2; ++np)
            w_at[np][kk] = tile_offset(warp_n * WN + np * 16 + (lane & 7) + ((lane >> 4) << 3),
                                       kk * 2 + ((lane >> 3) & 1));
    }

#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0;

#pragma unroll
    for (int stage = 0; stage < STAGES - 1; ++stage) {
        if (stage < k_tiles) load_stage(stage, stage);
        cp_async_commit();
    }
    int read = 0, write = STAGES - 1;
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        cp_async_wait<STAGES - 2>();
        group.sync();
        if (k_tile + STAGES - 1 < k_tiles) load_stage(write, k_tile + STAGES - 1);
        cp_async_commit();
        const unsigned a = a_base + read * A_STAGE, w = w_base + read * B_STAGE;
#pragma unroll
        for (int kk = 0; kk < BK / 32; ++kk) {
            unsigned a_frag[MT][4];
            unsigned w_frag[NT][2];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) ldmatrix_x4(a_frag[mt], a + a_at[mt][kk]);
#pragma unroll
            for (int np = 0; np < NT / 2; ++np) {
                unsigned r[4];
                ldmatrix_x4(r, w + w_at[np][kk]);
                w_frag[2 * np][0] = r[0];
                w_frag[2 * np][1] = r[1];
                w_frag[2 * np + 1][0] = r[2];
                w_frag[2 * np + 1][1] = r[3];
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) int8proj::mma_s8(acc[mt][nt], a_frag[mt], w_frag[nt]);
        }
        after_tile(k_first + k_tile * group.step, acc);
        read = read + 1 == STAGES ? 0 : read + 1;
        write = write + 1 == STAGES ? 0 : write + 1;
    }
    cp_async_wait<0>();
}

// attention/integer.rs scale_for. Not inlined: predicated into every caller,
// its f64 product would hold the f64 pipe whether or not a lane took it.
__device__ __noinline__ float repro_scale_for(float largest, float levels) {
    float scale = __fdiv_rn(largest, levels);
    bool subnormal = scale != 0.0f && abs_bits(scale) < 1.17549435e-38f;
    if (subnormal && __dmul_rn((double)scale, (double)levels) < (double)largest)
        return __int_as_float(__float_as_int(scale) + 1);
    return scale > 0.0f ? scale : 1.0f;
}

enum ReproEpilogue { REPRO_PLAIN, REPRO_RESIDUAL, REPRO_GELU, REPRO_QKV, REPRO_GELU_BLOCKS, REPRO_QKV_PACKED };

// The down projection's INT8 input, made in place of FP16 rows.
struct ReproBlocksOut {
    signed char* q;
    float* steps;
    const float* signs;
    const float* centers;
};

// Value magnitudes are kept as float bits, which order as nonnegative floats do,
// so integer atomics can take their maxima.
struct ReproAttentionPrep {
    signed char* queries;
    float* query_scales;
    const float* key_centers;
    signed char* keys;
    float* key_scales;
    __half* values16;
    unsigned* value_magnitudes;
    int* failed;
    const int* sequence_codes;
    int heads;
};

// Attention's keys a chunk, staged through shared memory: sequences' value
// planes start at multiples of the chunk.
#define ATT_CHUNK 64

// The values as repro_attention_pack's value warps make them.
struct ReproPackedValues {
    signed char* planes;
    float* value_scales;
    const int* starts;
    const int* lengths;
    const int* pstarts;
    int sequences, ptokens;
};

// repro_quotient where it is the division (repro_quotient_check's range);
// those steps are normal, so no subnormal correction applies.
__device__ __forceinline__ float repro_scale_for_127(float largest) {
    if (!(largest == 0.0f || (largest >= 0x1p-60f && largest <= 0x1p60f)))
        return repro_scale_for(largest, 127.0f);
    const float scale = repro_quotient(largest, 127.0f, __uint_as_float(0x3c010204u));
    return scale > 0.0f ? scale : 1.0f;
}

// Counts repro_scale_for_127's mismatches; an atomic a warp.
extern "C" __global__ void repro_scale_check(unsigned first, int n, unsigned long long* mismatches) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = __uint_as_float(first + (unsigned)i);
    bool differs = i < n && __float_as_uint(repro_scale_for_127(x)) !=
                                __float_as_uint(repro_scale_for(x, 127.0f));
    unsigned count = __popc(__ballot_sync(0xffffffffu, differs));
    if ((threadIdx.x & 31) == 0 && count) atomicAdd(mismatches, (unsigned long long)count);
}

// Sums start at int8proj::SUMS_FROM, so one FMA gives FACTOR times a sum exactly
// (int8proj::promoted), with no second accumulator set and no reset. With TWO the operand
// holds two digits a row, the second behind the first: K is twice the weights' depth, and
// the loop walks the weights twice in its one chain. A kernel of its own takes that walk:
// the weights' depth as a value of the loop costs every product of the widest tile.
template <class T, int FACTOR, bool ALIGNED, bool TWO>
__device__ __forceinline__ void repro_promoted_k_loop(const signed char* __restrict__ a_global,
                                                      const signed char* __restrict__ w_global,
                                                      const float* __restrict__ steps, int M, int K,
                                                      signed char* shared, float (&f)[T::MT][T::NT][4],
                                                      int k_first, int k_tiles) {
    constexpr int BM = T::BM, BN = T::BN, BK = T::BK, WM = T::WM, WN = T::WN;
    constexpr int STAGES = T::STAGES, THREADS = T::THREADS, CHUNKS = T::CHUNKS;
    constexpr int MT = T::MT, NT = T::NT, A_STAGE = T::A_STAGE, B_STAGE = T::B_STAGE;
    static_assert(BK == 64 && NT % 2 == 0, "16-column subtiles of 64-k tiles");
    static_assert(((BM * CHUNKS) % THREADS == 0 || THREADS % (BM * CHUNKS) == 0) &&
                      ((BN * CHUNKS) % THREADS == 0 || BN * CHUNKS < THREADS),
                  "even copies");
    // A thread's rows lie a multiple of 8 apart, so one swizzle term serves all.
    constexpr int ROWS = THREADS / CHUNKS;
    static_assert(THREADS % CHUNKS == 0 && ROWS % 8 == 0, "one swizzle term a thread's copies");
    // Tiles of fewer copies than threads copy from their first threads.
    constexpr int A_COPIES = BM * CHUNKS >= THREADS ? BM * CHUNKS / THREADS : 1;
    constexpr int B_COPIES = BN * CHUNKS >= THREADS ? BN * CHUNKS / THREADS : 1;
    const unsigned a_shared = shared_address(shared), w_shared = a_shared + STAGES * A_STAGE;
    const int thread = threadIdx.x, lane = thread & 31, warp = thread >> 5;
    const int warp_m = warp / T::WARPS_N, warp_n = warp % T::WARPS_N;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int period = TWO ? K / 2 : K;
    const int copy_row = thread / CHUNKS, chunk = thread % CHUNKS;
    const unsigned copy_target = int8proj::tile_offset(copy_row, chunk);
    const bool a_copies = BM * CHUNKS >= THREADS || thread < BM * CHUNKS;
    const bool b_copies = BN * CHUNKS >= THREADS || thread < BN * CHUNKS;
    const signed char* a_source[A_COPIES];
    unsigned a_valid[A_COPIES];
#pragma unroll
    for (int j = 0; j < A_COPIES; ++j) {
        const int global_row = m0 + copy_row + j * ROWS;
        a_source[j] = a_global + static_cast<size_t>(min(global_row, M - 1)) * K + chunk * 16;
        a_valid[j] = global_row < M ? 16 : 0;
    }
    const signed char* w_source = w_global + static_cast<size_t>(n0 + copy_row) * period + chunk * 16;
    // Fragment rows 16 apart share one swizzle term a k half.
    const int a_row = warp_m * WM + (lane & 15);
    const int w_row = warp_n * WN + (lane & 7) + ((lane >> 4) << 3);
    const unsigned a_at[2] = {static_cast<unsigned>(int8proj::tile_offset(a_row, lane >> 4)),
                              static_cast<unsigned>(int8proj::tile_offset(a_row, 2 + (lane >> 4)))};
    const unsigned w_at[2] = {static_cast<unsigned>(int8proj::tile_offset(w_row, (lane >> 3) & 1)),
                              static_cast<unsigned>(int8proj::tile_offset(w_row, 2 + ((lane >> 3) & 1)))};
    // Rows past M take the last row's steps, keeping reads in bounds.
    int step_at[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) step_at[mt][h] = min(m0 + warp_m * WM + mt * 16 + (lane >> 2) + 8 * h, M - 1);

    const unsigned long long weights = l2_evict_first();
    auto load_stage = [&](int stage, int k_tile) {
        // The second walk takes the weights' tiles again.
        const int k0 = (k_first + k_tile) * BK, kw = TWO && k0 >= period ? k0 - period : k0;
        const unsigned a = a_shared + stage * A_STAGE + copy_target;
        const unsigned w = w_shared + stage * B_STAGE + copy_target;
        if (a_copies)
#pragma unroll
            for (int j = 0; j < A_COPIES; ++j) cp_async_16(a + j * ROWS * BK, a_source[j] + k0, a_valid[j]);
        if (b_copies)
#pragma unroll
            for (int j = 0; j < B_COPIES; ++j)
                cp_async_16_policy(w + j * ROWS * BK, w_source + static_cast<size_t>(j) * ROWS * period + kw,
                                   weights);
    };

#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) f[mt][nt][i] = 0.0f;
#pragma unroll
    for (int stage = 0; stage < STAGES - 1; ++stage) {
        if (stage < k_tiles) load_stage(stage, stage);
        cp_async_commit();
    }
    auto meet = [&](int k_tile) {
        cp_async_wait<STAGES - 2>();
        if constexpr (ALIGNED)
            __syncthreads();
        else
            __barrier_sync(0);
        const int prefetch = k_tile + STAGES - 1;
        if (prefetch < k_tiles) load_stage(prefetch % STAGES, prefetch);
        cp_async_commit();
    };
    // Zero rows need copies and barriers but no products; their sums stay +0.
    if constexpr (!ALIGNED) {
        if (!__all_sync(0xffffffffu, m0 + warp_m * WM < M)) {
            for (int k_tile = 0; k_tile < k_tiles; ++k_tile) meet(k_tile);
            cp_async_wait<0>();
            return;
        }
    }
    // A k tile is a part, its rows' steps side by side in the table. The index forms in
    // unsigned arithmetic, which may wrap: no pointer can then walk the table through the
    // loop, and one would take the widest tile past the registers ptxas allows it.
    auto steps_of = [&](int k_tile, float (&to)[MT][2]) {
        const unsigned part = static_cast<unsigned>(k_first + k_tile) * static_cast<unsigned>(M);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) to[mt][h] = __ldg(steps + (static_cast<unsigned>(step_at[mt][h]) + part));
    };
    // The 16 x 16 warp tile loads a k tile's steps a tile ahead: each part's rows lie on a
    // line of the table that no earlier part brought in, and its k tile of four tensor
    // instructions is too short to hide the fetch. Wider warp tiles hide it, and the two
    // values held would cost them.
    constexpr bool AHEAD = MT == 1 && NT == 2;
    float step[MT][2];
    [[maybe_unused]] float next[MT][2];
    if constexpr (AHEAD) steps_of(0, next);
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        if constexpr (AHEAD) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int h = 0; h < 2; ++h) step[mt][h] = next[mt][h];
            // The last tile loads its own steps again, a read in bounds that nothing uses.
            steps_of(min(k_tile + 1, k_tiles - 1), next);
        } else {
            steps_of(k_tile, step);
        }
        meet(k_tile);
        const unsigned a = a_shared + (k_tile % STAGES) * A_STAGE;
        const unsigned w = w_shared + (k_tile % STAGES) * B_STAGE;
        unsigned a_frag[2][MT][4];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) ldmatrix_x4(a_frag[kk][mt], a + a_at[kk] + mt * 16 * BK);
#pragma unroll
        for (int np = 0; np < NT / 2; ++np) {
            unsigned r[2][4];
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) ldmatrix_x4(r[kk], w + w_at[kk] + np * 16 * BK);
            const unsigned w0[2][2] = {{r[0][0], r[0][1]}, {r[1][0], r[1][1]}};
            const unsigned w1[2][2] = {{r[0][2], r[0][3]}, {r[1][2], r[1][3]}};
            int acc[MT][2][4];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                int8proj::mma_s8_from(acc[mt][0], a_frag[0][mt], w0[0], int8proj::SUMS_FROM);
                int8proj::mma_s8_from(acc[mt][1], a_frag[0][mt], w1[0], int8proj::SUMS_FROM);
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                int8proj::mma_s8(acc[mt][0], a_frag[1][mt], w0[1]);
                int8proj::mma_s8(acc[mt][1], a_frag[1][mt], w1[1]);
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float part = int8proj::promoted<FACTOR>(acc[mt][j][i]);
                        f[mt][2 * np + j][i] = __fmaf_rn(part, step[mt][i >> 1], f[mt][2 * np + j][i]);
                    }
        }
    }
    cp_async_wait<0>();
}

// Partial tiles need non-aligned barriers because inactive warps meet in
// a separate loop. Full tiles keep one common barrier instruction.
template <class T, int FACTOR = 1, bool TWO = false>
__device__ __forceinline__ void repro_promoted_mainloop(const signed char* __restrict__ a_global,
                                                        const signed char* __restrict__ w_global,
                                                        const float* __restrict__ steps, int M,
                                                        int K, signed char* shared,
                                                        float (&f)[T::MT][T::NT][4], int k_first,
                                                        int k_tiles) {
    const int m0 = blockIdx.y * T::BM;
    if (m0 + T::BM <= M)
        repro_promoted_k_loop<T, FACTOR, true, TWO>(a_global, w_global, steps, M, K, shared, f, k_first, k_tiles);
    else
        repro_promoted_k_loop<T, FACTOR, false, TWO>(a_global, w_global, steps, M, K, shared, f, k_first, k_tiles);
}

// fused_epilogue.rs dot_restored_into: a part's exact dot 14 (q . a) is below
// 2^24, so its float is exact, and no row depends on another.
template <int STAGES>
using RestoredTile = int8proj::Tiling<16, 64, 16, 16, STAGES>;
static_assert(RestoredTile<3>::SHARED_BYTES == 15360 && RestoredTile<4>::SHARED_BYTES == 20480,
              "a group's stages take 15 or 20 KiB");

// repro.rs STATIC_CLASSES; a row's code is -c for its class c.
#define REPRO_CLASSES 3

// Named rows, so a code selects values: chosen from an array, the array would
// live in local memory.
struct ReproClassRows {
    float4 first, last, middle;
};
static_assert(REPRO_CLASSES == 3, "a row of each class");
__device__ __forceinline__ ReproClassRows repro_class_rows(const float* __restrict__ table, int n, int c) {
    return {__ldg(reinterpret_cast<const float4*>(table + c)),
            __ldg(reinterpret_cast<const float4*>(table + (size_t)n + c)),
            __ldg(reinterpret_cast<const float4*>(table + 2 * (size_t)n + c))};
}
__device__ __forceinline__ float4 repro_class_of(const ReproClassRows& rows, int code) {
    return code == -1 ? rows.first : code == -2 ? rows.last : rows.middle;
}

// Staged past the operand stages before the mainloop starts.
template <int ROWS>
struct RestoredInputs {
    int row_codes[ROWS];
    float col_scales[32];
    float statics[REPRO_CLASSES][32];
    float sums[32];
    float constants[32];
    float2 moments[ROWS];
};
static_assert(sizeof(RestoredInputs<16>) == 960 && sizeof(RestoredInputs<64>) == 1536 &&
                  sizeof(RestoredInputs<128>) == 2304,
              "the host adds these bytes");

// Null for every projection whose norm is not deferred.
struct DeferredTables {
    const float2* moments;
    const float* sums;
    const float* constants;
};

// The copies travel in the mainloop's first copy group, so the epilogue waits
// on no load; rows past `rows` read the last one's, never used.
template <int ROWS>
__device__ __forceinline__ void repro_restored_inputs(RestoredInputs<ROWS>& in,
                                                      const int* __restrict__ row_shared,
                                                      const float* __restrict__ col_scales,
                                                      const float* __restrict__ static_shared,
                                                      const DeferredTables& deferred, int m0,
                                                      int rows, int n) {
    static_assert(ROWS <= 64 || ROWS % 64 == 0, "rows and columns take separate threads");
    constexpr int PER = ROWS > 64 ? ROWS / 64 : 1;
    const int t = threadIdx.x, j0 = blockIdx.x * 32;
    if (t < (ROWS < 64 ? ROWS : 64)) {
#pragma unroll
        for (int u = 0; u < PER; ++u) {
            const int i = t + 64 * u, r = min(m0 + i, rows - 1);
            cp_async_4(shared_address(&in.row_codes[i]), row_shared + r);
            if (deferred.moments) cp_async_8(shared_address(&in.moments[i]), deferred.moments + r);
        }
    } else if (t >= 64 && t < 72) {
        const int c = 4 * (t - 64), j = j0 + c;
        cp_async_16(shared_address(&in.col_scales[c]), col_scales + j, 16);
#pragma unroll
        for (int s = 0; s < REPRO_CLASSES; ++s)
            cp_async_16(shared_address(&in.statics[s][c]), static_shared + (size_t)s * n + j, 16);
        if (deferred.moments) {
            cp_async_16(shared_address(&in.sums[c]), deferred.sums + j, 16);
            cp_async_16(shared_address(&in.constants[c]), deferred.constants + j, 16);
        }
    }
}

// prepared.rs finish_deferred: fma(fma(-mean, W gamma, u), rstd, W beta + b).
__device__ __forceinline__ float repro_deferred_value(float u, float2 moments, float sum,
                                                      float constant) {
    return __fmaf_rn(__fmaf_rn(-moments.x, sum, u), moments.y, constant);
}

// The residual rows stage past the inputs in 16-byte chunks.
static_assert(sizeof(RestoredInputs<16>) % 16 == 0, "aligned chunks");
using RestoredResidual = __half[16][32];

// Rounded additions cannot split, so group 0 chains every group's parts in k order; the
// others park theirs in `slot` behind barriers FULL and FREE, G + 3 barriers at most.
template <class T, int G>
struct ReproRestoredChain {
    static constexpr int FULL = G + 1, FREE = G + 2, PARTS = T::NT * 2;
    float (&f)[T::NT][2];
    float (&total)[T::NT][2];
    int& fold;
    int* slot;
    const float* steps;
    int group, rounds, segment;
    __device__ __forceinline__ void operator()(int k_tile, int (&acc)[T::MT][T::NT][4]) const {
        static_assert(T::MT == 1 && T::WM == 16, "one warp row");
        const int thread = (int)threadIdx.x - T::THREADS * group, round = k_tile / G;
        int part[T::NT][2];
#pragma unroll
        for (int nt = 0; nt < T::NT; ++nt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                part[nt][h] = 14 * acc[0][nt][2 * h];
                acc[0][nt][2 * h] = 0;
            }
        if constexpr (G > 1) {
            if (group > 0) {
                if (round > 0) repro_bar_sync<FREE>(G * T::THREADS);
                int* mine = slot + (group - 1) * PARTS * T::THREADS + thread;
#pragma unroll
                for (int nt = 0; nt < T::NT; ++nt)
#pragma unroll
                    for (int h = 0; h < 2; ++h) mine[(nt * 2 + h) * T::THREADS] = part[nt][h];
                repro_bar_arrive<FULL>(G * T::THREADS);
                return;
            }
            repro_bar_sync<FULL>(G * T::THREADS);
        }
        const int lane = thread & 31, first = round * G, ends = fold - first - 1;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float* row = steps + (lane >> 2) + 8 * h;
#pragma unroll
            for (int nt = 0; nt < T::NT; ++nt) {
                float v = f[nt][h];
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    const int p = g == 0 ? part[nt][h] : slot[((g - 1) * PARTS + nt * 2 + h) * T::THREADS + thread];
                    v = __fmaf_rn(__int2float_rn(p), row[16 * (first + g)], v);
                    if (g == ends) {
                        total[nt][h] = fold == segment ? v : __fadd_rn(total[nt][h], v);
                        v = 0.0f;
                    }
                }
                f[nt][h] = v;
            }
        }
        if (ends >= 0 && ends < G) fold += segment;
        if constexpr (G > 1)
            if (round + 1 < rounds) repro_bar_arrive<FREE>(G * T::THREADS);
    }
};

// repro_restored_epilogue's operations, by group 0's four warps.
template <class T, int E>
__device__ __forceinline__ void repro_restored_epilogue(const float (&f)[T::NT][2],
                                                        const RestoredInputs<16>& in,
                                                        const RestoredResidual& residual_rows,
                                                        bool deferred, const float* __restrict__ fc,
                                                        int rows, int n, void* __restrict__ out) {
    static_assert(T::BN == 64 && T::BM == 16, "a lane a column, the inputs' 16 rows");
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, j0 = blockIdx.x * 32;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = (lane >> 2) + 8 * h;
        if (r >= rows) continue;
        const float* statics = in.statics[-in.row_codes[r] - 1];
#pragma unroll
        for (int nt = 0; nt < T::NT; ++nt) {
            const int jl = (warp * T::WN + nt * 8) / 2 + (lane & 3);
            float v = __fadd_rn(__fmul_rn(f[nt][h], in.col_scales[jl]), statics[jl]);
            if (deferred) v = repro_deferred_value(v, in.moments[r], in.sums[jl], in.constants[jl]);
            const size_t at = (size_t)r * n + j0 + jl;
            if constexpr (E == REPRO_RESIDUAL) {
                v = __fadd_rn(v, __half2float(residual_rows[r][jl]));
                static_cast<__half*>(out)[at] = __float2half_rn(v);
            } else if constexpr (E == REPRO_GELU) {
                static_cast<__half*>(out)[at] = __float2half_rn(repro_gelu(v, fc));
            } else {
                static_cast<float*>(out)[at] = v;
            }
        }
    }
}

// Group g takes k tiles g, g + G, ... on its own stages and barrier, the parts
// chaining across groups as they go, so each tile streams its digits once.
template <int G>
constexpr int RESTORED_STAGES = G == 1 ? 4 : 3;

template <int G>
constexpr int repro_restored_bytes() {
    using T = RestoredTile<RESTORED_STAGES<G>>;
    constexpr int slot = (G - 1) * T::NT * 2 * T::THREADS * 4;
    constexpr int steps = 16 * (4096 / HALF) * 4;
    return G * T::SHARED_BYTES + sizeof(RestoredInputs<16>) + sizeof(RestoredResidual) + slot +
           steps;
}
static_assert(repro_restored_bytes<1>() == 20480 + 960 + 1024 + 4096 &&
                  repro_restored_bytes<2>() == 2 * 15360 + 960 + 1024 + 2048 + 4096 &&
                  repro_restored_bytes<4>() == 4 * 15360 + 960 + 1024 + 6144 + 4096,
              "the host adds these bytes");

// Each 128-block's count of tiles done is zero between launches.
struct NextInput {
    const float* centers;
    const float* signs;
    signed char* quantized;
    float* steps;
    int* done;
};

// Staged with the epilogue's inputs.
using NextStaged = float[4][BLOCK];
static_assert(sizeof(NextStaged) == 2048, "the host adds these bytes");

__device__ __forceinline__ NextStaged& next_staged(float* steps) {
    return *reinterpret_cast<NextStaged*>(steps + 16 * (4096 / HALF));
}

// The 128-block's last tile quantizes its rows back from L2; the group's barrier and
// thread 0's fence order the tiles' stores before their counts, as a grid barrier does.
__device__ __forceinline__ void repro_restored_next_input(const RestoredInputs<16>& in,
                                                          const NextStaged& staged,
                                                          const NextInput& next, const __half* out,
                                                          const float* fc, int rows, int n) {
    __shared__ int last;
    const int b = blockIdx.x / 4;
    GroupSync<0>{}();
    if (threadIdx.x == 0) {
        __threadfence();
        last = atomicAdd(next.done + b, 1) == 3;
        if (last) {
            next.done[b] = 0;
            __threadfence();
        }
    }
    GroupSync<0>{}();
    if (!last || threadIdx.x >= 64) return;
    const int lane = threadIdx.x & 31, r = 8 * (threadIdx.x >> 5) + lane / 4, part = lane % 4;
    const int c = BLOCK * b + 32 * part;
    const bool live = r < rows;
    uint4 raw[4] = {};
    if (live)
#pragma unroll
        for (int i = 0; i < 4; ++i) raw[i] = __ldcg(reinterpret_cast<const uint4*>(out + (size_t)r * n + c) + i);
    const float* center = staged[-in.row_codes[r] - 1] + 32 * part;
    const float* sign = staged[3] + 32 * part;
    float x[32];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned words[4] = {raw[i].x, raw[i].y, raw[i].z, raw[i].w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 pair = __half22float2(*reinterpret_cast<const __half2*>(&words[e]));
            const int v = 8 * i + 2 * e;
            x[v] = __fmul_rn(__fsub_rn(pair.x, center[v]), sign[v] < 0.0f ? -1.0f : 1.0f);
            x[v + 1] = __fmul_rn(__fsub_rn(pair.y, center[v + 1]), sign[v + 1] < 0.0f ? -1.0f : 1.0f);
        }
    }
    const float most = repro_rotate_x32(x, part, fc[FC_RECIP_SQRT_BLOCK]);
    const float scale = most > 0.0f ? __fdiv_rn(most, 127.0f) : 1.0f;
    if (!live) return;
    repro_store_block<32>(x, scale, next.quantized + (size_t)r * n + c);
    if (part % 2 == 0) *repro_step_at(next.steps, rows, 2 * b + part / 2, r) = scale;
}

// With NEXT the tile also makes the down projection's input.
template <int E, int G, bool NEXT = false>
__device__ __forceinline__ void repro_project_restored(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc,
    int rows, int n, int k, int segment, void* out, signed char* stages, const DeferredTables& deferred,
    const NextInput& next = {}) {
    static_assert(!NEXT || E == REPRO_GELU, "the up projection makes the down projection's input");
    using T = RestoredTile<RESTORED_STAGES<G>>;
    constexpr int RING = T::SHARED_BYTES, PARTS = T::NT * 2 * T::THREADS;
    signed char* inputs = stages + G * RING;
    RestoredInputs<16>& in = *reinterpret_cast<RestoredInputs<16>*>(inputs);
    RestoredResidual& residual_rows =
        *reinterpret_cast<RestoredResidual*>(inputs + sizeof(RestoredInputs<16>));
    int* slot = reinterpret_cast<int*>(inputs + sizeof(RestoredInputs<16>) + sizeof(RestoredResidual));
    float* steps = reinterpret_cast<float*>(slot + (G - 1) * PARTS);
    const int group = (int)threadIdx.x / T::THREADS, thread = (int)threadIdx.x % T::THREADS;
    const int parts = k / HALF, rounds = k / T::BK / G;
    // Group 0's copies land with its first stage.
    if (group == 0) {
        repro_restored_inputs<16>(in, row_shared, col_scales, static_shared, deferred, 0, rows, n);
        if constexpr (E == REPRO_RESIDUAL)
            if (thread < 64) {
                const int row = thread / 4, chunk = thread % 4;
                cp_async_16(shared_address(&residual_rows[row][8 * chunk]),
                            residual + (size_t)min(row, rows - 1) * n + blockIdx.x * 32 + 8 * chunk,
                            row < rows ? 16 : 0);
            }
        // The steps stage in the table's order, a part's rows side by side, so a copy
        // instruction's lanes read two lines of it; rows past the last take its steps,
        // keeping reads in bounds.
        for (int i = thread; i < 16 * parts; i += T::THREADS)
            cp_async_4(shared_address(steps + i), repro_step_at(row_steps, rows, i / 16, min(i % 16, rows - 1)));
        if constexpr (NEXT) {
            static_assert(T::THREADS == 128, "a chunk a thread");
            const int row = thread / 32, c = BLOCK * (blockIdx.x / 4) + 4 * (thread % 32);
            cp_async_16(shared_address(&next_staged(steps)[row][4 * (thread % 32)]),
                        row < 3 ? next.centers + (size_t)row * n + c : next.signs + c, 16);
        }
    }
    int acc[T::MT][T::NT][4];
    float promoted[T::NT][2] = {}, total[T::NT][2] = {};
    static_assert(T::BK == 64, "a 64-deep k tile a part");
    // The restart must be whole tiles, at least a round's, and divide k.
    if (threadIdx.x == 0 && (segment % T::BK != 0 || segment / T::BK < G || k % segment != 0)) __trap();
    int fold = segment / T::BK;
    repro_shared_mainloop<T>(
        qa, q, rows, k, stages + group * RING, acc, group, rounds,
        ReproRestoredChain<T, G>{promoted, total, fold, slot, steps, group, rounds, segment / T::BK},
        ReproLoopGroup<G>{group});
    if (group > 0) return;
    repro_restored_epilogue<T, E>(total, in, residual_rows, deferred.moments != nullptr, fc, rows, n,
                                  out);
    if constexpr (NEXT)
        repro_restored_next_input(in, next_staged(steps), next, static_cast<const __half*>(out), fc,
                                  rows, n);
}

// One kernel per epilogue and repro.rs GROUPINGS entry.
#define REPRO_RESTORED(EPILOGUE_NAME, EPILOGUE, G)                                               \
    extern "C" __global__ void __launch_bounds__(128 * G) repro_restored_##EPILOGUE_NAME##_g##G( \
        const signed char* qa, const float* row_steps, const signed char* q,                     \
        const float* col_scales, const float* static_shared, const int* row_shared,              \
        const __half* residual, const float* fc, int rows, int n, int k, int segment,            \
        void* out, const float2* moments, const float* sums, const float* constants) {           \
        extern __shared__ __align__(128) signed char stages[];                                   \
        repro_project_restored<EPILOGUE, G>(qa, row_steps, q, col_scales, static_shared,         \
                                            row_shared, residual, fc, rows, n, k, segment, out,  \
                                            stages, DeferredTables{moments, sums, constants});   \
    }
REPRO_RESTORED(plain, REPRO_PLAIN, 1)
REPRO_RESTORED(plain, REPRO_PLAIN, 2)
REPRO_RESTORED(plain, REPRO_PLAIN, 4)
REPRO_RESTORED(residual, REPRO_RESIDUAL, 1)
REPRO_RESTORED(residual, REPRO_RESIDUAL, 2)
REPRO_RESTORED(residual, REPRO_RESIDUAL, 4)
REPRO_RESTORED(gelu, REPRO_GELU, 1)
REPRO_RESTORED(gelu, REPRO_GELU, 2)
REPRO_RESTORED(gelu, REPRO_GELU, 4)

// Its shared bytes are repro_restored_bytes<G>() plus NextStaged's.
#define REPRO_RESTORED_GELU_NEXT(G)                                                              \
    extern "C" __global__ void __launch_bounds__(128 * G) repro_restored_gelu_next_g##G(        \
        const signed char* qa, const float* row_steps, const signed char* q,                     \
        const float* col_scales, const float* static_shared, const int* row_shared,              \
        const float* fc, int rows, int n, int k, int segment, __half* out, const float* centers, \
        const float* signs, signed char* quantized, float* steps, int* done,                     \
        const float2* moments, const float* sums, const float* constants) {                      \
        extern __shared__ __align__(128) signed char stages[];                                   \
        repro_project_restored<REPRO_GELU, G, true>(                                             \
            qa, row_steps, q, col_scales, static_shared, row_shared, nullptr, fc, rows, n, k,    \
            segment, out, stages, DeferredTables{moments, sums, constants},                      \
            NextInput{centers, signs, quantized, steps, done});                                  \
    }
REPRO_RESTORED_GELU_NEXT(1)
REPRO_RESTORED_GELU_NEXT(2)
REPRO_RESTORED_GELU_NEXT(4)

// Tall tiles of 64 rows, so a sequence's rows share each tile's weights.
#define TALL_B_STAGE (32 * 64)
#define TALL_NT 4

template <int ROWS>
__host__ __device__ constexpr int repro_tall_stage() {
    return ROWS * 64 + TALL_B_STAGE;
}
static_assert(repro_tall_stage<64>() == 6144, "a stage takes 6 KiB");

// Each B fragment serves all MT m-subtiles; addresses form once, and an
// m-subtile entirely past M takes no products.
template <int STAGES, int W, int MT, class Group, class AfterTile>
__device__ __forceinline__ void repro_tall_mainloop(const signed char* __restrict__ qa,
                                                    const signed char* __restrict__ q, int M,
                                                    int K, signed char* shared, int (&acc)[MT][TALL_NT][4],
                                                    int k_first, int k_tiles,
                                                    AfterTile after_tile, Group group) {
    using int8proj::tile_offset;
    static_assert(W >= 4, "the group's threads below 128 copy B");
    constexpr int NT = TALL_NT, SUB = 16 * W, A_STAGE = SUB * MT * 64, COPIES = 2 * MT;
    signed char* w_shared = shared + STAGES * A_STAGE;
    const int thread = group.thread(32 * W), lane = thread & 31, warp = thread >> 5;
    const int m0 = blockIdx.y * SUB * MT, c0 = blockIdx.x * (8 * NT);
    // Threads below 128 also copy their column's INT8 chunk.
    const int row = thread >> 2, chunk = thread & 3;
    const signed char* a_from[COPIES];
    int a_bytes[COPIES];
    unsigned a_to[COPIES];
#pragma unroll
    for (int i = 0; i < COPIES; ++i) {
        a_from[i] = qa + (size_t)min(m0 + 8 * W * i + row, M - 1) * K + chunk * 16;
        a_bytes[i] = m0 + 8 * W * i + row < M ? 16 : 0;
        a_to[i] = tile_offset(8 * W * i + row, chunk);
    }
    const bool copies_w = W == 4 || thread < 128;
    const signed char* w_from = q + (size_t)(c0 + row) * K + chunk * 16;
    const unsigned a_base = shared_address(shared), w_base = shared_address(w_shared);
    const unsigned w_to = tile_offset(row, chunk);
    const unsigned long long weights = l2_evict_first();
    auto load_stage = [&](int stage, int k_tile) {
        const int k0 = (k_first + k_tile * group.step) * 64;
        const unsigned a = a_base + stage * A_STAGE;
#pragma unroll
        for (int i = 0; i < COPIES; ++i) cp_async_16(a + a_to[i], a_from[i] + k0, a_bytes[i]);
        if (copies_w) cp_async_16_policy(w_base + stage * TALL_B_STAGE + w_to, w_from + k0, 16, weights);
    };
    // An m-subtile is live only if every one before it is.
    bool live[MT];
    unsigned rows_at[MT][2], pair_at[NT / 2][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) live[mt] = m0 + SUB * mt + 16 * warp < M;
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            rows_at[mt][kk] = tile_offset(SUB * mt + 16 * warp + (lane & 15), kk * 2 + (lane >> 4));
        const int b_row = (lane & 7) + ((lane >> 4) << 3), b_chunk = kk * 2 + ((lane >> 3) & 1);
#pragma unroll
        for (int p = 0; p < NT / 2; ++p) pair_at[p][kk] = tile_offset(16 * p + b_row, b_chunk);
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][t][i] = 0;
#pragma unroll
    for (int stage = 0; stage < STAGES - 1; ++stage) {
        if (stage < k_tiles) load_stage(stage, stage);
        cp_async_commit();
    }
    int read = 0, write = STAGES - 1;
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        cp_async_wait<STAGES - 2>();
        group.sync();
        if (k_tile + STAGES - 1 < k_tiles) load_stage(write, k_tile + STAGES - 1);
        cp_async_commit();
        const unsigned a = a_base + read * A_STAGE, w = w_base + read * TALL_B_STAGE;
        if (live[0])
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                unsigned rows_frag[MT][4];
#pragma unroll
                for (int mt = 0; mt < MT; ++mt)
                    if (live[mt]) ldmatrix_x4(rows_frag[mt], a + rows_at[mt][kk]);
#pragma unroll
                for (int p = 0; p < NT / 2; ++p) {
                    unsigned r[4];
                    ldmatrix_x4(r, w + pair_at[p][kk]);
                    const unsigned lower[2] = {r[0], r[1]}, upper[2] = {r[2], r[3]};
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt)
                        if (live[mt]) {
                            int8proj::mma_s8(acc[mt][2 * p], rows_frag[mt], lower);
                            int8proj::mma_s8(acc[mt][2 * p + 1], rows_frag[mt], upper);
                        }
                }
            }
        after_tile(k_first + k_tile * group.step, acc);
        read = read + 1 == STAGES ? 0 : read + 1;
        write = write + 1 == STAGES ? 0 : write + 1;
    }
    cp_async_wait<0>();
}

// A tile's rows' steps staged in shared memory in the table's order, a part's rows side by
// side: a copy instruction's lanes then read one line of the table.
template <int ROWS>
struct ReproStagedSteps {
    const float* steps;
    __device__ __forceinline__ float operator()(int local, int part) const { return steps[part * ROWS + local]; }
};

// The steps read from the product's table, where staging them would cost a block its place:
// the segments' rings leave no room for every segment's parts, and 128 rows of parts would
// take the taller tile past half a multiprocessor's shared memory.
struct ReproTableSteps {
    const float* steps;
    int rows, first_row, first_part;
    __device__ __forceinline__ float operator()(int local, int part) const {
        return __ldg(repro_step_at(steps, rows, first_part + part, min(first_row + local, rows - 1)));
    }
};

// ReproRestoredChain's chaining for the tall tile; G is at most 2: a round's parts are
// group 0's and one parked, each on its own step. Taller tiles take one group.
template <int G, int MT = 1, int W = 4, class Steps = ReproStagedSteps<16 * W>>
struct ReproTallChain {
    static_assert(G == 1 || G == 2, "a round's parts are group 0's and one parked");
    static_assert((MT == 1 && W == 4) || G == 1, "a taller tile's parts stay in its one group");
    static constexpr int FULL = G + 1, FREE = G + 2, SUB = 16 * W, NT = TALL_NT;
    float (&f)[MT][NT][4];
    float (&total)[MT][NT][4];
    int* slot;
    Steps steps;
    int group, rows, rounds, segment, fold;
    __device__ __forceinline__ void operator()(int k_tile, int (&acc)[MT][NT][4]) {
        const int thread = (int)threadIdx.x - 32 * W * group, round = k_tile / G;
        bool live[MT];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) live[mt] = (int)blockIdx.y * SUB * MT + SUB * mt + 16 * (thread >> 5) < rows;
        int part[MT][NT][4];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    part[mt][nt][i] = 14 * acc[mt][nt][i];
                    acc[mt][nt][i] = 0;
                }
        if constexpr (G > 1) {
            if (group > 0) {
                if (round > 0) repro_bar_sync<FREE>(G * 128);
                if (live[0])
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                        for (int i = 0; i < 4; ++i) slot[(nt * 4 + i) * 128 + thread] = part[0][nt][i];
                repro_bar_arrive<FULL>(G * 128);
                return;
            }
            repro_bar_sync<FULL>(G * 128);
        }
        const int ends = fold - round * G - 1;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            if (!live[mt]) continue;
            const int local = SUB * mt + 16 * (thread >> 5) + ((thread & 31) >> 2), p = round * G;
            const float now[2] = {steps(local, p), steps(local + 8, p)};
            // The parked part is the next k tile, on its own steps.
            [[maybe_unused]] float next[2];
            if constexpr (G > 1) next[0] = steps(local, p + 1), next[1] = steps(local + 8, p + 1);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float v = __fmaf_rn(__int2float_rn(part[mt][nt][i]), now[i >> 1], f[mt][nt][i]);
                    if (ends == 0) {
                        total[mt][nt][i] = fold == segment ? v : __fadd_rn(total[mt][nt][i], v);
                        v = 0.0f;
                    }
                    if constexpr (G > 1) {
                        v = __fmaf_rn(__int2float_rn(slot[(nt * 4 + i) * 128 + thread]),
                                      next[i >> 1], v);
                        if (ends == 1) {
                            total[mt][nt][i] = fold == segment ? v : __fadd_rn(total[mt][nt][i], v);
                            v = 0.0f;
                        }
                    }
                    f[mt][nt][i] = v;
                }
        }
        if (ends >= 0 && ends < G) fold += segment;
        if constexpr (G > 1)
            if (round + 1 < rounds) repro_bar_arrive<FREE>(G * 128);
    }
};

template <int NT>
struct ReproTallResidual {
    __half2 pair[2][NT];
};

// Loaded at the kernel's entry, so the loads land under the mainloop.
template <int NT, int ROWS>
__device__ __forceinline__ ReproTallResidual<NT> repro_tall_residual(const __half* __restrict__ residual, int rows,
                                                                     int n, int warp, int col0, int mrow = 0) {
    const int lane = threadIdx.x & 31;
    const int m0 = blockIdx.y * ROWS, j0 = blockIdx.x * 32;
    ReproTallResidual<NT> held{};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = min(m0 + mrow + 16 * warp + (lane >> 2) + 8 * h, rows - 1);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int j = j0 + col0 + 8 * nt + 2 * (lane & 3);
            held.pair[h][nt] = *reinterpret_cast<const __half2*>(residual + (size_t)r * n + j);
        }
    }
    return held;
}

// repro_restored_epilogue's operations, two adjacent columns stored as a pair.
template <int E, int NT, int ROWS>
__device__ __forceinline__ void repro_tall_epilogue(const float (&f)[NT][4],
                                                    const RestoredInputs<ROWS>& in,
                                                    const ReproTallResidual<NT>& residual,
                                                    bool deferred, const float* __restrict__ fc,
                                                    int rows, int n, void* __restrict__ out, int warp,
                                                    int col0, int mrow = 0) {
    const int lane = threadIdx.x & 31;
    const int m0 = blockIdx.y * ROWS, j0 = blockIdx.x * 32;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int rl = mrow + 16 * warp + (lane >> 2) + 8 * h, r = m0 + rl;
        if (r >= rows) continue;
        const float* statics = in.statics[-in.row_codes[rl] - 1];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int jl = col0 + 8 * nt + 2 * (lane & 3);
            float v[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                v[e] = __fadd_rn(__fmul_rn(f[nt][2 * h + e], in.col_scales[jl + e]), statics[jl + e]);
                if (deferred)
                    v[e] = repro_deferred_value(v[e], in.moments[rl], in.sums[jl + e], in.constants[jl + e]);
            }
            const size_t at = (size_t)r * n + j0 + jl;
            if constexpr (E == REPRO_RESIDUAL) {
                const float2 held = __half22float2(residual.pair[h][nt]);
                *reinterpret_cast<__half2*>(static_cast<__half*>(out) + at) =
                    __floats2half2_rn(__fadd_rn(v[0], held.x), __fadd_rn(v[1], held.y));
            } else if constexpr (E == REPRO_GELU) {
                *reinterpret_cast<__half2*>(static_cast<__half*>(out) + at) =
                    __floats2half2_rn(repro_gelu(v[0], fc), repro_gelu(v[1], fc));
            } else {
                *reinterpret_cast<float2*>(static_cast<float*>(out) + at) = make_float2(v[0], v[1]);
            }
        }
    }
}

// Group g takes k tiles g, g + G, ... on its own stages and barrier, the parts
// chaining across groups as they go.
template <int G, int STAGES, int W = 4>
__host__ __device__ constexpr int repro_tall_bytes() {
    return G * STAGES * repro_tall_stage<16 * W>() + sizeof(RestoredInputs<16 * W>) + (G - 1) * 16 * 128 * 4;
}
static_assert(repro_tall_bytes<2, 4>() == 49152 + 1536 + 8192 &&
                  repro_tall_bytes<1, 4>() == 24576 + 1536 &&
                  repro_tall_bytes<1, 3>() == 18432 + 1536 &&
                  repro_tall_bytes<1, 4, 8>() == 40960 + 2304 &&
                  repro_tall_bytes<1, 2, 8>() == 20480 + 2304,
              "the host adds these bytes");

// Taller tiles (W 8, 128 rows) take one group of the whole block.
template <int E, int G, int STAGES, int W = 4>
__device__ __forceinline__ void repro_project_restored_tall(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc,
    int rows, int n, int k, int segment, void* out, signed char* stages, const DeferredTables& deferred) {
    static_assert(W == 4 || G == 1, "a taller tile's parts stay in its one group");
    constexpr int ROWS = 16 * W, RING = STAGES * repro_tall_stage<ROWS>();
    signed char* inputs = stages + G * RING;
    RestoredInputs<ROWS>& in = *reinterpret_cast<RestoredInputs<ROWS>*>(inputs);
    int* slot = reinterpret_cast<int*>(inputs + sizeof(RestoredInputs<ROWS>));
    [[maybe_unused]] float* steps = reinterpret_cast<float*>(stages + repro_tall_bytes<G, STAGES, W>());
    const int group = (int)threadIdx.x / (32 * W), rounds = k / 64 / G, parts = k / HALF;
    // The tall tile stages its rows' steps; the taller one reads the table.
    constexpr bool STAGED = W == 4;
    // The restart must be whole tiles, at least a round's, and divide k.
    if (threadIdx.x == 0) {
        unsigned bytes;
        asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(bytes));
        if (segment % 64 != 0 || segment / 64 < G || k % segment != 0 ||
            bytes < (unsigned)(repro_tall_bytes<G, STAGES, W>() + (STAGED ? ROWS * parts * 4 : 0)))
            __trap();
    }
    int acc[1][4][4];
    float promoted[1][4][4] = {}, total[1][4][4] = {};
    auto chain_steps = [&]() {
        if constexpr (STAGED)
            return ReproStagedSteps<ROWS>{steps};
        else
            return ReproTableSteps{row_steps, rows, (int)blockIdx.y * ROWS, 0};
    };
    ReproTallChain<G, 1, W, decltype(chain_steps())> chain{promoted, total, slot, chain_steps(), group, rows, rounds,
                                                           segment / 64, segment / 64};
    // Group 0's copies land with its first stage. A thread stages its row's steps part after
    // part, only where the row's 16-row subtile holds a row of the input; rows past the last
    // take its steps, keeping reads in bounds.
    if (group == 0) {
        repro_restored_inputs<ROWS>(in, row_shared, col_scales, static_shared, deferred,
                                    blockIdx.y * ROWS, rows, n);
        if constexpr (STAGED) {
            const int row = (int)threadIdx.x % ROWS, m0 = (int)blockIdx.y * ROWS;
            if (m0 + row / 16 * 16 < rows)
                for (int part = (int)threadIdx.x / ROWS; part < parts; part += 32 * W / ROWS)
                    cp_async_4(shared_address(steps + part * ROWS + row),
                               repro_step_at(row_steps, rows, part, min(m0 + row, rows - 1)));
        }
    }
    if constexpr (W == 4)
        repro_tall_mainloop<STAGES, W>(qa, q, rows, k, stages + group * RING, acc, group, rounds, chain,
                                       ReproLoopGroup<G>{group});
    else
        repro_tall_mainloop<STAGES, W>(qa, q, rows, k, stages, acc, 0, rounds, chain, ReproWholeBlock{});
    if (group > 0) return;
    // The residual after the mainloop: held through it, its registers would
    // cost these tiles' blocks a multiprocessor.
    [[maybe_unused]] ReproTallResidual<4> held{};
    if constexpr (E == REPRO_RESIDUAL) held = repro_tall_residual<4, ROWS>(residual, rows, n, (int)threadIdx.x >> 5, 0);
    repro_tall_epilogue<E, 4>(total[0], in, held, deferred.moments != nullptr, fc, rows, n, out,
                              (int)threadIdx.x >> 5, 0);
}

// A segment group walks its own segment's k tiles in order.
template <int SEGS>
struct ReproSegmentGroup {
    static constexpr int step = 1;
    int index;
    __device__ __forceinline__ int thread(int threads) const { return (int)threadIdx.x - threads * index; }
    __device__ __forceinline__ void sync() const { sync_from<0>(); }
    template <int I>
    __device__ __forceinline__ void sync_from() const {
        if constexpr (I < SEGS) {
            if (index == I)
                GroupSync<I>{}();
            else
                sync_from<I + 1>();
        }
    }
};

// Holds its own copy of the chain, so the state stays in registers.
template <class Chain>
struct SegmentTiles {
    Chain chain;
    int first;
    template <class Acc>
    __device__ __forceinline__ void operator()(int k_tile, Acc& acc) { chain(k_tile - first, acc); }
};

// One group a segment, each chaining from +0 with no part handed between
// groups; group 0 joins the totals in order, ((s0 + s1) + s2) + ...
template <int SEGS, int STAGES, int MT = 1>
__device__ __forceinline__ void repro_project_restored_tall_segments(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc,
    int rows, int n, int k, int segment, void* out, signed char* stages, const DeferredTables& deferred) {
    constexpr int NT = TALL_NT, ROWS = 64 * MT, RING = STAGES * repro_tall_stage<ROWS>(), TOTALS = 16 * MT;
    signed char* inputs = stages + SEGS * RING;
    RestoredInputs<ROWS>& in = *reinterpret_cast<RestoredInputs<ROWS>*>(inputs);
    const int group = (int)threadIdx.x / 128, thread = (int)threadIdx.x - 128 * group;
    const int tiles = segment / 64;
    static_assert((SEGS - 1) * TOTALS * 128 * 4 <= SEGS * RING, "the totals meet in the stages");
    if (threadIdx.x == 0) {
        unsigned bytes;
        asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(bytes));
        if (segment % BLOCK != 0 || k != SEGS * segment ||
            bytes < (unsigned)(SEGS * RING + sizeof(RestoredInputs<ROWS>)))
            __trap();
    }
    int acc[MT][NT][4];
    float promoted[MT][NT][4] = {}, total[MT][NT][4] = {};
    using Chain = ReproTallChain<1, MT, 4, ReproTableSteps>;
    Chain chain{promoted, total, nullptr, ReproTableSteps{row_steps, rows, (int)blockIdx.y * ROWS, group * tiles},
                group, rows, tiles, tiles, tiles};
    if (group == 0)
        repro_restored_inputs<ROWS>(in, row_shared, col_scales, static_shared, deferred, blockIdx.y * ROWS, rows,
                                    n);
    repro_tall_mainloop<STAGES, 4>(qa, q, rows, k, stages + group * RING, acc, group * tiles, tiles,
                                   SegmentTiles<Chain>{chain, group * tiles}, ReproSegmentGroup<SEGS>{group});
    // Loaded under the totals' meeting: held through the mainloop, the
    // registers would spill.
    ReproTallResidual<NT> held[MT] = {};
    if (group == 0)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            held[mt] = repro_tall_residual<NT, ROWS>(residual, rows, n, (int)threadIdx.x >> 5, 0, 64 * mt);
    // Every group's mainloop done before the stages hold the totals.
    __syncthreads();
    float* totals = reinterpret_cast<float*>(stages);
    if (group > 0)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    totals[((group - 1) * TOTALS + mt * 16 + nt * 4 + i) * 128 + thread] = total[mt][nt][i];
    __syncthreads();
    if (group > 0) return;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float v = total[mt][nt][i];
#pragma unroll
                for (int s = 1; s < SEGS; ++s)
                    v = __fadd_rn(v, totals[((s - 1) * TOTALS + mt * 16 + nt * 4 + i) * 128 + thread]);
                total[mt][nt][i] = v;
            }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        repro_tall_epilogue<REPRO_RESIDUAL, NT, ROWS>(
            total[mt], in, held[mt], deferred.moments != nullptr, fc, rows, n,
            out, (int)threadIdx.x >> 5, 0, 64 * mt);
}

// One kernel per epilogue and repro.rs TALL_CONFIGS entry.
#define REPRO_TALL(EPILOGUE_NAME, EPILOGUE, G, STAGES)                                           \
    extern "C" __global__ void __launch_bounds__(128 * G)                                        \
        repro_restored_##EPILOGUE_NAME##_tall_g##G##s##STAGES(                                   \
            const signed char* qa, const float* row_steps, const signed char* q,                 \
            const float* col_scales, const float* static_shared, const int* row_shared,          \
            const __half* residual, const float* fc, int rows, int n, int k, int segment,        \
            void* out, const float2* moments, const float* sums, const float* constants) {       \
        extern __shared__ __align__(128) signed char stages[];                                   \
        repro_project_restored_tall<EPILOGUE, G, STAGES>(                                        \
            qa, row_steps, q, col_scales, static_shared, row_shared, residual, fc, rows, n, k,   \
            segment, out, stages, DeferredTables{moments, sums, constants});                     \
    }
#define REPRO_TALL_CONFIGS(EPILOGUE_NAME, EPILOGUE)                                              \
    REPRO_TALL(EPILOGUE_NAME, EPILOGUE, 2, 4)                                                    \
    REPRO_TALL(EPILOGUE_NAME, EPILOGUE, 1, 4)                                                    \
    REPRO_TALL(EPILOGUE_NAME, EPILOGUE, 1, 3)
REPRO_TALL_CONFIGS(plain, REPRO_PLAIN)
REPRO_TALL_CONFIGS(residual, REPRO_RESIDUAL)
REPRO_TALL_CONFIGS(gelu, REPRO_GELU)

// repro.rs TALL_W8_CONFIGS: 128-row tiles on eight warps.
#define REPRO_TALL_W8(EPILOGUE_NAME, EPILOGUE, STAGES)                                           \
    extern "C" __global__ void __launch_bounds__(256)                                            \
        repro_restored_##EPILOGUE_NAME##_tall_w8s##STAGES(                                       \
            const signed char* qa, const float* row_steps, const signed char* q,                 \
            const float* col_scales, const float* static_shared, const int* row_shared,          \
            const __half* residual, const float* fc, int rows, int n, int k, int segment,        \
            void* out, const float2* moments, const float* sums, const float* constants) {       \
        extern __shared__ __align__(128) signed char stages[];                                   \
        repro_project_restored_tall<EPILOGUE, 1, STAGES, 8>(                                     \
            qa, row_steps, q, col_scales, static_shared, row_shared, residual, fc, rows, n, k,   \
            segment, out, stages, DeferredTables{moments, sums, constants});                     \
    }
#define REPRO_TALL_W8_CONFIGS(EPILOGUE_NAME, EPILOGUE)                                           \
    REPRO_TALL_W8(EPILOGUE_NAME, EPILOGUE, 4)                                                    \
    REPRO_TALL_W8(EPILOGUE_NAME, EPILOGUE, 2)
REPRO_TALL_W8_CONFIGS(plain, REPRO_PLAIN)
REPRO_TALL_W8_CONFIGS(residual, REPRO_RESIDUAL)
REPRO_TALL_W8_CONFIGS(gelu, REPRO_GELU)
#undef REPRO_TALL_W8_CONFIGS
#undef REPRO_TALL_W8

// repro.rs TALL_SEGMENTS.
extern "C" __global__ void __launch_bounds__(128 * 4) repro_restored_residual_tall_seg4s3(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc, int rows, int n,
    int k, int segment, void* out, const float2* moments, const float* sums, const float* constants) {
    extern __shared__ __align__(128) signed char stages[];
    repro_project_restored_tall_segments<4, 3>(qa, row_steps, q, col_scales, static_shared, row_shared, residual,
                                               fc, rows, n, k, segment, out, stages,
                                               DeferredTables{moments, sums, constants});
}

extern "C" __global__ void __launch_bounds__(128 * 4) repro_restored_residual_tall_seg4m2s2(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc, int rows, int n,
    int k, int segment, void* out, const float2* moments, const float* sums, const float* constants) {
    extern __shared__ __align__(128) signed char stages[];
    repro_project_restored_tall_segments<4, 2, 2>(qa, row_steps, q, col_scales, static_shared, row_shared,
                                                  residual, fc, rows, n, k, segment, out, stages,
                                                  DeferredTables{moments, sums, constants});
}

// Tall tiles reuse each staged weight across 128 rows while retaining
// room for two blocks per multiprocessor.
using RestoredWide = int8proj::Tiling<128, 64, 64, 32, 4>;
static_assert(RestoredWide::THREADS == 128 && RestoredWide::SHARED_BYTES == 49152,
              "the host launches 128 threads with 48 KiB");

// A tile row's terms, read at the kernel's entry so their latency passes under the
// mainloop, where the tile leaves them registers (repro_project_restored_wide).
struct ReproRowTerms {
    int code;
    float2 moments;
    int sequence;
};

template <class T, int E>
__device__ __forceinline__ ReproRowTerms repro_tile_row_terms(const int* __restrict__ row_shared,
                                                              const DeferredTables& deferred,
                                                              const ReproAttentionPrep& prep, int rows) {
    static_assert(T::BM <= T::THREADS, "a thread a tile row");
    ReproRowTerms terms{0, make_float2(0.0f, 0.0f), 0};
    if (threadIdx.x < T::BM) {
        const int r = min((int)(blockIdx.y * T::BM + threadIdx.x), rows - 1);
        terms.code = row_shared[r];
        if (deferred.moments != nullptr) terms.moments = deferred.moments[r];
        if constexpr (E == REPRO_QKV || E == REPRO_QKV_PACKED) terms.sequence = prep.sequence_codes[r];
    }
    return terms;
}

// REPRO_QKV makes attention's operands from repro_attention_prep's FP32 values;
// REPRO_GELU_BLOCKS quantizes instead of storing FP16 rows.
template <class T, int E>
__device__ __forceinline__ void repro_restored_staged_epilogue(
    const float (&f)[T::MT][T::NT][4], const float* __restrict__ col_scales, const float* __restrict__ static_shared,
    const DeferredTables& deferred, const __half* __restrict__ residual, const float* __restrict__ fc, int rows,
    int n, void* __restrict__ out, signed char* stages, const ReproRowTerms& terms,
    const ReproAttentionPrep& prep = ReproAttentionPrep{}, int head_width = 0,
    const ReproBlocksOut& blocks = ReproBlocksOut{}, const ReproPackedValues& packed = ReproPackedValues{}) {
    constexpr bool QKV = E == REPRO_QKV || E == REPRO_QKV_PACKED;
    constexpr int PITCH = T::BN + 8, WARP_ROWS = T::BM / (T::THREADS / 32);
    constexpr int ROW_LANES = T::BN / 4, PASS_ROWS = 32 / ROW_LANES, PASSES = WARP_ROWS / PASS_ROWS;
    static_assert(T::THREADS % 32 == 0 && T::BM % (T::THREADS / 32) == 0, "whole rows per epilogue warp");
    static_assert(32 % ROW_LANES == 0 && WARP_ROWS % PASS_ROWS == 0, "whole rows a pass");
    constexpr int MAXIMA_BYTES = E == REPRO_QKV_PACKED ? T::THREADS * 4 : 0;
    static_assert(T::BM * PITCH * 4 + T::BM * 16 + MAXIMA_BYTES <= T::SHARED_BYTES, "the tile and its rows fit");
    int* tile = reinterpret_cast<int*>(stages);
    int* row_code = tile + T::BM * PITCH;
    float2* row_moments = reinterpret_cast<float2*>(row_code + T::BM);
    [[maybe_unused]] int* row_sequence = reinterpret_cast<int*>(row_moments + T::BM);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warp_m = warp / T::WARPS_N, warp_n = warp % T::WARPS_N;
    const int m0 = blockIdx.y * T::BM, n0 = blockIdx.x * T::BN;
    const int sub = lane / ROW_LANES, cl0 = 4 * (lane % ROW_LANES), c = n0 + cl0;
    const bool defer = deferred.moments != nullptr;
    const float4 sw = *reinterpret_cast<const float4*>(col_scales + c);
    const float4 su = defer ? *reinterpret_cast<const float4*>(deferred.sums + c) : make_float4(0, 0, 0, 0);
    const float4 co = defer ? *reinterpret_cast<const float4*>(deferred.constants + c) : make_float4(0, 0, 0, 0);
    // Part 0 is queries, 1 keys, 2 values.
    [[maybe_unused]] const int hidden = n / 3, part = n0 / hidden, pc = c - part * hidden;
    [[maybe_unused]] const int head = QKV ? pc / head_width : 0, within = pc - head * head_width;
    [[maybe_unused]] float4 kc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if constexpr (QKV)
        if (part == 1) kc = *reinterpret_cast<const float4*>(prep.key_centers + pc);
    [[maybe_unused]] float4 down_signs = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if constexpr (E == REPRO_GELU_BLOCKS) down_signs = *reinterpret_cast<const float4*>(blocks.signs + c);
    // Issue independent epilogue loads early to overlap their latency with staging.
    const ReproClassRows classes = repro_class_rows(static_shared, n, c);
    [[maybe_unused]] ReproGelu gelu{};
    [[maybe_unused]] float rotate = 0.0f;
    if constexpr (E == REPRO_GELU || E == REPRO_GELU_BLOCKS) gelu = repro_gelu_constants(fc);
    if constexpr (E == REPRO_GELU_BLOCKS) rotate = fc[FC_RECIP_SQRT_BLOCK];
    [[maybe_unused]] uint2 residual_rows[PASSES];
    if constexpr (E == REPRO_RESIDUAL) {
#pragma unroll
        for (int i = 0; i < PASSES; ++i) {
            const int r = m0 + warp * WARP_ROWS + i * PASS_ROWS + sub;
            residual_rows[i] =
                r < rows ? *reinterpret_cast<const uint2*>(residual + (size_t)r * n + c) : make_uint2(0u, 0u);
        }
    }
    // The epilogue reuses the operand tiles' shared memory.
    __syncthreads();
#pragma unroll
    for (int mt = 0; mt < T::MT; ++mt)
#pragma unroll
        for (int half_row = 0; half_row < 2; ++half_row)
#pragma unroll
            for (int nt = 0; nt < T::NT; ++nt) {
                const int rl = warp_m * T::WM + mt * 16 + (lane >> 2) + half_row * 8;
                const int cl = warp_n * T::WN + nt * 8 + 2 * (lane & 3);
                *reinterpret_cast<int2*>(tile + rl * PITCH + cl) =
                    make_int2(__float_as_int(f[mt][nt][2 * half_row]), __float_as_int(f[mt][nt][2 * half_row + 1]));
            }
    if (threadIdx.x < T::BM) {
        row_code[threadIdx.x] = terms.code;
        if (defer) row_moments[threadIdx.x] = terms.moments;
        if constexpr (QKV) row_sequence[threadIdx.x] = terms.sequence;
    }
    __syncthreads();
    // repro_tall_epilogue's operations on local row rl.
    auto row_values = [&](int rl, float (&v)[4]) {
        const float4 fixed = repro_class_of(classes, row_code[rl]);
        const int4 a = *reinterpret_cast<const int4*>(tile + rl * PITCH + cl0);
        const int bits[4] = {a.x, a.y, a.z, a.w};
        const float scale[4] = {sw.x, sw.y, sw.z, sw.w}, shared_row[4] = {fixed.x, fixed.y, fixed.z, fixed.w};
        const float sum[4] = {su.x, su.y, su.z, su.w}, constant[4] = {co.x, co.y, co.z, co.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] = __fadd_rn(__fmul_rn(__int_as_float(bits[j]), scale[j]), shared_row[j]);
            if (defer) v[j] = repro_deferred_value(v[j], row_moments[rl], sum[j], constant[j]);
        }
    };
    if constexpr (QKV) {
        if (part < 2) {
            // Rows reduce level by level so their shuffles overlap, each step
            // over every row first, as repro_on_steps does; extra rows store nothing.
            float x[PASSES][4];
            unsigned most_bits[PASSES];
#pragma unroll
            for (int i = 0; i < PASSES; ++i) {
                row_values(warp * WARP_ROWS + i * PASS_ROWS + sub, x[i]);
                if (part == 1) {
                    x[i][0] = __fsub_rn(x[i][0], kc.x);
                    x[i][1] = __fsub_rn(x[i][1], kc.y);
                    x[i][2] = __fsub_rn(x[i][2], kc.z);
                    x[i][3] = __fsub_rn(x[i][3], kc.w);
                }
                most_bits[i] = 0u;
#pragma unroll
                for (int j = 0; j < 4; ++j) most_bits[i] = max(most_bits[i], __float_as_uint(x[i][j]) & 0x7fffffffu);
            }
            for (int o = 1; o < head_width / 4; o *= 2)
#pragma unroll
                for (int i = 0; i < PASSES; ++i)
                    most_bits[i] = max(most_bits[i], __shfl_xor_sync(0xffffffffu, most_bits[i], o));
            float scale[PASSES], reciprocal[PASSES];
#pragma unroll
            for (int i = 0; i < PASSES; ++i) scale[i] = repro_scale_for_127(__uint_as_float(most_bits[i]));
#pragma unroll
            for (int i = 0; i < PASSES; ++i) reciprocal[i] = repro_step_reciprocal(scale[i]);
#pragma unroll
            for (int i = 0; i < PASSES; ++i) {
                const int rl = warp * WARP_ROWS + i * PASS_ROWS + sub, r = m0 + rl;
                if (r >= rows) break;
                int b[4];
                repro_on_steps<4>(x[i], scale[i], reciprocal[i], b);
                const size_t row = (size_t)head * rows + r;
                signed char* packed = part == 0 ? prep.queries : prep.keys;
                *reinterpret_cast<unsigned*>(packed + row * head_width + within) =
                    repro_low_bytes(b[0], b[1], b[2], b[3]);
                if (within == 0) {
                    (part == 0 ? prep.query_scales : prep.key_scales)[row] = scale[i];
                    if (most_bits[i] >= 0x7f800000u)
                        atomicOr(prep.failed + (size_t)(row_sequence[rl] >> 1) * prep.heads + head, 1);
                }
            }
        } else if constexpr (E == REPRO_QKV) {
            // One atomic maximum a run of a sequence's rows, not a row.
            int run = -1;
            unsigned most[4] = {0u, 0u, 0u, 0u};
            bool run_finite = true;
            auto end_run = [&]() {
                unsigned* top = prep.value_magnitudes + (size_t)run * hidden + pc;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (most[j]) atomicMax(top + j, most[j]);
                if (!run_finite) atomicOr(prep.failed + (size_t)run * prep.heads + head, 1);
            };
#pragma unroll 4
            for (int i = 0; i < PASSES; ++i) {
                const int rl = warp * WARP_ROWS + i * PASS_ROWS + sub, r = m0 + rl;
                if (r >= rows) break;
                float v[4];
                row_values(rl, v);
                const __half2 lo = __floats2half2_rn(v[0], v[1]), hi = __floats2half2_rn(v[2], v[3]);
                *reinterpret_cast<uint2*>(prep.values16 + (size_t)r * hidden + pc) =
                    make_uint2(*reinterpret_cast<const unsigned*>(&lo), *reinterpret_cast<const unsigned*>(&hi));
                const float h[4] = {__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi)};
                const int sequence = row_sequence[rl] >> 1;
                const bool sink = row_sequence[rl] & 1;
                if (sequence != run) {
                    if (run >= 0) end_run();
                    run = sequence;
                    run_finite = true;
#pragma unroll
                    for (int j = 0; j < 4; ++j) most[j] = 0u;
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    run_finite = run_finite && isfinite(h[j]);
                    if (!sink) most[j] = max(most[j], __float_as_uint(h[j]) & 0x7fffffffu);
                }
            }
            if (run >= 0) end_run();
        } else {
            // Channels lie across a warp's lanes; each sequence's bounds load
            // a sequence ahead of its pass, the first's before the tile's values.
            int sequence[3] = {0, 0, 0};
            auto load_sequence = [&](int s) {
                sequence[0] = packed.starts[s];
                sequence[1] = packed.lengths[s];
                sequence[2] = packed.pstarts[s];
            };
            if (packed.sequences > 0) load_sequence(0);
#pragma unroll
            for (int i = 0; i < PASSES; ++i) {
                const int rl = warp * WARP_ROWS + i * PASS_ROWS + sub;
                if (m0 + rl >= rows) break;
                float v[4];
                row_values(rl, v);
                *reinterpret_cast<int4*>(tile + rl * PITCH + cl0) =
                    make_int4(__float_as_int(fp16_value(v[0])), __float_as_int(fp16_value(v[1])),
                              __float_as_int(fp16_value(v[2])), __float_as_int(fp16_value(v[3])));
            }
            constexpr int PARTS = T::THREADS / T::BN;
            static_assert(T::THREADS % T::BN == 0 && 16 * PARTS >= T::BM, "parts hold every row");
            unsigned* maxima = reinterpret_cast<unsigned*>(row_sequence + T::BM);
            const int cc = threadIdx.x % T::BN, q = threadIdx.x / T::BN, channel = n0 - 2 * hidden + cc;
            for (int s = 0; s < packed.sequences; ++s) {
                const int start = sequence[0], len = sequence[1], pstart = sequence[2];
                if (s + 1 < packed.sequences) load_sequence(s + 1);
                __syncthreads();
                float x[16];
                unsigned most = 0;
                bool finite = true;
                float last = 0.0f;
                // Tokens past the sequence take zero, which leaves the flag and
                // the largest magnitude alone: selects, no branch.
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    const int j = 16 * q + e;
                    const float value = __int_as_float(tile[(start - m0 + min(j, len - 1)) * PITCH + cc]);
                    x[e] = j < len ? value : 0.0f;
                    finite = finite && isfinite(x[e]);
                    if (j != 0 && j != len - 1) most = max(most, __float_as_uint(x[e]) & 0x7fffffffu);
                    if (j == len - 1) last = x[e];
                }
                maxima[threadIdx.x] = most;
                if (!finite) atomicOr(prep.failed + (size_t)s * prep.heads + channel / head_width, 1);
                __syncthreads();
#pragma unroll
                for (int p = 0; p < PARTS; ++p) most = max(most, maxima[p * T::BN + cc]);
                const float scale = repro_scale_for_127(__uint_as_float(most));
                int b[16];
                repro_on_steps<16>(x, scale, repro_step_reciprocal(scale), b);
                unsigned words[4];
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    unsigned word = 0;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int j = 16 * q + 4 * w + e;
                        const int byte = j < len && j != 0 && j != len - 1 ? b[4 * w + e] : 0;
                        word |= (unsigned)(byte & 0xff) << (8 * e);
                    }
                    words[w] = word;
                }
                // A sequence's plane holds its tokens padded to whole key
                // chunks.
                if (16 * q < (len + ATT_CHUNK - 1) / ATT_CHUNK * ATT_CHUNK)
                    *reinterpret_cast<uint4*>(packed.planes + (size_t)channel * packed.ptokens + pstart + 16 * q) =
                        make_uint4(words[0], words[1], words[2], words[3]);
                if (q == 0) {
                    packed.value_scales[(size_t)s * hidden + channel] = scale;
                    prep.values16[(size_t)start * hidden + channel] = __float2half_rn(x[0]);
                }
                if ((len - 1) / 16 == q) prep.values16[(size_t)(start + len - 1) * hidden + channel] = __float2half_rn(last);
            }
        }
    } else if constexpr (E == REPRO_GELU_BLOCKS) {
        // Centers load first, their latency under the values'; GELU_ROWS rows
        // at a time let eight values' chains overlap in a thread's registers. Three stages
        // run three blocks a multiprocessor, whose budget holds four chains: with eight the
        // divider's slow path finds no free register and takes a stack frame.
        static_assert(T::BN == BLOCK && ROW_LANES == 32, "a warp holds a row's 128-block");
        constexpr int GELU_ROWS = PASSES < 2 || T::STAGES < 4 ? 1 : 2;
        static_assert(PASSES % GELU_ROWS == 0, "whole groups of rows");
        const float sign[4] = {down_signs.x < 0.0f ? -1.0f : 1.0f, down_signs.y < 0.0f ? -1.0f : 1.0f,
                               down_signs.z < 0.0f ? -1.0f : 1.0f, down_signs.w < 0.0f ? -1.0f : 1.0f};
        float4 centers[PASSES];
#pragma unroll
        for (int i = 0; i < PASSES; ++i) {
            const size_t class_row = (size_t)(-row_code[warp * WARP_ROWS + i * PASS_ROWS + sub] - 1) * n + c;
            centers[i] = __ldg(reinterpret_cast<const float4*>(blocks.centers + class_row));
        }
        float x[PASSES][4], most[PASSES];
#pragma unroll
        for (int i0 = 0; i0 < PASSES; i0 += GELU_ROWS) {
            float g[GELU_ROWS][4];
#pragma unroll
            for (int r = 0; r < GELU_ROWS; ++r) row_values(warp * WARP_ROWS + (i0 + r) * PASS_ROWS + sub, g[r]);
            repro_gelus(g, gelu);
#pragma unroll
            for (int r = 0; r < GELU_ROWS; ++r) {
                const float center[4] = {centers[i0 + r].x, centers[i0 + r].y, centers[i0 + r].z, centers[i0 + r].w};
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    x[i0 + r][j] = __fmul_rn(__fsub_rn(fp16_value(g[r][j]), center[j]), sign[j]);
            }
        }
        repro_rotate_rows<4, PASSES>(x, lane, rotate, most);
        float scale[PASSES], reciprocal[PASSES];
#pragma unroll
        for (int i = 0; i < PASSES; ++i) scale[i] = most[i] > 0.0f ? __fdiv_rn(most[i], 127.0f) : 1.0f;
#pragma unroll
        for (int i = 0; i < PASSES; ++i) reciprocal[i] = repro_step_reciprocal(scale[i]);
#pragma unroll
        for (int i = 0; i < PASSES; ++i) {
            const int r = m0 + warp * WARP_ROWS + i * PASS_ROWS + sub;
            if (r >= rows) break;
            repro_store_block<4>(x[i], scale[i], reciprocal[i], blocks.q + (size_t)r * n + c);
            if (lane % 16 == 0) *repro_step_at(blocks.steps, rows, 2 * (int)blockIdx.x + lane / 16, r) = scale[i];
        }
    } else {
        // The residual rows unrolled whole, so each takes its loaded word.
        constexpr int UNROLL = E == REPRO_RESIDUAL ? PASSES : 4;
#pragma unroll UNROLL
        for (int i = 0; i < PASSES; ++i) {
            const int rl = warp * WARP_ROWS + i * PASS_ROWS + sub, r = m0 + rl;
            if (r >= rows) break;
            float v[4];
            row_values(rl, v);
            const size_t at = (size_t)r * n + c;
            if constexpr (E == REPRO_RESIDUAL) {
                const __half2* pair = reinterpret_cast<const __half2*>(&residual_rows[i]);
                const float2 lo = __half22float2(pair[0]), hi = __half22float2(pair[1]);
                v[0] = __fadd_rn(v[0], lo.x);
                v[1] = __fadd_rn(v[1], lo.y);
                v[2] = __fadd_rn(v[2], hi.x);
                v[3] = __fadd_rn(v[3], hi.y);
                *reinterpret_cast<uint2*>(static_cast<__half*>(out) + at) = pack_half4(make_float4(v[0], v[1], v[2], v[3]));
            } else if constexpr (E == REPRO_GELU) {
                float g[1][4] = {{v[0], v[1], v[2], v[3]}};
                repro_gelus(g, gelu);
                *reinterpret_cast<uint2*>(static_cast<__half*>(out) + at) =
                    pack_half4(make_float4(g[0][0], g[0][1], g[0][2], g[0][3]));
            } else {
                *reinterpret_cast<float4*>(static_cast<float*>(out) + at) = make_float4(v[0], v[1], v[2], v[3]);
            }
        }
    }
}

// The chain's operations match the tall tile's on the same operands, so the
// tiling cannot change a bit. TWO: the operand's two digits, its chain twice the weights' depth.
template <class T, int E, bool TWO = false>
__device__ __forceinline__ void repro_project_restored_wide(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc,
    int rows, int n, int k, int segment, void* out, signed char* stages, const DeferredTables& deferred,
    const ReproAttentionPrep& prep = ReproAttentionPrep{}, int head_width = 0,
    const ReproBlocksOut& blocks = ReproBlocksOut{}, const ReproPackedValues& packed = ReproPackedValues{}) {
    if (threadIdx.x == 0) {
        unsigned bytes;
        asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(bytes));
        if (segment % BLOCK != 0 || segment != (TWO ? 2 * k : k) || bytes < (unsigned)T::SHARED_BYTES) __trap();
        // The epilogue needs whole heads in a tile and a tile within one of
        // Q, K and V.
        if ((E == REPRO_QKV || E == REPRO_QKV_PACKED) &&
            (head_width < 8 || head_width % 4 != 0 || T::BN % head_width != 0 || n % 3 != 0 ||
             (n / 3) % T::BN != 0))
            __trap();
        if (E == REPRO_QKV_PACKED && rows > T::BM) __trap();
    }
    // The 128-row tile's sums and the blocks tile's budget leave the row terms no register
    // through the mainloop, where the assembler keeps them on the stack: those tiles read
    // them behind it, with the epilogue's column terms.
    constexpr bool LATE_TERMS = T::MT >= 4 || E == REPRO_GELU_BLOCKS;
    const ReproRowTerms entry =
        LATE_TERMS ? ReproRowTerms{} : repro_tile_row_terms<T, E>(row_shared, deferred, prep, rows);
    float f[T::MT][T::NT][4];
    const int depth = TWO ? 2 * k : k;
    repro_promoted_mainloop<T, 14, TWO>(qa, q, row_steps, rows, depth, stages, f, 0, depth / T::BK);
    const ReproRowTerms terms =
        LATE_TERMS ? repro_tile_row_terms<T, E>(row_shared, deferred, prep, rows) : entry;
    repro_restored_staged_epilogue<T, E>(
        f, col_scales, static_shared, deferred, residual, fc, rows, n, out, stages, terms, prep, head_width, blocks, packed);
}

#define REPRO_RESTORED_WIDE(EPILOGUE_NAME, EPILOGUE)                                             \
    extern "C" __global__ void __launch_bounds__(RestoredWide::THREADS)                          \
        repro_restored_##EPILOGUE_NAME##_wide(                                                   \
            const signed char* qa, const float* row_steps, const signed char* q,                 \
            const float* col_scales, const float* static_shared, const int* row_shared,          \
            const __half* residual, const float* fc, int rows, int n, int k, int segment,        \
            void* out, const float2* moments, const float* sums, const float* constants) {       \
        extern __shared__ __align__(128) signed char stages[];                                   \
        repro_project_restored_wide<RestoredWide, EPILOGUE>(                                     \
            qa, row_steps, q, col_scales, static_shared, row_shared, residual, fc, rows, n, k,   \
            segment, out, stages, DeferredTables{moments, sums, constants});                     \
    }
REPRO_RESTORED_WIDE(plain, REPRO_PLAIN)
REPRO_RESTORED_WIDE(residual, REPRO_RESIDUAL)
REPRO_RESTORED_WIDE(gelu, REPRO_GELU)
#undef REPRO_RESTORED_WIDE

// Takes maxima and flags into scratch AttentionClear zeroed.
extern "C" __global__ void __launch_bounds__(RestoredWide::THREADS) repro_restored_qkv_wide(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc, int rows, int n,
    int k, int segment, void* out, const float2* moments, const float* sums, const float* constants,
    signed char* queries, float* query_scales, const float* key_centers, signed char* keys, float* key_scales,
    __half* values16, unsigned* value_magnitudes, int* failed, const int* sequence_codes, int heads, int head_width) {
    extern __shared__ __align__(128) signed char stages[];
    repro_project_restored_wide<RestoredWide, REPRO_QKV>(
        qa, row_steps, q, col_scales, static_shared, row_shared, residual, fc, rows, n, k, segment, out, stages,
        DeferredTables{moments, sums, constants},
        ReproAttentionPrep{queries, query_scales, key_centers, keys, key_scales, values16, value_magnitudes, failed,
                           sequence_codes, heads},
        head_width);
}

// For the down projection, and where 128 x 64 tiles would run alone.
using RestoredWideDown = int8proj::Tiling<64, 64, 32, 32, 4>;
static_assert(RestoredWideDown::THREADS == 128 && RestoredWideDown::SHARED_BYTES == 32768,
              "the host launches 128 threads with 32 KiB");

// Segments chain from +0 in turn and join in order, ((s0 + s1) + s2) + ...,
// as the tall segment groups join theirs.
template <class T>
__device__ __forceinline__ void repro_project_restored_wide_segments(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc,
    int rows, int n, int k, int segment, void* out, signed char* stages, const DeferredTables& deferred) {
    if (threadIdx.x == 0) {
        unsigned bytes;
        asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(bytes));
        if (segment % BLOCK != 0 || k % segment != 0 || bytes < (unsigned)T::SHARED_BYTES) __trap();
    }
    const int tiles = segment / T::BK;
    const ReproRowTerms terms = repro_tile_row_terms<T, REPRO_RESIDUAL>(row_shared, deferred, ReproAttentionPrep{}, rows);
    float f[T::MT][T::NT][4], total[T::MT][T::NT][4];
    for (int s = 0; s < k / segment; ++s) {
        repro_promoted_mainloop<T, 14>(qa, q, row_steps, rows, k, stages, f, s * tiles, tiles);
#pragma unroll
        for (int mt = 0; mt < T::MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < T::NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    total[mt][nt][i] = s == 0 ? f[mt][nt][i] : __fadd_rn(total[mt][nt][i], f[mt][nt][i]);
        // Every warp is done with the stages before the next segment fills them.
        __syncthreads();
    }
    repro_restored_staged_epilogue<T, REPRO_RESIDUAL>(total, col_scales, static_shared, deferred, residual, fc, rows,
                                                      n, out, stages, terms);
}

extern "C" __global__ void __launch_bounds__(RestoredWideDown::THREADS) repro_restored_residual_wide_segments(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc, int rows, int n,
    int k, int segment, void* out, const float2* moments, const float* sums, const float* constants) {
    extern __shared__ __align__(128) signed char stages[];
    repro_project_restored_wide_segments<RestoredWideDown>(qa, row_steps, q, col_scales, static_shared, row_shared,
                                                           residual, fc, rows, n, k, segment, out, stages,
                                                           DeferredTables{moments, sums, constants});
}

// 64 x 128 tiles draw the least power for their products: a layer's four
// projections take them once the narrowest's grid reaches every multiprocessor.
using RestoredWide64x128 = int8proj::Tiling<64, 128, 32, 64, 4>;
static_assert(RestoredWide64x128::THREADS == 128 && RestoredWide64x128::SHARED_BYTES == 49152,
              "the host launches 128 threads with 48 KiB");

extern "C" __global__ void __launch_bounds__(RestoredWide64x128::THREADS) repro_restored_residual_wide64x128(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc, int rows, int n,
    int k, int segment, void* out, const float2* moments, const float* sums, const float* constants) {
    extern __shared__ __align__(128) signed char stages[];
    repro_project_restored_wide<RestoredWide64x128, REPRO_RESIDUAL>(
        qa, row_steps, q, col_scales, static_shared, row_shared, residual, fc, rows, n, k, segment, out, stages,
        DeferredTables{moments, sums, constants});
}

extern "C" __global__ void __launch_bounds__(RestoredWide64x128::THREADS) repro_restored_gelu_wide64x128(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc, int rows, int n,
    int k, int segment, void* out, const float2* moments, const float* sums, const float* constants) {
    extern __shared__ __align__(128) signed char stages[];
    repro_project_restored_wide<RestoredWide64x128, REPRO_GELU>(qa, row_steps, q, col_scales, static_shared,
                                                                row_shared, residual, fc, rows, n, k, segment, out,
                                                                stages, DeferredTables{moments, sums, constants});
}

extern "C" __global__ void __launch_bounds__(RestoredWide64x128::THREADS) repro_restored_qkv_wide64x128(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc, int rows, int n,
    int k, int segment, void* out, const float2* moments, const float* sums, const float* constants,
    signed char* queries, float* query_scales, const float* key_centers, signed char* keys, float* key_scales,
    __half* values16, unsigned* value_magnitudes, int* failed, const int* sequence_codes, int heads, int head_width) {
    extern __shared__ __align__(128) signed char stages[];
    repro_project_restored_wide<RestoredWide64x128, REPRO_QKV>(
        qa, row_steps, q, col_scales, static_shared, row_shared, residual, fc, rows, n, k, segment, out, stages,
        DeferredTables{moments, sums, constants},
        ReproAttentionPrep{queries, query_scales, key_centers, keys, key_scales, values16, value_magnitudes, failed,
                           sequence_codes, heads},
        head_width);
}

extern "C" __global__ void __launch_bounds__(RestoredWide64x128::THREADS)
    repro_restored_residual_wide64x128_segments(
        const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
        const float* static_shared, const int* row_shared, const __half* residual, const float* fc, int rows, int n,
        int k, int segment, void* out, const float2* moments, const float* sums, const float* constants) {
    extern __shared__ __align__(128) signed char stages[];
    repro_project_restored_wide_segments<RestoredWide64x128>(qa, row_steps, q, col_scales, static_shared, row_shared,
                                                             residual, fc, rows, n, k, segment, out, stages,
                                                             DeferredTables{moments, sums, constants});
}

// For grids whose 128 x 64 tiles would leave multiprocessors a lone tile.
extern "C" __global__ void __launch_bounds__(RestoredWideDown::THREADS) repro_restored_qkv_wide64(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc, int rows, int n,
    int k, int segment, void* out, const float2* moments, const float* sums, const float* constants,
    signed char* queries, float* query_scales, const float* key_centers, signed char* keys, float* key_scales,
    __half* values16, unsigned* value_magnitudes, int* failed, const int* sequence_codes, int heads, int head_width) {
    extern __shared__ __align__(128) signed char stages[];
    repro_project_restored_wide<RestoredWideDown, REPRO_QKV>(
        qa, row_steps, q, col_scales, static_shared, row_shared, residual, fc, rows, n, k, segment, out, stages,
        DeferredTables{moments, sums, constants},
        ReproAttentionPrep{queries, query_scales, key_centers, keys, key_scales, values16, value_magnitudes, failed,
                           sequence_codes, heads},
        head_width);
}

extern "C" __global__ void __launch_bounds__(RestoredWideDown::THREADS) repro_restored_residual_wide64(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc, int rows, int n,
    int k, int segment, void* out, const float2* moments, const float* sums, const float* constants) {
    extern __shared__ __align__(128) signed char stages[];
    repro_project_restored_wide<RestoredWideDown, REPRO_RESIDUAL>(qa, row_steps, q, col_scales, static_shared,
                                                                  row_shared, residual, fc, rows, n, k, segment, out,
                                                                  stages, DeferredTables{moments, sums, constants});
}

// Few rows: each warp's chain is its own, so no group hands one across
// barriers.
using RestoredSmall = int8proj::Tiling<32, 32, 16, 16, 4>;
static_assert(RestoredSmall::THREADS == 128 && RestoredSmall::SHARED_BYTES == 16384,
              "the host launches 128 threads with 16 KiB");

extern "C" __global__ void __launch_bounds__(RestoredSmall::THREADS) repro_restored_residual_small(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc, int rows, int n,
    int k, int segment, void* out, const float2* moments, const float* sums, const float* constants) {
    extern __shared__ __align__(128) signed char stages[];
    repro_project_restored_wide<RestoredSmall, REPRO_RESIDUAL>(qa, row_steps, q, col_scales, static_shared,
                                                               row_shared, residual, fc, rows, n, k, segment, out,
                                                               stages, DeferredTables{moments, sums, constants});
}

// Smaller row tiles reduce underfilled rounds for short GELU projections.
extern "C" __global__ void __launch_bounds__(RestoredWideDown::THREADS) repro_restored_gelu_wide64(
    const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,
    const float* static_shared, const int* row_shared, const __half* residual, const float* fc, int rows, int n,
    int k, int segment, void* out, const float2* moments, const float* sums, const float* constants) {
    extern __shared__ __align__(128) signed char stages[];
    repro_project_restored_wide<RestoredWideDown, REPRO_GELU>(qa, row_steps, q, col_scales, static_shared,
                                                              row_shared, residual, fc, rows, n, k, segment, out,
                                                              stages, DeferredTables{moments, sums, constants});
}

// 128 columns, so a tile holds each row's 128-block whole.
template <int STAGES>
using RestoredBlocks = int8proj::Tiling<32, 128, 16, 32, STAGES>;
static_assert(RestoredBlocks<4>::THREADS == 256 && RestoredBlocks<4>::SHARED_BYTES == 40960 &&
                  RestoredBlocks<3>::SHARED_BYTES == 30720,
              "the host launches 256 threads with 40 or 30 KiB");

// Each depth binds its blocks a multiprocessor (repro.rs gelu_blocks_depth); the _two kernels
// take an operand of two digits.
#define REPRO_RESTORED_GELU_BLOCKS(NAME, STAGES, BLOCKS, TWO)                                            \
    extern "C" __global__ void __launch_bounds__(RestoredBlocks<STAGES>::THREADS, BLOCKS)               \
        repro_restored_##NAME(                                                                          \
        const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,   \
        const float* static_shared, const int* row_shared, const __half* residual, const float* fc,     \
        int rows, int n, int k, int segment, void* out, const float2* moments, const float* sums,       \
        const float* constants, const float* next_centers, const float* next_signs, signed char* next_q, \
        float* next_steps) {                                                                            \
        extern __shared__ __align__(128) signed char stages[];                                          \
        repro_project_restored_wide<RestoredBlocks<STAGES>, REPRO_GELU_BLOCKS, TWO>(                    \
            qa, row_steps, q, col_scales, static_shared, row_shared, residual, fc, rows, n, k, segment,  \
            out, stages, DeferredTables{moments, sums, constants}, ReproAttentionPrep{}, 0,             \
            ReproBlocksOut{next_q, next_steps, next_signs, next_centers});                              \
    }
REPRO_RESTORED_GELU_BLOCKS(gelu_blocks, 4, 2, false)
REPRO_RESTORED_GELU_BLOCKS(gelu_blocks3, 3, 3, false)
REPRO_RESTORED_GELU_BLOCKS(gelu_blocks_two, 4, 2, true)
REPRO_RESTORED_GELU_BLOCKS(gelu_blocks3_two, 3, 3, true)
#undef REPRO_RESTORED_GELU_BLOCKS

// One tile row holds every row of the batch.
using QkvPacked64 = int8proj::Tiling<64, 64, 16, 32, 4>;
using QkvPacked64Tall = int8proj::Tiling<128, 64, 16, 32, 4>;
static_assert(QkvPacked64::THREADS == 256 && QkvPacked64::SHARED_BYTES == 32768 &&
                  QkvPacked64Tall::THREADS == 512 && QkvPacked64Tall::SHARED_BYTES == 49152,
              "the host launches these threads and bytes");

// A minimum of one block: ptxas otherwise spills in the k loop for an
// occupancy the one-wave grid never takes. The _two kernels take an operand of two digits.
#define REPRO_RESTORED_QKV_PACKED(NAME, TILING, TWO)                                                     \
    extern "C" __global__ void __launch_bounds__(TILING::THREADS, 1) repro_restored_qkv_packed_##NAME(   \
        const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,   \
        const float* static_shared, const int* row_shared, const __half* residual, const float* fc,     \
        int rows, int n, int k, int segment, void* out, const float2* moments, const float* sums,       \
        const float* constants, signed char* queries, float* query_scales, const float* key_centers,    \
        signed char* keys, float* key_scales, __half* values16, int* failed, const int* sequence_codes,  \
        int heads, int head_width, signed char* planes, float* value_scales, const int* starts,        \
        const int* lengths, const int* pstarts, int sequences, int ptokens) {                           \
        extern __shared__ __align__(128) signed char stages[];                                          \
        repro_project_restored_wide<TILING, REPRO_QKV_PACKED, TWO>(                                     \
            qa, row_steps, q, col_scales, static_shared, row_shared, residual, fc, rows, n, k, segment,  \
            out, stages, DeferredTables{moments, sums, constants},                                      \
            ReproAttentionPrep{queries, query_scales, key_centers, keys, key_scales, values16, nullptr,  \
                               failed, sequence_codes, heads},                                          \
            head_width, ReproBlocksOut{},                                                               \
            ReproPackedValues{planes, value_scales, starts, lengths, pstarts, sequences, ptokens});      \
    }
REPRO_RESTORED_QKV_PACKED(64, QkvPacked64, false)
REPRO_RESTORED_QKV_PACKED(64_tall, QkvPacked64Tall, false)
REPRO_RESTORED_QKV_PACKED(64_two, QkvPacked64, true)
REPRO_RESTORED_QKV_PACKED(64_tall_two, QkvPacked64Tall, true)

// The wide tiles' kernels for an operand of two digits, whose chain is twice the weights'
// depth (repro_promoted_k_loop's TWO): the tilings and epilogues QKV and the up projection take.
#define REPRO_RESTORED_TWO(NAME, TILING, EPILOGUE)                                               \
    extern "C" __global__ void __launch_bounds__(TILING::THREADS) repro_restored_##NAME##_two(   \
        const signed char* qa, const float* row_steps, const signed char* q,                     \
        const float* col_scales, const float* static_shared, const int* row_shared,              \
        const __half* residual, const float* fc, int rows, int n, int k, int segment,            \
        void* out, const float2* moments, const float* sums, const float* constants) {           \
        extern __shared__ __align__(128) signed char stages[];                                   \
        repro_project_restored_wide<TILING, EPILOGUE, true>(                                     \
            qa, row_steps, q, col_scales, static_shared, row_shared, residual, fc, rows, n, k,   \
            segment, out, stages, DeferredTables{moments, sums, constants});                     \
    }
REPRO_RESTORED_TWO(plain_wide, RestoredWide, REPRO_PLAIN)
REPRO_RESTORED_TWO(gelu_wide, RestoredWide, REPRO_GELU)
REPRO_RESTORED_TWO(gelu_wide64x128, RestoredWide64x128, REPRO_GELU)
REPRO_RESTORED_TWO(plain_wide64, RestoredWideDown, REPRO_PLAIN)
REPRO_RESTORED_TWO(gelu_wide64, RestoredWideDown, REPRO_GELU)
#undef REPRO_RESTORED_TWO

#define REPRO_RESTORED_QKV_TWO(NAME, TILING)                                                             \
    extern "C" __global__ void __launch_bounds__(TILING::THREADS) repro_restored_qkv_##NAME##_two(       \
        const signed char* qa, const float* row_steps, const signed char* q, const float* col_scales,   \
        const float* static_shared, const int* row_shared, const __half* residual, const float* fc,     \
        int rows, int n, int k, int segment, void* out, const float2* moments, const float* sums,       \
        const float* constants, signed char* queries, float* query_scales, const float* key_centers,    \
        signed char* keys, float* key_scales, __half* values16, unsigned* value_magnitudes, int* failed, \
        const int* sequence_codes, int heads, int head_width) {                                         \
        extern __shared__ __align__(128) signed char stages[];                                          \
        repro_project_restored_wide<TILING, REPRO_QKV, true>(                                           \
            qa, row_steps, q, col_scales, static_shared, row_shared, residual, fc, rows, n, k, segment,  \
            out, stages, DeferredTables{moments, sums, constants},                                      \
            ReproAttentionPrep{queries, query_scales, key_centers, keys, key_scales, values16,           \
                               value_magnitudes, failed, sequence_codes, heads},                        \
            head_width);                                                                                \
    }
REPRO_RESTORED_QKV_TWO(wide, RestoredWide)
REPRO_RESTORED_QKV_TWO(wide64x128, RestoredWide64x128)
REPRO_RESTORED_QKV_TWO(wide64, RestoredWideDown)
#undef REPRO_RESTORED_QKV_TWO

#define ATT_THREADS 256
#define ATT_ROWS 64
#define ATT_ROW_THREADS 128
// Register budget for three resident blocks, matching the shared-memory budget.
#define ATT_MIN_BLOCKS 3
// attention/integer.rs weight_table, padded to whole 16-byte copies.
#define ATT_WEIGHT_SPAN 3840
#define ATT_WEIGHT_ENTRIES 3848
#define ATT_EXPONENT_UNITS 256.0f

// Their loads are in flight together.
#define PACK_ROWS 4

// pack_keys and pack_queries; false when a value is not finite. Every load leaves before
// any row is centered, which would otherwise wait for its value.
template <int R>
__device__ bool repro_pack_rows(const float* row0, size_t stride, int rows, const float* center,
                                int d, float* scales, signed char* out, int out_stride) {
    int lane = threadIdx.x & 31, c = 2 * lane;
    const bool live = c < d;
    float2 raw[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
        raw[r] = r < rows && live ? *reinterpret_cast<const float2*>(row0 + r * stride + c)
                                  : make_float2(0.0f, 0.0f);
    const float2 cv =
        center && live ? *reinterpret_cast<const float2*>(center + c) : make_float2(0.0f, 0.0f);
    float x[R][2];
    bool finite = true;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float2 v = raw[r];
        if (r < rows && live) {
            if (center) {
                v.x = __fsub_rn(v.x, cv.x);
                v.y = __fsub_rn(v.y, cv.y);
            }
            finite = finite && isfinite(v.x) && isfinite(v.y);
        }
        x[r][0] = v.x;
        x[r][1] = v.y;
    }
    // Padding lanes hold +0.
    float most[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
        most[r] = avx_max(avx_max(0.0f, abs_bits(x[r][0])), abs_bits(x[r][1]));
#pragma unroll
    for (int o = 16; o > 0; o /= 2)
#pragma unroll
        for (int r = 0; r < R; ++r)
            most[r] = avx_max(most[r], __shfl_xor_sync(0xffffffffu, most[r], o));
    float scale[R], reciprocal[R];
    bool normal = true;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        scale[r] = repro_scale_for_127(most[r]);
        reciprocal[r] = repro_step_reciprocal(scale[r]);
        normal = normal && reciprocal[r] != 0.0f;
        if (r < rows && lane == 0) scales[r] = scale[r];
    }
    // The rows' step tests made at once, as repro_on_steps makes them.
    auto store = [&](int r, int a, int b) {
        if (r < rows && c < d) {
            char2 bytes;
            bytes.x = (signed char)a;
            bytes.y = (signed char)b;
            *reinterpret_cast<char2*>(out + r * out_stride + c) = bytes;
        }
    };
    if (normal) {
#pragma unroll
        for (int r = 0; r < R; ++r)
            store(r, __float2int_rn(__fmul_rn(x[r][0], reciprocal[r])),
                  __float2int_rn(__fmul_rn(x[r][1], reciprocal[r])));
    } else {
#pragma unroll
        for (int r = 0; r < R; ++r)
            store(r, __float2int_rn(repro_on_step(x[r][0], scale[r], reciprocal[r])),
                  __float2int_rn(repro_on_step(x[r][1], scale[r], reciprocal[r])));
    }
    return __all_sync(0xffffffffu, finite);
}

// Two rounds of eight value loads a thread.
#define PREP_ROWS 16

// The QKV epilogue's attention operands from FP32 rows, for inputs past one key chunk;
// maxima and ors are the same in any grouping, so tiles merge into the cleared scratch.
extern "C" __global__ void __launch_bounds__(ATT_THREADS)
    repro_attention_prep(const float* qkv, const int* starts, const int* lengths, int tokens,
                         int hidden, int heads, int d, const float* key_centers,
                         float* query_scales, signed char* queries, float* key_scales,
                         signed char* keys, __half* values16, unsigned* value_magnitudes,
                         int* failed) {
    __shared__ int bad;
    int s = blockIdx.x, head = blockIdx.y;
    int start = starts[s], len = lengths[s];
    const int row0 = blockIdx.z * PREP_ROWS;
    if (row0 >= len) return;
    const int end = min(len, row0 + PREP_ROWS);
    int stride = 3 * hidden;
    const float* q_rows = qkv + (size_t)start * stride + head * d;
    const float* k_rows = q_rows + hidden;
    const float* v_rows = q_rows + 2 * hidden;
    int t = threadIdx.x, warp = t >> 5, warps = ATT_THREADS / 32;
    size_t first = (size_t)head * tokens + start;
    if (t == 0) bad = 0;
    __syncthreads();
    for (int j = row0 + warp * PACK_ROWS; j < end; j += warps * PACK_ROWS) {
        int count = min(PACK_ROWS, end - j);
        if (!repro_pack_rows<PACK_ROWS>(q_rows + (size_t)j * stride, stride, count, nullptr, d,
                                        query_scales + first + j, queries + (first + j) * d, d))
            bad = 1;
        if (!repro_pack_rows<PACK_ROWS>(k_rows + (size_t)j * stride, stride, count,
                                        key_centers + head * d, d, key_scales + first + j,
                                        keys + (first + j) * d, d))
            bad = 1;
    }
    if (t < d) {
        unsigned most = 0;
        bool finite = true;
        // The compiler cannot rule out the FP16 stores aliasing the loads, so
        // one at a time each would wait its own round trip.
        for (int j0 = row0; j0 < end; j0 += 8) {
            float raw[8];
#pragma unroll
            for (int u = 0; u < 8; ++u)
                raw[u] = j0 + u < end ? v_rows[(size_t)(j0 + u) * stride + t] : 0.0f;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int j = j0 + u;
                if (j >= end) break;
                __half v = __float2half_rn(raw[u]);
                values16[(size_t)(start + j) * hidden + head * d + t] = v;
                float x = __half2float(v);
                finite = finite && isfinite(x);
                if (j != 0 && j != len - 1) most = max(most, __float_as_uint(x) & 0x7fffffffu);
            }
        }
        atomicMax(value_magnitudes + (size_t)s * hidden + head * d + t, most);
        if (!finite) bad = 1;
    }
    __syncthreads();
    if (t == 0 && bad) atomicOr(failed + (size_t)s * heads + head, 1);
}

// A word of padding puts a warp's 32 channels in distinct banks.
template <int D>
struct AttentionValuesShared {
    float vscale[D], vreciprocal[D];
    unsigned staged[D][ATT_CHUNK / 4 + 1];
};

// attention/integer.rs pack_values. Rows are read across channels and the
// plane written along tokens, so the tile transposes through shared memory.
template <int D>
__device__ __forceinline__ void repro_attention_values_tile(
    int tile, AttentionValuesShared<D>& sh, const __half* values16, const unsigned* value_magnitudes,
    const float* query_scales, const float* key_scales, const int* starts, const int* pstarts,
    const int* lengths, int tokens, int ptokens, int hidden, int heads, float* value_scales,
    signed char* value_planes, int* failed) {
    constexpr int WORDS = ATT_CHUNK / 4;
    float* vscale = sh.vscale;
    float* vreciprocal = sh.vreciprocal;
    auto& staged = sh.staged;
    const int d = D, threads = blockDim.x;
    int head = blockIdx.y, s = blockIdx.z;
    int start = starts[s], len = lengths[s];
    int c0 = tile * ATT_CHUNK;
    if (c0 >= len) return;
    int count = min(ATT_CHUNK, len - c0);
    const __half* v_rows = values16 + (size_t)(start + c0) * hidden + head * d;
    int t = threadIdx.x;
    size_t first = (size_t)head * tokens + start + c0;
    if (t < d) {
        vscale[t] = repro_scale_for_127(
            __uint_as_float(value_magnitudes[(size_t)s * hidden + head * d + t]));
        vreciprocal[t] = repro_step_reciprocal(vscale[t]);
        if (c0 == 0) value_scales[((size_t)s * heads + head) * d + t] = vscale[t];
    }
    bool bad = false;
    for (int j = t; j < count; j += threads)
        bad = bad || isnan(query_scales[first + j]) || isnan(key_scales[first + j]);
    if (__syncthreads_or(bad) && t == 0) atomicOr(failed + (size_t)s * heads + head, 1);
    for (int i = t; i < d * WORDS; i += threads) {
        int c = i % d, word = i / d;
        float x[4];
        bool keep[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int j = 4 * word + u;
            keep[u] = j < count && c0 + j != 0 && c0 + j != len - 1;
            x[u] = keep[u] ? __half2float(v_rows[(size_t)j * hidden + c]) : 0.0f;
        }
        int v[4];
        repro_on_steps<4>(x, vscale[c], vreciprocal[c], v);
        unsigned packed = 0;
#pragma unroll
        for (int u = 0; u < 4; ++u) packed |= (unsigned)((keep[u] ? v[u] : 0) & 0xff) << (8 * u);
        staged[c][word] = packed;
    }
    __syncthreads();
    // Slots and tiles start at multiples of 64 tokens, so the words align.
    unsigned* plane =
        reinterpret_cast<unsigned*>(value_planes + (size_t)head * d * ptokens + pstarts[s] + c0);
    for (int i = t; i < d * WORDS; i += threads) {
        int row = i / WORDS, word = i % WORDS;
        plane[(size_t)row * (ptokens / 4) + word] = staged[row][word];
    }
}

// weigh's index: fmaxf, like _mm256_max_ps(e, low), takes the bound for a NaN.
__device__ __forceinline__ int repro_weight_index(float e) {
    return -__float2int_rn(fminf(fmaxf(e, -(float)ATT_WEIGHT_SPAN), 0.0f));
}

// out[i] = repro_weight_index of the float whose bits are first + i.
extern "C" __global__ void repro_weight_indices(unsigned first, int n, unsigned short* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (unsigned short)repro_weight_index(__uint_as_float(first + (unsigned)i));
}

// 7-bit digits fit signed INT8 products.
__device__ __forceinline__ void repro_weight_planes(int w0, int w1, int w2, int w3,
                                                    unsigned& high, unsigned& low) {
    unsigned a = (unsigned)w0 | ((unsigned)w1 << 16), b = (unsigned)w2 | ((unsigned)w3 << 16);
    low = __byte_perm(a & 0x007f007fu, b & 0x007f007fu, 0x6420);
    high = __byte_perm((a >> 7) & 0x007f007fu, (b >> 7) & 0x007f007fu, 0x6420);
}

// In this key order a thread's score fragments are its A fragments of the
// products, so no values pass between threads.
__device__ __forceinline__ int repro_key_of(int q, int c) {
    return 16 * (q >> 1) + 2 * (q & 1) + 4 * (c >> 1) + (c & 1);
}

template <int D>
__device__ __forceinline__ void repro_query_fragments(const signed char* __restrict__ queries,
                                                      size_t first, int r0, int r1, int len, int t,
                                                      unsigned (&qa)[D / 32][4]) {
#pragma unroll
    for (int st = 0; st < D / 32; ++st) {
        const unsigned* q0 = reinterpret_cast<const unsigned*>(queries + (first + r0) * D) + 8 * st;
        const unsigned* q1 = reinterpret_cast<const unsigned*>(queries + (first + r1) * D) + 8 * st;
        qa[st][0] = r0 < len ? q0[t] : 0u;
        qa[st][1] = r1 < len ? q1[t] : 0u;
        qa[st][2] = r0 < len ? q0[4 + t] : 0u;
        qa[st][3] = r1 < len ? q1[4 + t] : 0u;
    }
}

// A thread's copies lie a fixed stride of rows apart, so their addresses form once. Scales past
// the chunk's keys are NaN: such a key's score is NaN, which no row's largest takes and which
// weighs nothing.
template <int D, int KP>
__device__ __forceinline__ void repro_issue_keys(const signed char* __restrict__ keys,
                                                 const float* __restrict__ key_scales,
                                                 size_t first, int c0, int count, int* chunk,
                                                 float* scales) {
    constexpr int PIECES = D / 16, ROWS = ATT_ROW_THREADS / PIECES;
    static_assert(ATT_ROW_THREADS % PIECES == 0 && ATT_CHUNK % ROWS == 0 && ATT_CHUNK <= ATT_ROW_THREADS,
                  "whole strides of rows a thread");
    const int t = threadIdx.x, row = t / PIECES, piece = t % PIECES;
    const signed char* source = keys + (first + c0 + row) * D + piece * 16;
    const unsigned target = shared_address(chunk + row * KP + piece * 4);
#pragma unroll
    for (int j = 0; j < ATT_CHUNK / ROWS; ++j)
        if (row + j * ROWS < count)
            cp_async_16(target + j * ROWS * KP * 4, source + static_cast<size_t>(j) * ROWS * D, 16);
    if (t < ATT_CHUNK) {
        if (t < count)
            cp_async_4(shared_address(scales + t), key_scales + first + c0 + t);
        else
            scales[t] = __int_as_float(0x7fc00000);
    }
}

template <int D, int KP>
__device__ __forceinline__ void repro_score(const int* __restrict__ keys, int s, int q, int g,
                                            int t, const unsigned (&qa)[D / 32][4],
                                            int (&acc)[4]) {
    const int* key = keys + (32 * s + repro_key_of(q, g)) * KP;
    acc[0] = acc[1] = acc[2] = acc[3] = 0;
#pragma unroll
    for (int st = 0; st < D / 32; ++st) {
        unsigned kb[2] = {(unsigned)key[8 * st + t], (unsigned)key[8 * st + 4 + t]};
        int8proj::mma_s8(acc, qa[st], kb);
    }
}

// The query's scale enters through the row's exponent factor.
__device__ __forceinline__ float repro_scaled_score(int dot, float key_scale) {
    return __fmul_rn(__int2float_rn(dot), key_scale);
}

// With at most 16 keys the tiles of keys 16 to 31 hold none and are left out. PADDED: the
// scales past `count` are NaN, so fmaxf passes over those keys untested; a real key's score
// is never NaN, so fmaxf takes avx_max's largest.
template <int D, int KP, int KEYS = ATT_CHUNK, bool PADDED = false>
__device__ __forceinline__ void repro_row_max_steps(const int* __restrict__ keys,
                                                    const float* __restrict__ key_scales,
                                                    int count, int g, int t,
                                                    const unsigned (&qa)[D / 32][4], float& max0,
                                                    float& max1) {
    for (int step = 0; step < (KEYS + 31) / 32 && step < (count + 31) / 32; ++step) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (KEYS <= 16 && q >= 2) continue;
            int acc[4];
            repro_score<D, KP>(keys, step, q, g, t, qa, acc);
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int jj = 32 * step + repro_key_of(q, 2 * t + e);
                if constexpr (PADDED) {
                    const float ks = key_scales[jj];
                    max0 = fmaxf(max0, repro_scaled_score(acc[e], ks));
                    max1 = fmaxf(max1, repro_scaled_score(acc[2 + e], ks));
                } else if (jj < count) {
                    const float ks = key_scales[jj];
                    max0 = avx_max(max0, repro_scaled_score(acc[e], ks));
                    max1 = avx_max(max1, repro_scaled_score(acc[2 + e], ks));
                }
            }
        }
    }
}

__device__ __forceinline__ void repro_merge_row_max(float& max0, float& max1) {
    for (int o = 1; o < 4; o *= 2) {
        max0 = avx_max(max0, __shfl_xor_sync(0xffffffffu, max0, o));
        max1 = avx_max(max1, __shfl_xor_sync(0xffffffffu, max1, o));
    }
}

// attend's first pass: without NaN a row's largest is the same over any split, and the
// sign of a zero largest reaches no weight.
template <int D>
struct AttentionMaxShared {
    alignas(16) int kchunk[2][ATT_CHUNK * (D / 4 + 4)];
    float kscale[2][ATT_CHUNK];
};
template <int D>
__device__ __forceinline__ void repro_attention_max_tile(
    int x, AttentionMaxShared<D>& sh, const signed char* __restrict__ queries,
    const signed char* __restrict__ keys, const float* __restrict__ key_scales,
    const int* __restrict__ failed, const int* __restrict__ starts, const int* __restrict__ lengths,
    int tokens, int heads, int splits, float* __restrict__ row_max) {
    constexpr int STEPS = D / 32, KP = D / 4 + 4;
    auto& kchunk = sh.kchunk;
    auto& kscale = sh.kscale;
    int head = blockIdx.y, s = blockIdx.z, split = x % splits;
    int start = starts[s], len = lengths[s];
    int row0 = x / splits * ATT_ROWS;
    if (row0 >= len || failed[(size_t)s * heads + head]) return;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    int r0 = row0 + warp * 16 + g, r1 = r0 + 8;
    size_t first = (size_t)head * tokens + start;
    float max0 = -__int_as_float(0x7f800000), max1 = max0;
    int chunks = (len + ATT_CHUNK - 1) / ATT_CHUNK, share = (chunks + splits - 1) / splits;
    int begin = split * share, end = min(chunks, begin + share);
    if (begin < end) {
        unsigned qa[STEPS][4];
        repro_query_fragments<D>(queries, first, r0, r1, len, t, qa);
        repro_issue_keys<D, KP>(keys, key_scales, first, begin * ATT_CHUNK,
                                min(ATT_CHUNK, len - begin * ATT_CHUNK), kchunk[0], kscale[0]);
        cp_async_commit();
        for (int chunk = begin; chunk < end; ++chunk) {
            int b = (chunk - begin) & 1, c0 = chunk * ATT_CHUNK, count = min(ATT_CHUNK, len - c0);
            // The other buffer was released by the barrier ending the last chunk.
            if (chunk + 1 < end)
                repro_issue_keys<D, KP>(keys, key_scales, first, c0 + ATT_CHUNK,
                                        min(ATT_CHUNK, len - c0 - ATT_CHUNK), kchunk[b ^ 1],
                                        kscale[b ^ 1]);
            cp_async_commit();
            cp_async_wait<1>();
            __syncthreads();
            repro_row_max_steps<D, KP, ATT_CHUNK, true>(kchunk[b], kscale[b], count, g, t, qa, max0, max1);
            // Every thread is done with buffer b before chunk + 2 refills it.
            __syncthreads();
        }
        repro_merge_row_max(max0, max1);
    }
    float* slice = row_max + (size_t)split * heads * tokens + first;
    if (t == 0) {
        if (r0 < len) slice[r0] = max0;
        if (r1 < len) slice[r1] = max1;
    }
}

// Neither half reads the other's output. The maxima may take a head the values
// fail in this launch; the rows kernel reads the flags after both.
template <int D>
union AttentionValuesMaxShared {
    AttentionValuesShared<D> values;
    AttentionMaxShared<D> max;
};
#define REPRO_ATTENTION_VALUES_MAX(D)                                                                 \
    extern "C" __global__ void __launch_bounds__(ATT_ROW_THREADS) repro_attention_values_max_##D(     \
        const __half* values16, const unsigned* value_magnitudes, const float* query_scales,          \
        const float* key_scales, const int* starts, const int* pstarts, const int* lengths, int tokens,  \
        int ptokens, int hidden, int heads, float* value_scales, signed char* value_planes,           \
        int* failed, const signed char* queries, const signed char* keys, int splits, float* row_max,  \
        int value_tiles) {                                                                             \
        __shared__ AttentionValuesMaxShared<D> shared;                                                \
        if ((int)blockIdx.x < value_tiles)                                                            \
            repro_attention_values_tile<D>(blockIdx.x, shared.values, values16, value_magnitudes,     \
                                           query_scales, key_scales, starts, pstarts, lengths, tokens,  \
                                           ptokens, hidden, heads, value_scales, value_planes, failed);  \
        else                                                                                          \
            repro_attention_max_tile<D>(blockIdx.x - value_tiles, shared.max, queries, keys,          \
                                        key_scales, failed, starts, lengths, tokens, heads, splits,   \
                                        row_max);                                                     \
    }
REPRO_ATTENTION_VALUES_MAX(64)
#undef REPRO_ATTENTION_VALUES_MAX

template <int D>
__device__ __forceinline__ float repro_score_scale() {
    return __fdiv_rn(1.0f, __fsqrt_rn((float)D));
}

// attention/integer.rs exponent_factor: fl(fl(1 / fl(sqrt D)) log2 e) 2^8.
template <int D>
__device__ __forceinline__ float repro_exponent_factor() {
    return __fmul_rn(__fmul_rn(repro_score_scale<D>(), 0x1.715476p+0f), ATT_EXPONENT_UNITS);
}

// A row whose largest score is not finite takes no weights.
struct ReproRowShift {
    float max0, max1, factor0, factor1;
    bool finite0, finite1;
};

__device__ __forceinline__ ReproRowShift repro_row_shift(float max0, float max1, float qs0,
                                                         float qs1, float factor) {
    ReproRowShift r;
    r.max0 = max0;
    r.max1 = max1;
    r.finite0 = isfinite(max0);
    r.finite1 = isfinite(max1);
    r.factor0 = __fmul_rn(qs0, factor);
    r.factor1 = __fmul_rn(qs1, factor);
    return r;
}

// attention/integer.rs weigh.
__device__ __forceinline__ int repro_weight(float x, float max, float factor, bool finite,
                                            const unsigned short* __restrict__ table) {
    const int w = table[repro_weight_index(__fmul_rn(__fsub_rn(x, max), factor))];
    return finite ? w : 0;
}

// One thread of a row's quad holds each sink weight; the others hold zero.
struct ReproSinks {
    int first0 = 0, first1 = 0, last0 = 0, last1 = 0;
};

// A step holding a sink keeps its weights for the finish, which adds the
// sinks' terms from their FP16 values. PADDED: a key past the count has a NaN
// score and a row without a finite largest has +inf for it, so either exponent
// takes the table's last entry, zero, and no weight is tested.
template <int D, int KP, int CP, int CTW, bool SINKS, int KEYS = ATT_CHUNK, bool PADDED = false>
__device__ __forceinline__ void repro_attention_step(
    const int* __restrict__ keys, const float* __restrict__ key_scales,
    const int* __restrict__ plane, const unsigned short* __restrict__ table, int s, int count,
    int first, int last, int g, int t, const unsigned (&qa)[D / 32][4], const ReproRowShift& rs,
    int ct0, int& total0, int& total1, ReproSinks& sinks, int (&hh)[CTW][4],
    int (&ll)[CTW][4]) {
    int w0[8], w1[8];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        if (KEYS <= 16 && q >= 2) {
            w0[2 * q] = w0[2 * q + 1] = w1[2 * q] = w1[2 * q + 1] = 0;
            continue;
        }
        int acc[4];
        repro_score<D, KP>(keys, s, q, g, t, qa, acc);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int jj = 32 * s + repro_key_of(q, 2 * t + e);
            const bool real = PADDED || jj < count;
            const float ks = key_scales[jj];
            int a = repro_weight(repro_scaled_score(acc[e], ks), rs.max0, rs.factor0,
                                 PADDED || rs.finite0, table);
            int b = repro_weight(repro_scaled_score(acc[2 + e], ks), rs.max1, rs.factor1,
                                 PADDED || rs.finite1, table);
            a = real ? a : 0;
            b = real ? b : 0;
            if constexpr (SINKS) {
                if (jj == first) sinks.first0 = a, sinks.first1 = b;
                if (jj == last) sinks.last0 = a, sinks.last1 = b;
            }
            w0[2 * q + e] = a;
            w1[2 * q + e] = b;
            total0 += a;
            total1 += b;
        }
    }
    unsigned ah[4], al[4];
    repro_weight_planes(w0[0], w0[1], w0[2], w0[3], ah[0], al[0]);
    repro_weight_planes(w1[0], w1[1], w1[2], w1[3], ah[1], al[1]);
    repro_weight_planes(w0[4], w0[5], w0[6], w0[7], ah[2], al[2]);
    repro_weight_planes(w1[4], w1[5], w1[6], w1[7], ah[3], al[3]);
#pragma unroll
    for (int k = 0; k < CTW; ++k) {
        const int* v = plane + ((ct0 + k) * 8 + g) * CP + 8 * s;
        unsigned bv[2] = {(unsigned)v[t], (unsigned)v[4 + t]};
        int8proj::mma_s8(hh[k], ah, bv);
        int8proj::mma_s8(ll[k], al, bv);
    }
}

__device__ __forceinline__ void repro_join_rows(int& total0, int& total1, ReproSinks& sinks) {
    for (int o = 1; o < 4; o *= 2) {
        total0 += __shfl_xor_sync(0xffffffffu, total0, o);
        total1 += __shfl_xor_sync(0xffffffffu, total1, o);
        sinks.first0 += __shfl_xor_sync(0xffffffffu, sinks.first0, o);
        sinks.first1 += __shfl_xor_sync(0xffffffffu, sinks.first1, o);
        sinks.last0 += __shfl_xor_sync(0xffffffffu, sinks.last0, o);
        sinks.last1 += __shfl_xor_sync(0xffffffffu, sinks.last1, o);
    }
}

// attend's finish, ((p scale + w_first v_first) + w_last v_last) (1 / total):
// the products' term rounds once and each sink's term by an FMA.
template <int CTW>
__device__ __forceinline__ void repro_attention_finish(
    const float* __restrict__ vscale, const float* __restrict__ vfirst,
    const float* __restrict__ vlast, int total0, int total1, ReproSinks sinks,
    const int (&hh)[CTW][4], const int (&ll)[CTW][4], int ct0, int r0, int r1, int len, int t,
    __half* __restrict__ rows, int hidden, int rbase = 0) {
    // A nonzero total is an integer of at least 16383, so its reciprocal is
    // repro_reciprocal's; a zero total's output is NaN whatever it gives.
    const float reciprocal[2] = {repro_reciprocal(__int2float_rn(total0)),
                                 repro_reciprocal(__int2float_rn(total1))};
    const float first[2] = {__int2float_rn(sinks.first0), __int2float_rn(sinks.first1)};
    const float last[2] = {__int2float_rn(sinks.last0), __int2float_rn(sinks.last1)};
#pragma unroll
    for (int k = 0; k < CTW; ++k) {
        int c = (ct0 + k) * 8 + 2 * t;
#pragma unroll
        for (int half_row = 0; half_row < 2; ++half_row) {
            int r = half_row ? r1 : r0, total = half_row ? total1 : total0;
            if (r >= len) continue;
            __half out[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int i = 2 * half_row + e;
                float sum = __fmul_rn(__int2float_rn(128 * hh[k][i] + ll[k][i]), vscale[c + e]);
                sum = __fmaf_rn(first[half_row], vfirst[c + e], sum);
                if (len > 1) sum = __fmaf_rn(last[half_row], vlast[c + e], sum);
                float v = total == 0 ? __int_as_float(0x7fc00000)
                                     : __fmul_rn(sum, reciprocal[half_row]);
                out[e] = __float2half_rn(v);
            }
            *reinterpret_cast<__half2*>(rows + (size_t)(r - rbase) * hidden + c) =
                __halves2half2(out[0], out[1]);
        }
    }
}

// attend's weights, products and finish: 14-bit weights w = 128 h + l split into two INT8
// products whose exact sums fit I32 (512 * 16383 * 127 < 2^31), in any grouping.
template <int D>
__device__ __forceinline__ void repro_attention_tile(
    const signed char* __restrict__ queries, const float* __restrict__ query_scales,
    const signed char* __restrict__ keys, const float* __restrict__ key_scales,
    const signed char* __restrict__ value_planes, const float* __restrict__ value_scales,
    const __half* __restrict__ values16, const unsigned short* __restrict__ weights,
    const int* __restrict__ failed, const int* __restrict__ starts,
    const int* __restrict__ pstarts, const int* __restrict__ lengths, int tokens, int ptokens,
    int hidden, int heads, int splits, const float* __restrict__ row_max,
    __half* __restrict__ context) {
    // Four words of row padding put a value fragment's rows in distinct banks.
    constexpr int STEPS = D / 32, CHANNEL_TILES = D / 8;
    constexpr int KP = D / 4 + 4, CP = ATT_CHUNK / 4 + 4;
    __shared__ __align__(16) int kchunk[2][ATT_CHUNK * KP];
    __shared__ float kscale[2][ATT_CHUNK];
    __shared__ __align__(16) int vplane[2][D * CP];
    __shared__ __align__(16) unsigned short table[ATT_WEIGHT_ENTRIES];
    __shared__ float vscale[D], vfirst[D], vlast[D];
    int head = blockIdx.x, s = blockIdx.z;
    int start = starts[s], len = lengths[s];
    int row0 = blockIdx.y * ATT_ROWS;
    if (row0 >= len) return;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int g = lane >> 2, t = lane & 3;
    int r0 = row0 + warp * 16 + g, r1 = r0 + 8;
    size_t first = (size_t)head * tokens + start;
    if (failed[(size_t)s * heads + head]) {
        // A failed head writes NaN rows: its output is non-finite either way.
        int rows = min(ATT_ROWS, len - row0);
        for (int i = tid; i < rows * D; i += ATT_ROW_THREADS)
            context[(size_t)(start + row0 + i / D) * hidden + head * D + i % D] =
                __float2half_rn(__int_as_float(0x7fc00000));
        return;
    }
    // Loaded while the loop runs, so each output reads shared memory rather
    // than paying a memory latency apiece.
    size_t sh = ((size_t)s * heads + head) * D;
    if (tid < D) {
        vscale[tid] = value_scales[sh + tid];
        vfirst[tid] = __half2float(values16[(size_t)start * hidden + head * D + tid]);
        vlast[tid] = __half2float(values16[(size_t)(start + len - 1) * hidden + head * D + tid]);
    }
    unsigned qa[STEPS][4];
    repro_query_fragments<D>(queries, first, r0, r1, len, t, qa);
    // Rows past the sequence take no weights.
    float max0 = -__int_as_float(0x7f800000), max1 = max0;
    for (int split = 0; split < splits; ++split) {
        const float* slice = row_max + (size_t)split * heads * tokens + first;
        if (r0 < len) max0 = avx_max(max0, slice[r0]);
        if (r1 < len) max1 = avx_max(max1, slice[r1]);
    }
    // A largest score that is not finite gives way to +inf, under which every exponent is -inf
    // or NaN: the padded steps then test no row.
    const float inf = __int_as_float(0x7f800000);
    const ReproRowShift shift = repro_row_shift(
        isfinite(max0) ? max0 : inf, isfinite(max1) ? max1 : inf,
        r0 < len ? query_scales[first + r0] : 0.0f, r1 < len ? query_scales[first + r1] : 0.0f,
        repro_exponent_factor<D>());
    int total0 = 0, total1 = 0;
    ReproSinks sinks;
    int hh[CHANNEL_TILES][4] = {}, ll[CHANNEL_TILES][4] = {};
    const signed char* plane = value_planes + (size_t)head * D * ptokens + pstarts[s];
    int chunks = (len + ATT_CHUNK - 1) / ATT_CHUNK;
    // A thread's value copies lie a fixed stride of channels apart, as its key copies do.
    constexpr int PIECES = ATT_CHUNK / 16, CHANNELS = ATT_ROW_THREADS / PIECES;
    static_assert(ATT_ROW_THREADS % PIECES == 0 && D % CHANNELS == 0, "whole strides of channels a thread");
    const signed char* plane_source = plane + (size_t)(tid / PIECES) * ptokens + (tid % PIECES) * 16;
    const int plane_at = (tid / PIECES) * CP + (tid % PIECES) * 4;
    auto issue = [&](int chunk) {
        int b = chunk & 1, c0 = chunk * ATT_CHUNK;
        repro_issue_keys<D, KP>(keys, key_scales, first, c0, min(ATT_CHUNK, len - c0), kchunk[b],
                                kscale[b]);
        const unsigned target = shared_address(vplane[b] + plane_at);
#pragma unroll
        for (int j = 0; j < D / CHANNELS; ++j)
            cp_async_16(target + j * CHANNELS * CP * 4,
                        plane_source + static_cast<size_t>(j) * CHANNELS * ptokens + c0, 16);
        if (chunk == 0)
            for (int i = tid; i < ATT_WEIGHT_ENTRIES / 8; i += ATT_ROW_THREADS)
                cp_async_16(shared_address(table + 8 * i), weights + 8 * i, 16);
        cp_async_commit();
    };
    issue(0);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        int b = chunk & 1, c0 = chunk * ATT_CHUNK;
        // The other buffer was released by the barrier ending the last chunk.
        if (chunk + 1 < chunks) issue(chunk + 1);
        else cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        int count = min(ATT_CHUNK, len - c0), last = len - 1 - c0;
        // Keys past `count` in a whole step take zero weights.
        for (int s = 0; s < (count + 31) / 32; ++s) {
            const bool holds_first = c0 == 0 && s == 0, holds_last = last >> 5 == s;
            if (holds_first || holds_last)
                repro_attention_step<D, KP, CP, CHANNEL_TILES, true, ATT_CHUNK, true>(
                    kchunk[b], kscale[b], vplane[b], table, s, count, holds_first ? 0 : -1,
                    holds_last ? last : -1, g, t, qa, shift, 0, total0, total1, sinks, hh, ll);
            else
                repro_attention_step<D, KP, CP, CHANNEL_TILES, false, ATT_CHUNK, true>(
                    kchunk[b], kscale[b], vplane[b], table, s, count, -1, -1, g, t, qa, shift, 0,
                    total0, total1, sinks, hh, ll);
        }
        // Every thread is done with buffer b before chunk + 2 refills it.
        __syncthreads();
    }
    repro_join_rows(total0, total1, sinks);
    repro_attention_finish<CHANNEL_TILES>(vscale, vfirst, vlast, total0, total1, sinks, hh, ll, 0,
                                          r0, r1, len, t,
                                          context + (size_t)start * hidden + head * D, hidden);
}

// The out projection's input stage runs once the tile's heads are done.
template <int D>
__device__ __forceinline__ void repro_attention_rows(
    const signed char* queries, const float* query_scales, const signed char* keys,
    const float* key_scales, const signed char* value_planes, const float* value_scales,
    const __half* values16, const unsigned short* weights, const int* failed, const int* starts,
    const int* pstarts, const int* lengths, int tokens, int ptokens, int hidden, int heads,
    int splits, const float* row_max, __half* context, const ContextTail& tail) {
    repro_attention_tile<D>(queries, query_scales, keys, key_scales, value_planes, value_scales,
                            values16, weights, failed, starts, pstarts, lengths, tokens, ptokens,
                            hidden, heads, splits, row_max, context);
    const int s = blockIdx.z, len = lengths[s], row0 = blockIdx.y * ATT_ROWS;
    if (row0 < len)
        tail.finish<D>(context, starts[s] + row0, min(ATT_ROWS, len - row0), hidden,
                       blockIdx.x, blockIdx.z * gridDim.y + blockIdx.y);
}

#define REPRO_ATTENTION_ROWS(D)                                                                     \
    extern "C" __global__ void __launch_bounds__(ATT_ROW_THREADS, ATT_MIN_BLOCKS)                 \
        repro_attention_rows_##D(                                                                   \
            const signed char* queries, const float* query_scales, const signed char* keys,       \
            const float* key_scales, const signed char* value_planes, const float* value_scales,  \
            const __half* values16, const unsigned short* weights, const int* failed,             \
            const int* starts, const int* pstarts, const int* lengths, int tokens, int ptokens,   \
            int hidden, int heads, int splits, const float* row_max, __half* context,             \
            unsigned* tail_counts, const int* classes, const int* group_of_row,                   \
            const float* centers, const float* signs, const float* fc, signed char* quantized,    \
            float* steps) {                                                                         \
        repro_attention_rows<D>(queries, query_scales, keys, key_scales, value_planes,            \
                                value_scales, values16, weights, failed, starts, pstarts, lengths, \
                                tokens, ptokens, hidden, heads, splits, row_max, context,         \
                                ContextTail{tail_counts, classes, group_of_row, centers, signs,   \
                                            fc, quantized, steps, tokens});                       \
    }
REPRO_ATTENTION_ROWS(64)
#undef REPRO_ATTENTION_ROWS

// The rows' largest scores stay in registers between the passes.
template <int D, int CTW, int KEYS, int CP = ATT_CHUNK / 4 + 4>
__device__ __forceinline__ void repro_attention_narrow_rows(
    const int* __restrict__ qrows, const float* __restrict__ qscale,
    const int* __restrict__ krows, const float* __restrict__ kscale,
    const int* __restrict__ plane, const unsigned short* __restrict__ table,
    const float* __restrict__ vscale, const float* __restrict__ vfirst,
    const float* __restrict__ vlast, int len, int r0, int r1, int ct0, int g, int t,
    __half* __restrict__ rows, int hidden, int rbase = 0) {
    constexpr int KP = D / 4 + 4;
    unsigned qa[D / 32][4];
#pragma unroll
    for (int st = 0; st < D / 32; ++st) {
        const int* q0 = qrows + (r0 - rbase) * KP + 8 * st;
        const int* q1 = qrows + (r1 - rbase) * KP + 8 * st;
        qa[st][0] = r0 < len ? (unsigned)q0[t] : 0u;
        qa[st][1] = r1 < len ? (unsigned)q1[t] : 0u;
        qa[st][2] = r0 < len ? (unsigned)q0[4 + t] : 0u;
        qa[st][3] = r1 < len ? (unsigned)q1[4 + t] : 0u;
    }
    float max0 = -__int_as_float(0x7f800000), max1 = max0;
    repro_row_max_steps<D, KP, KEYS>(krows, kscale, len, g, t, qa, max0, max1);
    repro_merge_row_max(max0, max1);
    // Rows past the sequence take no weights.
    const ReproRowShift shift = repro_row_shift(
        r0 < len ? max0 : -__int_as_float(0x7f800000), r1 < len ? max1 : -__int_as_float(0x7f800000),
        r0 < len ? qscale[r0 - rbase] : 0.0f, r1 < len ? qscale[r1 - rbase] : 0.0f,
        repro_exponent_factor<D>());
    int total0 = 0, total1 = 0;
    ReproSinks sinks;
    int hh[CTW][4] = {}, ll[CTW][4] = {};
    for (int step = 0; step < (KEYS + 31) / 32 && step < (len + 31) / 32; ++step)
        repro_attention_step<D, KP, CP, CTW, true, KEYS>(
            krows, kscale, plane, table, step, len, step == 0 ? 0 : -1,
            (len - 1) >> 5 == step ? len - 1 : -1, g, t, qa, shift, ct0, total0, total1, sinks, hh,
            ll);
    repro_join_rows(total0, total1, sinks);
    repro_attention_finish<CTW>(vscale, vfirst, vlast, total0, total1, sinks, hh, ll, ct0, r0, r1,
                                len, t, rows, hidden, rbase);
}

// Warps 8-11 take a thread a channel of the 128-block's values.
#define ATT_PAIR_THREADS 384
#define PAIR_PACK_ROWS 8

#define PAIR_CENTER_ROWS 4

// repro.rs pairs_bytes mirrors this layout; every part is whole 16-byte words
// for cp.async.
template <int D, int KR = ATT_CHUNK, int QR = ATT_CHUNK, int THREADS = ATT_PAIR_THREADS>
struct PairLayout {
    static constexpr int HB = BLOCK / D, KP = D / 4 + 4, CP = KR / 4 + 4, CTX = BLOCK + 8;
    static constexpr int KROWS = KR, QROWS = QR;
    static constexpr int WARPS = THREADS / 32;
    static constexpr int QROWS_BYTES = HB * QR * KP * 4, KROWS_BYTES = HB * KR * KP * 4;
    static constexpr int PLANE_BYTES = HB * D * CP * 4;
    static constexpr int QSCALE_BYTES = HB * QR * 4, KSCALE_BYTES = HB * KR * 4, CHANNEL_BYTES = HB * D * 4;
    static constexpr int TABLE_BYTES = ATT_WEIGHT_ENTRIES * 2, FAILED_BYTES = 16 * ((WARPS * 4 + 15) / 16);
    static constexpr int CENTERS_BYTES = PAIR_CENTER_ROWS * BLOCK * 4, SIGNS_BYTES = BLOCK * 4;
    static constexpr int CTX_BYTES = QR * CTX * 2;
    static constexpr int TABLE_AT =
        QROWS_BYTES + KROWS_BYTES + PLANE_BYTES + QSCALE_BYTES + KSCALE_BYTES + 3 * CHANNEL_BYTES;
    static constexpr int BYTES = TABLE_AT + TABLE_BYTES + FAILED_BYTES + CENTERS_BYTES + SIGNS_BYTES + CTX_BYTES;
    static_assert(QROWS_BYTES % 16 == 0 && KROWS_BYTES % 16 == 0 && PLANE_BYTES % 16 == 0 &&
                      QSCALE_BYTES % 16 == 0 && KSCALE_BYTES % 16 == 0 && CHANNEL_BYTES % 16 == 0 &&
                      TABLE_BYTES % 16 == 0 && CTX_BYTES % 16 == 0,
                  "16-byte parts");
};

// repro.rs phase_bytes mirrors this layout.
template <int D, int KEYS, int TILE>
struct PhaseLayout {
    static constexpr int UNITS = BLOCK / D * (TILE / 16), STEPS = (KEYS + 31) / 32;
    static constexpr int ROW_WORDS = UNITS * 16, PLANE_WORDS = 2 * 4 * 32;
    static constexpr int BYTES = 4 * (4 * ROW_WORDS + UNITS * STEPS * PLANE_WORDS);
};

// Keys order as their floats do, so integer atomics take float maxima; no NaN
// comes here.
__device__ __forceinline__ int repro_max_key(float x) {
    const int bits = __float_as_int(x);
    return bits >= 0 ? bits : bits ^ 0x7fffffff;
}
__device__ __forceinline__ float repro_key_max(int key) {
    return __int_as_float(key >= 0 ? key : key ^ 0x7fffffff);
}

// Without NaN, a tree's maximum equals the sequential one.
template <int N>
__device__ __forceinline__ float repro_max_of(const float (&v)[N]) {
    static_assert(N == 4 || N == 8, "four or eight");
    const float low = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
    if constexpr (N == 4)
        return low;
    else
        return fmaxf(low, fmaxf(fmaxf(v[4], v[5]), fmaxf(v[6], v[7])));
}

// repro_attention_narrow_rows' values over every warp: A row maxima, B integer totals, C
// products. Callers start the maxima at repro_max_key(-inf) and the totals at 0.
template <int D, int KEYS, int TILE, int CP, int QN, int KN, int CTW, int THREADS = ATT_PAIR_THREADS>
__device__ __forceinline__ void repro_attention_phases(
    const int* __restrict__ qrows, const float* __restrict__ qscale,
    const int* __restrict__ krows, const float* __restrict__ kscale,
    const int* __restrict__ vplane, const unsigned short* __restrict__ table,
    const float* __restrict__ vscale, const float* __restrict__ vfirst,
    const float* __restrict__ vlast, int* __restrict__ phase, int len, int row0, int row_blocks,
    int qb, unsigned failed, int warp, int lane, __half* __restrict__ ctx, int ctx_stride) {
    using P = PhaseLayout<D, KEYS, TILE>;
    static_assert(KEYS >= ATT_CHUNK, "packed pairs hold at least one key chunk");
    constexpr int KP = D / 4 + 4, WARPS = THREADS / 32, GROUPS = D / 8 / CTW, RB = TILE / 16;
    // With warps to spare, a step splits in two parts.
    constexpr int PARTS = WARPS >= 2 * P::UNITS * P::STEPS ? 2 : 1, TT = 4 / PARTS;
    constexpr int TASKS = P::UNITS * P::STEPS * PARTS, ITEMS = P::UNITS * GROUPS;
    constexpr int TASK_ROUNDS = (TASKS + WARPS - 1) / WARPS, ITEM_ROUNDS = (ITEMS + WARPS - 1) / WARPS;
    int* maxima = phase;
    int* totals = maxima + P::ROW_WORDS;
    int* sinks = totals + P::ROW_WORDS;
    unsigned* planes = reinterpret_cast<unsigned*>(sinks + 2 * P::ROW_WORDS);
    const int g = lane >> 2, t = lane & 3, steps = (len + 31) / 32;
    const float none = -__int_as_float(0x7f800000), factor = repro_exponent_factor<D>();
    auto task_of = [&](int k, int& unit, int& step, int& part, int& h, int& rb) {
        const int task = warp + k * WARPS;
        unit = task / (P::STEPS * PARTS);
        step = task / PARTS % P::STEPS;
        part = task % PARTS;
        h = unit / RB;
        rb = unit % RB;
        return task < TASKS && rb < row_blocks && step < steps && !(failed >> h & 1);
    };
    auto past = [&](int step, int q, int e) { return 32 * step + repro_key_of(q, 2 * t + e) >= len; };
    // Kept from phase A for phase B.
    float x[TASK_ROUNDS][TT][4];
#pragma unroll
    for (int k = 0; k < TASK_ROUNDS; ++k) {
        int unit, step, part, h, rb;
        if (!task_of(k, unit, step, part, h, rb)) continue;
        const int r0 = row0 + 16 * rb + g, r1 = r0 + 8;
        unsigned qa[D / 32][4];
#pragma unroll
        for (int st = 0; st < D / 32; ++st) {
            const int* q0 = qrows + h * QN * KP + (r0 - qb) * KP + 8 * st;
            const int* q1 = qrows + h * QN * KP + (r1 - qb) * KP + 8 * st;
            qa[st][0] = r0 < len ? (unsigned)q0[t] : 0u;
            qa[st][1] = r1 < len ? (unsigned)q1[t] : 0u;
            qa[st][2] = r0 < len ? (unsigned)q0[4 + t] : 0u;
            qa[st][3] = r1 < len ? (unsigned)q1[4 + t] : 0u;
        }
        // A tile pair's four keys are adjacent, so their scales load as one word.
        float ks[TT][2];
#pragma unroll
        for (int j = 0; j < TT / 2; ++j) {
            const int p = part * (TT / 2) + j;
            const float4 s = *reinterpret_cast<const float4*>(kscale + h * KN + 32 * step + 16 * p + 4 * t);
            ks[2 * j][0] = s.x, ks[2 * j][1] = s.y, ks[2 * j + 1][0] = s.z, ks[2 * j + 1][1] = s.w;
        }
        float row_g[2 * TT], row_g8[2 * TT];
#pragma unroll
        for (int j = 0; j < TT; ++j) {
            int acc[4];
            repro_score<D, KP>(krows + h * KN * KP, step, part * TT + j, g, t, qa, acc);
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                x[k][j][e] = repro_scaled_score(acc[e], ks[j][e]);
                x[k][j][2 + e] = repro_scaled_score(acc[2 + e], ks[j][e]);
                row_g[2 * j + e] = x[k][j][e];
                row_g8[2 * j + e] = x[k][j][2 + e];
            }
        }
        // Keys past the sequence take no part in the maxima.
        if (32 * (step + 1) > len) {
#pragma unroll
            for (int j = 0; j < TT; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e)
                    if (past(step, part * TT + j, e)) row_g[2 * j + e] = row_g8[2 * j + e] = none;
        }
        float max0 = repro_max_of(row_g), max1 = repro_max_of(row_g8);
        repro_merge_row_max(max0, max1);
        if (t == 0) {
            atomicMax(maxima + unit * 16 + g, repro_max_key(max0));
            atomicMax(maxima + unit * 16 + g + 8, repro_max_key(max1));
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TASK_ROUNDS; ++k) {
        int unit, step, part, h, rb;
        if (!task_of(k, unit, step, part, h, rb)) continue;
        const int r0 = row0 + 16 * rb + g, r1 = r0 + 8, at0 = unit * 16 + g, at1 = at0 + 8;
        // Rows past the sequence take no weights.
        const ReproRowShift shift = repro_row_shift(
            r0 < len ? repro_key_max(maxima[at0]) : none, r1 < len ? repro_key_max(maxima[at1]) : none,
            r0 < len ? qscale[h * QN + r0 - qb] : 0.0f, r1 < len ? qscale[h * QN + r1 - qb] : 0.0f, factor);
        int w0[2 * TT], w1[2 * TT];
#pragma unroll
        for (int j = 0; j < TT; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                w0[2 * j + e] = repro_weight(x[k][j][e], shift.max0, shift.factor0, shift.finite0, table);
                w1[2 * j + e] = repro_weight(x[k][j][2 + e], shift.max1, shift.factor1, shift.finite1, table);
            }
        // Keys past the sequence weigh nothing.
        if (32 * (step + 1) > len) {
#pragma unroll
            for (int j = 0; j < TT; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e)
                    if (past(step, part * TT + j, e)) w0[2 * j + e] = w1[2 * j + e] = 0;
        }
        // Only the threads holding the first and last keys keep sink weights;
        // the last key's thread inverts repro_key_of.
        if (step == 0 && part == 0 && t == 0) {
            sinks[2 * at0] = w0[0];
            sinks[2 * at1] = w1[0];
        }
        if (step == (len - 1) / 32) {
            const int key = (len - 1) % 32, q = 2 * (key / 16) + key % 4 / 2;
            if (q / TT == part && t == key % 16 / 4) {
                const int at = 2 * (q % TT) + key % 2;
                int last0 = 0, last1 = 0;
#pragma unroll
                for (int i = 0; i < 2 * TT; ++i)
                    if (i == at) last0 = w0[i], last1 = w1[i];
                sinks[2 * at0 + 1] = last0;
                sinks[2 * at1 + 1] = last1;
            }
        }
        int total0 = 0, total1 = 0;
#pragma unroll
        for (int i = 0; i < 2 * TT; ++i) {
            total0 += w0[i];
            total1 += w1[i];
        }
        for (int o = 1; o < 4; o *= 2) {
            total0 += __shfl_xor_sync(0xffffffffu, total0, o);
            total1 += __shfl_xor_sync(0xffffffffu, total1, o);
        }
        if (t == 0) {
            atomicAdd(totals + at0, total0);
            atomicAdd(totals + at1, total1);
        }
        unsigned* words = planes + (unit * P::STEPS + step) * P::PLANE_WORDS + lane;
#pragma unroll
        for (int j = 0; j < TT / 2; ++j) {
            const int p = part * (TT / 2) + j;
            unsigned high0, low0, high1, low1;
            repro_weight_planes(w0[4 * j], w0[4 * j + 1], w0[4 * j + 2], w0[4 * j + 3], high0, low0);
            repro_weight_planes(w1[4 * j], w1[4 * j + 1], w1[4 * j + 2], w1[4 * j + 3], high1, low1);
            words[2 * p * 32] = high0;
            words[(2 * p + 1) * 32] = high1;
            words[(4 + 2 * p) * 32] = low0;
            words[(5 + 2 * p) * 32] = low1;
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < ITEM_ROUNDS; ++k) {
        const int item = warp + k * WARPS, unit = item / GROUPS, ct0 = item % GROUPS * CTW;
        const int h = unit / RB, rb = unit % RB;
        if (item >= ITEMS || rb >= row_blocks || (failed >> h & 1)) continue;
        const int r0 = row0 + 16 * rb + g, r1 = r0 + 8, at0 = unit * 16 + g, at1 = at0 + 8;
        // Steps past the sequence have no planes from phase B, so no products.
        int hh[CTW][4] = {}, ll[CTW][4] = {};
#pragma unroll
        for (int step = 0; step < P::STEPS; ++step) {
            const unsigned* words = planes + (unit * P::STEPS + step) * P::PLANE_WORDS + lane;
            const unsigned ah[4] = {words[0], words[32], words[64], words[96]};
            const unsigned al[4] = {words[128], words[160], words[192], words[224]};
#pragma unroll
            for (int c = 0; c < CTW; ++c) {
                const int* v = vplane + h * D * CP + ((ct0 + c) * 8 + g) * CP + 8 * step;
                const unsigned bv[2] = {(unsigned)v[t], (unsigned)v[4 + t]};
                if (step < steps) {
                    int8proj::mma_s8(hh[c], ah, bv);
                    int8proj::mma_s8(ll[c], al, bv);
                }
            }
        }
        ReproSinks rows_sinks;
        rows_sinks.first0 = sinks[2 * at0];
        rows_sinks.first1 = sinks[2 * at1];
        rows_sinks.last0 = sinks[2 * at0 + 1];
        rows_sinks.last1 = sinks[2 * at1 + 1];
        repro_attention_finish<CTW>(vscale + h * D, vfirst + h * D, vlast + h * D, totals[at0],
                                    totals[at1], rows_sinks, hh, ll, ct0, r0, r1, len, t,
                                    ctx + h * D, ctx_stride, qb);
    }
}

// Made once a layer and shared by every query tile's block.
struct NarrowOperands {
    signed char *queries, *keys;
    float *query_scales, *key_scales;
    signed char* planes;
    float* value_scales;
    __half* values16;
    int* failed;
    const int* pstarts;
    int tokens, ptokens;
};

// Borrows the context's storage, which phase C writes only after the copies'
// barrier.
struct PrepackedStage {
    __half* sinks;
    int* flags;
};
__device__ __forceinline__ PrepackedStage prepacked_stage(__half* ctx) {
    return {ctx, reinterpret_cast<int*>(ctx + 2 * BLOCK)};
}

template <bool PREPACKED>
struct PairOperands {
    struct type {};
};
template <>
struct PairOperands<true> {
    using type = NarrowOperands;
};

#define PACK_PART_ROWS 16

// repro_attention_pairs' packing split over parts of the sequence; a finite channel's
// largest magnitude is the same in any order across the lanes that meet it.
template <int D, int LMAX, int THREADS = ATT_PAIR_THREADS>
__device__ __forceinline__ void repro_attention_pack(const float* __restrict__ qkv,
                                                     const int* __restrict__ starts,
                                                     const int* __restrict__ lengths, int hidden,
                                                     const float* __restrict__ key_centers,
                                                     const NarrowOperands& no, int s, int blk, int part) {
    constexpr int HB = BLOCK / D, PARTS = LMAX / PACK_PART_ROWS, CHANNELS = BLOCK / PARTS, TPL = LMAX / 4;
    constexpr int PACK_WARPS = THREADS / 32 - 4;
    static_assert(LMAX % PACK_PART_ROWS == 0 && TPL % 16 == 0 && 4 * CHANNELS % 32 == 0 && PACK_WARPS >= 1 &&
                      32 * PACK_WARPS + 4 * CHANNELS <= THREADS,
                  "whole parts and warps");
    const int tid = threadIdx.x, lane = tid & 31;
    const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
    const int start = __shfl_sync(0xffffffffu, starts[s], 0);
    const int len = __shfl_sync(0xffffffffu, lengths[s], 0);
    if (len > LMAX) __trap();
    const size_t stride = 3 * (size_t)hidden;
    int* flags = no.failed + (size_t)s * (hidden / D);
    if (warp < PACK_WARPS) {
        const int row0 = PACK_PART_ROWS * part, rows_end = min(len, row0 + PACK_PART_ROWS);
        const int per = rows_end > row0 ? (rows_end - row0 + PAIR_PACK_ROWS - 1) / PAIR_PACK_ROWS : 0;
        for (int i = warp; i < HB * 2 * per; i += PACK_WARPS) {
            const int hk = i / per, h = hk >> 1, key = hk & 1, j = row0 + i % per * PAIR_PACK_ROWS;
            const int head = blk * HB + h, count = min(PAIR_PACK_ROWS, rows_end - j);
            const float* from = qkv + (size_t)(start + j) * stride + key * hidden + head * D;
            const float* center = key ? key_centers + head * D : nullptr;
            const size_t row = (size_t)head * no.tokens + start + j;
            float* scales = (key ? no.key_scales : no.query_scales) + row;
            signed char* out = (key ? no.keys : no.queries) + row * D;
            const bool packed = repro_pack_rows<PAIR_PACK_ROWS>(from, stride, count, center, D, scales, out, D);
            if (!packed && lane == 0) atomicOr(flags + head, 1);
        }
    } else if (tid - 32 * PACK_WARPS < 4 * CHANNELS) {
        const int v = tid - 32 * PACK_WARPS, q = v & 3, cc = CHANNELS * part + (v >> 2);
        const size_t c = (size_t)BLOCK * blk + cc;
        const float* col = qkv + 2 * (size_t)hidden + c;
        float x[TPL];
#pragma unroll
        for (int e = 0; e < TPL; ++e) {
            const int j = TPL * q + e;
            x[e] = j < len ? col[(size_t)(start + j) * stride] : 0.0f;
        }
        unsigned most = 0;
        bool finite = true;
        float last = 0.0f;
#pragma unroll
        for (int e = 0; e < TPL; ++e) {
            const int j = TPL * q + e;
            if (j >= len) continue;
            x[e] = __half2float(__float2half_rn(x[e]));
            finite = finite && isfinite(x[e]);
            if (j != 0 && j != len - 1) most = max(most, __float_as_uint(x[e]) & 0x7fffffffu);
            if (j == len - 1) last = x[e];
        }
        most = max(most, __shfl_xor_sync(0xffffffffu, most, 1));
        most = max(most, __shfl_xor_sync(0xffffffffu, most, 2));
        if (!finite) atomicOr(flags + c / D, 1);
        const float scale = repro_scale_for_127(__uint_as_float(most));
        int b[TPL];
        repro_on_steps<TPL>(x, scale, repro_step_reciprocal(scale), b);
        unsigned words[TPL / 4];
#pragma unroll
        for (int w = 0; w < TPL / 4; ++w) {
            unsigned packed = 0;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int j = TPL * q + 4 * w + e;
                const int byte = j < len && j != 0 && j != len - 1 ? b[4 * w + e] : 0;
                packed |= (unsigned)(byte & 0xff) << (8 * e);
            }
            words[w] = packed;
        }
        // A sequence's plane holds its tokens padded to whole key chunks.
        if (TPL * q < (len + ATT_CHUNK - 1) / ATT_CHUNK * ATT_CHUNK) {
            uint4* plane = reinterpret_cast<uint4*>(no.planes + c * no.ptokens + no.pstarts[s] + TPL * q);
#pragma unroll
            for (int u = 0; u < TPL / 16; ++u)
                plane[u] = make_uint4(words[4 * u], words[4 * u + 1], words[4 * u + 2], words[4 * u + 3]);
        }
        if (q == 0) {
            no.value_scales[(size_t)s * hidden + c] = scale;
            no.values16[(size_t)start * hidden + c] = __float2half_rn(x[0]);
        }
        if ((unsigned)(len - 1) / TPL == (unsigned)q)
            no.values16[(size_t)(start + len - 1) * hidden + c] = __float2half_rn(last);
    }
}

#define REPRO_ATTENTION_PACK(D, LMAX)                                                              \
    extern "C" __global__ void __launch_bounds__(ATT_PAIR_THREADS) repro_attention_pack_##D##_##LMAX( \
        const float* qkv, const int* starts, const int* lengths, int hidden,                       \
        const float* key_centers, signed char* queries, float* query_scales, signed char* keys,     \
        float* key_scales, signed char* planes, float* value_scales, __half* values16,              \
        int* failed, const int* pstarts, int tokens, int ptokens) {                                 \
        if (blockDim.x != ATT_PAIR_THREADS) __trap();                                               \
        repro_attention_pack<D, LMAX>(qkv, starts, lengths, hidden, key_centers,                   \
                                      NarrowOperands{queries, keys, query_scales, key_scales,      \
                                                     planes, value_scales, values16, failed,       \
                                                     pstarts, tokens, ptokens},                    \
                                      blockIdx.x, blockIdx.y, blockIdx.z);                         \
    }
REPRO_ATTENTION_PACK(64, 64)
REPRO_ATTENTION_PACK(64, 128)
#undef REPRO_ATTENTION_PACK

// Attention and the out projection's input stage in one launch, as the separate kernels
// make them; the warp index comes from lane 0 so tests on it stay warp-uniform.
template <int D, int LMAX, bool STAGED = false, int TILE = LMAX, bool PREPACKED = false, int THREADS = ATT_PAIR_THREADS>
__device__ __forceinline__ void repro_attention_pairs(
    const float* __restrict__ qkv, const int* __restrict__ starts,
    const int* __restrict__ lengths, int hidden, int heads, const float* __restrict__ key_centers,
    const unsigned short* __restrict__ weights, float* __restrict__ value_scales,
    const float* __restrict__ centers, int center_rows, const int* __restrict__ classes,
    const int* __restrict__ group_of_row, const float* __restrict__ signs,
    const float* __restrict__ fc, signed char* __restrict__ quantized, float* __restrict__ steps, int tokens,
    __half* __restrict__ context, int s, int blk, const int* codes_s = nullptr,
    const float* rotate_s = nullptr, int start_s = 0, int len_s = 0, int row0 = 0,
    const typename PairOperands<PREPACKED>::type& no = {}) {
    constexpr bool WIDE_KEYS = LMAX > ATT_CHUNK;
    using L = PairLayout<D, WIDE_KEYS ? LMAX : ATT_CHUNK, WIDE_KEYS ? TILE : ATT_CHUNK, THREADS>;
    constexpr int HB = L::HB, KP = L::KP, CP = L::CP, CTX = L::CTX, WARPS = THREADS / 32;
    constexpr int PACK_WARPS = WARPS - 4;
    static_assert(THREADS % 32 == 0 && PACK_WARPS >= 1, "whole warps");
    static_assert(THREADS == ATT_PAIR_THREADS || PREPACKED,
                  "another thread count only for the per-layer packed launches");
    constexpr int QN = L::QROWS, KN = L::KROWS;
    constexpr int TILES = D / 8, UNITS = HB * TILE / 16;
    // Group channel tiles to amortize plane words and row terms while
    // retaining enough items to distribute across the warps.
    constexpr int CTW = UNITS * TILES <= 2 * WARPS ? 2 : 4;
    static_assert(LMAX == 16 || LMAX == 32 || LMAX == ATT_CHUNK ||
                      (PREPACKED && (LMAX == 2 * ATT_CHUNK || LMAX == 160)),
                  "a length bucket");
    static_assert(!PREPACKED || (!STAGED && LMAX >= ATT_CHUNK && TILE % PACK_PART_ROWS == 0),
                  "the per-layer buckets take whole packed row groups");
    static_assert(TILE % 16 == 0 && TILE <= LMAX && (!STAGED || TILE == LMAX), "whole row blocks");
    extern __shared__ __align__(16) unsigned char pair_shared[];
    using Phases = PhaseLayout<D, LMAX, TILE>;
    constexpr int PHASE_BYTES = PREPACKED ? Phases::BYTES : 0;
    {
        unsigned bytes;
        asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(bytes));
        if (bytes < (unsigned)(L::BYTES + PHASE_BYTES) || center_rows > PAIR_CENTER_ROWS) __trap();
    }
    unsigned char* at = pair_shared;
    int* qrows = reinterpret_cast<int*>(at);
    int* krows = reinterpret_cast<int*>(at += L::QROWS_BYTES);
    int* vplane = reinterpret_cast<int*>(at += L::KROWS_BYTES);
    float* qscale = reinterpret_cast<float*>(at += L::PLANE_BYTES);
    float* kscale = reinterpret_cast<float*>(at += L::QSCALE_BYTES);
    float* vscale = reinterpret_cast<float*>(at += L::KSCALE_BYTES);
    float* vfirst = reinterpret_cast<float*>(at += L::CHANNEL_BYTES);
    float* vlast = reinterpret_cast<float*>(at += L::CHANNEL_BYTES);
    unsigned short* table = reinterpret_cast<unsigned short*>(at += L::CHANNEL_BYTES);
    unsigned* warp_failed = reinterpret_cast<unsigned*>(at += L::TABLE_BYTES);
    float* centers_s = reinterpret_cast<float*>(at += L::FAILED_BYTES);
    float* signs_s = reinterpret_cast<float*>(at += L::CENTERS_BYTES);
    __half* ctx = reinterpret_cast<__half*>(at += L::SIGNS_BYTES);
    const int tid = threadIdx.x, lane = tid & 31;
    const int warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
    // These copies depend only on launch pointers, so issuing them first
    // overlaps table loading with the remaining address calculations.
    if constexpr (!STAGED) {
        for (int i = tid; i < ATT_WEIGHT_ENTRIES / 8; i += THREADS)
            cp_async_16(shared_address(table + 8 * i), weights + 8 * i, 16);
        for (int i = tid; i < (center_rows + 1) * BLOCK / 4; i += THREADS) {
            const int row = i / (BLOCK / 4), c = 4 * (i % (BLOCK / 4));
            const float* src = row < center_rows ? centers + (size_t)row * hidden + BLOCK * blk + c
                                                 : signs + BLOCK * blk + c;
            float* dst = row < center_rows ? centers_s + row * BLOCK + c : signs_s + c;
            cp_async_16(shared_address(dst), src, 16);
        }
        cp_async_commit();
    }
    // Taken from lane 0, so the compiler knows tests on them take a warp one
    // way, as with the warp index.
    const int given_len = STAGED ? len_s : lengths[s];
    const int given_start = STAGED ? start_s : starts[s];
    [[maybe_unused]] int pstart = 0;
    if constexpr (PREPACKED) pstart = no.pstarts[s];
    if (given_len > LMAX) __trap();
    const int start = __shfl_sync(0xffffffffu, given_start, 0);
    const int len = __shfl_sync(0xffffffffu, given_len, 0);
    // A tile past the sequence still lets its copies land before it leaves.
    if (row0 >= len) {
        cp_async_wait<0>();
        return;
    }
    const int rows_end = min(len, row0 + TILE);
    const int qb = WIDE_KEYS ? row0 : 0;
    [[maybe_unused]] const size_t stride = 3 * (size_t)hidden;
    // Lanes a row shrink as the tile grows, so its rows take one pass.
    constexpr int QV = TILE > 2 * WARPS ? 16 : TILE > WARPS ? 8 : 4, QL = BLOCK / QV, QR = 32 / QL;
    const int qrow = row0 + QR * warp + lane / QL, part = lane % QL;
    const bool qlive = qrow < rows_end;
    [[maybe_unused]] const int qgroup = STAGED || !qlive ? 0 : group_of_row[start + qrow];
    const float rotate = STAGED ? 0.0f : fc[FC_RECIP_SQRT_BLOCK];
    // Bit h: a non-finite query, centered key or FP16 value of head h.
    unsigned failed = 0;
    if constexpr (PREPACKED) {
        // A head at a time, so no index divides by a run's length; rolled, since this
        // runs once and unrolled indexing would dominate its instructions.
        const int qn = rows_end - row0, rn = qn + len;
#pragma unroll 1
        for (int h = 0; h < HB; ++h) {
            const size_t head_rows = (size_t)(blk * HB + h) * no.tokens + start;
#pragma unroll 1
            for (int i = tid; i < rn * (D / 16); i += THREADS) {
                const int piece = i % (D / 16), r = i / (D / 16);
                const bool key = r >= qn;
                const int row = key ? r - qn : row0 + r;
                unsigned char* dst = reinterpret_cast<unsigned char*>((key ? krows : qrows) + h * (key ? KN : QN) * KP) +
                                     (key ? row : row - qb) * KP * 4 + 16 * piece;
                cp_async_16(shared_address(dst), (key ? no.keys : no.queries) + (head_rows + row) * D + 16 * piece, 16);
            }
#pragma unroll 1
            for (int r = tid; r < rn; r += THREADS) {
                const bool key = r >= qn;
                const int row = key ? r - qn : row0 + r;
                cp_async_4(shared_address(key ? kscale + h * KN + row : qscale + h * QN + row - qb),
                           (key ? no.key_scales : no.query_scales) + head_rows + row);
            }
        }
        // Only the padded tokens' words are copied; the steps read no other.
        static_assert(THREADS % BLOCK == 0, "whole threads a channel");
        const PrepackedStage staged = prepacked_stage(ctx);
        // A batch's shorter sequence holds fewer padded pieces.
        constexpr int PIECES = KN / 16;
        const int pieces = (len + ATT_CHUNK - 1) / ATT_CHUNK * (ATT_CHUNK / 16);
        const signed char* channels = no.planes + (size_t)BLOCK * blk * no.ptokens + pstart;
#pragma unroll
        for (int j = 0; j < (BLOCK * PIECES + THREADS - 1) / THREADS; ++j) {
            const int i = tid + j * THREADS, hc = i / PIECES, piece = i % PIECES;
            if (i < BLOCK * PIECES && piece < pieces)
                cp_async_16(shared_address(vplane + hc * CP + 4 * piece),
                            channels + (size_t)hc * no.ptokens + 16 * piece, 16);
        }
        if (tid < BLOCK / 4)
            cp_async_16(shared_address(vscale + 4 * tid), value_scales + (size_t)s * hidden + BLOCK * blk + 4 * tid,
                        16);
        if (tid < 2 * BLOCK / 8) {
            const int last = tid / (BLOCK / 8), piece = tid % (BLOCK / 8);
            const size_t token = (size_t)start + (last ? len - 1 : 0);
            cp_async_16(shared_address(staged.sinks + last * BLOCK + 8 * piece),
                        no.values16 + token * hidden + BLOCK * blk + 8 * piece, 16);
        }
        if (tid < HB) cp_async_4(shared_address(staged.flags + tid), no.failed + (size_t)s * heads + blk * HB + tid);
        cp_async_commit();
        // repro_attention_phases needs its maxima at -inf's key and totals at 0.
        int* phase = reinterpret_cast<int*>(pair_shared + L::BYTES);
        for (int i = tid; i < 2 * Phases::ROW_WORDS; i += THREADS)
            phase[i] = i < Phases::ROW_WORDS ? repro_max_key(-__int_as_float(0x7f800000)) : 0;
    } else if (warp < PACK_WARPS) {
        const int per_q = (rows_end - row0 + PAIR_PACK_ROWS - 1) / PAIR_PACK_ROWS;
        const int per = per_q + (len + PAIR_PACK_ROWS - 1) / PAIR_PACK_ROWS;
        for (int i = warp; i < HB * per; i += PACK_WARPS) {
            const int h = i / per, r = i % per, key = r >= per_q;
            const int j = key ? (r - per_q) * PAIR_PACK_ROWS : row0 + r * PAIR_PACK_ROWS;
            const int head = blk * HB + h, count = min(PAIR_PACK_ROWS, (key ? len : rows_end) - j);
            signed char* out = reinterpret_cast<signed char*>((key ? krows : qrows) + h * (key ? KN : QN) * KP);
            const float* from = qkv + (size_t)(start + j) * stride + key * hidden + head * D;
            const float* center = key ? key_centers + head * D : nullptr;
            // A tiled layout holds query rows from row0.
            const int at = key ? j : j - qb;
            float* scales = (key ? kscale + h * KN : qscale + h * QN) + at;
            const bool packed = repro_pack_rows<PAIR_PACK_ROWS>(from, stride, count, center, D, scales,
                                                                out + at * KP * 4, KP * 4);
            if (!packed) failed |= 1u << h;
        }
    } else {
        const int cc = tid - 32 * PACK_WARPS, h = cc / D, ch = cc % D, head = blk * HB + h;
        const float* col = qkv + 2 * (size_t)hidden + head * D + ch;
        float x[LMAX];
#pragma unroll
        for (int j = 0; j < LMAX; ++j) x[j] = j < len ? col[(size_t)(start + j) * stride] : 0.0f;
        unsigned most = 0;
        bool finite = true;
        float last = 0.0f;
#pragma unroll
        for (int j = 0; j < LMAX; ++j) {
            if (j >= len) continue;
            x[j] = __half2float(__float2half_rn(x[j]));
            finite = finite && isfinite(x[j]);
            if (j != 0 && j != len - 1) most = max(most, __float_as_uint(x[j]) & 0x7fffffffu);
            if (j == len - 1) last = x[j];
        }
        if (!finite) failed |= 1u << h;
        const float scale = repro_scale_for_127(__uint_as_float(most));
        vscale[h * D + ch] = scale;
        vfirst[h * D + ch] = x[0];
        vlast[h * D + ch] = last;
        value_scales[((size_t)s * heads + head) * D + ch] = scale;
        // Sinks and tokens past the sequence are zero, as are the words past
        // LMAX tokens that a 32-key step reads.
        int v[LMAX];
        repro_on_steps<LMAX>(x, scale, repro_step_reciprocal(scale), v);
#pragma unroll
        for (int w = 0; w < (LMAX < 32 ? 32 : LMAX) / 4; ++w) {
            unsigned packed = 0;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int j = 4 * w + e;
                const int q = j < LMAX && j < len && j != 0 && j != len - 1 ? v[j < LMAX ? j : 0] : 0;
                packed |= (unsigned)(q & 0xff) << (8 * e);
            }
            vplane[(h * D + ch) * CP + w] = packed;
        }
    }
    failed = __reduce_or_sync(0xffffffffu, failed);
    if (lane == 0) warp_failed[warp] = failed;
    cp_async_wait<0>();
    __syncthreads();
    [[maybe_unused]] const int qclass = STAGED ? 1 : qlive ? classes[qgroup] : 1;
    failed = 0;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) failed |= warp_failed[w];
    if constexpr (PREPACKED) {
        // Read out before phase C overwrites the context's storage.
        static_assert(L::CTX_BYTES >= 4 * BLOCK + 4 * HB, "the context's storage holds the staged sinks and flags");
        const PrepackedStage staged = prepacked_stage(ctx);
#pragma unroll
        for (int h = 0; h < HB; ++h)
            if (staged.flags[h]) failed |= 1u << h;
        if (tid < BLOCK) {
            vfirst[tid] = __half2float(staged.sinks[tid]);
            vlast[tid] = __half2float(staged.sinks[BLOCK + tid]);
        }
    }
    const int row_blocks = (rows_end - row0 + 15) / 16;
    if constexpr (PREPACKED) {
        repro_attention_phases<D, LMAX, TILE, CP, QN, KN, CTW, THREADS>(
            qrows, qscale, krows, kscale, vplane, table, vscale, vfirst, vlast,
            reinterpret_cast<int*>(pair_shared + L::BYTES), len, row0, row_blocks, qb, failed, warp, lane, ctx,
            CTX);
    } else {
        constexpr int GROUPS = TILES / CTW;
        for (int item = warp; item < HB * row_blocks * GROUPS; item += WARPS) {
            const int unit = item / GROUPS, h = unit / row_blocks, rb = unit % row_blocks;
            if (failed >> h & 1) continue;
            const int g = lane >> 2, t = lane & 3, r0 = row0 + 16 * rb + g, ct0 = item % GROUPS * CTW;
            repro_attention_narrow_rows<D, CTW, LMAX, CP>(
                qrows + h * QN * KP, qscale + h * QN, krows + h * KN * KP, kscale + h * KN,
                vplane + h * D * CP, table, vscale + h * D, vfirst + h * D, vlast + h * D, len, r0, r0 + 8,
                ct0, g, t, ctx + h * D, CTX, qb);
        }
    }
    const int tile_values = (rows_end - row0) * BLOCK;
    if (failed)
        for (int i = tid; i < tile_values; i += THREADS)
            if (failed >> (i % BLOCK / D) & 1)
                ctx[(row0 - qb + i / BLOCK) * CTX + i % BLOCK] = __float2half_rn(__int_as_float(0x7fc00000));
    __syncthreads();
    if (context)
        for (int i = tid; i < tile_values; i += THREADS)
            context[(size_t)(start + row0 + i / BLOCK) * hidden + BLOCK * blk + i % BLOCK] =
                ctx[(row0 - qb + i / BLOCK) * CTX + i % BLOCK];
    // A row past the tile forms from spare context rows and stores nowhere,
    // so its lanes still take part in the rotation's shuffles.
    for (int vrow = qrow; vrow - lane / QL < rows_end; vrow += QR * WARPS) {
        const bool live = vrow < rows_end;
        int center_row;
        if constexpr (STAGED)
            center_row = live ? -codes_s[start + vrow] - 1 : 0;
        else
            center_row = (vrow == qrow ? qclass : live ? classes[group_of_row[start + vrow]] : 1) - 1;
        const int c = QV * part;
        const float* center = centers_s + center_row * BLOCK + c;
        const __half* values = ctx + min(vrow - qb, QN - 1) * CTX + c;
        unsigned words[QV / 2];
        if constexpr (QV == 4) {
            const uint2 raw = *reinterpret_cast<const uint2*>(values);
            words[0] = raw.x, words[1] = raw.y;
        } else {
#pragma unroll
            for (int half = 0; half < QV / 8; ++half) {
                const uint4 raw = *reinterpret_cast<const uint4*>(values + 8 * half);
                words[4 * half] = raw.x, words[4 * half + 1] = raw.y;
                words[4 * half + 2] = raw.z, words[4 * half + 3] = raw.w;
            }
        }
        float x[QV];
#pragma unroll
        for (int e = 0; e < QV / 2; ++e) {
            const int k = 2 * e;
            const float2 pair = __half22float2(*reinterpret_cast<const __half2*>(&words[e]));
            // Explicit 8-byte loads: nvcc otherwise splits some from one build
            // to the next.
            const float2 cv = *reinterpret_cast<const float2*>(center + k);
            const float2 sv = *reinterpret_cast<const float2*>(signs_s + c + k);
            x[k] = __fmul_rn(__fsub_rn(pair.x, cv.x), sv.x < 0.0f ? -1.0f : 1.0f);
            x[k + 1] = __fmul_rn(__fsub_rn(pair.y, cv.y), sv.y < 0.0f ? -1.0f : 1.0f);
        }
        const float most = repro_rotate_block<QV>(x, part, STAGED ? *rotate_s : rotate);
        const float scale = most > 0.0f ? __fdiv_rn(most, 127.0f) : 1.0f;
        if (live) {
            repro_store_block<QV>(x, scale, quantized + (size_t)(start + vrow) * hidden + BLOCK * blk + c);
            if (part % (QL / 2) == 0)
                *repro_step_at(steps, tokens, 2 * blk + part / (QL / 2), start + vrow) = scale;
        }
    }
}

// A minimum of one block: the layout leaves a multiprocessor no second, and the assembler
// otherwise holds the registers to two blocks' budget and spills the longer rows' values.
#define REPRO_ATTENTION_PAIRS(D, LMAX)                                                             \
    extern "C" __global__ void __launch_bounds__(ATT_PAIR_THREADS, 1)                            \
        repro_attention_pairs_##D##_##LMAX(                                                       \
            const float* qkv, const int* starts, const int* lengths, int hidden, int heads,       \
            const float* key_centers, const unsigned short* weights, float* value_scales,         \
            const float* centers, int center_rows, const int* classes, const int* group_of_row,   \
            const float* signs, const float* fc, signed char* quantized, float* steps,            \
            __half* context, int tokens) {                                                        \
        repro_attention_pairs<D, LMAX, false, 16>(                                                \
            qkv, starts, lengths, hidden, heads, key_centers, weights, value_scales, centers,     \
            center_rows, classes, group_of_row, signs, fc, quantized, steps, tokens, context,     \
            blockIdx.x, blockIdx.y, nullptr, nullptr, 0, 0, 16 * (int)blockIdx.z);                \
    }
REPRO_ATTENTION_PAIRS(64, 16)
REPRO_ATTENTION_PAIRS(64, 32)

// repro.rs PACKED_BUCKETS; the 128-key buckets take 512 threads, a warp for
// each of their sixteen tasks.
#define REPRO_ATTENTION_PAIRS_PACKED(D, LMAX, TILE, THREADS)                                       \
    extern "C" __global__ void __launch_bounds__(THREADS)                                        \
        repro_attention_pairs_packed_##D##_##LMAX##_##TILE(                                       \
            const int* starts, const int* lengths, int hidden, int heads,                         \
            const unsigned short* weights, float* value_scales, const float* centers,             \
            int center_rows, const int* classes, const int* group_of_row, const float* signs,     \
            const float* fc, signed char* quantized, float* steps, __half* context,               \
            signed char* queries, float* query_scales, signed char* keys, float* key_scales,      \
            signed char* planes, __half* values16, int* failed, const int* pstarts,               \
            int tokens, int ptokens) {                                                            \
        repro_attention_pairs<D, LMAX, false, TILE, true, THREADS>(                               \
            nullptr, starts, lengths, hidden, heads, nullptr, weights, value_scales, centers,     \
            center_rows, classes, group_of_row, signs, fc, quantized, steps, tokens, context,     \
            blockIdx.x, blockIdx.y, nullptr, nullptr, 0, 0, TILE * (int)blockIdx.z,               \
            NarrowOperands{queries, keys, query_scales, key_scales, planes, value_scales,         \
                           values16, failed, pstarts, tokens, ptokens});                          \
    }
REPRO_ATTENTION_PAIRS_PACKED(64, 64, 16, ATT_PAIR_THREADS)
REPRO_ATTENTION_PAIRS_PACKED(64, 128, 16, 512)
REPRO_ATTENTION_PAIRS_PACKED(64, 128, 32, 512)

// encoder.rs lane_sum's order: the xor halving leaves the CPU's total in
// every lane.
__device__ __forceinline__ double repro_square_sum(const double* m, int h) {
    const int lane = threadIdx.x & 31;
    double part = 0.0;
    for (int c = lane; c < h; c += 32) part = __dadd_rn(part, __dmul_rn(m[c], m[c]));
#pragma unroll
    for (int w = 16; w > 0; w >>= 1) part = __dadd_rn(part, __shfl_xor_sync(0xffffffffu, part, w));
    return part;
}

extern "C" __global__ void repro_square_sum_check(const double* values, int h, int count,
                                                  double* sums) {
    const int i = blockIdx.x;
    if (i >= count) return;
    const double total = repro_square_sum(values + (size_t)i * h, h);
    if (threadIdx.x == 0) sums[i] = total;
}

// repro.rs POOL_ROWS.
#define POOL_ROWS 4

// Flags OR across rows, so a column's sum follows without the CPU's order.
__device__ __forceinline__ unsigned repro_pool_kind(__half x) {
    const unsigned bits = __half_as_ushort(x);
    const unsigned kind = (bits & 0x3ffu) ? 1u : (bits & 0x8000u) ? 4u : 2u;
    return (bits & 0x7c00u) == 0x7c00u ? kind : 0u;
}

// The CPU's row-order sum ends NaN with a NaN or both infinities, else an
// infinity, else the exact total.
__device__ __forceinline__ double repro_pool_sum(long long units, unsigned kinds) {
    if ((kinds & 1u) || (kinds & 6u) == 6u) return __longlong_as_double(0x7ff8000000000000ll);
    const double infinity = __longlong_as_double(0x7ff0000000000000ll);
    if (kinds) return (kinds & 2u) ? infinity : -infinity;
    return __dmul_rn(__ll2double_rn(units), 0x1p-24);
}

// Granlund and Montgomery's invariant reciprocal, formed once a block at entry.
struct ReproDivisor {
    unsigned d, m, sh1, sh2;
};
__device__ __forceinline__ ReproDivisor repro_divisor(unsigned d) {
    const unsigned l = 32 - __clz(d - 1);
    const unsigned m = (unsigned)((((1ull << l) - d) << 32) / d) + 1u;
    return {d, m, min(l, 1u), l ? l - 1 : 0u};
}

__device__ __forceinline__ unsigned repro_divide(unsigned n, const ReproDivisor& v) {
    const unsigned t = __umulhi(v.m, n);
    return (t + ((n - t) >> v.sh1)) >> v.sh2;
}

// RN(units / len) 2^-24 by integer long division: the quotient has at least 55 bits, so
// bit 0 carries the remainder's sticky bit and __ull2double_rn rounds as RN does.
__device__ __forceinline__ double repro_pool_mean(long long units, const ReproDivisor& v) {
    if (units == 0) return 0.0;
    const unsigned long long a = units < 0 ? 0ull - (unsigned long long)units : (unsigned long long)units;
    const int k = __clzll(a);
    const unsigned long long u = a << k;
    const unsigned high = (unsigned)(u >> 32), qh = repro_divide(high, v), rh = high - qh * v.d;
    const unsigned middle = (rh << 21) | ((unsigned)(u >> 11) & 0x1fffffu);
    const unsigned qm = repro_divide(middle, v), rm = middle - qm * v.d;
    const unsigned low = (rm << 11) | ((unsigned)u & 0x7ffu), ql = repro_divide(low, v), rl = low - ql * v.d;
    const unsigned long long q = ((unsigned long long)qh << 32) | ((unsigned long long)qm << 11) | ql;
    const double mean = __longlong_as_double(__double_as_longlong(__ull2double_rn(q | (rl != 0))) -
                                             ((long long)(k + 24) << 52));
    return units < 0 ? -mean : mean;
}

// The CPU's `(m / norm) as f32` from y = RN(1 / n): q = RN(m y) is within 3 ulp of
// RN(m / n), and a midpoint within 4 ulp of q takes the exact quotient instead.
__device__ __forceinline__ float repro_pool_quotient(double m, double n, double y) {
    const double q = __dmul_rn(m, y);
    const unsigned below = (unsigned)__double_as_longlong(q) & 0x1fffffffu;
    return __double2float_rn(below - (0x10000000u - 4u) <= 8u ? __ddiv_rn(m, n) : q);
}

// repro_pool_finish stages a sequence's means in shared memory.
#define POOL_MAX_HIDDEN 1024
// repro.rs POOL_THREADS.
#define POOL_THREADS 256
static_assert(NORM_BATCH_ROWS * 32 == POOL_THREADS, "repro_layernorm_pool's warps are its pooling threads");

// A sequence's values add to below 2^49 units, where every f64 partial sum
// the CPU forms is exact, so the INT64 total is that sum in any grouping.
__device__ __forceinline__ void repro_pool_add(__half x, long long& total, unsigned& kind) {
    const unsigned k = repro_pool_kind(x);
    total += k ? 0 : fp16_units(x);
    kind |= k;
}

struct ReproStageUnits {
    long long* units;
    unsigned* kinds;
    __device__ __forceinline__ void operator()(int c, __half value) const {
        const unsigned k = repro_pool_kind(value);
        units[c] = k ? 0 : fp16_units(value);
        if (k) atomicOr(kinds + c, k);
    }
};

// The last part to count itself finishes the sequence, ordered as repro_restored_next_input
// orders its stores; a value and its mark share one 64-bit store, so the host polls the mark.
template <int COLUMNS>
__device__ __forceinline__ void repro_pool_finish(long long (&total)[COLUMNS], unsigned (&kind)[COLUMNS], int s,
                                                  const ReproDivisor& v, int parts, int h, long long* units,
                                                  unsigned* kinds, int* counts, unsigned long long* out) {
    __shared__ double means[POOL_MAX_HIDDEN];
    __shared__ double norm, reciprocal;
    __shared__ int last;
    const size_t base = (size_t)s * h;
    if (parts > 1) {
#pragma unroll
        for (int k = 0; k < COLUMNS; ++k) {
            const int c = threadIdx.x + POOL_THREADS * k;
            if (c < h) {
                atomicAdd(reinterpret_cast<unsigned long long*>(units + base + c), (unsigned long long)total[k]);
                if (kind[k]) atomicOr(kinds + base + c, kind[k]);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            last = atomicAdd(counts + s, 1) == parts - 1;
            if (last) {
                counts[s] = 0;
                __threadfence();
            }
        }
        __syncthreads();
        if (!last) return;
#pragma unroll
        for (int k = 0; k < COLUMNS; ++k) {
            const int c = threadIdx.x + POOL_THREADS * k;
            if (c < h) {
                total[k] = __ldcg(units + base + c);
                kind[k] = __ldcg(kinds + base + c);
                units[base + c] = 0;
                kinds[base + c] = 0;
            }
        }
    }
    double mean[COLUMNS];
#pragma unroll
    for (int k = 0; k < COLUMNS; ++k) {
        const int c = threadIdx.x + POOL_THREADS * k;
        if (c < h) {
            mean[k] = kind[k] ? __ddiv_rn(repro_pool_sum(total[k], kind[k]), (double)v.d)
                              : repro_pool_mean(total[k], v);
            means[c] = mean[k];
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const double squares = repro_square_sum(means, h);
        if (threadIdx.x == 0) {
            norm = fmax(__dsqrt_rn(squares), 1e-12);
            reciprocal = __drcp_rn(norm);
        }
    }
    __syncthreads();
    const double n = norm, y = reciprocal;
#pragma unroll
    for (int k = 0; k < COLUMNS; ++k) {
        const int c = threadIdx.x + POOL_THREADS * k;
        if (c < h) out[base + c] = 1ull << 32 | __float_as_uint(repro_pool_quotient(mean[k], n, y));
    }
}

template <int COLUMNS>
__device__ __forceinline__ void repro_pool_sums(const __half* hidden, const int* starts, const int* lengths, int h,
                                                long long* units, unsigned* kinds, int* counts,
                                                unsigned long long* out) {
    __shared__ ReproDivisor divisor;
    if (threadIdx.x == 0 && (h > COLUMNS * POOL_THREADS || blockDim.x != POOL_THREADS)) __trap();
    const int s = blockIdx.x, len = lengths[s], first = POOL_ROWS * blockIdx.y;
    if (first >= len) return;
    const int rows = min(POOL_ROWS, len - first), parts = (len + POOL_ROWS - 1) / POOL_ROWS;
    if (threadIdx.x == 0) divisor = repro_divisor(len);
    const __half* part = hidden + (size_t)(starts[s] + first) * h;
    // Every load is in flight before any value is classified.
    __half x[POOL_ROWS][COLUMNS];
#pragma unroll
    for (int r = 0; r < POOL_ROWS; ++r)
#pragma unroll
        for (int k = 0; k < COLUMNS; ++k) {
            const int c = threadIdx.x + POOL_THREADS * k;
            x[r][k] = r < rows && c < h ? part[(size_t)r * h + c] : __ushort_as_half(0);
        }
    long long total[COLUMNS] = {};
    unsigned kind[COLUMNS] = {};
#pragma unroll
    for (int r = 0; r < POOL_ROWS; ++r)
#pragma unroll
        for (int k = 0; k < COLUMNS; ++k) repro_pool_add(x[r][k], total[k], kind[k]);
    // A single part has no join's barrier to show it the divisor.
    if (parts == 1) __syncthreads();
    repro_pool_finish(total, kind, s, divisor, parts, h, units, kinds, counts, out);
}

// Small row groups spread the joined sums' loads across multiprocessors.
#define POOL_JOINED_ROWS 2
#define POOL_STAGED_BLOCKS 4

// repro.rs pool_norm_bytes mirrors this layout.
struct PoolNormShared {
    float* params;
    long long* units;
    unsigned* kinds;
    float* stage;
};
template <int ROWS>
__device__ __forceinline__ PoolNormShared pool_norm_shared(void* base, int h) {
    float* params = static_cast<float*>(base);
    long long* units = reinterpret_cast<long long*>(params + 2 * h);
    unsigned* kinds = reinterpret_cast<unsigned*>(units + ROWS * h);
    return {params, units, kinds, reinterpret_cast<float*>(kinds + h)};
}

// Joined rows load four adjacent values a lane and pass through `stage` into
// the statistics' layout; each run's loads leave before its values form.
template <int PASSES, class Row>
__device__ __forceinline__ void repro_row_lanes(const Row& row, float* stage, int h, float (&x)[32]) {
    const int lane = threadIdx.x & 31;
    if constexpr (Row::STAGED) {
#pragma unroll
        for (int p = 0; p < PASSES; ++p) {
            typename Row::Raw raw[POOL_STAGED_BLOCKS];
#pragma unroll
            for (int i = 0; i < POOL_STAGED_BLOCKS; ++i)
                if (BLOCK * (POOL_STAGED_BLOCKS * p + i) < h)
                    raw[i] = row.load4(BLOCK * (POOL_STAGED_BLOCKS * p + i) + 4 * lane);
#pragma unroll
            for (int i = 0; i < POOL_STAGED_BLOCKS; ++i)
                if (BLOCK * (POOL_STAGED_BLOCKS * p + i) < h) {
                    float in[4];
                    row.form4(raw[i], in);
                    *reinterpret_cast<float4*>(stage + BLOCK * (POOL_STAGED_BLOCKS * p + i) + 4 * lane) =
                        make_float4(in[0], in[1], in[2], in[3]);
                }
        }
        __syncwarp();
#pragma unroll
        for (int i = 0; i < 32; ++i)
            if (32 * i < h) x[i] = stage[lane + 32 * i];
    } else {
#pragma unroll
        for (int i = 0; i < 32; ++i)
            if (32 * i < h) x[i] = row.value(lane + 32 * i);
    }
}

// The last layer's second LayerNorm stages the pooling's units in the same
// lane loop, so pooling needs no second pass over the rows.
template <int COLUMNS, int ROWS, class Rows>
__device__ __forceinline__ void repro_layernorm_pool(void* shared, const Rows& rows, __half* out,
                                                     const float* gamma, const float* beta, float eps, int h,
                                                     const int* starts, const int* lengths, long long* units,
                                                     unsigned* kinds, int* counts, unsigned long long* vectors) {
    using Row = decltype(rows.row(0, 0));
    constexpr int PASSES = COLUMNS * POOL_THREADS / (POOL_STAGED_BLOCKS * BLOCK);
    __shared__ ReproDivisor divisor;
    if (threadIdx.x == 0 && (h > COLUMNS * POOL_THREADS || blockDim.x != POOL_THREADS ||
                             (Row::STAGED && h > PASSES * POOL_STAGED_BLOCKS * BLOCK)))
        __trap();
    const PoolNormShared at = pool_norm_shared<ROWS>(shared, h);
    const int s = blockIdx.x, len = lengths[s], first = ROWS * blockIdx.y;
    if (first >= len) return;
    const int count = min(ROWS, len - first), row0 = starts[s] + first;
    if (threadIdx.x == 0) divisor = repro_divisor(len);
    repro_stage_norm(at.params, h, gamma, beta);
#pragma unroll
    for (int k = 0; k < COLUMNS; ++k) {
        const int c = threadIdx.x + POOL_THREADS * k;
        if (c < h) at.kinds[c] = 0;
    }
    const int warp = threadIdx.x >> 5;
    float x[32];
    if (warp < count) repro_row_lanes<PASSES>(rows.row(row0 + warp, h), at.stage + warp * h, h, x);
    cp_async_wait<0>();
    __syncthreads();
    if (warp < count)
        repro_layernorm_lanes_on(x, repro_row_moments(x, eps, h), out + (size_t)(row0 + warp) * h,
                                 ReproStageUnits{at.units + warp * h, at.kinds}, at.params, at.params + h, h);
    __syncthreads();
    long long total[COLUMNS] = {};
    unsigned kind[COLUMNS] = {};
#pragma unroll
    for (int k = 0; k < COLUMNS; ++k) {
        const int c = threadIdx.x + POOL_THREADS * k;
        if (c < h) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r)
                if (r < count) total[k] += at.units[r * h + c];
            kind[k] = at.kinds[c];
        }
    }
    repro_pool_finish(total, kind, s, divisor, (len + ROWS - 1) / ROWS, h, units, kinds, counts, vectors);
}

// repro.rs pool_columns.
#define REPRO_POOL(COLUMNS)                                                                                   \
    extern "C" __global__ void __launch_bounds__(POOL_THREADS) repro_pool_sums_##COLUMNS(                   \
        const __half* hidden, const int* starts, const int* lengths, int h, long long* units,               \
        unsigned* kinds, int* counts, unsigned long long* out) {                                             \
        repro_pool_sums<COLUMNS>(hidden, starts, lengths, h, units, kinds, counts, out);                    \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(POOL_THREADS) repro_layernorm_pool_##COLUMNS(              \
        const __half* rows, __half* out, const float* gamma, const float* beta, float eps, int h,           \
        const int* starts, const int* lengths, long long* units, unsigned* kinds, int* counts,              \
        unsigned long long* vectors) {                                                                       \
        extern __shared__ float4 shared[];                                                                   \
        repro_layernorm_pool<COLUMNS, NORM_BATCH_ROWS>(shared, ReproHalfRows{rows}, out, gamma, beta, eps, h, \
                                                       starts, lengths, units, kinds, counts, vectors);     \
    }
REPRO_POOL(4)
#undef REPRO_POOL

// In place over the residual stream, as repro_layernorm_quantize_joined forms
// the rows.
#define REPRO_LAYERNORM_POOL_JOINED(COLUMNS)                                                                       \
    extern "C" __global__ void __launch_bounds__(POOL_THREADS) repro_layernorm_pool_joined_##COLUMNS(             \
        const float* partials, int rows, const float* col_scales, const float* statics, const int* row_codes,     \
        __half* stream, const float* gamma, const float* beta, float eps, int h, const int* starts,               \
        const int* lengths, long long* units, unsigned* kinds, int* counts, unsigned long long* vectors) {        \
        extern __shared__ float4 shared[];                                                                         \
        repro_layernorm_pool<COLUMNS, POOL_JOINED_ROWS>(                                                           \
            shared, ReproJoinedRows<4>{partials, rows, col_scales, statics, row_codes, stream}, stream, gamma,     \
            beta, eps, h, starts, lengths, units, kinds, counts, vectors);                                         \
    }
REPRO_LAYERNORM_POOL_JOINED(4)
#undef REPRO_LAYERNORM_POOL_JOINED

extern "C" __global__ void repro_pool_mean_check(const long long* units, const int* lens, int count,
                                                 double* means) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) means[i] = repro_pool_mean(units[i], repro_divisor(lens[i]));
}

extern "C" __global__ void repro_pool_quotient_check(const double* means, const double* norms, int count,
                                                     float* quotients) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) quotients[i] = repro_pool_quotient(means[i], norms[i], __drcp_rn(norms[i]));
}

#include "repro_stream.cuh"
#include "repro_hop.cuh"

using QkvPacked160 = int8proj::Tiling<160, 64, 16, 32, 4>;
static_assert(QkvPacked160::THREADS == 640 && QkvPacked160::SHARED_BYTES == 57344,
              "the host launches these threads and bytes");
REPRO_RESTORED_QKV_PACKED(64_160, QkvPacked160, false)
REPRO_RESTORED_QKV_PACKED(64_160_two, QkvPacked160, true)
#undef REPRO_RESTORED_QKV_PACKED
REPRO_ATTENTION_PAIRS_PACKED(64, 160, 32, 512)
#undef REPRO_ATTENTION_PAIRS_PACKED
