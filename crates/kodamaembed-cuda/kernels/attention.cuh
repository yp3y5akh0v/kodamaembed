// Scores stay in registers and serve as P * V's A operand. CORRECTED adds 2^-11 (q_low . k_high +
// q_high . k_low) to each q_high . k_high, omitting q_low . k_low at 2^-22, for sequences shorter
// than `corrected_below` tokens. attention() and pipelined() take every query row through the same
// steps over key blocks of 64, so with the same CORRECTED they write identical bits. LOCAL keeps
// each query to the keys within `reach` rows of it and steps over the key blocks that hold one.
#pragma once

#include <cuda_fp16.h>

#include "common.cuh"

namespace attention_kernel {

constexpr int BLOCK = 64;        // keys per step, and query rows per CTA of attention()
constexpr int THREADS = 128;     // attention()'s four warps, 16 query rows each
constexpr float P_SHIFT = 8.0f;  // attention weights are rounded to FP16 as p * 2^8
constexpr float LOW_WEIGHT = 1.0f / LOW_SCALE;
constexpr float LOG2E = 1.4426950408889634f;

// Rows of D halfs are D / 8 chunks of 16 bytes, XOR-swizzled by row so the
// eight rows of an ldmatrix phase hit eight distinct bank groups.
template <int D>
__device__ __forceinline__ int tile_offset(int row, int chunk) {
    static_assert(D == 32 || D == 64, "four or eight chunks per row");
    if constexpr (D == 64) {
        return row * D + ((chunk ^ (row & 7)) << 3);
    } else {
        return row * D + ((chunk ^ ((row >> 1) & 3)) << 3);
    }
}

// Copies BLOCK rows of D halfs with COPIERS threads; rows at or past `valid` (at least 1) are
// zero-filled.
template <int D, int COPIERS>
__device__ __forceinline__ void load_rows(__half* tile, const __half* source, int valid, int stride) {
    constexpr int CHUNKS = D / 8;
    static_assert(BLOCK * CHUNKS % COPIERS == 0, "even copies");
#pragma unroll
    for (int j = 0; j < BLOCK * CHUNKS / COPIERS; ++j) {
        const int i = threadIdx.x + j * COPIERS;
        const int row = i / CHUNKS, chunk = i % CHUNKS;
        const __half* from = source + static_cast<size_t>(min(row, valid - 1)) * stride + chunk * 8;
        cp_async_16(shared_address(tile + tile_offset<D>(row, chunk)), from, row < valid ? 16 : 0);
    }
}

// This warp's 16 query rows of a staged tile as MMA A fragments.
template <int D>
__device__ __forceinline__ void load_query(unsigned (&frag)[D / 16][4], const __half* tile) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
        const int row = warp * 16 + (lane & 15);
        ldmatrix_x4(frag[ks], shared_address(tile + tile_offset<D>(row, ks * 2 + (lane >> 4))));
    }
}

// The fragments load_query gives, read straight from global memory, which frees the query tile;
// rows at or past `valid` are zero, as a zero-filled tile gives.
template <int D>
__device__ __forceinline__ void load_query_global(unsigned (&frag)[D / 16][4], const __half* rows,
                                                  int stride, int valid) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int r0 = warp * 16 + (lane >> 2), r1 = r0 + 8, c = (lane & 3) * 2;
    const auto at = [&](int r, int k) {
        return r < valid
                   ? __ldg(reinterpret_cast<const unsigned*>(rows + static_cast<size_t>(r) * stride + k))
                   : 0u;
    };
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
        const int col = ks * 16 + c;
        frag[ks][0] = at(r0, col);
        frag[ks][1] = at(r1, col);
        frag[ks][2] = at(r0, col + 8);
        frag[ks][3] = at(r1, col + 8);
    }
}

// One key block's scores: q_high . k_high in FP32 (s) and, if CORRECTED, the correction in FP16 (c).
template <int D, bool CORRECTED>
__device__ __forceinline__ void scores(float (&s)[BLOCK / 8][4], unsigned (&c)[BLOCK / 8][2],
                                       const unsigned (&q_high)[D / 16][4],
                                       const unsigned (&q_low)[D / 16][4], const __half* kh,
                                       const __half* kl) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int nt = 0; nt < BLOCK / 8; ++nt) {
        if constexpr (CORRECTED) c[nt][0] = c[nt][1] = 0u;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[nt][i] = 0.0f;
    }
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
#pragma unroll
        for (int np = 0; np < BLOCK / 16; ++np) {
            const int row = np * 16 + (lane & 7) + ((lane >> 4) << 3);
            const int at = tile_offset<D>(row, ks * 2 + ((lane >> 3) & 1));
            unsigned h[4];
            ldmatrix_x4(h, shared_address(kh + at));
            const unsigned h0[2] = {h[0], h[1]}, h1[2] = {h[2], h[3]};
            if constexpr (CORRECTED) {
                unsigned l[4];
                ldmatrix_x4(l, shared_address(kl + at));
                const unsigned l0[2] = {l[0], l[1]}, l1[2] = {l[2], l[3]};
                mma_16816(s[2 * np], q_high[ks], h0);
                mma_16816(s[2 * np + 1], q_high[ks], h1);
                mma_16816_f16(c[2 * np], q_low[ks], h0);
                mma_16816_f16(c[2 * np + 1], q_low[ks], h1);
                mma_16816_f16(c[2 * np], q_high[ks], l0);
                mma_16816_f16(c[2 * np + 1], q_high[ks], l1);
            } else {
                mma_16816(s[2 * np], q_high[ks], h0);
                mma_16816(s[2 * np + 1], q_high[ks], h1);
            }
        }
    }
}

// A warp's output and online-softmax state for its 16 query rows; each lane holds two rows.
template <int D>
struct Rows {
    float o[D / 8][4];
    float max[2], sum[2];

    __device__ __forceinline__ void start() {
#pragma unroll
        for (int dt = 0; dt < D / 8; ++dt)
#pragma unroll
            for (int i = 0; i < 4; ++i) o[dt][i] = 0.0f;
        max[0] = max[1] = negative_infinity();
        sum[0] = sum[1] = 0.0f;
    }
};

// MPNet's relative-position bias: `table` is the head's row of [head][key - query + longest - 1]
// at key - query = 0, and `query` the lane's two query rows, those past the sequence end clamped
// to its last row so they read inside the table.
struct RelativeBias {
    const float* table;
    int query[2];
};

// A local layer's window: the lane's two query rows and how far from one a key may lie.
struct Window {
    int query[2];
    int reach;
};

// Key block kb's scaled scores in the log2 domain (keys past the sequence end, or with LOCAL
// outside the query's window, get -inf), the online-softmax step and P * V.
template <int D, bool BIAS, bool CORRECTED, bool LOCAL = false>
__device__ __forceinline__ void softmax_pv(float (&s)[BLOCK / 8][4], const unsigned (&c)[BLOCK / 8][2],
                                           Rows<D>& rows, const __half* v, int kb, int len,
                                           float scale_log2e, const RelativeBias& bias,
                                           const Window& window = Window{}) {
    const int lane = threadIdx.x & 31;
    const int key0 = kb * BLOCK + (lane & 3) * 2;
    float block_max[2] = {negative_infinity(), negative_infinity()};
#pragma unroll
    for (int nt = 0; nt < BLOCK / 8; ++nt) {
        float correction[4] = {};
        if constexpr (CORRECTED) {
            const float2 top = __half22float2(*reinterpret_cast<const __half2*>(&c[nt][0]));
            const float2 bottom = __half22float2(*reinterpret_cast<const __half2*>(&c[nt][1]));
            correction[0] = top.x;
            correction[1] = top.y;
            correction[2] = bottom.x;
            correction[3] = bottom.y;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int key = key0 + nt * 8 + (i & 1);
            const float score = CORRECTED ? fmaf(correction[i], LOW_WEIGHT, s[nt][i]) : s[nt][i];
            bool seen = key < len;
            if constexpr (LOCAL) seen = seen && abs(key - window.query[i >> 1]) <= window.reach;
            s[nt][i] = seen ? score * scale_log2e : negative_infinity();
            if constexpr (BIAS) {
                if (key < len) s[nt][i] = fmaf(bias.table[key - bias.query[i >> 1]], LOG2E, s[nt][i]);
            }
            block_max[i >> 1] = fmaxf(block_max[i >> 1], s[nt][i]);
        }
    }
    float scale[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        block_max[r] = fmaxf(block_max[r], __shfl_xor_sync(0xffffffffu, block_max[r], 1));
        block_max[r] = fmaxf(block_max[r], __shfl_xor_sync(0xffffffffu, block_max[r], 2));
        const float next_max = fmaxf(rows.max[r], block_max[r]);
        // A row whose window starts past this block has seen no key yet: -inf - -inf is no scale.
        scale[r] = LOCAL && next_max == negative_infinity() ? 1.0f : exp2f(rows.max[r] - next_max);
        rows.max[r] = next_max;
        rows.sum[r] *= scale[r];
    }
#pragma unroll
    for (int dt = 0; dt < D / 8; ++dt) {
        rows.o[dt][0] *= scale[0];
        rows.o[dt][1] *= scale[0];
        rows.o[dt][2] *= scale[1];
        rows.o[dt][3] *= scale[1];
    }

    // 2^P_SHIFT keeps weights down to 2^-22 normal; the row sums carry it
    // and use the rounded weights, so P stays normalized.
    float shifted[2] = {rows.max[0] - P_SHIFT, rows.max[1] - P_SHIFT};
    if constexpr (LOCAL) {
        // Before a row's first key every score is -inf: a finite shift keeps its weights at zero.
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (rows.max[r] == negative_infinity()) shifted[r] = 0.0f;
        }
    }
#pragma unroll
    for (int kt = 0; kt < BLOCK / 16; ++kt) {
        unsigned p[4];
#pragma unroll
        for (int half_tile = 0; half_tile < 2; ++half_tile) {
            const float* t = s[2 * kt + half_tile];
            const __half2 top = __floats2half2_rn(exp2f(t[0] - shifted[0]), exp2f(t[1] - shifted[0]));
            const __half2 bottom =
                __floats2half2_rn(exp2f(t[2] - shifted[1]), exp2f(t[3] - shifted[1]));
            const float2 top_f = __half22float2(top), bottom_f = __half22float2(bottom);
            rows.sum[0] += top_f.x + top_f.y;
            rows.sum[1] += bottom_f.x + bottom_f.y;
            p[half_tile * 2] = *reinterpret_cast<const unsigned*>(&top);
            p[half_tile * 2 + 1] = *reinterpret_cast<const unsigned*>(&bottom);
        }
#pragma unroll
        for (int dp = 0; dp < D / 16; ++dp) {
            const int row = kt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            unsigned r[4];
            ldmatrix_x4_trans(r, shared_address(v + tile_offset<D>(row, dp * 2 + (lane >> 4))));
            const unsigned b0[2] = {r[0], r[1]}, b1[2] = {r[2], r[3]};
            mma_16816(rows.o[2 * dp], p, b0);
            mma_16816(rows.o[2 * dp + 1], p, b1);
        }
    }
}

// Normalizes this warp's rows of the query block at q0 and stores them in FP16.
template <int HEADS, int D>
__device__ __forceinline__ void finish(Rows<D>& rows, __half* __restrict__ out, int start, int q0,
                                       int len, int head) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        rows.sum[r] += __shfl_xor_sync(0xffffffffu, rows.sum[r], 1);
        rows.sum[r] += __shfl_xor_sync(0xffffffffu, rows.sum[r], 2);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = q0 + warp * 16 + (lane >> 2) + r * 8;
        if (row >= len) continue;
        const float inverse = 1.0f / rows.sum[r];
        __half* target =
            out + static_cast<size_t>(start + row) * (HEADS * D) + head * D + (lane & 3) * 2;
#pragma unroll
        for (int dt = 0; dt < D / 8; ++dt) {
            *reinterpret_cast<__half2*>(target + dt * 8) =
                __floats2half2_rn(rows.o[dt][r * 2] * inverse, rows.o[dt][r * 2 + 1] * inverse);
        }
    }
}

// With BIAS, `bias` is MPNet's [head][key - query + longest - 1] relative-position table. With
// LOCAL, a query row sees the keys within `reach` rows of it.
template <int HEADS, int D, bool BIAS, bool CORRECTED, bool LOCAL = false>
__device__ __forceinline__ void attention(const __half* __restrict__ qkv,
                                          const __half* __restrict__ low,
                                          const int* __restrict__ cu_seqlens,
                                          __half* __restrict__ out, float scale_log2e,
                                          const float* __restrict__ bias, int longest,
                                          int corrected_below, __half* shared, int reach = 0) {
    // A window's work does not grow with the sequence, so its kernel corrects every length.
    static_assert(!LOCAL || (CORRECTED && !BIAS), "a window's kernel is corrected and has no bias");
    constexpr int WIDTH = HEADS * D, STRIDE = 3 * WIDTH, LOW_STRIDE = 2 * WIDTH, TILE = BLOCK * D;
    // Tiles: Q, K high (two stages), K low (one stage: only scores read it, so the
    // next block's K low loads while softmax and P * V run), V (two stages).
    __half* q_tile = shared;
    auto k_high = [&](int stage) { return shared + (1 + stage) * TILE; };
    __half* k_low = shared + 3 * TILE;
    auto v_tile = [&](int stage) { return shared + (4 + stage) * TILE; };

    const int sequence = blockIdx.z, head = blockIdx.y;
    const int start = cu_seqlens[sequence], len = cu_seqlens[sequence + 1] - start;
    const int q0 = blockIdx.x * BLOCK;
    // Each sequence takes the kernel of its kind; a batch holding both kinds launches both.
    if (q0 >= len || (!LOCAL && (len < corrected_below) != CORRECTED)) return;
    const __half* base = qkv + static_cast<size_t>(start) * STRIDE + head * D;
    const __half* low_base = low + static_cast<size_t>(start) * LOW_STRIDE + head * D;
    // The keys this block's rows read: the sequence, or with LOCAL the key blocks their windows
    // reach, counted from the first of them.
    int origin = 0, key_len = len;
    if constexpr (LOCAL) {
        origin = max(q0 - reach, 0) / BLOCK * BLOCK;
        key_len = min(len, q0 + BLOCK + reach) - origin;
    }
    const __half* keys = base + WIDTH + static_cast<size_t>(origin) * STRIDE;
    const __half* values = base + 2 * WIDTH + static_cast<size_t>(origin) * STRIDE;
    [[maybe_unused]] const __half* key_lows =
        low_base + WIDTH + static_cast<size_t>(origin) * LOW_STRIDE;
    const int key_blocks = (key_len + BLOCK - 1) / BLOCK;

    load_rows<D, THREADS>(q_tile, base + static_cast<size_t>(q0) * STRIDE, len - q0, STRIDE);
    if constexpr (CORRECTED) {
        // Q's low parts pass through V's second stage, which the first prefetch
        // overwrites only after every warp has read them.
        load_rows<D, THREADS>(v_tile(1), low_base + static_cast<size_t>(q0) * LOW_STRIDE, len - q0,
                              LOW_STRIDE);
    }
    load_rows<D, THREADS>(k_high(0), keys, key_len, STRIDE);
    if constexpr (CORRECTED) load_rows<D, THREADS>(k_low, key_lows, key_len, LOW_STRIDE);
    load_rows<D, THREADS>(v_tile(0), values, key_len, STRIDE);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    unsigned q_high[D / 16][4];
    unsigned q_low[D / 16][4];
    load_query<D>(q_high, q_tile);
    if constexpr (CORRECTED) {
        load_query<D>(q_low, v_tile(1));
        __syncthreads();
    }

    Rows<D> rows;
    rows.start();
    RelativeBias relative{};
    if constexpr (BIAS) {
        const int query = q0 + (threadIdx.x >> 5) * 16 + ((threadIdx.x & 31) >> 2);
        relative = {bias + head * (2 * longest - 1) + longest - 1,
                    {min(query, len - 1), min(query + 8, len - 1)}};
    }
    Window window{};
    if constexpr (LOCAL) {
        const int query = q0 - origin + (threadIdx.x >> 5) * 16 + ((threadIdx.x & 31) >> 2);
        window = {{query, query + 8}, reach};
    }

    for (int kb = 0; kb < key_blocks; ++kb) {
        // Two copy groups per block: K high and V at the top, K low after the
        // scores. Waiting for all but the newest leaves this block's complete.
        const int next = (kb + 1) * BLOCK;
        if (kb + 1 < key_blocks) {
            const int stage = (kb + 1) & 1;
            load_rows<D, THREADS>(k_high(stage), keys + static_cast<size_t>(next) * STRIDE,
                                  key_len - next, STRIDE);
            load_rows<D, THREADS>(v_tile(stage), values + static_cast<size_t>(next) * STRIDE,
                                  key_len - next, STRIDE);
        }
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();

        float s[BLOCK / 8][4];
        unsigned c[BLOCK / 8][2];
        scores<D, CORRECTED>(s, c, q_high, q_low, k_high(kb & 1), k_low);
        if constexpr (CORRECTED) {
            __syncthreads();
            if (kb + 1 < key_blocks) {
                load_rows<D, THREADS>(k_low, key_lows + static_cast<size_t>(next) * LOW_STRIDE,
                                      key_len - next, LOW_STRIDE);
            }
        }
        cp_async_commit();
        softmax_pv<D, BIAS, CORRECTED, LOCAL>(s, c, rows, v_tile(kb & 1), kb, key_len, scale_log2e,
                                              relative, window);
        __syncthreads();
    }
    finish<HEADS, D>(rows, out, start, q0, len, head);
}

// Blocks of WARPS * 16 query rows with Q's fragments from global memory and two tiles each of K
// high, K low and V; block kb + 1's scores issue beside block kb's softmax and P * V, so the
// tensor cores have work while the exponentials run.
template <int HEADS, int D, int WARPS, bool CORRECTED>
__device__ __forceinline__ void pipelined(const __half* __restrict__ qkv,
                                          const __half* __restrict__ low,
                                          const int* __restrict__ cu_seqlens,
                                          __half* __restrict__ out, float scale_log2e,
                                          int corrected_below, __half* shared) {
    constexpr int WIDTH = HEADS * D, STRIDE = 3 * WIDTH, LOW_STRIDE = 2 * WIDTH, TILE = BLOCK * D;
    constexpr int COPIERS = WARPS * 32, ROWS = WARPS * 16;
    auto k_high = [&](int block) { return shared + (block & 1) * TILE; };
    auto k_low = [&](int block) { return shared + (2 + (block & 1)) * TILE; };
    auto v_tile = [&](int block) { return shared + (4 + (block & 1)) * TILE; };

    const int sequence = blockIdx.z, head = blockIdx.y;
    const int start = cu_seqlens[sequence], len = cu_seqlens[sequence + 1] - start;
    const int q0 = blockIdx.x * ROWS;
    if (q0 >= len || (len < corrected_below) != CORRECTED) return;
    const __half* base = qkv + static_cast<size_t>(start) * STRIDE + head * D;
    const __half* keys = base + WIDTH;
    const __half* values = base + 2 * WIDTH;
    const __half* low_base = low + static_cast<size_t>(start) * LOW_STRIDE + head * D;
    [[maybe_unused]] const __half* key_lows = low_base + WIDTH;
    const unsigned key_blocks = (len + BLOCK - 1) / BLOCK;

    const auto load_keys = [&](int block) {
        const int first = block * BLOCK;
        load_rows<D, COPIERS>(k_high(block), keys + static_cast<size_t>(first) * STRIDE,
                              len - first, STRIDE);
        if constexpr (CORRECTED) {
            load_rows<D, COPIERS>(k_low(block), key_lows + static_cast<size_t>(first) * LOW_STRIDE,
                                  len - first, LOW_STRIDE);
        }
    };
    const auto load_values = [&](int block) {
        const int first = block * BLOCK;
        load_rows<D, COPIERS>(v_tile(block), values + static_cast<size_t>(first) * STRIDE,
                              len - first, STRIDE);
    };

    load_keys(0);
    load_values(0);
    cp_async_commit();
    if (key_blocks > 1) load_keys(1);
    cp_async_commit();
    unsigned q_high[D / 16][4], q_low[D / 16][4];
    load_query_global<D>(q_high, base + static_cast<size_t>(q0) * STRIDE, STRIDE, len - q0);
    if constexpr (CORRECTED) {
        load_query_global<D>(q_low, low_base + static_cast<size_t>(q0) * LOW_STRIDE, LOW_STRIDE,
                             len - q0);
    }
    Rows<D> rows;
    rows.start();
    const RelativeBias none{};
    cp_async_wait<1>();
    __syncthreads();
    float s[BLOCK / 8][4];
    unsigned c[BLOCK / 8][2];
    scores<D, CORRECTED>(s, c, q_high, q_low, k_high(0), k_low(0));

    for (unsigned kb = 0; kb + 1 < key_blocks; ++kb) {
        // Block kb + 1's keys and block kb's values have landed, and every warp is past block
        // kb - 1, whose tiles the copies below reuse.
        cp_async_wait<0>();
        __syncthreads();
        if (kb + 2 < key_blocks) load_keys(kb + 2);
        load_values(kb + 1);
        cp_async_commit();
        float s_next[BLOCK / 8][4];
        unsigned c_next[BLOCK / 8][2];
        scores<D, CORRECTED>(s_next, c_next, q_high, q_low, k_high(kb + 1), k_low(kb + 1));
        softmax_pv<D, false, CORRECTED>(s, c, rows, v_tile(kb), kb, len, scale_log2e, none);
#pragma unroll
        for (int nt = 0; nt < BLOCK / 8; ++nt) {
            if constexpr (CORRECTED) {
                c[nt][0] = c_next[nt][0];
                c[nt][1] = c_next[nt][1];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) s[nt][i] = s_next[nt][i];
        }
    }
    cp_async_wait<0>();
    __syncthreads();
    softmax_pv<D, false, CORRECTED>(s, c, rows, v_tile(key_blocks - 1), key_blocks - 1, len,
                                    scale_log2e, none);
    finish<HEADS, D>(rows, out, start, q0, len, head);
}

}  // namespace attention_kernel
