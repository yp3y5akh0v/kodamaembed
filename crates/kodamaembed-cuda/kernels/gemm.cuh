// Every tiling sums each output in FP32 over K in the same order, so all
// tilings give identical bits.
#pragma once

#include <cuda_fp16.h>
#include <type_traits>

#include "common.cuh"

namespace gemm {

// Rows of BK halfs are BK / 8 chunks of 16 bytes, XOR-swizzled so the eight
// rows read by one ldmatrix phase land in eight distinct bank groups.
template <int BK>
__device__ __forceinline__ int tile_offset(int row, int chunk) {
    static_assert(BK == 32 || BK == 64, "four or eight chunks per row");
    if constexpr (BK == 32) {
        return row * BK + ((chunk ^ ((row >> 1) & 3)) << 3);
    } else {
        return row * BK + ((chunk ^ (row & 7)) << 3);
    }
}

// Widths that are not powers of two (88, 176) give the multiprocessors nearly equal work; N need
// not be a multiple of them, so the last column tile reads zeros past N and stores nothing there.
template <int BM_, int BN_, int BK_, int WM_, int WN_, int STAGES_>
struct Tiling {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, WM = WM_, WN = WN_, STAGES = STAGES_;
    static constexpr int CHUNKS = BK / 8;
    static constexpr int WARPS_N = BN / WN;
    static constexpr int THREADS = (BM / WM) * WARPS_N * 32;
    static constexpr int MT = WM / 16;
    static constexpr int NT = WN / 8;
    static constexpr int A_STAGE = BM * BK;
    static constexpr int B_STAGE = BN * BK;
    static constexpr int A_LOADS = BM * CHUNKS, W_LOADS = BN * CHUNKS;
    static constexpr int SHARED_BYTES = STAGES * (A_STAGE + B_STAGE) * 2;
    static constexpr bool PADDED = (BN & (BN - 1)) != 0;
    // Loading a lane's residual vectors before its stores paid on these tilings (out and down at
    // 193-384 rows); on the 512-row tiling the registers it took cost more than it saved.
    static constexpr bool GROUPED_RESIDUAL =
        (BM == 64 && BN == 128 && BK == 64) || (BM == 112 && BN == 64) || (BM == 96 && BN <= 80) ||
        (BM == 80 && BN == 96);
    static_assert(BM % WM == 0 && BN % WN == 0 && WM % 16 == 0 && WN % 8 == 0, "whole warp tiles");
};

// The r<rows>c<columns> tilings, whose blocks walk the output's tiles: every epilogue has a
// kernel in each of them.

// A few rows: narrow column tiles spread the weight stream over every multiprocessor.
using r16c32 = Tiling<16, 32, 64, 16, 16, 6>;
using r16c64 = Tiling<16, 64, 32, 16, 32, 6>;
using r32c32 = Tiling<32, 32, 32, 16, 16, 6>;
// One row band for 33-48 and 65-96 rows: padded rows cost one band's spare rows, not a band.
using r48c32 = Tiling<48, 32, 64, 48, 16, 4>;
using r48c64 = Tiling<48, 64, 64, 48, 16, 4>;
using r48c128 = Tiling<48, 128, 32, 48, 32, 4>;
using r80c64 = Tiling<80, 64, 64, 80, 16, 5>;
using r80c128 = Tiling<80, 128, 32, 80, 32, 7>;
using r96c128 = Tiling<96, 128, 32, 96, 32, 4>;
using r64c64s3 = Tiling<64, 64, 64, 32, 32, 3>;
using r64c64s4 = Tiling<64, 64, 64, 32, 32, 4>;
using r64c128 = Tiling<64, 128, 32, 32, 64, 4>;
using r128c64 = Tiling<128, 64, 32, 64, 32, 4>;
using r128c128 = Tiling<128, 128, 32, 64, 64, 3>;
using r128c128w8 = Tiling<128, 128, 32, 64, 32, 4>;
using r256c128 = Tiling<256, 128, 32, 64, 64, 4>;
// r64c128 and r128c128w8 with 64-wide K steps, for the row bands where they ran faster.
using r64c128k64 = Tiling<64, 128, 64, 32, 64, 3>;
using r128c128w8k64 = Tiling<128, 128, 64, 64, 32, 3>;
// 1024 columns in 12 tiles of 88 and 4096 in 24 of 176 give whole waves of 48 blocks: out and
// down from 385 rows, up past 512.
using r128c88 = Tiling<128, 88, 64, 32, 88, 3>;
using r128c176 = Tiling<128, 176, 32, 32, 88, 3>;

// BIAS_SPLIT_F16 also keeps what FP16 lost, as common.cuh's split values. BIAS_GEGLU_F16 takes
// each pair of columns as a gated feed-forward's input and its gate, and writes
// gelu(input) * gate, one column a pair. BIAS_TURNED_SPLIT_F16 splits after turning each pair of
// its first columns by the row's position.
enum Epilogue {
    BIAS_GELU_F16,
    BIAS_RESIDUAL_F32,
    BIAS_SPLIT_F16,
    BIAS_GEGLU_F16,
    BIAS_TURNED_SPLIT_F16
};

// Rotary positions for BIAS_TURNED_SPLIT_F16: row r of the output sits at `positions[r]`, and
// `table` holds each position's (cosine, sine) for every pair of a head, `head` columns wide, a
// power of two. The first `columns` columns turn; a pair holds a head's coordinates m and
// m + head / 2.
struct Turns {
    const int* positions;
    const float2* table;
    int head;
    int columns;
};

// The epilogue's arithmetic element by element, staged through each warp's own shared-memory region
// so global stores are 16-byte chunks of whole rows, 8-byte ones for the gated products.
template <class T, int EPILOGUE>
__device__ __forceinline__ void epilogue(float (&acc)[T::MT][T::NT][4],
                                         const float* __restrict__ bias,
                                         const float* __restrict__ residual, void* __restrict__ out,
                                         __half* __restrict__ low, int low_columns, int M, int N,
                                         int m0, int n0, __half* shared, const Turns& turns) {
    constexpr int WM = T::WM, WN = T::WN, NT = T::NT;
    // FP16 rows: the eight rows one store instruction touches start on distinct four-bank groups.
    constexpr int LD = WN % 16 == 0 ? WN + 8 : WN;
    // FP32 leaves in passes of NT / 2 and NT - NT / 2 column groups, rows starting four (even NT)
    // or eight banks apart.
    constexpr int HMAX = (NT - NT / 2) * 8;
    constexpr int LDF = NT % 2 == 0 && WN % 16 == 0 ? WN / 2 + 4 : HMAX + ((8 - HMAX) % 16 + 16) % 16;
    constexpr int REGION = WM * (LD * 2 > LDF * 4 ? LD * 2 : LDF * 4);
    static_assert(REGION * (T::THREADS / 32) <= T::SHARED_BYTES, "staging fits the stages' memory");
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warp_m = warp / T::WARPS_N, warp_n = warp % T::WARPS_N;
    const int row0 = m0 + warp_m * WM, col0 = n0 + warp_n * WN;
    unsigned char* region = reinterpret_cast<unsigned char*>(shared) + warp * REGION;
    // Every warp is past the main loop's last shared-memory reads before any region is written.
    __syncthreads();

    float2 b[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int col = col0 + nt * 8;
        // Groups of eight columns lie wholly below or past N, a multiple of eight.
        b[nt] = !T::PADDED || col < N
            ? *reinterpret_cast<const float2*>(bias + col + (lane & 3) * 2)
            : make_float2(0.0f, 0.0f);
    }

    if constexpr (EPILOGUE == BIAS_TURNED_SPLIT_F16) {
        // The biased values turn in place, once for the high and the low parts.
        const int pairs = turns.head / 2;
#pragma unroll
        for (int mt = 0; mt < T::MT; ++mt)
#pragma unroll
            for (int half_row = 0; half_row < 2; ++half_row) {
                const int row = row0 + mt * 16 + (lane >> 2) + half_row * 8;
                // Rows past M hold nothing, so any position's turn serves them.
                const float2* turn =
                    turns.table + static_cast<size_t>(row < M ? turns.positions[row] : 0) * pairs;
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const int col = col0 + nt * 8 + (lane & 3) * 2;
                    float& x = acc[mt][nt][half_row * 2];
                    float& y = acc[mt][nt][half_row * 2 + 1];
                    x += b[nt].x;
                    y += b[nt].y;
                    // Groups of eight columns lie wholly below or past the turned ones.
                    if (col < turns.columns) {
                        const float2 at = turn[(col & (turns.head - 1)) >> 1];
                        const float turned = fmaf(x, at.x, -(y * at.y));
                        y = fmaf(y, at.x, x * at.y);
                        x = turned;
                    }
                }
            }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) b[nt] = make_float2(0.0f, 0.0f);
    }

    if constexpr (EPILOGUE == BIAS_RESIDUAL_F32) {
        float* staged = reinterpret_cast<float*>(region);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int first = h == 0 ? 0 : NT / 2, last = h == 0 ? NT / 2 : NT;
            const int offset = first * 8, chunks = (last - first) * 2;
#pragma unroll
            for (int mt = 0; mt < T::MT; ++mt)
#pragma unroll
                for (int half_row = 0; half_row < 2; ++half_row) {
                    const int r = mt * 16 + (lane >> 2) + half_row * 8;
#pragma unroll
                    for (int nt = first; nt < last; ++nt) {
                        const int c = nt * 8 + (lane & 3) * 2 - offset;
                        *reinterpret_cast<float2*>(staged + r * LDF + c) =
                            make_float2(acc[mt][nt][half_row * 2] + b[nt].x,
                                        acc[mt][nt][half_row * 2 + 1] + b[nt].y);
                    }
                }
            if constexpr (T::GROUPED_RESIDUAL) {
                constexpr int STORES = (WM * (NT - NT / 2) * 2 + 31) / 32;
                float4 additions[STORES];
#pragma unroll
                for (int i = 0; i < STORES; ++i) {
                    const int q = lane + i * 32;
                    const int r = q / chunks, c = (q % chunks) * 4, row = row0 + r;
                    const int col = col0 + offset + c;
                    additions[i] = q < WM * chunks && row < M && (!T::PADDED || col < N)
                        ? *reinterpret_cast<const float4*>(residual + static_cast<size_t>(row) * N + col)
                        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
                // Keep independent residual loads together so their latency overlaps.
                asm volatile("" ::: "memory");
                __syncwarp();
#pragma unroll
                for (int i = 0; i < STORES; ++i) {
                    const int q = lane + i * 32;
                    const int r = q / chunks, c = (q % chunks) * 4, row = row0 + r;
                    const int col = col0 + offset + c;
                    if (q < WM * chunks && row < M && (!T::PADDED || col < N)) {
                        const size_t at = static_cast<size_t>(row) * N + col;
                        const float4 v = *reinterpret_cast<const float4*>(staged + r * LDF + c);
                        const float4 add = additions[i];
                        *reinterpret_cast<float4*>(static_cast<float*>(out) + at) =
                            make_float4(v.x + add.x, v.y + add.y, v.z + add.z, v.w + add.w);
                    }
                }
            } else {
                __syncwarp();
                for (int q = lane; q < WM * chunks; q += 32) {
                    const int r = q / chunks, c = (q % chunks) * 4, row = row0 + r;
                    const int col = col0 + offset + c;
                    if (row < M && (!T::PADDED || col < N)) {
                        const size_t at = static_cast<size_t>(row) * N + col;
                        const float4 v = *reinterpret_cast<const float4*>(staged + r * LDF + c);
                        const float4 add = *reinterpret_cast<const float4*>(residual + at);
                        *reinterpret_cast<float4*>(static_cast<float*>(out) + at) =
                            make_float4(v.x + add.x, v.y + add.y, v.z + add.z, v.w + add.w);
                    }
                }
            }
            __syncwarp();
        }
    } else {
        constexpr int CHUNKS = WN / 8;
        __half* staged = reinterpret_cast<__half*>(region);
#pragma unroll
        for (int mt = 0; mt < T::MT; ++mt)
#pragma unroll
            for (int half_row = 0; half_row < 2; ++half_row) {
                const int r = mt * 16 + (lane >> 2) + half_row * 8;
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    float v0 = acc[mt][nt][half_row * 2] + b[nt].x;
                    float v1 = acc[mt][nt][half_row * 2 + 1] + b[nt].y;
                    if constexpr (EPILOGUE == BIAS_GELU_F16) {
                        v0 = gelu(v0);
                        v1 = gelu(v1);
                    }
                    // The product takes its pair's first place; the store leaves the second out.
                    if constexpr (EPILOGUE == BIAS_GEGLU_F16) v0 = gelu(v0) * v1;
                    *reinterpret_cast<__half2*>(staged + r * LD + nt * 8 + (lane & 3) * 2) =
                        __floats2half2_rn(v0, v1);
                }
            }
        __syncwarp();
        for (int q = lane; q < WM * CHUNKS; q += 32) {
            const int r = q / CHUNKS, c = (q % CHUNKS) * 8, row = row0 + r;
            if (row < M && (!T::PADDED || col0 + c < N)) {
                if constexpr (EPILOGUE == BIAS_GEGLU_F16) {
                    // Four pairs' products, the low half of each 32-bit word, in one 8-byte store.
                    const uint4 pairs = *reinterpret_cast<const uint4*>(staged + r * LD + c);
                    *reinterpret_cast<uint2*>(static_cast<__half*>(out) +
                                              static_cast<size_t>(row) * (N / 2) + (col0 + c) / 2) =
                        make_uint2((pairs.x & 0xffffu) | (pairs.y << 16),
                                   (pairs.z & 0xffffu) | (pairs.w << 16));
                } else {
                    *reinterpret_cast<uint4*>(static_cast<__half*>(out) + static_cast<size_t>(row) * N + col0 + c) =
                        *reinterpret_cast<const uint4*>(staged + r * LD + c);
                }
            }
        }
        if constexpr (EPILOGUE == BIAS_SPLIT_F16 || EPILOGUE == BIAS_TURNED_SPLIT_F16) {
            if (col0 < low_columns) {
                __syncwarp();
#pragma unroll
                for (int mt = 0; mt < T::MT; ++mt)
#pragma unroll
                    for (int half_row = 0; half_row < 2; ++half_row) {
                        const int r = mt * 16 + (lane >> 2) + half_row * 8;
#pragma unroll
                        for (int nt = 0; nt < NT; ++nt) {
                            const float v0 = acc[mt][nt][half_row * 2] + b[nt].x;
                            const float v1 = acc[mt][nt][half_row * 2 + 1] + b[nt].y;
                            const float2 kept = __half22float2(__floats2half2_rn(v0, v1));
                            *reinterpret_cast<__half2*>(staged + r * LD + nt * 8 + (lane & 3) * 2) =
                                __floats2half2_rn((v0 - kept.x) * LOW_SCALE, (v1 - kept.y) * LOW_SCALE);
                        }
                    }
                __syncwarp();
                for (int q = lane; q < WM * CHUNKS; q += 32) {
                    const int r = q / CHUNKS, c = (q % CHUNKS) * 8, row = row0 + r;
                    // The low output can end inside a warp even when N contains whole tiles.
                    if (row < M && col0 + c < low_columns) {
                        *reinterpret_cast<uint4*>(low + static_cast<size_t>(row) * low_columns + col0 + c) =
                            *reinterpret_cast<const uint4*>(staged + r * LD + c);
                    }
                }
            }
        }
    }
}

// One output tile at rows m0.., columns n0..: each k16 step loads the next fragments, issues its
// MMAs, then the next stage's copies; the stage barrier follows the MMAs of the step before the
// last, so the tensor cores work through each wait.
template <class T, int EPILOGUE>
__device__ __forceinline__ void tile(const __half* __restrict__ a_global,
                                     const __half* __restrict__ w_global,
                                     const float* __restrict__ bias,
                                     const float* __restrict__ residual, void* __restrict__ out,
                                     __half* __restrict__ low, int low_columns, int M, int N, int K,
                                     int m0, int n0, __half* shared, const Turns& turns) {
    constexpr int BK = T::BK, KK = BK / 16;
    static_assert(KK >= 2 && KK % 2 == 0, "fragment buffers alternate by k16 step");
    __half* a_shared = shared;
    __half* w_shared = shared + T::STAGES * T::A_STAGE;
    const int thread = threadIdx.x, lane = thread & 31, warp = thread >> 5;
    const int warp_m = warp / T::WARPS_N, warp_n = warp % T::WARPS_N;
    const int k_tiles = K / BK;

    // This thread copies rows row0 + j * APART of each stage, all at chunk thread % CHUNKS. APART
    // is a multiple of eight, so the swizzle repeats and copy j lies j * APART rows past copy 0 in
    // both memories: a stage adds only its k0 to offsets fixed for the tile. Rows past M (or N)
    // zero-fill from the tile's first row, which exists.
    constexpr int APART = T::THREADS / T::CHUNKS;
    static_assert(T::THREADS % T::CHUNKS == 0 && APART % 8 == 0, "the swizzle repeats every 8 rows");
    const int row0 = thread / T::CHUNKS, first = tile_offset<BK>(row0, thread % T::CHUNKS);
    const int from = row0 * K + (thread % T::CHUNKS) * 8, step = APART * K;
    const __half* a_tile = a_global + static_cast<size_t>(m0) * K;
    const __half* w_tile = w_global + static_cast<size_t>(n0) * K;
    const int a_rows = M - m0 - row0, w_rows = N - n0 - row0;

    auto load_stage = [&](int stage, int k_tile) {
        const int k0 = k_tile * BK;
        __half* a = a_shared + stage * T::A_STAGE + first;
        __half* w = w_shared + stage * T::B_STAGE + first;
#pragma unroll
        for (int j = 0; j < (T::A_LOADS + T::THREADS - 1) / T::THREADS; ++j) {
            if (T::A_LOADS % T::THREADS == 0 || thread + j * T::THREADS < T::A_LOADS) {
                const bool inside = j * APART < a_rows;
                cp_async_16(shared_address(a + j * APART * BK),
                            a_tile + (inside ? from + j * step + k0 : k0), inside ? 16 : 0);
            }
        }
#pragma unroll
        for (int j = 0; j < (T::W_LOADS + T::THREADS - 1) / T::THREADS; ++j) {
            if (T::W_LOADS % T::THREADS == 0 || thread + j * T::THREADS < T::W_LOADS) {
                const bool inside = !T::PADDED || j * APART < w_rows;
                cp_async_16(shared_address(w + j * APART * BK),
                            w_tile + (inside ? from + j * step + k0 : k0), inside ? 16 : 0);
            }
        }
    };

    auto load_fragments = [&](unsigned (&a_frag)[T::MT][4], unsigned (&w_frag)[T::NT][2],
                              int stage, int kk) {
        const __half* a = a_shared + stage * T::A_STAGE;
        const __half* w = w_shared + stage * T::B_STAGE;
#pragma unroll
        for (int mt = 0; mt < T::MT; ++mt) {
            const int row = warp_m * T::WM + mt * 16 + (lane & 15);
            ldmatrix_x4(a_frag[mt], shared_address(a + tile_offset<BK>(row, kk * 2 + (lane >> 4))));
        }
#pragma unroll
        for (int np = 0; np < T::NT / 2; ++np) {
            const int row = warp_n * T::WN + np * 16 + (lane & 7) + ((lane >> 4) << 3);
            unsigned r[4];
            ldmatrix_x4(r, shared_address(w + tile_offset<BK>(row, kk * 2 + ((lane >> 3) & 1))));
            w_frag[2 * np][0] = r[0];
            w_frag[2 * np][1] = r[1];
            w_frag[2 * np + 1][0] = r[2];
            w_frag[2 * np + 1][1] = r[3];
        }
        if constexpr (T::NT % 2 == 1) {
            const int row = warp_n * T::WN + (T::NT - 1) * 8 + (lane & 7);
            unsigned r[2];
            ldmatrix_x2(r, shared_address(w + tile_offset<BK>(row, kk * 2 + ((lane >> 3) & 1))));
            w_frag[T::NT - 1][0] = r[0];
            w_frag[T::NT - 1][1] = r[1];
        }
    };

    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int mt = 0; mt < T::MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < T::NT; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.0f;
        }

#pragma unroll
    for (int stage = 0; stage < T::STAGES - 1; ++stage) {
        if (stage < k_tiles) load_stage(stage, stage);
        cp_async_commit();
    }
    cp_async_wait<T::STAGES - 2>();
    __syncthreads();

    unsigned a_frag[2][T::MT][4];
    unsigned w_frag[2][T::NT][2];
    int read_stage = 0, write_stage = T::STAGES - 1, next_tile = T::STAGES - 1;
    load_fragments(a_frag[0], w_frag[0], read_stage, 0);

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
#pragma unroll
        for (int kk = 0; kk < KK; ++kk) {
            if (kk + 1 < KK || k_tile + 1 < k_tiles) {
                load_fragments(a_frag[(kk + 1) & 1], w_frag[(kk + 1) & 1], read_stage,
                               (kk + 1) % KK);
            }
#pragma unroll
            for (int mt = 0; mt < T::MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < T::NT; ++nt)
                    mma_16816(acc[mt][nt], a_frag[kk & 1][mt], w_frag[kk & 1][nt]);
            // The refilled stage's last reads came before the previous k-tile's barrier.
            if (kk == 0 && next_tile < k_tiles) load_stage(write_stage, next_tile);
            if (kk == KK - 2) {
                cp_async_commit();
                cp_async_wait<T::STAGES - 2>();
                __syncthreads();
                read_stage = read_stage + 1 == T::STAGES ? 0 : read_stage + 1;
                write_stage = write_stage + 1 == T::STAGES ? 0 : write_stage + 1;
                ++next_tile;
            }
        }
    }
    cp_async_wait<0>();
    epilogue<T, EPILOGUE>(acc, bias, residual, out, low, low_columns, M, N, m0, n0, shared, turns);
}

// Grid: up to ceil(M / BM) * ceil(N / BN) blocks; block b takes tiles b, b + grid, ... in
// column-major order, so the blocks running at once share weight columns in L2.
template <class T, int EPILOGUE>
__device__ __forceinline__ void run(const __half* __restrict__ a_global,
                                    const __half* __restrict__ w_global,
                                    const float* __restrict__ bias,
                                    const float* __restrict__ residual, void* __restrict__ out,
                                    __half* __restrict__ low, int low_columns, int M, int N,
                                    int K, __half* shared, const Turns& turns = {}) {
    // Unsigned division lets the largest GELU and residual tiles keep their grid state in registers.
    constexpr bool SPLITS = EPILOGUE == BIAS_SPLIT_F16 || EPILOGUE == BIAS_TURNED_SPLIT_F16;
    using Index = std::conditional_t<T::BM == 256 && !SPLITS, unsigned, int>;
    const Index m_tiles = (static_cast<Index>(M) + T::BM - 1) / T::BM;
    const Index n_tiles = (static_cast<Index>(N) + T::BN - 1) / T::BN;
    const Index tiles = m_tiles * n_tiles;
    for (Index t = blockIdx.x; t < tiles; t += gridDim.x) {
        tile<T, EPILOGUE>(a_global, w_global, bias, residual, out, low, low_columns, M, N, K,
                          (t % m_tiles) * T::BM, (t / m_tiles) * T::BN, shared, turns);
        // The staged epilogue's shared-memory reads end before the next tile's copies land.
        __syncthreads();
    }
}

// Grid: ceil(M / BM) row tiles along x by ceil(N / BN) column tiles along y, one block a tile.
template <class T, int EPILOGUE>
__device__ __forceinline__ void run_grid(const __half* __restrict__ a_global,
                                         const __half* __restrict__ w_global,
                                         const float* __restrict__ bias,
                                         const float* __restrict__ residual, void* __restrict__ out,
                                         __half* __restrict__ low, int low_columns, int M, int N,
                                         int K, __half* shared) {
    tile<T, EPILOGUE>(a_global, w_global, bias, residual, out, low, low_columns, M, N, K,
                      blockIdx.x * T::BM, blockIdx.y * T::BN, shared, {});
}

}  // namespace gemm
