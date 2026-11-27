// INT8 tensor-core products with exact INT32 sums, so every tiling of a
// product gives identical bits.
#pragma once

#include "common.cuh"

namespace int8proj {

// Rows of 64 bytes are four 16-byte chunks, XOR-swizzled so the eight rows
// read by one ldmatrix phase land in distinct bank groups.
__device__ __forceinline__ int tile_offset(int row, int chunk) {
    return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4);
}

__device__ __forceinline__ void mma_s8(int (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// d = a b + (c, c, c, c): sums started at a constant, with no reset.
__device__ __forceinline__ void mma_s8_from(int (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2], int c) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c));
}

// The bits of 1.5 2^23, where an FP32 value is its own integer: sums started there
// (mma_s8_from) need no conversion.
constexpr int SUMS_FROM = 0x4B400000;

// FACTOR times a sum started at SUMS_FROM, exactly and in one FMA.
template <int FACTOR>
__device__ __forceinline__ float promoted(int sum) {
    static_assert(FACTOR == 1 || FACTOR == 14, "FACTOR 1.5 2^23 exact and FACTOR times a sum below 2^24");
    return __fmaf_rn(__int_as_float(sum), static_cast<float>(FACTOR), -static_cast<float>(FACTOR) * 12582912.0f);
}

// BM x BN output tiles of WM x WN warp tiles, K in tiles of 64 (two k32
// steps), STAGES operand stages.
template <int BM_, int BN_, int WM_, int WN_, int STAGES_>
struct Tiling {
    static constexpr int BM = BM_, BN = BN_, BK = 64, WM = WM_, WN = WN_, STAGES = STAGES_;
    static constexpr int WARPS_N = BN / WN;
    static constexpr int THREADS = (BM / WM) * WARPS_N * 32;
    static constexpr int CHUNKS = BK / 16, MT = WM / 16, NT = WN / 8;
    static constexpr int A_STAGE = BM * BK, B_STAGE = BN * BK;
    static constexpr int SHARED_BYTES = STAGES * (A_STAGE + B_STAGE);
    static_assert(NT % 2 == 0, "B fragments load in pairs");
    // Narrow tiles have fewer copy chunks than threads; the first ones copy.
    static_assert(((BM * CHUNKS) % THREADS == 0 || THREADS % (BM * CHUNKS) == 0) &&
                      ((BN * CHUNKS) % THREADS == 0 || BN * CHUNKS < THREADS),
                  "even loads");
};

// A mainloop's hook after each k tile's products: none.
struct NoAfterTile {
    template <class Acc>
    __device__ __forceinline__ void operator()(int, Acc&) const {}
};

}  // namespace int8proj
