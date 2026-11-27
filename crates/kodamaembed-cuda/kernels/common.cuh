#pragma once

#include <cuda_fp16.h>

__device__ __forceinline__ float negative_infinity() { return __int_as_float(0xff800000); }

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// The exact GELU, x times the normal distribution's share below x.
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

// Four floats rounded to nearest FP16, packed for one 8-byte store.
__device__ __forceinline__ uint2 pack_half4(float4 v) {
    const __half2 lo = __floats2half2_rn(v.x, v.y);
    const __half2 hi = __floats2half2_rn(v.z, v.w);
    uint2 packed;
    packed.x = *reinterpret_cast<const unsigned*>(&lo);
    packed.y = *reinterpret_cast<const unsigned*>(&hi);
    return packed;
}

// A split value is high + low * 2^-11: FP16 and what that rounding lost, scaled into FP16's
// normal range, about 22 significant bits together.
constexpr float LOW_SCALE = 2048.0f;

__device__ __forceinline__ unsigned shared_address(const void* pointer) {
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

// Copies 16 bytes global -> shared without registers; bytes beyond `valid`
// (0 or 16) are zero-filled.
__device__ __forceinline__ void cp_async_16(unsigned target, const void* source,
                                            unsigned valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(target),
                 "l"(source), "r"(valid));
}

// Weights are read once an encode, so evicting them first keeps activations
// and small tables in L2.
__device__ __forceinline__ unsigned long long l2_evict_first() {
    unsigned long long policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}

// cp_async_16 under an L2 cache policy.
__device__ __forceinline__ void cp_async_16_policy(unsigned target, const void* source,
                                                   unsigned long long policy) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" ::"r"(target),
                 "l"(source), "l"(policy));
}
__device__ __forceinline__ void cp_async_16_policy(unsigned target, const void* source,
                                                   unsigned valid, unsigned long long policy) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n" ::"r"(target),
                 "l"(source), "r"(valid), "l"(policy));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int PENDING>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(PENDING));
}

// Lane i addresses row i % 8 of matrix i / 8.
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], unsigned address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
}

// Lanes 0-15 address row i % 8 of matrix i / 8.
__device__ __forceinline__ void ldmatrix_x2(unsigned (&r)[2], unsigned address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(address));
}

// As ldmatrix_x4, with each 8x8 matrix transposed on the way into registers.
__device__ __forceinline__ void ldmatrix_x4_trans(unsigned (&r)[4], unsigned address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(address));
}

// D += A * B for a 16x16 FP16 A fragment and a 16x8 FP16 B fragment, FP32 accumulate.
__device__ __forceinline__ void mma_16816(float (&d)[4], const unsigned (&a)[4],
                                          const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// As mma_16816 with FP16 accumulators in the same element layout.
__device__ __forceinline__ void mma_16816_f16(unsigned (&d)[2], const unsigned (&a)[4],
                                              const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, "
        "{%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
