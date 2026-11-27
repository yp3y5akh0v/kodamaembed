// Integer arithmetic keeps FP16 rounding (ties to even, normal range) when a
// driver eliminates float -> float16_t -> float casts.
float round_to_half(float v) {
    const uint bits = floatBitsToUint(v);
    return uintBitsToFloat((bits + 0x0FFFu + ((bits >> 13) & 1u)) & 0xFFFFE000u);
}
