// EPILOGUE 0: out16 = acc + bias, out32 keeping what FP16 lost in the first
// LOW_COLUMNS columns; 1: out16 = gelu(acc + bias); 2: out32 = acc + bias + residual.
// Built with PAIRED, columns 2j and 2j + 1 are a pair, and EPILOGUE 3: out16 = gelu(input) * gate
// from an input and its gate, one column a pair; 4: as 0 after each pair of the first
// LOW_COLUMNS columns turns by its row's position.

layout(std430, binding = 2) readonly buffer Bias { float bias[]; };
layout(std430, binding = 3) readonly buffer Residual { float residual[]; };
layout(std430, binding = 4) writeonly buffer Out16 { float16_t out16[]; };
layout(std430, binding = 5) writeonly buffer Out32 { float out32[]; };

#include "gelu.glsl"
#include "half.glsl"

void write_output(uint row, uint col, uint N, float acc) {
    const float v = acc + bias[col];
    const uint at = row * N + col;
    if (EPILOGUE == 2) {
        out32[at] = v + residual[at];
    } else if (EPILOGUE == 1) {
        out16[at] = float16_t(gelu(v));
    } else {
        // Integer rounding (half.glsl) makes the FP16 store exact and the low part true.
        const float high = round_to_half(v);
        out16[at] = float16_t(high);
        if (col < LOW_COLUMNS) out32[row * LOW_COLUMNS + col] = v - high;
    }
}

#ifdef PAIRED
// The columns of a head, a power of two.
layout(constant_id = 3) const uint HEAD = 64;
layout(std430, binding = 6) readonly buffer Positions { int positions[]; };
// [position][pair of a head]: the pair's cosine and sine.
layout(std430, binding = 7) readonly buffer Turns { vec2 turns[]; };

// write_output for `acc` at `col`, with the sum at the other column of its pair.
void write_pair(uint row, uint col, uint N, float acc, float partner) {
    const float v = acc + bias[col], other = partner + bias[col ^ 1u];
    const bool first = (col & 1u) == 0u;
    if (EPILOGUE == 3) {
        if (first) out16[row * (N / 2) + col / 2] = float16_t(gelu(v) * other);
    } else if (col < LOW_COLUMNS) {
        // (x, y) turns to (x cos - y sin, y cos + x sin).
        const vec2 turn = turns[uint(positions[row]) * (HEAD / 2) + (col % HEAD) / 2];
        const float turned = fma(v, turn.x, first ? -(other * turn.y) : other * turn.y);
        const float high = round_to_half(turned);
        out16[row * N + col] = float16_t(high);
        out32[row * LOW_COLUMNS + col] = turned - high;
    } else {
        write_output(row, col, N, acc);
    }
}

#define WRITE(row, col, N, acc, partner) write_pair(row, col, N, acc, partner)
#else
#define WRITE(row, col, N, acc, partner) write_output(row, col, N, acc)
#endif
