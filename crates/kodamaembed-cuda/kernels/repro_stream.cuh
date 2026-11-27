// One resident block a multiprocessor streams weights while workers prepare row-local
// values; each output keeps the CPU's operation order.

#define STREAM_WARPS 11
#define STREAM_THREADS ((STREAM_WARPS + 1) * 32)
#define STREAM_WORKERS 8
// The largest chunk: 8 columns of 512 inputs.
#define STREAM_CHUNK 4096
// The most out projection units a streamer owns, their residual stashed.
#define STREAM_STASH_UNITS 4
// The chunks a producer keeps in flight: fewer starve the memory, more delay the chunk
// a consumer waits for.
#define STREAM_AHEAD 4
// A row tile's rows: the products' MMA rows.
#define STREAM_TILE_ROWS 16
#define STREAM_LN_ROWS 2
// Statistics warps follow the item warps: h / 128 items, two a warp, h <= 1024.
#define STREAM_STATS_WARP 8
#define STREAM_MAX_LAYERS 32

// A layer's hand-offs in phase order; an even count, as repro_clear clears pairs.
enum StreamHandoff {
    STREAM_QKV_IN, STREAM_QKV, STREAM_ATT, STREAM_OUT, STREAM_LN1, STREAM_UP, STREAM_BLOCKS, STREAM_PARTS,
    STREAM_DOWN, STREAM_HANDOFFS = STREAM_DOWN + 2
};

// repro.rs StreamLayer mirrors this; undeferred projections' terms are null.
struct StreamLayer {
    const signed char* q[4];
    const float* scales[4];
    const float* statics[4];
    const float* centers[4];
    const float* signs[4];
    const float* sums[4];
    const float* constants[4];
    const float* gamma1;
    const float* beta1;
    const float* gamma2;
    const float* beta2;
    const float* key_centers;
    float eps1, eps2;
};
static_assert(sizeof(StreamLayer) == 272, "repro.rs StreamLayer");

// A ring slot's `full` completes when the producer's 32 cp.async groups of its chunk land,
// `empty` when every lane of the consumer warp has used it.
__device__ __forceinline__ void stream_bar_init(unsigned long long* bar, unsigned count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(shared_address(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void stream_bar_arrive(unsigned long long* bar) {
    asm volatile("{\n .reg .b64 state;\n mbarrier.arrive.shared.b64 state, [%0];\n}" ::"r"(
                     shared_address(bar))
                 : "memory");
}
__device__ __forceinline__ bool stream_bar_test(unsigned long long* bar, unsigned parity) {
    unsigned done;
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(shared_address(bar)), "r"(parity)
        : "memory");
    return done != 0;
}
// The lane's copies so far, tracked by `bar` (no pending count added).
__device__ __forceinline__ void stream_cp_arrive(unsigned long long* bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(shared_address(bar)) : "memory");
}
// Out of line, so the parity is defined before the barrier state loads; an inlined copy
// can load the state first, which compute-sanitizer's racecheck reports.
__device__ __noinline__ void stream_bar_wait(unsigned long long* bar, unsigned parity) {
    while (!stream_bar_test(bar, parity)) __nanosleep(64);
    __syncwarp();
}

// A streamer's consumers meet on named barrier 1 while the producer streams on.
using StreamConsumers = ReproNamedSync<1, STREAM_WARPS * 32>;
using StreamBlock = ReproBlockSync;

// The floor split of n / 8 units over the streamers.
struct StreamUnits {
    int first, count;
};
__device__ __forceinline__ StreamUnits stream_units(int n, int streamer, int streamers) {
    const int units = n / 8;
    const int first = units * streamer / streamers;
    return {first, units * (streamer + 1) / streamers - first};
}

// sm_86's shared memory a block opts into (99 KiB): a streamer's ring takes as
// many chunks as the rest of its layout leaves room for.
constexpr int STREAM_SHARED_LIMIT = 99 * 1024;
// The out projection's stashed columns a row.
constexpr int STREAM_STASH = STREAM_STASH_UNITS * 8;

// repro.rs STREAM_BYTES mirrors this layout. Every layer's addresses sit in
// shared memory, since a global read queued behind the ring's copies waits for them.
struct StreamShared {
    unsigned char* ring;
    signed char* a;
    float* steps;
    __half* stash_out;
    int* codes;
    const signed char** weights;
    const float** scales;
    const float** statics;
    const float** sums;
    const float** constants;
    int* issued;
    unsigned long long* done;
    unsigned long long* full;
    unsigned long long* empty;
};

struct StreamPlan {
    static constexpr int ROWS = STREAM_TILE_ROWS;
    static constexpr int A_BYTES = ROWS * 1024;
    static constexpr int STEPS = ROWS * (4096 / HALF);
    static constexpr int STASH_BYTES = ROWS * STREAM_STASH * 2;
    // The layers' address tables and the issue count.
    static constexpr int ADDRESSES = 3 * STREAM_MAX_LAYERS * 4 * 8 + 2 * STREAM_MAX_LAYERS * 2 * 8 + 16;
    static constexpr int FIXED = A_BYTES + STEPS * 4 + STASH_BYTES + ROWS * 4 + ADDRESSES;
    // A slot: its chunk and its two barriers.
    static constexpr int SLOT_BYTES = STREAM_CHUNK + 2 * 8;
    static constexpr int SLOTS = (STREAM_SHARED_LIMIT - FIXED) / SLOT_BYTES;
    static constexpr int BYTES = SLOTS * SLOT_BYTES + FIXED;
    // The rows' codes' offset: a worker keeps them beside its stages and
    // attention, which end before them.
    static constexpr int CODES_AT = SLOTS * STREAM_CHUNK + A_BYTES + STEPS * 4 + STASH_BYTES;
    static_assert(PairLayout<64>::BYTES + 16 <= CODES_AT,
                  "a worker's attention and its staged rotation factor end before the rows' codes");
};
static_assert(StreamPlan::SLOTS == 18 && StreamPlan::BYTES == 99696, "the host reserves these bytes");

__device__ __forceinline__ StreamShared stream_shared(unsigned char* base) {
    using P = StreamPlan;
    StreamShared s;
    unsigned char* at = base;
    s.ring = at;
    s.a = reinterpret_cast<signed char*>(at += P::SLOTS * STREAM_CHUNK);
    s.steps = reinterpret_cast<float*>(at += P::A_BYTES);
    s.stash_out = reinterpret_cast<__half*>(at += P::STEPS * 4);
    s.codes = reinterpret_cast<int*>(at += P::STASH_BYTES);
    s.weights = reinterpret_cast<const signed char**>(at += P::ROWS * 4);
    s.scales = reinterpret_cast<const float**>(at += STREAM_MAX_LAYERS * 4 * 8);
    s.statics = reinterpret_cast<const float**>(at += STREAM_MAX_LAYERS * 4 * 8);
    s.sums = reinterpret_cast<const float**>(at += STREAM_MAX_LAYERS * 4 * 8);
    s.constants = reinterpret_cast<const float**>(at += STREAM_MAX_LAYERS * 2 * 8);
    s.issued = reinterpret_cast<int*>(at += STREAM_MAX_LAYERS * 2 * 8);
    s.done = reinterpret_cast<unsigned long long*>(at + 8);
    s.full = reinterpret_cast<unsigned long long*>(at += 16);
    s.empty = s.full + P::SLOTS;
    return s;
}

// Issue publication prevents a consumer from observing an earlier slot phase.
__device__ __forceinline__ int stream_issued(int* issued) {
    int value = 0;
    if ((threadIdx.x & 31) == 0)
        asm volatile("atom.acquire.cta.shared.add.u32 %0, [%1], 0;" : "=r"(value)
                     : "r"(shared_address(issued)) : "memory");
    return __shfl_sync(0xffffffffu, value, 0);
}
__device__ __forceinline__ void stream_issue(int* issued, int value) {
    asm volatile("{ .reg .b32 old; atom.release.cta.shared.exch.b32 old, [%0], %1; }"
                 :: "r"(shared_address(issued)), "r"(value) : "memory");
}

// A streamer's chunks count in 32 bits, as the issue count does: wider counters cost the
// consumers' loop a 64-bit division a chunk and the kernel its registers.
static_assert(STREAM_MAX_LAYERS * 2LL * (9 * StreamPlan::A_BYTES / StreamPlan::ROWS / 8) *
                      (StreamPlan::A_BYTES / StreamPlan::ROWS / 64) < (1LL << 31),
              "every layer's units, piece by piece in two digits, at the plan's widest rows");

template <int SLOTS>
__device__ __forceinline__ void ring_take(int* issued, unsigned long long* full, unsigned c) {
    while ((unsigned)stream_issued(issued) <= c) __nanosleep(64);
    __syncwarp();
    stream_bar_wait(&full[c % SLOTS], (c / SLOTS) & 1);
}
// Frees chunk c's slot once the warp is done with it.
template <int SLOTS>
__device__ __forceinline__ void ring_free(unsigned long long* empty, unsigned c) {
    __syncwarp();
    stream_bar_arrive(&empty[c % SLOTS]);
}

// Chained as ReproRestoredChain does; `steps` holds a row's parts side by side from part
// `kt0`, so one load takes a row's steps of two k tiles.
template <int TILES>
__device__ __forceinline__ void stream_products(const unsigned char* slot, const signed char* a,
                                                const float* steps, int parts, int kt0, float (&f)[4]) {
    // Four k tiles at a time: a chunk's tiles unrolled whole keep twice the fragments in
    // flight, and the kernel's long-lived values go to the stack.
    constexpr int GROUP = 4;
    static_assert(TILES % GROUP == 0 && GROUP % 2 == 0, "a load takes two k tiles' steps of a group");
    const int lane = threadIdx.x & 31, g = lane >> 2;
    const unsigned w = shared_address(slot) + int8proj::tile_offset(lane & 7, lane >> 3);
    const unsigned a0 = shared_address(a) + int8proj::tile_offset(lane & 15, lane >> 4);
    const unsigned a1 = shared_address(a) + int8proj::tile_offset(lane & 15, 2 + (lane >> 4));
    const float2* row[2] = {reinterpret_cast<const float2*>(steps + g * parts + kt0),
                            reinterpret_cast<const float2*>(steps + (g + 8) * parts + kt0)};
#pragma unroll 1
    for (int group = 0; group < TILES / GROUP; ++group) {
        float2 step[2];
#pragma unroll
        for (int t = 0; t < GROUP; ++t) {
            const int kt = GROUP * group + t;
            unsigned b[4];
            ldmatrix_x4(b, w + kt * 512);
            const unsigned b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
            unsigned x[4], y[4];
            ldmatrix_x4(x, a0 + kt * 1024);
            ldmatrix_x4(y, a1 + kt * 1024);
            int acc[4];
            int8proj::mma_s8_from(acc, x, b0, int8proj::SUMS_FROM);
            int8proj::mma_s8(acc, y, b1);
            if (t % 2 == 0) step[0] = row[0][kt / 2], step[1] = row[1][kt / 2];
            const float s0 = t % 2 == 0 ? step[0].x : step[0].y, s1 = t % 2 == 0 ? step[1].x : step[1].y;
            f[0] = __fmaf_rn(int8proj::promoted<14>(acc[0]), s0, f[0]);
            f[1] = __fmaf_rn(int8proj::promoted<14>(acc[1]), s0, f[1]);
            f[2] = __fmaf_rn(int8proj::promoted<14>(acc[2]), s1, f[2]);
            f[3] = __fmaf_rn(int8proj::promoted<14>(acc[3]), s1, f[3]);
        }
    }
}

// A deferred projection (`sums` set) takes its shared rows without the bias.
struct StreamColumns {
    float scale[2], fixed[2][2], sum[2], constant[2];
};
__device__ __forceinline__ StreamColumns stream_columns(const float* scales, const float* statics,
                                                        const float* sums, const float* constants,
                                                        const int* codes, int n, int j) {
    const int g = (threadIdx.x & 31) >> 2;
    StreamColumns c{};
    const float2 sc = *reinterpret_cast<const float2*>(scales + j);
    c.scale[0] = sc.x, c.scale[1] = sc.y;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const float2 st = *reinterpret_cast<const float2*>(statics + (size_t)(-codes[g + 8 * h] - 1) * n + j);
        c.fixed[h][0] = st.x, c.fixed[h][1] = st.y;
    }
    if (sums) {
        const float2 su = *reinterpret_cast<const float2*>(sums + j);
        const float2 co = *reinterpret_cast<const float2*>(constants + j);
        c.sum[0] = su.x, c.sum[1] = su.y, c.constant[0] = co.x, c.constant[1] = co.y;
    }
    return c;
}

// (mean, rstd) in one 64-bit word, into a slot zeroed at launch: rstd is
// positive or NaN, so a written word is never zero.
__device__ __forceinline__ unsigned long long stream_moment_word(const RowMoments& m) {
    const float2 v = repro_deferred_moments(m);
    return (unsigned long long)__float_as_uint(v.y) << 32 | __float_as_uint(v.x);
}
__device__ __forceinline__ unsigned long long stream_load_word(const unsigned long long* at) {
    unsigned long long word;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(word) : "l"(at));
    return word;
}
__device__ __forceinline__ void stream_store_word(unsigned long long* at, unsigned long long word) {
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(at), "l"(word) : "memory");
}
// A word not yet written is polled by its lane alone, in a loop entered only then.
__device__ __forceinline__ void stream_await_moments(const unsigned long long* words,
                                                     unsigned long long (&seen)[2], int rows,
                                                     float2 (&m)[2]) {
    const int g = (threadIdx.x & 31) >> 2;
    auto missing = [&](int h) { return g + 8 * h < rows && seen[h] == 0; };
    if (missing(0) || missing(1)) {
        do {
#pragma unroll
            for (int h = 0; h < 2; ++h)
                if (missing(h)) seen[h] = stream_load_word(words + g + 8 * h);
        } while (missing(0) || missing(1));
    }
#pragma unroll
    for (int h = 0; h < 2; ++h)
        m[h] = make_float2(__uint_as_float((unsigned)seen[h]), __uint_as_float((unsigned)(seen[h] >> 32)));
}

// Zeros past `rows`; the copies are issued, not committed.
__device__ __forceinline__ void stream_load_inputs(signed char* buffer, const signed char* rows_q, int rows,
                                                   int k, int depth, int piece, int t, int count) {
    const int chunks = depth / 16;
    for (int i = t; i < STREAM_TILE_ROWS * chunks; i += count) {
        const int r = i / chunks, c = i % chunks;
        cp_async_16(shared_address(buffer + (c / 4) * 1024 + int8proj::tile_offset(r, c % 4)),
                    rows_q + (size_t)min(r, rows - 1) * k + piece * depth + 16 * c, r < rows ? 16 : 0);
    }
}

// A piece's PER parts for the tile's rows, a row's side by side; rows past `rows` take the
// last row's steps; issued, not committed.
template <int PER>
__device__ __forceinline__ void stream_load_piece_steps(float* steps, const float* row_steps, int rows, int parts,
                                                        int piece, int t, int count) {
    for (int i = t; i < STREAM_TILE_ROWS * PER; i += count) {
        const int r = i / PER, c = piece * PER + i % PER;
        cp_async_4(shared_address(steps + r * parts + c), repro_step_at(row_steps, rows, c, min(r, rows - 1)));
    }
}

// Issued by the consumers, not committed.
__device__ __forceinline__ void stream_load_stash(__half* stash, const __half* hidden, int rows, int h,
                                                  StreamUnits units) {
    if ((int)threadIdx.x < STREAM_TILE_ROWS * units.count) {
        const int r = threadIdx.x / units.count, unit = threadIdx.x % units.count;
        cp_async_16(shared_address(stash + r * STREAM_STASH + 8 * unit),
                    hidden + (size_t)min(r, rows - 1) * h + 8 * (units.first + unit), 16);
    }
}

// repro_restored_epilogue's operations, then repro_deferred_value's if deferred.
__device__ __forceinline__ float restored_value(float f, float scale, float fixed, bool deferred, float2 m,
                                                float sum, float constant) {
    const float v = __fadd_rn(__fmul_rn(f, scale), fixed);
    return deferred ? repro_deferred_value(v, m, sum, constant) : v;
}

// The restored epilogues for a row tile, in repro_restored_epilogue's order.
template <int E>
__device__ __forceinline__ void stream_epilogue(const float (&f)[4], const StreamColumns& cols,
                                                bool deferred, const float2 (&m)[2],
                                                const __half* stash, int unit, const float* fc,
                                                int rows, int n, int j, void* out) {
    const int lane = threadIdx.x & 31, g = lane >> 2;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        const int r = g + 8 * hh;
        if (r >= rows) continue;
        float v0 = restored_value(f[2 * hh], cols.scale[0], cols.fixed[hh][0], deferred, m[hh], cols.sum[0],
                                  cols.constant[0]);
        float v1 = restored_value(f[2 * hh + 1], cols.scale[1], cols.fixed[hh][1], deferred, m[hh], cols.sum[1],
                                  cols.constant[1]);
        const size_t at = (size_t)r * n + j;
        if constexpr (E == REPRO_PLAIN) {
            *reinterpret_cast<float2*>(static_cast<float*>(out) + at) = make_float2(v0, v1);
        } else if constexpr (E == REPRO_RESIDUAL) {
            const __half2 residual =
                *reinterpret_cast<const __half2*>(stash + r * STREAM_STASH + 8 * unit + 2 * (lane & 3));
            v0 = __fadd_rn(v0, __low2float(residual));
            v1 = __fadd_rn(v1, __high2float(residual));
            *reinterpret_cast<__half2*>(static_cast<__half*>(out) + at) =
                __halves2half2(__float2half_rn(v0), __float2half_rn(v1));
        } else {
            *reinterpret_cast<__half2*>(static_cast<__half*>(out) + at) =
                __halves2half2(__float2half_rn(repro_gelu(v0, fc)), __float2half_rn(repro_gelu(v1, fc)));
        }
    }
}

// The kernel's shapes and buffers, the same for every layer.
struct StreamRun {
    const StreamLayer* layers;
    int layer_count, rows, h, inner, heads, sequences;
    const int *starts, *lengths, *row_codes, *classes, *group_of_row;
    const float* fc;
    const unsigned short* weights;
    __half *hidden, *pre;
    signed char* quantized;
    float *steps, *qkv, *value_scales;
    __half* intermediate;
    signed char* next_q;
    float* next_steps;
    unsigned* counters;
    // Statistics words for the deferred projections, zero at launch.
    unsigned long long* moments;
    // Bit l: layer l's QKV, and its up projection, take an operand of two digits.
    unsigned two_qkv, two_up;

    // The layers, a bit each, whose projection `p` (QKV, out, up, down) takes two digits.
    __device__ __forceinline__ unsigned two_digit(int p) const { return p == 0 ? two_qkv : p == 2 ? two_up : 0u; }
    // The operand's digits of layer `layer`'s projection `p`.
    __device__ __forceinline__ int digits(int layer, int p) const { return 1 + (int)(two_digit(p) >> layer & 1u); }
    // A layer's hand-off counters, at an unsigned offset: a signed one's extension is a
    // register the consumers' loops have no room for.
    __device__ __forceinline__ unsigned* handoffs(int layer) const {
        return counters + (unsigned)layer * STREAM_HANDOFFS;
    }
};

// Each input chunk has its own copy group, so consumers can start before all chunks arrive.
template <int TILES, int E>
__device__ __forceinline__ void stream_project(const StreamRun& run, const StreamShared& s, int layer, int p,
                                               int n, StreamUnits units, unsigned base, int waited,
                                               unsigned target, bool stash, int signalled, void* out) {
    using P = StreamPlan;
    constexpr int DEPTH = TILES * 64, PER = DEPTH / HALF;
    const int warp = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0), lane = threadIdx.x & 31;
    const int h = run.h, walks = run.digits(layer, p), parts = walks * h / HALF, chunks = h / DEPTH;
    const int deferred_at = 2 * layer + p / 2;
    const float* sums = p % 2 == 0 ? s.sums[deferred_at] : nullptr;
    const unsigned long long* words = sums ? run.moments + deferred_at * P::ROWS : nullptr;
    unsigned* counters = run.handoffs(layer);
    int before = 0;
    do {
        const int unit = before + warp, among = min(units.count - before, STREAM_WARPS);
        const bool owned = unit < units.count;
        const int j = 8 * (units.first + unit) + 2 * (lane & 3);
        StreamColumns cols = {};
        if (owned)
            cols = stream_columns(s.scales[4 * layer + p], s.statics[4 * layer + p], sums,
                                  sums ? s.constants[deferred_at] : nullptr, s.codes, n, j);
        float f[4] = {};
        unsigned long long seen[2] = {};
        for (int walk = 0; walk < walks; ++walk) {
            // The shared buffer holds one digit. Reusing it for a second digit requires a consumer barrier.
            const bool stages = before == 0 || walks > 1;
            if (stages) {
                if (before == 0 && walk == 0) {
                    if (waited >= 0) repro_wait(&counters[waited], target, StreamConsumers{});
                } else {
                    StreamConsumers{}();
                }
                for (int i = 0; i < chunks; ++i) {
                    const int live = min(STREAM_TILE_ROWS, run.rows), piece = walk * chunks + i;
                    // Both digits' steps fit their place, staged once.
                    if (before == 0)
                        stream_load_piece_steps<PER>(s.steps, run.steps, live, parts, piece, threadIdx.x,
                                                     STREAM_WARPS * 32);
                    stream_load_inputs(s.a + i * TILES * 1024, run.quantized, live, walks * h, DEPTH, piece,
                                       threadIdx.x, STREAM_WARPS * 32);
                    if (stash && before == 0 && i == chunks - 1)
                        stream_load_stash(s.stash_out, run.hidden, live, h, units);
                    cp_async_commit();
                }
            }
            for (int i = 0; i < chunks; ++i) {
                if (stages) {
                    // Piece i landed: at most the pieces after it still in flight.
                    if (i + 1 < chunks)
                        cp_async_wait<1>();
                    else
                        cp_async_wait<0>();
                    StreamConsumers{}();
                }
                if (owned) {
                    const int piece = walk * chunks + i;
                    const unsigned c = base + (unsigned)(before * walks * chunks + piece * among + warp);
                    if (words && piece == walks * chunks - 1) {
                        seen[0] = stream_load_word(words + (lane >> 2));
                        seen[1] = stream_load_word(words + (lane >> 2) + 8);
                    }
                    ring_take<P::SLOTS>(s.issued, s.full, c);
                    stream_products<TILES>(s.ring + (c % P::SLOTS) * STREAM_CHUNK, s.a + i * TILES * 1024,
                                           s.steps, parts, piece * TILES, f);
                    ring_free<P::SLOTS>(s.empty, c);
                }
            }
        }
        float2 m[2] = {};
        if (words && owned) stream_await_moments(words, seen, min(STREAM_TILE_ROWS, run.rows), m);
        if (owned)
            stream_epilogue<E>(f, cols, words != nullptr, m, s.stash_out, unit, run.fc,
                               min(STREAM_TILE_ROWS, run.rows), n, j, out);
        before += STREAM_WARPS;
    } while (before < units.count);
    repro_signal(&counters[signalled], StreamConsumers{});
}

// The down projection's input in one round, as repro_restored_next_input makes it; rows
// past the input clamp their indices and store nothing, keeping a warp's halves together.
template <int ROWS>
__device__ __forceinline__ void stream_blocks_all(const StreamRun& run, const float* centers, const float* signs,
                                                  const int* codes, int sm, int sms, int warps) {
    const int warp = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0), lane = threadIdx.x & 31;
    const int inner = run.inner, blocks = inner / BLOCK, pairs = blocks * (ROWS / 2);
    if (warp >= warps) return;
    const float rotate = run.fc[FC_RECIP_SQRT_BLOCK];
    for (int p = sm + sms * warp; p < pairs; p += sms * warps) {
        const int b = p / (ROWS / 2), r = 2 * (p % (ROWS / 2)) + lane / 16;
        const int rr = min(r, run.rows - 1), part = lane % 16, c = BLOCK * b + 8 * part;
        const uint4 raw = *reinterpret_cast<const uint4*>(run.intermediate + (size_t)rr * inner + c);
        const float4* center = reinterpret_cast<const float4*>(centers + (size_t)(-codes[rr] - 1) * inner + c);
        const float4* sign = reinterpret_cast<const float4*>(signs + c);
        const float4 c0 = center[0], c1 = center[1], s0 = sign[0], s1 = sign[1];
        const float cv[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
        const unsigned words[4] = {raw.x, raw.y, raw.z, raw.w};
        float x[8];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 pair = __half22float2(*reinterpret_cast<const __half2*>(&words[e]));
            x[2 * e] = __fmul_rn(__fsub_rn(pair.x, cv[2 * e]), sv[2 * e] < 0.0f ? -1.0f : 1.0f);
            x[2 * e + 1] = __fmul_rn(__fsub_rn(pair.y, cv[2 * e + 1]), sv[2 * e + 1] < 0.0f ? -1.0f : 1.0f);
        }
        const float most = repro_rotate_block<8>(x, part, rotate);
        const float scale = most > 0.0f ? __fdiv_rn(most, 127.0f) : 1.0f;
        if (r < run.rows) {
            repro_store_block<8>(x, scale, run.next_q + (size_t)r * inner + c);
            if (part % 8 == 0) *repro_step_at(run.next_steps, run.rows, 2 * b + part / 8, r) = scale;
        }
    }
}

// Quarter 3's streamers finish the down projection.
__device__ __forceinline__ unsigned stream_finishers(int streamers) { return (unsigned)(streamers / 4); }

// Each streamer chains a quarter of the depth from +0, parking FP32 parts in the idle QKV
// buffer; the finishers add ((f0 + f1) + f2) + f3, the CPU's restarted chain.
template <int TILES>
__device__ __forceinline__ void stream_down(const StreamRun& run, const StreamShared& s, int layer,
                                            StreamUnits units, unsigned base) {
    using P = StreamPlan;
    constexpr int DEPTH = TILES * 64, PER = DEPTH / HALF;
    const int warp = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0), lane = threadIdx.x & 31, g = lane >> 2;
    const int h = run.h, inner = run.inner, steps_a_row = inner / HALF, rows = run.rows;
    const int streamers = gridDim.x - STREAM_WORKERS, quarter = blockIdx.x % 4;
    const int pieces = h / DEPTH, first_piece = quarter * pieces;
    const bool finisher = quarter == 3;
    const StreamLayer& l = run.layers[layer];
    unsigned* counters = run.handoffs(layer);
    float* parts = run.qkv;
    // A round as stream_project's; the other quarters count their parts after their last.
    int before = 0;
    do {
        const int index = before + warp, among = min(units.count - before, STREAM_WARPS);
        const bool unit = index < units.count;
        const int j = 8 * (units.first + index) + 2 * (lane & 3);
        StreamColumns cols = {};
        if (finisher && unit)
            cols = stream_columns(s.scales[4 * layer + 3], s.statics[4 * layer + 3], nullptr, nullptr, s.codes,
                                  h, j);
        if (before == 0) {
            repro_wait(&counters[STREAM_UP], streamers, StreamConsumers{});
            stream_blocks_all<P::ROWS>(run, l.centers[3], l.signs[3], s.codes, blockIdx.x, gridDim.x, STREAM_WARPS);
            repro_signal(&counters[STREAM_BLOCKS], StreamConsumers{});
            repro_wait(&counters[STREAM_BLOCKS], gridDim.x, StreamConsumers{});
            for (int i = 0; i < pieces; ++i) {
                const int live = min(STREAM_TILE_ROWS, rows);
                stream_load_piece_steps<PER>(s.steps, run.next_steps, live, steps_a_row, first_piece + i,
                                             threadIdx.x, STREAM_WARPS * 32);
                stream_load_inputs(s.a + i * TILES * 1024, run.next_q, live, inner, DEPTH, first_piece + i,
                                   threadIdx.x, STREAM_WARPS * 32);
                cp_async_commit();
            }
        }
        __half2 residual[2] = {};
        if (finisher && unit)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
                residual[hh] = __ldcg(
                    reinterpret_cast<const __half2*>(run.hidden + (size_t)min(g + 8 * hh, rows - 1) * h + j));
        float f[4] = {};
        for (int i = 0; i < pieces; ++i) {
            if (before == 0) {
                // Piece i landed: at most the pieces after it still in flight.
                if (i + 1 < pieces)
                    cp_async_wait<1>();
                else
                    cp_async_wait<0>();
                StreamConsumers{}();
            }
            if (unit) {
                const unsigned c = base + (unsigned)(before * pieces + i * among + warp);
                ring_take<P::SLOTS>(s.issued, s.full, c);
                stream_products<TILES>(s.ring + (c % P::SLOTS) * STREAM_CHUNK, s.a + i * TILES * 1024, s.steps,
                                       steps_a_row, (first_piece + i) * TILES, f);
                ring_free<P::SLOTS>(s.empty, c);
            }
        }
        if (!finisher) {
            if (unit)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int r = g + 8 * hh;
                    if (r < rows)
                        __stcg(reinterpret_cast<float2*>(parts + ((size_t)quarter * rows + r) * h + j),
                               make_float2(f[2 * hh], f[2 * hh + 1]));
                }
        } else {
            repro_wait(&counters[STREAM_PARTS], (unsigned)streamers - stream_finishers(streamers),
                       StreamConsumers{});
            if (unit) {
                float2 part[2][3];
#pragma unroll
                for (int hh = 0; hh < 2; ++hh)
#pragma unroll
                    for (int q = 0; q < 3; ++q)
                        part[hh][q] = __ldcg(reinterpret_cast<const float2*>(
                            parts + ((size_t)q * rows + min(g + 8 * hh, rows - 1)) * h + j));
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int r = g + 8 * hh;
                    if (r >= rows) continue;
                    const float f0 =
                        __fadd_rn(__fadd_rn(__fadd_rn(part[hh][0].x, part[hh][1].x), part[hh][2].x), f[2 * hh]);
                    const float f1 = __fadd_rn(__fadd_rn(__fadd_rn(part[hh][0].y, part[hh][1].y), part[hh][2].y),
                                               f[2 * hh + 1]);
                    float v0 = restored_value(f0, cols.scale[0], cols.fixed[hh][0], false, float2{}, 0.0f, 0.0f);
                    float v1 = restored_value(f1, cols.scale[1], cols.fixed[hh][1], false, float2{}, 0.0f, 0.0f);
                    v0 = __fadd_rn(v0, __low2float(residual[hh]));
                    v1 = __fadd_rn(v1, __high2float(residual[hh]));
                    *reinterpret_cast<__half2*>(run.pre + (size_t)r * h + j) =
                        __halves2half2(__float2half_rn(v0), __float2half_rn(v1));
                }
            }
        }
        before += STREAM_WARPS;
    } while (before < units.count);
    repro_signal(&counters[finisher ? STREAM_DOWN : STREAM_PARTS], StreamConsumers{});
}

// Staged during the phase before the norm, when the worker's attention layout,
// also at the base, is done.
struct NormStage {
    float *gamma, *beta, *signs, *centers, *rotate;
    RowMoments* moments;
};
__device__ __forceinline__ NormStage norm_stage(unsigned char* base, int h) {
    float* at = reinterpret_cast<float*>(base);
    NormStage st;
    st.gamma = at;
    st.beta = at + h;
    st.signs = at + 2 * h;
    st.centers = at + 3 * h;
    st.rotate = at + (3 + REPRO_CLASSES) * h;
    st.moments = reinterpret_cast<RowMoments*>(at + (3 + REPRO_CLASSES) * h + 4);
    return st;
}
// A LayerNorm stage's bytes at the widest h (1024).
constexpr int STREAM_NORM_STAGE_BYTES = (3 + REPRO_CLASSES) * 1024 * 4 + 16 + 2 * (int)sizeof(RowMoments);

// Committed, not waited.
__device__ __forceinline__ void stream_stage_norm(const NormStage& st, const float* gamma, const float* beta,
                                                  const float* centers, const float* signs, const float* fc,
                                                  int h) {
    for (int i = threadIdx.x; i < h / 4; i += STREAM_THREADS) {
        cp_async_16(shared_address(st.gamma + 4 * i), gamma + 4 * i, 16);
        cp_async_16(shared_address(st.beta + 4 * i), beta + 4 * i, 16);
        if (centers) cp_async_16(shared_address(st.signs + 4 * i), signs + 4 * i, 16);
    }
    if (centers) {
        for (int i = threadIdx.x; i < REPRO_CLASSES * h / 4; i += STREAM_THREADS)
            cp_async_16(shared_address(st.centers + 4 * i), centers + 4 * i, 16);
        if (threadIdx.x == 0) cp_async_4(shared_address(st.rotate), fc + FC_RECIP_SQRT_BLOCK);
    }
    cp_async_commit();
}

// Statistics warps hold a row in the CPU's lane order (repro_row_moments);
// item warps hold two (row, 128-block) items each.
struct NormItems {
    float x[32];
    uint4 raw;
    int s, sr, row, ir, ib, c, part;
    bool stats, any, live;
};
__device__ __forceinline__ NormItems stream_norm_items(const StreamRun& run, int unit) {
    const int warp = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0), lane = threadIdx.x & 31;
    const int h = run.h, blocks = h / BLOCK, first = STREAM_LN_ROWS * unit;
    NormItems it;
    it.s = warp - STREAM_STATS_WARP;
    it.sr = first + it.s;
    it.stats = it.s >= 0 && it.s < STREAM_LN_ROWS && it.sr < run.rows;
    if (it.stats) {
#pragma unroll
        for (int i = 0; i < 32; ++i)
            if (32 * i < h) it.x[i] = __half2float(run.pre[(size_t)it.sr * h + lane + 32 * i]);
    }
    const int item = 2 * warp + lane / 16;
    it.part = lane % 16;
    it.row = item / blocks;
    it.ir = first + it.row, it.ib = item % blocks, it.c = BLOCK * it.ib + 8 * it.part;
    it.any = 2 * warp < STREAM_LN_ROWS * blocks;
    it.live = item < STREAM_LN_ROWS * blocks && it.ir < run.rows;
    it.raw = make_uint4(0, 0, 0, 0);
    if (it.live) it.raw = *reinterpret_cast<const uint4*>(run.pre + (size_t)it.ir * h + it.c);
    return it;
}

// repro_layernorm_quantize's values.
__device__ __forceinline__ void stream_norm_store(const StreamRun& run, const NormStage& st,
                                                  const NormItems& it) {
    if (!it.live) return;
    const RowMoments m = st.moments[it.row];
    unsigned words[4] = {it.raw.x, it.raw.y, it.raw.z, it.raw.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int k = it.c + 2 * e;
        const float2 pair = __half22float2(*reinterpret_cast<const __half2*>(&words[e]));
        const __half2 value = __halves2half2(__float2half_rn(repro_normalized(pair.x, m, st.gamma[k], st.beta[k])),
                                             __float2half_rn(repro_normalized(pair.y, m, st.gamma[k + 1],
                                                                              st.beta[k + 1])));
        words[e] = *reinterpret_cast<const unsigned*>(&value);
    }
    *reinterpret_cast<uint4*>(run.hidden + (size_t)it.ir * run.h + it.c) =
        make_uint4(words[0], words[1], words[2], words[3]);
}

// The stores follow the block's meeting on the rows' statistics.
__device__ __forceinline__ void stream_layernorm_items(const StreamRun& run, const NormStage& st, int worker,
                                                       float eps) {
    const NormItems it = stream_norm_items(run, worker);
    if (it.stats) {
        const RowMoments m = repro_row_moments(it.x, eps, run.h);
        if ((threadIdx.x & 31) == 0) st.moments[it.s] = m;
    }
    StreamBlock{}();
    stream_norm_store(run, st, it);
}

// Item warps make the next input from the raw rows, in `digits` digits, and meet on barrier
// 3, so their stores precede thread 0's count, while statistics warps publish words.
__device__ __forceinline__ void stream_deferred_layernorm(const StreamRun& run, const NormStage& st,
                                                          const int* codes, int unit, float eps,
                                                          unsigned* signalled, unsigned long long* words,
                                                          int digits) {
    const int h = run.h, blocks = h / BLOCK;
    const NormItems it = stream_norm_items(run, unit);
    if (it.any) {
        const float* center = st.centers + (size_t)(-codes[min(it.ir, run.rows - 1)] - 1) * h + it.c;
        const unsigned raw[4] = {it.raw.x, it.raw.y, it.raw.z, it.raw.w};
        float v[8];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 pair = __half22float2(*reinterpret_cast<const __half2*>(&raw[e]));
            const int k = 2 * e;
            v[k] = __fmul_rn(__fsub_rn(__fmul_rn(pair.x, st.gamma[it.c + k]), center[k]),
                             st.signs[it.c + k] < 0.0f ? -1.0f : 1.0f);
            v[k + 1] = __fmul_rn(__fsub_rn(__fmul_rn(pair.y, st.gamma[it.c + k + 1]), center[k + 1]),
                                 st.signs[it.c + k + 1] < 0.0f ? -1.0f : 1.0f);
        }
        const float most = repro_rotate_block<8>(v, it.part, *st.rotate);
        const float scale = most > 0.0f ? __fdiv_rn(most, 127.0f) : 1.0f;
        if (it.live) {
            signed char* q = run.quantized + (size_t)it.ir * digits * h + it.c;
            repro_store_block<8>(v, scale, q, digits == 2 ? q + h : nullptr);
            if (it.part % 8 == 0) {
                const int part = 2 * it.ib + it.part / 8;
                *repro_step_at(run.steps, run.rows, part, it.ir) = scale;
                if (digits == 2) *repro_step_at(run.steps, run.rows, h / HALF + part, it.ir) = repro_second_step(scale);
            }
        }
        repro_bar_sync<3>(32 * blocks);
        if (threadIdx.x == 0) {
            __threadfence();
            atomicAdd(signalled, 1u);
        }
    }
    if (it.stats) {
        const RowMoments m = repro_row_moments(it.x, eps, h);
        if ((threadIdx.x & 31) == 0) {
            st.moments[it.s] = m;
            stream_store_word(words + it.sr, stream_moment_word(m));
        }
    }
    StreamBlock{}();
    stream_norm_store(run, st, it);
}

// No other worker phase writes the table's bytes, so it is copied once a launch.
template <int D>
__host__ __device__ constexpr int pair_table_at() {
    return PairLayout<D>::TABLE_AT;
}
template <int D>
__host__ __device__ constexpr int pair_centers_at() {
    return pair_table_at<D>() + PairLayout<D>::TABLE_BYTES + PairLayout<D>::FAILED_BYTES;
}
template <int D>
__device__ __forceinline__ void stream_stage_attention_table(unsigned char* base, const unsigned short* weights) {
    static_assert(pair_table_at<D>() >= STREAM_NORM_STAGE_BYTES, "the LayerNorm's stage lies below the table");
    for (int i = threadIdx.x; i < ATT_WEIGHT_ENTRIES / 8; i += STREAM_THREADS)
        cp_async_16(shared_address(base + pair_table_at<D>() + 16 * i), weights + 8 * i, 16);
    cp_async_commit();
}
template <int D>
__device__ __forceinline__ void stream_stage_attention(unsigned char* base, const float* centers, const float* signs,
                                                       const float* fc, int h, int blk) {
    float* cs = reinterpret_cast<float*>(base + pair_centers_at<D>());
    float* ss = cs + PAIR_CENTER_ROWS * BLOCK;
    for (int i = threadIdx.x; i < (REPRO_CLASSES + 1) * BLOCK / 4; i += STREAM_THREADS) {
        const int row = i / (BLOCK / 4), c = 4 * (i % (BLOCK / 4));
        const float* src = row < REPRO_CLASSES ? centers + (size_t)row * h + BLOCK * blk + c : signs + BLOCK * blk + c;
        float* dst = row < REPRO_CLASSES ? cs + row * BLOCK + c : ss + c;
        cp_async_16(shared_address(dst), src, 16);
    }
    if (threadIdx.x == 0) cp_async_4(shared_address(base + PairLayout<D>::BYTES), fc + FC_RECIP_SQRT_BLOCK);
    cp_async_commit();
}
template <int D>
__device__ __forceinline__ void stream_attention(const StreamRun& run, const StreamLayer& l, const StreamShared& s,
                                                 const unsigned char* base, int seq, int blk, int seq_start,
                                                 int seq_len) {
    repro_attention_pairs<D, StreamPlan::ROWS, true>(
        run.qkv, run.starts, run.lengths, run.h, run.heads, l.key_centers, run.weights, run.value_scales,
        l.centers[1], REPRO_CLASSES, run.classes, run.group_of_row, l.signs[1], run.fc, run.quantized, run.steps,
        run.rows, nullptr, seq, blk, s.codes, reinterpret_cast<const float*>(base + PairLayout<D>::BYTES),
        seq_start, seq_len);
}

// Every layer in one launch, with `counters` and `moments` zero and every norm deferred.
template <int D, int TILES>
__device__ __forceinline__ void repro_stream_layers_impl(const StreamRun& run) {
    using P = StreamPlan;
    constexpr int DEPTH = TILES * 64;
    static_assert(STREAM_WORKERS * STREAM_LN_ROWS == P::ROWS, "a worker's LayerNorm unit for every two rows");
    extern __shared__ __align__(128) unsigned char stream_base[];
    const StreamShared s = stream_shared(stream_base);
    const int streamers = gridDim.x - STREAM_WORKERS, h = run.h, inner = run.inner;
    if (blockIdx.x >= streamers) {
        const int worker = blockIdx.x - streamers;
        const int blocks = h / BLOCK, units = run.sequences * blocks;
        const NormStage st = norm_stage(stream_base, h);
        // The down projection's rows: quarter 3's streamers'.
        const unsigned finished = stream_finishers(streamers);
        if (threadIdx.x < P::ROWS) s.codes[threadIdx.x] = run.row_codes[min((int)threadIdx.x, run.rows - 1)];
        // With one item a worker, its parameters stage ahead.
        const bool staged = units <= STREAM_WORKERS;
        const bool attends = worker < units;
        if (staged) stream_stage_attention_table<D>(stream_base, run.weights);
        // Only an attending worker's sequence is used.
        const int seq = min(worker / blocks, run.sequences - 1);
        const int seq_start = run.starts[seq], seq_len = run.lengths[seq];
        for (int layer = 0; layer < run.layer_count; ++layer) {
            const StreamLayer& l = run.layers[layer];
            unsigned* counters = run.handoffs(layer);
            if (staged && attends)
                stream_stage_attention<D>(stream_base, l.centers[1], l.signs[1], run.fc, h, worker % blocks);
            if (layer > 0) {
                const StreamLayer& last = run.layers[layer - 1];
                repro_wait(&counters[STREAM_DOWN - STREAM_HANDOFFS], finished, StreamBlock{});
                cp_async_wait<0>();
                StreamBlock{}();
                stream_deferred_layernorm(run, st, s.codes, worker, last.eps2, &counters[STREAM_QKV_IN],
                                          run.moments + 2 * layer * P::ROWS, run.digits(layer, 0));
            }
            repro_wait(&counters[STREAM_QKV], streamers, StreamBlock{});
            if (staged) {
                if (attends)
                    stream_attention<D>(run, l, s, stream_base, worker / blocks, worker % blocks, seq_start, seq_len);
                __syncthreads();
            } else {
                for (int unit = worker; unit < units; unit += STREAM_WORKERS) {
                    repro_attention_pairs<D, P::ROWS>(run.qkv, run.starts, run.lengths, h, run.heads,
                                                      l.key_centers, run.weights, run.value_scales, l.centers[1],
                                                      REPRO_CLASSES, run.classes, run.group_of_row, l.signs[1],
                                                      run.fc, run.quantized, run.steps, run.rows, nullptr,
                                                      unit / blocks, unit % blocks);
                    __syncthreads();
                }
            }
            repro_signal(&counters[STREAM_ATT], StreamBlock{});
            stream_stage_norm(st, l.gamma1, l.beta1, l.centers[2], l.signs[2], run.fc, h);
            repro_wait(&counters[STREAM_OUT], streamers, StreamBlock{});
            cp_async_wait<0>();
            StreamBlock{}();
            stream_deferred_layernorm(run, st, s.codes, worker, l.eps1, &counters[STREAM_LN1],
                                      run.moments + (2 * layer + 1) * P::ROWS, run.digits(layer, 2));
            // The worker's items of the down projection's input, counted
            // after its normalized rows (the down projection's residual).
            repro_wait(&counters[STREAM_UP], streamers, StreamBlock{});
            stream_blocks_all<P::ROWS>(run, l.centers[3], l.signs[3], s.codes, blockIdx.x, gridDim.x,
                                       STREAM_WARPS);
            repro_signal(&counters[STREAM_BLOCKS], StreamBlock{});
            // The block met in the signal, so the stage is free to restage.
            const StreamLayer* next = layer + 1 < run.layer_count ? &run.layers[layer + 1] : nullptr;
            stream_stage_norm(st, l.gamma2, l.beta2, next ? next->centers[0] : nullptr,
                              next ? next->signs[0] : nullptr, run.fc, h);
        }
        const StreamLayer& last = run.layers[run.layer_count - 1];
        repro_wait(&run.handoffs(run.layer_count - 1)[STREAM_DOWN], finished, StreamBlock{});
        cp_async_wait<0>();
        StreamBlock{}();
        stream_layernorm_items(run, st, worker, last.eps2);
        return;
    }

    // Taken past the workers' branch: from the kernel's entry the assembler holds it across
    // that branch on the stack.
    const int warp = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0), lane = threadIdx.x & 31;
    // Every projection is h deep a streamer: the down projection's quarter.
    const int streamer = blockIdx.x;
    const StreamUnits units[4] = {stream_units(3 * h, streamer, streamers), stream_units(h, streamer, streamers),
                                  stream_units(inner, streamer, streamers),
                                  stream_units(h, streamer / 4, (streamers - streamer % 4 + 3) / 4)};
    if (threadIdx.x == 0) {
        for (int i = 0; i < P::SLOTS; ++i) {
            stream_bar_init(&s.full[i], 32);
            stream_bar_init(&s.empty[i], 32);
        }
        stream_bar_init(s.done, 32);
        *s.issued = 0;
        // The plan's bounds, which repro.rs checks before choosing the stream.
        if (units[1].count > STREAM_STASH_UNITS || inner != 4 * h || streamers < 4 || P::ROWS * h > P::A_BYTES ||
            run.rows <= 0 || run.rows > P::ROWS || run.layer_count > STREAM_MAX_LAYERS)
            __trap();
    }
    if (threadIdx.x < P::ROWS) s.codes[threadIdx.x] = run.row_codes[min((int)threadIdx.x, run.rows - 1)];
    for (int i = threadIdx.x; i < 4 * run.layer_count; i += STREAM_THREADS) {
        const StreamLayer& l = run.layers[i / 4];
        s.weights[i] = l.q[i % 4];
        s.scales[i] = l.scales[i % 4];
        s.statics[i] = l.statics[i % 4];
    }
    // The QKV and up projections' column terms (stream_project's 2 layer +
    // p / 2).
    for (int i = threadIdx.x; i < 2 * run.layer_count; i += STREAM_THREADS) {
        const StreamLayer& l = run.layers[i / 2];
        s.sums[i] = l.sums[2 * (i % 2)];
        s.constants[i] = l.constants[2 * (i % 2)];
    }
    __syncthreads();

    if (warp == STREAM_WARPS) {
        const unsigned long long policy = l2_evict_first();
        unsigned c = 0;
        for (int layer = 0; layer < run.layer_count; ++layer)
            for (int p = 0; p < 4; ++p) {
                const int stride = p == 3 ? inner : h, from = p == 3 ? (streamer % 4) * h : 0;
                const signed char* weights = s.weights[4 * layer + p];
                // A round's chunks: the units its consumer warps take, piece by piece, the
                // pieces again for an operand's second digits.
                const int pieces = h / DEPTH;
                for (int before = 0; before < units[p].count; before += STREAM_WARPS)
                    for (int walked = 0; walked < run.digits(layer, p) * pieces; ++walked)
                        for (int unit = before; unit < min(units[p].count, before + STREAM_WARPS); ++unit, ++c) {
                            const int j = walked % pieces;
                            const int slot = (int)(c % P::SLOTS);
                            if (c >= P::SLOTS) stream_bar_wait(&s.empty[slot], (c / P::SLOTS - 1) & 1);
                            // Chunks in flight land in no order: issued without bound, the chunk a
                            // consumer waits for lands behind the ring's whole lookahead. Tested in
                            // line: where the memory leads the producer, a call a chunk is its delay.
                            if (c >= STREAM_AHEAD) {
                                unsigned long long* landed = &s.full[(c - STREAM_AHEAD) % P::SLOTS];
                                const unsigned lap = ((c - STREAM_AHEAD) / P::SLOTS) & 1;
                                if (!stream_bar_test(landed, lap)) stream_bar_wait(landed, lap);
                            }
                            const signed char* src =
                                weights + (size_t)(units[p].first + unit) * 8 * stride + from + (size_t)j * DEPTH;
                            const unsigned dst = shared_address(s.ring + slot * STREAM_CHUNK);
                            constexpr int PIECES = 8 * DEPTH / 16;
#pragma unroll
                            for (int i = lane; i < PIECES; i += 32) {
                                const int col = i / (DEPTH / 16), piece = i % (DEPTH / 16);
                                cp_async_16_policy(dst + (piece / 4) * 512 + int8proj::tile_offset(col, piece % 4),
                                                   src + (size_t)col * stride + 16 * piece, policy);
                            }
                            stream_cp_arrive(&s.full[slot]);
                            __syncwarp();
                            if (lane == 0) stream_issue(s.issued, (int)(c + 1));
                        }
            }
        // A separate phase drains copies without grouping reused destinations.
        stream_cp_arrive(s.done);
        stream_bar_wait(s.done, 0);
        return;
    }

    // Recompute the producer's chunk index per layer to free a register in the consumer loop.
    // Each earlier two-digit projection accounts for one extra pass over its weights.
    unsigned chunks[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) chunks[p] = (unsigned)(units[p].count * (h / DEPTH));
    for (int layer = 0; layer < run.layer_count; ++layer) {
        const unsigned earlier = (1u << layer) - 1;
        unsigned next = 0;
#pragma unroll
        for (int p = 0; p < 4; ++p) next += chunks[p] * ((unsigned)layer + __popc(run.two_digit(p) & earlier));
        unsigned first[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            first[p] = next;
            next += chunks[p] * (unsigned)run.digits(layer, p);
        }
        stream_project<TILES, REPRO_PLAIN>(run, s, layer, 0, 3 * h, units[0], first[0], layer > 0 ? STREAM_QKV_IN : -1,
                                           STREAM_WORKERS, false, STREAM_QKV, run.qkv);
        stream_project<TILES, REPRO_RESIDUAL>(run, s, layer, 1, h, units[1], first[1], STREAM_ATT, STREAM_WORKERS, true,
                                              STREAM_OUT, run.pre);
        stream_project<TILES, REPRO_GELU>(run, s, layer, 2, inner, units[2], first[2], STREAM_LN1, STREAM_WORKERS,
                                          false, STREAM_UP, run.intermediate);
        stream_down<TILES>(run, s, layer, units[3], first[3]);
    }
}

// The kernel at e5's head width and chunk depth (64, 512 inputs).
extern "C" __global__ void __launch_bounds__(STREAM_THREADS, 1) repro_stream_layers_64_8(
    const StreamLayer* layers, int layer_count, int rows, int h, int inner, int heads, int sequences, const int* starts,
    const int* lengths, const int* row_codes, const int* classes, const int* group_of_row, const float* fc,
    const unsigned short* weights, __half* hidden, __half* pre, signed char* quantized, float* steps, float* qkv,
    float* value_scales, __half* intermediate, signed char* next_q, float* next_steps, unsigned* counters,
    unsigned long long* moments, unsigned two_qkv, unsigned two_up) {
    repro_stream_layers_impl<64, 8>(StreamRun{
        layers, layer_count, rows, h, inner, heads, sequences, starts, lengths, row_codes, classes, group_of_row, fc,
        weights, hidden, pre, quantized, steps, qkv, value_scales, intermediate, next_q, next_steps, counters, moments,
        two_qkv, two_up});
}
