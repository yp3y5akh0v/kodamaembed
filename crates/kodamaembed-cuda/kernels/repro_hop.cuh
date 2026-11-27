// Each compute warp has its own input ring to avoid a block-wide barrier per tile.

// A computing warp's input ring: stages of its 16 rows' 64-input tile.
#define HOP_STAGES 4
// A twelfth compute warp would leave 128 registers per thread where the kernel needs 170.
#define HOP_MAX_TILES 11
// A warp a row tile, then the warp that streams the weights.
#define HOP_THREADS ((HOP_MAX_TILES + 1) * 32)
// Limit live fragments and sums to the block's register budget.
#define HOP_PART_UNITS 5
// The ring's stages: the streaming warp runs this far ahead of the slowest row tile.
#define HOP_SLOTS 4
// The stages a block keeps in flight: fewer starve the memory, more delay the stage its
// lanes wait for.
#define HOP_AHEAD 3

// repro.rs hop_bytes mirrors this layout.
template <int K, int U>
struct HopPlan {
    static constexpr int KT = K / 64, PARTS = K / HALF, SLOT = U * 512, SLOTS = HOP_SLOTS;
    static constexpr int INPUTS = HOP_MAX_TILES * HOP_STAGES * 1024;
    static constexpr int STEPS = HOP_MAX_TILES * 16 * PARTS * 4;
    // A barrier a slot, then the streaming warp's last.
    static constexpr int BARS = ((SLOTS + 1) * 8 + 127) / 128 * 128;
    static constexpr int BYTES = SLOTS * SLOT + BARS + INPUTS + STEPS;
    static_assert(K % BLOCK == 0 && KT >= HOP_STAGES - 1 && SLOTS <= KT && BYTES <= STREAM_SHARED_LIMIT,
                  "the ring and the input rings fit");
};
static_assert(HopPlan<1024, 10>::BYTES == 76928, "repro.rs hop_bytes");

// The streaming warp meets a slot's readers on the slot's named barrier before overwriting it.
__device__ __forceinline__ void hop_slot_take(int slot, int readers) {
    __syncwarp();
    asm volatile("barrier.cta.sync %0, %1;" :: "r"(slot + 2), "r"(readers + 32) : "memory");
}

// A reader's last use of the slot's stage.
__device__ __forceinline__ void hop_slot_free(int slot, int readers) {
    asm volatile("barrier.cta.arrive %0, %1;" :: "r"(slot + 2), "r"(readers + 32) : "memory");
}

// Reuse a slot only after its readers release it. A separate wait bounds the number of copies in flight.
template <int K, int U>
__device__ __forceinline__ void hop_stream_weights(unsigned char* ring, unsigned long long* full,
                                                   unsigned long long* done, const signed char* q, int ldw,
                                                   StreamUnits share, int tiles) {
    using P = HopPlan<K, U>;
    static_assert(HOP_AHEAD >= 1 && HOP_AHEAD <= P::SLOTS, "a stage in flight keeps its slot");
    const int lane = threadIdx.x & 31;
    const unsigned long long weights = l2_evict_first();
    const unsigned target = shared_address(ring) + int8proj::tile_offset(lane >> 2, lane & 3);
    const signed char* source = q + (size_t)(8 * share.first + (lane >> 2)) * ldw + 16 * (lane & 3);
    for (int t = 0; t < P::KT; ++t) {
        const int slot = t % P::SLOTS;
        if (t >= P::SLOTS) hop_slot_take(slot, 32 * tiles);
        if (t >= HOP_AHEAD)
            stream_bar_wait(&full[(t - HOP_AHEAD) % P::SLOTS], (unsigned)(((t - HOP_AHEAD) / P::SLOTS) & 1));
#pragma unroll
        for (int u = 0; u < U; ++u)
            cp_async_16_policy(target + slot * P::SLOT + u * 512, source + (size_t)(u < share.count ? u : 0) * 8 * ldw + t * 64,
                               u < share.count ? 16u : 0u, weights);
        stream_cp_arrive(&full[slot]);
    }
    stream_cp_arrive(done);
    stream_bar_wait(done, 0);
}

// A tile's 16 rows' steps from part `first_part` on, a row's side by side; rows past `rows`
// take the last row's, keeping reads in bounds.
template <int PARTS>
__device__ __forceinline__ void hop_load_steps(float* steps, const float* row_steps, int first_part, int rows,
                                               int first) {
    const int lane = threadIdx.x & 31;
    for (int i = lane; i < 16 * PARTS; i += 32) {
        const int r = i / PARTS, c = i % PARTS;
        cp_async_4(shared_address(steps + r * PARTS + c),
                   repro_step_at(row_steps, rows, first_part + c, min(first + r, rows - 1)));
    }
}

template <int K, int U>
__device__ __forceinline__ void hop_products(float (&f)[U][4], const unsigned char* ring, unsigned long long* full,
                                             signed char* input, float* steps, const signed char* qa, int lda,
                                             const float* row_steps, int first_part, int rows, int tile) {
    using P = HopPlan<K, U>;
    const int lane = threadIdx.x & 31, g = lane >> 2, first = 16 * tile;
    const signed char* source[2];
    unsigned valid[2], target[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int r = first + g + 8 * j;
        source[j] = qa + (size_t)min(r, rows - 1) * lda + 16 * (lane & 3);
        valid[j] = r < rows ? 16u : 0u;
        target[j] = shared_address(input) + int8proj::tile_offset(g + 8 * j, lane & 3);
    }
    auto load = [&](int stage, int t) {
#pragma unroll
        for (int j = 0; j < 2; ++j) cp_async_16(target[j] + stage * 1024, source[j] + t * 64, valid[j]);
    };
    hop_load_steps<P::PARTS>(steps, row_steps, first_part, rows, first);
#pragma unroll
    for (int s = 0; s < HOP_STAGES - 1; ++s) {
        load(s, s);
        cp_async_commit();
    }
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
        for (int i = 0; i < 4; ++i) f[u][i] = 0.0f;
    const unsigned a_at[2] = {shared_address(input) + int8proj::tile_offset(lane & 15, lane >> 4),
                              shared_address(input) + int8proj::tile_offset(lane & 15, 2 + (lane >> 4))};
    const unsigned w_at = shared_address(ring) + int8proj::tile_offset(lane & 7, lane >> 3);
    for (int t = 0; t < P::KT; ++t) {
        const int slot = t % P::SLOTS, stage = t % HOP_STAGES;
        const unsigned parity = (unsigned)(t / P::SLOTS) & 1u;
        // The test voted across the warp, so the wait's branch is uniform.
        const bool landed = __all_sync(0xffffffffu, stream_bar_test(&full[slot], parity));
        cp_async_wait<HOP_STAGES - 2>();
        __syncwarp();
        if (t + HOP_STAGES - 1 < P::KT) load((t + HOP_STAGES - 1) % HOP_STAGES, t + HOP_STAGES - 1);
        cp_async_commit();
        unsigned a0[4], a1[4];
        ldmatrix_x4(a0, a_at[0] + stage * 1024);
        ldmatrix_x4(a1, a_at[1] + stage * 1024);
        const float step[2] = {steps[g * P::PARTS + t], steps[(g + 8) * P::PARTS + t]};
        if (!landed) stream_bar_wait(&full[slot], parity);
        const unsigned w = w_at + slot * P::SLOT;
#pragma unroll
        for (int u0 = 0; u0 < U; u0 += HOP_PART_UNITS) {
            constexpr int PART = U < HOP_PART_UNITS ? U : HOP_PART_UNITS;
            unsigned b[PART][4];
            int acc[PART][4];
#pragma unroll
            for (int p = 0; p < PART; ++p)
                if (u0 + p < U) ldmatrix_x4(b[p], w + (u0 + p) * 512);
#pragma unroll
            for (int p = 0; p < PART; ++p)
                if (u0 + p < U) {
                    const unsigned b0[2] = {b[p][0], b[p][1]};
                    int8proj::mma_s8_from(acc[p], a0, b0, int8proj::SUMS_FROM);
                }
#pragma unroll
            for (int p = 0; p < PART; ++p)
                if (u0 + p < U) {
                    const unsigned b1[2] = {b[p][2], b[p][3]};
                    int8proj::mma_s8(acc[p], a1, b1);
                }
#pragma unroll
            for (int p = 0; p < PART; ++p)
                if (u0 + p < U)
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        f[u0 + p][i] = __fmaf_rn(int8proj::promoted<14>(acc[p][i]), step[i >> 1], f[u0 + p][i]);
                    }
        }
        hop_slot_free(slot, 32 * ((rows + 15) / 16));
    }
    cp_async_wait<0>();
}

// Each segment's chain starts at +0, as the restored chains restart every h
// inputs, so repro_layernorm_quantize_joined joins the sums in order; a projection of one
// segment leaves its whole sums.
template <int K, int U, int SEGMENTS>
__device__ __forceinline__ void repro_hop_partials_impl(const signed char* qa, const float* row_steps,
                                                        const signed char* q, int rows, int n, float* partials) {
    using P = HopPlan<K, U>;
    constexpr int DEPTH = SEGMENTS * K;
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* ring = smem;
    auto* full = reinterpret_cast<unsigned long long*>(smem + P::SLOTS * P::SLOT);
    auto* done = full + P::SLOTS;
    auto* inputs = reinterpret_cast<signed char*>(smem + P::SLOTS * P::SLOT + P::BARS);
    auto* steps = reinterpret_cast<float*>(inputs + P::INPUTS);
    const int warp = __shfl_sync(0xffffffffu, threadIdx.x >> 5, 0), lane = threadIdx.x & 31;
    // The segments split the grid's blocks as stream_units splits a segment's units, so a
    // multiprocessor count of any remainder fills.
    const int tiles = (rows + 15) / 16;
    const int segment = SEGMENTS == 1 ? 0 : (SEGMENTS * blockIdx.x + SEGMENTS - 1) / gridDim.x;
    const int first = gridDim.x * segment / SEGMENTS, per = gridDim.x * (segment + 1) / SEGMENTS - first;
    const StreamUnits share = stream_units(n, blockIdx.x - first, per);
    if (threadIdx.x == 0) {
        unsigned bytes;
        asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(bytes));
        if (gridDim.x < SEGMENTS || tiles > HOP_MAX_TILES || (int)blockDim.x != 32 * (tiles + 1) || share.count > U ||
            bytes < (unsigned)P::BYTES)
            __trap();
        for (int s = 0; s < P::SLOTS; ++s) stream_bar_init(&full[s], 32);
        stream_bar_init(done, 32);
    }
    __syncthreads();
    if (warp == tiles) {
        hop_stream_weights<K, U>(ring, full, done, q + segment * K, DEPTH, share, tiles);
        return;
    }
    // Where the lane's sums go, formed before the matrix loop: one index through it, where
    // the segment and the share's first unit are two values and the wider shares have no
    // register for both.
    const unsigned at = (unsigned)(segment * rows * n + 8 * share.first + 2 * (lane & 3));
    float f[U][4];
    hop_products<K, U>(f, ring, full, inputs + warp * HOP_STAGES * 1024, steps + warp * 16 * P::PARTS,
                       qa + segment * K, DEPTH, row_steps, segment * P::PARTS, rows, warp);
    float* sums = partials + at;
#pragma unroll
    for (int u = 0; u < U; ++u) {
        if (u >= share.count) break;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = 16 * warp + (lane >> 2) + 8 * h;
            if (r < rows)
                *reinterpret_cast<float2*>(sums + (size_t)r * n + 8 * u) = make_float2(f[u][2 * h], f[u][2 * h + 1]);
        }
    }
}

// E5's 1,024-deep products by the most units a block takes, which repro.rs HOP_UNITS names:
// `partials`, the down projection's four segments; `sums`, a projection of one segment, for
// the pass that takes its rows whole (repro_gelu_half_blocks). With a resident-block minimum
// the shares past eleven units spill.
#define REPRO_HOP(NAME, SEGMENTS, UNITS, ...)                                                             \
    extern "C" __global__ void __launch_bounds__(__VA_ARGS__) repro_hop_##NAME##_1024_##UNITS(           \
        const signed char* qa, const float* row_steps, const signed char* q, int rows, int n,            \
        float* sums) {                                                                                   \
        repro_hop_partials_impl<1024, UNITS, SEGMENTS>(qa, row_steps, q, rows, n, sums);                 \
    }
#define REPRO_HOP_BELOW_9(NAME, SEGMENTS)                                                                 \
    REPRO_HOP(NAME, SEGMENTS, 4, HOP_THREADS, 1)                                                          \
    REPRO_HOP(NAME, SEGMENTS, 5, HOP_THREADS, 1)                                                          \
    REPRO_HOP(NAME, SEGMENTS, 6, HOP_THREADS, 1)                                                          \
    REPRO_HOP(NAME, SEGMENTS, 7, HOP_THREADS, 1)                                                          \
    REPRO_HOP(NAME, SEGMENTS, 8, HOP_THREADS, 1)
#define REPRO_HOP_FROM_9(NAME, SEGMENTS)                                                                  \
    REPRO_HOP(NAME, SEGMENTS, 9, HOP_THREADS, 1)                                                          \
    REPRO_HOP(NAME, SEGMENTS, 10, HOP_THREADS, 1)                                                         \
    REPRO_HOP(NAME, SEGMENTS, 11, HOP_THREADS, 1)                                                         \
    REPRO_HOP(NAME, SEGMENTS, 12, HOP_THREADS)                                                            \
    REPRO_HOP(NAME, SEGMENTS, 13, HOP_THREADS)                                                            \
    REPRO_HOP(NAME, SEGMENTS, 15, HOP_THREADS)                                                            \
    REPRO_HOP(NAME, SEGMENTS, 16, HOP_THREADS)
REPRO_HOP_BELOW_9(partials, 4)
REPRO_HOP_FROM_9(partials, 4)
// A share under nine units means 64 multiprocessors or more, where the blocks kernel takes
// the strips' rows in one wave (repro.rs up_blocks).
REPRO_HOP_FROM_9(sums, 1)
#undef REPRO_HOP_FROM_9
#undef REPRO_HOP_BELOW_9
#undef REPRO_HOP

// repro.rs GELU_PASS_WARPS and GELU_PASS_ROWS.
#define GELU_PASS_WARPS 8
#define GELU_PASS_ROWS 4

// Load all rows before applying the REPRO_GELU_BLOCKS arithmetic so their loads overlap.
extern "C" __global__ void __launch_bounds__(GELU_PASS_WARPS * 32, 3) repro_gelu_half_blocks(
    const float* __restrict__ sums, const float* __restrict__ col_scales, const float* __restrict__ static_shared,
    const int* __restrict__ row_shared, const float2* __restrict__ moments, const float* __restrict__ deferred_sums,
    const float* __restrict__ constants, const float* __restrict__ fc, int rows, int n,
    const float* __restrict__ next_centers, const float* __restrict__ next_signs, signed char* __restrict__ next_q,
    float* __restrict__ next_steps) {
    constexpr int R = GELU_PASS_ROWS;
    const int lane = threadIdx.x & 31, block = GELU_PASS_WARPS * blockIdx.x + (threadIdx.x >> 5);
    const int c = BLOCK * block + 4 * lane, first = R * blockIdx.y;
    if (c >= n) return;
    const bool defer = moments != nullptr;
    int code[R];
    float2 moment[R];
    float4 raw[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        // A row past the last takes the last one's values and stores nothing.
        const int r = min(first + i, rows - 1);
        code[i] = row_shared[r];
        moment[i] = defer ? moments[r] : make_float2(0.0f, 0.0f);
        raw[i] = *reinterpret_cast<const float4*>(sums + (size_t)r * n + c);
    }
    const float4 sw = __ldg(reinterpret_cast<const float4*>(col_scales + c));
    const float4 su = defer ? __ldg(reinterpret_cast<const float4*>(deferred_sums + c)) : make_float4(0, 0, 0, 0);
    const float4 co = defer ? __ldg(reinterpret_cast<const float4*>(constants + c)) : make_float4(0, 0, 0, 0);
    const ReproClassRows shared = repro_class_rows(static_shared, n, c);
    const ReproGelu gelu = repro_gelu_constants(fc);
    // The centers by class as the shared rows, where a row's own would wait for its code.
    const ReproClassRows centers = repro_class_rows(next_centers, n, c);
    const float4 sg = __ldg(reinterpret_cast<const float4*>(next_signs + c));
    const float rotate = fc[FC_RECIP_SQRT_BLOCK];
    const float scale[4] = {sw.x, sw.y, sw.z, sw.w}, sum[4] = {su.x, su.y, su.z, su.w};
    const float constant[4] = {co.x, co.y, co.z, co.w};
    float x[R][4];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const float4 fixed = repro_class_of(shared, code[i]);
        const float value[4] = {raw[i].x, raw[i].y, raw[i].z, raw[i].w};
        const float shared_row[4] = {fixed.x, fixed.y, fixed.z, fixed.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            x[i][j] = __fadd_rn(__fmul_rn(value[j], scale[j]), shared_row[j]);
            if (defer) x[i][j] = repro_deferred_value(x[i][j], moment[i], sum[j], constant[j]);
        }
    }
    repro_gelus(x, gelu);
    const float sign[4] = {sg.x < 0.0f ? -1.0f : 1.0f, sg.y < 0.0f ? -1.0f : 1.0f, sg.z < 0.0f ? -1.0f : 1.0f,
                           sg.w < 0.0f ? -1.0f : 1.0f};
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const float4 m = repro_class_of(centers, code[i]);
        const float center[4] = {m.x, m.y, m.z, m.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) x[i][j] = __fmul_rn(__fsub_rn(fp16_value(x[i][j]), center[j]), sign[j]);
    }
    float most[R], step[R], reciprocal[R];
    repro_rotate_rows<4, R>(x, lane, rotate, most);
#pragma unroll
    for (int i = 0; i < R; ++i) step[i] = most[i] > 0.0f ? __fdiv_rn(most[i], 127.0f) : 1.0f;
#pragma unroll
    for (int i = 0; i < R; ++i) reciprocal[i] = repro_step_reciprocal(step[i]);
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const int r = first + i;
        if (r >= rows) break;
        repro_store_block<4>(x[i], step[i], reciprocal[i], next_q + (size_t)r * n + c);
        if (lane % 16 == 0) *repro_step_at(next_steps, rows, 2 * block + lane / 16, r) = step[i];
    }
}
