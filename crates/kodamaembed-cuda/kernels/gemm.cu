// The encoder's GEMMs as gemm_bias_f16_<NAME>, gemm_bias_gelu_f16_<NAME> and
// gemm_bias_residual_f32_<NAME>: gemm.cuh's r<rows>c<columns> tilings in all three epilogues, the
// g<rows>c<columns> ones in the epilogue of their projection, all writing identical bits (see
// gemm.cuh). Grid of an r tiling: up to ceil(M / BM) * ceil(N / BN) blocks along x; of a g tiling:
// ceil(M / BM) blocks along x by ceil(N / BN) along y. Block: TILING::THREADS. Dynamic shared:
// TILING::SHARED_BYTES. low: [M, low_columns] with low_columns a multiple of eight and at most N,
// or null with 0.
#include "gemm.cuh"

#define TILING(NAME) DEFINE_TILING(NAME, (gemm::NAME::THREADS))
// A one-block minimum keeps these tiles' state in registers on Ampere, Ada and Blackwell.
#define TILING_MIN_ONE(NAME) DEFINE_TILING(NAME, (gemm::NAME::THREADS, 1))
#define DEFINE_TILING(NAME, BOUNDS)                                                               \
    extern "C" __global__ void __launch_bounds__ BOUNDS                                           \
        gemm_bias_f16_##NAME(const __half* a, const __half* w, const float* bias, __half* out,     \
                             __half* low, int low_columns, int M, int N, int K) {                 \
        extern __shared__ __align__(128) __half shared[];                                         \
        gemm::run<gemm::NAME, gemm::BIAS_SPLIT_F16>(a, w, bias, nullptr, out, low, low_columns,   \
                                                    M, N, K, shared);                             \
    }                                                                                             \
    extern "C" __global__ void __launch_bounds__ BOUNDS                                           \
        gemm_bias_gelu_f16_##NAME(const __half* a, const __half* w, const float* bias,            \
                                  __half* out, int M, int N, int K) {                             \
        extern __shared__ __align__(128) __half shared[];                                         \
        gemm::run<gemm::NAME, gemm::BIAS_GELU_F16>(a, w, bias, nullptr, out, nullptr, 0, M, N, K, \
                                                   shared);                                       \
    }                                                                                             \
    extern "C" __global__ void __launch_bounds__ BOUNDS                                           \
        gemm_bias_residual_f32_##NAME(const __half* a, const __half* w, const float* bias,        \
                                      const float* residual, float* out, int M, int N, int K) {   \
        extern __shared__ __align__(128) __half shared[];                                         \
        gemm::run<gemm::NAME, gemm::BIAS_RESIDUAL_F32>(a, w, bias, residual, out, nullptr, 0, M,  \
                                                       N, K, shared);                             \
    }

TILING_MIN_ONE(r16c32)
TILING(r16c64)
TILING(r32c32)
TILING(r48c32)
TILING_MIN_ONE(r48c64)
TILING(r48c128)
TILING(r80c64)
TILING(r80c128)
TILING(r96c128)
TILING_MIN_ONE(r64c64s3)
TILING(r64c64s4)
TILING(r64c128)
TILING(r128c64)
TILING(r128c128)
TILING(r128c128w8)
TILING(r256c128)
TILING(r64c128k64)
TILING(r128c128w8k64)
TILING(r128c88)
TILING(r128c176)

// One block a tile on a grid of row tiles by column tiles (gemm::run_grid) for the projections of
// a 1024-wide layer, at the row counts where that launch ran faster than blocks walking the tiles.
// The row tile grows with the rows, so a launch computes few rows that hold nothing.
#define GRID_BIAS(NAME, ...)                                                                      \
    using GridBias_##NAME = gemm::Tiling<__VA_ARGS__>;                                            \
    extern "C" __global__ void __launch_bounds__(GridBias_##NAME::THREADS)                        \
        gemm_bias_f16_##NAME(const __half* a, const __half* w, const float* bias, __half* out,     \
                             __half* low, int low_columns, int M, int N, int K) {                 \
        extern __shared__ __align__(128) __half shared[];                                         \
        gemm::run_grid<GridBias_##NAME, gemm::BIAS_SPLIT_F16>(a, w, bias, nullptr, out, low,      \
                                                              low_columns, M, N, K, shared);      \
    }
#define GRID_GELU(NAME, ...)                                                                      \
    using GridGelu_##NAME = gemm::Tiling<__VA_ARGS__>;                                            \
    extern "C" __global__ void __launch_bounds__(GridGelu_##NAME::THREADS)                        \
        gemm_bias_gelu_f16_##NAME(const __half* a, const __half* w, const float* bias,            \
                                  __half* out, int M, int N, int K) {                             \
        extern __shared__ __align__(128) __half shared[];                                         \
        gemm::run_grid<GridGelu_##NAME, gemm::BIAS_GELU_F16>(a, w, bias, nullptr, out, nullptr,   \
                                                             0, M, N, K, shared);                 \
    }
#define GRID_RESIDUAL(NAME, ...)                                                                  \
    using GridResidual_##NAME = gemm::Tiling<__VA_ARGS__>;                                        \
    extern "C" __global__ void __launch_bounds__(GridResidual_##NAME::THREADS)                    \
        gemm_bias_residual_f32_##NAME(const __half* a, const __half* w, const float* bias,        \
                                      const float* residual, float* out, int M, int N, int K) {   \
        extern __shared__ __align__(128) __half shared[];                                         \
        gemm::run_grid<GridResidual_##NAME, gemm::BIAS_RESIDUAL_F32>(a, w, bias, residual, out,   \
                                                                     nullptr, 0, M, N, K, shared); \
    }

// QKV at 129-512 rows: two row tiles by 24 column tiles, one wave of 48 blocks.
GRID_BIAS(g80c128, 80, 128, 32, 80, 32, 7)
GRID_BIAS(g96c128, 96, 128, 32, 96, 32, 4)
GRID_BIAS(g112c128, 112, 128, 64, 112, 16, 3)
GRID_BIAS(g128c128, 128, 128, 64, 64, 32, 3)
GRID_BIAS(g144c128, 144, 128, 32, 48, 32, 4)
GRID_BIAS(g160c128, 160, 128, 32, 80, 32, 4)
GRID_BIAS(g192c128, 192, 128, 32, 96, 32, 4)
GRID_BIAS(g256c128, 256, 128, 32, 64, 64, 4)
// The expansion at 193-512 rows: 44 to 48 blocks of eight warps, 96 at 385-512 rows. Two warps on
// a scheduler keep its tensor pipe busy through each other's loads.
GRID_GELU(g80c256, 80, 256, 32, 80, 32, 4)
GRID_GELU(g128c192, 128, 192, 32, 64, 48, 4)
GRID_GELU(g96c256, 96, 256, 32, 48, 64, 4)
GRID_GELU(g112c256, 112, 256, 32, 112, 32, 4)
GRID_GELU(g192c176, 192, 176, 32, 48, 88, 4)
GRID_GELU(g128c176, 128, 176, 32, 32, 88, 3)
// out at 193-320 rows and down at 193-288, four warps a block.
GRID_RESIDUAL(g112c64, 112, 64, 64, 112, 16, 3)
GRID_RESIDUAL(g64c128, 64, 128, 64, 32, 64, 3)
GRID_RESIDUAL(g96c64, 96, 64, 32, 48, 32, 4)
// down at 225-240 and 257-288 rows: three row tiles read the weights three times where tiles of
// 64 rows read them four and five times. Its 11 column tiles of 96 and 13 of 80 end in a tile
// that holds 64 columns; at 257-288 rows, where down is paced by the bytes its blocks pull
// through L2, 13 tiles read the activations 13 times where 64-wide tiles read them 16.
GRID_RESIDUAL(g80c96, 80, 96, 64, 80, 24, 4)
GRID_RESIDUAL(g96c80, 96, 80, 64, 48, 40, 4)
