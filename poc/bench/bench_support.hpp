// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

#pragma once
// Internal API of the benchmark program (poc/bench): the constants and the
// workgroup geometry / family packing that both translation units need, the
// inline unit conversions, and the entry point of the whole-system pass.
//
// bench_support.cpp  dispatch geometry + family packing (one table per kernel)
// bench_system.cpp   the --system pass: sites, oracle guard, both regimes
// bench_main.cpp     kernel matrix, probes, CLI, main

#include "backend/backend.hpp"
#include "backend/gpu_profiles.h"
#include "backend/variant_selection.hpp"
#include "common/env.hpp"
#include "common/tensor_gen.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// ---------------------------------------------------------------------------
// Shape and size constants of the model the harness benches (report section 10)
// ---------------------------------------------------------------------------

// Whole-system pass: per-token cost of the REAL BitNet-2B transformer
// layer matrices -
// q 2560x2560, k 640x2560, v 640x2560, o 2560x2560, gate 6912x2560,
// up 6912x2560, down 2560x6912 (69.5 M weights per layer) - plus the
// output head 128256x2560 as its OWN measured row (it runs once per
// token, not once per layer). Decode runs each site at m=1 through
// the profile's champion GEMV kernel (deployable 2-bit gemv_plane_rowcoop on the
// Mali-G57 profile); prefill runs all seven matrices at m=64 through the
// champion GEMM. SCOPE: linear layers only - attention, norms, residuals,
// activation quantize/pack between layers are NOT included, so tokens per
// second here are an UPPER BOUND (labeled in the report). Weights: the
// layer-12 real shapes reused for every layer (timing-equivalent
// same-shape ops; documented PoC simplification). Two submission regimes
// per pass: sync (one submit + fence per call) vs batched (ONE submit +
// fence per token - the realistic engine shape).
inline constexpr uint32_t kSystemLayersDefault = 30;  // verified model depth (report section 10)
inline constexpr uint32_t kSystemDecodeTokens = 20;   // repeats for avg/min/p90
inline constexpr uint32_t kSystemPrefillTokens = 3;   // fewer: prefill passes are heavy
inline constexpr uint32_t kSystemSeed = 7U;           // same seed as the shape matrix
inline constexpr uint32_t kBitNetHidden = 2560;       // BitNet-2B hidden size (report section 6)
inline constexpr uint32_t kBitNetFfn = 6912;          // BitNet-2B FFN intermediate size
inline constexpr uint32_t kBitNetVocab = 128256;      // output head rows (tied embeddings)
inline constexpr uint32_t kBitNetKvHeadsDim = 640;    // 5 KV heads * 128 head dim
inline constexpr uint32_t kPrefillTokensPerCall = 64; // prefill token block (m)

inline constexpr double kBytesPerGB = 1e9;  // bytes moved -> GB/s denominator
inline constexpr double kMsPerSecond = 1e3; // ms -> seconds in the GB/s numerator

// ---------------------------------------------------------------------------
// Kernel contracts: group sizes and tiling constants mirror the [numthreads]
// literals of poc/slang/bitnet_kernels.slang; central here so the dispatch math
// and the kernels cannot drift apart.
// ---------------------------------------------------------------------------

// constexpr std::array: no static-storage-duration vector ctor (bugprone).
// Group-size / tiling constants: they MIRROR the kernel contracts in
// poc/slang/*.slang (v1: 64-thread groups and 16-wide tiles, v2: 256-thread
// slices; see also KernelParams docs). Central here so the harness dispatch
// math and the kernels cannot drift apart. Defined BEFORE the site factories
// and dispatch helpers that consume them.
inline constexpr uint32_t kV1GroupThreads = 64;        // v1 gemv/v2-gemm group rounding
inline constexpr uint32_t kV1GemmTile = 16;            // v1 gemm fixed tile
inline constexpr uint32_t kProbeThreads = 256;         // bw/alu probe group size
inline constexpr uint32_t kWarmupIters = 10;           // probe warmup iterations
inline constexpr uint32_t kProbeIters = 50;            // timed probe samples (avg/min/p90)
inline constexpr uint32_t kProbeUnitBytes = 16;        // bw_probe touches uint4 per element
inline constexpr uint32_t kV2GroupThreads = 256;       // v2 gemv slice-group rounding
inline constexpr uint32_t kProbeElements = 4'000'000U; // bw_probe: 64 MB of uint4
inline constexpr uint32_t kProbeTotalThreads = 16384U;
inline constexpr uint32_t kAluProbeIters = 1728U; // ~ matches gemv inner-op count
inline constexpr uint32_t kNullProbeThreads = 64; // null_probe: one 64-thread group
inline constexpr uint32_t kKSliceWords = 8;       // ksplit: kwords per K-slice (KernelParams.mode)
inline constexpr int kSentinelByte = 0x5A;   // site-verification output fill (never a valid sum)
inline constexpr size_t kCpuProbePasses = 5; // cpu_memsum_64mb: single-thread read passes

// Experimental-kernel opt-in: the microtile family benches only under
// BITNET_EXPERIMENTAL=1, mirroring poc_tests.
inline bool experiments_enabled() {
    return bitnet::env_nonempty("BITNET_EXPERIMENTAL");
}

inline double tok_per_s(double ms) {
    return ms > 0.0 ? kMsPerSecond / ms : 0.0;
}

// CSV bytes-moved/GB/s pair; the bytes model is per-kernel (see the
// stream-bytes model comment in run_shape_matrix).
inline double gb_per_s(double bytes, double ms) {
    return bytes / kBytesPerGB / (ms / kMsPerSecond);
}

// Packing is a per-kernel-family contract (poc/common/format.hpp): v6 kernels
// read the uint4 unit format, *_dot_* kernels the packed-dot format, the
// bitplane generations the 2/8-plane format. The champion fields name the
// kernel; the packer follows the family.
struct SiteParams {
    uint32_t kwords = 0; // v6: units; dots: dots-words; plane: ceil(k/32); coopmat: padded K
    uint32_t mode = 0;   // coopmat: padded M (transposed-act stride); else 0
    uint64_t wbytes = 0;
    uint64_t abytes = 0;
    std::vector<uint32_t> wbits;
    std::vector<uint32_t> abits;
};

// ---------------------------------------------------------------------------
// Definitions (bench_support.cpp, bench_system.cpp)
// ---------------------------------------------------------------------------

bool is_ksplit_kernel(std::string_view kernel);

uint32_t kernel_param_mode(std::string_view kernel, uint32_t packed_mode);

uint32_t ceil_div(uint32_t value, uint32_t divisor);

std::array<uint32_t, 3> bench_dispatch_groups(std::string_view kernel,
                                              const bitnet::GpuProfile& gpu, uint32_t m, uint32_t n,
                                              uint32_t kernel_kwords);

bool dispatch_within_limits(const std::array<uint32_t, 3>& groups, const bitnet::DeviceCaps& caps);

std::string groups_text(const std::array<uint32_t, 3>& groups);

SiteParams pack_site(const bitnet::TernaryTensor& t, uint32_t m, uint32_t n, uint32_t k,
                     std::string_view kernel);

// The whole-system pass (bench_system.cpp): per-token cost of the real
// layer matrices through the profile champions, decode and prefill.
void run_system(bitnet::IComputeBackend& backend, const bitnet::GpuProfile& gpu,
                const bitnet::DeviceCaps& caps, uint32_t layers, uint32_t decode_tokens,
                uint32_t prefill_tokens, bitnet::DotVariant dot_variant, bool fused_projections,
                bool decode_only);
