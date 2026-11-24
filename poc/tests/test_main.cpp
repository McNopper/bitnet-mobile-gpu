// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Bit-exact validation harness.
//
// Runs on Windows desktop and on the Android device (same binary, same
// seeds -> identical golden inputs on both). For every test shape:
//   1. generate deterministic ternary/binary weights + int8 activations
//   2. CPU oracle: naive definition matmul (reference.cpp)
//   3. GPU kernel: bit-plane popcount strategy via IComputeBackend
//   4. require int32 outputs to be EXACTLY equal (no tolerance)
// Also validates the packing roundtrip, and dumps device caps.

#include "backend/backend.hpp"
#include "backend/caps_conformance.hpp"
#include "backend/extension_families.hpp"
#include "backend/gpu_profiles.h"
#include "backend/kernel_family.hpp"
#include "backend/kernel_threads.h"
#include "backend/variant_selection.hpp"
#include "common/bitnet1.hpp"
#include "common/env.hpp"
#include "common/format.hpp"
#include "common/reference.hpp"
#include "common/tensor_gen.hpp"
#include "tests/suites.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef BITNET_SPV_DIR
#define BITNET_SPV_DIR "spv"
#endif

#ifndef BITNET_GOLDEN_PATH
#define BITNET_GOLDEN_PATH "data/golden_vector.txt"
#endif

// Forced variant override (testing, capability-agnostic): --variant CLI
// wins over the BITNET_FORCE_VARIANT env fallback; BITNET_FORCE_NO_DOT=1
// is the legacy alias for the emulated-dot variant. Empty = caps-driven
// selection (variant_selection.hpp).
static std::string forced_variant_override() {
    std::string forced = bitnet::env_string("BITNET_FORCE_VARIANT");
    if (!forced.empty()) {
        return forced;
    }
    if (bitnet::env_string("BITNET_FORCE_NO_DOT") == "1") {
        return "dot-emulated";
    }
    return "";
}

// The variant selection for THIS run, set once in run_tests after the caps
// probe (the harness is single-threaded). Globals so every per-shape path
// (compare_tensor overloads, golden and real-model lanes) sees the same
// selection without threading a new parameter through every call site.
bitnet::DotVariant g_dot_variant = bitnet::DotVariant::HardwareDot;
bool g_dot_lane_active = true;

namespace {

// constexpr std::array: no static-storage-duration vector ctor (bugprone).
// Group-size constants mirroring the kernel contracts (see bench_main and
// poc/slang/bitnet_kernels.slang): rowserial 64-thread groups, ksplit
// 256-thread slices, rowserial gemm
// 16-wide tiles.
constexpr uint32_t kV1GroupThreads = 64;
constexpr uint32_t kV2GroupThreads = 256;
constexpr uint32_t kV1GemmTile = 16;
constexpr uint32_t kDescThreads = 64; // load_kernel numthreads description
constexpr int32_t kMaxMismatchPrints = 5;

constexpr std::array<Shape, 20> kShapes = {{
    // gemv (decode path, M=1)
    {.m = 1, .n = 2560, .k = 2560, .binary = false},
    {.m = 1, .n = 2560, .k = 6912, .binary = false},
    {.m = 1, .n = 6912, .k = 2560, .binary = false},
    {.m = 1, .n = 257, .k = 259, .binary = false}, // odd, non-multiple-of-tile
    {.m = 1, .n = 32, .k = 33, .binary = false},   // minimal
    // gemm (prefill path)
    {.m = 4, .n = 2560, .k = 2560, .binary = false},
    {.m = 8, .n = 257, .k = 513, .binary = false},
    {.m = 16, .n = 2560, .k = 6912, .binary = false},
    // binary (1-bit) coverage: weights {0,1}, plane1 naturally zero
    {.m = 1, .n = 2560, .k = 2560, .binary = true},
    {.m = 4, .n = 257, .k = 259, .binary = true},
    // Edge shapes: the m-edge paths of the tile kernels, k
    // below one word, n=1, and m that no TILE_M divides.
    {.m = 1, .n = 1, .k = 1, .binary = false},      // single weight
    {.m = 1, .n = 5, .k = 31, .binary = false},     // k < 32 (sub-word padding)
    {.m = 3, .n = 65, .k = 31, .binary = false},    // odd m, odd n, k=31
    {.m = 5, .n = 1, .k = 31, .binary = false},     // n=1 GEMM column
    {.m = 7, .n = 33, .k = 1, .binary = false},     // k=1 minimal reduction
    {.m = 65, .n = 257, .k = 513, .binary = false}, // m not a TILE_M multiple
    // Extreme-value vectors: random data never reaches these.
    {.m = 1, .n = 64, .k = 128, .binary = false, .data = Shape::Data::Extreme},
    {.m = 4, .n = 64, .k = 128, .binary = false, .data = Shape::Data::Extreme},
    {.m = 4, .n = 32, .k = 96, .binary = false, .data = Shape::Data::Zeros},
    // The {-1,+1} "1-bit binary" alphabet: BitNet/XNOR-Net
    // encoding, no zeros, plain ternary mode.
    {.m = 4, .n = 65, .k = 257, .binary = false, .data = Shape::Data::SignedBinary},
}};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (arg == "--spv-dir") {
            {
                a.spv_dir = next();
            }
        } else if (arg == "--cpu-only") {
            {
                a.cpu_only = true;
            }
        } else if (arg == "--seed") {
            {
                a.seed = static_cast<uint32_t>(std::stoul(next()));
            }
        } else if (arg == "--caps-out") {
            {
                a.caps_out = next();
            }
        } else if (arg == "--tensors") {
            a.tensors = next();
        } else if (arg == "--golden") {
            a.golden = next();
        } else if (arg == "--variant") {
            a.variant = next();
        } else {
            std::printf("unknown arg: %s\n", arg.c_str());
            std::printf(
                "usage: poc_tests [--spv-dir DIR] [--cpu-only] [--seed N] [--caps-out FILE] "
                "[--tensors FILE] [--golden FILE]\n");
            std::exit(2);
        }
    }
    // The backend discovers SPIR-V via BITNET_SPV_DIR; forward the flag there.
    bitnet::set_env("BITNET_SPV_DIR", a.spv_dir.c_str());
    return a;
}

} // namespace

// Prepares the dot-family packed-dot buffers for one shape (format v2). Returns
// false when a pack roundtrip fails (already reported); leaves the output
// parameters untouched on failure. Extracted from compare_tensor (clang-tidy
// cognitive complexity; the dot-family gating stays at the call site).
static bool prepare_v5_buffers(const bitnet::TernaryTensor& t, uint32_t m, uint32_t n, uint32_t k,
                               bitnet::IComputeBackend& backend, bitnet::PackedTernaryDots& dw,
                               bitnet::PackedActivationsDots& da,
                               std::unique_ptr<bitnet::IBuffer>& wd5,
                               std::unique_ptr<bitnet::IBuffer>& ad5) {
    dw = bitnet::PackedTernaryDots::pack(t.weights.data(), n, k);
    if (dw.unpack() != t.weights) {
        std::printf("  PACK ROUNDTRIP FAILED (dots weights)\n");
        return false;
    }
    da = bitnet::PackedActivationsDots::pack(t.activations.data(), m, k);
    if (da.unpack() != t.activations) {
        std::printf("  PACK ROUNDTRIP FAILED (dots activations)\n");
        return false;
    }
    wd5 = backend.create_buffer(dw.bytes(), true);
    ad5 = backend.create_buffer(da.bytes(), true);
    std::memcpy(wd5->mapped(), dw.bits.data(), dw.bytes());
    std::memcpy(ad5->mapped(), da.bits.data(), da.bytes());
    return true;
}

// Kernel variants under test. ksplit accumulates atomically -> output must
// be zeroed first; its params.mode carries the slice length. Namespace
// scope so the per-shape construction helpers below can build them.

// Per-shape variant matrix construction (extracted from compare_tensor for
// clang-tidy cognitive complexity). kSliceWords: ksplit's params.mode slice.
struct ShapeVariants {
    Variant v1;
    Variant v2;
    Variant v3;
    Variant v4;
};

static ShapeVariants base_variants_for(const Shape& s, uint32_t kSliceWords,
                                       const bitnet::GpuProfile* gpu) {
    ShapeVariants out{};
    out.v1 = Variant{.name = (s.m == 1) ? "gemv_bitplane_rowserial" : "gemm_bitplane_naive",
                     .mode = s.binary ? 1U : 0U,
                     .kwords = 0U,
                     .needs_zero = false,
                     .groups = {0, 1, 1}};
    out.v2 = Variant{.name = (s.m == 1) ? "gemv_bitplane_ksplit" : "gemm_bitplane_ksplit",
                     .mode = kSliceWords,
                     .kwords = 0U,
                     .needs_zero = true,
                     .groups = {0, 1, 1}};
    out.v3 = Variant{.name = (s.m == 1) ? "gemv_bitplane_rowcoop" : "gemm_bitplane_cellcoop",
                     .mode = 0U,
                     .kwords = 0U,
                     .needs_zero = false,
                     .groups = {0, 1, 1}};
    // gemm_bitplane_tiled: one workgroup per TILE_M x TILE_N output tile,
    // one thread per output, accumulators in registers over the whole K
    // loop (gemm only; gemv keeps rowcoop cooperative rows).
    out.v4 = Variant{.name = "gemm_bitplane_tiled",
                     .mode = 0U,
                     .kwords = 0U,
                     .needs_zero = false,
                     .groups = {(s.n + gpu->gemm_tile_n - 1) / gpu->gemm_tile_n,
                                (s.m + gpu->gemm_tile_m - 1) / gpu->gemm_tile_m, 1}};
    if (s.m == 1) {
        out.v1.groups[0] = (s.n + kV1GroupThreads - 1) / kV1GroupThreads;
        out.v2.groups[0] = (s.n + kV2GroupThreads - 1) / kV2GroupThreads;
        out.v2.groups[2] = (bitnet::kwords_for(s.k) + kSliceWords - 1) / kSliceWords;
        out.v3.groups[0] = s.n;
    } else {
        out.v1.groups[0] = (s.n + kV1GemmTile - 1) / kV1GemmTile;
        out.v1.groups[1] = (s.m + kV1GemmTile - 1) / kV1GemmTile;
        out.v2.groups[0] = (s.n + kV1GroupThreads - 1) / kV1GroupThreads;
        out.v2.groups[1] = s.m;
        out.v2.groups[2] = (bitnet::kwords_for(s.k) + kSliceWords - 1) / kSliceWords;
        out.v3.groups[0] = s.m * s.n;
    }
    return out;
}

// gemv_dot_rowcoop/gemm_dot_tiled variant for one shape (gemv: one
// workgroup per output row; gemm: one workgroup per TILE_M x TILE_N
// tile). The HOST is authoritative for the row stride: kwords carries the
// packed-dot stride (ceil(K/4) words) and the kernel indexes with it in
// plain word arithmetic (a variant that derived the stride in the shader
// produced shifted rows on Mali-G57); params.mode is unused by the dot
// family.
static Variant v5_variant_for(const Shape& s, const bitnet::GpuProfile* gpu, uint32_t dots_kwords,
                              bitnet::DotVariant variant) {
    // The emulated variant loads the -DNO_INT_DOT modules (same entry
    // points, same LocalSize contract, bit-exact); the printed PASS note
    // marks the slow path so no performance claim can hide in it.
    const char* pass_note = variant == bitnet::DotVariant::EmulatedDot
                                ? "PASS (bit-exact, emulated dot - slow path)"
                                : nullptr;
    if (s.m == 1) {
        return Variant{.name = bitnet::dot_module_name("gemv_dot_rowcoop", variant),
                       .mode = 0U,
                       .kwords = dots_kwords,
                       .needs_zero = false,
                       .groups = {s.n, 1, 1},
                       .pass_note = pass_note};
    }
    return Variant{.name = bitnet::dot_module_name("gemm_dot_tiled", variant),
                   .mode = 0U,
                   .kwords = dots_kwords,
                   .needs_zero = false,
                   .groups = {(s.n + gpu->gemm_tile_n - 1) / gpu->gemm_tile_n,
                              (s.m + gpu->gemm_tile_m - 1) / gpu->gemm_tile_m, 1},
                   .pass_note = pass_note};
}

// Negative control: under-dispatch the reference
// kernel into a sentinel-filled output buffer. A correct harness MUST see
// mismatches here; zero mismatches means it is blind to unwritten outputs.
// Returns the mismatch count, or -1 after reporting a blind harness.
static int64_t run_negative_control(const Variant& ctrl, const bitnet::KernelParams& params,
                                    bitnet::IComputeBackend* backend, bitnet::IBuffer* wbuf,
                                    bitnet::IBuffer* abuf, bitnet::IBuffer* obuf,
                                    std::vector<int32_t>& gpu_out,
                                    const std::vector<int32_t>& ref) {
    std::memset(obuf->mapped(), 0x5A, gpu_out.size() * sizeof(int32_t));
    Variant half = ctrl;
    half.groups[0] /= 2U;
    if (half.groups[0] == 0U) {
        half.groups[0] = 1U;
    }
    auto* kernel = backend->load_kernel({.name = half.name, .numthreads = {kDescThreads, 1, 1}});
    backend->bind(wbuf, abuf, obuf);
    backend->set_params(&params, sizeof(params));
    bitnet::IComputeBackend::Dispatch d;
    d.kernel = kernel;
    d.groups[0] = half.groups[0];
    d.groups[1] = half.groups[1];
    d.groups[2] = half.groups[2];
    backend->dispatch(d);
    backend->wait_idle();
    std::memcpy(gpu_out.data(), obuf->mapped(), gpu_out.size() * sizeof(int32_t));
    int64_t ctrl_mismatches = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != gpu_out[i]) {
            ++ctrl_mismatches;
        }
    }
    if (ctrl_mismatches == 0) {
        std::printf("  NEGATIVE CONTROL FAILED: under-dispatched %s matched the "
                    "reference - this harness is blind to unwritten outputs\n",
                    ctrl.name.c_str());
        return -1;
    }
    return ctrl_mismatches;
}

// Once-per-process wrapper: runs the negative control on the first shape of
// each dispatch family (lane 0 = GEMV, lane 1 = GEMM) and
// remembers the verdict. Returns 0 (validated) or -1 (blind harness).
static int64_t run_negative_control_once(const Variant& ctrl, const bitnet::KernelParams& params,
                                         bitnet::IComputeBackend* backend, bitnet::IBuffer* wbuf,
                                         bitnet::IBuffer* abuf, bitnet::IBuffer* obuf,
                                         std::vector<int32_t>& gpu_out,
                                         const std::vector<int32_t>& ref, int lane) {
    static std::array<bool, 2> done{false, false};
    if (done[lane]) {
        return 0;
    }
    done[lane] = true;
    const int64_t mismatches =
        run_negative_control(ctrl, params, backend, wbuf, abuf, obuf, gpu_out, ref);
    if (mismatches == 0) {
        std::printf("  NEGATIVE CONTROL FAILED: under-dispatched %s matched the "
                    "reference - this harness is blind to unwritten outputs\n",
                    ctrl.name.c_str());
        return -1; // blind harness: caller must abort the shape
    }
    std::printf("  negative control (%s): %lld sentinel mismatches on half dispatch - instrument "
                "validated\n",
                lane == 0 ? "gemv lane" : "gemm lane", static_cast<long long>(mismatches));
    std::memset(obuf->mapped(), 0, gpu_out.size() * sizeof(int32_t));
    return 0; // instrument validated
}

// plane-uint4 buffers (rowcoop/tiled + microtile) and dispatch geometry:
// extracted from compare_tensor to keep its cognitive complexity in budget.
// microtile shares the rowcoop/tiled format v3 buffers.
static void prepare_uint4_kernels(const Shape& s, const bitnet::GpuProfile* gpu,
                                  const bitnet::TernaryTensor& t, bitnet::IComputeBackend& backend,
                                  Variant& v6, Variant& v7, std::unique_ptr<bitnet::IBuffer>& wbuf6,
                                  std::unique_ptr<bitnet::IBuffer>& abuf6) {
    if (s.m > 1) {
        v6.groups = {(s.n + gpu->gemm_tile_n - 1) / gpu->gemm_tile_n,
                     (s.m + gpu->gemm_tile_m - 1) / gpu->gemm_tile_m, 1};
        // microtile register geometry: from the generated profile table
        // (the same values CMake passes as GEMM_V7_* defines).
        const uint32_t tiles_x = (s.n + gpu->gemm_v7_cols - 1) / gpu->gemm_v7_cols;
        const uint32_t tiles_y = (s.m + gpu->gemm_v7_rows - 1) / gpu->gemm_v7_rows;
        const auto tiles = static_cast<uint32_t>(tiles_x * tiles_y);
        v7.groups = {(tiles + gpu->gemm_v7_threads - 1) / gpu->gemm_v7_threads, 1, 1};
    } else {
        v6.groups = {(s.n + gpu->gemv6_rows_per_group - 1) / gpu->gemv6_rows_per_group, 1, 1};
        v7.groups = {0, 1, 0}; // gemm-only kernel: not dispatched for gemv
    }
    const auto w6 = bitnet::PackedTernaryV6::pack(t.weights.data(), s.n, s.k);
    const auto a6 = bitnet::PackedActivationsV6::pack(t.activations.data(), s.m, s.k);
    wbuf6 = backend.create_buffer(w6.bytes(), true);
    abuf6 = backend.create_buffer(a6.bytes(), true);
    std::memcpy(wbuf6->mapped(), w6.bits.data(), w6.bytes());
    std::memcpy(abuf6->mapped(), a6.bits.data(), a6.bytes());
}

// Load, dispatch and bit-exact-compare ONE kernel variant (LocalSize
// guard: expected thread count per kernel). Extracted from compare_tensor
// to keep its cognitive complexity in budget. Returns the mismatch count.
int64_t run_and_compare_variant(const Variant& variant, const bitnet::GpuProfile* gpu,
                                bitnet::IComputeBackend& backend, bitnet::IBuffer* wb,
                                bitnet::IBuffer* ab, bitnet::IBuffer* out,
                                const bitnet::KernelParams& base, std::vector<int32_t>& gpu_out,
                                const std::vector<int32_t>& ref, void* binding_override) {
    const uint32_t expect = bitnet::expected_local_size(variant.name, *gpu);
    auto* kernel = backend.load_kernel({.name = variant.name, .numthreads = {expect, 1, 1}});
    const bitnet::KernelParams kernel_params{.m = base.m,
                                             .n = base.n,
                                             .kwords = variant.kwords != 0U ? variant.kwords
                                                                            : base.kwords,
                                             .mode = variant.mode};
    int64_t mismatches = 0;
    if (binding_override == nullptr) {
        // Sticky-bind path: one command buffer, one submit + fence.
        backend.bind(wb, ab, out);
        backend.set_params(&kernel_params, sizeof(kernel_params));
        if (variant.needs_zero) {
            std::memset(out->mapped(), 0, gpu_out.size() * sizeof(int32_t));
        }
        bitnet::IComputeBackend::Dispatch d;
        d.kernel = kernel;
        d.groups[0] = variant.groups[0];
        d.groups[1] = variant.groups[1];
        d.groups[2] = variant.groups[2];
        backend.dispatch(d);
        backend.wait_idle();
    } else {
        // Binding-set path (v7i texture weights): the set carries its own
        // params slot; one batched dispatch + fence.
        const bitnet::IComputeBackend::BatchedDispatch bd{
            .kernel = kernel,
            .binding = binding_override,
            .groups = {variant.groups[0], variant.groups[1], variant.groups[2]}};
        static_cast<void>(backend.dispatch_batch_stats({bd}, 1U));
    }
    std::memcpy(gpu_out.data(), out->mapped(), gpu_out.size() * sizeof(int32_t));
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != gpu_out[i]) {
            if (mismatches < kMaxMismatchPrints) {
                std::printf("  mismatch @[%zu]: gpu=%d ref=%d\n", i, gpu_out[i], ref[i]);
            }
            ++mismatches;
        }
    }
    const char* verdict = "FAIL";
    if (mismatches == 0) {
        verdict = variant.pass_note != nullptr ? variant.pass_note : "PASS (bit-exact)";
    }
    std::printf("  %s: %s\n", variant.name.c_str(), verdict);
    return mismatches;
}

// Extreme / zero / signed-binary tensors: deterministic vectors
// the seeded random generator cannot produce. Compare path identical to the
// random shapes (pack roundtrip, CPU oracle, every core kernel bit-exact).
static void fill_checkerboard_weights(bitnet::TernaryTensor& t, const Shape& s) {
    for (uint32_t r = 0; r < s.n; ++r) {
        for (uint32_t j = 0; j < s.k; ++j) {
            t.weights[static_cast<size_t>(r) * s.k + j] =
                ((r + j) % 2 == 0) ? int8_t{1} : int8_t{-1};
        }
    }
}

static void fill_extreme_activations(bitnet::TernaryTensor& t, const Shape& s) {
    for (uint32_t r = 0; r < s.m; ++r) {
        for (uint32_t j = 0; j < s.k; ++j) {
            t.activations[static_cast<size_t>(r) * s.k + j] =
                (j % 2 == 0) ? int8_t{127} : int8_t{-128};
        }
    }
}

static void fill_alternating_activations(bitnet::TernaryTensor& t, const Shape& s) {
    for (uint32_t r = 0; r < s.m; ++r) {
        for (uint32_t j = 0; j < s.k; ++j) {
            t.activations[static_cast<size_t>(r) * s.k + j] = (j % 2 == 0) ? int8_t{1} : int8_t{-1};
        }
    }
}

static bitnet::TernaryTensor make_special_tensor(const Shape& s) {
    bitnet::TernaryTensor t;
    t.m = s.m;
    t.n = s.n;
    t.k = s.k;
    t.binary = s.binary;
    t.weights.assign(static_cast<size_t>(s.n) * s.k, 0);
    t.activations.assign(static_cast<size_t>(s.m) * s.k, 0);
    // Zeros keeps both buffers zero (kernels must still write every output);
    // the others share the +1/-1 checkerboard weight alphabet.
    if (s.data == Shape::Data::Extreme || s.data == Shape::Data::SignedBinary) {
        fill_checkerboard_weights(t, s);
    }
    if (s.data == Shape::Data::Extreme) {
        fill_extreme_activations(t, s);
    }
    if (s.data == Shape::Data::SignedBinary) {
        fill_alternating_activations(t, s); // signs exercised, no accumulator extremes
    }
    return t;
}

static bitnet::TernaryTensor make_shape_tensor(const Shape& s, uint32_t seed) {
    return s.data == Shape::Data::Random ? bitnet::make_tensor(s.m, s.n, s.k, s.binary, seed)
                                         : make_special_tensor(s);
}

static int64_t compare_shape(const Shape& s, uint32_t seed, std::vector<int32_t>& gpu_out) {
    const auto t = make_shape_tensor(s, seed);
    return compare_tensor(s, t, gpu_out);
}

// Shared dispatch+compare core (random-tensor and real-weight paths).
int64_t compare_tensor(const Shape& s, const bitnet::TernaryTensor& t,
                       std::vector<int32_t>& gpu_out) {
    // Packing roundtrip check (guards the format contract itself).
    const auto packed_w = bitnet::PackedTernary::pack(t.weights.data(), s.n, s.k);
    if (packed_w.unpack() != t.weights) {
        std::printf("  PACK ROUNDTRIP FAILED (weights)\n");
        return -1;
    }
    const auto packed_a = bitnet::PackedActivations::pack(t.activations.data(), s.m, s.k);

    // CPU oracle: the definition.
    std::vector<int32_t> ref(static_cast<size_t>(s.m) * s.n);
    bitnet::matmul_ref(t.weights.data(), t.activations.data(), ref.data(), s.m, s.n, s.k);

    gpu_out.assign(static_cast<size_t>(s.m) * s.n, 0);

    auto backend = bitnet::make_vulkan_backend();
    const auto caps = backend->probe_caps();
    const auto* gpu = bitnet::gpu_profile_for_device(caps.device_name);
    std::printf("  device: %s (driver %s, api %u.%u.%u, profile %s)\n", caps.device_name.c_str(),
                caps.driver_version.c_str(), bitnet::version_major(caps.api_version),
                bitnet::version_minor(caps.api_version), bitnet::version_patch(caps.api_version),
                gpu->profile_dir);

    auto wbuf = backend->create_buffer(packed_w.bytes(), true);
    auto abuf = backend->create_buffer(packed_a.bytes(), true);
    auto obuf =
        backend->create_buffer(static_cast<uint64_t>(gpu_out.size()) * sizeof(int32_t), true);
    std::memcpy(wbuf->mapped(), packed_w.bits.data(), packed_w.bytes());
    std::memcpy(abuf->mapped(), packed_a.bits.data(), packed_a.bytes());

    const bitnet::KernelParams params{
        .m = s.m, .n = s.n, .kwords = bitnet::kwords_for(s.k), .mode = s.binary ? 1U : 0U};
    const uint32_t kSliceWords = 8; // ksplit: kwords per K-slice (bitnet_kernels.slang)

    // Kernel variants: constructed by base_variants_for (rowserial/ksplit/
    // rowcoop/tiled) and v5_variant_for (capability-gated packed-dot generation).
    const ShapeVariants sv = base_variants_for(s, kSliceWords, gpu);
    const auto& [v1, v2, v3, v4] = sv;

    // ksplit_il: the interleaved-16 weight layout (format.hpp
    // repack_interleaved16) for the coalesced Adreno path. Same dispatch
    // contract as ksplit, its own repacked weight buffer. GEMV only (the
    // il kernel reads one activation row, like gemv_bitplane_ksplit);
    // multi-token shapes keep the gemm variant below.
    Variant v2il = sv.v2;
    v2il.name = "gemv_bitplane_ksplit_il";
    const auto packed_il = bitnet::repack_interleaved16(packed_w);
    // The interleaved buffer pads rows to a multiple of 16: use the real
    // bit-vector size, not PackedTernary::bytes() (unpadded rows).
    const std::size_t il_bytes = packed_il.bits.size() * sizeof(uint32_t);
    auto wbuf_il = backend->create_buffer(il_bytes, true);
    std::memcpy(wbuf_il->mapped(), packed_il.bits.data(), il_bytes);

    // dot-family kernels: integer dot product is core since Vulkan 1.3
    // (docs.vulkan.org refpage VK_KHR_shader_integer_dot_product). The
    // capability-driven variant (variant_selection.hpp) picks the module:
    // hardware dot when the capability is present, the -DNO_INT_DOT
    // emulated modules otherwise - the lane always runs, slow but correct;
    // only --variant popcount skips it.
    Variant v5{.name = "", .mode = 0U, .kwords = 0U, .needs_zero = false, .groups = {0, 1, 1}};
    bitnet::PackedTernaryDots dots_w;
    bitnet::PackedActivationsDots dots_a;
    std::unique_ptr<bitnet::IBuffer> wd5;
    std::unique_ptr<bitnet::IBuffer> ad5;
    if (g_dot_lane_active &&
        prepare_v5_buffers(t, s.m, s.n, s.k, *backend, dots_w, dots_a, wd5, ad5)) {
        v5 = v5_variant_for(s, gpu, dots_w.kwords, g_dot_variant);
    }

    // plane-uint4 generations (format v3): gemv_plane_rowcoop /
    // gemm_plane_tiled and gemm_plane_microtile[_image] register micro-tile
    // (GEMM only, experimental). No capability gate: uint4 storage loads
    // are core, under the standard buffer layouts.
    Variant v6{.name = (s.m == 1) ? "gemv_plane_rowcoop" : "gemm_plane_tiled",
               .mode = 0U,
               .kwords = bitnet::v6_units_for(s.k),
               .needs_zero = false,
               .groups = {0, 1, 1}};
    Variant v7{.name = "gemm_plane_microtile",
               .mode = 0U,
               .kwords = bitnet::v6_units_for(s.k),
               .needs_zero = false,
               .groups = {0, 1, 1}};
    Variant v7i{.name = "gemm_plane_microtile_image",
                .mode = bitnet::kWeightImageWidthTexels,
                .kwords = bitnet::v6_units_for(s.k),
                .needs_zero = false,
                .groups = {0, 1, 1}};
    std::unique_ptr<bitnet::IBuffer> wbuf6;
    std::unique_ptr<bitnet::IBuffer> abuf6;
    prepare_uint4_kernels(s, gpu, t, *backend, v6, v7, wbuf6, abuf6);
    if (s.m > 1) {
        // gemm_plane_microtile_image shares gemm_plane_microtile's work
        // decomposition (same gemmPlaneMicrotileBody, only the weight
        // source differs), so it takes the same dispatch grid. With the
        // construction default {0,1,1} the dispatch would be a silent
        // no-op and the comparison would read the previous kernel's
        // output; validation cannot flag a 0-group dispatch.
        v7i.groups = v7.groups;
    }

    // Negative control (once per family): the harness must SEE unwritten
    // outputs - see run_negative_control_once above. Lane 0 (GEMV) fires
    // on the first shape, lane 1 (GEMM) on the first m>1 shape.
    if (int64_t nc = run_negative_control_once(sv.v1, params, backend.get(), wbuf.get(), abuf.get(),
                                               obuf.get(), gpu_out, ref, s.m > 1 ? 1 : 0);
        nc != 0) {
        return nc;
    }

    int64_t total_mismatches = 0;
    const auto run_variant = [&](const Variant& variant, bitnet::IBuffer* wb, bitnet::IBuffer* ab,
                                 void* binding_override = nullptr) {
        total_mismatches += run_and_compare_variant(variant, gpu, *backend, wb, ab, obuf.get(),
                                                    params, gpu_out, ref, binding_override);
    };
    for (const Variant& variant : {v1, v2, v3}) {
        run_variant(variant, wbuf.get(), abuf.get());
    }
    if (s.m == 1) {
        run_variant(v2il, wbuf_il.get(), abuf.get()); // il: gemv shapes only
    }
    if (s.m > 1) {
        run_variant(v4, wbuf.get(), abuf.get()); // shared-tile GEMM: gemm-only
        // The microtile family is the experimental set:
        // opt-in via BITNET_EXPERIMENTAL=1 so the default harness is the
        // core set that must pass everywhere.
        if (experiments_enabled()) {
            run_variant(v7, wbuf6.get(), abuf6.get()); // register micro-tile
        } else {
            std::printf("  gemm_plane_microtile: skipped (experimental; BITNET_EXPERIMENTAL=1 "
                        "to enable)\n");
        }
    }
    run_variant(v6, wbuf6.get(), abuf6.get()); // plane-uint4 generation (format v3)
    if (s.m > 1) {
        total_mismatches +=
            run_coopmat_lane(s, caps, t, gpu, *backend, obuf.get(), params, gpu_out, ref);
    }
    if (!experiments_enabled()) {
        std::printf("  gemm_plane_microtile_image: skipped (experimental; BITNET_EXPERIMENTAL=1 "
                    "to enable)\n");
    } else if (s.m > 1 && !s.binary) {
        // gemm_plane_microtile_image: weight stream through the 2D sampled
        // image. The image upload packs the format v3 weights.
        // The catch below is defense-in-depth: it reports and aborts the
        // shape; later shapes re-validate.
        const auto w6i = bitnet::PackedTernaryV6::pack(t.weights.data(), s.n, s.k);
        const bitnet::KernelParams v7i_params{
            .m = s.m, .n = s.n, .kwords = v7.kwords, .mode = bitnet::kWeightImageWidthTexels};
        void* image_binding = backend->make_image_bindings(
            w6i.bits.data(), w6i.bytes(), abuf6.get(), obuf.get(), &v7i_params, sizeof(v7i_params));
        try {
            run_variant(v7i, nullptr, nullptr, image_binding);
        } catch (const std::exception& e) {
            std::printf("  gemm_plane_microtile_image: EXCEPTION (%s) - device loss, shape "
                        "aborted\n",
                        e.what());
            total_mismatches += static_cast<int64_t>(ref.size());
        }
    } else {
        std::printf("  gemm_plane_microtile_image: skipped (gemv/binary shape)\n");
    }
    if (g_dot_lane_active) {
        run_variant(v5, wd5.get(), ad5.get()); // packed-dot generation (hardware or emulated)
    } else {
        std::printf("  gemv/gemm_dot_rowcoop/tiled: skipped (--variant popcount)\n");
    }
    return total_mismatches;
}

// Run every synthetic shape (CPU-only checksum or full GPU matrix).
// Extracted from run_tests (clang-tidy cognitive complexity).
static const char* shape_kind(const Shape& s) {
    if (s.data == Shape::Data::Extreme) {
        return "extreme";
    }
    if (s.data == Shape::Data::Zeros) {
        return "zeros";
    }
    if (s.data == Shape::Data::SignedBinary) {
        return "signed-binary";
    }
    return s.binary ? "binary" : "ternary";
}

static int run_shape_suite(const Args& args) {
    int failed = 0;
    for (const Shape& s : kShapes) {
        std::printf("[%s] m=%u n=%u k=%u %s\n", s.m == 1 ? "gemv" : "gemm", s.m, s.n, s.k,
                    shape_kind(s));
        try {
            if (args.cpu_only) {
                const auto t = make_shape_tensor(s, args.seed);
                std::vector<int32_t> ref(static_cast<size_t>(s.m) * s.n);
                bitnet::matmul_ref(t.weights.data(), t.activations.data(), ref.data(), s.m, s.n,
                                   s.k);
                // checksum smoke output so CPU runs are comparable across machines
                const int64_t sum = std::accumulate(ref.begin(), ref.end(), int64_t{0});
                std::printf("  cpu-only checksum: %lld\n", static_cast<long long>(sum));
            } else {
                std::vector<int32_t> gpu_out;
                const int64_t mm = compare_shape(s, args.seed, gpu_out);
                if (mm != 0) {
                    std::printf("  FAIL: %lld mismatches\n", static_cast<long long>(mm));
                    ++failed;
                } else {
                    std::printf("  PASS (bit-exact)\n");
                }
            }
        } catch (const std::exception& e) {
            std::printf("  FAIL: %s\n", e.what());
            ++failed;
        }
    }
    return failed;
}

static int run_tests(int argc, char** argv);

int main(int argc, char** argv) {
    // Full-body catch: nothing may escape main (bugprone-exception-escape),
    // and filesystem/allocator failures become readable FAIL lines.
    try {
        return run_tests(argc, argv);
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    } catch (...) {
        std::printf("FAIL: unknown exception\n");
        return 1;
    }
}

// Capability-driven variant selection for the run (variant_selection.hpp):
// caps probe by default, a forced variant (--variant CLI > env override)
// pins ONE variant for the whole run. Forcing a variant the device cannot
// run fails LOUDLY with the missing capability named - no silent fallback.
// Sets the g_dot_variant / g_dot_lane_active globals; returns the process
// exit code (0 = selection resolved).
static int resolve_dot_variant(const bitnet::DeviceCaps& caps, const std::string& variant_cli) {
    const std::string forced = !variant_cli.empty() ? variant_cli : forced_variant_override();
    if (!forced.empty()) {
        bitnet::DotVariant parsed = bitnet::DotVariant::HardwareDot;
        if (!bitnet::parse_dot_variant(forced, parsed)) {
            std::printf("FATAL: unknown --variant '%s' (accepted: popcount | dot | "
                        "dot-emulated)\n",
                        forced.c_str());
            return 2;
        }
        if (parsed == bitnet::DotVariant::HardwareDot && !caps.shader_integer_dot_product) {
            std::printf("FATAL: --variant dot needs shaderIntegerDotProduct (core 1.3 OR "
                        "VK_KHR_shader_integer_dot_product); device '%s' does not provide "
                        "it. Use dot-emulated (always available, slow path).\n",
                        caps.device_name.c_str());
            return 2;
        }
        g_dot_variant = parsed;
    } else {
        g_dot_variant = bitnet::auto_dot_variant(caps.shader_integer_dot_product);
    }
    g_dot_lane_active = g_dot_variant != bitnet::DotVariant::Popcount;
    const char* variant_source = " (forced)";
    if (forced.empty()) {
        variant_source =
            caps.shader_integer_dot_product ? " (caps)" : " (caps - no shaderIntegerDotProduct)";
    }
    std::printf("variant: %s%s\n", bitnet::dot_variant_name(g_dot_variant), variant_source);
    if (g_dot_variant == bitnet::DotVariant::EmulatedDot) {
        std::printf("  emulated dot: integer-dot family runs the -DNO_INT_DOT modules "
                    "(slow path, correctness only)\n");
    }
    return 0;
}

// Validation gate (machine-checked cleanliness): prints the definitive
// per-run state line and reports whether the run must FAIL. The debug
// callback counts every message process-wide (all backend instances share
// the counter); any message with validation ON fails the run. "ON" means
// the validation LAYER is loaded in the instance and its messenger exists;
// the OFF states print which of the two reasons applies (disabled by env,
// or the layer is not available to this process - the case on every
// Android device measured here) and cannot fail (no messages can arrive).
static bool report_validation_state() {
    auto gate_backend = bitnet::make_vulkan_backend();
    const bool active = gate_backend->validation_active();
    const uint64_t messages = gate_backend->validation_message_count();
    if (active) {
        if (messages == 0) {
            std::printf("validation: ON (0 messages)\n");
            return false;
        }
        std::printf("validation: ON (%llu messages - FAILING)\n",
                    static_cast<unsigned long long>(messages));
        return true;
    }
    if (!gate_backend->validation_requested()) {
        std::printf("validation: OFF (BITNET_VALIDATION=0)\n");
        return false;
    }
    std::printf("validation: OFF (requested, but the validation layer is not available to this "
                "process on this device; GPU results are NOT validation-checked)\n");
    return false;
}

static int run_tests(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    bitnet::LoadedWeights loaded; // real-model weights when --tensors is given
    const bool have_real = !args.tensors.empty() && bitnet::load_bitnet1(args.tensors, loaded);
    if (!args.tensors.empty() && !have_real) {
        std::printf("FAIL: cannot load BITNET1 container: %s\n", args.tensors.c_str());
        return 1;
    }
    std::printf("bitnet poc_tests: spv='%s' seed=%u cpu_only=%d\n", args.spv_dir.c_str(), args.seed,
                static_cast<int>(args.cpu_only));

    if (!std::filesystem::exists(args.spv_dir)) {
        std::printf("FAIL: spv dir not found: %s\n", args.spv_dir.c_str());
        return 1;
    }

    // Per-GPU profile selection: probe the device, then point the
    // backend's kernel discovery at the matching compiled profile.
    if (!args.cpu_only) {
        auto probe_backend = bitnet::make_vulkan_backend();
        const auto caps = probe_backend->probe_caps();
        print_extended_caps(caps);
        print_relevant_extensions(probe_backend->probe_extensions());
        const auto* gpu_profile = bitnet::gpu_profile_for_device(caps.device_name);
        const std::string device_spv_dir =
            (std::filesystem::path(args.spv_dir) / gpu_profile->profile_dir).string();
        bitnet::set_env("BITNET_SPV_DIR", device_spv_dir.c_str());
        std::printf("gpu profile: %s (spv: %s)\n", gpu_profile->profile_dir,
                    device_spv_dir.c_str());
        // Capability-ladder baseline: the popcount
        // path must run on Vulkan 1.0 (kMinApi10; the backend resolves its
        // 1.1 queries through VK_KHR_get_physical_device_properties2 there).
        // Integer dot is capability-gated (variant selection below).
        if (caps.api_version < bitnet::kMinApi10) {
            std::printf("UNAVAILABLE: Vulkan 1.0 required, device reports %u.%u.%u - GPU tests "
                        "cannot run\n",
                        bitnet::version_major(caps.api_version),
                        bitnet::version_minor(caps.api_version),
                        bitnet::version_patch(caps.api_version));
            return 4;
        }
        // Caps conformance (machine-checked knob-vs-limit validation):
        // every dispatched knob and the largest buffers against the
        // ADVERTISED device limits. A violation FAILS the run (a limit
        // breach is a dispatch-contract bug, never noise).
        const bitnet::ConformanceResult conformance =
            bitnet::check_caps_conformance(caps, *gpu_profile);
        if (conformance.violations != 0) {
            std::printf("FATAL: %d caps-conformance violation(s) - the profile does not fit this "
                        "device\n",
                        conformance.violations);
            return 2;
        }

        // Capability-driven variant selection (variant_selection.hpp): caps
        // probe by default; a forced variant (--variant CLI > env override)
        // pins ONE variant for the whole run. Forcing a variant the device
        // cannot run fails LOUDLY here - no silent fallback.
        const int variant_exit = resolve_dot_variant(caps, args.variant);
        if (variant_exit != 0) {
            return variant_exit;
        }
    }

    if (!args.caps_out.empty()) {
        auto caps_backend = bitnet::make_vulkan_backend();
        write_caps(args.caps_out.c_str(), caps_backend->probe_caps(),
                   caps_backend->probe_extensions());
    }

    int failed = 0;
    failed += run_shape_suite(args);

    // Known-answer golden vector: hand-computed values through
    // every core kernel, with the committed file cross-checked (GPU only).
    if (!args.cpu_only) {
        failed += run_golden_vector_test(args.golden);
    }

    // Real-model weights test (Path A): the shipped BITNET1 container holds a
    // real BitNet-2B layer; activations stay synthetic.
    if (have_real) {
        if (run_real_model_test(args, loaded) != 0) {
            ++failed;
        }
        // BitLinear forward: float -> quantize -> kernel ->
        // epilogue on the same real layer, float outputs exactly equal.
        if (run_bitlinear_test(loaded) != 0) {
            ++failed;
        }
    }

    // Validation gate: machine-checked cleanliness. The debug callback
    // counts every message process-wide (all backend instances share the
    // counter); any message with validation ON fails the run on top of the
    // stderr output. The two OFF states are printed as what they are
    // (disabled by the environment, or the layer is not available to this
    // process, as on the measured Android devices).
    if (!args.cpu_only && report_validation_state()) {
        ++failed;
    }

    std::printf("\n%s\n", failed == 0 ? "ALL TESTS PASSED" : "TESTS FAILED");
    return failed == 0 ? 0 : 1;
}
