// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Kernel microbenchmarks -> CSV on stdout.
//
// Shapes mirror a BitNet-2B layer (report section 6): decode = GEMV on
// 2560x2560 (q/o), 2560x6912 (down), 6912x2560 (gate/up); prefill = GEMM
// at M in {64, 256}. Plus a bandwidth probe kernel that measures the
// device's ACHIEVABLE read bandwidth — the roofline denominator.
//
// Measurement integrity: every timed dispatch (matrix rows AND --system
// sites) is first checked against the device's maxComputeWorkGroupCount
// and must reproduce the CPU oracle bit-exactly from a sentinel-filled
// output buffer. A row that fails either check is reported, never timed.
//
// CSV columns:
//   kernel,m,n,kwords,mode,iters,ms_avg,ms_min,ms_p90,bytes_moved,gb_per_s,ideal_ms
// Kernel rows carry all twelve. Probe rows (bw_probe, stream_bw_probe,
// alu_countbits_probe, null_probe) stop after gb_per_s: ideal_ms is a
// deployment-floor quantity for kernel rows and is undefined for a probe.
// ms_avg/ms_min/ms_p90 are real per-iteration statistics of the SAME
// sample set in both kinds of row - never one average printed three times.
// `# timing:` lines pair that host number with the GPU timestamp mean of the
// same dispatch (the submit + fence floor is host_avg - gpu_avg).

#include "backend/backend.hpp"
#include "backend/caps_conformance.hpp"
#include "backend/gpu_profiles.h"
#include "backend/kernel_family.hpp"
#include "backend/kernel_threads.h"
#include "backend/variant_selection.hpp"
#include "bench/bench_support.hpp"
#include "common/env.hpp"
#include "common/format.hpp"
#include "common/reference.hpp"
#include "common/tensor_gen.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef BITNET_SPV_DIR
#define BITNET_SPV_DIR "spv"
#endif

namespace {

// Forced variant override (testing): --variant CLI wins over the
// BITNET_FORCE_VARIANT env fallback; BITNET_FORCE_NO_DOT=1 is the legacy
// alias for dot-emulated. Empty = caps-driven.
std::string forced_variant_override() {
    std::string forced = bitnet::env_string("BITNET_FORCE_VARIANT");
    if (!forced.empty()) {
        return forced;
    }
    if (bitnet::env_string("BITNET_FORCE_NO_DOT") == "1") {
        return "dot-emulated";
    }
    return "";
}

struct BenchShape {
    const char* kernel;
    uint32_t m, n, k;
    uint32_t iters;
};

constexpr std::array<BenchShape, 45> kShapes = {{
    {.kernel = "gemv_bitplane_rowserial", .m = 1, .n = 2560, .k = 2560, .iters = 300},
    {.kernel = "gemv_bitplane_rowserial", .m = 1, .n = 2560, .k = 6912, .iters = 300},
    {.kernel = "gemv_bitplane_rowserial", .m = 1, .n = 6912, .k = 2560, .iters = 300},
    {.kernel = "gemm_bitplane_naive", .m = 64, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_bitplane_naive", .m = 256, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_bitplane_naive", .m = 64, .n = 6912, .k = 2560, .iters = 30},
    // ksplit (K-split + atomics): same shapes for the naive-vs-ksplit comparison
    {.kernel = "gemv_bitplane_ksplit", .m = 1, .n = 2560, .k = 2560, .iters = 300},
    {.kernel = "gemv_bitplane_ksplit", .m = 1, .n = 2560, .k = 6912, .iters = 300},
    {.kernel = "gemv_bitplane_ksplit", .m = 1, .n = 6912, .k = 2560, .iters = 300},
    {.kernel = "gemm_bitplane_ksplit", .m = 64, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_bitplane_ksplit", .m = 256, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_bitplane_ksplit", .m = 64, .n = 6912, .k = 2560, .iters = 30},
    // rowcoop/cellcoop (cooperative rows): same shapes for the generation comparison
    {.kernel = "gemv_bitplane_rowcoop", .m = 1, .n = 2560, .k = 2560, .iters = 300},
    {.kernel = "gemv_bitplane_rowcoop", .m = 1, .n = 2560, .k = 6912, .iters = 300},
    {.kernel = "gemv_bitplane_rowcoop", .m = 1, .n = 6912, .k = 2560, .iters = 300},
    {.kernel = "gemm_bitplane_cellcoop", .m = 64, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_bitplane_cellcoop", .m = 256, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_bitplane_cellcoop", .m = 64, .n = 6912, .k = 2560, .iters = 30},
    // tiled (shared-tile GEMM): same gemm shapes for the generation comparison
    {.kernel = "gemm_bitplane_tiled", .m = 64, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_bitplane_tiled", .m = 256, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_bitplane_tiled", .m = 64, .n = 6912, .k = 2560, .iters = 30},
    // dot (packed dot, core since Vulkan 1.3): same shapes, capability-gated
    {.kernel = "gemv_dot_rowcoop", .m = 1, .n = 2560, .k = 2560, .iters = 300},
    {.kernel = "gemv_dot_rowcoop", .m = 1, .n = 2560, .k = 6912, .iters = 300},
    {.kernel = "gemv_dot_rowcoop", .m = 1, .n = 6912, .k = 2560, .iters = 300},
    {.kernel = "gemm_dot_tiled", .m = 64, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_dot_tiled", .m = 256, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_dot_tiled", .m = 64, .n = 6912, .k = 2560, .iters = 30},
    // plane (plane-uint4 staging, format v3): same gemm shapes as tiled
    {.kernel = "gemm_plane_tiled", .m = 64, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_plane_tiled", .m = 256, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_plane_tiled", .m = 64, .n = 6912, .k = 2560, .iters = 30},
    // coop-mat MMA lane (VK_KHR_cooperative_matrix): capability-gated rows
    // over the SAME shapes as the gemm_plane_tiled block above (the A/B pair
    // the champion decision consumes). Skipped with a printed note when the
    // device lacks the enumerated sint8->sint32 workgroup shape.
    {.kernel = "gemm_coopmat", .m = 64, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_coopmat", .m = 256, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_coopmat", .m = 64, .n = 6912, .k = 2560, .iters = 30},
    // Fused decode shapes (fused qkv 3840, fused up+gate 13824) and
    // gemv_plane_rowcoop on all five decode shapes. The cooperative-row
    // champion candidates are measured on the SAME shape set - no
    // inference across kernels.
    {.kernel = "gemv_bitplane_rowcoop", .m = 1, .n = 3840, .k = 2560, .iters = 300},
    {.kernel = "gemv_bitplane_rowcoop", .m = 1, .n = 13824, .k = 2560, .iters = 300},
    {.kernel = "gemv_dot_rowcoop", .m = 1, .n = 3840, .k = 2560, .iters = 300},
    {.kernel = "gemv_dot_rowcoop", .m = 1, .n = 13824, .k = 2560, .iters = 300},
    {.kernel = "gemv_plane_rowcoop", .m = 1, .n = 2560, .k = 2560, .iters = 300},
    {.kernel = "gemv_plane_rowcoop", .m = 1, .n = 2560, .k = 6912, .iters = 300},
    {.kernel = "gemv_plane_rowcoop", .m = 1, .n = 6912, .k = 2560, .iters = 300},
    {.kernel = "gemv_plane_rowcoop", .m = 1, .n = 3840, .k = 2560, .iters = 300},
    {.kernel = "gemv_plane_rowcoop", .m = 1, .n = 13824, .k = 2560, .iters = 300},
    // microtile (register micro-tile over plane units): EXPERIMENTAL
    // set - a measured negative that can exceed the Mali fence
    // budget at full work size; opt-in only, kept at the END of the
    // matrix for the same reason.
    {.kernel = "gemm_plane_microtile", .m = 64, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_plane_microtile", .m = 256, .n = 2560, .k = 6912, .iters = 30},
    {.kernel = "gemm_plane_microtile", .m = 64, .n = 6912, .k = 2560, .iters = 30},
}};

// Per-generation packed buffers + the family's stride parameter. Packing
// is per layout family (format v2 dots / format v3 plane-uint4 / plane);
// extracted from bench_shape to keep its cognitive complexity in budget.
struct BenchBuffers {
    std::unique_ptr<bitnet::IBuffer> wbuf;
    std::unique_ptr<bitnet::IBuffer> abuf;
    uint32_t kwords = 0; // KernelParams.kwords: plane words, dot words or v6 units
    uint32_t mode = 0;   // KernelParams.mode: coopmat padded M, else 0
};

BenchBuffers pack_bench_buffers(bitnet::IComputeBackend& backend, const BenchShape& s,
                                const bitnet::TernaryTensor& t) {
    BenchBuffers out;
    const std::string_view kernel_sv(s.kernel);
    if (bitnet::is_coopmat_kernel(kernel_sv)) {
        // Coop-mat lane: plain padded int8 operands, activations K-major
        // (transposed; see format.hpp). kwords = padded K, mode = padded M.
        const auto pair = bitnet::PackedInt8MatPair::pack(
            t.weights.data(), t.activations.data(), s.m, s.n, s.k, bitnet::kCoopMatMSize,
            bitnet::kCoopMatNSize, bitnet::kCoopMatKSize, true);
        out.kwords = pair.paddedK;
        out.mode = pair.paddedM;
        out.wbuf = backend.create_buffer(pair.weightBytes(), true);
        out.abuf = backend.create_buffer(pair.actBytes(), true);
        std::memcpy(out.wbuf->mapped(), pair.weights.data(), pair.weightBytes());
        std::memcpy(out.abuf->mapped(), pair.acts.data(), pair.actBytes());
    } else if (bitnet::is_dot_kernel(kernel_sv)) {
        // dot packed-dot layouts (format v2): int8 values, 4 per word.
        const auto pwd = bitnet::PackedTernaryDots::pack(t.weights.data(), s.n, s.k);
        const auto pad = bitnet::PackedActivationsDots::pack(t.activations.data(), s.m, s.k);
        out.kwords = pwd.kwords;
        out.wbuf = backend.create_buffer(pwd.bytes(), true);
        out.abuf = backend.create_buffer(pad.bytes(), true);
        std::memcpy(out.wbuf->mapped(), pwd.bits.data(), pwd.bytes());
        std::memcpy(out.abuf->mapped(), pad.bits.data(), pad.bytes());
    } else if (bitnet::is_plane_kernel(kernel_sv)) {
        // plane layouts (format v3): 64-weight units + regrouped activation
        // planes; kwords = weight units per row. microtile[_image] consumes
        // the IDENTICAL format (its change is the compute mapping, register
        // micro-tile, not the layout).
        const auto pw6 = bitnet::PackedTernaryV6::pack(t.weights.data(), s.n, s.k);
        const auto pa6 = bitnet::PackedActivationsV6::pack(t.activations.data(), s.m, s.k);
        out.kwords = pw6.units;
        out.wbuf = backend.create_buffer(pw6.bytes(), true);
        out.abuf = backend.create_buffer(pa6.bytes(), true);
        std::memcpy(out.wbuf->mapped(), pw6.bits.data(), pw6.bytes());
        std::memcpy(out.abuf->mapped(), pa6.bits.data(), pa6.bytes());
    } else {
        const auto pw = bitnet::PackedTernary::pack(t.weights.data(), s.n, s.k);
        const auto pa = bitnet::PackedActivations::pack(t.activations.data(), s.m, s.k);
        // ksplit_il consumes the interleaved-16 layout (format.hpp
        // repack_interleaved16); everything else keeps the default layout.
        const auto pwx =
            (kernel_sv == "gemv_bitplane_ksplit_il") ? bitnet::repack_interleaved16(pw) : pw;
        out.kwords = pwx.kwords;
        out.wbuf = backend.create_buffer(pwx.bytes(), true);
        out.abuf = backend.create_buffer(pa.bytes(), true);
        std::memcpy(out.wbuf->mapped(), pwx.bits.data(), pwx.bytes());
        std::memcpy(out.abuf->mapped(), pa.bits.data(), pa.bytes());
    }
    return out;
}

// CPU oracle per bench shape: the matrix tensors are deterministic per
// (m, n, k, seed), so one oracle result serves every kernel row of a shape.
class OracleCache {
  public:
    const std::vector<int32_t>& get(const bitnet::TernaryTensor& t) {
        const std::array<uint32_t, 3> key{t.m, t.n, t.k};
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            std::vector<int32_t> ref(static_cast<size_t>(t.m) * t.n);
            bitnet::matmul_ref(t.weights.data(), t.activations.data(), ref.data(), t.m, t.n, t.k);
            it = cache_.emplace(key, std::move(ref)).first;
        }
        return it->second;
    }

  private:
    std::map<std::array<uint32_t, 3>, std::vector<int32_t>> cache_;
};

// Outcome of one matrix row: timed, or the reason it was NOT timed. A row
// is timed only after its dispatch (a) fits the device's group-count limits
// and (b) reproduced the CPU oracle bit-exactly.
enum class RowStatus : std::uint8_t { Timed, OutsideLimits, OracleMismatch };

struct RowResult {
    RowStatus status = RowStatus::Timed;
    bitnet::BenchStats stats{};
    std::array<uint32_t, 3> groups{};
};

RowResult bench_shape(bitnet::IComputeBackend& backend, const bitnet::GpuProfile& gpu,
                      const bitnet::DeviceCaps& caps, const BenchShape& s, uint32_t seed,
                      bitnet::DotVariant dot_variant, OracleCache& oracle) {
    const auto t = bitnet::make_tensor(s.m, s.n, s.k, false, seed);
    const std::string_view kernel_sv(s.kernel);
    const bool accumulates = is_ksplit_kernel(kernel_sv);

    const BenchBuffers bufs = pack_bench_buffers(backend, s, t);
    RowResult row;
    row.groups = bench_dispatch_groups(kernel_sv, gpu, s.m, s.n, bufs.kwords);
    if (!dispatch_within_limits(row.groups, caps)) {
        row.status = RowStatus::OutsideLimits;
        return row;
    }
    const size_t out_bytes = static_cast<size_t>(s.m) * s.n * sizeof(int32_t);
    auto obuf = backend.create_buffer(out_bytes, true);

    const uint32_t expect = bitnet::expected_local_size(s.kernel, gpu);
    // The row name stays the LOGICAL kernel; the module name carries the
    // capability-driven _emulated suffix for the dot family.
    auto* kernel = backend.load_kernel(
        {.name = bitnet::dot_module_name(s.kernel, dot_variant), .numthreads = {expect, 1, 1}});
    // mode: ksplit K-slice width, coop-mat padded M (bufs.mode), else 0.
    const bitnet::KernelParams params{
        .m = s.m, .n = s.n, .kwords = bufs.kwords, .mode = kernel_param_mode(kernel_sv, bufs.mode)};
    backend.bind(bufs.wbuf.get(), bufs.abuf.get(), obuf.get());
    backend.set_params(&params, sizeof(params));

    bitnet::IComputeBackend::Dispatch d;
    d.kernel = kernel;
    d.groups[0] = row.groups[0];
    d.groups[1] = row.groups[1];
    d.groups[2] = row.groups[2];

    // Work-was-done guard: ONE dispatch from a sentinel-filled output (zero
    // for the atomically accumulating ksplit kernels) must reproduce the CPU
    // oracle before anything is timed.
    std::memset(obuf->mapped(), accumulates ? 0 : kSentinelByte, out_bytes);
    backend.dispatch(d);
    backend.wait_idle();
    if (std::memcmp(oracle.get(t).data(), obuf->mapped(), out_bytes) != 0) {
        row.status = RowStatus::OracleMismatch;
        return row;
    }

    backend.dispatch_bench_stats(d, kWarmupIters);
    backend.wait_idle();
    row.stats = backend.dispatch_bench_stats(d, s.iters);
    return row;
}

bitnet::BenchStats bench_bw_probe(bitnet::IComputeBackend& backend, uint32_t elems,
                                  uint32_t total_threads, uint32_t iters) {
    auto src = backend.create_buffer(static_cast<uint64_t>(elems) * kProbeUnitBytes, true);
    auto sink = backend.create_buffer(static_cast<uint64_t>(total_threads) * sizeof(int32_t), true);
    auto* kernel = backend.load_kernel({.name = "bw_probe", .numthreads = {kProbeThreads, 1, 1}});
    const bitnet::KernelParams params{.m = 1U, .n = total_threads, .kwords = elems, .mode = 0U};
    backend.bind(src.get(), sink.get(),
                 nullptr); // bw_probe binding table: b0=Src, b1=Sink, b2 unused
    backend.set_params(&params, sizeof(params));

    bitnet::IComputeBackend::Dispatch d;
    d.kernel = kernel;
    d.groups[0] = total_threads / kProbeThreads;

    backend.dispatch_bench_stats(d, kWarmupIters);
    backend.wait_idle();
    return backend.dispatch_bench_stats(d, iters);
}

// Contiguous-stream probe: per-thread sequential runs -
// the strict streaming pattern, complementing the strided bw_probe.
bitnet::BenchStats bench_stream_probe(bitnet::IComputeBackend& backend, uint32_t elems,
                                      uint32_t total_threads, uint32_t iters) {
    auto src = backend.create_buffer(static_cast<uint64_t>(elems) * kProbeUnitBytes, true);
    auto sink = backend.create_buffer(static_cast<uint64_t>(total_threads) * sizeof(int32_t), true);
    auto* kernel =
        backend.load_kernel({.name = "stream_bw_probe", .numthreads = {kProbeThreads, 1, 1}});
    const bitnet::KernelParams params{.m = 1U, .n = total_threads, .kwords = elems, .mode = 0U};
    backend.bind(src.get(), sink.get(), nullptr);
    backend.set_params(&params, sizeof(params));
    bitnet::IComputeBackend::Dispatch d;
    d.kernel = kernel;
    d.groups[0] = total_threads / kProbeThreads;
    backend.dispatch_bench_stats(d, kWarmupIters);
    backend.wait_idle();
    const bitnet::BenchStats stats = backend.dispatch_bench_stats(d, iters);
    const double bytes = static_cast<double>(elems) * kProbeUnitBytes;

    // Host CPU cross-check: ONE thread reading the same 64 MB volume from
    // ordinary heap memory - the single-core share of the DRAM bandwidth the
    // GPU probe competes for. The checksum is printed and feeds the next
    // pass, so the loops cannot be eliminated or merged.
    std::vector<uint32_t> host_words(static_cast<size_t>(elems) * 4U);
    // Touch every page with non-zero, non-constant data.
    std::ranges::generate(host_words, [value = 0U]() mutable { return ++value; });
    std::array<double, kCpuProbePasses> cpu_pass_ms{};
    uint32_t cpu_xor = 0U;
    for (size_t pass = 0; pass < cpu_pass_ms.size(); ++pass) {
        host_words[pass] ^= cpu_xor; // each pass depends on the previous result
        const auto cpu_start = std::chrono::steady_clock::now();
        cpu_xor = std::accumulate(host_words.begin(), host_words.end(), cpu_xor,
                                  [](uint32_t acc, uint32_t word) { return acc ^ word; });
        cpu_pass_ms[pass] =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_start)
                .count();
    }
    const double cpu_avg_ms = std::accumulate(cpu_pass_ms.begin(), cpu_pass_ms.end(), 0.0) /
                              static_cast<double>(cpu_pass_ms.size());
    const auto [cpu_min_ms, cpu_max_ms] = std::ranges::minmax(cpu_pass_ms);

    // Columns as the kernel rows (avg, min, p90); GB/s from the average
    // pass, checksum last.
    std::printf("stream_bw_probe,1,%u,%u,0,%u,%.4f,%.4f,%.4f,%.0f,%.3f\n", total_threads, elems,
                iters, stats.avg_ms, stats.min_ms, stats.p90_ms, bytes,
                gb_per_s(bytes, stats.avg_ms));
    // CPU row: columns as the kernel rows (avg, min, then MAX in the p90
    // slot); GB/s from the average pass, checksum last.
    std::printf("cpu_memsum_64mb,1,%u,%u,0,%zu,%.4f,%.4f,%.4f,%.0f,%.3f,xor=%08x\n", total_threads,
                elems, cpu_pass_ms.size(), cpu_avg_ms, cpu_min_ms, cpu_max_ms, bytes,
                gb_per_s(bytes, cpu_avg_ms), cpu_xor);
    return stats;
}

// Vector-load isolation probe: runs BOTH access paths
// (uint4-typed and scalar uint loads) over the same known pattern and
// compares each against the CPU XOR - the execution probe that settles
// whether a driver compiles uint4 storage loads correctly (queries cannot).
// Returns 0 when both modes match the CPU, else the mismatch count.
uint32_t run_vec_load_probe(bitnet::IComputeBackend& backend) {
    constexpr uint32_t kUnits = 1024U;
    constexpr uint32_t kWordsPerUnit = 4U;
    constexpr uint32_t kSeedMixPrime = 2654435761U; // Knuth hash (tensor_gen contract)
    constexpr uint32_t kSeedMixBias = 0x9E3779B9U;  // golden-ratio constant
    constexpr uint32_t kVecProbeThreads = 256U;
    auto src = backend.create_buffer(
        static_cast<uint64_t>(kUnits) * kWordsPerUnit * bitnet::kWordBytes, true);
    auto* psrc = static_cast<uint32_t*>(src->mapped());
    for (uint32_t w = 0U; w < kUnits * kWordsPerUnit; ++w) {
        psrc[w] = w * kSeedMixPrime + kSeedMixBias;
    }
    auto out = backend.create_buffer(static_cast<uint64_t>(kUnits) * sizeof(uint32_t), true);
    auto* kernel =
        backend.load_kernel({.name = "vec_load_probe", .numthreads = {kVecProbeThreads, 1, 1}});
    const bitnet::KernelParams params{
        .m = kUnits, .n = kUnits, .kwords = kWordsPerUnit, .mode = 0U};
    backend.bind(src.get(), src.get(), out.get());
    backend.set_params(&params, sizeof(params));
    bitnet::IComputeBackend::Dispatch d;
    d.kernel = kernel;
    d.groups[0] = kUnits / kVecProbeThreads;
    uint32_t bad = 0U;
    for (const uint32_t mode : {0U, 1U}) {
        const auto p = params;
        const bitnet::KernelParams mparams{.m = p.m, .n = p.n, .kwords = p.kwords, .mode = mode};
        backend.set_params(&mparams, sizeof(mparams));
        backend.dispatch(d);
        backend.wait_idle();
        const auto* rout = static_cast<const uint32_t*>(out->mapped());
        uint32_t rout0 = 0U;
        uint32_t expect0 = 0U;
        for (uint32_t u = 0U; u < kUnits; ++u) {
            uint32_t expect = 0U;
            for (uint32_t lane = 0U; lane < 4U; ++lane) {
                expect ^= psrc[u * 4U + lane];
            }
            if (bad == 0U && rout[u] != expect) {
                rout0 = rout[u]; // first mismatch: report the actual GPU value
                expect0 = expect;
            }
            if (rout[u] != expect) {
                ++bad;
            }
        }
        std::printf("vec_load_probe_vector,%s bad=%u first_mismatch_gpu=%u cpu=%u\n",
                    bad == 0U ? "PASS" : "FAIL", bad, rout0, expect0);
    }
    return bad;
}

bitnet::BenchStats bench_fma_probe(bitnet::IComputeBackend& backend, uint32_t iters_per_thread,
                                   uint32_t total_threads, uint32_t iters) {
    auto sink = backend.create_buffer(static_cast<uint64_t>(total_threads) * sizeof(int32_t), true);
    auto* kernel =
        backend.load_kernel({.name = "alu_fma_probe", .numthreads = {kProbeThreads, 1, 1}});
    const bitnet::KernelParams params{
        .m = 1U, .n = total_threads, .kwords = iters_per_thread, .mode = 0U};
    backend.bind(nullptr, sink.get(), nullptr); // fma probe: b1 = Sink, b3 = params
    backend.set_params(&params, sizeof(params));

    bitnet::IComputeBackend::Dispatch d;
    d.kernel = kernel;
    d.groups[0] = total_threads / kProbeThreads;

    backend.dispatch_bench_stats(d, kWarmupIters);
    backend.wait_idle();
    return backend.dispatch_bench_stats(d, iters);
}

bitnet::BenchStats bench_alu_probe(bitnet::IComputeBackend& backend, uint32_t iters_per_thread,
                                   uint32_t total_threads, uint32_t iters) {
    auto sink = backend.create_buffer(static_cast<uint64_t>(total_threads) * sizeof(int32_t), true);
    auto* kernel =
        backend.load_kernel({.name = "alu_countbits_probe", .numthreads = {kProbeThreads, 1, 1}});
    const bitnet::KernelParams params{
        .m = 1U, .n = total_threads, .kwords = iters_per_thread, .mode = 0U};
    backend.bind(nullptr, sink.get(), nullptr); // alu probe: b1 = Sink, b3 = params
    backend.set_params(&params, sizeof(params));

    bitnet::IComputeBackend::Dispatch d;
    d.kernel = kernel;
    d.groups[0] = total_threads / kProbeThreads;

    backend.dispatch_bench_stats(d, kWarmupIters);
    backend.wait_idle();
    return backend.dispatch_bench_stats(d, iters);
}

// Coop-mat row gate: returns the skip-note text ("" = run the row). Opt-in
// experimental (known-broken) AND capability-gated;
// the printed note is the fall-through evidence (no coop-mat dispatch
// attempted on devices without the enumerated shape).
std::string coopmat_skip_note(const std::string& kernel, const bitnet::DeviceCaps& caps) {
    if (!bitnet::is_coopmat_kernel(kernel)) {
        return "";
    }
    if (!experiments_enabled()) {
        return " (experimental; BITNET_EXPERIMENTAL=1 to enable - KNOWN BROKEN on RTX 5070, "
               "see the report, section 5.3)";
    }
    if (!bitnet::coop_mat_shape_matches(caps)) {
        return " (no enumerated sint8*sint8->sint32 subgroup shape; gemm_plane_tiled serves)";
    }
    return "";
}

// Per-kernel stream-bytes model: dot int8
// layouts move 1 B/weight (n + m rows); bitplane kernels move 2 + 8
// bit-planes; plane moves uint4 units (16 B per weight unit + 64 B per act
// block); rowserial/ksplit single pass, rowcoop/cellcoop gemm re-reads per
// output row; the coop-mat lane moves PADDED 8-bit operands (4x the
// deployed weight bytes).
double stream_bytes_model(std::string_view kernel, uint32_t m, uint32_t n, uint32_t k) {
    const auto kw = static_cast<double>(bitnet::dots_kwords_for(k));
    const auto pkw = static_cast<double>(bitnet::kwords_for(k));
    const auto v6u = static_cast<double>(bitnet::v6_units_for(k));
    const auto w_row_dots = kw * static_cast<double>(bitnet::kWordBytes);
    const auto w_row_plane = pkw * static_cast<double>(bitnet::kWeightBitPlanes) *
                             static_cast<double>(bitnet::kWordBytes);
    const auto a_row_plane =
        pkw * static_cast<double>(bitnet::kActBitPlanes) * static_cast<double>(bitnet::kWordBytes);
    const auto w_row_v6 =
        v6u * static_cast<double>(bitnet::kV6UnitWords) * static_cast<double>(bitnet::kWordBytes);
    const auto a_row_v6 = v6u * static_cast<double>(bitnet::kV6ActWordsPerBlock) *
                          static_cast<double>(bitnet::kWordBytes);
    const auto out_bytes = static_cast<double>(m) * n * bitnet::kInt32Bytes;
    if (bitnet::is_dot_kernel(kernel)) {
        return (static_cast<double>(n) + static_cast<double>(m)) * w_row_dots + out_bytes;
    }
    if (bitnet::is_coopmat_kernel(kernel)) {
        const auto kPad = (static_cast<double>(k) + bitnet::kCoopMatKSize - 1.0) /
                          bitnet::kCoopMatKSize * bitnet::kCoopMatKSize;
        const auto nPad = (static_cast<double>(n) + bitnet::kCoopMatNSize - 1.0) /
                          bitnet::kCoopMatNSize * bitnet::kCoopMatNSize;
        const auto mPad = (static_cast<double>(m) + bitnet::kCoopMatMSize - 1.0) /
                          bitnet::kCoopMatMSize * bitnet::kCoopMatMSize;
        return nPad * kPad + mPad * kPad + out_bytes;
    }
    if (bitnet::is_plane_kernel(kernel)) {
        // plane and microtile both read format-v3 buffers in a single pass
        // (identical stream bytes; microtile only reorders the compute).
        return static_cast<double>(n) * w_row_v6 + static_cast<double>(m) * a_row_v6 + out_bytes;
    }
    if (m == 1) {
        return static_cast<double>(n) * w_row_plane + a_row_plane + out_bytes;
    }
    if (kernel.contains("cellcoop")) {
        return (static_cast<double>(m) * n * w_row_plane) + (static_cast<double>(m) * a_row_plane) +
               out_bytes;
    }
    return static_cast<double>(n) * w_row_plane + static_cast<double>(m) * a_row_plane + out_bytes;
}

// Static skip reasons for a matrix row (capability / experimental gates);
// empty = the row runs. Extracted from run_shape_matrix for the
// cognitive-complexity gate.
std::string row_skip_note(const BenchShape& s, const bitnet::DeviceCaps& caps,
                          bitnet::DotVariant dot_variant) {
    if (bitnet::is_dot_kernel(s.kernel) && dot_variant == bitnet::DotVariant::Popcount) {
        return " (--variant popcount)";
    }
    // Experimental set: the microtile family is a measured negative
    //; it benches only under BITNET_EXPERIMENTAL=1
    // so the default run stays within fence budgets on every device.
    if (bitnet::is_microtile_kernel(s.kernel) && !experiments_enabled()) {
        return " (experimental; BITNET_EXPERIMENTAL=1 to enable)";
    }
    return coopmat_skip_note(s.kernel, caps);
}

// Prints one matrix row. Returns false when the row failed oracle
// verification (the run then exits non-zero - a kernel that does not
// reproduce the oracle is never reported as a timing).
bool print_row(const BenchShape& shape, const RowResult& row, const bitnet::DeviceCaps& caps,
               double ceiling) {
    if (row.status == RowStatus::OutsideLimits) {
        std::printf("# %s m=%u n=%u k=%u: skipped (dispatch %s is outside "
                    "maxComputeWorkGroupCount %u/%u/%u)\n",
                    shape.kernel, shape.m, shape.n, shape.k, groups_text(row.groups).c_str(),
                    caps.max_workgroup_count[0], caps.max_workgroup_count[1],
                    caps.max_workgroup_count[2]);
        return true;
    }
    if (row.status == RowStatus::OracleMismatch) {
        std::printf("# %s m=%u n=%u k=%u: NOT TIMED - GPU output differs from the CPU oracle "
                    "(dispatch %s)\n",
                    shape.kernel, shape.m, shape.n, shape.k, groups_text(row.groups).c_str());
        return false;
    }
    const double bytes = stream_bytes_model(shape.kernel, shape.m, shape.n, shape.k);
    // Deployment floor per row: 2-bit weights (n*k/4 B) + int8 activations
    // (m*k B) + int32 outputs (m*n*4 B), divided by the measured ceiling.
    const auto ideal_bytes = static_cast<double>(shape.n) * shape.k / 4.0 +
                             static_cast<double>(shape.m) * shape.k +
                             static_cast<double>(shape.m) * shape.n * bitnet::kInt32Bytes;
    const double ideal_ms = ideal_bytes / kBytesPerGB / (ceiling / kMsPerSecond);
    std::printf("%s,%u,%u,%u,0,%u,%.4f,%.4f,%.4f,%.0f,%.3f,%.4f\n", shape.kernel, shape.m, shape.n,
                bitnet::kwords_for(shape.k), shape.iters, row.stats.avg_ms, row.stats.min_ms,
                row.stats.p90_ms, bytes, gb_per_s(bytes, row.stats.avg_ms), ideal_ms);
    return true;
}

// The kernel matrix: one CSV row per (shape, kernel), each verified against
// the CPU oracle before it is timed. Returns the number of rows that
// FAILED verification.
int run_shape_matrix(bitnet::IComputeBackend& backend, const bitnet::GpuProfile& gpu,
                     const bitnet::DeviceCaps& caps, bitnet::DotVariant dot_variant,
                     const std::string& only_kernel, uint32_t iters_override) {
    // Measured device ceiling for the ideal-time column:
    // ideal_ms = deployment-format bytes / ceiling.
    const double ceiling =
        gb_per_s(static_cast<double>(kProbeElements) * kProbeUnitBytes,
                 bench_bw_probe(backend, kProbeElements, kProbeTotalThreads, kProbeIters).avg_ms);
    std::printf("kernel,m,n,kwords,mode,iters,ms_avg,ms_min,ms_p90,bytes_moved,gb_per_s,"
                "ideal_ms\n");
    OracleCache oracle;
    int oracle_failures = 0;
    for (const BenchShape& s : kShapes) {
        if (!only_kernel.empty() && only_kernel != s.kernel) {
            continue;
        }
        const std::string skip = row_skip_note(s, caps, dot_variant);
        if (!skip.empty()) {
            std::printf("# %s: skipped%s\n", s.kernel, skip.c_str());
            continue;
        }
        BenchShape shape = s;
        if (iters_override > 0) {
            shape.iters = iters_override;
        }
        const RowResult row =
            bench_shape(backend, gpu, caps, shape, kSystemSeed, dot_variant, oracle);
        if (!print_row(shape, row, caps, ceiling)) {
            ++oracle_failures;
        }
    }
    return oracle_failures;
}

// Roofline probes: achievable bandwidth (bw_probe) and zero-traffic ALU
// cost of the countbits math (alu_countbits_probe). If the ALU probe
// dominates gemv time on a device, the kernel is ALU-bound (instruction
// reduction is the lever); if bw_probe dominates, it is memory-bound
// (tiling/coalescing).
// Null probe: one store per workgroup - the per-submit + fence floor of the
// dispatch_bench timing loop.
bitnet::BenchStats bench_null_probe(bitnet::IComputeBackend& backend, uint32_t groups,
                                    uint32_t iters) {
    auto sink = backend.create_buffer(static_cast<uint64_t>(groups) * sizeof(int32_t), true);
    auto* kernel =
        backend.load_kernel({.name = "null_probe", .numthreads = {kNullProbeThreads, 1, 1}});
    const bitnet::KernelParams params{.m = 1U, .n = groups, .kwords = 0U, .mode = 0U};
    backend.bind(nullptr, sink.get(), nullptr);
    backend.set_params(&params, sizeof(params));
    bitnet::IComputeBackend::Dispatch d;
    d.kernel = kernel;
    d.groups[0] = groups;
    backend.dispatch_bench_stats(d, kWarmupIters);
    backend.wait_idle();
    return backend.dispatch_bench_stats(d, iters);
}

// The timing probe verdict: two probes whose HOST times differ by over a
// millisecond (null_probe = one store per workgroup, bw_probe = a 64 MB
// stream) must show the same difference in their GPU timestamps. When they
// do not, the timestamp pair does not bracket the dispatch on this driver
// and the host wall clock over submit + fence is the number that stands.
// Same-difference thresholds of the timing verdict: the GPU delta has to
// land within [kTimingGpuRatioMin, kTimingGpuRatioMax] x the host delta,
// and only when the host delta is large enough to test against.
constexpr double kTimingMinHostDeltaMs = 0.05;
constexpr double kTimingGpuRatioMin = 0.5;
constexpr double kTimingGpuRatioMax = 2.0;

struct TimingVerdict {
    const char* state = "unavailable"; // on | rejected | unavailable
    double host_delta_ms = 0.0;
    double gpu_delta_ms = 0.0;
};

TimingVerdict timing_verdict(const bitnet::BenchStats& quiet, const bitnet::BenchStats& busy) {
    TimingVerdict verdict;
    if (quiet.gpu_ms <= 0.0 || busy.gpu_ms <= 0.0) {
        return verdict;
    }
    verdict.state = "on";
    verdict.host_delta_ms = busy.avg_ms - quiet.avg_ms;
    verdict.gpu_delta_ms = busy.gpu_ms - quiet.gpu_ms;
    const double host =
        verdict.host_delta_ms < 0.0 ? -verdict.host_delta_ms : verdict.host_delta_ms;
    const double gpu = verdict.gpu_delta_ms < 0.0 ? -verdict.gpu_delta_ms : verdict.gpu_delta_ms;
    // Too small a difference to test with => keep "on" (untested), not
    // "rejected": a test that proves nothing must not read as a failure.
    if (host >= kTimingMinHostDeltaMs &&
        (gpu < kTimingGpuRatioMin * host || gpu > kTimingGpuRatioMax * host)) {
        verdict.state = "rejected";
    }
    return verdict;
}

// The timing probe line: the SAME dispatch read twice - host wall clock
// around submit + fence, GPU timestamps inside that window. For null_probe
// the GPU work is one store per workgroup, so host_avg_ms - gpu_avg_ms is
// the submit + fence floor every kernel row carries (only meaningful while
// the verdict says the timestamps track the work).
void print_timing_line(const char* probe, const bitnet::BenchStats& stats, uint32_t iters,
                       const char* state) {
    if (stats.gpu_ms > 0.0) {
        std::printf("# timing: %s iters=%u host_avg_ms=%.4f host_min_ms=%.4f host_p90_ms=%.4f "
                    "gpu_avg_ms=%.4f submit_fence_ms=%.4f timestamps=%s\n",
                    probe, iters, stats.avg_ms, stats.min_ms, stats.p90_ms, stats.gpu_ms,
                    stats.avg_ms - stats.gpu_ms, state);
        return;
    }
    std::printf("# timing: %s iters=%u host_avg_ms=%.4f host_min_ms=%.4f host_p90_ms=%.4f "
                "gpu_avg_ms=0 submit_fence_ms=0 timestamps=unavailable\n",
                probe, iters, stats.avg_ms, stats.min_ms, stats.p90_ms);
}

void run_probes(bitnet::IComputeBackend& backend) {
    // Achievable read bandwidth: 64 MB of uint4, 16384 threads.
    const uint32_t elems = kProbeElements;
    const uint32_t threads = kProbeTotalThreads;
    const bitnet::BenchStats bw = bench_bw_probe(backend, elems, threads, kProbeIters);
    const double bytes = static_cast<double>(elems) * kProbeUnitBytes;
    std::printf("bw_probe,1,%u,%u,0,%u,%.4f,%.4f,%.4f,%.0f,%.3f\n", threads, elems, kProbeIters,
                bw.avg_ms, bw.min_ms, bw.p90_ms, bytes, gb_per_s(bytes, bw.avg_ms));

    // Contiguous-stream variant: the strict sequential pattern vs the
    // strided probe above - bounds the ceiling from both sides. The row is
    // printed once, by the probe itself (it owns the CPU cross-check row).
    const bitnet::BenchStats stream = bench_stream_probe(backend, elems, threads, kProbeIters);

    const uint32_t alu_iters = kAluProbeIters; // ~ matches gemv inner-op count
    const bitnet::BenchStats alu = bench_alu_probe(backend, alu_iters, threads, kProbeIters);
    std::printf("alu_countbits_probe,1,%u,%u,0,%u,%.4f,%.4f,%.4f,0,0.000\n", threads, alu_iters,
                kProbeIters, alu.avg_ms, alu.min_ms, alu.p90_ms);

    // FP32-FMA formulation of the same ternary arithmetic: one FMA per
    // weight against the AND + popcount integer pipeline above.
    const bitnet::BenchStats fma = bench_fma_probe(backend, alu_iters, threads, kProbeIters);
    std::printf("alu_fma_probe,1,%u,%u,0,%u,%.4f,%.4f,%.4f,0,0.000\n", threads, alu_iters,
                kProbeIters, fma.avg_ms, fma.min_ms, fma.p90_ms);

    // Vector-load isolation: settles uint4 storage-load
    // correctness per driver - the execution probe that property queries
    // cannot replace.
    const uint32_t vec_bad = run_vec_load_probe(backend);
    std::printf("vec_load_probe,1,1024,4,0,1,0.0000,0.0000,0,%s\n",
                vec_bad == 0U ? "PASS" : "FAIL");

    // Per-submit + fence floor: every kernel time carries this baseline.
    const bitnet::BenchStats nulls =
        bench_null_probe(backend, threads / kProbeThreads, kProbeIters);
    std::printf("null_probe,1,%u,0,0,%u,%.4f,%.4f,%.4f,0,0.000\n", threads / kProbeThreads,
                kProbeIters, nulls.avg_ms, nulls.min_ms, nulls.p90_ms);

    // Host vs GPU on the same dispatch, and whether the GPU side is worth
    // believing on THIS driver (the same-difference test above).
    const TimingVerdict verdict = timing_verdict(nulls, bw);
    std::printf("# timing: verdict=%s host_delta_ms=%.4f gpu_delta_ms=%.4f "
                "(null_probe vs bw_probe - quiet vs busy dispatch; host wall time stands "
                "either way)\n",
                verdict.state, verdict.host_delta_ms, verdict.gpu_delta_ms);
    print_timing_line("null_probe", nulls, kProbeIters, verdict.state);
    print_timing_line("bw_probe", bw, kProbeIters, verdict.state);
    print_timing_line("stream_bw_probe", stream, kProbeIters, verdict.state);
}

} // namespace

static int run_bench(int argc, char** argv);

int main(int argc, char** argv) {
    // Full-body catch: nothing may escape main (bugprone-exception-escape).
    try {
        return run_bench(argc, argv);
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    } catch (...) {
        std::printf("FAIL: unknown exception\n");
        return 1;
    }
}

// CLI options for run_bench, parsed separately to keep run_bench's
// cognitive complexity in budget (clang-tidy readability-function-cognitive-
// complexity).
struct BenchArgs {
    std::string spv_dir = BITNET_SPV_DIR;
    uint32_t iters_override = 0;
    bool skip_probe = false;
    std::string only_kernel;
    bool system_mode = false;
    bool fused_projections = false;
    bool decode_only = false;
    uint32_t system_layers = kSystemLayersDefault;
    unsigned tile_override_m = 0;
    unsigned tile_override_n = 0;
    unsigned tile_override_threads = 0;
    std::string variant; // forced capability variant (testing); empty = caps-driven
    // --system champion A/B overrides (exact logical kernel names; empty =
    // the profile's measured champions).
    std::string decode_champion;
    std::string prefill_champion;
};

// --decode-champion / --prefill-champion: parsed by their own scan so the
// main option parser's cognitive complexity stays in budget.
static void parse_champion_args(int argc, char** argv, BenchArgs& args) {
    for (int i = 1; i + 1 < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--decode-champion") {
            args.decode_champion = argv[i + 1];
        } else if (arg == "--prefill-champion") {
            args.prefill_champion = argv[i + 1];
        }
    }
}

// Log marker: a run through overridden champions must never be mistaken
// for a run through the profile table.
static const char* champion_source_note(const BenchArgs& args) {
    const bool overridden = !args.decode_champion.empty() || !args.prefill_champion.empty();
    return overridden ? " (A/B override - NOT the profile table)" : "";
}

BenchArgs parse_bench_args(int argc, char** argv) {
    BenchArgs args;
    parse_champion_args(argc, argv, args);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--spv-dir" && i + 1 < argc) {
            args.spv_dir = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            args.iters_override = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--skip-probe") {
            args.skip_probe = true;
        } else if (arg == "--system") {
            args.system_mode = true;
        } else if (arg == "--fused-projections") {
            args.fused_projections = true;
        } else if (arg == "--decode-only") {
            args.decode_only = true;
        } else if (arg == "--layers" && i + 1 < argc) {
            args.system_layers = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--only" && i + 1 < argc) {
            args.only_kernel = argv[++i];
        } else if (arg == "--tile-m" && i + 1 < argc) {
            args.tile_override_m = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--tile-n" && i + 1 < argc) {
            args.tile_override_n = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--tile-threads" && i + 1 < argc) {
            args.tile_override_threads = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--variant" && i + 1 < argc) {
            args.variant = argv[++i];
        }
    }
    return args;
}

// Graceful-exit contract helper: --system dispatches THROUGH the profile
// champions; an unfilled champion ("" until the per-device A/B fills it)
// would fail later on a bogus module load. Prints the rejection and
// returns false when --system cannot run on this profile (the caller
// exits 2 before any GPU work, like the knob gate above).
static bool system_champions_filled(const bitnet::GpuProfile& gpu) {
    if (std::strlen(gpu.gemv_champion) != 0 && std::strlen(gpu.gemm_champion) != 0) {
        return true;
    }
    std::printf("FATAL: --system needs champion kernels for profile '%s' "
                "(decode='%s' prefill='%s') - fill them from the per-device A/B bench matrix "
                "first\n",
                gpu.profile_dir, gpu.gemv_champion, gpu.gemm_champion);
    return false;
}

// Sweep-override application (--tile-m/--tile-n/--tile-threads probe knob
// combos; extracted from run_bench for the cognitive-complexity gate): the
// SPIR-V must be compiled with the SAME defines (CMakeLists sweep profiles).
static void apply_tile_overrides(bitnet::GpuProfile& gpu, const BenchArgs& args) {
    if (args.tile_override_m != 0U) {
        gpu.gemm_tile_m = args.tile_override_m;
    }
    if (args.tile_override_n != 0U) {
        gpu.gemm_tile_n = args.tile_override_n;
    }
    if (args.tile_override_threads != 0U) {
        gpu.gemm_tile_threads = args.tile_override_threads;
    }
}

// Champion overrides (--decode-champion / --prefill-champion): system-level
// A/B of champion CANDIDATES through the same --system pass, instead of
// inferring a system ranking from isolated bench rows. Exact logical kernel
// names of the right role only; anything else is rejected before any GPU
// work (exit 2). The overridden champion passes the same caps-conformance
// gate and the same oracle verification as a profile champion.
static bool apply_champion_overrides(bitnet::GpuProfile& gpu, const BenchArgs& args) {
    const auto valid = [](const std::string& name, bool want_gemv) {
        const auto* info = bitnet::find_kernel_info(name);
        return info != nullptr && info->family != bitnet::KernelFamily::Probe &&
               info->is_gemv == want_gemv;
    };
    if (!args.decode_champion.empty()) {
        if (!valid(args.decode_champion, true)) {
            std::printf("FATAL: --decode-champion '%s' is not a GEMV kernel name\n",
                        args.decode_champion.c_str());
            return false;
        }
        gpu.gemv_champion = args.decode_champion.c_str();
    }
    if (!args.prefill_champion.empty()) {
        if (!valid(args.prefill_champion, false)) {
            std::printf("FATAL: --prefill-champion '%s' is not a GEMM kernel name\n",
                        args.prefill_champion.c_str());
            return false;
        }
        gpu.gemm_champion = args.prefill_champion.c_str();
    }
    return true;
}

// Gate #2 evidence in the bench log itself: which validation state THIS run
// observed and how many messages the process counted. Printed only as a
// `#` line, so the CSV parsers keep reading rows, not comments.
static void print_validation_line(const bitnet::IComputeBackend& backend) {
    if (!backend.validation_requested()) {
        std::printf("# validation: OFF (BITNET_VALIDATION=0)\n");
        return;
    }
    if (!backend.validation_active()) {
        std::printf("# validation: OFF (requested, but the layer is not available on this "
                    "loader)\n");
        return;
    }
    std::printf("# validation: ON (messages=%llu)\n",
                static_cast<unsigned long long>(backend.validation_message_count()));
}

// Matrix mode: the per-kernel rows, then the roofline probes. Exit code 1
// when any row failed oracle verification (extracted from run_bench for the
// cognitive-complexity gate).
static int run_matrix_mode(bitnet::IComputeBackend& backend, const bitnet::GpuProfile& gpu,
                           const bitnet::DeviceCaps& caps, bitnet::DotVariant dot_variant,
                           const BenchArgs& args) {
    const int oracle_failures =
        run_shape_matrix(backend, gpu, caps, dot_variant, args.only_kernel, args.iters_override);
    if (!args.skip_probe) {
        run_probes(backend);
    }
    print_validation_line(backend);
    if (oracle_failures != 0) {
        std::printf("FAIL: %d bench row(s) did not reproduce the CPU oracle\n", oracle_failures);
        return 1;
    }
    return 0;
}

static int run_bench(int argc, char** argv) {
    const BenchArgs args = parse_bench_args(argc, argv);
    const std::string& spv_dir = args.spv_dir;
    // Forward to the backend's env-var discovery (portable wrapper).
    bitnet::set_env("BITNET_SPV_DIR", spv_dir.c_str());
    if (!std::filesystem::exists(spv_dir)) {
        std::printf("FAIL: spv dir not found: %s\n", spv_dir.c_str());
        return 1;
    }

    try {
        auto backend = bitnet::make_vulkan_backend();
        const auto caps = backend->probe_caps();
        // Per-GPU profile selection: point kernel discovery at the
        // compiled profile matching this device. Sweep overrides (--tile-m /
        // --tile-n) probe knob combos: the SPIR-V must be compiled with the
        // SAME defines (see CMakeLists.txt sweep profiles). All other host
        // dispatch knobs (v7 geometry, v6 rows-per-group) come ONLY from the
        // generated table - there is no host-side override for them.
        bitnet::GpuProfile gpu = *bitnet::gpu_profile_for_device(caps.device_name);
        apply_tile_overrides(gpu, args);
        if (!apply_champion_overrides(gpu, args)) {
            return 2;
        }
        // Knob validation gate (the code must never crash the
        // device): reject any profile exceeding advertised device limits
        // BEFORE touching the GPU. Exit code 2 = rejected config.
        std::string reject_reason;
        if (!bitnet::gpu_profile_valid(gpu, caps.max_workgroup_invocations,
                                       caps.max_shared_memory_bytes, &reject_reason)) {
            std::printf("REJECTED config: %s\n", reject_reason.c_str());
            return 2;
        }
        const std::string profile_dir = gpu.profile_dir;
        const std::string device_spv_dir = (std::filesystem::path(spv_dir) / profile_dir).string();
        bitnet::set_env("BITNET_SPV_DIR", device_spv_dir.c_str());
        std::printf("gpu profile: %s (spv: %s)\n", profile_dir.c_str(), device_spv_dir.c_str());
        std::printf("# champions: decode=%s prefill=%s%s\n", gpu.gemv_champion, gpu.gemm_champion,
                    champion_source_note(args));
        std::printf("# device: %s driver %s api %u.%u.%u shared_mem %u workgroup %u\n",
                    caps.device_name.c_str(), caps.driver_version.c_str(),
                    bitnet::version_major(caps.api_version),
                    bitnet::version_minor(caps.api_version),
                    bitnet::version_patch(caps.api_version), caps.max_shared_memory_bytes,
                    caps.max_workgroup_invocations);
        // Baseline requirement: Vulkan 1.0 (capability-ladder floor
        // kMinApi10 - the popcount path degrades, never skips); the packed
        // dot family is capability-gated (variant selection below).
        if (caps.api_version < bitnet::kMinApi10) {
            std::printf("UNAVAILABLE: Vulkan 1.0 required, device reports %u.%u.%u - nothing to "
                        "measure\n",
                        bitnet::version_major(caps.api_version),
                        bitnet::version_minor(caps.api_version),
                        bitnet::version_patch(caps.api_version));
            return 4;
        }
        // Caps conformance (machine-checked knob-vs-limit validation): a
        // violation fails the run before any GPU work.
        const bitnet::ConformanceResult conformance = bitnet::check_caps_conformance(caps, gpu);
        if (conformance.violations != 0) {
            std::printf("FATAL: %d caps-conformance violation(s) - the profile does not fit this "
                        "device\n",
                        conformance.violations);
            return 2;
        }

        // Capability-driven variant selection (variant_selection.hpp): caps
        // probe by default; a forced variant (--variant CLI > env) pins ONE
        // variant for the whole run and fails loudly when the device lacks
        // the required capability - no silent fallback.
        const std::string forced = !args.variant.empty() ? args.variant : forced_variant_override();
        bitnet::DotVariant dot_variant = bitnet::DotVariant::HardwareDot;
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
            dot_variant = parsed;
        } else {
            dot_variant = bitnet::auto_dot_variant(caps.shader_integer_dot_product);
        }
        const char* variant_source = " (forced)";
        if (forced.empty()) {
            variant_source = caps.shader_integer_dot_product
                                 ? " (caps)"
                                 : " (caps - no shaderIntegerDotProduct)";
        }
        std::printf("# variant: %s%s\n", bitnet::dot_variant_name(dot_variant), variant_source);
        if (dot_variant == bitnet::DotVariant::EmulatedDot) {
            std::printf("# emulated dot: dot rows measure the -DNO_INT_DOT modules - "
                        "emulated (slow); correctness lane, NO performance claims\n");
        }

        if (args.system_mode) {
            if (!system_champions_filled(gpu)) {
                return 2;
            }
            std::printf("# system: fused_projections=%u decode_only=%u\n",
                        static_cast<uint32_t>(args.fused_projections),
                        static_cast<uint32_t>(args.decode_only));
            run_system(*backend, gpu, caps, args.system_layers, kSystemDecodeTokens,
                       kSystemPrefillTokens, dot_variant, args.fused_projections, args.decode_only);
        } else {
            return run_matrix_mode(*backend, gpu, caps, dot_variant, args);
        }
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
