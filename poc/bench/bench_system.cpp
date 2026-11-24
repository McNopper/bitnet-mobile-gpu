// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// The whole-system pass: one site per layer projection (plus the output
// head as its own row), every site verified against the CPU oracle before it
// is timed, then both submission regimes (sync and batched) reported.

#include "backend/kernel_threads.h"
#include "bench/bench_support.hpp"
#include "common/reference.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct SystemSite {
    const char* name = "";
    std::string kernel_name; // LOGICAL family name maps to the module name
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t calls_per_layer = 1; // sq site: q, k, v, o = 4
    std::unique_ptr<bitnet::IBuffer> wbuf;
    std::unique_ptr<bitnet::IBuffer> abuf;
    std::unique_ptr<bitnet::IBuffer> obuf;
    void* kernel = nullptr;
    void* binding = nullptr;
    bitnet::IComputeBackend::BatchedDispatch dispatch;
    bitnet::KernelParams params{}; // sticky-bind path (sync regime)
    bitnet::IComputeBackend::Dispatch sync_dispatch;
    double isolated_ms = 0.0;      // per-call avg, measured separately
    bool accumulates = false;      // ksplit: InterlockedAdd into a zeroed output
    std::vector<int32_t> expected; // CPU oracle for this site (verify_site)
};

// The real BitNet-2B layer projections (safetensors header shapes).
// n = output rows, k = reduction dim.
struct LayerShape {
    const char* name;
    uint32_t n;
    uint32_t k;
};

// One --system call site through the champion `kernel`. `kernel` (the
// LOGICAL champion name) drives family matching - packer selection,
// parameter block and grid; only the MODULE name handed to load_kernel
// carries the capability-driven _emulated suffix (dot_module_name). The CPU
// oracle for the site's deterministic tensor is computed here, while the
// unpacked tensor is alive; verify_site compares against it before any
// timing.
SystemSite make_site(bitnet::IComputeBackend& backend, const bitnet::GpuProfile& gpu,
                     const bitnet::DeviceCaps& caps, std::string_view champion,
                     const LayerShape& shape, uint32_t m, bitnet::DotVariant dot_variant) {
    const std::string kernel(champion);
    const uint32_t n = shape.n;
    const uint32_t k = shape.k;
    const auto t = bitnet::make_tensor(m, n, k, false, kSystemSeed);
    const SiteParams pk = pack_site(t, m, n, k, kernel);
    const auto groups = bench_dispatch_groups(kernel, gpu, m, n, pk.kwords);
    if (!dispatch_within_limits(groups, caps)) {
        throw std::runtime_error("dispatch " + groups_text(groups) + " of " + kernel +
                                 " is outside maxComputeWorkGroupCount");
    }
    SystemSite site;
    site.name = shape.name;
    site.kernel_name = bitnet::dot_module_name(kernel, dot_variant);
    site.n = n;
    site.k = k;
    site.accumulates = is_ksplit_kernel(kernel);
    site.expected.resize(static_cast<size_t>(m) * n);
    bitnet::matmul_ref(t.weights.data(), t.activations.data(), site.expected.data(), m, n, k);
    site.wbuf = backend.create_buffer(pk.wbytes, true);
    site.abuf = backend.create_buffer(pk.abytes, true);
    site.obuf = backend.create_buffer(static_cast<uint64_t>(m) * n * bitnet::kInt32Bytes, true);
    std::memcpy(site.wbuf->mapped(), pk.wbits.data(), pk.wbytes);
    std::memcpy(site.abuf->mapped(), pk.abits.data(), pk.abytes);
    const uint32_t expect = bitnet::expected_local_size(kernel, gpu);
    site.kernel = backend.load_kernel({.name = site.kernel_name, .numthreads = {expect, 1, 1}});
    const bitnet::KernelParams params{
        .m = m, .n = n, .kwords = pk.kwords, .mode = kernel_param_mode(kernel, pk.mode)};
    site.binding = backend.make_bindings(site.wbuf.get(), site.abuf.get(), site.obuf.get(), &params,
                                         sizeof(params));
    site.params = params;
    site.sync_dispatch = {.kernel = site.kernel, .groups = {groups[0], groups[1], groups[2]}};
    site.dispatch = {.kernel = site.kernel,
                     .binding = site.binding,
                     .groups = {groups[0], groups[1], groups[2]}};
    return site;
}

// Work-was-done guard: before a
// site is timed, ONE dispatch must reproduce the CPU oracle bit-exactly
// from a sentinel-filled output buffer (zero-filled for the atomically
// accumulating ksplit kernels - their documented contract). A kernel that
// returns early, an under-sized grid or a wrong parameter block fails the
// run here instead of being timed as if it were the workload.
void verify_site(bitnet::IComputeBackend& backend, const SystemSite& site) {
    const size_t bytes = site.expected.size() * sizeof(int32_t);
    std::memset(site.obuf->mapped(), site.accumulates ? 0 : kSentinelByte, bytes);
    static_cast<void>(backend.dispatch_batch_stats({site.dispatch}, 1U));
    if (std::memcmp(site.expected.data(), site.obuf->mapped(), bytes) != 0) {
        throw std::runtime_error("system site '" + std::string(site.name) + "' (" +
                                 site.kernel_name +
                                 "): GPU output differs from the CPU oracle - refusing to time "
                                 "a kernel that does not do the work");
    }
}

// Negative control for the work-was-done guard (once per --system run): a
// site dispatched with HALF its grid must be REJECTED by verify_site. A
// guard that cannot fail proves nothing; this line in the log is the
// evidence that it can.
void guard_negative_control(bitnet::IComputeBackend& backend, SystemSite& site) {
    const bitnet::IComputeBackend::BatchedDispatch full = site.dispatch;
    if (full.groups[0] < 2U) {
        std::printf("system,negative_control: skipped (single-group site)\n");
        return;
    }
    site.dispatch.groups[0] = full.groups[0] / 2U;
    bool rejected = false;
    try {
        verify_site(backend, site);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    site.dispatch = full;
    if (!rejected) {
        throw std::runtime_error("negative control: an under-dispatched site passed oracle "
                                 "verification - the guard is broken");
    }
    std::printf("system,negative_control,site=%s,half_grid_rejected=1\n", site.name);
}

// Verify every site of a pass against the oracle, then time it in
// isolation (batch-of-1 per repeat, incl. the per-submit floor).
void verify_and_time_sites(bitnet::IComputeBackend& backend, const char* pass,
                           std::vector<SystemSite>& sites, uint32_t repeats) {
    for (SystemSite& site : sites) {
        verify_site(backend, site);
        site.isolated_ms = backend.dispatch_batch_stats({site.dispatch}, repeats).avg_ms;
    }
    std::printf("system,%s,verified_sites=%zu (bit-exact vs CPU oracle before timing)\n", pass,
                sites.size());
}

constexpr std::array<LayerShape, 7> kLayerShapes{{
    {.name = "q", .n = 2560, .k = 2560},
    {.name = "k", .n = kBitNetKvHeadsDim, .k = kBitNetHidden},
    {.name = "v", .n = kBitNetKvHeadsDim, .k = kBitNetHidden},
    {.name = "o", .n = 2560, .k = 2560},
    {.name = "gate", .n = kBitNetFfn, .k = kBitNetHidden},
    {.name = "up", .n = kBitNetFfn, .k = kBitNetHidden},
    {.name = "dn", .n = kBitNetHidden, .k = kBitNetFfn},
}};

// Decode-only fused projection alternative: identical total weight count,
// but q/k/v share one stream and gate/up share another (4 calls/layer
// instead of 7). --fused-projections selects this list for the A/B.
constexpr std::array<LayerShape, 4> kFusedLayerShapes{{
    {.name = "qkv", .n = kBitNetHidden + 2U * kBitNetKvHeadsDim, .k = kBitNetHidden},
    {.name = "o", .n = kBitNetHidden, .k = kBitNetHidden},
    {.name = "up_gate", .n = 2U * kBitNetFfn, .k = kBitNetHidden},
    {.name = "dn", .n = kBitNetHidden, .k = kBitNetFfn},
}};

void run_system_pass(bitnet::IComputeBackend& backend, const char* pass, uint32_t layers,
                     uint32_t tokens, const std::vector<SystemSite>& per_layer_sites) {
    // Sequence per token: per layer, the sites (sq repeated for q/k/v/o).
    std::vector<bitnet::IComputeBackend::BatchedDispatch> seq;
    const auto calls_total =
        std::accumulate(per_layer_sites.begin(), per_layer_sites.end(), size_t{0},
                        [](size_t sum, const SystemSite& site) {
                            return sum + static_cast<size_t>(site.calls_per_layer);
                        });
    seq.reserve(static_cast<size_t>(layers) * calls_total);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        for (const SystemSite& site : per_layer_sites) {
            for (uint32_t rep = 0; rep < site.calls_per_layer; ++rep) {
                seq.push_back(site.dispatch);
            }
        }
    }
    const auto calls = static_cast<uint32_t>(seq.size());

    // Sync regime: one submit + fence per call, with the sticky bind +
    // set_params per site switch (the unbatched engine has to rebind
    // between different call sites - that cost is part of the
    // sync-regime number). The FULL layer stack per token: L x sites.
    std::vector<double> token_ms;
    token_ms.reserve(tokens);
    for (uint32_t t = 0; t < tokens; ++t) {
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t layer = 0; layer < layers; ++layer) {
            for (const SystemSite& site : per_layer_sites) {
                backend.bind(site.wbuf.get(), site.abuf.get(), site.obuf.get());
                backend.set_params(&site.params, sizeof(site.params));
                for (uint32_t rep = 0; rep < site.calls_per_layer; ++rep) {
                    static_cast<void>(backend.dispatch_bench_stats(site.sync_dispatch, 1U));
                }
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        token_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::ranges::sort(token_ms);
    const double sync_avg =
        std::accumulate(token_ms.begin(), token_ms.end(), 0.0) / static_cast<double>(tokens);
    const size_t p90_index = token_ms.size() >= 10U ? (token_ms.size() * 9U) / 10U : 0U;

    // Batched regime: ONE submit + fence per token.
    const auto batch = backend.dispatch_batch_stats(seq, tokens);

    std::printf("system,%s,sync,layers=%u,calls_per_token=%u,ms_avg=%.3f,ms_min=%.3f,ms_p90=%.3f,"
                "tok_per_s=%.2f\n",
                pass, layers, calls, sync_avg, token_ms.front(), token_ms[p90_index],
                tok_per_s(sync_avg));
    std::printf(
        "system,%s,batched,layers=%u,calls_per_token=%u,ms_avg=%.3f,ms_min=%.3f,ms_p90=%.3f,"
        "tok_per_s=%.2f\n",
        pass, layers, calls, batch.avg_ms, batch.min_ms, batch.p90_ms, tok_per_s(batch.avg_ms));
    double kernel_only = 0.0;
    for (const SystemSite& site : per_layer_sites) {
        const double calls_per_token =
            static_cast<double>(layers) * static_cast<double>(site.calls_per_layer);
        kernel_only += site.isolated_ms * calls_per_token;
        std::printf("site,%s,%s,%s,n=%u,k=%u,ms_per_call=%.4f\n", pass, site.name,
                    site.kernel_name.c_str(), site.n, site.k, site.isolated_ms);
    }
    std::printf(
        "system,%s,kernel_only_sum_ms=%.3f,overhead_sync_ms=%.3f,overhead_batched_ms=%.3f\n", pass,
        kernel_only, sync_avg - kernel_only, batch.avg_ms - kernel_only);
}

} // namespace

void run_system(bitnet::IComputeBackend& backend, const bitnet::GpuProfile& gpu,
                const bitnet::DeviceCaps& caps, uint32_t layers, uint32_t decode_tokens,
                uint32_t prefill_tokens, bitnet::DotVariant dot_variant, bool fused_projections,
                bool decode_only) {
    // Champion kernels come from the generated profile table:
    // one measured decode (GEMV) and one measured prefill (GEMM) kernel per
    // device profile (or the --decode-champion / --prefill-champion A/B
    // overrides). A dot-generation champion runs on EVERY device: the
    // capability-driven variant picks the hardware or the emulated (slow)
    // module; only --variant popcount skips the dot champion lane.
    const std::string_view gemv_champ(gpu.gemv_champion);
    const bool champ_is_dot = bitnet::is_dot_kernel(gemv_champ);
    if (!champ_is_dot || dot_variant != bitnet::DotVariant::Popcount) {
        // Decode: real projections per layer, m=1 (one token), champion
        // GEMV. The default is seven unfused calls; the A/B knob selects
        // four fused calls with exactly the same total weight count.
        const std::span<const LayerShape> decode_shapes =
            fused_projections ? std::span<const LayerShape>(kFusedLayerShapes)
                              : std::span<const LayerShape>(kLayerShapes);
        const char* decode_pass = fused_projections ? "decode_fused" : "decode";
        std::vector<SystemSite> decode_sites;
        decode_sites.reserve(decode_shapes.size());
        std::ranges::transform(
            decode_shapes, std::back_inserter(decode_sites), [&](const LayerShape& ls) {
                return make_site(backend, gpu, caps, gemv_champ, ls, 1U, dot_variant);
            });
        guard_negative_control(backend, decode_sites.front());
        verify_and_time_sites(backend, decode_pass, decode_sites, decode_tokens);
        run_system_pass(backend, decode_pass, layers, decode_tokens, decode_sites);
    } else {
        std::printf("system,decode: skipped (champion %s is a dot kernel; --variant "
                    "popcount)\n",
                    gpu.gemv_champion);
    }
    // Output head: 128256x2560, ONCE per token (not per
    // layer) - its own measured row so the decode upper bound stays
    // separable from the head cost. Weights at 2 bits would be 82 MB in one
    // buffer; a device that rejects the allocation or whose limits the
    // champion's grid exceeds reports the site as unavailable (reason in
    // the log). Oracle verification runs OUTSIDE the catch: a head that
    // allocates but computes wrong results fails the run.
    std::vector<SystemSite> head_sites;
    try {
        head_sites.push_back(make_site(
            backend, gpu, caps, gemv_champ,
            LayerShape{.name = "head", .n = kBitNetVocab, .k = kBitNetHidden}, 1U, dot_variant));
    } catch (const std::exception& e) {
        std::printf("system,head: UNAVAILABLE (%s) - skipped\n", e.what());
    }
    if (!head_sites.empty()) {
        verify_and_time_sites(backend, "head", head_sites, decode_tokens);
        run_system_pass(backend, "head", 1U, decode_tokens, head_sites);
    }
    if (decode_only) {
        std::printf("system,prefill: skipped (--decode-only)\n");
        return;
    }
    // Prefill: all seven real matrices at m=64 through the champion GEMM.
    std::vector<SystemSite> prefill_sites;
    prefill_sites.reserve(kLayerShapes.size());
    std::ranges::transform(kLayerShapes, std::back_inserter(prefill_sites),
                           [&](const LayerShape& ls) {
                               return make_site(backend, gpu, caps, gpu.gemm_champion, ls,
                                                kPrefillTokensPerCall, dot_variant);
                           });
    verify_and_time_sites(backend, "prefill", prefill_sites, prefill_tokens);
    run_system_pass(backend, "prefill", layers, prefill_tokens, prefill_sites);
}
