// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

#pragma once
// Machine-checked knob-vs-limit conformance. Every value the host compiles or dispatches
// is validated against the device's ADVERTISED limits at startup:
// workgroup size/count dimensions, per-kernel-variant groupshared bytes
// (computed from the tile knobs), the largest weight/activation/output
// buffers per bench/test shape vs maxStorageBufferRange and
// maxMemoryAllocationSize (the Adreno 128 MiB SSBO limit is the
// motivation), and the subgroup/coop-mat/dot capability ladder as
// informational lines. EXCEEDING A LIMIT FAILS THE RUN - one
// "limit/used/verdict" line per check, "caps: N checks, 0 violations" on
// success. Capability-driven: checks consume DeviceCaps flags + the knob
// table for the matched profile - never device names. Kernel names are
// matched EXACTLY against poc/backend/kernel_family.hpp's table.

#include "backend/backend.hpp"
#include "backend/gpu_profiles.h"
#include "backend/kernel_family.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

namespace bitnet {

struct ConformanceResult {
    int checks = 0;
    int violations = 0;
};

// Buffer-size models per family contract (poc/common/format.hpp packers):
// plane format stores 2 bit-planes of 32 weights per uint word pair
// (2 * ceil(k/32) words per row); the uint4 plane-family format packs 64
// weights per 16-byte unit; the packed-dot format stores dequantized int8
// (1 byte per weight).
[[nodiscard]] inline uint64_t plane_weight_bytes(uint64_t n, uint64_t k) {
    return n * 4ULL * (2ULL * ((k + 31ULL) / 32ULL));
}
[[nodiscard]] inline uint64_t unit_weight_bytes(uint64_t n, uint64_t k) {
    return n * 16ULL * ((k + 63ULL) / 64ULL);
}
[[nodiscard]] inline uint64_t dots_weight_bytes(uint64_t n, uint64_t k) {
    return n * 4ULL * ((k + 3ULL) / 4ULL);
}
[[nodiscard]] inline uint64_t activation_bytes(uint64_t m, uint64_t k) {
    return m * k;
}
[[nodiscard]] inline uint64_t output_bytes(uint64_t m, uint64_t n) {
    return m * n * 4ULL;
}

// groupshared bytes per kernel, computed from the SAME tile knobs CMake
// passes as -D defines (mirrors the groupshared declarations in
// poc/slang/bitnet_kernels.slang: gemm_bitplane_tiled
// (TILE_N*2 + TILE_M*8)*CHUNK words, gemm_dot_tiled
// (TILE_N + TILE_M)*CHUNK words, gemm_plane_tiled
// (TILE_N*4 + TILE_M*16)*(CHUNK/2) words, cooperative-row families THREADS
// words, naive/ksplit/microtile/coopmat: none counted here).
[[nodiscard]] inline uint64_t family_shared_bytes(std::string_view kernel, const GpuProfile& gpu) {
    if (kernel == "gemm_bitplane_tiled") {
        return (static_cast<uint64_t>(gpu.gemm_tile_n) * 2U +
                static_cast<uint64_t>(gpu.gemm_tile_m) * 8U) *
               gpu.gemm_chunk * 4ULL;
    }
    if (kernel == "gemm_dot_tiled") {
        return (static_cast<uint64_t>(gpu.gemm_tile_n) + gpu.gemm_tile_m) * gpu.gemm_chunk * 4ULL;
    }
    if (kernel == "gemm_plane_tiled") {
        return (static_cast<uint64_t>(gpu.gemm_tile_n) * 4U +
                static_cast<uint64_t>(gpu.gemm_tile_m) * 16U) *
               (gpu.gemm_chunk / 2U) * 4ULL;
    }
    if (kernel == "gemv_dot_rowcoop") {
        return static_cast<uint64_t>(gpu.gemv_dot_threads) * 4ULL;
    }
    if (kernel == "gemv_plane_rowcoop") {
        return static_cast<uint64_t>(gpu.gemv6_threads) * 4ULL;
    }
    if (kernel == "gemv_bitplane_rowcoop" || kernel == "gemm_bitplane_cellcoop") {
        return static_cast<uint64_t>(gpu.gemv_threads) * 4ULL;
    }
    return 0ULL; // rowserial/naive/ksplit/microtile/probes/coopmat: not counted here
}

// Workgroup (numthreads) contract per kernel: x from the profile knob
// (same values CMake passes as -D defines), y/z from the kernel literals.
struct WorkgroupContract {
    const char* kernel;
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

[[nodiscard]] inline std::array<WorkgroupContract, 12> workgroup_contracts(const GpuProfile& gpu) {
    return {WorkgroupContract{"gemv_bitplane_rowserial", 64U, 1U, 1U},
            WorkgroupContract{"gemv_bitplane_ksplit", 256U, 1U, 1U},
            WorkgroupContract{"gemv_bitplane_ksplit_il", 256U, 1U, 1U},
            WorkgroupContract{"gemv_bitplane_rowcoop", gpu.gemv_threads, 1U, 1U},
            WorkgroupContract{"gemm_bitplane_cellcoop", gpu.gemv_threads, 1U, 1U},
            WorkgroupContract{"gemm_bitplane_tiled", gpu.gemm_tile_threads, 1U, 1U},
            WorkgroupContract{"gemv_dot_rowcoop", gpu.gemv_dot_threads, 1U, 1U},
            WorkgroupContract{"gemm_dot_tiled", gpu.gemm_tile_threads, 1U, 1U},
            WorkgroupContract{"gemv_plane_rowcoop", gpu.gemv6_threads, 1U, 1U},
            WorkgroupContract{"gemm_plane_tiled", gpu.gemm_tile_threads, 1U, 1U},
            WorkgroupContract{"gemm_plane_microtile", gpu.gemm_v7_threads, 1U, 1U},
            WorkgroupContract{"gemm_bitplane_naive", 16U, 16U, 1U}};
}

// Dispatch geometry per champion for the LARGEST shapes the harness runs
// (head 128256x2560, gemv bench max 13824x2560, gemm bench max 256x2560x6912
// - the real BitNet-2B projections). Names are exact and
// unambiguously gemv-xor-gemm. EVERY matmul kernel has an entry (a
// champion override may select any of them); an UNKNOWN name fails closed
// with an unrepresentable grid, so no champion can pass the gate
// unchecked.
inline constexpr uint64_t kUncheckedGroups = UINT64_MAX;

[[nodiscard]] inline std::array<uint64_t, 3> dispatch_groups(std::string_view champion,
                                                             const GpuProfile& gpu, uint64_t m,
                                                             uint64_t n, uint64_t k) {
    const uint64_t plane_words = (k + 31ULL) / 32ULL; // bit-plane words per row
    if (champion.empty()) {
        // Unfilled champion (sweep profiles): nothing to dispatch-check;
        // poc_bench --system refuses to run without champions.
        return {1ULL, 1ULL, 1ULL};
    }
    if (champion == "gemv_plane_rowcoop") {
        return {(n + gpu.gemv6_rows_per_group - 1ULL) / gpu.gemv6_rows_per_group, 1ULL, 1ULL};
    }
    if (is_tiled_kernel(champion)) {
        // Tile-grid families (gemm_bitplane_tiled / gemm_dot_tiled /
        // gemm_plane_tiled): one workgroup per TILE_M x TILE_N output tile.
        return {(n + gpu.gemm_tile_n - 1ULL) / gpu.gemm_tile_n,
                (m + gpu.gemm_tile_m - 1ULL) / gpu.gemm_tile_m, 1ULL};
    }
    if (champion == "gemv_dot_rowcoop" || champion == "gemv_bitplane_rowcoop") {
        return {n, 1ULL, 1ULL}; // one group per output row
    }
    if (champion == "gemv_bitplane_rowserial") {
        return {(n + 63ULL) / 64ULL, 1ULL, 1ULL}; // numthreads(64, 1, 1)
    }
    if (champion == "gemv_bitplane_ksplit" || champion == "gemv_bitplane_ksplit_il") {
        // numthreads(256, 1, 1) over n; one dispatch layer per 8-word K-slice.
        return {(n + 255ULL) / 256ULL, 1ULL, (plane_words + 7ULL) / 8ULL};
    }
    if (champion == "gemm_bitplane_ksplit") {
        // numthreads(64, 1, 1) over n; m on y; one layer per 8-word K-slice.
        return {(n + 63ULL) / 64ULL, m, (plane_words + 7ULL) / 8ULL};
    }
    if (champion == "gemm_bitplane_naive") {
        return {(n + 15ULL) / 16ULL, (m + 15ULL) / 16ULL, 1ULL}; // numthreads(16, 16, 1)
    }
    if (champion == "gemm_bitplane_cellcoop") {
        return {m * n, 1ULL, 1ULL}; // one group per output cell
    }
    if (is_microtile_kernel(champion)) {
        const uint64_t tiles = ((n + gpu.gemm_v7_cols - 1ULL) / gpu.gemm_v7_cols) *
                               ((m + gpu.gemm_v7_rows - 1ULL) / gpu.gemm_v7_rows);
        return {(tiles + gpu.gemm_v7_threads - 1ULL) / gpu.gemm_v7_threads, 1ULL, 1ULL};
    }
    if (is_coopmat_kernel(champion)) {
        // One 16x16 output tile per 32-lane subgroup, one subgroup per group.
        return {((n + 15ULL) / 16ULL) * ((m + 15ULL) / 16ULL), 1ULL, 1ULL};
    }
    return {kUncheckedGroups, kUncheckedGroups, kUncheckedGroups}; // unknown: fail closed
}

[[nodiscard]] inline uint64_t champion_weight_bytes(std::string_view champion, uint64_t n,
                                                    uint64_t k) {
    if (champion == "gemv_dot_rowcoop" || champion == "gemm_dot_tiled") {
        return dots_weight_bytes(n, k);
    }
    if (champion == "gemv_plane_rowcoop" || champion == "gemm_plane_tiled") {
        return unit_weight_bytes(n, k);
    }
    return plane_weight_bytes(n, k); // bitplane-family champions
}

inline ConformanceResult check_caps_conformance(const DeviceCaps& caps, const GpuProfile& gpu) {
    ConformanceResult r;
    auto check = [&](bool ok, const std::string& name, std::string_view limit,
                     std::string_view used) {
        ++r.checks;
        if (!ok) {
            ++r.violations;
        }
        std::printf("[caps] %-38s limit %-24s used %-24s %s\n", name.c_str(),
                    std::string(limit).c_str(), std::string(used).c_str(), ok ? "OK" : "EXCEEDS");
    };

    // 1. Workgroup SIZE dimensions + invocations per kernel family.
    for (const WorkgroupContract& w : workgroup_contracts(gpu)) {
        const uint64_t total = static_cast<uint64_t>(w.x) * w.y * w.z;
        check(w.x <= caps.max_workgroup_size[0] && w.y <= caps.max_workgroup_size[1] &&
                  w.z <= caps.max_workgroup_size[2],
              std::string("workgroup_size ") + w.kernel,
              std::to_string(caps.max_workgroup_size[0]) + "/" +
                  std::to_string(caps.max_workgroup_size[1]) + "/" +
                  std::to_string(caps.max_workgroup_size[2]),
              std::to_string(w.x) + "/" + std::to_string(w.y) + "/" + std::to_string(w.z));
        check(total <= caps.max_workgroup_invocations,
              std::string("workgroup_invocations ") + w.kernel,
              std::to_string(caps.max_workgroup_invocations), std::to_string(total));
    }

    // 2. groupshared bytes per kernel family (tile knobs).
    for (const WorkgroupContract& w : workgroup_contracts(gpu)) {
        const uint64_t shared = family_shared_bytes(w.kernel, gpu);
        check(shared <= caps.max_shared_memory_bytes, std::string("shared_memory ") + w.kernel,
              std::to_string(caps.max_shared_memory_bytes) + " B", std::to_string(shared) + " B");
    }

    // 3. Dispatch GROUP counts for the largest shapes vs the count limits.
    struct LargestDispatch {
        const char* what;
        const char* champion;
        uint64_t m;
        uint64_t n;
        uint64_t k;
    };
    const std::array<LargestDispatch, 3> largest{{
        {"head gemv (champion)", gpu.gemv_champion, 1ULL, 128256ULL, 2560ULL},
        {"gemv bench max (champion)", gpu.gemv_champion, 1ULL, 13824ULL, 2560ULL},
        {"gemm bench max (champion)", gpu.gemm_champion, 256ULL, 2560ULL, 6912ULL},
    }};
    for (const LargestDispatch& d : largest) {
        const std::array<uint64_t, 3> groups = dispatch_groups(d.champion, gpu, d.m, d.n, d.k);
        const bool known = groups[0] != kUncheckedGroups;
        check(known && groups[0] <= caps.max_workgroup_count[0] &&
                  groups[1] <= caps.max_workgroup_count[1] &&
                  groups[2] <= caps.max_workgroup_count[2],
              std::string("workgroup_count ") + d.what,
              std::to_string(caps.max_workgroup_count[0]) + "/" +
                  std::to_string(caps.max_workgroup_count[1]) + "/" +
                  std::to_string(caps.max_workgroup_count[2]),
              known ? std::to_string(groups[0]) + "/" + std::to_string(groups[1]) + "/" +
                          std::to_string(groups[2])
                    : std::string("no dispatch contract for '") + d.champion + "'");
    }

    // 4. Largest buffers per shape vs SSBO range + allocation size.
    for (const LargestDispatch& d : largest) {
        const uint64_t weights = champion_weight_bytes(d.champion, d.n, d.k);
        const uint64_t acts = activation_bytes(d.m, d.k);
        const uint64_t out = output_bytes(d.m, d.n);
        const uint64_t biggest = std::max({weights, acts, out});
        check(biggest <= caps.max_storage_buffer_range,
              std::string("storage_buffer_range ") + d.what,
              std::to_string(caps.max_storage_buffer_range) + " B",
              std::to_string(biggest) + " B (w=" + std::to_string(weights) +
                  " a=" + std::to_string(acts) + " o=" + std::to_string(out) + ")");
        check(
            biggest <= caps.max_memory_allocation_size, std::string("memory_allocation ") + d.what,
            std::to_string(caps.max_memory_allocation_size) + " B", std::to_string(biggest) + " B");
    }

    // 5. Capability ladder visibility (informational, never a violation):
    // selection flags are read from caps, never from device names.
    check(true, "integer dot capability", caps.shader_integer_dot_product ? "present" : "absent",
          caps.shader_integer_dot_product ? "hardware dot variant" : "emulated dot variant");
    // Coop-mat lane: SELECTION FOLLOWS THE ENUMERATED TYPES - the lane runs
    // only the exact enumerated 16x16x32 sint8*sint8->sint32 subgroup shape.
    // int8 is 4x the 2-bit weight-stream floor, and the ceiling math in
    // the bench reflects that. The lane is opt-in experimental
    // (known-broken by a slangc operand-plumbing bug).
    check(true, "cooperative matrix (VK_KHR_cooperative_matrix)",
          caps.cooperative_matrix ? "present" : "absent",
          caps.coop_mat_shape_s8_s32
              ? "enumerated sint8->sint32 subgroup shape " + std::to_string(caps.coop_mat_msize) +
                    "x" + std::to_string(caps.coop_mat_nsize) + "x" +
                    std::to_string(caps.coop_mat_ksize) + " (gemm_coopmat opt-in, known-broken)"
              : "no matching sint8->sint32 shape; gemm_plane_tiled serves");
    check(true, "cooperative vector (NV fallback - no KHR variant registered)",
          caps.cooperative_vector_nv ? "present" : "absent", "recorded only; no kernel uses it");
    check(true, "subgroup ops", caps.subgroup_arithmetic ? "arithmetic exposed" : "absent",
          "no shipped kernel dispatches on subgroup ops");
    check(caps.api_version >= kMinApi10, "process baseline (popcount path)",
          "Vulkan 1.0 (kMinApi10)",
          std::to_string(version_major(caps.api_version)) + "." +
              std::to_string(version_minor(caps.api_version)));

    std::printf("caps: %d checks, %d violations\n", r.checks, r.violations);
    return r;
}

} // namespace bitnet
