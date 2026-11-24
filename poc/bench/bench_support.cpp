// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Workgroup geometry and family packing of the benchmark: ONE dispatch-grid
// table per kernel (bench matrix and --system sites share it) and the packer
// selection that follows the kernel family. An unknown kernel throws - a
// guessed grid is how a kernel gets timed without doing its work.

#include "bench/bench_support.hpp"

#include "backend/kernel_family.hpp"
#include "common/format.hpp"

#include <stdexcept>
#include <string>

// K-split kernels take their K-slice width in KernelParams.mode and one
// dispatch layer (groups z) per slice. A ZERO slice width makes every
// thread return before its loop - a silent no-op that still costs one
// submit and would be timed as if it were the workload. The bench matrix
// and the system pass therefore resolve mode and grid through the two
// helpers below.
bool is_ksplit_kernel(std::string_view kernel) {
    return kernel == "gemv_bitplane_ksplit" || kernel == "gemm_bitplane_ksplit";
}

uint32_t kernel_param_mode(std::string_view kernel, uint32_t packed_mode) {
    return is_ksplit_kernel(kernel) ? kKSliceWords : packed_mode;
}

uint32_t ceil_div(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1U) / divisor;
}
// Per-kernel workgroup grid: the ONE dispatch-geometry table of this file
// (bench matrix AND --system sites). Exact-match per kernel
// (kernel_family.hpp), mirroring the [numthreads] literals in
// poc/slang/bitnet_kernels.slang and the dispatch contracts of the test
// harness. An unknown kernel throws - a guessed grid is how a kernel gets
// timed without doing its work.
std::array<uint32_t, 3> bench_dispatch_groups(std::string_view kernel,
                                              const bitnet::GpuProfile& gpu, uint32_t m, uint32_t n,
                                              uint32_t kernel_kwords) {
    if (bitnet::is_coopmat_kernel(kernel)) {
        // Coop-mat MMA: one MSize x NSize tile per 32-lane subgroup - a
        // flat grid over ceil(tiles / kCoopMatSubgroups).
        const uint32_t tiles =
            ceil_div(n, bitnet::kCoopMatNSize) * ceil_div(m, bitnet::kCoopMatMSize);
        return {ceil_div(tiles, bitnet::kCoopMatSubgroups), 1U, 1U};
    }
    if (bitnet::is_microtile_kernel(kernel)) {
        // Register micro-tile: one THREAD per (rows x cols) output tile,
        // gemm_v7_threads per workgroup - a flat 1D grid over the tiles.
        const uint32_t tiles = ceil_div(n, gpu.gemm_v7_cols) * ceil_div(m, gpu.gemm_v7_rows);
        return {ceil_div(tiles, gpu.gemm_v7_threads), 1U, 1U};
    }
    if (bitnet::is_tiled_kernel(kernel)) {
        // gemm_bitplane_tiled / gemm_dot_tiled / gemm_plane_tiled: one
        // workgroup per TILE_N x TILE_M output tile (profile or sweep sizes).
        return {ceil_div(n, gpu.gemm_tile_n), ceil_div(m, gpu.gemm_tile_m), 1U};
    }
    if (kernel == "gemv_plane_rowcoop") {
        // The module is compiled with the SAME -DROWS_PER_GROUP value.
        return {ceil_div(n, gpu.gemv6_rows_per_group), 1U, 1U};
    }
    if (kernel == "gemv_dot_rowcoop" || kernel == "gemv_bitplane_rowcoop") {
        return {n, 1U, 1U}; // one workgroup per output row
    }
    if (kernel == "gemm_bitplane_cellcoop") {
        return {m * n, 1U, 1U}; // one workgroup per output cell
    }
    if (kernel == "gemv_bitplane_ksplit") {
        return {ceil_div(n, kV2GroupThreads), 1U, ceil_div(kernel_kwords, kKSliceWords)};
    }
    if (kernel == "gemm_bitplane_ksplit") {
        return {ceil_div(n, kV1GroupThreads), m, ceil_div(kernel_kwords, kKSliceWords)};
    }
    if (kernel == "gemm_bitplane_naive") {
        // numthreads(16, 16, 1): n on x, m on y.
        return {ceil_div(n, kV1GemmTile), ceil_div(m, kV1GemmTile), 1U};
    }
    if (kernel == "gemv_bitplane_rowserial") {
        return {ceil_div(n, kV1GroupThreads), 1U, 1U};
    }
    throw std::runtime_error("bench: no dispatch geometry for kernel '" + std::string(kernel) +
                             "'");
}

// vkCmdDispatch requires every group count to stay within
// maxComputeWorkGroupCount (VUID-vkCmdDispatch-groupCountX-00386 ff.); a
// ZERO count is legal Vulkan but always a harness bug here (nothing runs).
// The validation layer cannot be loaded on the Android test devices, so the
// bench checks the limit itself before every dispatch it issues.
bool dispatch_within_limits(const std::array<uint32_t, 3>& groups, const bitnet::DeviceCaps& caps) {
    for (size_t axis = 0; axis < groups.size(); ++axis) {
        if (groups[axis] == 0U || groups[axis] > caps.max_workgroup_count[axis]) {
            return false;
        }
    }
    return true;
}

std::string groups_text(const std::array<uint32_t, 3>& groups) {
    return std::to_string(groups[0]) + "/" + std::to_string(groups[1]) + "/" +
           std::to_string(groups[2]);
}
SiteParams pack_site(const bitnet::TernaryTensor& t, uint32_t m, uint32_t n, uint32_t k,
                     std::string_view kernel) {
    SiteParams out;
    if (bitnet::is_coopmat_kernel(kernel)) {
        // Coop-mat lane: plain padded int8 operands, activations K-major
        // (transposed; see format.hpp). kwords = padded K, mode = padded M.
        const auto pair = bitnet::PackedInt8MatPair::pack(
            t.weights.data(), t.activations.data(), m, n, k, bitnet::kCoopMatMSize,
            bitnet::kCoopMatNSize, bitnet::kCoopMatKSize, true);
        out.kwords = pair.paddedK;
        out.mode = pair.paddedM;
        out.wbytes = pair.weightBytes();
        out.abytes = pair.actBytes();
        out.wbits.assign((static_cast<size_t>(out.wbytes) + 3U) / 4U, 0U);
        out.abits.assign((static_cast<size_t>(out.abytes) + 3U) / 4U, 0U);
        std::memcpy(out.wbits.data(), pair.weights.data(), out.wbytes);
        std::memcpy(out.abits.data(), pair.acts.data(), out.abytes);
    } else if (bitnet::is_plane_kernel(kernel)) {
        const auto pw = bitnet::PackedTernaryV6::pack(t.weights.data(), n, k);
        const auto pa = bitnet::PackedActivationsV6::pack(t.activations.data(), m, k);
        out.kwords = pw.units;
        out.wbits = pw.bits;
        out.abits = pa.bits;
        out.wbytes = pw.bytes();
        out.abytes = pa.bytes();
    } else if (bitnet::is_dot_kernel(kernel)) {
        const auto pw = bitnet::PackedTernaryDots::pack(t.weights.data(), n, k);
        const auto pa = bitnet::PackedActivationsDots::pack(t.activations.data(), m, k);
        out.kwords = pw.kwords;
        out.wbits = pw.bits;
        out.abits = pa.bits;
        out.wbytes = pw.bytes();
        out.abytes = pa.bytes();
    } else {
        const auto pw = bitnet::PackedTernary::pack(t.weights.data(), n, k);
        const auto pa = bitnet::PackedActivations::pack(t.activations.data(), m, k);
        out.kwords = pw.kwords;
        out.wbits = pw.bits;
        out.abits = pa.bits;
        out.wbytes = pw.bytes();
        out.abytes = pa.bytes();
    }
    return out;
}
