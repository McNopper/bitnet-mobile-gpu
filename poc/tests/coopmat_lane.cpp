// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Cooperative-matrix lane of the harness (VK_KHR_cooperative_matrix), opt-in
// experimental like the microtile family: it runs only when the device
// enumerates the exact sint8 x sint8 -> sint32 subgroup shape and the
// environment asks for experimental kernels. The module is loaded fresh from
// the profile SPIR-V directory and must pass the backend's LocalSize guard.

#include "backend/variant_selection.hpp"
#include "tests/suites.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

// Coop-mat lane provenance print: the dispatch
// contract printed per run, so a provenance bug (stale module, swapped
// slot, wrong grid) can never masquerade as a codegen finding. Cross-checked
// against the static SPIR-V disassembly: the module is loaded fresh from
// the profile spv dir
// and REJECTED by the backend guard when its OpExecutionMode LocalSize.x
// disagrees with kCoopMatThreads; update_storage_set writes bindings
// 0/1/2/3 in exactly the shader's vk::binding order (weights, acts, out,
// params); the grid covers every 16x16 tile with kCoopMatSubgroups=1
// workgroup (32 threads = one 32-lane subgroup) each.
static void print_coopmat_provenance(const Shape& s, const bitnet::PackedInt8MatPair& pair,
                                     uint32_t tiles_x, uint32_t tiles_y, const Variant& coop) {
    const std::filesystem::path module =
        std::filesystem::path(bitnet::env_string("BITNET_SPV_DIR")) / "gemm_coopmat.spv";
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(module, ec);
    std::printf("  gemm_coopmat provenance: module %s (%s%llu bytes)\n",
                module.generic_string().c_str(), ec ? "stat failed, " : "",
                static_cast<unsigned long long>(ec ? 0U : bytes));
    std::printf("    entry: slangc -entry gemm_coopmat; local-size guard expects %u threads "
                "(kCoopMatThreads == GEMM_COOPMAT_THREADS)\n",
                bitnet::kCoopMatThreads);
    std::printf("    descriptor slots: 0=weights %llu B, 1=acts %llu B, 2=out, 3=params "
                "(backend write order == shader vk::binding order)\n",
                static_cast<unsigned long long>(pair.weightBytes()),
                static_cast<unsigned long long>(pair.actBytes()));
    std::printf("    grid: %u x %u tiles of 16x16x32, %u subgroup(s)/workgroup -> groups "
                "(%u, %u, %u)\n",
                tiles_x, tiles_y, bitnet::kCoopMatSubgroups, coop.groups[0], coop.groups[1],
                coop.groups[2]);
    std::printf("    params: m=%u n=%u kwords(paddedK)=%u mode(paddedM)=%u "
                "(PackedInt8MatPair, transposed acts)\n",
                s.m, s.n, pair.paddedK, pair.paddedM);
}

// Coop-mat MMA lane (VK_KHR_cooperative_matrix), opt-in experimental (like
// the microtile family). The module loads ONLY when the device
// passes coop_mat_shape_matches (exact enumerated shape + int8 storage +
// fixed 32-lane subgroups); the printed skip notes are the fall-through
// evidence on devices without the lane (Mali): no coop-mat dispatch is
// attempted, gemm_plane_tiled serves. The runtime result is WRONG on the only
// coop-mat desktop device (RTX 5070 + slangc 2026.17): slangc cannot
// wire CoopMat.Load results into coopMatMulAdd operands - the MulAdd
// consumes OpUndef and outputs are zero.
// Returns the mismatch count (0 on skip/pass).
int64_t run_coopmat_lane(const Shape& s, const bitnet::DeviceCaps& caps,
                         const bitnet::TernaryTensor& t, const bitnet::GpuProfile* gpu,
                         bitnet::IComputeBackend& backend, bitnet::IBuffer* out,
                         const bitnet::KernelParams& base, std::vector<int32_t>& gpu_out,
                         const std::vector<int32_t>& ref) {
    if (!experiments_enabled()) {
        std::printf("  gemm_coopmat: skipped (experimental; BITNET_EXPERIMENTAL=1 to "
                    "enable - KNOWN BROKEN on RTX 5070, see the report, section 5.3)\n");
        return 0;
    }
    if (!bitnet::coop_mat_shape_matches(caps)) {
        std::printf("  gemm_coopmat: skipped (no enumerated sint8*sint8->sint32 subgroup "
                    "shape; gemm_plane_tiled serves)\n");
        return 0;
    }
    // Direct-pattern operands: plain padded int8, activations K-major
    // (transposed) per the measured ColumnMajor-load note; P.kwords = padded
    // K, P.mode = padded M (see format.hpp).
    const auto pair = bitnet::PackedInt8MatPair::pack(
        t.weights.data(), t.activations.data(), s.m, s.n, s.k, bitnet::kCoopMatMSize,
        bitnet::kCoopMatNSize, bitnet::kCoopMatKSize, true);
    auto wbuf8 = backend.create_buffer(pair.weightBytes(), true);
    auto abuf8 = backend.create_buffer(pair.actBytes(), true);
    std::memcpy(wbuf8->mapped(), pair.weights.data(), pair.weightBytes());
    std::memcpy(abuf8->mapped(), pair.acts.data(), pair.actBytes());
    const uint32_t coop_tiles_x = (s.n + bitnet::kCoopMatNSize - 1U) / bitnet::kCoopMatNSize;
    const uint32_t coop_tiles_y = (s.m + bitnet::kCoopMatMSize - 1U) / bitnet::kCoopMatMSize;
    const auto coop_tiles = static_cast<uint32_t>(coop_tiles_x * coop_tiles_y);
    const Variant coop{
        .name = "gemm_coopmat",
        .mode = pair.paddedM,
        .kwords = pair.paddedK,
        .needs_zero = false,
        .groups = {(coop_tiles + bitnet::kCoopMatSubgroups - 1U) / bitnet::kCoopMatSubgroups, 1U,
                   1U},
        .pass_note = "PASS (bit-exact, coopmat MMA)"};
    print_coopmat_provenance(s, pair, coop_tiles_x, coop_tiles_y, coop);
    return run_and_compare_variant(coop, gpu, backend, wbuf8.get(), abuf8.get(), out, base, gpu_out,
                                   ref, nullptr);
}