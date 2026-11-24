#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Expected LocalSize per kernel - the single mapping between kernel names
// and their compiled group contracts (Slang numthreads / -D defines in
// CMakeLists.txt). The Vulkan backend's LocalSize guard compares this
// against the loaded module's OpExecutionMode, so a stale or
// wrong-profile .spv is REJECTED instead of silently computing a fraction
// of the output.
//
// Fixed sizes here mirror the [numthreads] literals in the kernels; the
// tunable ones come from the GpuProfile (which CMake generates from the
// same values it passes as -D defines - one source of truth). Kernels
// are classified by exact name (kernel_family.hpp).

#include "backend/gpu_profiles.h"
#include "backend/kernel_family.hpp"
#include "backend/variant_selection.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace bitnet {

inline uint32_t expected_local_size(const std::string& kernel, const GpuProfile& gpu) {
    // The on-disk emulated-dot suffix changes the module, not its geometry.
    std::string_view logical = kernel;
    constexpr std::string_view emulated_suffix = "_emulated";
    if (logical.ends_with(emulated_suffix)) {
        logical.remove_suffix(emulated_suffix.size());
    }
    if (logical == "gemv_dot_rowcoop") {
        return gpu.gemv_dot_threads; // GEMV_DOT_ROWCOOP_THREADS
    }
    if (logical == "gemv_plane_rowcoop") {
        return gpu.gemv6_threads; // GEMV_PLANE_ROWCOOP_THREADS, own profile knob
    }
    if (is_coopmat_kernel(logical)) {
        return kCoopMatThreads; // gemm_coopmat: one 32-lane subgroup per workgroup
    }
    if (is_microtile_kernel(logical)) {
        return gpu.gemm_v7_threads; // gemm_plane_microtile[_image] (generated table)
    }
    if (is_tiled_kernel(logical)) {
        // gemm_bitplane_tiled / gemm_dot_tiled / gemm_plane_tiled (TILE_THREADS)
        return gpu.gemm_tile_threads;
    }
    if (logical == "gemv_bitplane_rowcoop" || logical == "gemm_bitplane_cellcoop") {
        return gpu.gemv_threads; // shared cooperative-row contract
                                 // (GEMV_*/GEMM_BITPLANE_CELLCOOP_THREADS)
    }
    // Fixed-contract kernels.
    if (logical == "gemv_bitplane_ksplit" || logical == "gemv_bitplane_ksplit_il") {
        return 256U;
    }
    if (logical == "gemm_bitplane_ksplit") {
        return 64U;
    }
    if (logical == "gemm_bitplane_naive") {
        return 16U; // (16, 16, 1)
    }
    if (logical == "bw_probe" || logical == "stream_bw_probe" || logical == "alu_countbits_probe" ||
        logical == "alu_fma_probe" || logical == "vec_load_probe") {
        return 256U;
    }
    return 64U; // gemv_bitplane_rowserial, null_probe and the default contract
}

} // namespace bitnet
