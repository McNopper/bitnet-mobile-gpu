// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

#pragma once
// Kernel family classification - the single source of truth: names are
// mechanism-descriptive, {operation}_{family}_{assignment}, e.g.
// gemm_bitplane_tiled, gemv_plane_rowcoop. This header is the ONE place
// that answers "what is this kernel", via exact-match lookup (never fuzzy
// substring guessing), so every consumer (LocalSize guard, groupshared
// sizing, dispatch-grid construction, byte model) reads the same
// classification instead of re-deriving its own.
//
// Families:
//   bitplane - AND+popcount over 2/8 bit-planes (format v1), the
//              always-portable base (Vulkan 1.0 floor).
//   dot      - packed integer dot product (format v2); hardware
//              dot4add_i8packed or the bit-exact emulated fallback.
//   plane    - plane-uint4 staging (format v3); microtile
//              is the register-tile sub-strategy, microtile_image its
//              2D-sampled-image weight stream (both experimental).
//   coopmat  - VK_KHR_cooperative_matrix hardware MMA (capability-gated,
//              opt-in experimental).
//   probe    - measurement instruments (bw/stream/null/vec_load/alu),
//              never part of the matmul family ladder.

#include <algorithm>
#include <array>
#include <string_view>

namespace bitnet {

enum class KernelFamily : std::uint8_t { Bitplane, Dot, Plane, CoopMat, Probe };
enum class KernelAssignment : std::uint8_t {
    RowSerial,      // gemv_bitplane_rowserial: one thread per row, full-K serial
    Naive,          // gemm_bitplane_naive: one thread per output, full-K serial
    KSplit,         // *_ksplit: K-split partials + integer atomics
    RowCoop,        // gemv_*_rowcoop: one workgroup per row, groupshared reduce
    CellCoop,       // gemm_bitplane_cellcoop: one workgroup per output cell
    Tiled,          // *_tiled: TILE_M x TILE_N blocking, K-chunk groupshared staging
    Microtile,      // gemm_plane_microtile: register micro-tile (RM x RN/thread)
    MicrotileImage, // gemm_plane_microtile_image: microtile + sampled-image weights
    CoopMat,        // gemm_coopmat: hardware cooperative-matrix MMA
    Probe,          // measurement instruments
};

struct KernelInfo {
    std::string_view name;
    KernelFamily family{};
    KernelAssignment assignment{};
    bool is_gemv{}; // false = gemm or probe
};

// Exact-match table: every shipped kernel entry point, once. Using
// std::to_array (C++20) so the size is deduced - no hand-counted
// template argument to drift from the initializer list.
inline constexpr auto kKernelTable = std::to_array<KernelInfo>({
    {"gemv_bitplane_rowserial", KernelFamily::Bitplane, KernelAssignment::RowSerial, true},
    {"gemv_bitplane_ksplit", KernelFamily::Bitplane, KernelAssignment::KSplit, true},
    {"gemv_bitplane_ksplit_il", KernelFamily::Bitplane, KernelAssignment::KSplit, true},
    {"gemv_bitplane_rowcoop", KernelFamily::Bitplane, KernelAssignment::RowCoop, true},
    {"gemm_bitplane_naive", KernelFamily::Bitplane, KernelAssignment::Naive, false},
    {"gemm_bitplane_ksplit", KernelFamily::Bitplane, KernelAssignment::KSplit, false},
    {"gemm_bitplane_cellcoop", KernelFamily::Bitplane, KernelAssignment::CellCoop, false},
    {"gemm_bitplane_tiled", KernelFamily::Bitplane, KernelAssignment::Tiled, false},
    {"gemv_dot_rowcoop", KernelFamily::Dot, KernelAssignment::RowCoop, true},
    {"gemm_dot_tiled", KernelFamily::Dot, KernelAssignment::Tiled, false},
    {"gemv_plane_rowcoop", KernelFamily::Plane, KernelAssignment::RowCoop, true},
    {"gemm_plane_tiled", KernelFamily::Plane, KernelAssignment::Tiled, false},
    {"gemm_plane_microtile", KernelFamily::Plane, KernelAssignment::Microtile, false},
    {"gemm_plane_microtile_image", KernelFamily::Plane, KernelAssignment::MicrotileImage, false},
    {"gemm_coopmat", KernelFamily::CoopMat, KernelAssignment::CoopMat, false},
    {"bw_probe", KernelFamily::Probe, KernelAssignment::Probe, false},
    {"stream_bw_probe", KernelFamily::Probe, KernelAssignment::Probe, false},
    {"null_probe", KernelFamily::Probe, KernelAssignment::Probe, false},
    {"vec_load_probe", KernelFamily::Probe, KernelAssignment::Probe, false},
    {"alu_countbits_probe", KernelFamily::Probe, KernelAssignment::Probe, false},
    {"alu_fma_probe", KernelFamily::Probe, KernelAssignment::Probe, false},
});

// name may carry the "_emulated" module suffix (dot_module_name) or a
// generated-profile decoration - strip nothing here, match the LOGICAL
// name (the suffix is appended only to the on-disk module name, never to
// the logical name callers pass to classification helpers).
[[nodiscard]] inline const KernelInfo* find_kernel_info(std::string_view name) {
    const auto it = std::ranges::find_if(
        kKernelTable, [name](const KernelInfo& info) { return name == info.name; });
    return it == kKernelTable.end() ? nullptr : &*it;
}

[[nodiscard]] inline bool is_bitplane_kernel(std::string_view name) {
    const auto* info = find_kernel_info(name);
    return info != nullptr && info->family == KernelFamily::Bitplane;
}
[[nodiscard]] inline bool is_dot_kernel(std::string_view name) {
    const auto* info = find_kernel_info(name);
    return info != nullptr && info->family == KernelFamily::Dot;
}
[[nodiscard]] inline bool is_plane_kernel(std::string_view name) {
    const auto* info = find_kernel_info(name);
    return info != nullptr && info->family == KernelFamily::Plane;
}
[[nodiscard]] inline bool is_coopmat_kernel(std::string_view name) {
    const auto* info = find_kernel_info(name);
    return info != nullptr && info->family == KernelFamily::CoopMat;
}
[[nodiscard]] inline bool is_microtile_kernel(std::string_view name) {
    const auto* info = find_kernel_info(name);
    return info != nullptr && (info->assignment == KernelAssignment::Microtile ||
                               info->assignment == KernelAssignment::MicrotileImage);
}
[[nodiscard]] inline bool is_microtile_image_kernel(std::string_view name) {
    const auto* info = find_kernel_info(name);
    return info != nullptr && info->assignment == KernelAssignment::MicrotileImage;
}
// The plane family's non-microtile strategies (gemv_plane_rowcoop,
// gemm_plane_tiled): the old bare "ends_with(_v6)" check, now exact.
[[nodiscard]] inline bool is_plane_tiled_family(std::string_view name) {
    return is_plane_kernel(name) && !is_microtile_kernel(name);
}
[[nodiscard]] inline bool is_tiled_kernel(std::string_view name) {
    const auto* info = find_kernel_info(name);
    return info != nullptr && info->assignment == KernelAssignment::Tiled;
}
[[nodiscard]] inline bool is_rowcoop_kernel(std::string_view name) {
    const auto* info = find_kernel_info(name);
    return info != nullptr && info->assignment == KernelAssignment::RowCoop;
}

} // namespace bitnet
