// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

#pragma once
// Capability-driven kernel-variant selection. Selection consumes CAPABILITY FLAGS from the caps
// probe - never device names; two different GPUs with the same capability
// set run the same variants with possibly different knob values (device
// profiles stay knob tables only). The popcount/bitplane kernels are the
// always-portable base and need no selection; only the packed integer-dot
// family has two compiled modules per profile:
//   dot            - hardware dot4add_i8packed (needs shaderIntegerDotProduct:
//                    core 1.3 OR VK_KHR_shader_integer_dot_product)
//   dot-emulated   - the SAME sources with -DNO_INT_DOT: own shift/mask/
//                    extract multiply-accumulate per int8 lane, bit-exact,
//                    ALWAYS available (slow path, correctness only)
//   popcount       - run only the popcount lane; the dot family is skipped
// A forced variant (--variant CLI, else BITNET_FORCE_VARIANT env; CLI wins;
// BITNET_FORCE_NO_DOT=1 is the legacy alias for dot-emulated) overrides the
// caps view for the whole run and FAILS LOUDLY when the required capability
// is missing - no silent fallback (only dot-emulated is unconditionally
// runnable by design).

#include "backend/backend.hpp"
#include "backend/kernel_family.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace bitnet {

enum class DotVariant { Popcount, HardwareDot, EmulatedDot };

[[nodiscard]] inline const char* dot_variant_name(DotVariant v) {
    switch (v) {
    case DotVariant::Popcount:
        return "popcount";
    case DotVariant::HardwareDot:
        return "dot";
    case DotVariant::EmulatedDot:
        return "dot-emulated";
    }
    return "unknown";
}

// "popcount" | "dot" | "dot-emulated" -> enum; false on anything else.
[[nodiscard]] inline bool parse_dot_variant(std::string_view name, DotVariant& out) {
    if (name == "popcount") {
        out = DotVariant::Popcount;
        return true;
    }
    if (name == "dot") {
        out = DotVariant::HardwareDot;
        return true;
    }
    if (name == "dot-emulated") {
        out = DotVariant::EmulatedDot;
        return true;
    }
    return false;
}

// Caps-driven default: hardware packed dot iff the device provides
// shaderIntegerDotProduct, emulated otherwise - the integer path runs on
// EVERY device, slow but correct without the capability.
[[nodiscard]] inline DotVariant auto_dot_variant(bool shader_integer_dot_product) {
    return shader_integer_dot_product ? DotVariant::HardwareDot : DotVariant::EmulatedDot;
}

// Module (.spv) name for a logical kernel name under the selected variant:
// the dot family (gemv_dot_rowcoop / gemm_dot_tiled) has two modules per
// profile, every other kernel one. The suffix keeps the LocalSize guard
// working (kernel_threads.h's exact-match table recognizes both names).
[[nodiscard]] inline std::string dot_module_name(std::string_view logical_name,
                                                 DotVariant variant) {
    if (variant == DotVariant::EmulatedDot && is_dot_kernel(logical_name)) {
        return std::string(logical_name).append("_emulated");
    }
    return std::string(logical_name);
}

// Coop-mat MMA lane contract (gemm_coopmat in bitnet_kernels.slang): the exact
// shape the module is compiled for. A device runs the lane only when its
// vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR enumeration contains
// EXACTLY this shape (sint8*sint8 -> sint32, non-saturating, SUBGROUP scope
// - the KHR enum struct carries no row/col-major fields) and the SPIR-V's
// feature dependencies query TRUE. The fixed-32-lane requirement matches the
// kernel's tile mapping (GEMM_COOPMAT_SUBGROUPS tiles per workgroup under
// 32-lane subgroups). Measured enumeration (RTX 5070): 21
// shapes, all scope 3 (subgroup); the 16x16x32 s8->s32 entry is the match.
// Keep the constants in sync with the #ifndef defaults in the Slang source
// (the runtime caps check is the authority; a mismatched device degrades to
// gemm_plane_tiled, never mis-dispatches). Operands: plain padded row-major int8
// (PackedInt8MatPair in format.hpp); KernelParams.kwords = padded raw K.
constexpr uint32_t kCoopMatMSize = 16;
constexpr uint32_t kCoopMatNSize = 16;
constexpr uint32_t kCoopMatKSize = 32;
constexpr uint32_t kCoopMatScopeSubgroup = 3; // VK_SCOPE_SUBGROUP_KHR
constexpr uint32_t kCoopMatSubgroups = 1;     // one 32-lane subgroup per workgroup
constexpr uint32_t kCoopMatThreads = 32;      // GEMM_COOPMAT_THREADS (tutorial geometry)

[[nodiscard]] inline bool coop_mat_shape_matches(const DeviceCaps& caps) {
    // storage_8bit is part of the contract: the lane's operands are int8
    // SSBOs (the backend chains storageBuffer8BitAccess on the same
    // capability check; a device matching the shape but lacking 8-bit
    // storage would produce a module whose declared capability queries
    // FALSE - skip-noted fall-through, gemm_plane_tiled serves).
    return caps.storage_8bit && caps.coop_mat_shape_s8_s32 &&
           caps.coop_mat_msize == kCoopMatMSize && caps.coop_mat_nsize == kCoopMatNSize &&
           caps.coop_mat_ksize == kCoopMatKSize && caps.coop_mat_scope == kCoopMatScopeSubgroup &&
           caps.min_subgroup_size == 32 && caps.max_subgroup_size == 32;
}

} // namespace bitnet
