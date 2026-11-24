// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

#pragma once
// Backend-neutral relevant-extension inventory. Presence is reported from
// the driver's own enumeration, never inferred from the device/vendor name.
// Feature/property support still needs its separate query before a kernel
// may consume an extension.
//
// EXTENSION SELECTION PRECEDENCE: KHR > EXT >
// vendor. A ratified KHR extension is preferred for a capability; EXT
// (multi-vendor, non-ratified) second; a single-vendor extension
// (ARM/QCOM/NV) LAST - and only when no KHR/EXT form exists, documented as
// such, never as the long-term target if a KHR form appears later. The
// standing example: cooperative matrices run on VK_KHR_cooperative_matrix;
// cooperative vectors have no registered KHR/EXT form, so the NV extension
// is the decode-rung target by necessity. Current
// variants already comply; this block codifies the rule the variants follow.

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace bitnet {

inline constexpr std::array<std::string_view, 27> kRelevantExtensionNames = {
    // KHR/EXT family (capability targets).
    "VK_KHR_shader_integer_dot_product",
    "VK_KHR_cooperative_matrix",
    "VK_EXT_subgroup_size_control",
    "VK_KHR_8bit_storage",
    "VK_KHR_16bit_storage",
    "VK_KHR_shader_subgroup_extended_types",
    "VK_KHR_shader_subgroup_rotate",
    "VK_KHR_shader_expect_assume",
    "VK_EXT_descriptor_heap",
    "VK_KHR_pipeline_binary",
    "VK_KHR_shader_untyped_pointers",
    // ARM family
    "VK_ARM_shader_core_properties",
    "VK_ARM_tensors",
    "VK_ARM_tensor_controls",
    "VK_ARM_cooperative_matrix_layouts",
    "VK_ARM_performance_counters_by_region",
    // QCOM family
    "VK_QCOM_image_processing",
    "VK_QCOM_image_processing2",
    "VK_QCOM_image_processing3",
    "VK_QCOM_tile_properties",
    // NV family (desktop compute surface; all names enumerated by the RTX
    // 5070 driver, capability dump of poc_tests --caps-out - the
    // per-device present/absent split is the evidence). The one deliberate
    // KHR-less capability target lives HERE: no VK_KHR_cooperative_vector
    // exists in the registry - VK_NV_cooperative_vector is the only
    // registered cooperative-vector extension, so it is the decode-rung
    // target by necessity.
    // VK_NV_cooperative_matrix_decode_vector is enumerated by the RTX 5070
    // driver but absent from the published registry (checked 2026-11-12) -
    // tracked as enumerated evidence, not as a capability target.
    "VK_NV_cooperative_vector",
    "VK_NV_cooperative_matrix2",
    "VK_NV_cooperative_matrix_decode_vector",
    "VK_NV_compute_shader_derivatives",
    "VK_NV_raw_access_chains",
    "VK_NV_shader_sm_builtins",
    "VK_NV_shader_subgroup_partitioned",
};

// Family summary boundaries for print_relevant_extensions (test_main):
// [0, kKhrExtFamilyCount) KHR/EXT, [kArmFamilyBegin, kArmFamilyBegin +
// kArmFamilyCount) ARM, [kQcomFamilyBegin, kQcomFamilyBegin +
// kQcomFamilyCount) QCOM, [kNvFamilyBegin, kNvFamilyBegin + kNvFamilyCount)
// NV.
inline constexpr size_t kKhrExtFamilyCount = 11;
inline constexpr size_t kArmFamilyBegin = kKhrExtFamilyCount;
inline constexpr size_t kArmFamilyCount = 5;
inline constexpr size_t kQcomFamilyBegin = kArmFamilyBegin + kArmFamilyCount;
inline constexpr size_t kQcomFamilyCount = 4;
inline constexpr size_t kNvFamilyBegin = kQcomFamilyBegin + kQcomFamilyCount;
inline constexpr size_t kNvFamilyCount = 7;

[[nodiscard]] inline bool extension_present(const std::vector<std::string>& extensions,
                                            std::string_view wanted) {
    return std::ranges::find(extensions, wanted) != extensions.end();
}

} // namespace bitnet
