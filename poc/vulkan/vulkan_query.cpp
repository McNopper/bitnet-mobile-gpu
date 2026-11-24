// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Device capability queries: properties, features, the integer-dot and
// cooperative-matrix surface, and the DeviceCaps aggregation the harness
// reads (caps probe and caps-conformance gate).

// MSVC's CRT marks getenv/fopen deprecated (C4996); they are the portable
// choice for this TU (Windows desktop + Android arm64) and are used safely
// (no writes through the returned env pointer, read-only fopen). The project
// allows a suppression only with this comment + rationale.
#ifdef _MSC_VER
// _CRT_SECURE_NO_WARNINGS is the CRT's own documented macro name, not ours
// to rename (the reserved-identifier autofix broke the build).
// NOLINTNEXTLINE(bugprone-reserved-identifier)
#define _CRT_SECURE_NO_WARNINGS
#endif

// Vulkan structs are partially initialized by design (sType pattern);
// documented exception (one TU-local pragma, not a global flag).
// -Wmissing-designated-field-initializers landed in clang 20 (VS 18 ships
// clang 22; NDK r27 clang 18 lacks it). Version guard instead of
// __has_warning: cppcheck parses every preprocessor branch and cannot
// evaluate function-like builtin macros; MSVC rejects them too.
#ifdef __clang__
#if __clang_major__ >= 20
#pragma clang diagnostic ignored "-Wmissing-designated-field-initializers"
#endif

// MSVC CRT deprecations for getenv/fopen are noise: both are the portable
// Windows+Android choices for this TU.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

#include "backend/backend.hpp"
#include "common/format.hpp"
#include "vulkan/vulkan_backend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace bitnet::vulkan {

void VulkanBackend::query_device_properties() {
    const auto get_properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceProperties2"));
    const auto get_properties2_khr = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2KHR>(
        vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceProperties2KHR"));
    if (get_properties2 != nullptr || get_properties2_khr != nullptr) {
        subgroup_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        // maintenance3 carries maxMemoryAllocationSize (core 1.1; the
        // caps-conformance buffer-limit check consumes it - unknown
        // struct types are ignored by older drivers and the field stays
        // 0, which the conformance check reports as absent).
        maint3_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
        VkPhysicalDeviceProperties2 props2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                           .pNext = nullptr,
                                           .properties = {}};
        auto* props_tail = reinterpret_cast<VkBaseOutStructure*>(&props2);
        const auto append_props = [&props_tail](auto& item) {
            props_tail->pNext = reinterpret_cast<VkBaseOutStructure*>(&item);
            props_tail = props_tail->pNext;
        };
        append_props(maint3_);
        append_props(subgroup_);
        if (has_subgroup_size_control_) {
            subgroup_size_properties_.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
            append_props(subgroup_size_properties_);
        }
        if (has_arm_shader_core_properties_) {
            arm_core_properties_.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_ARM;
            append_props(arm_core_properties_);
        }
        if (has_qcom_image_processing_) {
            qcom_image_properties_.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_PROPERTIES_QCOM;
            append_props(qcom_image_properties_);
        }
        if (has_cooperative_matrix_ext_) {
            coop_matrix_properties_.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            append_props(coop_matrix_properties_);
        }
        if (get_properties2 != nullptr) {
            get_properties2(physical_device_, &props2);
        } else {
            get_properties2_khr(physical_device_, &props2);
        }
        properties_ = props2.properties;
    } else {
        vkGetPhysicalDeviceProperties(physical_device_, &properties_);
    }
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_);
}

void VulkanBackend::query_device_features() {
    const auto get_properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceProperties2"));
    const auto get_properties2_khr = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2KHR>(
        vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceProperties2KHR"));
    if (get_properties2 == nullptr && get_properties2_khr == nullptr) {
        return; // 1.0 fallback route: no Features2 - emulated dot serves
    }
    const auto get_features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
        vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceFeatures2"));
    const auto get_features2_khr = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(
        vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceFeatures2KHR"));
    if (get_features2 == nullptr && get_features2_khr == nullptr) {
        return;
    }
    VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR dot_props{};
    dot_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR;
    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dot_feature_query{};
    dot_feature_query.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
    VkPhysicalDeviceVulkan13Features v13_query{};
    v13_query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceFeatures2 features_query{};
    features_query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    auto* features_tail = reinterpret_cast<VkBaseOutStructure*>(&features_query);
    const auto append_features = [&features_tail](auto& item) {
        features_tail->pNext = reinterpret_cast<VkBaseOutStructure*>(&item);
        features_tail = features_tail->pNext;
    };
    if (has_dot_product_ext_ || properties_.apiVersion >= VK_API_VERSION_1_3) {
        append_features(dot_props);
    }
    if (has_dot_product_ext_) {
        append_features(dot_feature_query);
    }
    if (properties_.apiVersion >= VK_API_VERSION_1_3) {
        append_features(v13_query);
    }
    if (has_subgroup_size_control_) {
        subgroup_size_features_.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
        append_features(subgroup_size_features_);
    }
    if (has_8bit_storage_) {
        storage_8bit_features_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
        append_features(storage_8bit_features_);
    }
    if (has_16bit_storage_) {
        storage_16bit_features_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        append_features(storage_16bit_features_);
    }
    if (has_qcom_image_processing_) {
        qcom_image_features_.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_PROCESSING_FEATURES_QCOM;
        append_features(qcom_image_features_);
    }
    if (has_cooperative_matrix_ext_) {
        coop_matrix_features_.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        append_features(coop_matrix_features_);
    }
    if (properties_.apiVersion >= VK_API_VERSION_1_2) {
        // Core-1.2 feature support for the coop-mat SPIR-V capabilities
        // (queried even when the extension is absent; only REPORTED).
        coop_vulkan12_features_.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        append_features(coop_vulkan12_features_);
    }
    if (get_features2 != nullptr) {
        get_features2(physical_device_, &features_query);
    } else {
        get_features2_khr(physical_device_, &features_query);
    }
    // Extension presence alone is not support: both the core-1.3 and KHR
    // routes must report the optional shaderIntegerDotProduct feature.
    dot_product_core_ =
        (properties_.apiVersion >= VK_API_VERSION_1_3 &&
         v13_query.shaderIntegerDotProduct == VK_TRUE) ||
        (has_dot_product_ext_ && dot_feature_query.shaderIntegerDotProduct == VK_TRUE);
    dot_packed_signed_accelerated_ =
        dot_props.integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE;
    dot_8bit_signed_accelerated_ = dot_props.integerDotProduct8BitSignedAccelerated == VK_TRUE;
}

void VulkanBackend::query_device_capabilities() {
    const std::vector<VkExtensionProperties> extensions =
        enumerate_device_extensions(physical_device_);
    has_dot_product_ext_ =
        has_extension(extensions, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
    has_cooperative_matrix_ext_ =
        has_extension(extensions, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    // Cooperative vectors: NV-only (no KHR variant registered);
    // recorded as fallback-compat evidence, never
    // selected as a capability target. The spec-registered name literal
    // stands in for the VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME macro:
    // the NDK 27.1 headers predate the extension and do not define it
    // (toolchain difference - the desktop SDK
    // headers do).
    has_cooperative_vector_nv_ext_ = has_extension(extensions, "VK_NV_cooperative_vector");
    has_subgroup_size_control_ =
        has_extension(extensions, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    has_8bit_storage_ = has_extension(extensions, VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
    has_16bit_storage_ = has_extension(extensions, VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
    has_arm_shader_core_properties_ =
        has_extension(extensions, VK_ARM_SHADER_CORE_PROPERTIES_EXTENSION_NAME);
    has_qcom_image_processing_ = has_extension(extensions, VK_QCOM_IMAGE_PROCESSING_EXTENSION_NAME);

    // Capability ladder floor (kMinApi10): GetPhysicalDeviceProperties2 /
    // Features2 are core since 1.1; on a 1.0 driver they resolve through
    // VK_KHR_get_physical_device_properties2 (enabled at instance level
    // when present). Without either route, the plain 1.0 properties call
    // is used and the 1.1-only flags stay unset - the popcount path
    // still runs (degrade, don't skip).
    query_device_properties();

    // Core feature query (Vulkan 1.3): shaderIntegerDotProduct lives in
    // VkPhysicalDeviceVulkan13Features - no extension struct needed.
    // The ACCELERATION properties ride the same query chain: feature=TRUE
    // alone does not guarantee the op runs natively (if packed-dot is
    // not "accelerated", the driver may lower SDot to per-byte ops).
    // Skipped on the 1.0 fallback route (no Properties2): the dot
    // capability stays false and the emulated-dot variant serves the
    // integer path.
    query_device_features();

    // VK_KHR_cooperative_matrix enumerated shapes (the coop-mat lane's
    // BEFORE-COMPILING validation: the module loads only when the driver
    // itself lists the exact sint8->sint32 shape it was compiled for).
    if (has_cooperative_matrix_ext_) {
        query_cooperative_matrix_shapes();
    }
}

void VulkanBackend::query_cooperative_matrix_shapes() {
    const auto get_shapes = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
        vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    if (get_shapes == nullptr) {
        return;
    }
    uint32_t count = 0;
    if (get_shapes(physical_device_, &count, nullptr) != VK_SUCCESS || count == 0) {
        return;
    }
    std::vector<VkCooperativeMatrixPropertiesKHR> shapes(
        count, VkCooperativeMatrixPropertiesKHR{
                   .sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
                   .pNext = nullptr,
                   .MSize = 0,
                   .NSize = 0,
                   .KSize = 0,
                   .AType = VK_COMPONENT_TYPE_SINT8_KHR,
                   .BType = VK_COMPONENT_TYPE_SINT8_KHR,
                   .CType = VK_COMPONENT_TYPE_SINT32_KHR,
                   .ResultType = VK_COMPONENT_TYPE_SINT32_KHR,
                   .saturatingAccumulation = VK_FALSE,
                   .scope = VK_SCOPE_WORKGROUP_KHR});
    if (get_shapes(physical_device_, &count, shapes.data()) != VK_SUCCESS) {
        return;
    }
    coop_matrix_shape_count_ = count;
    // Prefer the exact gemm_coopmat contract (subgroup scope - NVIDIA's
    // KHR enumeration carries ONLY scope-3 shapes); otherwise record the
    // first sint8->sint32 entry of any scope as caps evidence. The lane
    // contract (variant_selection.hpp) still requires scope 3, so a
    // workgroup-only device falls through.
    const auto s8_to_s32 = [](const VkCooperativeMatrixPropertiesKHR& s) {
        return s.AType == VK_COMPONENT_TYPE_SINT8_KHR && s.BType == VK_COMPONENT_TYPE_SINT8_KHR &&
               s.CType == VK_COMPONENT_TYPE_SINT32_KHR &&
               s.ResultType == VK_COMPONENT_TYPE_SINT32_KHR && s.saturatingAccumulation == VK_FALSE;
    };
    const auto exact =
        std::ranges::find_if(shapes, [&s8_to_s32](const VkCooperativeMatrixPropertiesKHR& s) {
            return s8_to_s32(s) && s.scope == VK_SCOPE_SUBGROUP_KHR;
        });
    const auto fallback = std::ranges::find_if(
        shapes, [&s8_to_s32](const VkCooperativeMatrixPropertiesKHR& s) { return s8_to_s32(s); });
    const auto chosen = exact != shapes.end() ? exact : fallback;
    if (chosen != shapes.end()) {
        coop_matrix_shape_ = *chosen;
        has_coop_matrix_shape_ = true;
    }
}

[[nodiscard]] DeviceCaps VulkanBackend::build_caps() const {
    DeviceCaps caps;
    caps.device_name = properties_.deviceName;
    caps.driver_version = format_driver_version(properties_.vendorID, properties_.driverVersion);
    caps.api_version = properties_.apiVersion;
    caps.max_shared_memory_bytes = properties_.limits.maxComputeSharedMemorySize;
    caps.max_workgroup_invocations = properties_.limits.maxComputeWorkGroupInvocations;
    caps.subgroup_size = subgroup_.subgroupSize;
    caps.subgroup_arithmetic =
        (subgroup_.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    caps.shader_integer_dot_product = dot_product_core_;
    caps.dot_4x8_packed_signed_accelerated = dot_packed_signed_accelerated_;
    caps.dot_8bit_signed_accelerated = dot_8bit_signed_accelerated_;
    caps.cooperative_matrix = has_cooperative_matrix_ext_;
    caps.cooperative_vector_nv = has_cooperative_vector_nv_ext_;
    caps.coop_mat_shape_s8_s32 =
        has_coop_matrix_shape_ && properties_.apiVersion >= VK_API_VERSION_1_2 &&
        (coop_matrix_properties_.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) !=
            0 &&
        coop_matrix_features_.cooperativeMatrix == VK_TRUE &&
        coop_vulkan12_features_.shaderInt8 == VK_TRUE &&
        coop_vulkan12_features_.vulkanMemoryModel == VK_TRUE &&
        coop_vulkan12_features_.vulkanMemoryModelDeviceScope == VK_TRUE;
    caps.coop_mat_msize = coop_matrix_shape_.MSize;
    caps.coop_mat_nsize = coop_matrix_shape_.NSize;
    caps.coop_mat_ksize = coop_matrix_shape_.KSize;
    caps.coop_mat_scope = coop_matrix_shape_.scope;
    caps.coop_mat_shape_count = coop_matrix_shape_count_;
    caps.coop_mat_supported_stages = coop_matrix_properties_.cooperativeMatrixSupportedStages;
    caps.storage_8bit = storage_8bit_features_.storageBuffer8BitAccess == VK_TRUE;
    caps.storage_16bit = storage_16bit_features_.storageBuffer16BitAccess == VK_TRUE;
    caps.subgroup_size_control = subgroup_size_features_.subgroupSizeControl == VK_TRUE;
    caps.compute_full_subgroups = subgroup_size_features_.computeFullSubgroups == VK_TRUE;
    caps.min_subgroup_size = subgroup_size_properties_.minSubgroupSize;
    caps.max_subgroup_size = subgroup_size_properties_.maxSubgroupSize;
    caps.max_compute_workgroup_subgroups = subgroup_size_properties_.maxComputeWorkgroupSubgroups;
    caps.required_subgroup_size_compute =
        (subgroup_size_properties_.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    caps.arm_shader_core_properties = has_arm_shader_core_properties_;
    caps.arm_pixel_rate = arm_core_properties_.pixelRate;
    caps.arm_texel_rate = arm_core_properties_.texelRate;
    caps.arm_fma_rate = arm_core_properties_.fmaRate;
    caps.qcom_image_processing = has_qcom_image_processing_;
    caps.qcom_texture_sample_weighted = qcom_image_features_.textureSampleWeighted == VK_TRUE;
    caps.qcom_texture_box_filter = qcom_image_features_.textureBoxFilter == VK_TRUE;
    caps.qcom_texture_block_match = qcom_image_features_.textureBlockMatch == VK_TRUE;
    caps.qcom_max_weight_filter_phases = qcom_image_properties_.maxWeightFilterPhases;
    caps.qcom_max_weight_filter_width = qcom_image_properties_.maxWeightFilterDimension.width;
    caps.qcom_max_weight_filter_height = qcom_image_properties_.maxWeightFilterDimension.height;
    caps.qcom_max_block_match_width = qcom_image_properties_.maxBlockMatchRegion.width;
    caps.qcom_max_block_match_height = qcom_image_properties_.maxBlockMatchRegion.height;
    caps.qcom_max_box_filter_width = qcom_image_properties_.maxBoxFilterBlockSize.width;
    caps.qcom_max_box_filter_height = qcom_image_properties_.maxBoxFilterBlockSize.height;
    // Caps-conformance limits (VkPhysicalDeviceLimits - already in the
    // queried Properties2 payload, no extra Vulkan calls).
    for (int i = 0; i < 3; ++i) {
        caps.max_workgroup_size[i] = (&properties_.limits.maxComputeWorkGroupSize[0])[i];
        caps.max_workgroup_count[i] = (&properties_.limits.maxComputeWorkGroupCount[0])[i];
    }
    caps.max_storage_buffer_range = properties_.limits.maxStorageBufferRange;
    caps.max_memory_allocation_size = maint3_.maxMemoryAllocationSize;
    return caps;
}

} // namespace bitnet::vulkan
