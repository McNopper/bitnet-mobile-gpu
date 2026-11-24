// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Device objects of the Vulkan backend: descriptor set layout, pipeline
// layout and pool, host-visible buffers, and their teardown.

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

void VulkanBackend::create_device_objects() {
    const float queue_priority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                       .pNext = nullptr,
                                       .flags = 0,
                                       .queueFamilyIndex = queue_family_,
                                       .queueCount = 1,
                                       .pQueuePriorities = &queue_priority};

    std::vector<const char*> extensions;
    // Integer dot product enablement: this
    // Mali BSP advertises both the core feature and the KHR extension,
    // but REJECTS device creation when the core Vulkan13Features struct
    // is chained (-3 EXTENSION_NOT_PRESENT with an empty extension
    // list). The spec-compliant extension route works: enable
    // VK_KHR_shader_integer_dot_product + chain its own feature struct.
    // Desktop NVIDIA accepts both routes; we use one path everywhere.
    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dot_features{};
    dot_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
    dot_features.shaderIntegerDotProduct = dot_product_core_ ? VK_TRUE : VK_FALSE;
    // Cooperative-matrix lane enablement (least privilege): the feature
    // structs are chained ONLY when the enumerated shape matched AND the
    // SPIR-V's core-1.2 feature dependencies are supported. The core-1.2
    // struct is chained only on a 1.2+ device (spec valid usage); the
    // Vulkan 1.1 Mali lane never reaches this (extension absent). 8-bit
    // SSBO storage is part of the contract since the operands are int8
    // buffers (the glslang-compiled twin and the staged kernels declare
    // int8_t SSBO arrays; the slang direct-load module never surfaced
    // the requirement because its loads were dropped entirely).
    const bool coop_api_ok = properties_.apiVersion >= VK_API_VERSION_1_2;
    // slangc declares BOTH int8-storage capabilities for the staged
    // module (int8 SSBO operand views + int8 workgroup staging tiles),
    // so both query TRUE before the lane may be enabled.
    const bool coop_8bit_ok = has_8bit_storage_ &&
                              storage_8bit_features_.storageBuffer8BitAccess == VK_TRUE &&
                              storage_8bit_features_.uniformAndStorageBuffer8BitAccess == VK_TRUE;
    const bool use_coop_mat = has_coop_matrix_shape_ && coop_api_ok && coop_8bit_ok &&
                              coop_matrix_features_.cooperativeMatrix == VK_TRUE &&
                              coop_vulkan12_features_.shaderInt8 == VK_TRUE &&
                              coop_vulkan12_features_.vulkanMemoryModel == VK_TRUE &&
                              coop_vulkan12_features_.vulkanMemoryModelDeviceScope == VK_TRUE;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_enable{};
    VkPhysicalDeviceVulkan12Features v12_enable{};
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_enable = subgroup_size_features_;
    if (use_coop_mat) {
        coop_enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        coop_enable.pNext = nullptr;
        coop_enable.cooperativeMatrix = VK_TRUE;
        coop_enable.cooperativeMatrixRobustBufferAccess = VK_FALSE;
        // storageBuffer8BitAccess lives INSIDE the core-1.2 struct on a
        // 1.2+ device (VUID-VkDeviceCreateInfo-pNext-02830 forbids
        // chaining VkPhysicalDevice8BitStorageFeatures alongside it).
        v12_enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        v12_enable.pNext = nullptr;
        v12_enable.shaderInt8 = VK_TRUE;
        v12_enable.vulkanMemoryModel = VK_TRUE;
        v12_enable.vulkanMemoryModelDeviceScope = VK_TRUE;
        v12_enable.storageBuffer8BitAccess = VK_TRUE;
        v12_enable.uniformAndStorageBuffer8BitAccess = VK_TRUE;
        coop_enable.pNext = &v12_enable;
        subgroup_enable.pNext = &coop_enable;
    } else {
        subgroup_enable.pNext = nullptr;
    }
    dot_features.pNext = has_subgroup_size_control_ ? &subgroup_enable : nullptr;
    VkPhysicalDeviceFeatures2 features2_enable{};
    features2_enable.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2_enable.pNext = &dot_features;
    features2_enable.features = {}; // no classic features required (v1)

    const bool use_dot = dot_product_core_;
    const bool use_features2 = use_dot || has_subgroup_size_control_ || use_coop_mat;
    if (use_dot && has_dot_product_ext_) {
        extensions.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
    }
    if (use_coop_mat && has_8bit_storage_) {
        extensions.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
    }
    if (has_cooperative_matrix_ext_) {
        extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }
    if (has_subgroup_size_control_) {
        extensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    }

    VkPhysicalDeviceFeatures features{}; // v1: no extra features required

    // Spec rule: when a Features2 struct is in pNext, pEnabledFeatures
    // must be null - handled by the ternary below.
    VkDeviceCreateInfo device_info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = use_features2 ? static_cast<void*>(&features2_enable) : nullptr,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data(),
        .pEnabledFeatures = use_features2 ? nullptr : &features};
    require_vk(vkCreateDevice(physical_device_, &device_info, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

    VkCommandPoolCreateInfo pool_info{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      .pNext = nullptr,
                                      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                      .queueFamilyIndex = queue_family_};
    require_vk(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_),
               "vkCreateCommandPool");

    VkCommandBufferAllocateInfo command_info{.sType =
                                                 VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                             .pNext = nullptr,
                                             .commandPool = command_pool_,
                                             .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                             .commandBufferCount = 1};
    require_vk(vkAllocateCommandBuffers(device_, &command_info, &command_buffer_),
               "vkAllocateCommandBuffers");

    VkFenceCreateInfo fence_info{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = nullptr, .flags = 0};
    require_vk(vkCreateFence(device_, &fence_info, nullptr, &fence_), "vkCreateFence");

    VkQueryPoolCreateInfo query_info{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                     .pNext = nullptr,
                                     .flags = 0,
                                     .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                     .queryCount = kQueryCount,
                                     .pipelineStatistics = 0};
    require_vk(vkCreateQueryPool(device_, &query_info, nullptr, &query_pool_), "vkCreateQueryPool");

    // One set layout shared by all kernels (binding table, set 0):
    // 0/1/2 storage (weights, activations, accumulators), 3 uniform params,
    // 4 uniform texel buffer — the v7 texture-path weight stream
    //. Kernels that don't declare binding 4
    // are unaffected; the binding stays valid-but-unwritten in their
    // sets (statically unused descriptors need no initialization).
    std::array<VkDescriptorSetLayoutBinding, kBindingCount> bindings{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i] = {.binding = i,
                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                       .descriptorCount = 1,
                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                       .pImmutableSamplers = nullptr};
    }
    bindings[3] = {.binding = 3,
                   .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                   .descriptorCount = 1,
                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                   .pImmutableSamplers = nullptr};
    bindings[4] = {.binding = 4,
                   .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
                   .descriptorCount = 1,
                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                   .pImmutableSamplers = nullptr};
    bindings[kBindingWeightImage] = {.binding = kBindingWeightImage,
                                     .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                     .descriptorCount = 1,
                                     .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                     .pImmutableSamplers = nullptr};
    VkDescriptorSetLayoutCreateInfo layout_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = kBindingCount,
        .pBindings = bindings.data()};
    require_vk(vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &set_layout_),
               "vkCreateDescriptorSetLayout");

    // No push constants in v1.
    VkPipelineLayoutCreateInfo pipeline_layout_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout_,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr};
    require_vk(vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_),
               "vkCreatePipelineLayout");

    // Batched sequences (make_bindings) allocate one set per call site:
    // 16 sites = the single-dispatch set + 15 sequence entries.
    constexpr uint32_t kMaxDescriptorSets = 16;
    constexpr uint32_t kStoragePerSet = 3;
    constexpr uint32_t kUniformPerSet = 1;
    std::array<VkDescriptorPoolSize, 4> pool_sizes = {
        {{.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = kStoragePerSet * kMaxDescriptorSets},
         {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          .descriptorCount = kUniformPerSet * kMaxDescriptorSets},
         {.type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, .descriptorCount = kMaxDescriptorSets},
         {.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = kMaxDescriptorSets}}};
    VkDescriptorPoolCreateInfo pool_create_info{.sType =
                                                    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                .pNext = nullptr,
                                                .flags = 0,
                                                .maxSets = kMaxDescriptorSets,
                                                .poolSizeCount = 4,
                                                .pPoolSizes = pool_sizes.data()};
    require_vk(vkCreateDescriptorPool(device_, &pool_create_info, nullptr, &descriptor_pool_),
               "vkCreateDescriptorPool");

    timestamps_usable_ = properties_.limits.timestampPeriod > 0.0F &&
                         properties_.limits.timestampComputeAndGraphics != VK_FALSE &&
                         queue_timestamp_bits_ > 0;
}

std::unique_ptr<VulkanBuffer> VulkanBackend::create_buffer_impl(uint64_t bytes, bool host_visible,
                                                                VkBufferUsageFlags extra_usage) {
    // UNIFORM_TEXEL_BUFFER usage is baked into every storage buffer so
    // a VkBufferView (texture-path weight stream) can be
    // created over any of them without a separate allocation path.
    const VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra_usage;

    // RAII so any failure below cleans up what was already created.
    // device is captured at construction so ~Handles() frees on the
    // right VkDevice even on early throws.
    struct Handles {
        VkDevice device = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        ~Handles() {
            if (memory != VK_NULL_HANDLE) {
                vkFreeMemory(device, memory, nullptr);
            }
            if (buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device, buffer, nullptr);
            }
        }
    } handles{.device = device_};

    VkBufferCreateInfo buffer_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                   .pNext = nullptr,
                                   .flags = 0,
                                   .size = static_cast<VkDeviceSize>(bytes),
                                   .usage = usage,
                                   .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                                   .queueFamilyIndexCount = 0,
                                   .pQueueFamilyIndices = nullptr};
    require_vk(vkCreateBuffer(device_, &buffer_info, nullptr, &handles.buffer), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, handles.buffer, &requirements);

    // HOST_VISIBLE|HOST_COHERENT first (guaranteed to exist per spec),
    // then any HOST_VISIBLE; device-local otherwise.
    uint32_t type_index = UINT32_MAX;
    if (host_visible) {
        type_index =
            find_memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (type_index == UINT32_MAX) {
            type_index =
                find_memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        }
    } else {
        type_index =
            find_memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }
    if (type_index == UINT32_MAX) {
        throw std::runtime_error("bitnet vulkan: no suitable memory type for a " +
                                 std::string(host_visible ? "host-visible" : "device-local") +
                                 " buffer");
    }

    const VkMemoryPropertyFlags type_props = memory_.memoryTypes[type_index].propertyFlags;
    VkMemoryAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                    .pNext = nullptr,
                                    .allocationSize =
                                        std::max(bytes, static_cast<uint64_t>(requirements.size)),
                                    .memoryTypeIndex = type_index};
    require_vk(vkAllocateMemory(device_, &alloc_info, nullptr, &handles.memory),
               "vkAllocateMemory");
    require_vk(vkBindBufferMemory(device_, handles.buffer, handles.memory, 0),
               "vkBindBufferMemory");

    // Persistent mapping happens in the VulkanBuffer constructor.
    auto buffer =
        std::make_unique<VulkanBuffer>(device_, handles.buffer, handles.memory, bytes, host_visible,
                                       (type_props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0,
                                       host_visible ? &live_host_buffers_ : nullptr);
    handles.buffer = VK_NULL_HANDLE; // ownership moved into the VulkanBuffer
    handles.memory = VK_NULL_HANDLE;
    return buffer;
}

void VulkanBackend::ensure_uniform_buffer() {
    if (uniform_ == nullptr) {
        uniform_ = create_buffer_impl(kMaxUniformBytes, true, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    }
}

void VulkanBackend::flush_buffer(VulkanBuffer& buffer) {
    VkMappedMemoryRange range{.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                              .pNext = nullptr,
                              .memory = buffer.memory(),
                              .offset = 0,
                              .size = VK_WHOLE_SIZE};
    require_vk(vkFlushMappedMemoryRanges(device_, 1, &range), "vkFlushMappedMemoryRanges");
}

void VulkanBackend::flush_live_host_buffers() {
    for (VulkanBuffer* buffer : live_host_buffers_) {
        flush_buffer(*buffer);
    }
}

void VulkanBackend::destroy_all() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }
    uniform_.reset();          // must die before the device
    binding_uniforms_.clear(); // per-site params slots: before the device
    for (const WeightImage& wi : weight_images_) {
        vkDestroyImageView(device_, wi.view, nullptr);
        vkDestroyImage(device_, wi.image, nullptr);
        vkFreeMemory(device_, wi.memory, nullptr);
    }
    weight_images_.clear();
    for (const auto& kernel : pipelines_) {
        if (kernel.second != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, kernel.second, nullptr);
        }
    }
    pipelines_.clear();
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr); // frees the set too
    }
    if (set_layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    }
    if (query_pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device_, query_pool_, nullptr);
    }
    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_, fence_, nullptr);
    }
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr); // frees the command buffer
    }
    // Free lingering buffers before the device (validation gate: no leaks).
    dummy_buffer_.reset();
    uniform_.reset();
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }
    if (messenger_ != VK_NULL_HANDLE && destroy_messenger_ != nullptr) {
        destroy_messenger_(instance_, messenger_, nullptr);
        messenger_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

} // namespace bitnet::vulkan
