// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Dispatch and timing of the Vulkan backend: one submit per dispatch, the
// per-iteration timing loop (host wall clock around submit + fence, plus a
// GPU timestamp pair per iteration when the driver reports usable
// timestamps), and the descriptor/image bindings that carry it (sticky
// single-dispatch and batched per-site sets).

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

// Nanoseconds per millisecond: the timestamp period the driver reports is
// nanoseconds per tick, and the bench contract reports milliseconds.
constexpr double kNanosecondsPerMillisecond = 1.0e6;

void VulkanBackend::dispatch(const Dispatch& d) {
    auto* const pipeline = static_cast<VkPipeline>(d.kernel);
    if (pipeline == VK_NULL_HANDLE) {
        throw std::runtime_error("bitnet vulkan: dispatch() without a kernel (load_kernel first)");
    }
    if (!descriptors_written_) {
        throw std::runtime_error("bitnet vulkan: dispatch() before bind()");
    }
    ensure_uniform_buffer();
    // HOST_VISIBLE without HOST_COHERENT is a legal fallback memory type;
    // make pending host writes visible before every submit.
    flush_live_host_buffers();

    require_vk(vkResetCommandBuffer(command_buffer_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                        .pNext = nullptr,
                                        .flags = 0,
                                        .pInheritanceInfo = nullptr};
    require_vk(vkBeginCommandBuffer(command_buffer_, &begin_info), "vkBeginCommandBuffer");
    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1,
                            &descriptor_set_, 0, nullptr);
    vkCmdDispatch(command_buffer_, d.groups[0], d.groups[1], d.groups[2]);
    require_vk(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");
    submit_and_fence();
}

BenchStats VulkanBackend::dispatch_bench_stats(const Dispatch& d, uint32_t count) {
    BenchStats stats{};
    if (count == 0) {
        return stats;
    }
    auto* const pipeline = static_cast<VkPipeline>(d.kernel);
    if (pipeline == VK_NULL_HANDLE) {
        throw std::runtime_error(
            "bitnet vulkan: dispatch_bench() without a kernel (load_kernel first)");
    }
    if (!descriptors_written_) {
        throw std::runtime_error("bitnet vulkan: dispatch_bench() before bind()");
    }
    ensure_uniform_buffer();
    flush_live_host_buffers();

    VkCommandBufferBeginInfo begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                        .pNext = nullptr,
                                        .flags = 0,
                                        .pInheritanceInfo = nullptr};
    std::vector<double> iter_ms;
    iter_ms.reserve(count);
    std::vector<double> gpu_iter_ms;
    gpu_iter_ms.reserve(timestamps_usable_ ? count : 0U);
    // Wrap mask: timestampValidBits can be < 64, so the delta is taken
    // modulo 2^validBits (correct as long as one iteration spans less than
    // one wrap, which a fence-bounded dispatch always does).
    const uint64_t ts_mask =
        queue_timestamp_bits_ >= 64U ? ~0ULL : ((1ULL << queue_timestamp_bits_) - 1ULL);
    for (uint32_t i = 0; i < count; ++i) {
        const auto iter_start = std::chrono::steady_clock::now();
        require_vk(vkResetCommandBuffer(command_buffer_, 0), "vkResetCommandBuffer");
        require_vk(vkBeginCommandBuffer(command_buffer_, &begin_info), "vkBeginCommandBuffer");
        if (timestamps_usable_) {
            // Reset inside this recording: the queries must be reset before
            // they are written, in submission order (VUID-vkCmdWriteTimestamp-
            // None-00830). vkCmdResetQueryPool is a void-returning command.
            vkCmdResetQueryPool(command_buffer_, query_pool_, 0, kQueryCount);
            vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_,
                                kTimestampQueryStart);
        }
        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_,
                                0, 1, &descriptor_set_, 0, nullptr);
        vkCmdDispatch(command_buffer_, d.groups[0], d.groups[1], d.groups[2]);
        if (timestamps_usable_) {
            // Execution dependency across the dispatch: the end timestamp
            // must observe completed compute work, not the command
            // processor reaching the write (observed on the RTX 5070:
            // timestamp pairs without this barrier return a constant
            // ~0.6 us regardless of the kernel).
            const VkMemoryBarrier done{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                       .pNext = nullptr,
                                       .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                       .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &done, 0, nullptr, 0,
                                 nullptr);
            vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool_,
                                kTimestampQueryEnd);
        }
        require_vk(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");
        VkSubmitInfo submit_info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                 .pNext = nullptr,
                                 .waitSemaphoreCount = 0,
                                 .pWaitSemaphores = nullptr,
                                 .pWaitDstStageMask = nullptr,
                                 .commandBufferCount = 1,
                                 .pCommandBuffers = &command_buffer_,
                                 .signalSemaphoreCount = 0,
                                 .pSignalSemaphores = nullptr};
        require_vk(vkQueueSubmit(queue_, 1, &submit_info, fence_), "vkQueueSubmit");
        require_vk(vkWaitForFences(device_, 1, &fence_, VK_TRUE, kFenceTimeoutNs),
                   "vkWaitForFences");
        require_vk(vkResetFences(device_, 1, &fence_), "vkResetFences");
        // Host sample closed BEFORE the query read, so the GPU result read
        // never inflates the host wall time it is compared against.
        const auto iter_end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> iter = iter_end - iter_start;
        iter_ms.push_back(iter.count());
        if (timestamps_usable_) {
            std::array<uint64_t, kQueryCount> stamps{};
            require_vk(vkGetQueryPoolResults(device_, query_pool_, 0, kQueryCount, sizeof(stamps),
                                             stamps.data(), sizeof(stamps[0]),
                                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                       "vkGetQueryPoolResults");
            const uint64_t ticks =
                (stamps[kTimestampQueryEnd] - stamps[kTimestampQueryStart]) & ts_mask;
            const auto period_ns = static_cast<double>(properties_.limits.timestampPeriod);
            gpu_iter_ms.push_back(static_cast<double>(ticks) * period_ns /
                                  kNanosecondsPerMillisecond);
        }
    }
    std::ranges::sort(iter_ms);
    stats.min_ms = iter_ms.front();
    const size_t p90_index = (iter_ms.size() * 9U + 9U) / 10U;
    stats.p90_ms = iter_ms[p90_index >= iter_ms.size() ? iter_ms.size() - 1 : p90_index];
    // cppcheck-suppress useStlAlgorithm - std::accumulate over a small
    // sampled vector reads no clearer than the loop; kept explicit.
    const double sum = std::accumulate(iter_ms.begin(), iter_ms.end(), 0.0);
    stats.avg_ms = sum / static_cast<double>(iter_ms.size());
    if (!gpu_iter_ms.empty()) {
        const double gpu_sum = std::accumulate(gpu_iter_ms.begin(), gpu_iter_ms.end(), 0.0);
        stats.gpu_ms = gpu_sum / static_cast<double>(gpu_iter_ms.size());
    }
    return stats;
}

double VulkanBackend::dispatch_bench(const Dispatch& d, uint32_t count) {
    return dispatch_bench_stats(d, count).avg_ms;
}

void* VulkanBackend::make_bindings(IBuffer* b0, IBuffer* b1, IBuffer* b2, const void* params,
                                   uint64_t param_bytes) {
    // Each binding set carries its own params slot: the shared uniform
    // buffer (set_params) only ever holds ONE site's params, while a
    // sequence mixes sites with different shapes. Created with the
    // UNIFORM usage - the generic create_buffer() is storage-only and
    // a UNIFORM descriptor would reject it (VUID-00330).
    auto ub = create_buffer_impl(param_bytes, true, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    std::memcpy(ub->mapped(), params, param_bytes);
    auto* vulkan_ub = static_cast<VulkanBuffer*>(ub.get());
    if (!vulkan_ub->host_coherent()) {
        flush_buffer(*vulkan_ub);
    }
    binding_uniforms_.push_back(std::move(ub));
    // The null-slot placeholder is created lazily in bind(); a batched
    // sequence may reach make_bindings without any prior bind().
    if (!dummy_buffer_) {
        dummy_buffer_ = std::unique_ptr<VulkanBuffer>(
            static_cast<VulkanBuffer*>(create_buffer(kDummyBufferBytes, true).release()));
    }
    const std::array<IBuffer*, 3> inputs = {b0, b1, b2};
    const VkBuffer dummy = static_cast<VulkanBuffer*>(dummy_buffer_.get())->handle();
    std::array<VkBuffer, 3> buffers = {dummy, dummy, dummy};
    std::array<VkDeviceSize, 3> ranges = {kDummyBufferBytes, kDummyBufferBytes, kDummyBufferBytes};
    for (uint32_t i = 0; i < 3; ++i) {
        if (inputs[i] == nullptr) {
            continue;
        }
        const auto* buffer = static_cast<const VulkanBuffer*>(inputs[i]);
        buffers[i] = buffer->handle();
        ranges[i] = static_cast<VkDeviceSize>(buffer->size());
    }
    VkDescriptorSetAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                           .pNext = nullptr,
                                           .descriptorPool = descriptor_pool_,
                                           .descriptorSetCount = 1,
                                           .pSetLayouts = &set_layout_};
    VkDescriptorSet set = VK_NULL_HANDLE;
    require_vk(vkAllocateDescriptorSets(device_, &alloc_info, &set), "vkAllocateDescriptorSets");
    update_storage_set(set, buffers, ranges, vulkan_ub->handle());
    binding_sets_.push_back(set);
    return static_cast<void*>(set);
}

void* VulkanBackend::make_image_bindings(const void* weight_bytes, uint64_t weight_size_bytes,
                                         IBuffer* acts, IBuffer* out, const void* params,
                                         uint64_t param_bytes) {
    constexpr VkDeviceSize texel_bytes = 16; // R32G32B32A32_UINT
    const uint32_t width = kWeightImageWidthTexels;
    const auto height = static_cast<uint32_t>((weight_size_bytes + width * texel_bytes - 1) /
                                              (width * texel_bytes));

    VkImageCreateInfo image_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R32G32B32A32_UINT,
        .extent = VkExtent3D{.width = width, .height = height, .depth = 1U},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage image = VK_NULL_HANDLE;
    require_vk(vkCreateImage(device_, &image_info, nullptr, &image), "vkCreateImage");
    VkMemoryRequirements reqs{};
    vkGetImageMemoryRequirements(device_, image, &reqs);
    VkDeviceMemory image_memory = VK_NULL_HANDLE;
    VkMemoryAllocateInfo alloc_info{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                    .pNext = nullptr,
                                    .allocationSize = reqs.size,
                                    .memoryTypeIndex = find_memory_type(
                                        reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    if (alloc_info.memoryTypeIndex == UINT32_MAX) {
        alloc_info.memoryTypeIndex =
            find_memory_type(reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    }
    require_vk(vkAllocateMemory(device_, &alloc_info, nullptr, &image_memory),
               "vkAllocateMemory (weight image)");
    require_vk(vkBindImageMemory(device_, image, image_memory, 0), "vkBindImageMemory");

    VkImageViewCreateInfo view_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R32G32B32A32_UINT,
        .components = VkComponentMapping{.r = VK_COMPONENT_SWIZZLE_IDENTITY,
                                         .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                                         .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                                         .a = VK_COMPONENT_SWIZZLE_IDENTITY},
        .subresourceRange = VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                    .baseMipLevel = 0U,
                                                    .levelCount = 1U,
                                                    .baseArrayLayer = 0U,
                                                    .layerCount = 1U}};
    VkImageView view = VK_NULL_HANDLE;
    require_vk(vkCreateImageView(device_, &view_info, nullptr, &view), "vkCreateImageView");

    // Stage the packed bytes host-side, then copy once into the image.
    // The copy region spans the FULL padded extent (height is rounded
    // up to whole texel rows), so the stage buffer must cover the
    // entire extent — allocating it at weight_size_bytes made
    // vkCmdCopyBufferToImage read up to one row (16 KiB) past the
    // buffer end, a GPU page fault NVIDIA surfaces as a deferred
    // VK_ERROR_DEVICE_LOST on a LATER submit (observed on the RTX 5070).
    // The zero-filled tail reads as weight 0: bit-exactness untouched.
    const VkDeviceSize stage_bytes =
        static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * texel_bytes;
    auto stage = create_buffer_impl(stage_bytes, true, 0);
    std::memset(stage->mapped(), 0, static_cast<size_t>(stage_bytes));
    std::memcpy(stage->mapped(), weight_bytes, weight_size_bytes);
    if (!stage->host_coherent()) {
        flush_buffer(*stage);
    }

    VkCommandBufferBeginInfo begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                        .pNext = nullptr,
                                        .flags = 0,
                                        .pInheritanceInfo = nullptr};
    require_vk(vkResetCommandBuffer(command_buffer_, 0), "vkResetCommandBuffer");
    require_vk(vkBeginCommandBuffer(command_buffer_, &begin_info), "vkBeginCommandBuffer");
    VkImageMemoryBarrier to_dst{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                .pNext = nullptr,
                                .srcAccessMask = 0U,
                                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = image,
                                .subresourceRange =
                                    VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                            .baseMipLevel = 0U,
                                                            .levelCount = 1U,
                                                            .baseArrayLayer = 0U,
                                                            .layerCount = 1U}};
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_dst);
    VkBufferImageCopy region{
        .bufferOffset = 0,
        .bufferRowLength = width, // tightly packed host rows
        .bufferImageHeight = height,
        .imageSubresource = VkImageSubresourceLayers{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                     .mipLevel = 0U,
                                                     .baseArrayLayer = 0U,
                                                     .layerCount = 1U},
        .imageOffset = VkOffset3D{.x = 0, .y = 0, .z = 0},
        .imageExtent = VkExtent3D{.width = width, .height = height, .depth = 1U}};
    vkCmdCopyBufferToImage(command_buffer_, stage->handle(), image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VkImageMemoryBarrier to_read{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                    .baseMipLevel = 0U,
                                                    .levelCount = 1U,
                                                    .baseArrayLayer = 0U,
                                                    .layerCount = 1U}};
    vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &to_read);
    require_vk(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");
    VkSubmitInfo submit_info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                             .pNext = nullptr,
                             .waitSemaphoreCount = 0,
                             .pWaitSemaphores = nullptr,
                             .pWaitDstStageMask = nullptr,
                             .commandBufferCount = 1,
                             .pCommandBuffers = &command_buffer_,
                             .signalSemaphoreCount = 0,
                             .pSignalSemaphores = nullptr};
    require_vk(vkQueueSubmit(queue_, 1, &submit_info, fence_), "vkQueueSubmit");
    require_vk(vkWaitForFences(device_, 1, &fence_, VK_TRUE, kFenceTimeoutNs), "vkWaitForFences");
    require_vk(vkResetFences(device_, 1, &fence_), "vkResetFences");

    // Descriptor set: b0 dummy storage (unused by the kernel), b1 acts,
    // b2 out, b3 params slot, b5 the sampled weight image.
    const std::array<IBuffer*, 3> inputs = {nullptr, acts, out}; // b0: dummy (unused)
    const VkBuffer dummy = static_cast<VulkanBuffer*>(dummy_buffer_.get())->handle();
    std::array<VkBuffer, 3> buffers = {dummy, dummy, dummy};
    std::array<VkDeviceSize, 3> ranges = {kDummyBufferBytes, kDummyBufferBytes, kDummyBufferBytes};
    for (uint32_t i = 0; i < 3; ++i) {
        if (inputs[i] == nullptr) {
            continue;
        }
        const auto* buffer = static_cast<const VulkanBuffer*>(inputs[i]);
        buffers[i] = buffer->handle();
        ranges[i] = static_cast<VkDeviceSize>(buffer->size());
    }
    auto ub = create_buffer_impl(param_bytes, true, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    std::memcpy(ub->mapped(), params, param_bytes);
    auto* vulkan_ub = static_cast<VulkanBuffer*>(ub.get());
    if (!vulkan_ub->host_coherent()) {
        flush_buffer(*vulkan_ub);
    }
    binding_uniforms_.push_back(std::move(ub));
    VkDescriptorSetAllocateInfo alloc_set{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                          .pNext = nullptr,
                                          .descriptorPool = descriptor_pool_,
                                          .descriptorSetCount = 1,
                                          .pSetLayouts = &set_layout_};
    VkDescriptorSet set = VK_NULL_HANDLE;
    require_vk(vkAllocateDescriptorSets(device_, &alloc_set, &set), "vkAllocateDescriptorSets");
    update_storage_set(set, buffers, ranges, vulkan_ub->handle());
    VkDescriptorImageInfo image_info_bind{.sampler = VK_NULL_HANDLE,
                                          .imageView = view,
                                          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet image_write{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                     .pNext = nullptr,
                                     .dstSet = set,
                                     .dstBinding = kBindingWeightImage,
                                     .dstArrayElement = 0,
                                     .descriptorCount = 1,
                                     .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                     .pImageInfo = &image_info_bind,
                                     .pBufferInfo = nullptr,
                                     .pTexelBufferView = nullptr};
    vkUpdateDescriptorSets(device_, 1, &image_write, 0, nullptr);
    weight_images_.push_back(WeightImage{.image = image, .memory = image_memory, .view = view});
    binding_sets_.push_back(set);
    return static_cast<void*>(set);
}

} // namespace bitnet::vulkan
