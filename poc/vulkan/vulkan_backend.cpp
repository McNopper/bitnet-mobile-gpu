// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Vulkan compute backend for the BitNet PoC.
//
// Implements bitnet::IComputeBackend with the raw Vulkan C API (no volk, no
// vulkan-hpp) so this single translation unit builds on Windows desktop and
// cross-compiles for Android arm64.
//
// v1 scope (deliberate):
// - one compute queue family (compute-only preferred), one queue
// - host-visible I/O buffers, device-local buffers unmapped
// - one shared descriptor set layout (3 storage + 1 uniform binding, set 0)
// - one SPIR-V compute pipeline per kernel, cached by name
// - timing via dispatch_bench: per-iteration submit + fence, host wall clock
//   around each sample, plus a GPU timestamp pair bracketing the same
//   dispatch (kQueryCount queries) when the driver reports usable
//   timestamps; BenchStats::gpu_ms is 0 when it does not
// - validation layer ON by default (repo gate): only BITNET_VALIDATION=0
//   disables it; messages WARNING and above go to stderr with their id
//
// v2 TODO: device-local staging uploads (TRANSFER_DST is already reserved on
// every buffer), dedicated transfer queue.
//
// Threading: single-threaded by contract (the harness drives the backend).
// Lifetimes: buffers handed out by this backend must be destroyed before the
// backend itself (their destructors reference the VkDevice).

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

#include "vulkan/vulkan_backend.hpp"

#include "backend/backend.hpp"
#include "common/format.hpp"

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

// Throws with a readable message naming the failing call.
[[noreturn]] void throw_vk(VkResult result, const char* what) {
    throw std::runtime_error(std::string("bitnet vulkan: ") + what + " failed (VkResult " +
                             std::to_string(static_cast<int>(result)) + ")");
}

void require_vk(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        throw_vk(result, what);
    }
}

bool env_is(const char* name, const char* value) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strcmp(v, value) == 0;
}

std::vector<VkLayerProperties> enumerate_instance_layers() {
    uint32_t count = 0;
    require_vk(vkEnumerateInstanceLayerProperties(&count, nullptr),
               "vkEnumerateInstanceLayerProperties");
    std::vector<VkLayerProperties> out(count);
    if (count > 0) {
        require_vk(vkEnumerateInstanceLayerProperties(&count, out.data()),
                   "vkEnumerateInstanceLayerProperties");
    }
    return out;
}

std::vector<VkExtensionProperties> enumerate_instance_extensions() {
    uint32_t count = 0;
    require_vk(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr),
               "vkEnumerateInstanceExtensionProperties");
    std::vector<VkExtensionProperties> out(count);
    if (count > 0) {
        require_vk(vkEnumerateInstanceExtensionProperties(nullptr, &count, out.data()),
                   "vkEnumerateInstanceExtensionProperties");
    }
    return out;
}

std::vector<VkExtensionProperties> enumerate_device_extensions(VkPhysicalDevice physical_device) {
    uint32_t count = 0;
    require_vk(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr),
               "vkEnumerateDeviceExtensionProperties");
    std::vector<VkExtensionProperties> out(count);
    if (count > 0) {
        require_vk(
            vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, out.data()),
            "vkEnumerateDeviceExtensionProperties");
    }
    return out;
}

bool has_extension(const std::vector<VkExtensionProperties>& list, const char* name) {
    return std::ranges::any_of(
        list, [name](const auto& entry) { return std::strcmp(entry.extensionName, name) == 0; });
}

bool has_layer(const std::vector<VkLayerProperties>& list, const char* name) {
    return std::ranges::any_of(
        list, [name](const auto& entry) { return std::strcmp(entry.layerName, name) == 0; });
}

// VkPhysicalDeviceProperties::driverVersion is vendor-encoded. NVIDIA packs
// 10.8.8.6 bits; decoding it with the generic 10.10.12 VK_VERSION macros
// prints driver 621.24 as "621.96.0". The other vendors measured here (Arm,
// Qualcomm) use the generic packing. Display formatting only - nothing
// selects kernels or knobs from the vendor id.
constexpr uint32_t kVendorIdNvidia = 0x10DE;
constexpr uint32_t kNvidiaMajorShift = 22;
constexpr uint32_t kNvidiaMinorShift = 14;
constexpr uint32_t kNvidiaSecondaryShift = 6;
constexpr uint32_t kNvidiaMajorMask = 0x3FF;   // 10 bits
constexpr uint32_t kNvidiaMinorMask = 0xFF;    // 8 bits (minor and secondary)
constexpr uint32_t kNvidiaTertiaryMask = 0x3F; // 6 bits

std::string format_driver_version(uint32_t vendor_id, uint32_t version) {
    if (vendor_id == kVendorIdNvidia) {
        return std::to_string((version >> kNvidiaMajorShift) & kNvidiaMajorMask) + "." +
               std::to_string((version >> kNvidiaMinorShift) & kNvidiaMinorMask) + "." +
               std::to_string((version >> kNvidiaSecondaryShift) & kNvidiaMinorMask) + "." +
               std::to_string(version & kNvidiaTertiaryMask);
    }
    return std::to_string(VK_VERSION_MAJOR(version)) + "." +
           std::to_string(VK_VERSION_MINOR(version)) + "." +
           std::to_string(VK_VERSION_PATCH(version));
}

// Validation gate: PROCESS-WIDE count of validation messages observed by
// any backend instance's debug callback during this run. The harness creates
// several instances (probe, caps, suites), so a per-instance counter would
// undercount; every instance reports this total.
std::atomic<uint64_t> g_validation_message_count{0};

// stderr debug messenger (repo gate: validation on by default). Prints every
// WARNING-or-above message with its id, counts it (the harness fails the
// run on any message). Never throws.
VKAPI_ATTR VkBool32 VKAPI_CALL debug_messenger_callback(
    [[maybe_unused]] VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT types,
    const VkDebugUtilsMessengerCallbackDataEXT* data, [[maybe_unused]] void* user_data) {
    const char* id = data->pMessageIdName != nullptr ? data->pMessageIdName : "unknown";
    std::fprintf(stderr, "bitnet vulkan [validation] [%s] %s\n", id, data->pMessage);
    std::fflush(stderr);
    g_validation_message_count.fetch_add(1, std::memory_order_relaxed);
    return VK_FALSE;
}

// "<spv-dir>" = $BITNET_SPV_DIR if set, else "spv" (relative to the CWD).
std::string spv_directory() {
    const char* env = std::getenv("BITNET_SPV_DIR");
    if (env != nullptr && env[0] != '\0') {
        return {env};
    }
    return "spv";
}

// Reads a whole SPIR-V module; throws naming the full attempted path when the
// file is missing, empty, or not word-aligned.
// Returns the LocalSize.x of the first OpExecutionMode LocalSize instruction
// in the module, or 0 when none is present. SPIR-V words: opcode = w & 0xFFFF,
// word count = w >> 16; OpExecutionMode (opcode 16) operands are
// [entry point, execution mode, literals...]; LocalSize is execution mode 17.
// SPIR-V word-stream constants for the LocalSize scan (OpExecutionMode).
constexpr uint32_t kSpirvHeaderWords = 5U; // magic, version, gen, bound, schema
constexpr uint32_t kSpirvOpExecutionMode = 16U;
constexpr uint32_t kSpirvExecModeLocalSize = 17U;
constexpr uint32_t kSpirvLocalSizeXIndex = 3U;  // first size literal after the mode
constexpr uint32_t kSpirvMinExecModeWords = 5U; // opcode, target, mode, x, y

uint32_t local_size_x_of(const std::vector<uint32_t>& code) {
    size_t i = kSpirvHeaderWords; // skip the header (magic, version, generator, bound, schema)
    while (i < code.size()) {
        const uint32_t insn = code[i];
        const uint32_t opcode = insn & 0xFFFFU;
        const size_t count = insn >> 16;
        if (count == 0U || count > code.size() - i) {
            break; // malformed - do not loop forever or read past the module
        }
        if (opcode == kSpirvOpExecutionMode && count >= kSpirvMinExecModeWords &&
            code[i + 2] == kSpirvExecModeLocalSize) {
            return code[i + kSpirvLocalSizeXIndex];
        }
        i += count;
    }
    return 0U;
}
std::vector<uint32_t> read_spirv_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("bitnet vulkan: cannot open SPIR-V kernel file '" + path + "'");
    }
    const std::streamoff byte_count = file.tellg();
    if (byte_count <= 0 || (byte_count % 4) != 0) {
        throw std::runtime_error("bitnet vulkan: SPIR-V kernel file '" + path +
                                 "' is empty or not a whole number of 32-bit words");
    }
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> words(static_cast<size_t>(byte_count) / 4);
    file.read(reinterpret_cast<char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    if (!file) {
        throw std::runtime_error("bitnet vulkan: short read on SPIR-V kernel file '" + path + "'");
    }
    return words;
}

// Preferred family: compute-only; else any compute-capable family.
// Free function (no captures) so select_physical_device stays simple enough
// for the cognitive-complexity gate.
int32_t best_compute_family(VkPhysicalDevice physical_device, uint32_t* timestamp_bits) {
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, nullptr);
    if (family_count == 0) {
        return -1;
    }
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, families.data());
    int32_t compute_only = -1;
    int32_t any_compute = -1;
    for (uint32_t i = 0; i < family_count; ++i) {
        const VkQueueFlags flags = families[i].queueFlags;
        if ((flags & VK_QUEUE_COMPUTE_BIT) == 0) {
            continue;
        }
        if (any_compute < 0) {
            any_compute = static_cast<int32_t>(i);
        }
        if (compute_only < 0 && (flags & VK_QUEUE_GRAPHICS_BIT) == 0) {
            compute_only = static_cast<int32_t>(i);
        }
    }
    if (compute_only >= 0) {
        *timestamp_bits = families[static_cast<size_t>(compute_only)].timestampValidBits;
        return compute_only;
    }
    if (any_compute >= 0) {
        *timestamp_bits = families[static_cast<size_t>(any_compute)].timestampValidBits;
        return any_compute;
    }
    return -1;
}

// --------------------------------------------------------------- interface

VulkanBackend::VulkanBackend() {
    create_instance();
    select_physical_device();
    create_device_objects();
    caps_ = build_caps();
}

VulkanBackend::~VulkanBackend() {
    destroy_all();
}

DeviceCaps VulkanBackend::probe_caps() {
    return caps_;
}

[[nodiscard]] bool VulkanBackend::validation_active() const {
    return validation_active_;
}

[[nodiscard]] bool VulkanBackend::validation_requested() const {
    return validation_requested_;
}

[[nodiscard]] uint64_t VulkanBackend::validation_message_count() const {
    return g_validation_message_count.load(std::memory_order_relaxed);
}

std::vector<std::string> VulkanBackend::probe_extensions() {
    std::vector<std::string> names;
    names.reserve(enumerate_device_extensions(physical_device_).size());
    for (const VkExtensionProperties& e : enumerate_device_extensions(physical_device_)) {
        // cppcheck-suppress useStlAlgorithm - emplace_back of a member
        // copy reads clearest here; std::transform needs a lambda that
        // hides the member access.
        names.emplace_back(e.extensionName);
    }
    std::ranges::sort(names);
    return names;
}

std::unique_ptr<IBuffer> VulkanBackend::create_buffer(uint64_t bytes, bool host_visible) {
    if (bytes == 0) {
        throw std::runtime_error("bitnet vulkan: create_buffer requires bytes > 0");
    }
    return create_buffer_impl(bytes, host_visible, 0);
}

void* VulkanBackend::load_kernel(const KernelDesc& desc) {
    const auto cached = pipelines_.find(desc.name);
    if (cached != pipelines_.end()) {
        return cached->second;
    }

    const std::string path = spv_directory() + "/" + desc.name + ".spv";
    const std::vector<uint32_t> code = read_spirv_file(path);

    // Instrument guard: verify the module's OpExecutionMode LocalSize.x
    // against the caller-expected thread count. A stale or
    // mismatched-profile .spv is REJECTED here instead of silently
    // computing a fraction of the output.
    if (const uint32_t local_x = local_size_x_of(code);
        local_x != 0U && local_x != desc.numthreads[0]) {
        throw std::runtime_error("bitnet vulkan: REJECTED '" + desc.name + "' from '" + path +
                                 "': SPIR-V LocalSize.x = " + std::to_string(local_x) +
                                 ", expected " + std::to_string(desc.numthreads[0]) +
                                 " (stale or wrong-profile kernel)");
    }

    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo module_info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                         .pNext = nullptr,
                                         .flags = 0,
                                         .codeSize = code.size() * sizeof(uint32_t),
                                         .pCode = code.data()};
    require_vk(vkCreateShaderModule(device_, &module_info, nullptr, &shader_module),
               "vkCreateShaderModule");

    // Test-only A/B knob: pin the required subgroup width when the
    // driver advertises subgroupSizeControl for compute. Selection is
    // capability-checked and fails loudly; no vendor/device branch.
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_stage{};
    const char* required_subgroup = std::getenv("BITNET_REQUIRED_SUBGROUP_SIZE");
    if (required_subgroup != nullptr && required_subgroup[0] != '\0') {
        const uint32_t size = static_cast<uint32_t>(std::stoul(required_subgroup));
        if (!caps_.subgroup_size_control || !caps_.required_subgroup_size_compute ||
            size < caps_.min_subgroup_size || size > caps_.max_subgroup_size) {
            throw std::runtime_error(
                "bitnet vulkan: BITNET_REQUIRED_SUBGROUP_SIZE=" + std::to_string(size) +
                " is outside the device's compute subgroup-control contract [" +
                std::to_string(caps_.min_subgroup_size) + ", " +
                std::to_string(caps_.max_subgroup_size) + "]");
        }
        subgroup_stage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
        subgroup_stage.pNext = nullptr;
        subgroup_stage.requiredSubgroupSize = size;
        std::printf("bitnet vulkan: requiring subgroup size %u for '%s'\n", size,
                    desc.name.c_str());
    }

    // v1: no specialization constants, entry point "main".
    VkPipelineShaderStageCreateInfo stage_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = required_subgroup != nullptr && required_subgroup[0] != '\0' ? &subgroup_stage
                                                                              : nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = nullptr};
    VkComputePipelineCreateInfo pipeline_info{.sType =
                                                  VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                              .pNext = nullptr,
                                              .flags = 0,
                                              .stage = stage_info,
                                              .layout = pipeline_layout_,
                                              .basePipelineHandle = VK_NULL_HANDLE,
                                              .basePipelineIndex = -1};
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result =
        vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
    // The pipeline keeps its own copy of the module; free ours either way.
    vkDestroyShaderModule(device_, shader_module, nullptr);
    require_vk(result, "vkCreateComputePipelines");

    pipelines_.emplace(desc.name, pipeline);
    return pipeline;
}

void VulkanBackend::update_storage_set(VkDescriptorSet set, const std::array<VkBuffer, 3>& buffers,
                                       const std::array<VkDeviceSize, 3>& ranges,
                                       VkBuffer params_buf) {
    std::array<VkDescriptorBufferInfo, 3> storage_infos = {{
        {.buffer = buffers[0], .offset = 0, .range = ranges[0]},
        {.buffer = buffers[1], .offset = 0, .range = ranges[1]},
        {.buffer = buffers[2], .offset = 0, .range = ranges[2]},
    }};
    VkDescriptorBufferInfo params_info{.buffer = params_buf,
                                       .offset = 0,
                                       .range = static_cast<VkDeviceSize>(sizeof(KernelParams))};

    std::array<VkWriteDescriptorSet, 4> writes{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                     .pNext = nullptr,
                     .dstSet = set,
                     .dstBinding = i,
                     .dstArrayElement = 0,
                     .descriptorCount = 1,
                     .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                     .pImageInfo = nullptr,
                     .pBufferInfo = &storage_infos[i],
                     .pTexelBufferView = nullptr};
    }
    writes[3] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 .pNext = nullptr,
                 .dstSet = set,
                 .dstBinding = 3,
                 .dstArrayElement = 0,
                 .descriptorCount = 1,
                 .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                 .pImageInfo = nullptr,
                 .pBufferInfo = &params_info,
                 .pTexelBufferView = nullptr};
    vkUpdateDescriptorSets(device_, 4, writes.data(), 0, nullptr);
}

void VulkanBackend::bind(IBuffer* b0, IBuffer* b1, IBuffer* b2) {
    ensure_uniform_buffer();
    if (!dummy_buffer_) {
        dummy_buffer_ = std::unique_ptr<VulkanBuffer>(
            static_cast<VulkanBuffer*>(create_buffer(kDummyBufferBytes, true).release()));
    }

    // Any of the three may be nullptr (e.g. poc_bench binds (src, sink,
    // nullptr) for bw_probe, which statically uses only bindings 0/1).
    // Writing a null descriptor requires the nullDescriptor feature
    // (VUID-VkDescriptorBufferInfo-buffer-02998), so null slots are
    // bound to a small dummy buffer instead — spec-clean on every
    // driver, and never dereferenced by kernels that statically ignore
    // the slot (documented exception: null slots are never dereferenced).
    const VkBuffer dummy = static_cast<VulkanBuffer*>(dummy_buffer_.get())->handle();
    const std::array<IBuffer*, 3> inputs = {b0, b1, b2};
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
    bound_[0] = buffers[0];
    bound_[1] = buffers[1];
    bound_[2] = buffers[2];

    if (descriptor_set_ == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo alloc_info{.sType =
                                                   VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                               .pNext = nullptr,
                                               .descriptorPool = descriptor_pool_,
                                               .descriptorSetCount = 1,
                                               .pSetLayouts = &set_layout_};
        require_vk(vkAllocateDescriptorSets(device_, &alloc_info, &descriptor_set_),
                   "vkAllocateDescriptorSets");
    }

    // Rewritten on every bind: safe because every dispatch fences before
    // the next bind/dispatch pair runs (no in-flight consumers).
    update_storage_set(descriptor_set_, bound_, ranges, uniform_->handle());
    descriptors_written_ = true;
}

void VulkanBackend::set_params(const void* data, uint64_t bytes) {
    // Host passes exactly sizeof(KernelParams) == 16 bytes.
    assert(bytes <= kMaxUniformBytes);
    ensure_uniform_buffer();
    void* dst = uniform_->mapped();
    assert(dst != nullptr);
    std::memcpy(dst, data, static_cast<size_t>(bytes));
    if (!uniform_->host_coherent()) {
        flush_buffer(*uniform_);
    }
}

BenchStats VulkanBackend::dispatch_batch_stats(const std::vector<BatchedDispatch>& seq,
                                               uint32_t repeat) {
    BenchStats stats{};
    if (seq.empty() || repeat == 0U) {
        return stats;
    }
    for (const auto& d : seq) {
        if (static_cast<VkPipeline>(d.kernel) == VK_NULL_HANDLE) {
            throw std::runtime_error(
                "bitnet vulkan: dispatch_batch() without a kernel (load_kernel first)");
        }
        if (static_cast<VkDescriptorSet>(d.binding) == VK_NULL_HANDLE) {
            throw std::runtime_error(
                "bitnet vulkan: dispatch_batch() without bindings (make_bindings first)");
        }
    }
    flush_live_host_buffers();

    VkCommandBufferBeginInfo begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                        .pNext = nullptr,
                                        .flags = 0,
                                        .pInheritanceInfo = nullptr};
    // Shader-write -> shader-read/write between successive dispatches:
    // the sequence is a chain (later kernels may consume earlier
    // outputs; within the timing bench they overwrite, and the barrier
    // keeps both cases correct).
    VkMemoryBarrier barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                            .pNext = nullptr,
                            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                            .dstAccessMask =
                                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    std::vector<double> iter_ms;
    iter_ms.reserve(repeat);
    for (uint32_t r = 0; r < repeat; ++r) {
        const auto iter_start = std::chrono::steady_clock::now();
        require_vk(vkResetCommandBuffer(command_buffer_, 0), "vkResetCommandBuffer");
        require_vk(vkBeginCommandBuffer(command_buffer_, &begin_info), "vkBeginCommandBuffer");
        const void* bound_pipeline = nullptr;
        for (const auto& d : seq) {
            if (d.kernel != bound_pipeline) {
                vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  static_cast<VkPipeline>(d.kernel));
                bound_pipeline = d.kernel;
            }
            // std::array (not C-array): modernize-avoid-c-arrays; the
            // data() pointer matches the bind API's const set pointer.
            const std::array<VkDescriptorSet, 1> sets = {static_cast<VkDescriptorSet>(d.binding)};
            vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_layout_, 0, 1, sets.data(), 0, nullptr);
            vkCmdDispatch(command_buffer_, d.groups[0], d.groups[1], d.groups[2]);
            vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr,
                                 0, nullptr);
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
        // Scaled timeout: one batch = a whole token of GPU work; a
        // flat 5 s budget times out on prefill sequences (a 64-token
        // block takes 32 s on Mali-G57).
        const uint64_t batch_timeout_ns =
            kFenceTimeoutNs * (static_cast<uint64_t>(seq.size()) + 1U);
        require_vk(vkWaitForFences(device_, 1, &fence_, VK_TRUE, batch_timeout_ns),
                   "vkWaitForFences");
        require_vk(vkResetFences(device_, 1, &fence_), "vkResetFences");
        const auto iter_end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> iter = iter_end - iter_start;
        iter_ms.push_back(iter.count());
    }
    std::ranges::sort(iter_ms);
    stats.min_ms = iter_ms.front();
    const size_t p90_index = (iter_ms.size() * 9U + 9U) / 10U;
    stats.p90_ms = iter_ms[p90_index >= iter_ms.size() ? iter_ms.size() - 1 : p90_index];
    // cppcheck-suppress useStlAlgorithm - std::accumulate over a small
    // sampled vector reads no clearer than the loop; kept explicit.
    const double sum = std::accumulate(iter_ms.begin(), iter_ms.end(), 0.0);
    stats.avg_ms = sum / static_cast<double>(iter_ms.size());
    return stats;
}

void VulkanBackend::wait_idle() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }
}

void VulkanBackend::submit_and_fence() {
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
}

void VulkanBackend::create_instance() {
    // Gate: the Vulkan validation layer is ON by default
    // and every harness run must be validation-clean; only
    // BITNET_VALIDATION=0 turns it off. Both states are exposed through
    // the seam (validation_active) so the harness can print the
    // per-run state line and fail on any message.
    validation_requested_ = !env_is("BITNET_VALIDATION", "0");
    const bool validation_enabled = validation_requested_;

    VkApplicationInfo app{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                          .pNext = nullptr,
                          .pApplicationName = "bitnet-poc",
                          .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
                          .pEngineName = "bitnet",
                          .engineVersion = VK_MAKE_VERSION(0, 1, 0),
                          .apiVersion = VK_API_VERSION_1_3};

    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    bool debug_utils_enabled = false;

    // Capability-ladder floor (kMinApi10): on a 1.0 instance the
    // Properties2/Features2 queries exist only through this extension
    // (core 1.1 otherwise). Enable when present - harmless above 1.0.
    // (The extension-name constant predates vulkan_core.h inclusion of
    // all promoted KHR macros in some SDK layouts - literal string.)
    if (has_extension(enumerate_instance_extensions(), "VK_KHR_get_physical_device_properties2")) {
        extensions.push_back("VK_KHR_get_physical_device_properties2");
    }

    if (validation_enabled) {
        if (has_layer(enumerate_instance_layers(), kValidationLayerName)) {
            layers.push_back(kValidationLayerName);
            // The messenger exists to count the LAYER's messages. Without
            // the layer the extension alone (some Android drivers expose
            // it) would only forward driver chatter while nothing checks
            // the API usage - the run must then not report validation as
            // active, so the extension is enabled only with the layer.
            if (has_extension(enumerate_instance_extensions(), VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
                extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                debug_utils_enabled = true;
            }
        } else {
            std::fprintf(stderr,
                         "bitnet vulkan: %s is unavailable; continuing without "
                         "validation (set BITNET_VALIDATION=0 to silence this)\n",
                         kValidationLayerName);
        }
    }

    VkInstanceCreateInfo info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                              .pNext = nullptr,
                              .flags = 0,
                              .pApplicationInfo = &app,
                              .enabledLayerCount = static_cast<uint32_t>(layers.size()),
                              .ppEnabledLayerNames = layers.empty() ? nullptr : layers.data(),
                              .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
                              .ppEnabledExtensionNames =
                                  extensions.empty() ? nullptr : extensions.data()};
    const VkResult result = vkCreateInstance(&info, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("bitnet vulkan: vkCreateInstance failed (VkResult " +
                                 std::to_string(static_cast<int>(result)) +
                                 "); is a Vulkan loader/runtime installed?");
    }

    if (debug_utils_enabled) {
        attach_debug_messenger();
    }
}

void VulkanBackend::attach_debug_messenger() {
    const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    destroy_messenger_ = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
    if (create == nullptr || destroy_messenger_ == nullptr) {
        std::fprintf(stderr, "bitnet vulkan: debug-utils functions not exported; validation "
                             "output disabled\n");
        return;
    }
    VkDebugUtilsMessengerCreateInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debug_messenger_callback,
        .pUserData = nullptr};
    if (create(instance_, &info, nullptr, &messenger_) == VK_SUCCESS) {
        // Counting state is authoritative only once the messenger exists:
        // layer + debug-utils + exported functions all had to succeed.
        validation_active_ = true;
    } else {
        messenger_ = VK_NULL_HANDLE;
        std::fprintf(stderr, "bitnet vulkan: failed to create debug messenger\n");
    }
}

void VulkanBackend::select_physical_device() {
    uint32_t count = 0;
    require_vk(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
               "vkEnumeratePhysicalDevices");
    if (count == 0) {
        throw std::runtime_error("bitnet vulkan: no Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> devices(count);
    require_vk(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
               "vkEnumeratePhysicalDevices");

    // Preferred family: computed by the free function above. Extracted
    // from the query chain below to keep cognitive complexity in budget.
    const auto try_select = [&](bool require_discrete) {
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(candidate, &props);
            if (require_discrete && props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                continue;
            }
            uint32_t timestamp_bits = 0;
            const int32_t family = best_compute_family(candidate, &timestamp_bits);
            if (family < 0) {
                continue;
            }
            physical_device_ = candidate;
            queue_family_ = static_cast<uint32_t>(family);
            queue_timestamp_bits_ = timestamp_bits;
            return true;
        }
        return false;
    };
    // First a discrete GPU with a compute queue, else any compute device.
    if (!try_select(true) && !try_select(false)) {
        throw std::runtime_error("bitnet vulkan: no physical device with a compute queue found");
    }
    query_device_capabilities();
}

[[nodiscard]] uint32_t VulkanBackend::find_memory_type(uint32_t type_bits,
                                                       VkMemoryPropertyFlags wanted) const {
    for (uint32_t i = 0; i < memory_.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags props = memory_.memoryTypes[i].propertyFlags;
        if ((type_bits & (1U << i)) != 0 && (props & wanted) == wanted) {
            return i;
        }
    }
    return UINT32_MAX;
}

} // namespace bitnet::vulkan

namespace bitnet {

std::unique_ptr<IComputeBackend> make_vulkan_backend() {
    return std::make_unique<vulkan::VulkanBackend>();
}

} // namespace bitnet
