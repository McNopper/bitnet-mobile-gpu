// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper
#pragma once

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

// Module constants shared by the translation units:
constexpr uint32_t kQueryCount = 2;                       // dispatch top / bottom timestamps
constexpr uint32_t kTimestampQueryStart = 0;              // index of the top timestamp
constexpr uint32_t kTimestampQueryEnd = kQueryCount - 1U; // index of the bottom timestamp
constexpr uint32_t kMaxUniformBytes = 64;                 // set_params contract
constexpr uint64_t kFenceTimeoutNs = 5ULL * 1000ULL * 1000ULL * 1000ULL; // 5 s
constexpr VkDeviceSize kDummyBufferBytes = 256; // null-slot placeholder (see bind)
// Binding table (set 0): 0/1/2 storage, 3 uniform params, 4 texel weight
// stream (no shipped kernel declares it; the binding stays in the shared
// set layout and pool), 5 sampled weight image
// (gemm_plane_microtile_image) - must match the layout created in init
// and the [vk::binding] annotations in the kernels.
constexpr uint32_t kBindingWeightImage = 5;
constexpr uint32_t kBindingCount = 6;

// Not exported by vulkan_core.h; kept here so the literal lives in one place.
constexpr const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";

// Helpers (definitions in vulkan_backend.cpp):
// Throws with a readable message naming the failing call.
[[noreturn]] void throw_vk(VkResult result, const char* what);

void require_vk(VkResult result, const char* what);

bool env_is(const char* name, const char* value);

std::vector<VkLayerProperties> enumerate_instance_layers();

std::vector<VkExtensionProperties> enumerate_instance_extensions();

std::vector<VkExtensionProperties> enumerate_device_extensions(VkPhysicalDevice physical_device);

bool has_extension(const std::vector<VkExtensionProperties>& list, const char* name);

bool has_layer(const std::vector<VkLayerProperties>& list, const char* name);

std::string format_driver_version(uint32_t vendor_id, uint32_t version);

// "<spv-dir>" = $BITNET_SPV_DIR if set, else "spv" (relative to the CWD).
std::string spv_directory();

// Returns the LocalSize.x of the first OpExecutionMode LocalSize instruction
// in the module, or 0 when none is present. SPIR-V words: opcode = w & 0xFFFF,
// word count = w >> 16; OpExecutionMode (opcode 16) operands are
// [entry point, execution mode, literals...]; LocalSize is execution mode 6.
uint32_t local_size_x_of(const std::vector<uint32_t>& code);

// Reads a whole SPIR-V module; throws naming the full attempted path when the
// file is missing, empty, or not word-aligned.
std::vector<uint32_t> read_spirv_file(const std::string& path);

// Preferred family: compute-only; else any compute-capable family.
// Free function (no captures) so select_physical_device stays simple enough
// for the cognitive-complexity gate.
int32_t best_compute_family(VkPhysicalDevice physical_device, uint32_t* timestamp_bits);

// Buffer handle wrapper (IBuffer over one VkBuffer + device memory):
// IBuffer over one VkBuffer + one VkDeviceMemory. Host-visible buffers are
// persistently mapped for their whole lifetime.
class VulkanBuffer final : public IBuffer {
  public:
    VulkanBuffer(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, uint64_t bytes,
                 bool host_visible, bool host_coherent, std::vector<VulkanBuffer*>* live)
        : device_(device), buffer_(buffer), memory_(memory), bytes_(bytes),
          host_coherent_(host_coherent), live_(live) {
        if (host_visible) {
            require_vk(
                vkMapMemory(device_, memory_, 0, static_cast<VkDeviceSize>(bytes_), 0, &mapped_),
                "vkMapMemory");
        }
        // Non-coherent host memory needs an explicit flush before device use;
        // dispatch() flushes everything registered here before every submit.
        if (mapped_ != nullptr && !host_coherent_ && live_ != nullptr) {
            live_->push_back(this);
        }
    }

    ~VulkanBuffer() override {
        if (mapped_ != nullptr) {
            vkUnmapMemory(device_, memory_);
        }
        if (live_ != nullptr) {
            // std::ranges::find over iterator+sentinel with a const pointer
            // needle does not resolve in the MSVC 14.51 STL under clang-tidy's
            // parse; std::find is the portable form here.
            // NOLINTNEXTLINE(modernize-use-ranges)
            const auto it = std::find(live_->begin(), live_->end(), this);
            if (it != live_->end()) {
                live_->erase(it);
            }
        }
        if (buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buffer_, nullptr);
        }
        if (memory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, memory_, nullptr);
        }
    }

    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    void* mapped() override { return mapped_; }
    [[nodiscard]] uint64_t size() const override { return bytes_; }

    [[nodiscard]] VkBuffer handle() const { return buffer_; }
    [[nodiscard]] VkDeviceMemory memory() const { return memory_; }
    [[nodiscard]] bool host_coherent() const { return host_coherent_; }

  private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    uint64_t bytes_ = 0;
    void* mapped_ = nullptr;
    bool host_coherent_ = false;
    std::vector<VulkanBuffer*>* live_ = nullptr;
};

// The backend implementation. The method bodies live in the translation
// units named above; this header is the only type every TU needs.
class VulkanBackend final : public IComputeBackend {
  public:
    VulkanBackend();

    ~VulkanBackend() override;

    VulkanBackend(const VulkanBackend&) = delete;
    VulkanBackend& operator=(const VulkanBackend&) = delete;

    DeviceCaps probe_caps() override;

    [[nodiscard]] bool validation_active() const override;

    [[nodiscard]] bool validation_requested() const override;

    [[nodiscard]] uint64_t validation_message_count() const override;

    std::vector<std::string> probe_extensions() override;

    std::unique_ptr<IBuffer> create_buffer(uint64_t bytes, bool host_visible) override;

    void* load_kernel(const KernelDesc& desc) override;

    // The 4-descriptor write shared by the single-dispatch path (bind) and
    // the batched-sequence path (make_bindings): 3 storage bindings + the
    // params uniform slot.
    void update_storage_set(VkDescriptorSet set, const std::array<VkBuffer, 3>& buffers,
                            const std::array<VkDeviceSize, 3>& ranges, VkBuffer params_buf);

    void bind(IBuffer* b0, IBuffer* b1, IBuffer* b2) override;

    void set_params(const void* data, uint64_t bytes) override;

    void dispatch(const Dispatch& d) override;

    // count separate submissions on the same queue. Same-queue submissions
    // execute in submission order (implicit ordering across submits), so no
    // intra-batch barriers are needed. One batched submit with W->W-hazard
    // barriers was tried and WEDGES this Mali BSP driver: post-barrier
    // dispatches slow ~28x, then the queue stops completing (VK_TIMEOUT).
    BenchStats dispatch_bench_stats(const Dispatch& d, uint32_t count) override;

    double dispatch_bench(const Dispatch& d, uint32_t count) override;

    void* make_bindings(IBuffer* b0, IBuffer* b1, IBuffer* b2, const void* params,
                        uint64_t param_bytes) override;

    // Texture path of gemm_plane_microtile_image: uploads the weight
    // stream into a 2D R32G32B32A32_UINT sampled image (kWeightImageWidth
    // texels wide, height sized to fit; tightly packed host rows) and
    // returns a descriptor set binding it at binding 5 (SAMPLED_IMAGE)
    // alongside the usual acts/out/params slots. OpImageFetch then runs
    // on the texture pipeline — the path the static probe showed at 1
    // texture cycle vs 5 load_store cycles.
    void* make_image_bindings(const void* weight_bytes, uint64_t weight_size_bytes, IBuffer* acts,
                              IBuffer* out, const void* params, uint64_t param_bytes) override;

    BenchStats dispatch_batch_stats(const std::vector<BatchedDispatch>& seq,
                                    uint32_t repeat) override;

    void wait_idle() override;

  private:
    void submit_and_fence();

    void create_instance();

    void attach_debug_messenger();

    void select_physical_device();

    // Properties2 chain + the 1.0 fallback (extracted from
    // query_device_capabilities to keep its cognitive complexity in
    // budget).
    void query_device_properties();

    // Features2 chain (core 1.3 dot feature + acceleration properties;
    // extracted for the same complexity budget).
    void query_device_features();

    // Capability snapshot (Properties2 + Features2 chains; see the
    // capability-ladder note inside).
    void query_device_capabilities();

    // Enumerates VkCooperativeMatrixPropertiesKHR via the instance-level
    // function pointer (the extension may be absent; the pointer is null
    // then). Records the total shape count (evidence) and the entry matching
    // the gemm_coopmat module contract: AType=BType=sint8, CType=ResultType=
    // sint32, non-saturating accumulation, SUBGROUP scope (NVIDIA's KHR
    // enumeration carries only scope-3 shapes; a hypothetical
    // workgroup-only device is recorded but fails the lane contract). The
    // KHR struct carries no row/col-major fields (NV-only) - operand
    // layouts are the kernel's free choice on the KHR path.
    void query_cooperative_matrix_shapes();

    void create_device_objects();

    [[nodiscard]] DeviceCaps build_caps() const;

    [[nodiscard]] uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags wanted) const;

    std::unique_ptr<VulkanBuffer> create_buffer_impl(uint64_t bytes, bool host_visible,
                                                     VkBufferUsageFlags extra_usage);

    void ensure_uniform_buffer();

    void flush_buffer(VulkanBuffer& buffer);

    void flush_live_host_buffers();

    void destroy_all();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger_ = nullptr;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties_{};
    VkPhysicalDeviceMaintenance3Properties maint3_{};
    VkPhysicalDeviceSubgroupProperties subgroup_{};
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup_size_properties_{};
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_features_{};
    VkPhysicalDevice8BitStorageFeatures storage_8bit_features_{};
    VkPhysicalDevice16BitStorageFeatures storage_16bit_features_{};
    VkPhysicalDeviceShaderCorePropertiesARM arm_core_properties_{};
    VkPhysicalDeviceImageProcessingPropertiesQCOM qcom_image_properties_{};
    VkPhysicalDeviceImageProcessingFeaturesQCOM qcom_image_features_{};
    // VK_KHR_cooperative_matrix: features + the core-1.2 features the
    // coop-mat SPIR-V needs (Int8 capability -> shaderInt8; VulkanMemoryModel
    // + DeviceScope capabilities -> vulkanMemoryModel[DeviceScope]). Both
    // queried AND enabled only when the extension and the module's shape are
    // enumerated (least privilege - never on the Mali 1.1 lane).
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_matrix_features_{};
    VkPhysicalDeviceVulkan12Features coop_vulkan12_features_{};
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR coop_matrix_properties_{};
    // scope/saturatingAccumulation get explicit valid values (the enums have
    // no zero enumerator - bugprone-invalid-enum-default-initialization).
    VkCooperativeMatrixPropertiesKHR coop_matrix_shape_{
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
        .scope = VK_SCOPE_SUBGROUP_KHR};
    uint32_t coop_matrix_shape_count_ = 0;
    bool has_coop_matrix_shape_ = false;
    VkPhysicalDeviceMemoryProperties memory_{};
    uint32_t queue_family_ = 0;
    uint32_t queue_timestamp_bits_ = 0;
    bool has_dot_product_ext_ = false;
    bool has_cooperative_matrix_ext_ = false;
    bool has_cooperative_vector_nv_ext_ = false;
    bool has_subgroup_size_control_ = false;
    bool has_8bit_storage_ = false;
    bool has_16bit_storage_ = false;
    bool has_arm_shader_core_properties_ = false;
    bool has_qcom_image_processing_ = false;
    // Core feature bit (VkPhysicalDeviceVulkan13Features.shaderIntegerDotProduct).
    // Authoritative since Vulkan 1.3; the extension above stays enabled only
    // for pre-1.3 drivers.
    bool dot_product_core_ = false;
    bool dot_packed_signed_accelerated_ = false;
    bool dot_8bit_signed_accelerated_ = false;
    bool timestamps_usable_ = false;
    // Validation gate: requested = BITNET_VALIDATION != 0; active =
    // requested AND layer present AND debug messenger created (only then can
    // messages be observed and counted).
    bool validation_requested_ = false;
    bool validation_active_ = false;
    DeviceCaps caps_{};

    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    bool descriptors_written_ = false;
    // Batched-sequence call sites (make_bindings): one descriptor set +
    // one params uniform slot per site.
    std::vector<VkDescriptorSet> binding_sets_;
    std::vector<std::unique_ptr<IBuffer>> binding_uniforms_;
    // Texel views for the v7 texture-path sets (make_texel_bindings).
    // v7 sampled weight images (make_image_bindings): image + memory + view
    // per uploaded weight stream.
    struct WeightImage {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };
    std::vector<WeightImage> weight_images_;
    std::array<VkBuffer, 3> bound_ = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::unique_ptr<VulkanBuffer> dummy_buffer_; // null-slot placeholder (bind)
    std::map<std::string, VkPipeline> pipelines_;
    std::unique_ptr<VulkanBuffer> uniform_;
    std::vector<VulkanBuffer*> live_host_buffers_;
};

} // namespace bitnet::vulkan
