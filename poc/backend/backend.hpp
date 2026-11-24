// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

#pragma once
// POC compute-backend abstraction: common core + swappable backends.
//
// Dependency rule: this header is part of the backend layer. It must not
// include Vulkan/Metal headers, and poc/common/ must not include it — the
// core talks to backends only through this interface. Vulkan implements it
// now; Metal (metal-cpp) is a drop-in later.
//
// Design notes:
// - Shader sources are Slang. A backend compiles/loads each
//   KernelDesc into its own target (SPIR-V for Vulkan, AIR for Metal).
// - All tensor layouts and packed formats live in poc/common/ so every
//   backend consumes identical bytes and the harness validates each backend
//   against the same golden vectors.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bitnet {

// Vulkan packs versions as (major << 22) | (minor << 12) | patch. Decoding
// here (bit math only) keeps apps free of Vulkan headers.
inline uint32_t version_major(uint32_t v) {
    return v >> 22;
}
inline uint32_t version_minor(uint32_t v) {
    return (v >> 12) & 0x3FF;
}
inline uint32_t version_patch(uint32_t v) {
    return v & 0xFFF;
}

// Capability-ladder floors. kMinApi10: the popcount/
// bitplane baseline path must run on a Vulkan 1.0 device (slow but correct,
// degrade don't skip); the 1.1-only queries the backend needs resolve
// through VK_KHR_get_physical_device_properties2 there. Integer dot
// is NOT an API floor - it is capability-gated per feature (core 1.3
// shaderIntegerDotProduct OR VK_KHR_shader_integer_dot_product), with a
// bit-exact emulated fallback when absent. kMinApi10 is the PHYSICAL-DEVICE
// floor; instance creation currently requests a Vulkan 1.3-capable loader.
constexpr uint32_t kMinApi10 = 1U << 22;

// Capability snapshot from the physical device (written as a Markdown
// file by poc_tests --caps-out).
struct DeviceCaps {
    std::string device_name;
    std::string driver_version;
    uint32_t api_version{};
    uint32_t max_shared_memory_bytes{};
    uint32_t max_workgroup_invocations{};
    uint32_t subgroup_size{};
    bool subgroup_arithmetic{};        // reduce/add/... in shaders
    bool shader_integer_dot_product{}; // core feature (1.3): shaderIntegerDotProduct
    // VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR acceleration
    // bits: the FEATURE can be advertised while the driver EMULATES the op
    // (all three measured mobile GPUs report the op as not accelerated). The
    // packed-SIGNED variant is the one our kernels use.
    bool dot_4x8_packed_signed_accelerated{};
    bool dot_8bit_signed_accelerated{};
    bool cooperative_matrix{}; // VK_KHR_cooperative_matrix
    // Cooperative vectors: NO KHR variant exists in the registry -
    // VK_NV_cooperative_vector is the only registered
    // extension, recorded here as the evidence-backed fallback-compat target
    // of the decode-GEMV rung (ADOPT-PROBE only, no implementation).
    bool cooperative_vector_nv{};
    // Coop-mat MMA lane contract (gemm_coopmat): true when the device
    // ENUMERATES (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) the
    // exact shape the module was compiled for - sint8*sint8 -> sint32,
    // 16x16x32, subgroup scope, non-saturating - AND the SPIR-V's feature
    // dependencies (cooperativeMatrix, vulkanMemoryModel +
    // vulkanMemoryModelDeviceScope, shaderInt8) all query TRUE. Kernel
    // loading gates on this BEFORE any coop-mat dispatch (degrade, don't
    // skip: gemm_plane_tiled serves when false).
    bool coop_mat_shape_s8_s32{};
    uint32_t coop_mat_msize{};            // matched entry MSize
    uint32_t coop_mat_nsize{};            // matched entry NSize
    uint32_t coop_mat_ksize{};            // matched entry KSize
    uint32_t coop_mat_scope{};            // matched entry scope (3 = subgroup)
    uint32_t coop_mat_shape_count{};      // total enumerated shapes (evidence)
    uint32_t coop_mat_supported_stages{}; // cooperativeMatrixSupportedStages bitmask
    bool storage_8bit{};
    bool storage_16bit{};
    bool subgroup_size_control{};
    bool compute_full_subgroups{};
    uint32_t min_subgroup_size{};
    uint32_t max_subgroup_size{};
    uint32_t max_compute_workgroup_subgroups{};
    bool required_subgroup_size_compute{};
    bool arm_shader_core_properties{};
    uint32_t arm_pixel_rate{};
    uint32_t arm_texel_rate{};
    uint32_t arm_fma_rate{};
    bool qcom_image_processing{};
    bool qcom_texture_sample_weighted{};
    bool qcom_texture_box_filter{};
    bool qcom_texture_block_match{};
    uint32_t qcom_max_weight_filter_phases{};
    uint32_t qcom_max_weight_filter_width{};
    uint32_t qcom_max_weight_filter_height{};
    uint32_t qcom_max_block_match_width{};
    uint32_t qcom_max_block_match_height{};
    uint32_t qcom_max_box_filter_width{};
    uint32_t qcom_max_box_filter_height{};
    // Caps-conformance limits (machine-checked knob-vs-limit validation;
    // the 128 MiB storage-buffer limit of Adreno is the motivation for the
    // buffer-limit checks).
    uint32_t max_workgroup_size[3]{};      // maxComputeWorkGroupSize x/y/z
    uint32_t max_workgroup_count[3]{};     // maxComputeWorkGroupCount x/y/z
    uint64_t max_storage_buffer_range{};   // per-buffer SSBO range limit
    uint64_t max_memory_allocation_size{}; // per-allocation (VkMemoryAllocateInfo) limit
};

// Host-visible buffer (staging + device-local handling is a backend detail).
class IBuffer {
  public:
    virtual ~IBuffer() = default;
    virtual void* mapped() = 0;
    virtual uint64_t size() const = 0;
};

// A compiled strategy specialization (Slang entry point + constants).
// The backend loads the SPIR-V from "<spv-dir>/<name>.spv".
struct KernelDesc {
    std::string name; // e.g. "gemv_bitplane" -> gemv_bitplane.spv
    uint32_t numthreads[3]{1, 1, 1};
};

// gemm_plane_microtile_image texture-path geometry (mirrors
// bitnet_kernels.slang's texel mapping and
// the vulkan backend's image width).
constexpr uint32_t kWeightImageWidthTexels = 1024;

struct BenchStats {
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double p90_ms = 0.0;
    // GPU-side mean of the same iterations, from a timestamp pair bracketing
    // each dispatch. 0.0 = the driver reports no usable timestamps, so the
    // host samples above are the whole measurement. The gap between avg_ms
    // and gpu_ms is the submit + fence constant every kernel time carries.
    double gpu_ms = 0.0;
};

class IComputeBackend {
  public:
    virtual ~IComputeBackend() = default;

    virtual DeviceCaps probe_caps() = 0;

    // All device-extension names the driver reports (capability probing for
    // optional kernel generations, e.g. gemv_dot_rowcoop candidates). No Vulkan
    // types cross the seam.
    virtual std::vector<std::string> probe_extensions() = 0;

    virtual std::unique_ptr<IBuffer> create_buffer(uint64_t bytes, bool host_visible) = 0;

    virtual void* load_kernel(const KernelDesc& desc) = 0;

    // Validation-gate introspection: did THIS instance run with
    // the debug layer enabled, and how many validation messages did the
    // PROCESS observe? The count aggregates across instances (the harness
    // creates several: probe, caps, suites), so any non-zero value means
    // some run step violated an API contract. requested vs active lets the
    // harness distinguish "OFF by env" from "OFF because the layer is
    // unavailable on this loader".
    [[nodiscard]] virtual bool validation_requested() const = 0;
    [[nodiscard]] virtual bool validation_active() const = 0;
    [[nodiscard]] virtual uint64_t validation_message_count() const = 0;
    // Sticky bindings for subsequent dispatches. Binding table (set 0):
    //   b0: storage read-only  - packed weight bit-planes (uint words)
    //   b1: storage read-only  - packed activation bit-planes (uint words)
    //   b2: storage read-write - int32 accumulator output
    // plus a uniform block (set 0, binding 3) written by set_params.
    virtual void bind(IBuffer* b0, IBuffer* b1, IBuffer* b2) = 0;
    virtual void set_params(const void* data, uint64_t bytes) = 0;

    // Batched-sequence support: the realistic
    // engine shape - many call sites per token, ONE submit + fence per
    // token instead of one per call. A binding set is an immutable
    // descriptor set with its OWN params slot, so call sites with
    // different shapes/params coexist in one sequence (bind + set_params
    // carry exactly one set and stay untouched for single-dispatch use).
    virtual void* make_bindings(IBuffer* b0, IBuffer* b1, IBuffer* b2, const void* params,
                                uint64_t param_bytes) = 0;

    // gemm_plane_microtile_image path: uploads the packed weight stream into a 2D
    // R32G32B32A32_UINT sampled image (width = kWeightImageWidthTexels
    // texels, tightly packed host rows) and binds it at binding 5
    // (SAMPLED_IMAGE). The image upload (stage + copy + layout
    // transitions) happens inside this call; weights are static per
    // layer, so the cost amortizes across the stream's lifetime.
    virtual void* make_image_bindings(const void* weight_bytes, uint64_t weight_size_bytes,
                                      IBuffer* acts, IBuffer* out, const void* params,
                                      uint64_t param_bytes) = 0;

    struct BatchedDispatch {
        void* kernel = nullptr;  // from load_kernel
        void* binding = nullptr; // from make_bindings
        uint32_t groups[3] = {1, 1, 1};
    };

    // Record the sequence into ONE command buffer (a memory barrier
    // between successive dispatches), one submit + fence per repeat.
    // Returns per-repeat (e.g. per-token) host-side avg/min/p90 - the
    // whole-system number the isolated per-call bench cannot see.
    virtual BenchStats dispatch_batch_stats(const std::vector<BatchedDispatch>& seq,
                                            uint32_t repeat) = 0;

    struct Dispatch {
        void* kernel = nullptr;
        uint32_t groups[3] = {1, 1, 1};
    };
    virtual void dispatch(const Dispatch& d) = 0;

    // Per-iteration timing (profiling contract): `count` timed samples of ONE
    // dispatch, each its own submit + fence - the call shape a real engine
    // pays when it does not batch. Host steady_clock brackets submit..fence
    // (avg/min/p90 over the samples) and, when the driver reports usable
    // timestamps, the SAME iteration is bracketed by a GPU timestamp pair so
    // `gpu_ms` separates kernel time from the submit + fence constant. The
    // `null_probe` timing probe reads exactly that split; use it before
    // believing any sub-millisecond kernel row.
    virtual BenchStats dispatch_bench_stats(const Dispatch& d, uint32_t count) = 0;
    virtual double dispatch_bench(const Dispatch& d, uint32_t count) = 0;

    virtual void wait_idle() = 0;
};

// Backend factories (linked per target). make_metal_backend is the
// prepared-only stub in poc/metal/ (unavailable at runtime, no Metal code).
std::unique_ptr<IComputeBackend> make_vulkan_backend();
std::unique_ptr<IComputeBackend> make_metal_backend();

} // namespace bitnet
