// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Metal backend STUB: Metal is PREPARED
// ONLY - a placeholder at the IComputeBackend seam for later usage. It
// compiles cleanly on every platform, touches no Metal API, and reports
// Metal as UNAVAILABLE at runtime (clear error / capability flags - never
// a silent fallback). No Metal GPU execution exists in this project; the
// Slang sources are compile-verified to MSL via slangc -target metal.

#include "backend/backend.hpp"

#include <stdexcept>
#include <utility>

namespace bitnet {

namespace {

[[noreturn]] void metal_stub_unavailable(const char* what) {
    throw std::runtime_error(std::string("bitnet metal: ") + what +
                             " - Metal backend is a PREPARED-ONLY stub; no Metal GPU "
                             "execution exists in this build. Runtime baseline is Vulkan "
                             "(capability-driven selection); the Slang kernels are "
                             "compile-verified to MSL (slangc -target metal) but Metal is "
                             "untested on Apple hardware (none available).");
}

// The stub's capability snapshot: reports Metal as unavailable
// (device name carries the stub marker, every capability flag false) so
// capability-driven selection can never pick it as a Vulkan alternative.
class MetalBackendStub final : public IComputeBackend {
  public:
    MetalBackendStub() = default;
    ~MetalBackendStub() override = default;
    MetalBackendStub(const MetalBackendStub&) = delete;
    MetalBackendStub& operator=(const MetalBackendStub&) = delete;

    [[nodiscard]] DeviceCaps probe_caps() override {
        DeviceCaps caps;
        caps.device_name = "Metal (prepared-only stub - unavailable)";
        caps.driver_version = "none";
        caps.api_version = 0U;
        return caps; // every limit/capability stays 0/false: unavailable
    }

    std::vector<std::string> probe_extensions() override {
        return {}; // no Metal runtime - no extensions to report
    }

    [[nodiscard]] bool validation_requested() const override { return false; }
    [[nodiscard]] bool validation_active() const override { return false; }
    [[nodiscard]] uint64_t validation_message_count() const override { return 0U; }

    std::unique_ptr<IBuffer> create_buffer(uint64_t /*bytes*/, bool /*host_visible*/) override {
        metal_stub_unavailable("create_buffer");
    }

    void* load_kernel(const KernelDesc& /*desc*/) override {
        metal_stub_unavailable("load_kernel");
    }

    void bind(IBuffer* /*b0*/, IBuffer* /*b1*/, IBuffer* /*b2*/) override {
        metal_stub_unavailable("bind");
    }

    void set_params(const void* /*data*/, uint64_t /*bytes*/) override {
        metal_stub_unavailable("set_params");
    }

    void* make_bindings(IBuffer* /*b0*/, IBuffer* /*b1*/, IBuffer* /*b2*/, const void* /*params*/,
                        uint64_t /*param_bytes*/) override {
        metal_stub_unavailable("make_bindings");
    }

    void* make_image_bindings(const void* /*weight_bytes*/, uint64_t /*weight_size_bytes*/,
                              IBuffer* /*acts*/, IBuffer* /*out*/, const void* /*params*/,
                              uint64_t /*param_bytes*/) override {
        metal_stub_unavailable("make_image_bindings");
    }

    BenchStats dispatch_batch_stats(const std::vector<BatchedDispatch>& /*seq*/,
                                    uint32_t /*repeat*/) override {
        metal_stub_unavailable("dispatch_batch_stats");
    }

    void dispatch(const Dispatch& /*d*/) override { metal_stub_unavailable("dispatch"); }

    BenchStats dispatch_bench_stats(const Dispatch& /*d*/, uint32_t /*count*/) override {
        metal_stub_unavailable("dispatch_bench_stats");
    }

    double dispatch_bench(const Dispatch& /*d*/, uint32_t /*count*/) override {
        metal_stub_unavailable("dispatch_bench");
    }

    void wait_idle() override { metal_stub_unavailable("wait_idle"); }
};

} // namespace

// Factory (Metal lives in its own folder, never touches common or
// Vulkan code). Registered at the seam next to make_vulkan_backend; the
// harnesses keep the Vulkan lane (capability-driven selection) - this
// factory exists so a real Metal backend can drop in without touching
// common code or the callers.
std::unique_ptr<IComputeBackend> make_metal_backend() {
    return std::make_unique<MetalBackendStub>();
}

} // namespace bitnet
