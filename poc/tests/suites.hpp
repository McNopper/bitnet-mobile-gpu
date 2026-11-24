// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

#pragma once
// Shared types of the bit-exact validation harness and the entry points of
// its suites. The harness core (dispatch, compare, shape suite, CLI) stays in
// test_main.cpp; each suite lives in its own translation unit so that every
// file of poc/tests stays below the 1000-line header-size limit.
//
// Own TU per suite: coopmat_lane.cpp, golden_test.cpp, caps_report.cpp,
// real_model_test.cpp.

#include "backend/backend.hpp"
#include "backend/gpu_profiles.h"
#include "common/bitnet1.hpp"
#include "common/env.hpp"
#include "common/format.hpp"
#include "common/tensor_gen.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#ifndef BITNET_SPV_DIR
#define BITNET_SPV_DIR "spv"
#endif

#ifndef BITNET_GOLDEN_PATH
#define BITNET_GOLDEN_PATH "data/golden_vector.txt"
#endif

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

struct Shape {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    bool binary = false;
    // Data pattern: Random = seeded mt19937 tensor; the others
    // are deterministic vectors that random data cannot reach:
    // - Extreme: activations alternate +127/-128, weights alternate +1/-1
    //   (accumulator extremes; both weight planes fully populated).
    // - Zeros: weights AND activations all zero (kernels must still write
    //   every output element).
    // - SignedBinary: the 1-bit {-1,+1} alphabet (BitNet/XNOR-Net encoding),
    //   no zeros - runs in plain ternary mode (both planes).
    enum class Data : std::uint8_t { Random = 0, Extreme, Zeros, SignedBinary };
    Data data = Data::Random;
};

// Experimental-kernel opt-in: the microtile family
// runs only when BITNET_EXPERIMENTAL is set; the default matrix is the
// core set that must exit 0 on every platform.
inline bool experiments_enabled() {
    return bitnet::env_nonempty("BITNET_EXPERIMENTAL");
}

struct Args {
    std::string spv_dir = BITNET_SPV_DIR;
    bool cpu_only = false;
    uint32_t seed = 1234;
    std::string caps_out;
    std::string tensors;                     // BITNET1 container: replaces the random first shape
    std::string golden = BITNET_GOLDEN_PATH; // known-answer vector file
    std::string variant; // forced capability variant (testing); empty = caps-driven
};

struct Variant {
    std::string name;
    uint32_t mode = 0;
    uint32_t kwords = 0; // 0 = plane format (kwords_for(k)); dot family: dots words
    bool needs_zero = false;
    std::array<uint32_t, 3> groups = {0, 1, 1};
    // Non-null replaces the default PASS verdict text (emulated-dot slow
    // path note); FAIL always prints plain.
    const char* pass_note = nullptr;
};

// ---------------------------------------------------------------------------
// Harness core (defined in test_main.cpp), used by the suites
// ---------------------------------------------------------------------------

// Run one shape end-to-end; returns number of mismatching elements (0 = pass).
int64_t compare_tensor(const Shape& s, const bitnet::TernaryTensor& t,
                       std::vector<int32_t>& gpu_out);

int64_t run_and_compare_variant(const Variant& variant, const bitnet::GpuProfile* gpu,
                                bitnet::IComputeBackend& backend, bitnet::IBuffer* wb,
                                bitnet::IBuffer* ab, bitnet::IBuffer* out,
                                const bitnet::KernelParams& base, std::vector<int32_t>& gpu_out,
                                const std::vector<int32_t>& ref, void* binding_override);

// ---------------------------------------------------------------------------
// Suites (one translation unit each)
// ---------------------------------------------------------------------------

int run_golden_vector_test(const std::string& golden_path);
int run_real_model_test(const Args& args, const bitnet::LoadedWeights& loaded);
int run_bitlinear_test(const bitnet::LoadedWeights& loaded);
int64_t run_coopmat_lane(const Shape& s, const bitnet::DeviceCaps& caps,
                         const bitnet::TernaryTensor& t, const bitnet::GpuProfile* gpu,
                         bitnet::IComputeBackend& backend, bitnet::IBuffer* out,
                         const bitnet::KernelParams& base, std::vector<int32_t>& gpu_out,
                         const std::vector<int32_t>& ref);

void print_extended_caps(const bitnet::DeviceCaps& caps);
void print_relevant_extensions(const std::vector<std::string>& extensions);
void write_caps(const char* path, const bitnet::DeviceCaps& caps,
                const std::vector<std::string>& extensions);
