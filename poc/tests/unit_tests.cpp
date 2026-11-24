// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Host-side unit tests: packing contract, reference math, tensor generation
// determinism, BITNET1 container roundtrip, version decoding. These prove
// the components in isolation; kernel-vs-oracle exactness is proven per
// device by poc_tests (the GPU harness).
//
// Framework: minimal assert-style runner - llama.cpp's "avoid third-party
// dependencies" guideline applies to the repo, and the assertions needed
// here fit on one screen. If the suite grows beyond ~30 cases, move to
// Catch2 via FetchContent (documented decision required).

#include "backend/backend.hpp"
#include "backend/caps_conformance.hpp"
#include "common/bitnet1.hpp"
#include "common/format.hpp"
#include "common/reference.hpp"
#include "common/tensor_gen.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

int g_count = 0;
int g_failed = 0;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        ++g_count;                                                                                 \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                            \
            ++g_failed;                                                                            \
        }                                                                                          \
    } while (0)

void test_kwords_for() {
    CHECK(bitnet::kwords_for(1) == 1);
    CHECK(bitnet::kwords_for(31) == 1);
    CHECK(bitnet::kwords_for(32) == 1);
    CHECK(bitnet::kwords_for(33) == 2);
    CHECK(bitnet::kwords_for(64) == 2);
    CHECK(bitnet::kwords_for(65) == 3);
    CHECK(bitnet::kwords_for(6912) == 216);
}

void test_pack_layout() {
    // One row of three weights: +1, 0, -1 -> plane0 bit0, plane1 bit2.
    const std::array<int8_t, 3> w = {1, 0, -1};
    const auto p = bitnet::PackedTernary::pack(w.data(), 1, 3);
    CHECK(p.rows == 1);
    CHECK(p.cols == 3);
    CHECK(p.kwords == 1);
    CHECK(p.bits.size() == 2);
    CHECK(p.bits[0] == 0x1U); // plane0: bit0 = +1
    CHECK(p.bits[1] == 0x4U); // plane1: bit2 = -1
}

void test_pack_roundtrip_padding() {
    // K = 33 crosses a word boundary; padding bits must stay 0 and must not
    // perturb the unpacked values (roundtrip) or the accumulator (all-zero
    // padding contributes nothing - proven bit-exactly by the GPU harness).
    // fixed seed IS the test: determinism.
    // NOLINTNEXTLINE(bugprone-random-generator-seed)
    std::mt19937 rng(7U);
    std::uniform_int_distribution<int> wd(0, 2);
    std::vector<int8_t> w(65);
    std::ranges::generate(w, [&wd, &rng] { return static_cast<int8_t>(wd(rng) - 1); });
    const auto p = bitnet::PackedTernary::pack(w.data(), 1, 65);
    CHECK(p.kwords == 3);
    const auto back = p.unpack();
    CHECK(back == w);
}

void test_pack_rejects_non_ternary() {
    const std::array<int8_t, 2> bad = {1, 5};
    bool threw = false;
    try {
        static_cast<void>(bitnet::PackedTernary::pack(bad.data(), 1, 2));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_activation_planes() {
    // a = -128 -> u = 0 -> all planes zero; a = 127 -> u = 255 -> every plane
    // has its bits set. kwords = 1: both weights share ONE word per plane
    // (weight col0 -> bit0, col1 -> bit1).
    const std::array<int8_t, 2> a = {-128, 127};
    const auto p = bitnet::PackedActivations::pack(a.data(), 1, 2);
    CHECK(p.kwords == 1);
    bool all_zero = true;
    bool all_set = true;
    for (uint32_t plane = 0; plane < 8; ++plane) {
        const uint32_t word = p.bits[static_cast<size_t>(plane) * p.kwords];
        all_zero = all_zero && ((word & 1U) == 0);       // col0 (a=-128)
        all_set = all_set && (((word >> 1) & 1U) == 1U); // col1 (a=127)
    }
    CHECK(all_zero);
    CHECK(all_set);
}

void test_matmul_ref_known() {
    // W = [[1, 0, -1], [0, 1, 1]], A = [[2, 1, 0], [1, -1, 3]]
    // out[0][0] = W0.A0 = 2 | out[0][1] = W1.A0 = 1
    // out[1][0] = W0.A1 = -2 | out[1][1] = W1.A1 = 2
    const std::array<int8_t, 6> w = {1, 0, -1, 0, 1, 1};
    const std::array<int8_t, 6> a = {2, 1, 0, 1, -1, 3};
    std::array<int32_t, 4> out = {0, 0, 0, 0};
    bitnet::matmul_ref(w.data(), a.data(), out.data(), 2, 2, 3);
    CHECK(out[0] == 2);
    CHECK(out[1] == 1);
    CHECK(out[2] == -2);
    CHECK(out[3] == 2);
}

void test_row_scale() {
    const std::array<int32_t, 2> in = {100, -200};
    const std::array<float, 2> gamma = {0.5F, 2.0F};
    std::array<float, 2> out = {0.0F, 0.0F};
    bitnet::apply_row_scale(in.data(), gamma.data(), out.data(), 1, 2);
    CHECK(out[0] == 50.0F);
    CHECK(out[1] == -400.0F);
}

void test_tensor_gen_determinism() {
    const auto a = bitnet::make_tensor(4, 8, 96, false, 42);
    const auto b = bitnet::make_tensor(4, 8, 96, false, 42);
    CHECK(a.weights == b.weights);
    CHECK(a.activations == b.activations);
    CHECK(a.gamma == b.gamma);

    const auto c = bitnet::make_tensor(4, 8, 96, false, 43);
    CHECK(c.weights != a.weights);
}

void test_tensor_gen_properties() {
    const auto t = bitnet::make_tensor(8, 16, 96, false, 5);
    bool ternary = true;
    for (const int8_t v : t.weights) {
        ternary = ternary && (v == -1 || v == 0 || v == 1);
    }
    CHECK(ternary);
    for (uint32_t r = 0; r < t.n; ++r) {
        int64_t abs_sum = 0;
        for (uint32_t c = 0; c < t.k; ++c) {
            abs_sum += t.weights[static_cast<size_t>(r) * t.k + c] < 0
                           ? -static_cast<int64_t>(t.weights[static_cast<size_t>(r) * t.k + c])
                           : static_cast<int64_t>(t.weights[static_cast<size_t>(r) * t.k + c]);
        }
        const auto expected =
            static_cast<float>(static_cast<double>(abs_sum) / static_cast<double>(t.k));
        CHECK(std::fabs(t.gamma[r] - expected) < 1e-6F);
    }
}

void test_version_decode() {
    // 0x403000 encodes 1.3.0 (observed on the Mali device).
    const uint32_t v = (1U << 22) | (3U << 12);
    CHECK(bitnet::version_major(v) == 1);
    CHECK(bitnet::version_minor(v) == 3);
    CHECK(bitnet::version_patch(v) == 0);
}

void test_bitnet1_roundtrip() {
    const std::array<int8_t, 6> w = {1, 0, -1, 0, 1, 1};
    const std::array<float, 2> gamma = {0.25F, 1.5F};
    const std::string path = "bitnet1_unit_test.bin";
    CHECK(bitnet::write_bitnet1(path, w.data(), 2, 3, gamma.data()));

    bitnet::LoadedWeights loaded;
    CHECK(bitnet::load_bitnet1(path, loaded));
    CHECK(loaded.n == 2);
    CHECK(loaded.k == 3);
    CHECK(loaded.weights.size() == 6);
    CHECK(std::memcmp(loaded.weights.data(), w.data(), 6) == 0);
    CHECK(loaded.gamma[0] == 0.25F);
    CHECK(loaded.gamma[1] == 1.5F);
    std::remove(path.c_str());
}

void test_bitnet1_rejects_corrupt() {
    const std::string path = "bitnet1_unit_test_corrupt.bin";
    {
        std::ofstream bad(path, std::ios::binary);
        bad << "XXXX";
    }
    bitnet::LoadedWeights rejected;
    CHECK(!bitnet::load_bitnet1(path, rejected));
    std::remove(path.c_str());
}

void test_v6_units_for() {
    CHECK(bitnet::v6_units_for(1) == 1);
    CHECK(bitnet::v6_units_for(63) == 1);
    CHECK(bitnet::v6_units_for(64) == 1);
    CHECK(bitnet::v6_units_for(65) == 2);
    CHECK(bitnet::v6_units_for(2560) == 40);
    CHECK(bitnet::v6_units_for(6912) == 108);
}

void test_v6_ternary_layout() {
    // The plane-major mapping as a known answer: one 64-weight unit, plane-major
    // then half {plus_lo, plus_hi, minus_lo, minus_hi}. A packing in which
    // lo-minus and hi-plus share a word, or hi-minus leaves the unit,
    // fails here.
    std::vector<int8_t> w(64, 0);
    w[0] = 1;   // plus_lo,  bit 0  -> word 0
    w[1] = -1;  // minus_lo, bit 1  -> word 2
    w[32] = 1;  // plus_hi,  bit 0  -> word 1
    w[33] = -1; // minus_hi, bit 1  -> word 3
    w[63] = -1; // minus_hi, bit 31 -> word 3
    const auto p = bitnet::PackedTernaryV6::pack(w.data(), 1, 64);
    CHECK(p.units == 1);
    CHECK(p.bits.size() == 4);
    CHECK(p.bits[0] == 0x1U);
    CHECK(p.bits[1] == 0x1U);
    CHECK(p.bits[2] == 0x2U);
    CHECK(p.bits[3] == 0x80000002U);
    const auto back = p.unpack();
    CHECK(back == w);
}

void test_v6_ternary_roundtrip() {
    // Random ternary rows incl. lengths crossing the unit and half
    // boundaries; padding weights must stay 0 (roundtrip proves it).
    // fixed seed IS the test: determinism.
    // NOLINTNEXTLINE(bugprone-random-generator-seed)
    std::mt19937 rng(11U);
    std::uniform_int_distribution<int> wd(-1, 1);
    for (uint32_t k : {1U, 33U, 63U, 64U, 65U, 513U, 6912U}) {
        const uint32_t n = 2; // two rows: proves the per-row unit offset
        std::vector<int8_t> w(static_cast<size_t>(n) * k);
        std::ranges::generate(w, [&wd, &rng] { return static_cast<int8_t>(wd(rng)); });
        const auto p = bitnet::PackedTernaryV6::pack(w.data(), n, k);
        CHECK(p.units == bitnet::v6_units_for(k));
        const auto back = p.unpack();
        CHECK(back == w);
    }
}

void test_v6_activations_layout() {
    // Two activations in one block: a[0] = 1 (u = 129 -> planes 0 and 7,
    // half 0 bit 0) and a[63] = -1 (u = 127 -> planes 0..6, half 1 bit 31;
    // u = 127 has no bit 7). Padding (none here) would be u = 128.
    std::vector<int8_t> a(64, 0);
    a[0] = 1;
    a[63] = -1;
    const auto p = bitnet::PackedActivationsV6::pack(a.data(), 1, 64);
    CHECK(p.units == 4);
    CHECK(p.bits.size() == 16);
    CHECK(p.bits[0] == 0x1U);         // plane 0, half 0, bit 0 (a[0] = 1 -> 129)
    CHECK(p.bits[7] == 0xFFFFFFFFU);  // plane 7, half 0: a[0] = 1 -> 129 sets
                                      // bit 0; a[1..31] = 0 -> 128 sets bits 1..31
    CHECK(p.bits[8] == 0x80000000U);  // plane 0, half 1, bit 31 (a[63] -> 127)
    CHECK(p.bits[14] == 0x80000000U); // plane 6, half 1, bit 31
    CHECK(p.bits[15] == 0x7FFFFFFFU); // plane 7, half 1: a[32..62] = 0 -> 128
                                      // sets bits 0..30; 127 has no bit 7
    const auto back = p.unpack();
    CHECK(back == a);
}

void test_v6_activations_roundtrip() {
    // Random int8 rows incl. lengths crossing the block boundary; padding
    // activations (u = kActOffset) must recover as exactly 0.
    // NOLINTNEXTLINE(bugprone-random-generator-seed)
    std::mt19937 rng(12U);
    std::uniform_int_distribution<int> ad(-128, 127);
    for (uint32_t k : {1U, 33U, 63U, 64U, 65U, 513U}) {
        const uint32_t n = 2; // two rows: proves the per-row block offset
        std::vector<int8_t> a(static_cast<size_t>(n) * k);
        std::ranges::generate(a, [&ad, &rng] { return static_cast<int8_t>(ad(rng)); });
        const auto p = bitnet::PackedActivationsV6::pack(a.data(), n, k);
        CHECK(p.units == 4 * bitnet::v6_units_for(k));
        const auto back = p.unpack();
        CHECK(back == a);
    }
}

// Metal stub contract (prepared-only): the factory constructs, the
// capability snapshot reports Metal as unavailable, and every dispatch
// method throws the clear error - never a silent fallback.
void test_metal_stub_contract() {
    auto metal = bitnet::make_metal_backend();
    const auto caps = metal->probe_caps();
    CHECK(caps.device_name.contains("stub"));
    CHECK(caps.api_version == 0U); // no Metal runtime: unavailable
    CHECK(!metal->validation_active());
    CHECK(metal->validation_message_count() == 0U);
    bool threw = false;
    try {
        static_cast<void>(metal->create_buffer(16U, true));
    } catch (const std::exception& e) {
        threw = std::string(e.what()).contains("PREPARED-ONLY stub");
    }
    CHECK(threw);
}

bool groups_equal(const std::array<uint64_t, 3>& groups, uint64_t x, uint64_t y, uint64_t z) {
    return groups == std::array<uint64_t, 3>{x, y, z};
}

// Caps-conformance dispatch geometry: every matmul kernel has an exact
// grid and the K-split champion is checked with its real 3D grid.
void test_conformance_dispatch_groups() {
    const bitnet::GpuProfile* gpu = bitnet::gpu_profile_for_device("no such device");
    CHECK(std::string(gpu->profile_dir) == "default");
    CHECK(groups_equal(bitnet::dispatch_groups("gemv_bitplane_ksplit", *gpu, 1, 128256, 2560), 501,
                       1, 10));
    CHECK(groups_equal(bitnet::dispatch_groups("gemv_bitplane_rowcoop", *gpu, 1, 128256, 2560),
                       128256, 1, 1));
    CHECK(groups_equal(bitnet::dispatch_groups("gemm_bitplane_cellcoop", *gpu, 64, 2560, 6912),
                       163840, 1, 1));
    CHECK(groups_equal(bitnet::dispatch_groups("gemm_bitplane_naive", *gpu, 64, 2560, 6912), 160, 4,
                       1));
    CHECK(groups_equal(bitnet::dispatch_groups("gemm_bitplane_ksplit", *gpu, 64, 2560, 6912), 40,
                       64, 27));
    // An unfilled champion (sweep profiles) has nothing to dispatch-check.
    CHECK(groups_equal(bitnet::dispatch_groups("", *gpu, 1, 2560, 2560), 1, 1, 1));
}

// An unknown champion fails closed, and every matmul kernel of the table
// has a dispatch contract (a new kernel without one fails here, not as an
// unchecked "OK" on a device).
void test_conformance_dispatch_fail_closed() {
    const bitnet::GpuProfile* gpu = bitnet::gpu_profile_for_device("no such device");
    const auto unknown = bitnet::dispatch_groups("gemv_does_not_exist", *gpu, 1, 2560, 2560);
    CHECK(unknown[0] == bitnet::kUncheckedGroups);
    const bool all_known =
        std::ranges::all_of(bitnet::kKernelTable, [gpu](const bitnet::KernelInfo& info) {
            return info.family == bitnet::KernelFamily::Probe ||
                   bitnet::dispatch_groups(info.name, *gpu, info.is_gemv ? 1U : 64U, 2560,
                                           2560)[0] != bitnet::kUncheckedGroups;
        });
    CHECK(all_known);
}

} // namespace

int main() {
    // Nothing may escape main (bugprone-exception-escape).
    try {
        test_kwords_for();
        test_pack_layout();
        test_pack_roundtrip_padding();
        test_pack_rejects_non_ternary();
        test_activation_planes();
        test_matmul_ref_known();
        test_row_scale();
        test_tensor_gen_determinism();
        test_tensor_gen_properties();
        test_version_decode();
        test_bitnet1_roundtrip();
        test_bitnet1_rejects_corrupt();
        test_v6_units_for();
        test_v6_ternary_layout();
        test_v6_ternary_roundtrip();
        test_v6_activations_layout();
        test_v6_activations_roundtrip();
        test_metal_stub_contract();
        test_conformance_dispatch_groups();
        test_conformance_dispatch_fail_closed();

        std::printf("unit tests: %d run, %d failed\n", g_count, g_failed);
        return g_failed == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    } catch (...) {
        std::printf("FAIL: unknown exception\n");
        return 1;
    }
}
