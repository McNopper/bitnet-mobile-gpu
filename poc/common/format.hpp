// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

#pragma once
// BitNet packed-format contract.
//
// Single source of truth for the CPU reference, the Slang kernels and the
// backends. Changing anything here is a contract change: bump
// kFormatVersion, regenerate golden vectors, re-validate all strategies.
//
// Weights: ternary {-1, 0, +1} matrix W[N][K] (N outputs, K inputs).
//   Packed as two bit-planes of uint32 words, row-major over N:
//     bit0 plane, word j, bit b : weight (32*j + b) == +1
//     bit1 plane, word j, bit b : weight (32*j + b) == -1
//   so weight = bit0 - bit1. 1-bit weights (PoC): values {0,1} -> bit1 all
//   zero. Padding bits (K not a multiple of 32) stay 0 on both planes, which
//   contributes exactly zero to the accumulator.
//
// Activations: int8 matrix A[M][K]. Stored as uint8 with +128 offset
//   (u = a + 128), then split into 8 bit-planes of uint32 words, row-major
//   over M: plane p, word j, bit b : bit p of u (32*j + b).
//   Correction identity used by every kernel:
//     sum_j w_j * a_j = sum_j w_j * u_j - 128 * sum_j w_j
//                     = (popcount planes of u vs w0/w1) - 128 * rowsum_w
//   Padding words of u are 0 -> contribute zero.
//
// Accumulation: int32 throughout. The largest intermediate sum for
// K=6912 is 255 * 6912 = 1,762,560 — far below 2^31, so the integer part
// is bit-exact reproducible on CPU and GPU.
// The per-row fp32 scale gamma (absmean) is applied by the caller AFTER the
// integer pass, keeping the kernel pure integer.

#include <cstdint>
#include <vector>

namespace bitnet {

constexpr uint32_t kFormatVersion = 3; // v3: adds the v6 plane-uint4 layouts

constexpr uint32_t kWeightsPerWord = 32;    // plane format: weights per uint32 word
constexpr uint32_t kWeightBitPlanes = 2;    // ternary: +1 plane and -1 plane
constexpr uint32_t kActBitPlanes = 8;       // activation bit-planes (int8, +128 offset)
constexpr int32_t kActOffset = 128;         // activation storage offset (u = a + kActOffset)
constexpr uint32_t kWeightsPerDotsWord = 4; // int8 values, 4 per uint32 word

constexpr uint32_t kwords_for(uint32_t k) {
    return (k + 31) / 32;
}

// Packed-dot unit geometry (central constants - no magic numbers; the
// Slang kernels in poc/slang/bitnet_kernels.slang use the same values, and
// the bit-exact harness enforces host/shader agreement on every run).
constexpr uint32_t kDotsUnitWords = 4; // int8 values packed per uint32 word
constexpr uint32_t kWordBytes = 4;     // bytes per uint32 word (all layouts)
constexpr uint32_t kInt32Bytes = 4;    // sizeof(int32_t) accumulator/output value
constexpr uint32_t kDotsLaneBits = 8;  // bits per packed lane (one int8 byte)

// ceil(K / kDotsUnitWords): uint32 data words per row in BOTH packed-dot
// layouts (weights and activations share the stride, so the kernel uses one
// KernelParams.kwords).
constexpr uint32_t dots_kwords_for(uint32_t k) {
    return (k + kDotsUnitWords - 1) / kDotsUnitWords;
}

// v6 unit geometry (format v3): one uint4 weight unit = 64 ternary weights
// stored as FOUR plane words {plus_lo, plus_hi, minus_lo, minus_hi}
// (plane-major, then half), so a single 128-bit load delivers both planes
// of both 32-weight halves. Activations share the 64-column grid: 16 words
// per block grouped into 4 uint4 units, so one weight unit pairs with
// exactly four activation uint4 units - 5 uint4 loads per 64 columns
// instead of 20 scalar word loads in the bit-plane layout.
constexpr uint32_t kV6WeightsPerUnit = 64;   // weights per uint4 unit
constexpr uint32_t kV6UnitWords = 4;         // words per weight unit
constexpr uint32_t kV6ActWordsPerBlock = 16; // words per 64-activation block

constexpr uint32_t v6_units_for(uint32_t k) {
    return (k + kV6WeightsPerUnit - 1) / kV6WeightsPerUnit;
}

struct PackedTernary {
    uint32_t rows = 0;   // N
    uint32_t cols = 0;   // K (unpadded)
    uint32_t kwords = 0; // ceil(K / 32)
    // Layout: [row][plane(0..1)][word(0..kwords)]  -> size rows * 2 * kwords
    std::vector<uint32_t> bits;
    static PackedTernary pack(const int8_t* w, uint32_t n, uint32_t k);
    // Test helper: recover {-1,0,+1} values incl. padding = 0.
    std::vector<int8_t> unpack() const;
    uint64_t bytes() const { return uint64_t(rows) * 2 * kwords * sizeof(uint32_t); }
};

// Interleaved-16 layout (alternate buffer, not a change to the default
// contract): blocks of 16 rows; within a block, for each word index w, the
// 16 rows' plus-plane words are contiguous, then their minus-plane words.
// Addressing: out[b*(2*kwords*16) + w*16 + r] (plus) and
//             out[b*(2*kwords*16) + kwords*16 + w*16 + r] (minus),
// so a workgroup whose threads own consecutive rows reads 16 CONTIGUOUS
// words per plane per step (coalesced), instead of 16 rows' strided planes.
PackedTernary repack_interleaved16(const PackedTernary& src);
// Inverse permutation for the roundtrip unit test (proof it is lossless).
std::vector<int8_t> unpack_interleaved16(const PackedTernary& src);

struct PackedActivations {
    uint32_t rows = 0;   // M
    uint32_t cols = 0;   // K (unpadded)
    uint32_t kwords = 0; // ceil(K / 32)
    // Layout: [row][plane(0..7)][word]  -> size rows * 8 * kwords
    std::vector<uint32_t> bits;

    static PackedActivations pack(const int8_t* a, uint32_t m, uint32_t k);
    uint64_t bytes() const { return uint64_t(rows) * 8 * kwords * sizeof(uint32_t); }
};

// Uniform block (set 0, binding 3), scalar layout, shared with the shaders.
struct KernelParams {
    uint32_t m;      // activation rows
    uint32_t n;      // weight rows (outputs)
    uint32_t kwords; // plane kernels (v1-v4): ceil(K / 32); dot kernels
                     // (v5): ceil(K / 16) - each kernel family reads its own
                     // layout definition from this header
    uint32_t mode;   // 0 = ternary, 1 = binary (reserved; v1 kernels always
                     // read both planes, binary inputs simply carry a zero
                     // second plane)
};
static_assert(sizeof(KernelParams) == 16, "shader layout contract");
//
// Motivation: the dot path replaces the per-plane popcount chain with
// ONE 4x8 packed integer dot per 4 weights. Integer dot product is core
// since Vulkan 1.3 (docs.vulkan.org refpage
// VK_KHR_shader_integer_dot_product).
//
// Weights: same ternary matrix W[N][K]. DEQUANTIZED int8 values packed
// 4 per uint32 word (little-endian byte order; byte = int8 bit pattern,
// -1 -> 0xFF, 0 -> 0x00, +1 -> 0x01), so dot4add_i8packed(weightWord,
// actWord) is EXACTLY the 4-element sum of w*a - no correction term is
// needed (the row-sum correction belongs to the bit-plane formats with
// offset activations). 2-bit codes cannot be dotted directly - code bytes
// are not linear in the weights. This layout stores 1 B/weight against
// 0.25 B/weight for the plane formats. Padding bytes are 0x00 (weight 0).
struct PackedTernaryDots {
    uint32_t rows = 0;   // N
    uint32_t cols = 0;   // K (unpadded)
    uint32_t kwords = 0; // ceil(K / 4)
    // Layout: [row][word] -> size rows * kwords
    std::vector<uint32_t> bits;
    static PackedTernaryDots pack(const int8_t* w, uint32_t n, uint32_t k);
    // Test helper: recover {-1,0,+1} values incl. padding = 0.
    std::vector<int8_t> unpack() const;
    uint64_t bytes() const { return uint64_t(rows) * kwords * sizeof(uint32_t); }
};

// Packed-dot activation layout (v5).
//
// Raw int8 activations (NO +128 offset - the dot path needs signed values),
// 4 bytes per uint32 word (little-endian byte order), ceil(K / 4) words per
// row - the SAME stride as the weight layout, so the kernel uses one
// KernelParams.kwords for both buffers. Padding bytes are 0.
struct PackedActivationsDots {
    uint32_t rows = 0;   // M
    uint32_t cols = 0;   // K (unpadded)
    uint32_t kwords = 0; // ceil(K / 4) - same stride as the weight layout
    // Layout: [row][word] -> size rows * kwords.
    std::vector<uint32_t> bits;
    static PackedActivationsDots pack(const int8_t* a, uint32_t m, uint32_t k);
    // Test helper: recover the int8 activations incl. padding = 0.
    std::vector<int8_t> unpack() const;
    uint64_t bytes() const { return uint64_t(rows) * kwords * sizeof(uint32_t); }
};

// v6 plane-uint4 weight layout (format v3).
//
// One unit = kV6WeightsPerUnit weights = kV6UnitWords words, ordered
// plane-major then half:
//   word 0: +1 plane, weights 64u + 0..31   (plus_lo)
//   word 1: +1 plane, weights 64u + 32..63  (plus_hi)
//   word 2: -1 plane, weights 64u + 0..31   (minus_lo)
//   word 3: -1 plane, weights 64u + 32..63  (minus_hi)
// so ONE uint4 load delivers both planes of both halves, and the v4
// popcount identity (weight = plus_bit - minus_bit) is preserved. The
// unit test test_v6_ternary_layout encodes this exact mapping, so a
// packing in which two planes share a word cannot pass silently.
// Padding weights stay 0 on both planes.
struct PackedTernaryV6 {
    uint32_t rows = 0;  // N
    uint32_t cols = 0;  // K (unpadded)
    uint32_t units = 0; // ceil(K / 64)
    // Layout: [row][unit u][word 0..3], size rows * units * kV6UnitWords.
    std::vector<uint32_t> bits;

    static PackedTernaryV6 pack(const int8_t* w, uint32_t n, uint32_t k);
    // Test helper: recover {-1,0,+1} values incl. padding = 0.
    std::vector<int8_t> unpack() const;
    uint64_t bytes() const { return uint64_t(rows) * units * kV6UnitWords * sizeof(uint32_t); }
};

// v6 activation layout: the 8 bit-planes of PackedActivations, re-grouped
// onto the same 64-column unit grid as the weights. Per 64-activation
// block a: kV6ActWordsPerBlock words, indexed 8*half + plane (half 0 =
// columns 64a + 0..31, half 1 = columns 64a + 32..63), grouped into four
// uint4 units:
//   unit 4a + 0 = planes 0..3, half 0     unit 4a + 1 = planes 4..7, half 0
//   unit 4a + 2 = planes 0..3, half 1     unit 4a + 3 = planes 4..7, half 1
// A weight unit u (all four words) therefore pairs with exactly the four
// activation uint4 units 4u + 0..3. Padding columns are stored as
// activation 0 (u = kActOffset) so unpack recovers them exactly; padding
// WEIGHTS are 0, so padding lanes contribute exactly zero to the
// accumulator either way (proven bit-exactly by the GPU harness).
struct PackedActivationsV6 {
    uint32_t rows = 0;  // M
    uint32_t cols = 0;  // K (unpadded)
    uint32_t units = 0; // kV6UnitWords * ceil(K / 64) uint4 units per row
    // Layout: [row][block a][word 8*half + plane], size
    // rows * units * kV6UnitWords words.
    std::vector<uint32_t> bits;

    static PackedActivationsV6 pack(const int8_t* a, uint32_t m, uint32_t k);
    // Test helper: recover the int8 activations incl. padding = 0.
    std::vector<int8_t> unpack() const;
    uint64_t bytes() const { return uint64_t(rows) * units * kV6UnitWords * sizeof(uint32_t); }
};

// Coop-mat lane operand layout (ADDITIVE - the packed layouts above and
// kFormatVersion are untouched): plain row-major int8 as consumed directly
// by the VK_KHR_cooperative_matrix kernel (gemm_coopmat in
// bitnet_kernels.slang). Rows/columns are zero-padded to the lane's tile
// multiples so the collective 16x16x32 loads never leave the buffer and the
// padding contributes exactly 0 (integer-exact vs the CPU oracle). Weights
// are the ternary {-1,0,+1} values verbatim; activations the int8 values
// verbatim. The kernel's KernelParams.kwords carries paddedK (row stride).
// transposeActs=true stores the ACTIVATIONS transposed (K-major:
// paddedK rows x paddedM columns): the kernel loads A ColumnMajor.
struct PackedInt8MatPair {
    uint32_t paddedM = 0;
    uint32_t paddedN = 0;
    uint32_t paddedK = 0;
    std::vector<uint8_t> weights; // paddedN * paddedK bytes (n-major rows)
    std::vector<uint8_t> acts;    // paddedK * paddedM bytes when transposed

    static PackedInt8MatPair pack(const int8_t* w, const int8_t* a, uint32_t m, uint32_t n,
                                  uint32_t k, uint32_t tileM, uint32_t tileN, uint32_t tileK,
                                  bool transposeActs);
    uint64_t weightBytes() const { return uint64_t(paddedN) * paddedK; }
    uint64_t actBytes() const { return uint64_t(paddedK) * paddedM; }
};

} // namespace bitnet
