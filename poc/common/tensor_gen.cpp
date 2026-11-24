// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

#include "tensor_gen.hpp"

#include <algorithm>
#include <random>

namespace bitnet {

// Deterministic tensor generation constants (seed-hash mixing + the int8
// activation range): part of the test-data contract - changing them changes
// every tensor and every golden comparison.
constexpr uint32_t kSeedMixPrime = 2654435761U; // Knuth multiplicative hash
constexpr uint32_t kSeedMixM = 97U;
constexpr uint32_t kSeedMixN = 31U;
constexpr int32_t kActMin = -128;
constexpr int32_t kActMax = 127;

TernaryTensor make_tensor(uint32_t m, uint32_t n, uint32_t k, bool binary, uint32_t seed) {
    std::mt19937 rng((seed * kSeedMixPrime) ^ ((m * kSeedMixM) + (n * kSeedMixN) + k));
    TernaryTensor t;
    t.m = m;
    t.n = n;
    t.k = k;
    t.binary = binary;
    t.weights.resize(static_cast<size_t>(n) * k);
    t.activations.resize(static_cast<size_t>(m) * k);
    t.gamma.resize(n);

    // Ternary: uniform over {-1,0,+1}; binary: uniform over {0,1}.
    std::uniform_int_distribution<int> wd(0, binary ? 1 : 2);
    std::ranges::generate(t.weights, [&wd, &rng, binary] {
        return static_cast<int8_t>(binary ? wd(rng) : wd(rng) - 1);
    });
    std::uniform_int_distribution<int> ad(kActMin, kActMax);
    std::ranges::generate(t.activations, [&ad, &rng] { return static_cast<int8_t>(ad(rng)); });

    // Synthetic per-row scales exercise the harness scale path; model
    // exports use tensor-wide gamma (BitNet b1.58 reference quantizer).
    // With ternary values that equals the fraction of non-zero weights.
    for (uint32_t r = 0; r < n; ++r) {
        const int8_t* row = &t.weights[static_cast<size_t>(r) * k];
        int64_t abs_sum = 0;
        for (uint32_t c = 0; c < k; ++c) {
            abs_sum += row[c] < 0 ? -static_cast<int64_t>(row[c]) : static_cast<int64_t>(row[c]);
        }
        t.gamma[r] = static_cast<float>(static_cast<double>(abs_sum) / static_cast<double>(k));
    }
    return t;
}

} // namespace bitnet
