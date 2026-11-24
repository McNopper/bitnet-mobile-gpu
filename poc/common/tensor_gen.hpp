// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

#pragma once
// Deterministic tensor generation for golden vectors.
// Same seed -> same tensors on every machine/toolchain (mt19937 + fixed
// distributions), so desktop and Android runs validate identical inputs.

#include "format.hpp"

namespace bitnet {

struct TernaryTensor {
    uint32_t m = 0;                  // activation rows
    uint32_t n = 0;                  // weight rows (outputs)
    uint32_t k = 0;                  // reduction dim
    bool binary = false;             // true: weights in {0,1}; false: {-1,0,+1}
    std::vector<int8_t> weights;     // n * k
    std::vector<int8_t> activations; // m * k
    std::vector<float> gamma;        // per-row absmean scale, size n
};

// seed is mixed with (m, n, k, binary) so shape changes don't shift all
// other vectors.
TernaryTensor make_tensor(uint32_t m, uint32_t n, uint32_t k, bool binary, uint32_t seed);

} // namespace bitnet
