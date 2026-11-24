// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

#pragma once
// Independent CPU oracle: the DEFINITION of the operation, computed
// naively. It deliberately does NOT use bit-planes or popcount — agreement
// with the GPU kernel therefore proves the bitplane trick, not a shared
// implementation.

#include "format.hpp"

namespace bitnet {

// out[m * n + c] = sum_k w[c * k + t] * a[m * k + t], exact int32.
void matmul_ref(const int8_t* w, const int8_t* a, int32_t* out, uint32_t m, uint32_t n, uint32_t k);

// out[m * n + c] = in[m * n + c] * gamma[c]  (fp32, IEEE, no fast-math).
void apply_row_scale(const int32_t* in, const float* gamma, float* out, uint32_t m, uint32_t n);

} // namespace bitnet
