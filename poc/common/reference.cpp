// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

#include "reference.hpp"

namespace bitnet {

void matmul_ref(const int8_t* w, const int8_t* a, int32_t* out, uint32_t m, uint32_t n,
                uint32_t k) {
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t c = 0; c < n; ++c) {
            const int8_t* wrow = w + (static_cast<size_t>(c) * k);
            const int8_t* arow = a + (static_cast<size_t>(i) * k);
            int32_t acc = 0;
            for (uint32_t t = 0; t < k; ++t) {
                acc += static_cast<int32_t>(wrow[t]) * static_cast<int32_t>(arow[t]);
            }
            out[(static_cast<size_t>(i) * n) + c] = acc;
        }
    }
}

void apply_row_scale(const int32_t* in, const float* gamma, float* out, uint32_t m, uint32_t n) {
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t c = 0; c < n; ++c) {
            out[(static_cast<size_t>(i) * n) + c] =
                static_cast<float>(in[(static_cast<size_t>(i) * n) + c]) * gamma[c];
        }
    }
}

} // namespace bitnet
