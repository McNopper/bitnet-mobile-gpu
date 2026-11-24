// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

#include "common/format.hpp"

#include <stdexcept>

namespace bitnet {

PackedTernary PackedTernary::pack(const int8_t* w, uint32_t n, uint32_t k) {
    PackedTernary p;
    p.rows = n;
    p.cols = k;
    p.kwords = kwords_for(k);
    p.bits.assign(static_cast<size_t>(n) * kWeightBitPlanes * p.kwords, 0);
    for (uint32_t row = 0; row < n; ++row) {
        uint32_t* plane0 = &p.bits[static_cast<size_t>(row) * 2 * p.kwords];
        uint32_t* plane1 = plane0 + p.kwords;
        for (uint32_t col = 0; col < k; ++col) {
            const int8_t v = w[(static_cast<size_t>(row) * k) + col];
            if (v == 1) {
                plane0[col / kWeightsPerWord] |= 1U << (col % kWeightsPerWord);
            } else if (v == -1) {
                plane1[col / kWeightsPerWord] |= 1U << (col % kWeightsPerWord);
            } else if (v != 0) {
                // Contract violation: caller must pass ternary (or binary) values.
                throw std::invalid_argument("PackedTernary::pack: weight not in {-1,0,+1}");
            }
        }
    }
    return p;
}

std::vector<int8_t> PackedTernary::unpack() const {
    std::vector<int8_t> out(static_cast<size_t>(rows) * cols, 0);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t* plane0 = &bits[static_cast<size_t>(row) * 2 * kwords];
        const uint32_t* plane1 = plane0 + kwords;
        for (uint32_t col = 0; col < cols; ++col) {
            const int v =
                static_cast<int>((plane0[col / kWeightsPerWord] >> (col % kWeightsPerWord)) & 1U) -
                static_cast<int>((plane1[col / kWeightsPerWord] >> (col % kWeightsPerWord)) & 1U);
            out[(static_cast<size_t>(row) * cols) + col] = static_cast<int8_t>(v);
        }
    }
    return out;
}

PackedActivations PackedActivations::pack(const int8_t* a, uint32_t m, uint32_t k) {
    PackedActivations p;
    p.rows = m;
    p.cols = k;
    p.kwords = kwords_for(k);
    p.bits.assign(static_cast<size_t>(m) * kActBitPlanes * p.kwords, 0);
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < k; ++col) {
            // +128 offset: signed int8 -> unsigned planes (see contract).
            const auto u =
                static_cast<uint8_t>(a[(static_cast<size_t>(row) * k) + col] + kActOffset);
            uint32_t* planes = &p.bits[static_cast<size_t>(row) * kActBitPlanes * p.kwords];
            for (uint32_t plane = 0; plane < kActBitPlanes; ++plane) {
                if (u & (1U << plane)) {
                    planes[(plane * p.kwords) + (col / kWeightsPerWord)] |=
                        1U << (col % kWeightsPerWord);
                }
            }
        }
    }
    return p;
}

PackedTernaryDots PackedTernaryDots::pack(const int8_t* w, uint32_t n, uint32_t k) {
    PackedTernaryDots p;
    p.rows = n;
    p.cols = k;
    p.kwords = dots_kwords_for(k);
    // Pad bytes = 0x00 (weight 0).
    p.bits.assign(static_cast<size_t>(n) * p.kwords, 0U);
    for (uint32_t row = 0; row < n; ++row) {
        uint32_t* words = &p.bits[static_cast<size_t>(row) * p.kwords];
        for (uint32_t col = 0; col < k; ++col) {
            const int8_t v = w[(static_cast<size_t>(row) * k) + col];
            if (v < -1 || v > 1) {
                // Contract violation: caller must pass ternary (or binary) values.
                throw std::invalid_argument("PackedTernaryDots::pack: weight not in {-1,0,+1}");
            }
            // Dequantized int8 value (two's complement): -1 -> 0xFF, 0 -> 0x00,
            // +1 -> 0x01. dot4add_i8packed multiplies these bytes directly.
            const auto byte = static_cast<uint32_t>(static_cast<uint8_t>(v));
            words[col / kDotsUnitWords] |= byte << (kDotsLaneBits * (col % kDotsUnitWords));
        }
    }
    return p;
}

std::vector<int8_t> PackedTernaryDots::unpack() const {
    std::vector<int8_t> out(static_cast<size_t>(rows) * cols, 0);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t* words = &bits[static_cast<size_t>(row) * kwords];
        for (uint32_t col = 0; col < cols; ++col) {
            const uint32_t byte =
                (words[col / kDotsUnitWords] >> (kDotsLaneBits * (col % kDotsUnitWords))) & 0xFFU;
            const auto v = static_cast<int8_t>(byte);
            out[(static_cast<size_t>(row) * cols) + col] = v;
        }
    }
    return out;
}

PackedActivationsDots PackedActivationsDots::pack(const int8_t* a, uint32_t m, uint32_t k) {
    PackedActivationsDots p;
    p.rows = m;
    p.cols = k;
    p.kwords = dots_kwords_for(k); // same stride as the weight layout
    p.bits.assign(static_cast<size_t>(m) * p.kwords, 0);
    for (uint32_t row = 0; row < m; ++row) {
        uint32_t* words = &p.bits[static_cast<size_t>(row) * p.kwords];
        for (uint32_t col = 0; col < k; ++col) {
            // Bit pattern of the int8 value (two's complement).
            const auto byte = static_cast<uint32_t>(
                static_cast<uint8_t>(a[(static_cast<size_t>(row) * k) + col]));
            words[col / kDotsUnitWords] |= byte << (kDotsLaneBits * (col % kDotsUnitWords));
        }
    }
    return p;
}

std::vector<int8_t> PackedActivationsDots::unpack() const {
    std::vector<int8_t> out(static_cast<size_t>(rows) * cols, 0);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t* words = &bits[static_cast<size_t>(row) * kwords];
        for (uint32_t col = 0; col < cols; ++col) {
            const uint32_t byte =
                (words[col / kDotsUnitWords] >> (kDotsLaneBits * (col % kDotsUnitWords))) & 0xFFU;
            out[(static_cast<size_t>(row) * cols) + col] = static_cast<int8_t>(byte);
        }
    }
    return out;
}

PackedTernaryV6 PackedTernaryV6::pack(const int8_t* w, uint32_t n, uint32_t k) {
    PackedTernaryV6 p;
    p.rows = n;
    p.cols = k;
    p.units = v6_units_for(k);
    // Pad weights = 0 -> both plane bits 0.
    p.bits.assign(static_cast<size_t>(n) * p.units * kV6UnitWords, 0U);
    for (uint32_t row = 0; row < n; ++row) {
        uint32_t* words = &p.bits[static_cast<size_t>(row) * p.units * kV6UnitWords];
        for (uint32_t col = 0; col < k; ++col) {
            const int8_t v = w[(static_cast<size_t>(row) * k) + col];
            if (v < -1 || v > 1) {
                // Contract violation: caller must pass ternary (or binary) values.
                throw std::invalid_argument("PackedTernaryV6::pack: weight not in {-1,0,+1}");
            }
            if (v == 0) {
                continue;
            }
            const uint32_t unit = col / kV6WeightsPerUnit;
            const uint32_t in_unit = col % kV6WeightsPerUnit;
            // plane-major then half, within the unit: plus plane words
            // 0..1, minus plane words 2..3 (the plane-major mapping; see
            // format.hpp).
            const uint32_t word =
                (unit * kV6UnitWords) + (in_unit / kWeightsPerWord) + (v < 0 ? 2U : 0U);
            const uint32_t bit = in_unit % kWeightsPerWord;
            words[word] |= 1U << bit;
        }
    }
    return p;
}

std::vector<int8_t> PackedTernaryV6::unpack() const {
    std::vector<int8_t> out(static_cast<size_t>(rows) * cols, 0);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t* words = &bits[static_cast<size_t>(row) * units * kV6UnitWords];
        for (uint32_t col = 0; col < cols; ++col) {
            const uint32_t unit = col / kV6WeightsPerUnit;
            const uint32_t in_unit = col % kV6WeightsPerUnit;
            const uint32_t base = (unit * kV6UnitWords) + (in_unit / kWeightsPerWord);
            const uint32_t bit = in_unit % kWeightsPerWord;
            const int v = static_cast<int>((words[base] >> bit) & 1U) -
                          static_cast<int>((words[base + 2U] >> bit) & 1U);
            out[(static_cast<size_t>(row) * cols) + col] = static_cast<int8_t>(v);
        }
    }
    return out;
}

PackedActivationsV6 PackedActivationsV6::pack(const int8_t* a, uint32_t m, uint32_t k) {
    PackedActivationsV6 p;
    p.rows = m;
    p.cols = k;
    p.units = kV6UnitWords * v6_units_for(k);
    p.bits.assign(static_cast<size_t>(m) * p.units * kV6UnitWords, 0U);
    const uint32_t padded = v6_units_for(k) * kV6WeightsPerUnit;
    for (uint32_t row = 0; row < m; ++row) {
        uint32_t* words = &p.bits[static_cast<size_t>(row) * p.units * kV6UnitWords];
        // Padding columns store activation 0 (u = kActOffset) so unpack
        // recovers them exactly; they contribute zero regardless because
        // the padding weights are 0.
        for (uint32_t col = 0; col < padded; ++col) {
            const int8_t v =
                col < k ? a[(static_cast<size_t>(row) * k) + col] : static_cast<int8_t>(0);
            const auto u = static_cast<uint8_t>(v + kActOffset);
            const uint32_t block = col / kV6WeightsPerUnit;
            const uint32_t in_block = col % kV6WeightsPerUnit;
            const uint32_t half = in_block / kWeightsPerWord;
            const uint32_t bit = in_block % kWeightsPerWord;
            uint32_t* block_words = words + (static_cast<size_t>(block) * kV6ActWordsPerBlock);
            for (uint32_t plane = 0; plane < kActBitPlanes; ++plane) {
                if (u & (1U << plane)) {
                    block_words[(half * kActBitPlanes) + plane] |= 1U << bit;
                }
            }
        }
    }
    return p;
}

std::vector<int8_t> PackedActivationsV6::unpack() const {
    std::vector<int8_t> out(static_cast<size_t>(rows) * cols, 0);
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t* words = &bits[static_cast<size_t>(row) * units * kV6UnitWords];
        for (uint32_t col = 0; col < cols; ++col) {
            const uint32_t block = col / kV6WeightsPerUnit;
            const uint32_t in_block = col % kV6WeightsPerUnit;
            const uint32_t half = in_block / kWeightsPerWord;
            const uint32_t bit = in_block % kWeightsPerWord;
            const uint32_t* block_words =
                words + (static_cast<size_t>(block) * kV6ActWordsPerBlock);
            uint32_t u = 0U;
            for (uint32_t plane = 0; plane < kActBitPlanes; ++plane) {
                u |= ((block_words[(half * kActBitPlanes) + plane] >> bit) & 1U) << plane;
            }
            const int v = static_cast<int>(u) - static_cast<int>(kActOffset);
            out[(static_cast<size_t>(row) * cols) + col] = static_cast<int8_t>(v);
        }
    }
    return out;
}

PackedInt8MatPair PackedInt8MatPair::pack(const int8_t* w, const int8_t* a, uint32_t m, uint32_t n,
                                          uint32_t k, uint32_t tileM, uint32_t tileN,
                                          uint32_t tileK, bool transposeActs) {
    PackedInt8MatPair out;
    out.paddedM = (m + tileM - 1) / tileM * tileM;
    out.paddedN = (n + tileN - 1) / tileN * tileN;
    out.paddedK = (k + tileK - 1) / tileK * tileK;
    out.weights.assign(static_cast<size_t>(out.paddedN) * out.paddedK, 0);
    out.acts.assign(static_cast<size_t>(out.paddedK) * out.paddedM, 0);
    for (uint32_t row = 0; row < n; ++row) {
        for (uint32_t col = 0; col < k; ++col) {
            out.weights[(static_cast<size_t>(row) * out.paddedK) + col] =
                static_cast<uint8_t>(w[(static_cast<size_t>(row) * k) + col]);
        }
    }
    if (transposeActs) {
        // K-major: element (col k, row r) at k * paddedM + r.
        for (uint32_t row = 0; row < m; ++row) {
            for (uint32_t col = 0; col < k; ++col) {
                out.acts[(static_cast<size_t>(col) * out.paddedM) + row] =
                    static_cast<uint8_t>(a[(static_cast<size_t>(row) * k) + col]);
            }
        }
    } else {
        for (uint32_t row = 0; row < m; ++row) {
            for (uint32_t col = 0; col < k; ++col) {
                out.acts[(static_cast<size_t>(row) * out.paddedK) + col] =
                    static_cast<uint8_t>(a[(static_cast<size_t>(row) * k) + col]);
            }
        }
    }
    return out;
}

} // namespace bitnet
