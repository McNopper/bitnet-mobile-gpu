// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Real-model lanes of the harness: a layer of the model from the BITNET1
// container (activations synthetic) and the BitLinear forward pass that wraps
// the kernel with the reference quantizer and the rescale epilogue.

#include "common/reference.hpp"
#include "common/tensor_gen.hpp"
#include "tests/suites.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

constexpr int32_t kSynthActMin = -128; // real-model synthetic activation range
constexpr int32_t kSynthActMax = 127;

// Real-model weights container: loaded via bitnet::load_bitnet1
// (poc/common/bitnet1.hpp) - written by tools/export_weights.py.

// Real-model weights test (Path A): weights from the BITNET1 container,
// activations synthetic. Returns 0 on pass.
int run_real_model_test(const Args& args, const bitnet::LoadedWeights& loaded) {
    const Shape s{.m = 8, .n = loaded.n, .k = loaded.k, .binary = false};
    std::printf("[real-model] m=8 n=%u k=%u ternary (weights: BitNet-2B layer; "
                "activations synthetic)\n",
                loaded.n, loaded.k);
    try {
        bitnet::TernaryTensor t;
        t.m = s.m;
        t.n = s.n;
        t.k = s.k;
        t.binary = false;
        t.weights = loaded.weights;
        t.gamma = loaded.gamma;
        std::mt19937 rng(args.seed ^ loaded.k);
        std::uniform_int_distribution<int> ad(kSynthActMin, kSynthActMax);
        t.activations.resize(static_cast<size_t>(s.m) * s.k);
        std::ranges::generate(t.activations, [&ad, &rng] { return static_cast<int8_t>(ad(rng)); });
        std::vector<int32_t> gpu_out;
        const int64_t mm = compare_tensor(s, t, gpu_out);
        if (mm != 0) {
            std::printf("  FAIL: %lld mismatches\n", static_cast<long long>(mm));
            return 1;
        }
        std::printf("  PASS (bit-exact)\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("  FAIL: %s\n", e.what());
        return 1;
    }
}

// BitLinear forward: float activations in, float
// outputs out, on the REAL layer weights. The chain under test:
//   1. per-row (= per-token) absmax int8 activation quantization, as in the
//      reference implementation (Hugging Face transformers,
//      integrations/bitnet.py): scale = 127 / max|x| with a 1e-5 floor,
//      q = clamp(round(x * scale), -128, 127), torch.round = half-to-EVEN
//   2. the int8 bit-plane kernels (GPU) vs the CPU oracle on the same q
//      (bit-exact int32, the existing gate)
//   3. the epilogue y[r][c] = acc[r][c] * gamma_W[c] / s_x[r]  (float)
// The epilogue is ONE shared function applied to both accumulator paths, so
// the float outputs compare with EXACT equality (identical ops, identical
// order). Quantization is host-side in both paths; a GPU quantize+pack
// kernel does not exist in the PoC.
constexpr float kQuantScale = 127.0F; // BitNet b1.58 absmax activation scale
constexpr float kQuantFloor = 1e-5F;  // reference clamp(min=1e-5) on max|x|

static int8_t quantize_activation(float x, float s_x) {
    // std::nearbyint under the default FE_TONEAREST mode rounds halves to
    // even, like torch.round; std::round would round them away from zero.
    const float q = std::nearbyint(x * s_x);
    return static_cast<int8_t>(std::clamp(q, -128.0F, 127.0F));
}

// Shared epilogue (the ONLY float step): y[r][c] = acc[r][c] * gamma[c] / s_x[r].
static void bitlinear_epilogue(const std::vector<int32_t>& acc, const std::vector<float>& gamma,
                               const std::vector<float>& s_x, uint32_t m, uint32_t n,
                               std::vector<float>& y) {
    y.resize(static_cast<size_t>(m) * n);
    for (uint32_t r = 0; r < m; ++r) {
        for (uint32_t c = 0; c < n; ++c) {
            y[static_cast<size_t>(r) * n + c] = static_cast<float>(
                static_cast<float>(acc[static_cast<size_t>(r) * n + c]) * gamma[c] / s_x[r]);
        }
    }
}

int run_bitlinear_test(const bitnet::LoadedWeights& loaded) {
    const uint32_t m = 8;
    std::printf("[bitlinear] m=%u n=%u k=%u float->int8 quantize -> kernel -> epilogue "
                "(float-exact)\n",
                m, loaded.n, loaded.k);
    try {
        // Deterministic float input in [-1, 1] (integer-derived, IEEE divides
        // only - identical on every platform/toolchain).
        std::vector<float> x(static_cast<size_t>(m) * loaded.k);
        for (uint32_t r = 0; r < m; ++r) {
            for (uint32_t j = 0; j < loaded.k; ++j) {
                const auto raw = static_cast<float>((r * 131U + j * 17U) % 401U);
                x[static_cast<size_t>(r) * loaded.k + j] = (raw - 200.0F) / 200.0F;
            }
        }
        // Step 1: per-row absmax quantization.
        std::vector<float> s_x(m, 0.0F);
        std::vector<int8_t> q(static_cast<size_t>(m) * loaded.k);
        for (uint32_t r = 0; r < m; ++r) {
            float max_abs = 0.0F;
            for (uint32_t j = 0; j < loaded.k; ++j) {
                max_abs = std::max(max_abs, std::fabs(x[static_cast<size_t>(r) * loaded.k + j]));
            }
            s_x[r] = kQuantScale / std::max(max_abs, kQuantFloor);
            for (uint32_t j = 0; j < loaded.k; ++j) {
                q[static_cast<size_t>(r) * loaded.k + j] =
                    quantize_activation(x[static_cast<size_t>(r) * loaded.k + j], s_x[r]);
            }
        }
        // Step 2: every core kernel vs the CPU oracle on the quantized input.
        const Shape s{.m = m, .n = loaded.n, .k = loaded.k, .binary = false};
        bitnet::TernaryTensor t;
        t.m = m;
        t.n = loaded.n;
        t.k = loaded.k;
        t.binary = false;
        t.weights = loaded.weights;
        t.gamma = loaded.gamma;
        t.activations = q;
        std::vector<int32_t> gpu_acc;
        const int64_t mm = compare_tensor(s, t, gpu_acc);
        if (mm != 0) {
            std::printf("  FAIL: %lld accumulator mismatches\n", static_cast<long long>(mm));
            return 1;
        }
        // CPU reference accumulators: same operations as the oracle.
        std::vector<int32_t> cpu_acc(static_cast<size_t>(m) * loaded.n, 0);
        bitnet::matmul_ref(loaded.weights.data(), q.data(), cpu_acc.data(), m, loaded.n, loaded.k);
        // Step 3: ONE epilogue applied to both accumulator paths.
        std::vector<float> y_gpu;
        std::vector<float> y_cpu;
        bitlinear_epilogue(gpu_acc, loaded.gamma, s_x, m, loaded.n, y_gpu);
        bitlinear_epilogue(cpu_acc, loaded.gamma, s_x, m, loaded.n, y_cpu);
        if (y_gpu != y_cpu) {
            size_t bad = 0;
            for (size_t i = 0; i < y_gpu.size(); ++i) {
                if (y_gpu[i] != y_cpu[i]) {
                    ++bad;
                }
            }
            std::printf("  FAIL: %zu float outputs differ (must be exactly equal)\n", bad);
            return 1;
        }
        // Sample record: input, quantized input, output, expectation.
        std::printf("  sample x[0][0..3]:     %.4f %.4f %.4f %.4f\n", x[0], x[1], x[2], x[3]);
        std::printf("  sample q[0][0..3]:     %d %d %d %d (s_x[0]=%.4f)\n", q[0], q[1], q[2], q[3],
                    s_x[0]);
        std::printf("  sample y[0][0..3]:     %.4f %.4f %.4f %.4f (gamma[0]=%.6f)\n", y_gpu[0],
                    y_gpu[1], y_gpu[2], y_gpu[3], loaded.gamma[0]);
        std::printf("  PASS (all kernels bit-exact on q; %zu float outputs exactly equal)\n",
                    y_gpu.size());
        return 0;
    } catch (const std::exception& e) {
        std::printf("  FAIL: %s\n", e.what());
        return 1;
    }
}