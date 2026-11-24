// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// FP32-FMA formulation probe: measures the alternative arithmetic of the
// ternary matmul (one FMA per weight, acc += w * a with w in {-1,0,+1}).
// Like alu_countbits_probe it reads NOTHING from memory - inputs are computed
// in registers - so it isolates the arithmetic rate of the FMA pipeline from
// the AND + popcount integer pipeline on the same device.
//
// Exactness of the formulation (not of this throughput probe): int8
// activations are exactly representable in FP32, and a matmul output sum is
// bounded by K*127 < 2^24 for every layer shape, so an FP32 accumulator is
// exact for the ternary matmul. The probe itself only measures the rate.
//
// Uniform reuse: kwords = loop iterations per thread, n = total threads.
// Each thread writes one int to Sink (b1), matching the shared layout.

struct KernelParams {
    uint m;
    uint n;
    uint kwords;
    uint mode;
};

[vk::binding(1, 0)] RWStructuredBuffer<int> Sink;
[vk::binding(3, 0)] ConstantBuffer<KernelParams> P;

[shader("compute")][numthreads(256, 1, 1)] void
alu_fma_probe(uint3 tid : SV_DispatchThreadID) {
    // A seed that evolves per weight: every FMA has different operands, so
    // the compiler cannot fold the products of an iteration (or of the whole
    // loop) into one constant. This mirrors the countbits probe's anti-fold.
    uint seed = tid.x * 2654435761U + 1U;
    float acc = 0.0f;
    for (uint i = 0U; i < P.kwords; ++i) {
        for (uint j = 0U; j < 32U; ++j) {
            seed = (seed << 1) | (seed >> 31);
            const float w = (seed & 1U) != 0U ? 1.0f : -1.0f; // exact +/-1
            const float a = float(int(seed & 127U) - 64);      // exact int8
            acc += w * a;
        }
    }
    Sink[tid.x] = int(acc);
}
