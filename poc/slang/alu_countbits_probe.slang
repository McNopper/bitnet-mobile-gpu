// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// ALU decomposition probe: reproduces the EXACT inner math of the bit-plane
// kernels (8 planes x [2 AND + 2 countbits + shift-scale-add] + rowsum
// popcounts) but reads NOTHING from memory - all inputs are computed in
// registers. Comparing this against bw_probe (pure memory) and the real
// gemv decomposes kernel time into ALU vs memory components on any device.
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
alu_countbits_probe(uint3 tid : SV_DispatchThreadID) {
    // Deterministic register-resident "weights": no memory traffic.
    uint x = tid.x * 2654435761U + 1U;
    uint w0 = x;
    uint w1 = x ^ 0xA5A5A5A5U;
    int s = 0;
    for (uint i = 0; i < P.kwords; ++i) {
        // Rotate inputs each iteration so the compiler cannot hoist the
        // whole loop into a constant fold.
        w0 = (w0 << 1) | (w0 >> 31);
        w1 = (w1 << 1) | (w1 >> 31);
        for (int p = 0; p < 8; ++p) {
            uint mask = (1U << p) * 0x11111111U;
            s += (1 << p) * (int(countbits(w0 & mask)) - int(countbits(w1 & mask)));
        }
        s -= 128 * (int(countbits(w0)) - int(countbits(w1)));
    }
    Sink[tid.x] = s;
}
