// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Contiguous-stream bandwidth probe: variant of bw_probe
// where each thread sweeps its OWN contiguous run of uint4 words (start =
// tid * run), instead of striding by the thread count. At the warp level
// both patterns coalesce; per thread, this variant additionally keeps every
// next word in the same cache line - the strict sequential-stream pattern.
// Compare against bw_probe (strided) in the same run to bound the ceiling.
// Every word is accumulated so nothing optimizes away.

// Same 16-byte uniform as ::bitnet::KernelParams in poc/common/format.hpp;
// fields reinterpreted by the probe host (kwords, n as above).
struct KernelParams {
    uint m;
    uint n;
    uint kwords;
    uint mode;
};

[vk::binding(0, 0)] StructuredBuffer<uint4> Src;
[vk::binding(1, 0)] RWStructuredBuffer<int> Sink;
[vk::binding(3, 0)] ConstantBuffer<KernelParams>
    P; // reuse struct: kwords = number of uint4 elements, n = total thread count

[shader("compute")][numthreads(256, 1, 1)] void stream_bw_probe(uint3 tid : SV_DispatchThreadID) {
    const uint run = P.kwords / P.n; // uint4 elements per thread (contiguous)
    const uint start = tid.x * run;
    int s = 0;
    for (uint i = 0U; i < run; ++i) {
        const uint4 v = Src[start + i];
        s += int(v.x ^ v.y ^ v.z ^ v.w);
    }
    Sink[tid.x] = s;
}
