// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// BitNet PoC memory-bandwidth probe (Slang -> SPIR-V).
//
// Streams Src (uint4 words) with a grid-stride loop and reduces to a single
// int per thread, so the kernel is DRAM-bound. Uniform reuse: kwords =
// number of uint4 elements to read, n = total thread count (= grid-stride
// stride). Each thread writes exactly one int; the host allocates Sink with
// totalThreads ints.
//
// Binding table matches the shared descriptor layout (poc/backend/
// backend.hpp): b0 Src, b1 Sink (read-write), b3 uniform. Binding 2 is
// OutAcc in the gemv/gemm table and stays statically unused here.

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

[shader("compute")][numthreads(256, 1, 1)] void bw_probe(uint3 tid : SV_DispatchThreadID) {
    int s = 0;
    for (uint i = tid.x; i < P.kwords; i += P.n)
        s += Src[i].x + Src[i].y + Src[i].z + Src[i].w;
    Sink[tid.x] = s;
}
