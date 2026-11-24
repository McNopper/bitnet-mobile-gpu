// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Vector-load isolation probe. Settles whether
// uint4-typed storage loads compile correctly on a given device, independent
// of kernel structure:
//
//   mode 0 (vector): Out[u] = xor of the four components of Src4[u]
//   mode 1 (scalar): Out[u] = xor of the four words of Src1[4u..4u+3]
//
// The host fills Src with a known pattern and compares both outputs against
// the same XOR computed on the CPU. A driver that miscompiles either access
// path produces mismatches here. Bindings: b0 Src as uint4 view, b1 Src as
// uint view (the SAME buffer bound twice - both views cover identical
// bytes), b2 Out, b3 uniform.

struct KernelParams
{
    uint m;      // units to process
    uint n;      // total threads
    uint kwords; // words per unit (4)
    uint mode;   // 0 = vector load, 1 = scalar load
};

[vk::binding(0, 0)] StructuredBuffer<uint4> Src4;
[vk::binding(1, 0)] StructuredBuffer<uint> Src1;
[vk::binding(2, 0)] RWStructuredBuffer<uint> Out;
[vk::binding(3, 0)] ConstantBuffer<KernelParams> P;

[shader("compute")]
[numthreads(256, 1, 1)]
void vec_load_probe(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= P.m)
    {
        return;
    }
    uint r = 0U;
    if (P.mode == 0U)
    {
        const uint4 v = Src4[tid.x];
        r = v.x ^ v.y ^ v.z ^ v.w;
    }
    else
    {
        const uint b = tid.x * 4U;
        r = Src1[b + 0U] ^ Src1[b + 1U] ^ Src1[b + 2U] ^ Src1[b + 3U];
    }
    Out[tid.x] = r;
}
