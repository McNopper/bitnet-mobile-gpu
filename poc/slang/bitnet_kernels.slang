// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// ONE Slang source, MANY SPIR-V variants:
// every matmul kernel this project ships lives here. CMake compiles one
// entry point per variant with per-profile -D knobs; slangc
// strips every entry point and groupshared allocation this invocation
// does not reference, so one file costs nothing at runtime. Names are
// mechanism-descriptive ({operation}_{family}_{assignment}) - poc/backend/
// kernel_family.hpp is the single place that classifies them back from
// a string, so the backend/harness never re-derives family membership
// ad hoc. The MACHINE-CHECKED choice of WHICH variant runs on a given
// device is made at runtime by the caps probe + caps-conformance gate +
// variant_selection.hpp (capability flags only, never device names) -
// this file supplies every candidate; it never decides between them.
//
// Families:
//   bitplane - AND+popcount over 2/8 bit-planes (format v1). The
//              always-portable base: no capability needed, the Vulkan
//              1.0 floor (degrade, don't skip).
//   dot      - packed integer dot product (format v2); hardware
//              dot4add_i8packed (needs shaderIntegerDotProduct, core
//              1.3 or VK_KHR_shader_integer_dot_product) or the
//              bit-exact -DNO_INT_DOT emulated fallback (always runs).
//   plane    - plane-uint4 staging (format v3); microtile
//              is the register-tile sub-strategy (experimental), its
//              _image variant streams weights through a 2D sampled
//              image instead of a storage buffer (also experimental).
//   coopmat  - VK_KHR_cooperative_matrix hardware MMA, staged through
//              groupshared memory (the only known-good int8 pattern).
//              Capability-gated, opt-in experimental:
//              the shipped kernel is architecturally correct and
//              waiting for a slangc fix (the installed
//              toolchain cannot wire CoopMat.Load results into
//              coopMatMulAdd operands for int8 in any form; the driver
//              itself is proven correct by a GLSL twin of the kernel that
//              is bit-exact on the same driver).
//
// Integer partials are linear and addition is associative in every
// family, so every work decomposition below is bit-exact against the
// SAME CPU oracle (poc/common/reference.cpp) - that is the project's
// entire correctness argument, independent of which variant a device
// picks.

struct KernelParams {
    uint m;
    uint n;
    uint kwords; // meaning is per-family: bitplane/dot words, plane units, coopmat padded K
    uint mode;   // per-family: v2-style K-slice, coopmat padded M, microtile_image width; else unused
};

// ===========================================================================
// Tunable knobs: slangc -D overrides, one #ifndef block per knob.
// CMake passes ONE shared -D list per GPU profile to every family in this
// file (CMakeLists.txt bitnet_compile_kernels_for); a knob unused by the
// entry point being compiled is a dead define, never a conflict.
// ===========================================================================

#ifndef GEMV_BITPLANE_ROWCOOP_THREADS
#define GEMV_BITPLANE_ROWCOOP_THREADS 128
#endif
#ifndef GEMM_BITPLANE_CELLCOOP_THREADS
#define GEMM_BITPLANE_CELLCOOP_THREADS 128
#endif
#ifndef GEMV_DOT_ROWCOOP_THREADS
#define GEMV_DOT_ROWCOOP_THREADS 64
#endif
#ifndef GEMV_PLANE_ROWCOOP_THREADS
#define GEMV_PLANE_ROWCOOP_THREADS 64
#endif
#ifndef ROWS_PER_GROUP
#define ROWS_PER_GROUP 1
#endif
// Shared tiled-GEMM geometry (gemm_bitplane_tiled / gemm_dot_tiled /
// gemm_plane_tiled): the SAME -D values reach all three families per
// profile today (CMakeLists.txt _defs is one list) - merging into one
// file makes that sharing explicit instead of implicit-via-identical-args.
#ifndef TILE_M
#define TILE_M 4
#endif
#ifndef TILE_N
#define TILE_N 32
#endif
#ifndef CHUNK
#define CHUNK 8 // bitplane/dot: words per K-chunk; plane: CHUNK/2 units per chunk
#endif
#ifndef TILE_THREADS
#define TILE_THREADS 128
#endif
#ifndef GEMM_PLANE_MICROTILE_RM
#define GEMM_PLANE_MICROTILE_RM 4
#endif
#ifndef GEMM_PLANE_MICROTILE_RN
#define GEMM_PLANE_MICROTILE_RN 2
#endif
#ifndef GEMM_PLANE_MICROTILE_THREADS
#define GEMM_PLANE_MICROTILE_THREADS 64
#endif
#ifndef GEMM_COOPMAT_THREADS
#define GEMM_COOPMAT_THREADS 32
#endif
#ifndef GEMM_COOPMAT_SUBGROUPS
#define GEMM_COOPMAT_SUBGROUPS 1 // one 32-lane subgroup per workgroup (tutorial geometry)
#endif
#ifndef GEMM_COOPMAT_M
#define GEMM_COOPMAT_M 16
#endif
#ifndef GEMM_COOPMAT_N
#define GEMM_COOPMAT_N 16
#endif
#ifndef GEMM_COOPMAT_K
#define GEMM_COOPMAT_K 32
#endif

#define PLANE_CU (CHUNK / 2)

// ===========================================================================
// Resources shared by EVERY family (identical type, binding and name in
// every source this file replaces - deduplicated here to exactly one
// declaration).
// ===========================================================================

[vk::binding(2, 0)] RWStructuredBuffer<int> OutAcc;
[vk::binding(3, 0)] ConstantBuffer<KernelParams> P;

// ===========================================================================
// Bitplane family: AND+popcount over 2/8 bit-planes (format v1). Shared
// helper + the strategy-interface demonstration, then one
// entry point per assignment strategy (gemv: rowserial/ksplit/rowcoop;
// gemm: naive/ksplit/cellcoop/tiled).
// ===========================================================================

[vk::binding(0, 0)] StructuredBuffer<uint> WeightBits;
[vk::binding(1, 0)] StructuredBuffer<uint> ActBits;

// Per word w of weight row rowN (both planes), accumulate
// 2^p * (popcount(w0 & act_p) - popcount(w1 & act_p)) and the row-sum
// correction into the caller's accumulators (the correction applied once
// at the end; identical integer-exact result in every assignment below).
static void planeWordAccumulate(uint wBase, uint aBase, uint w, inout int accU, inout int rowSum) {
    const uint w0 = WeightBits[wBase + w];
    const uint w1 = WeightBits[wBase + P.kwords + w];
    for (int p = 0; p < 8; ++p) {
        const uint act = ActBits[aBase + uint(p) * P.kwords + w];
        accU += (1 << p) * (int(countbits(w0 & act)) - int(countbits(w1 & act)));
    }
    rowSum += int(countbits(w0)) - int(countbits(w1));
}

interface IBitMatMulStrategy {
    void accumulate(int rowM, int rowN, int wordBegin, int wordEnd, inout int acc);
}

struct BitplanePopcountStrategy : IBitMatMulStrategy {
    // Single pass over the full row (rowserial/naive call (0, kwords) once).
    void accumulate(int rowM, int rowN, int wordBegin, int wordEnd, inout int acc) {
        int accU = 0;
        int rowSum = 0;
        const uint wBase = uint(rowN) * (2u * P.kwords);
        const uint aBase = uint(rowM) * (8u * P.kwords);
        for (int w = wordBegin; w < wordEnd; ++w) {
            planeWordAccumulate(wBase, aBase, uint(w), accU, rowSum);
        }
        acc += accU - 128 * rowSum;
    }
}

static int dotBitplaneUnit<S : IBitMatMulStrategy>(int rowM, int rowN) {
    int acc = 0;
    S strategy;
    strategy.accumulate(rowM, rowN, 0, int(P.kwords), acc);
    return acc;
}

// ---- gemv_bitplane_rowserial: one thread per output row, full-K serial walk ----

[shader("compute")][numthreads(64, 1, 1)] void
gemv_bitplane_rowserial(uint3 tid : SV_DispatchThreadID) {
    uint n = tid.x;
    if (n < P.n)
        OutAcc[n] = dotBitplaneUnit<BitplanePopcountStrategy>(0, int(n));
}

// ---- gemv_bitplane_ksplit: K-split slices + InterlockedAdd ----
// (the host MUST zero the output buffer first: accumulation contract.
// The grid stays small for very wide outputs; the interleaved-16 variant
// below is the Adreno 740 decode champion.)

static int dotRowSlice(int rowN, int begin, int end) {
    int accU = 0;
    int rowSum = 0;
    uint wBase = uint(rowN) * (2u * P.kwords);
    for (int w = begin; w < end; ++w) {
        planeWordAccumulate(wBase, 0u, uint(w), accU, rowSum);
    }
    return accU - 128 * rowSum;
}

[shader("compute")][numthreads(256, 1, 1)] void
gemv_bitplane_ksplit(uint3 tid : SV_DispatchThreadID) {
    uint n = tid.x;
    if (n >= P.n)
        return;
    int slice = int(tid.z);
    int begin = slice * int(P.mode);
    int end = min(begin + int(P.mode), int(P.kwords));
    if (begin >= end)
        return;
    int partial = dotRowSlice(int(n), begin, end);
    InterlockedAdd(OutAcc[n], partial);
}

// ---- gemv_bitplane_ksplit_il: K-split over the INTERLEAVED-16 layout ----
// Same math and InterlockedAdd accumulation contract as gemv_bitplane_ksplit,
// but weights are read in the interleaved-16 layout (format.hpp
// repack_interleaved16): for each word index w, the 16 rows of a block read
// 16 CONTIGUOUS words per plane, instead of 16 rows' strided planes. On
// Adreno, where each thread owns a different output row, this is the
// coalesced form of the same dispatch.

static void planeWordAccumulateIL(uint wBase, uint aBase, uint w, inout int accU, inout int rowSum) {
    const uint w0 = WeightBits[wBase + w * 16U];
    const uint w1 = WeightBits[wBase + P.kwords * 16U + w * 16U];
    for (int p = 0; p < 8; ++p) {
        const uint act = ActBits[aBase + uint(p) * P.kwords + w];
        accU += (1 << p) * (int(countbits(w0 & act)) - int(countbits(w1 & act)));
    }
    rowSum += int(countbits(w0)) - int(countbits(w1));
}

[shader("compute")][numthreads(256, 1, 1)] void
gemv_bitplane_ksplit_il(uint3 tid : SV_DispatchThreadID) {
    const uint n = tid.x;
    if (n >= P.n) {
        return;
    }
    const uint block = n / 16U;
    const uint r = n % 16U;
    const uint wBase = block * (2U * P.kwords * 16U) + r;
    const int slice = int(tid.z);
    const int begin = slice * int(P.mode);
    const int end = min(begin + int(P.mode), int(P.kwords));
    if (begin >= end) {
        return;
    }
    int accU = 0;
    int rowSum = 0;
    for (int w = begin; w < end; ++w) {
        planeWordAccumulateIL(wBase, 0U, uint(w), accU, rowSum);
    }
    InterlockedAdd(OutAcc[n], accU - 128 * rowSum);
}

// ---- gemv_bitplane_rowcoop: cooperative rows, one workgroup per row ----
// (thread t walks words t, t+T, t+2T for cache-line utilization; partials
// combine in groupshared memory - no atomics, no zero-init contract).

groupshared int g_gemv_bitplane_rowcoop_partial[GEMV_BITPLANE_ROWCOOP_THREADS];

[shader("compute")][numthreads(GEMV_BITPLANE_ROWCOOP_THREADS, 1, 1)] void
gemv_bitplane_rowcoop(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint n = gid.x;
    const uint t = gtid.x;
    const uint wBase = n * (2u * P.kwords);
    const uint aBase = 0u; // gemv: activation row 0

    int partial = 0;
    for (uint w = t; w < P.kwords; w += GEMV_BITPLANE_ROWCOOP_THREADS) {
        int accU = 0;
        int rowSum = 0;
        planeWordAccumulate(wBase, aBase, w, accU, rowSum);
        partial += accU - 128 * rowSum;
    }

    g_gemv_bitplane_rowcoop_partial[t] = partial;
    GroupMemoryBarrierWithGroupSync();

    if (t == 0u) {
        int total = 0;
        for (uint i = 0u; i < GEMV_BITPLANE_ROWCOOP_THREADS; ++i) {
            total += g_gemv_bitplane_rowcoop_partial[i];
        }
        OutAcc[n] = total;
    }
}

// ---- gemm_bitplane_naive: one thread per output, full-K serial walk ----

[shader("compute")][numthreads(16, 16, 1)] void
gemm_bitplane_naive(uint3 tid : SV_DispatchThreadID) {
    const uint n = tid.x;
    const uint m = tid.y;
    if (n < P.n && m < P.m)
        OutAcc[m * P.n + n] = dotBitplaneUnit<BitplanePopcountStrategy>(int(m), int(n));
}

// ---- gemm_bitplane_ksplit: K-split slices + InterlockedAdd (zero-init contract) ----

static int dotCellSlice(int rowM, int rowN, int begin, int end) {
    int accU = 0;
    int rowSum = 0;
    const uint wBase = uint(rowN) * (2u * P.kwords);
    const uint aBase = uint(rowM) * (8u * P.kwords);
    for (int w = begin; w < end; ++w) {
        planeWordAccumulate(wBase, aBase, uint(w), accU, rowSum);
    }
    return accU - 128 * rowSum;
}

[shader("compute")][numthreads(64, 1, 1)] void
gemm_bitplane_ksplit(uint3 tid : SV_DispatchThreadID) {
    const uint n = tid.x;
    const uint m = tid.y;
    if (n >= P.n || m >= P.m)
        return;
    const int slice = int(tid.z);
    const int begin = slice * int(P.mode);
    const int end = min(begin + int(P.mode), int(P.kwords));
    if (begin >= end)
        return;
    const int partial = dotCellSlice(int(m), int(n), begin, end);
    InterlockedAdd(OutAcc[m * P.n + n], partial);
}

// ---- gemm_bitplane_cellcoop: one workgroup per output CELL, cooperative K-walk ----

groupshared int g_gemm_bitplane_cellcoop_partial[GEMM_BITPLANE_CELLCOOP_THREADS];

[shader("compute")][numthreads(GEMM_BITPLANE_CELLCOOP_THREADS, 1, 1)] void
gemm_bitplane_cellcoop(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint cell = gid.x;
    const uint n = cell % P.n;
    const uint m = cell / P.n;
    const uint t = gtid.x;
    const uint wBase = n * (2u * P.kwords);
    const uint aBase = m * (8u * P.kwords);

    int partial = 0;
    for (uint w = t; w < P.kwords; w += GEMM_BITPLANE_CELLCOOP_THREADS) {
        int accU = 0;
        int rowSum = 0;
        planeWordAccumulate(wBase, aBase, w, accU, rowSum);
        partial += accU - 128 * rowSum;
    }

    g_gemm_bitplane_cellcoop_partial[t] = partial;
    GroupMemoryBarrierWithGroupSync();

    if (t == 0u) {
        int total = 0;
        for (uint i = 0u; i < GEMM_BITPLANE_CELLCOOP_THREADS; ++i) {
            total += g_gemm_bitplane_cellcoop_partial[i];
        }
        OutAcc[cell] = total;
    }
}

// ---- gemm_bitplane_tiled: TILE_M x TILE_N blocking, K-chunk groupshared staging ----

groupshared uint s_gemm_bitplane_tiled_w[TILE_N * 2 * CHUNK]; // [n-row][plane][chunk-word]
groupshared uint s_gemm_bitplane_tiled_a[TILE_M * 8 * CHUNK]; // [m-row][plane][chunk-word]

[shader("compute")][numthreads(TILE_THREADS, 1, 1)] void
gemm_bitplane_tiled(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint n0 = gid.x * TILE_N;
    const uint m0 = gid.y * TILE_M;
    const uint t = gtid.x;

    int accU = 0;
    int rowSum = 0;

    for (uint g0 = 0; g0 < P.kwords; g0 += CHUNK) {
        for (uint i = t; i < TILE_N * 2 * CHUNK; i += TILE_THREADS) {
            const uint rn = i / (2 * CHUNK);
            const uint rem = i - rn * (2 * CHUNK);
            const uint plane = rem / CHUNK;
            const uint w = rem - plane * CHUNK;
            const uint gRow = n0 + rn;
            const uint gWord = g0 + w;
            s_gemm_bitplane_tiled_w[i] =
                (gRow < P.n && gWord < P.kwords)
                    ? WeightBits[gRow * (2u * P.kwords) + plane * P.kwords + gWord]
                    : 0u;
        }
        for (uint i = t; i < TILE_M * 8 * CHUNK; i += TILE_THREADS) {
            const uint rm = i / (8 * CHUNK);
            const uint rem = i - rm * (8 * CHUNK);
            const uint plane = rem / CHUNK;
            const uint w = rem - plane * CHUNK;
            const uint aRow = m0 + rm;
            const uint aWord = g0 + w;
            s_gemm_bitplane_tiled_a[i] =
                (aRow < P.m && aWord < P.kwords)
                    ? ActBits[aRow * (8u * P.kwords) + plane * P.kwords + aWord]
                    : 0u;
        }
        GroupMemoryBarrierWithGroupSync();

        const uint mm = t / TILE_N;
        const uint nn = t - mm * TILE_N;
        if (m0 + mm < P.m && n0 + nn < P.n) {
            const uint wBase = nn * (2 * CHUNK);
            const uint aBase = mm * (8 * CHUNK);
            for (uint w = 0; w < CHUNK; ++w) {
                const uint ww0 = s_gemm_bitplane_tiled_w[wBase + w];
                const uint ww1 = s_gemm_bitplane_tiled_w[wBase + CHUNK + w];
                for (int p = 0; p < 8; ++p) {
                    const uint act = s_gemm_bitplane_tiled_a[aBase + uint(p) * CHUNK + w];
                    accU += (1 << p) * (int(countbits(ww0 & act)) - int(countbits(ww1 & act)));
                }
                rowSum += int(countbits(ww0)) - int(countbits(ww1));
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const uint mm = t / TILE_N;
    const uint nn = t - mm * TILE_N;
    if (m0 + mm < P.m && n0 + nn < P.n) {
        const int acc = accU - 128 * rowSum;
        OutAcc[(m0 + mm) * P.n + (n0 + nn)] = acc;
    }
}

// ===========================================================================
// Dot family: packed integer dot product (format v2). dot4() is the
// shared accumulate primitive: dot4add_i8packed on devices that advertise
// shaderIntegerDotProduct, or the SAME-bit-exact own shift/mask/extract
// multiply-accumulate when compiled with -DNO_INT_DOT (the integer
// path always runs, slow but correct without the capability). Selection is capability-driven (variant_selection.hpp),
// never by device name.
// ===========================================================================

[vk::binding(0, 0)] StructuredBuffer<uint> WeightCodes; // [n][kwords] word 0 of each unit = weights 4w..4w+3
[vk::binding(1, 0)] StructuredBuffer<uint> ActWords;    // [m][kwords]

int dot4(uint x, uint y, int acc) {
#ifdef NO_INT_DOT
    int s = acc;
    [unroll] for (uint lane = 0u; lane < 4u; ++lane) {
        const uint shift = lane * 8u;
        const int xv = int(((x >> shift) & 0xFFu) ^ 0x80u) - 128;
        const int yv = int(((y >> shift) & 0xFFu) ^ 0x80u) - 128;
        s += xv * yv;
    }
    return s;
#else
    return dot4add_i8packed(x, y, acc);
#endif
}

// ---- gemv_dot_rowcoop: decode (m = 1), one workgroup per output row ----
// (mirrors gemv_bitplane_rowcoop/gemv_plane_rowcoop's cooperative-row
// contract; two independent accumulator chains for ILP - integer addition
// stays associative, so this is still bit-exact).

groupshared int g_gemv_dot_rowcoop_partial[GEMV_DOT_ROWCOOP_THREADS];

[shader("compute")][numthreads(GEMV_DOT_ROWCOOP_THREADS, 1, 1)] void
gemv_dot_rowcoop(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint n = gid.x;
    const uint t = gtid.x;
    // The row stride comes from the HOST via KernelParams.kwords (a variant
    // that derived it in the shader produced shifted rows on Mali-G57); the
    // host is authoritative, plain per-word index arithmetic is correct
    // everywhere.
    const uint wBase = n * P.kwords;

    int acc0 = 0;
    int acc1 = 0;
    uint w = t;
    for (; w + GEMV_DOT_ROWCOOP_THREADS < P.kwords; w += 2U * GEMV_DOT_ROWCOOP_THREADS) {
        acc0 = dot4(WeightCodes[wBase + w], ActWords[w], acc0);
        acc1 = dot4(WeightCodes[wBase + w + GEMV_DOT_ROWCOOP_THREADS],
                    ActWords[w + GEMV_DOT_ROWCOOP_THREADS], acc1);
    }
    for (; w < P.kwords; w += GEMV_DOT_ROWCOOP_THREADS) {
        acc0 = dot4(WeightCodes[wBase + w], ActWords[w], acc0);
    }

    g_gemv_dot_rowcoop_partial[t] = acc0 + acc1;
    GroupMemoryBarrierWithGroupSync();

    if (t == 0u) {
        int total = 0;
        for (uint i = 0u; i < GEMV_DOT_ROWCOOP_THREADS; ++i) {
            total += g_gemv_dot_rowcoop_partial[i];
        }
        OutAcc[n] = total; // int8 values dot directly - no correction term
    }
}

// ---- gemm_dot_tiled: TILE_M x TILE_N blocking, K-chunk groupshared staging ----
// (scalar word addressing: a uint4 vector-load variant of this kernel was
// bit-exact on the desktop but gave wrong results on Mali-G57 for
// stride-padded shapes; the cause was not isolated and the variant is not
// shipped. The plane family uses uint4 loads and is bit-exact there.)

groupshared uint s_gemm_dot_tiled_w[TILE_N * CHUNK]; // [n-row][chunk-word]
groupshared uint s_gemm_dot_tiled_a[TILE_M * CHUNK]; // [m-row][chunk-word]

[shader("compute")][numthreads(TILE_THREADS, 1, 1)] void
gemm_dot_tiled(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint n0 = gid.x * TILE_N;
    const uint m0 = gid.y * TILE_M;
    const uint t = gtid.x;

    int acc0 = 0;
    int acc1 = 0;

    for (uint g0 = 0; g0 < P.kwords; g0 += CHUNK) {
        for (uint i = t; i < TILE_N * CHUNK; i += TILE_THREADS) {
            const uint rn = i / CHUNK;
            const uint w = i - rn * CHUNK;
            const uint gRow = n0 + rn;
            const uint gWord = g0 + w;
            s_gemm_dot_tiled_w[i] = (gRow < P.n && gWord < P.kwords)
                                        ? WeightCodes[gRow * P.kwords + gWord]
                                        : 0u;
        }
        for (uint i = t; i < TILE_M * CHUNK; i += TILE_THREADS) {
            const uint rm = i / CHUNK;
            const uint w = i - rm * CHUNK;
            const uint gRow = m0 + rm;
            const uint gWord = g0 + w;
            s_gemm_dot_tiled_a[i] =
                (gRow < P.m && gWord < P.kwords) ? ActWords[gRow * P.kwords + gWord] : 0u;
        }
        GroupMemoryBarrierWithGroupSync();

        const uint mm = t / TILE_N;
        const uint nn = t - mm * TILE_N;
        if (m0 + mm < P.m && n0 + nn < P.n) {
            const uint wBase = nn * CHUNK;
            const uint aBase = mm * CHUNK;
            for (uint w = 0; w < CHUNK; w += 2U) {
                acc0 = dot4(s_gemm_dot_tiled_w[wBase + w], s_gemm_dot_tiled_a[aBase + w], acc0);
                acc1 = dot4(s_gemm_dot_tiled_w[wBase + w + 1U], s_gemm_dot_tiled_a[aBase + w + 1U],
                            acc1);
            }
        }
        GroupMemoryBarrierWithGroupSync(); // s_w/s_a rewritten next chunk
    }

    const uint mm = t / TILE_N;
    const uint nn = t - mm * TILE_N;
    if (m0 + mm < P.m && n0 + nn < P.n) {
        OutAcc[(m0 + mm) * P.n + (n0 + nn)] = acc0 + acc1;
    }
}

// ===========================================================================
// Plane family: plane-uint4 staging (format v3). Each weight
// uint4 is {plus_lo, plus_hi, minus_lo, minus_hi}; activation units carry
// the eight bit-planes on the same 64-column grid.
// ===========================================================================

[vk::binding(0, 0)] StructuredBuffer<uint4> WeightUnits;
[vk::binding(1, 0)] StructuredBuffer<uint4> ActUnits;
[vk::binding(5, 0)] Texture2D<uint4> WeightImg;

// ---- gemv_plane_rowcoop: cooperative rows over plane-uint4 units ----

groupshared int g_gemv_plane_rowcoop_partial[GEMV_PLANE_ROWCOOP_THREADS];

[shader("compute")][numthreads(GEMV_PLANE_ROWCOOP_THREADS, 1, 1)] void
gemv_plane_rowcoop(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint lanes = GEMV_PLANE_ROWCOOP_THREADS / ROWS_PER_GROUP;
    const uint t = gtid.x;
    const uint row = gid.x * ROWS_PER_GROUP + t / lanes;
    const uint lane = t % lanes;

    int partial = 0;
    if (row < P.n) {
        const uint wBase = row * P.kwords;
        for (uint u = lane; u < P.kwords; u += lanes) {
            const uint4 wu = WeightUnits[wBase + u];
            const uint4 a0 = ActUnits[u * 4U + 0U];
            const uint4 a1 = ActUnits[u * 4U + 1U];
            const uint4 a2 = ActUnits[u * 4U + 2U];
            const uint4 a3 = ActUnits[u * 4U + 3U];
            partial += (1 << 0) * (int(countbits(wu.x & a0.x)) + int(countbits(wu.y & a2.x)) -
                                   int(countbits(wu.z & a0.x)) - int(countbits(wu.w & a2.x)));
            partial += (1 << 1) * (int(countbits(wu.x & a0.y)) + int(countbits(wu.y & a2.y)) -
                                   int(countbits(wu.z & a0.y)) - int(countbits(wu.w & a2.y)));
            partial += (1 << 2) * (int(countbits(wu.x & a0.z)) + int(countbits(wu.y & a2.z)) -
                                   int(countbits(wu.z & a0.z)) - int(countbits(wu.w & a2.z)));
            partial += (1 << 3) * (int(countbits(wu.x & a0.w)) + int(countbits(wu.y & a2.w)) -
                                   int(countbits(wu.z & a0.w)) - int(countbits(wu.w & a2.w)));
            partial += (1 << 4) * (int(countbits(wu.x & a1.x)) + int(countbits(wu.y & a3.x)) -
                                   int(countbits(wu.z & a1.x)) - int(countbits(wu.w & a3.x)));
            partial += (1 << 5) * (int(countbits(wu.x & a1.y)) + int(countbits(wu.y & a3.y)) -
                                   int(countbits(wu.z & a1.y)) - int(countbits(wu.w & a3.y)));
            partial += (1 << 6) * (int(countbits(wu.x & a1.z)) + int(countbits(wu.y & a3.z)) -
                                   int(countbits(wu.z & a1.z)) - int(countbits(wu.w & a3.z)));
            partial += (1 << 7) * (int(countbits(wu.x & a1.w)) + int(countbits(wu.y & a3.w)) -
                                   int(countbits(wu.z & a1.w)) - int(countbits(wu.w & a3.w)));
            partial -= 128 * (int(countbits(wu.x)) + int(countbits(wu.y)) - int(countbits(wu.z)) -
                              int(countbits(wu.w)));
        }
    }

    g_gemv_plane_rowcoop_partial[t] = partial;
    GroupMemoryBarrierWithGroupSync();

    if (t % lanes == 0U) {
        int total = 0;
        for (uint i = 0U; i < lanes; ++i) {
            total += g_gemv_plane_rowcoop_partial[t - lane + i];
        }
        if (row < P.n) {
            OutAcc[row] = total;
        }
    }
}

// ---- gemm_plane_tiled: shared-tile blocking over plane-uint4 units ----

groupshared uint s_gemm_plane_tiled_w[TILE_N * 4 * PLANE_CU];
groupshared uint s_gemm_plane_tiled_a[TILE_M * 16 * PLANE_CU];

[shader("compute")][numthreads(TILE_THREADS, 1, 1)] void
gemm_plane_tiled(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint n0 = gid.x * TILE_N;
    const uint m0 = gid.y * TILE_M;
    const uint t = gtid.x;

    int accU = 0;
    int rowSum = 0;

    for (uint g0 = 0; g0 < P.kwords; g0 += PLANE_CU) {
        for (uint i = t; i < TILE_N * PLANE_CU; i += TILE_THREADS) {
            const uint rn = i / PLANE_CU;
            const uint u = i - rn * PLANE_CU;
            const uint gRow = n0 + rn;
            const uint gUnit = g0 + u;
            const uint4 wu = (gRow < P.n && gUnit < P.kwords)
                                 ? WeightUnits[gRow * P.kwords + gUnit]
                                 : uint4(0U, 0U, 0U, 0U);
            const uint base = (rn * PLANE_CU + u) * 4U;
            s_gemm_plane_tiled_w[base + 0U] = wu.x;
            s_gemm_plane_tiled_w[base + 1U] = wu.y;
            s_gemm_plane_tiled_w[base + 2U] = wu.z;
            s_gemm_plane_tiled_w[base + 3U] = wu.w;
        }
        for (uint i = t; i < TILE_M * PLANE_CU * 4U; i += TILE_THREADS) {
            const uint rm = i / (PLANE_CU * 4U);
            const uint rem = i - rm * (PLANE_CU * 4U);
            const uint u = rem / 4U;
            const uint q = rem - u * 4U;
            const uint aRow = m0 + rm;
            const uint aIdx = aRow * (4U * P.kwords) + (g0 + u) * 4U + q;
            const uint4 au =
                (aRow < P.m && (g0 + u) < P.kwords) ? ActUnits[aIdx] : uint4(0U, 0U, 0U, 0U);
            const uint base = (rm * PLANE_CU + u) * 16U + q * 4U;
            s_gemm_plane_tiled_a[base + 0U] = au.x;
            s_gemm_plane_tiled_a[base + 1U] = au.y;
            s_gemm_plane_tiled_a[base + 2U] = au.z;
            s_gemm_plane_tiled_a[base + 3U] = au.w;
        }
        GroupMemoryBarrierWithGroupSync();

        const uint mm = t / TILE_N;
        const uint nn = t - mm * TILE_N;
        if (m0 + mm < P.m && n0 + nn < P.n) {
            const uint wBase = nn * (PLANE_CU * 4U);
            const uint aBase = mm * (PLANE_CU * 16U);
            for (uint u = 0; u < PLANE_CU; ++u) {
                const uint pLo = s_gemm_plane_tiled_w[wBase + u * 4U + 0U];
                const uint pHi = s_gemm_plane_tiled_w[wBase + u * 4U + 1U];
                const uint mLo = s_gemm_plane_tiled_w[wBase + u * 4U + 2U];
                const uint mHi = s_gemm_plane_tiled_w[wBase + u * 4U + 3U];
                const uint uA = aBase + u * 16U;
                for (int p = 0; p < 8; ++p) {
                    const uint aLo = s_gemm_plane_tiled_a[uA + uint(p)];
                    const uint aHi = s_gemm_plane_tiled_a[uA + 8U + uint(p)];
                    accU += (1 << p) * (int(countbits(pLo & aLo)) + int(countbits(pHi & aHi)) -
                                        int(countbits(mLo & aLo)) - int(countbits(mHi & aHi)));
                }
                rowSum += int(countbits(pLo)) + int(countbits(pHi)) - int(countbits(mLo)) -
                          int(countbits(mHi));
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const uint mm = t / TILE_N;
    const uint nn = t - mm * TILE_N;
    if (m0 + mm < P.m && n0 + nn < P.n) {
        const int acc = accU - 128 * rowSum;
        OutAcc[(m0 + mm) * P.n + (n0 + nn)] = acc;
    }
}

// ---- gemm_plane_microtile / gemm_plane_microtile_image: register
// micro-tile (experimental); only the weight SOURCE differs between them,
// expressed as a static specialization so the shared
// register discipline and body exist exactly once.

static int2 accum_act_unit(uint w0_plus, uint w0_minus, uint w1_plus, uint w1_minus, uint4 a,
                           uint plane_base) {
    const int c0 =
        (1 << (plane_base + 0U)) * (int(countbits(w0_plus & a.x)) - int(countbits(w0_minus & a.x))) +
        (1 << (plane_base + 1U)) * (int(countbits(w0_plus & a.y)) - int(countbits(w0_minus & a.y))) +
        (1 << (plane_base + 2U)) * (int(countbits(w0_plus & a.z)) - int(countbits(w0_minus & a.z))) +
        (1 << (plane_base + 3U)) * (int(countbits(w0_plus & a.w)) - int(countbits(w0_minus & a.w)));
    const int c1 =
        (1 << (plane_base + 0U)) * (int(countbits(w1_plus & a.x)) - int(countbits(w1_minus & a.x))) +
        (1 << (plane_base + 1U)) * (int(countbits(w1_plus & a.y)) - int(countbits(w1_minus & a.y))) +
        (1 << (plane_base + 2U)) * (int(countbits(w1_plus & a.z)) - int(countbits(w1_minus & a.z))) +
        (1 << (plane_base + 3U)) * (int(countbits(w1_plus & a.w)) - int(countbits(w1_minus & a.w)));
    return int2(c0, c1);
}

static int row_sum_unit(uint4 w) {
    return int(countbits(w.x)) + int(countbits(w.y)) - int(countbits(w.z)) - int(countbits(w.w));
}

interface IPlaneWeightSource {
    uint4 load(uint index);
}

struct BufferPlaneWeightSource : IPlaneWeightSource {
    uint4 load(uint index) {
        return WeightUnits[index];
    }
}

struct ImagePlaneWeightSource : IPlaneWeightSource {
    uint4 load(uint index) {
        return WeightImg.Load(int3(index % P.mode, index / P.mode, 0U));
    }
}

static void gemmPlaneMicrotileBody<S : IPlaneWeightSource>(uint3 gtid) {
    S weightSource;
    const uint tiles_x = (P.n + GEMM_PLANE_MICROTILE_RN - 1U) / GEMM_PLANE_MICROTILE_RN;
    const uint tile = gtid.x;
    const uint n0 = (tile % tiles_x) * GEMM_PLANE_MICROTILE_RN;
    const uint m0 = (tile / tiles_x) * GEMM_PLANE_MICROTILE_RM;
    const bool col1 = n0 + 1U < P.n;

    int acc00 = 0;
    int acc01 = 0;
    int acc10 = 0;
    int acc11 = 0;
    int acc20 = 0;
    int acc21 = 0;
    int acc30 = 0;
    int acc31 = 0;
    int rowSum0 = 0;
    int rowSum1 = 0;

    for (uint u = 0U; u < P.kwords; ++u) {
        const uint4 w0 = weightSource.load((n0 + 0U) * P.kwords + u);
        const uint4 w1 = col1 ? weightSource.load((n0 + 1U) * P.kwords + u)
                              : uint4(0U, 0U, 0U, 0U);

        {
            const uint act_base = (m0 + 0U) * (4U * P.kwords) + u * 4U;
            const bool live = m0 + 0U < P.m;
            const uint4 a = live ? ActUnits[act_base + 0U] : uint4(0U, 0U, 0U, 0U);
            const int2 s = accum_act_unit(w0.x, w0.z, w1.x, w1.z, a, 0U);
            acc00 += s.x;
            acc01 += s.y;
            const uint4 a1 = live ? ActUnits[act_base + 1U] : uint4(0U, 0U, 0U, 0U);
            const int2 s1 = accum_act_unit(w0.x, w0.z, w1.x, w1.z, a1, 4U);
            acc00 += s1.x;
            acc01 += s1.y;
            const uint4 a2 = live ? ActUnits[act_base + 2U] : uint4(0U, 0U, 0U, 0U);
            const int2 s2 = accum_act_unit(w0.y, w0.w, w1.y, w1.w, a2, 0U);
            acc00 += s2.x;
            acc01 += s2.y;
            const uint4 a3 = live ? ActUnits[act_base + 3U] : uint4(0U, 0U, 0U, 0U);
            const int2 s3 = accum_act_unit(w0.y, w0.w, w1.y, w1.w, a3, 4U);
            acc00 += s3.x;
            acc01 += s3.y;
            rowSum0 += row_sum_unit(w0);
            rowSum1 += row_sum_unit(w1);
        }
        {
            const uint act_base = (m0 + 1U) * (4U * P.kwords) + u * 4U;
            const bool live = m0 + 1U < P.m;
            const uint4 a = live ? ActUnits[act_base + 0U] : uint4(0U, 0U, 0U, 0U);
            const int2 s = accum_act_unit(w0.x, w0.z, w1.x, w1.z, a, 0U);
            acc10 += s.x;
            acc11 += s.y;
            const uint4 a1 = live ? ActUnits[act_base + 1U] : uint4(0U, 0U, 0U, 0U);
            const int2 s1 = accum_act_unit(w0.x, w0.z, w1.x, w1.z, a1, 4U);
            acc10 += s1.x;
            acc11 += s1.y;
            const uint4 a2 = live ? ActUnits[act_base + 2U] : uint4(0U, 0U, 0U, 0U);
            const int2 s2 = accum_act_unit(w0.y, w0.w, w1.y, w1.w, a2, 0U);
            acc10 += s2.x;
            acc11 += s2.y;
            const uint4 a3 = live ? ActUnits[act_base + 3U] : uint4(0U, 0U, 0U, 0U);
            const int2 s3 = accum_act_unit(w0.y, w0.w, w1.y, w1.w, a3, 4U);
            acc10 += s3.x;
            acc11 += s3.y;
        }
        {
            const uint act_base = (m0 + 2U) * (4U * P.kwords) + u * 4U;
            const bool live = m0 + 2U < P.m;
            const uint4 a = live ? ActUnits[act_base + 0U] : uint4(0U, 0U, 0U, 0U);
            const int2 s = accum_act_unit(w0.x, w0.z, w1.x, w1.z, a, 0U);
            acc20 += s.x;
            acc21 += s.y;
            const uint4 a1 = live ? ActUnits[act_base + 1U] : uint4(0U, 0U, 0U, 0U);
            const int2 s1 = accum_act_unit(w0.x, w0.z, w1.x, w1.z, a1, 4U);
            acc20 += s1.x;
            acc21 += s1.y;
            const uint4 a2 = live ? ActUnits[act_base + 2U] : uint4(0U, 0U, 0U, 0U);
            const int2 s2 = accum_act_unit(w0.y, w0.w, w1.y, w1.w, a2, 0U);
            acc20 += s2.x;
            acc21 += s2.y;
            const uint4 a3 = live ? ActUnits[act_base + 3U] : uint4(0U, 0U, 0U, 0U);
            const int2 s3 = accum_act_unit(w0.y, w0.w, w1.y, w1.w, a3, 4U);
            acc20 += s3.x;
            acc21 += s3.y;
        }
        {
            const uint act_base = (m0 + 3U) * (4U * P.kwords) + u * 4U;
            const bool live = m0 + 3U < P.m;
            const uint4 a = live ? ActUnits[act_base + 0U] : uint4(0U, 0U, 0U, 0U);
            const int2 s = accum_act_unit(w0.x, w0.z, w1.x, w1.z, a, 0U);
            acc30 += s.x;
            acc31 += s.y;
            const uint4 a1 = live ? ActUnits[act_base + 1U] : uint4(0U, 0U, 0U, 0U);
            const int2 s1 = accum_act_unit(w0.x, w0.z, w1.x, w1.z, a1, 4U);
            acc30 += s1.x;
            acc31 += s1.y;
            const uint4 a2 = live ? ActUnits[act_base + 2U] : uint4(0U, 0U, 0U, 0U);
            const int2 s2 = accum_act_unit(w0.y, w0.w, w1.y, w1.w, a2, 0U);
            acc30 += s2.x;
            acc31 += s2.y;
            const uint4 a3 = live ? ActUnits[act_base + 3U] : uint4(0U, 0U, 0U, 0U);
            const int2 s3 = accum_act_unit(w0.y, w0.w, w1.y, w1.w, a3, 4U);
            acc30 += s3.x;
            acc31 += s3.y;
        }
    }

    const int out00 = acc00 - 128 * rowSum0;
    const int out01 = acc01 - 128 * rowSum1;
    const int out10 = acc10 - 128 * rowSum0;
    const int out11 = acc11 - 128 * rowSum1;
    const int out20 = acc20 - 128 * rowSum0;
    const int out21 = acc21 - 128 * rowSum1;
    const int out30 = acc30 - 128 * rowSum0;
    const int out31 = acc31 - 128 * rowSum1;
    if (n0 < P.n) {
        if (m0 + 0U < P.m) {
            OutAcc[(m0 + 0U) * P.n + n0] = out00;
            if (col1) {
                OutAcc[(m0 + 0U) * P.n + (n0 + 1U)] = out01;
            }
        }
        if (m0 + 1U < P.m) {
            OutAcc[(m0 + 1U) * P.n + n0] = out10;
            if (col1) {
                OutAcc[(m0 + 1U) * P.n + (n0 + 1U)] = out11;
            }
        }
        if (m0 + 2U < P.m) {
            OutAcc[(m0 + 2U) * P.n + n0] = out20;
            if (col1) {
                OutAcc[(m0 + 2U) * P.n + (n0 + 1U)] = out21;
            }
        }
        if (m0 + 3U < P.m) {
            OutAcc[(m0 + 3U) * P.n + n0] = out30;
            if (col1) {
                OutAcc[(m0 + 3U) * P.n + (n0 + 1U)] = out31;
            }
        }
    }
}

[shader("compute")][numthreads(GEMM_PLANE_MICROTILE_THREADS, 1, 1)] void
gemm_plane_microtile(uint3 gtid : SV_DispatchThreadID) {
    gemmPlaneMicrotileBody<BufferPlaneWeightSource>(gtid);
}

[shader("compute")][numthreads(GEMM_PLANE_MICROTILE_THREADS, 1, 1)] void
gemm_plane_microtile_image(uint3 gtid : SV_DispatchThreadID) {
    gemmPlaneMicrotileBody<ImagePlaneWeightSource>(gtid);
}

// ===========================================================================
// Coop-mat family: VK_KHR_cooperative_matrix hardware MMA - STAGED VARIANT
// ONLY: the int8 operands stage through
// GROUPSHARED memory (SSBO load -> shared -> collective coopmat load). The
// matrix units read from on-chip tile storage - that is the entire value
// proposition, and NVIDIA's own int8 reference benchmark
// (jeffbolznv/vk_cooperative_matrix_perf, shmem.comp) loads from shared
// exclusively; a direct SSBO->coopmat load has no known-good upstream
// reference under ANY compiler.
//
// On this toolchain (slangc 2026.17-25-gc7954ebd1) the
// compiler cannot wire CoopMat.Load results into coopMatMulAdd operands
// for int8 in EITHER the direct or the staged form (confirmed broken in
// both: the MulAdd consumes loop-preheader OpUndef regardless of load
// path) - this is an upstream slangc operand-plumbing bug, not an
// architecture problem: the DRIVER is proven correct by independent
// falsification evidence (GLSL compiled by glslangValidator with the
// IDENTICAL staged semantics was bit-exact on the same driver). This
// kernel IS the correct architecture, staying opt-in experimental until
// slangc fixes the bug - no code change will be needed here when it does.
//
// Staging budget per 32-column K-chunk (all in groupshared, one 32-thread
// subgroup per workgroup): A tile 16x32 int8 = 512 B + B tile 32x16 int8
// = 512 B + the 16x16 int32 accumulator window = 1 KB -> ~2 KB total, far
// below the 32 KiB class limit (caps-conformance check enforces the
// contract). The chunked-K staging loop is part of the known-good
// pattern, not a workaround.
//
// Operands are HOST-PACKED plain int8 (the tensor bytes verbatim: weights
// ternary {-1,0,+1}, activations int8), zero-padded to tile/K multiples so
// the collective math never sees padding values that contribute non-zero.
// P.kwords carries the PADDED raw K (the row stride); P.mode carries the
// padded M (transposed-act stride). The lane moves 8-bit operands (4x the
// 2-bit deployed weights), so decode shapes stay gemm_plane_tiled
// territory; it only measures whether hardware MMA wins prefill compute.
//
// Capability-gated at RUNTIME (variant_selection.hpp
// coop_mat_shape_matches): the module loads only when the device's
// vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR enumeration contains
// the exact shape compiled below (sint8*sint8 -> sint32, 16x16x32,
// SUBGROUP scope, non-saturating) AND the SPIR-V feature dependencies
// query TRUE (cooperativeMatrix, vulkanMemoryModel[DeviceScope],
// shaderInt8, storageBuffer8BitAccess) AND the device reports fixed
// 32-lane subgroups. Without it the printed skip note documents the
// fall-through and gemm_plane_tiled serves - no coop-mat dispatch is
// attempted.
// ===========================================================================

// Host-packed plain-int8 operands (tensor bytes verbatim, tile/K padded):
// int8 views of the SAME two storage slots the plane family uses above -
// each compiled entry point references exactly one view, the PROVEN
// pattern already shipped in this file (binding 2/3 are the shared ones).
[vk::binding(0, 0)] StructuredBuffer<int8_t> CoopMatWeights; // n x K, n-major rows
[vk::binding(1, 0)] StructuredBuffer<int8_t> CoopMatActs;    // m x K, K-major (transposed)
// Staging tiles: one A tile (M x K) and one B tile
// (K x N) per workgroup, int8 in Workgroup storage.
groupshared int8_t s_coopmat_a[GEMM_COOPMAT_M * GEMM_COOPMAT_K];
groupshared int8_t s_coopmat_b[GEMM_COOPMAT_K * GEMM_COOPMAT_N];
groupshared int32_t s_coopmat_c[GEMM_COOPMAT_SUBGROUPS * GEMM_COOPMAT_M * GEMM_COOPMAT_N];

[shader("compute")]
[numthreads(GEMM_COOPMAT_THREADS, 1, 1)] void gemm_coopmat(uint3 gtid : SV_GroupThreadID,
                                                           uint3 gid : SV_GroupID) {
    typealias CoopMatA =
        linalg.CoopMat<int8_t, MemoryScope.Subgroup, GEMM_COOPMAT_M, GEMM_COOPMAT_K,
                       linalg.CoopMatMatrixUse.MatrixA>;
    typealias CoopMatB =
        linalg.CoopMat<int8_t, MemoryScope.Subgroup, GEMM_COOPMAT_K, GEMM_COOPMAT_N,
                       linalg.CoopMatMatrixUse.MatrixB>;
    typealias CoopMatC =
        linalg.CoopMat<int32_t, MemoryScope.Subgroup, GEMM_COOPMAT_M, GEMM_COOPMAT_N,
                       linalg.CoopMatMatrixUse.MatrixAccumulator>;

    const uint lane = gtid.x;
    const uint tilesX = (P.n + GEMM_COOPMAT_N - 1U) / GEMM_COOPMAT_N;
    const uint tilesY = (P.m + GEMM_COOPMAT_M - 1U) / GEMM_COOPMAT_M;
    const uint tile = gid.x;
    const bool live = tile < tilesX * tilesY; // uniform within the subgroup
    const uint n0 = (tile % tilesX) * GEMM_COOPMAT_N;
    const uint m0 = (tile / tilesX) * GEMM_COOPMAT_M;

    CoopMatC acc;
    acc.fill(0);

    // Staged pattern (the known-good int8 architecture): each K-chunk goes
    // SSBO -> groupshared -> collective coopmat load -> ONE hardware
    // MulAdd. A (M x K) stages transposed-on-read from the K-major acts
    // buffer into row-major shared (stride K); B (K x N) stages verbatim
    // from the n-major weights rows into n-major shared (ColumnMajor load,
    // stride K). Padding is zero from the host packing, so padded lanes
    // contribute exactly 0; the collective reads never leave shared memory.
    for (uint kb = 0; live && kb < P.kwords; kb += GEMM_COOPMAT_K) {
        CoopMatA matA;
        CoopMatB matB;
        for (uint i = lane; i < GEMM_COOPMAT_M * GEMM_COOPMAT_K; i += GEMM_COOPMAT_THREADS) {
            const uint r = i / GEMM_COOPMAT_K; // m row
            const uint c = i % GEMM_COOPMAT_K; // k col
            s_coopmat_a[i] = CoopMatActs[(kb + c) * P.mode + m0 + r];
        }
        for (uint i = lane; i < GEMM_COOPMAT_K * GEMM_COOPMAT_N; i += GEMM_COOPMAT_THREADS) {
            const uint cn = i / GEMM_COOPMAT_K; // n col
            const uint c = i % GEMM_COOPMAT_K;  // k col
            s_coopmat_b[i] = CoopMatWeights[(n0 + cn) * P.kwords + kb + c];
        }
        GroupMemoryBarrierWithGroupSync();

        matA.Load<linalg.CoopMatMatrixLayout.RowMajor>(s_coopmat_a, 0U, GEMM_COOPMAT_K);
        matB.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(s_coopmat_b, 0U, GEMM_COOPMAT_K);
        acc = linalg.coopMatMulAdd<int32_t, false, int8_t, int8_t, int32_t,
                                   MemoryScope.Subgroup, GEMM_COOPMAT_M, GEMM_COOPMAT_K,
                                   GEMM_COOPMAT_N>(matA, matB, acc);
        GroupMemoryBarrierWithGroupSync(); // s_a/s_b rewritten next chunk
    }

    // Bounds-checked epilogue: fragments live in registers, so the tile goes
    // through shared memory (a store leg - not the math) to reach the valid
    // m x n window of OutAcc.
    const uint cElems = GEMM_COOPMAT_M * GEMM_COOPMAT_N;
    if (live) {
        acc.Store<linalg.CoopMatMatrixLayout.RowMajor>(s_coopmat_c, 0U, GEMM_COOPMAT_N);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint i = lane; live && i < cElems; i += GEMM_COOPMAT_THREADS) {
        const uint r = i / GEMM_COOPMAT_N;
        const uint col = i % GEMM_COOPMAT_N;
        if (m0 + r < P.m && n0 + col < P.n) {
            OutAcc[(m0 + r) * P.n + (n0 + col)] = s_coopmat_c[i];
        }
    }
}
