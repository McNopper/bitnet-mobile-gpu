// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Norbert Nopper

// Null probe: one store per workgroup - measures the per-submit + fence
// floor of the dispatch_bench timing loop. The CSV row
// "null_probe" is the per-submit baseline every kernel time carries.
RWStructuredBuffer<int> Out;

[shader("compute")]
[numthreads(64, 1, 1)]
void null_probe(uint3 gid : SV_GroupID) {
    Out[gid.x] = int(gid.x);
}