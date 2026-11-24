# poc/ — the proof of concept

Bit-exact ternary (1.58-bit) matrix multiplication for a single BitLinear
layer of a transformer, written for mobile GPUs: Slang kernels compiled to
SPIR-V (one variant set per GPU profile), a C++23 host behind a single
backend interface, a harness that requires GPU == CPU integer equality, and
benchmarks that time only what the oracle verified. Plan and evidence:
[Technical_Plan.pdf](../Technical_Plan.pdf) (Section 9).

## Layout

| Folder | Contents |
| --- | --- |
| `common/` | packed-format contract (`format.hpp`), the CPU oracle (`reference.*`), deterministic tensor generation, the BITNET1 container reader, portable environment access |
| `backend/` | `IComputeBackend` — the only interface between the harness and a GPU implementation — plus capability conformance, extension families and the generated GPU profiles |
| `vulkan/` | the Vulkan implementation: construction and device setup, dispatch and timing, capability queries, device objects |
| `metal/` | prepared-only stub: compiled, never executed |
| `slang/` | `bitnet_kernels.slang` (every kernel in one source) and the measurement probes |
| `tests/` | the `poc_tests` harness, its suites (golden vector, capabilities, cooperative matrix, real weights) and the host unit tests |
| `bench/` | the `poc_bench` kernel matrix and whole-system pass, the `poc_membw` CPU bandwidth probe |

## Build and run

From the folder root (full sequence: [../README.md](../README.md)):

```powershell
pwsh scripts/build-windows.ps1
build\windows\poc_tests.exe --spv-dir build\windows\spv
build\windows\poc_bench.exe --spv-dir build\windows\spv
```

Expected: `ALL TESTS PASSED`, `caps: 47 checks, 0 violations`,
`validation: ON (0 messages)` — the last one on the desktop only; the
validation layer does not load on production Android builds, where the
integer oracle is the gate.

## What the tests enforce

- **Bit-exact or broken:** GPU output must equal the CPU oracle integer for
  integer — zero tolerance, every shape, every kernel.
- **The check can fail:** a deliberately under-dispatched grid must fail the
  comparison (negative control, every run); the known-answer vector is
  [golden_vector.txt](../data/golden_vector.txt), checked against the file,
  a closed form and every kernel.
- **Limits before work:** every knob and dispatch grid is checked against
  the device limits before any GPU work; a breach fails the run.
- **Timing only after the oracle:** a benchmark row exists only after its
  output was verified.

Setup, gates and the reproduction contract: [../AGENTS.md](../AGENTS.md).
