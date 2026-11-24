# AGENTS.md — set up, test and reproduce

Instructions for an agent working in this repository.
Everything runs from the repository root; every expectation below is
the observed output of these exact commands. Human-oriented walkthrough:
[README.md](README.md). Plan and evidence:
[Technical_Plan.pdf](Technical_Plan.pdf).

## Prerequisites

| Needed | Version / note |
| --- | --- |
| CMake + Ninja | on PATH |
| C++23 toolchain | MSVC (Windows host) or Android NDK r27 (device lane) |
| Vulkan SDK | provides `slangc`; set `VULKAN_SDK` or edit the hint in `CMakeLists.txt` |
| Python | 3.12+ with `numpy` and `huggingface_hub` (real-weight lane only) |
| Quality-gate tools | markdownlint-cli2 (Node.js), ruff, mypy, pytest, PSScriptAnalyzer, clang-tidy, cppcheck, clang-format on PATH |
| TinyTeX (`pdflatex`, `chktex`) | only to lint and rebuild the report (`python tools/build_report.py`) |
| adb | only for the device lane (`run-on-device.ps1`) |

## 1. Build

```powershell
pwsh scripts/build-windows.ps1
```

Expect exit 0 with zero warnings. Compiles every Slang kernel to SPIR-V
(one set per GPU profile) and the host programs.

## 2. Test harness

```powershell
build\windows\poc_tests.exe --spv-dir build\windows\spv
```

Expect exit 0, `ALL TESTS PASSED`, `caps: 50 checks, 0 violations`,
`validation: ON (0 messages)`. The run covers deterministic synthetic
shapes, the hand-computed known-answer vector
([data/golden_vector.txt](data/golden_vector.txt)), and negative controls
that must fail; real-weight lanes join only with `--tensors` (step 5).

Prove the check can fail (once per fresh copy):

```powershell
Rename-Item data\golden_vector.txt golden_vector.txt.away
build\windows\poc_tests.exe --spv-dir build\windows\spv   # expect FAIL, exit 1
Rename-Item data\golden_vector.txt.away golden_vector.txt
```

## 3. Quality gate

```powershell
pwsh scripts/verify.ps1
```

Expect `VERIFY: PASS`: markdownlint, ruff, mypy, PSScriptAnalyzer,
clang-tidy (production + tests), cppcheck, C++ unit tests, pytest,
clang-format. The C++ lanes need the build of step 1.

## 4. Benchmarks (CSV on stdout)

```powershell
build\windows\poc_bench.exe --spv-dir build\windows\spv
build\windows\poc_bench.exe --spv-dir build\windows\spv --system
```

Expect 39 matrix rows and no `NOT TIMED` lines; `--system` reports
`verified_sites=7` for decode and prefill and `verified_sites=1` for the
head, and fires its own negative control (`half_grid_rejected=1`). A row
exists only if the oracle verified it.

## 5. Real-weight lane (optional)

```powershell
python tools/export_weights.py
build\windows\poc_tests.exe --spv-dir build\windows\spv --tensors models\real_weights.bitnet1
```

Expect 167 PASS lines instead of 153 (the real-model tests join). The
container regenerates byte-identically from the checkpoint (sha256
documented in [data/README.md](data/README.md)); model files are
downloaded or generated, never shipped — see
[models/README.md](models/README.md).

## 6. Device lane (optional)

```powershell
pwsh scripts/build-android.ps1
pwsh scripts\run-on-device.ps1 -Serial <adb serial> -Tensors models\real_weights.bitnet1
```

Expect the same `ALL TESTS PASSED` on the device. The validation layer
does not load on production Android builds; correctness there is the
bit-exact integer oracle — the real-weight scope is described in
[data/README.md](data/README.md).

## Reproduction contract

- Every claim reduces to a command above and its output; no number without
  a run behind it.
- The harness cannot pass by accident: exact integer equality, seeded
  inputs, negative controls, limits checked before any GPU work.
- Never edit `data/golden_vector.txt` or a measured number; regenerate
  derived files (`models/*`) instead.
