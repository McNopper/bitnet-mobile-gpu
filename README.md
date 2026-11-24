# 1.58-bit BitNet Inference on Mobile GPUs via llama.cpp

This repository contains the two deliverables:

1. **A technical plan report (PDF)**: `Technical_Plan.pdf` (this
   folder, next to this README; A4). The report documents the design,
   the measurements and the evaluation of the proof of concept (PoC);
   Section 9 describes the PoC itself.
2. **Scripts containing the source code**: the proof of concept in `poc/`,
   with the scripts that build, run and check it.

## Contents

| Path | Purpose |
| --- | --- |
| `Technical_Plan.pdf` | the technical plan report (built from the source below) |
| `Technical_Plan.tex` | **the report's single source file**: prose, tables and all three figures as in-source TikZ pictures; rebuild with `python tools/build_report.py` |
| [`poc/`](poc/README.md) | source of the PoC |
| [`scripts/`](scripts/README.md) | build, run and check scripts |
| [`tools/`](tools/README.md) | export weights from real models |
| [`data/`](data/README.md) | test and validation data |
| [`models/`](models/README.md) | optional real-weight inputs for the harness |
| [`AGENTS.md`](AGENTS.md) | setup, testing and reproduction for an agent |
| [`README.md`](README.md), [`LICENSE`](LICENSE) | this file; MIT license |

## How to execute

Prerequisites: CMake + Ninja (PATH), Vulkan SDK 1.4.363.1 (`slangc`; set
`VULKAN_SDK` or edit the hint in `CMakeLists.txt`), a Vulkan 1.3-capable
loader (physical devices may use the 1.0 popcount fallback), MSVC or clang
with C++23, Python 3.12+. For the device steps: Android NDK r27 (the `android`
preset in `CMakePresets.json` names its path) and `adb` on PATH
(`scripts/run-on-device.ps1` reads the platform-tools path — adjust it to
your SDK location). For rebuilding the report: a LaTeX distribution with
`pdflatex` and `chktex`, for example TinyTeX
(<https://yihui.org/tinytex/>) with
`tlmgr install chktex pgf parskip microtype upquote`. The quality gate also
requires markdownlint-cli2 (Node.js), ruff, mypy and pytest (Python),
PSScriptAnalyzer, clang-tidy, cppcheck and clang-format on PATH.

```powershell
# 1. Desktop build: Slang -> SPIR-V per GPU profile, then the host programs
pwsh scripts/build-windows.ps1

# 2. Test harness: bit-exact GPU against the CPU oracle, sample input and
#    expected output. Expect "ALL TESTS PASSED", exit 0,
#    "caps: 50 checks, 0 violations" and "validation: ON (0 messages)".
.\build\windows\poc_tests.exe --spv-dir build\windows\spv

# 3. Benchmarks (CSV on stdout): kernel matrix, then the whole-system pass
.\build\windows\poc_bench.exe --spv-dir build\windows\spv
.\build\windows\poc_bench.exe --spv-dir build\windows\spv --system

# 4. Android device (arm64-v8a): build, push, run, stream the output
pwsh scripts\build-android.ps1
pwsh scripts\run-on-device.ps1 -Serial <adb serial>

# 5. Quality gate (all lanes); it needs the build tree of step 1
pwsh scripts\verify.ps1

# 6. Report: lint the LaTeX source (structural + ChkTeX, zero findings)
#    and rebuild Technical_Plan.pdf next to its source
python tools\build_report.py
```

The build is standard CMake + Ninja with `slangc` from the Vulkan SDK; the
PowerShell scripts wrap it. Built and tested: Windows (MSVC 19.52) and
Android arm64-v8a (NDK r27, clang 18). Other hosts are not tested.

## Test harness and sample input/output validation

For every shape the harness generates deterministic inputs, computes the
reference with a naive CPU loop, runs every applicable core kernel on the
GPU and requires equal int32 outputs; a deliberately under-dispatched
kernel must fail the comparison (negative control, in every run). The
sample input and expected output are the known-answer vector
`data/golden_vector.txt` ($K = 64$, expected $+160$ for even rows and
$-160$ for odd ones), checked against the file, against the closed form
and by every applicable core kernel. The 20 synthetic shapes cover the
q/o, gate/up and down-projection dimensions, odd sizes, extreme values
(+127/−128), all zeros and 1-bit weights. The 640 × 2560 K/V projection is
covered by the system benchmark, not by every core kernel.

One pytest case reads the bf16 checkpoint of the model (4.8 GB, not part
of this folder) and is skipped when it is not present locally; the
`--tensors` path of `poc_tests` takes such a container if you supply
one; `tools/export_weights.py` writes it to `models/real_weights.bitnet1`
by default (see [`models/README.md`](models/README.md)).
