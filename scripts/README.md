# scripts/ — build, run, verify

Four scripts, run from the folder root; every script fails loudly (non-zero
exit on error).

| Script | Purpose |
| --- | --- |
| `build-windows.ps1` | Windows host build: `windows` preset → `build/windows`; host programs + Slang → SPIR-V per GPU profile |
| `build-android.ps1` | Android arm64-v8a cross-build: `android` preset → `build/android` (NDK r27) |
| `run-on-device.ps1` | adb push of binaries and SPIR-V kernels to `/data/local/tmp/bitnet/`, run, stream output (`-Serial`, `-AppArgs`, `-Bench`, `-Tensors`, `-Validation`) |
| `verify.ps1` | the quality gate (single entry point): markdownlint, ruff, mypy, PSScriptAnalyzer, clang-tidy, cppcheck, C++ unit tests, pytest, clang-format; needs the build tree of `build-windows.ps1` |

## Typical session

```powershell
pwsh scripts/build-windows.ps1                              # 1. build
build\windows\poc_tests.exe --spv-dir build\windows\spv     # 2. harness
pwsh scripts/verify.ps1                                     # 3. gate
```

Device loop: `build-android.ps1` → `run-on-device.ps1 -Serial <serial>`.
Optional real-weight run: append
`--tensors models\real_weights.bitnet1` to the harness.

Expected outputs of every step and the full reproduction recipe:
[../AGENTS.md](../AGENTS.md).
