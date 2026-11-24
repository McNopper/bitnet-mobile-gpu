# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Norbert Nopper

# Push the Android PoC binaries + SPIR-V kernels to the device over adb and
# run them, forwarding stdout. Needs build/android/ artifacts from
# scripts/build-android.ps1.
# Usage: pwsh scripts/run-on-device.ps1 -Serial <adb-serial> [-Bench] [-AppArgs '<flags>']

param(
  [string]$Serial = '',
  [string]$AppArgs = '',
  [switch]$Bench,
  [string]$Tensors = '',
  # Validation layers on-device: the Allwinner BSP loader (Mali 40.0.0)
  # attempts layer init when setprop debug.vulkan.layers is set but fails
  # silently (VK_ERROR_INITIALIZATION_FAILED; no loader debug output), so
  # ON-DEVICE validation is not achievable on this device. Correctness
  # gates: desktop runs validate; device runs rely on bit-exact GPU==CPU.
  [switch]$Validation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
# adb: PATH first (the documented prerequisite), fallback to the SDK layout
$adbOnPath = Get-Command adb -ErrorAction SilentlyContinue
if ($adbOnPath) { $adb = $adbOnPath.Source } else { $adb = 'C:/Development/AndroidSdk/platform-tools/adb.exe' }
$deviceDir = '/data/local/tmp/bitnet'
$deviceSpvDir = "$deviceDir/spv"

if (-not (Test-Path $adb)) { Write-Output "adb not found at $adb"; exit 1 }
if ($Serial -eq '') { Write-Output 'Serial is required: pass -Serial <adb serial> (see adb devices)'; exit 2 }

# Native adb failures do not trigger PowerShell's ErrorActionPreference.
# Check every state-changing call so stale files on the device cannot mask
# a failed mkdir/push/chmod. The final device-state query stays best-effort.
function Invoke-Adb {
  param([Parameter(Mandatory = $true)][string[]]$AdbArgs)
  & $script:adb -s $script:Serial @AdbArgs
  if ($LASTEXITCODE -ne 0) { Write-Output "FAIL adb $($AdbArgs[0]) (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}

$exe = 'poc_tests'
$hostBin = Join-Path $repo 'build/android/poc_tests'
$validationOff = ''
if ($Bench) {
  # Benchmark runs DISABLE validation layers - layer overhead would skew
  # the numbers. Correctness runs (poc_tests) keep validation ON.
  $exe = 'poc_bench'
  $hostBin = Join-Path $repo 'build/android/poc_bench'
  $validationOff = 'BITNET_VALIDATION=0 '
}
if (-not (Test-Path $hostBin)) {
  Write-Output "$exe not found at $hostBin - run scripts/build-android.ps1 first"
  exit 1
}

# Kernels were compiled on the host by the spv_kernels target (slangc runs
# locally even for the android preset). Compiled per GPU profile: the whole
# spv tree (spv/<profile>/<entry>.spv) is pushed; the binary selects the
# profile at runtime from the caps probe.
$hostSpvDir = Join-Path $repo 'build/android/spv'
$spvFiles = Get-ChildItem $hostSpvDir -Recurse -Filter *.spv -ErrorAction SilentlyContinue
if (-not $spvFiles) {
  Write-Output "no .spv kernels in $hostSpvDir - run scripts/build-android.ps1 first"
  exit 1
}

Write-Output '=== mkdir'
Invoke-Adb -AdbArgs @('shell', 'mkdir', '-p', $deviceSpvDir)

Write-Output '=== push binary'
Invoke-Adb -AdbArgs @('push', $hostBin, "$deviceDir/$exe")

Write-Output '=== push kernels (recursive, all profiles)'
foreach ($f in $spvFiles) {
  $rel = $f.FullName.Substring((Get-Item $hostSpvDir).FullName.Length + 1).Replace('\', '/')
  # Keep Android paths POSIX: Windows Split-Path emits backslashes, which the
  # remote shell consumes as escapes (turning /data/local/... into datalocal...).
  $relParent = $rel -replace '/[^/]+$', ''
  $parent = "$deviceSpvDir/$relParent"
  Invoke-Adb -AdbArgs @('shell', 'mkdir', '-p', $parent)
  Invoke-Adb -AdbArgs @('push', $f.FullName, "$deviceSpvDir/$rel")
}

Write-Output '=== chmod'
Invoke-Adb -AdbArgs @('shell', 'chmod', '755', "$deviceDir/$exe")

# Validation gate: push the Khronos layer and point the Vulkan loader at it
# via VK_LAYER_PATH when present; without it we continue unvalidated and
# the backend prints a warning.
$layerLocal = Join-Path $repo 'third_party/vulkan-validation/arm64-v8a/libVkLayer_khronos_validation.so'
$layerJson = Join-Path $repo 'third_party/vulkan-validation/arm64-v8a/VkLayer_khronos_validation.json'
$envPrefix = ''
if ($Validation -and (Test-Path $layerLocal) -and (Test-Path $layerJson)) {
  Write-Output '=== push validation layer'
  Invoke-Adb -AdbArgs @('shell', 'mkdir', '-p', "$deviceDir/layers")
  Invoke-Adb -AdbArgs @('push', $layerLocal, "$deviceDir/layers/")
  Invoke-Adb -AdbArgs @('push', $layerJson, "$deviceDir/layers/")
  # Khronos-loader recipe: VK_LAYER_PATH points the loader at the manifest.
  # We deliberately do NOT use `setprop debug.vulkan.layers` (the global
  # property recipe for native executables): on this BSP loader ANY value of
  # that property poisons vkCreateInstance (VK_ERROR_INITIALIZATION_FAILED)
  # and only a reboot clears it.
  # On proper loaders VK_LAYER_PATH alone enables the layer.
  $envPrefix = "VK_LAYER_PATH=$deviceDir/layers "
}
else {
  Write-Output 'validation layer unavailable - continuing unvalidated (pass -Validation to attempt it)'
}

# Real-model weights (Path A): push the BITNET1 container when provided.
$tensorsArg = ''
if ($Tensors -ne '') {
  if (-not (Test-Path $Tensors)) { Write-Output "tensors file not found: $Tensors"; exit 1 }
  Invoke-Adb -AdbArgs @('push', $Tensors, "$deviceDir/real_weights.bitnet1")
  $tensorsArg = '--tensors /data/local/tmp/bitnet/real_weights.bitnet1'
}

# Known-answer golden vector: pushed for every test run; the
# harness cross-checks it against the in-source closed form.
$goldenArg = ''
if (-not $Bench) {
  $goldenLocal = Join-Path $repo 'data/golden_vector.txt'
  if (Test-Path $goldenLocal) {
    Invoke-Adb -AdbArgs @('push', $goldenLocal, "$deviceDir/golden_vector.txt") | Out-Null
    $goldenArg = '--golden /data/local/tmp/bitnet/golden_vector.txt'
  } else {
    Write-Output 'golden_vector.txt not found in data/ - golden test will report it missing'
  }
}

Write-Output '=== run'
& $adb -s $Serial shell "$validationOff$envPrefix$deviceDir/$exe --spv-dir $deviceSpvDir $tensorsArg $goldenArg $AppArgs"
if ($LASTEXITCODE -ne 0) { Write-Output "FAIL run (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

# Device-state recording: GPU frequency/governor and
# thermal state around the run - best effort; some nodes need a userdebug
# build. Recorded facts go into the bench log for every measurement.
Write-Output '=== device state (devfreq/thermal, best effort)'
& $adb -s $Serial shell "for f in /sys/class/devfreq/*[Gg][Pp][Uu]*/governor /sys/class/devfreq/*[Gg][Pp][Uu]*/cur_freq; do (cat `$f 2>/dev/null | head -1) && echo `$f; done; for t in /sys/class/thermal/thermal_zone*/temp; do (cat `$t 2>/dev/null) || true; done" 2>&1 | ForEach-Object { Write-Output "  $_" }
