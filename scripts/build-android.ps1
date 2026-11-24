# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Norbert Nopper

# Android arm64-v8a cross-build: configure with the 'android' preset (NDK r27
# toolchain), build the PoC targets, print artifact paths.
# Note: Slang -> SPIR-V always compiles on the host - slangc from the Vulkan
# SDK runs locally even in the android preset; only the C++ targets are
# cross-compiled by the NDK toolchain.
# Usage: pwsh scripts/build-android.ps1

param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$buildDir = Join-Path $repo 'build/android'

cmake --preset android
if ($LASTEXITCODE -ne 0) { Write-Output "FAIL configure (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

cmake --build $buildDir --target poc_tests poc_bench poc_membw spv_kernels
if ($LASTEXITCODE -ne 0) { Write-Output "FAIL build (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Output ''
Write-Output 'Artifacts:'
Write-Output (Join-Path $buildDir 'poc_tests')
Write-Output (Join-Path $buildDir 'poc_bench')
Write-Output (Join-Path $buildDir 'poc_membw')
$spv = Get-ChildItem (Join-Path $buildDir 'spv') -Filter *.spv -ErrorAction SilentlyContinue
foreach ($f in $spv) { Write-Output $f.FullName }
