# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Norbert Nopper

# Windows host build: configure with the 'windows' preset, build the PoC
# targets, print artifact paths.
# Usage: pwsh scripts/build-windows.ps1

param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$buildDir = Join-Path $repo 'build/windows'

cmake --preset windows
if ($LASTEXITCODE -ne 0) { Write-Output "FAIL configure (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

cmake --build $buildDir --target poc_tests poc_bench poc_membw spv_kernels
if ($LASTEXITCODE -ne 0) { Write-Output "FAIL build (exit $LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Output ''
Write-Output 'Artifacts:'
Write-Output (Join-Path $buildDir 'poc_tests.exe')
Write-Output (Join-Path $buildDir 'poc_bench.exe')
Write-Output (Join-Path $buildDir 'poc_membw.exe')
$spv = Get-ChildItem (Join-Path $buildDir 'spv') -Filter *.spv -ErrorAction SilentlyContinue
foreach ($f in $spv) { Write-Output $f.FullName }
