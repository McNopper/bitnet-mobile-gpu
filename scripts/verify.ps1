# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Norbert Nopper

# Quality gate (single entry point).
# Usage: pwsh scripts/verify.ps1
# Lanes: markdownlint, ruff (check + format), mypy, PSScriptAnalyzer,
# clang-tidy (production and test lanes), cppcheck, C++ unit tests, pytest,
# clang-format. The C++ lanes need the build tree of
# scripts/build-windows.ps1 (compile database, unit-test target).

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$failed = $false

function Invoke-Step {
  param([string]$Name, [scriptblock]$Action)
  Write-Output "=== $Name"
  $global:LASTEXITCODE = 0 # fresh baseline: some lanes are pure PS and never set it
  & $Action
  if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
    Write-Output "FAIL $Name (exit $LASTEXITCODE)"
    $script:failed = $true
  }
}

# Markdown structure/style lint (all prose, incl. mermaid fences).
Invoke-Step 'markdownlint-cli2' {
  Push-Location $repo
  try { npx.cmd --yes markdownlint-cli2 '**/*.md' '!third_party/**' '#node_modules' } finally { Pop-Location }
}

# Python lint + format check (ruff) over tools/ and config files. Two
# steps, not one: a step chaining two commands observes only the LAST
# exit code and would hide a failure of the first.
Invoke-Step 'ruff check' {
  Push-Location $repo
  try { ruff check . } finally { Pop-Location }
}
Invoke-Step 'ruff format' {
  Push-Location $repo
  try { ruff format --check tools } finally { Pop-Location }
}

# Type check (mypy strict) over tools/.
Invoke-Step 'mypy (strict)' {
  python -m mypy
}

# PowerShell lint over scripts/.
Invoke-Step 'PSScriptAnalyzer' {
  $hits = Invoke-ScriptAnalyzer -Path (Join-Path $repo 'scripts') -Recurse -ErrorAction SilentlyContinue
  if ($hits) { $hits | ForEach-Object { Write-Output "$($_.ScriptName):$($_.Line): $($_.Message)" } }
  if ($hits | Where-Object { $_.Severity -in 'Error', 'Warning' }) { $global:LASTEXITCODE = 1 }
}

# clang-tidy (AST analysis) over poc/ using the compile DB. Two lanes:
# production code gets the full profile including
# readability-magic-numbers (layout constants must be named); poc/tests is
# DATA-driven (tensor shapes, seeds) - magic numbers there are test
# vectors, so that single check is lifted for the tests lane only. Two
# steps, not one, so that each lane's exit code is observed. Findings fail
# the gate directly (there is no debt file).
Invoke-Step 'clang-tidy (prod)' {
  $prod = Get-ChildItem (Join-Path $repo 'poc') -Recurse -Include *.cpp |
    Where-Object { $_.FullName -notmatch '\\poc\\tests\\' } | ForEach-Object FullName
  clang-tidy -p (Join-Path $repo 'build/windows') '--warnings-as-errors=*' @prod
}
Invoke-Step 'clang-tidy (tests)' {
  $tests = Get-ChildItem (Join-Path $repo 'poc\tests') -Recurse -Include *.cpp | ForEach-Object FullName
  clang-tidy -p (Join-Path $repo 'build/windows') '--warnings-as-errors=*' '--checks=-*,bugprone-*,-bugprone-easily-swappable-parameters,performance-*,modernize-*,-modernize-use-trailing-return-type,-modernize-use-std-print,readability-*,-readability-implicit-bool-conversion,-readability-identifier-length,-readability-math-missing-parentheses,-readability-magic-numbers' @tests
}

# cppcheck (pattern/data-flow analysis; no compile DB needed). Cross-check
# for the Android build is the NDK cross-compile itself (-Werror there).
# No suppressions: the code avoids cppcheck-unevaluable constructs
# (__has_warning is short-circuited behind !defined(__CPPCHECK__)).
Invoke-Step 'cppcheck' {
  cppcheck --quiet --enable=warning,style,performance,portability --std=c++23 `
    --platform=win64 -I (Join-Path $repo 'poc') --inline-suppr --error-exitcode=2 `
    (Join-Path $repo 'poc')
}

# C++ unit tests (host-side components; kernel exactness lives in poc_tests).
if (Test-Path (Join-Path $repo 'build/windows/build.ninja')) {
  Invoke-Step 'unit tests (C++)' {
    cmake --build (Join-Path $repo 'build/windows') --target poc_unit_tests
    & (Join-Path $repo 'build/windows/poc_unit_tests.exe')
  }
}

# Python unit tests (tooling logic).
if (Test-Path (Join-Path $repo 'tools/tests')) {
  Invoke-Step 'pytest' {
    Push-Location $repo
    try { python -m pytest tools/tests -q } finally { Pop-Location }
  }
}

# C++ format check.
$cpp = Get-ChildItem (Join-Path $repo 'poc') -Recurse -Include *.cpp,*.hpp -ErrorAction SilentlyContinue
if ($cpp) {
  Invoke-Step 'clang-format' {
    foreach ($f in $cpp) {
      clang-format.exe --dry-run --Werror $f.FullName
      if ($LASTEXITCODE -ne 0) { $script:failed = $true }
    }
  }
}
else {
  Write-Output '=== clang-format: SKIP (no C++ sources yet)'
}

Write-Output ''
if ($failed) { Write-Output 'VERIFY: FAIL'; exit 1 }
Write-Output 'VERIFY: PASS'
