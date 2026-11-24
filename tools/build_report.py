# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Norbert Nopper

"""Lint the report's LaTeX source and rebuild the PDF from it.

The folder carries ONE source file for the report,
`Technical_Plan.tex` (its three figures are TikZ pictures inside
it - no external image files), and the PDF built from it. This tool runs
the full loop:

 1. structural lint (fail-hard): UTF-8 without BOM, pure ASCII (pdflatex
    chokes on un-mapped Unicode), balanced environments and braces, no
    `\\input`/`\\include` (single-source guarantee), no absolute local
    paths, every `\\includegraphics` target present in the folder;
 2. ChkTeX lint (the ecosystem-standard LaTeX linter): ZERO findings are
    the gate. The only excluded class is W8 (dash length), which fires on
    content that is provably correct here: ISO dates (2026-11-13), TikZ
    path operators (--) and negative coordinates (-3.5mm) - a minimal
    reproduction flags a correct ISO date, so the class cannot be
    satisfied without corrupting correct text. Every other class stays on;
 3. build: two `pdflatex` passes into `build/tex/`;
 4. verify: A4 page size and a plausible page count;
 5. install: copy the PDF next to the source (overwrites the previous
    build - the source is the only thing edited by hand).

Prerequisites: a LaTeX distribution with `pdflatex` and `chktex` on PATH,
for example TinyTeX (https://yihui.org/tinytex/) with
`tlmgr install chktex pgf parskip microtype upquote`.

Usage:
    python tools/build_report.py [--source Technical_Plan.tex]
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import subprocess
from pathlib import Path

# ChkTeX runs with W8 excluded (dash length): a minimal reproduction shows
# it flags a correct ISO date (2026-11-13), and this document's other dash
# uses are TikZ path operators and negative numbers - all correct. Every
# other warning class is enabled and must report ZERO findings.
CHKTEx_ARGS = ("-n8",)
CHKTEx_BASELINE = 0

A4 = (595.28, 841.89)  # pt


def find_tool(*names: str) -> str | None:
    """Locates a tool on PATH, then in the usual TinyTeX bin folders."""
    for name in names:
        found = shutil.which(name)
        if found:
            return found
        stem = name + (".exe" if os.name == "nt" else "")
        for base in (
            Path(os.environ.get("APPDATA", "")) / "TinyTeX" / "bin" / "windows",
            Path.home() / "Library" / "TeX" / "bin" / "darwin",
            Path("/usr/local/texlive/bin/x86_64-linux"),
        ):
            if (base / stem).exists():
                return str(base / stem)
    return None


def lint_structural(text: str, folder: Path) -> list[str]:
    """Fail-hard structural checks; returns the findings (empty = pass)."""
    findings: list[str] = []
    if text.startswith("\ufeff"):
        findings.append("file starts with a byte-order mark")
    non_ascii = sorted({c for c in text if ord(c) > 127})
    if non_ascii:
        findings.append(
            "non-ASCII characters (pdflatex cannot typeset them): "
            + ", ".join(f"U+{ord(c):04X}" for c in non_ascii)
        )
    begins = re.findall(r"\\begin\{(\w+\*?)\}", text)
    ends = re.findall(r"\\end\{(\w+\*?)\}", text)
    for env in sorted(set(begins) | set(ends)):
        if begins.count(env) != ends.count(env):
            findings.append(
                f"environment {env}: {begins.count(env)} begin vs {ends.count(env)} end"
            )
    if text.count("{") != text.count("}"):
        findings.append(f"unbalanced braces: {text.count('{{')} {{ vs {text.count('}')}}}")
    for forbidden in (r"\input{", r"\include{", r"\subfile{"):
        if forbidden in text:
            findings.append(f"external include {forbidden!r} breaks the single source")
    for match in re.finditer(r"[A-Za-z]:[\\/](?:Users|Temp)|/(?:Users|home)/", text):
        findings.append(f"absolute local path: {match.group(0)!r}")
    for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        target = match.group(1).strip()
        if not (folder / target).exists():
            findings.append(f"includegraphics target missing: {target}")
    return findings


def lint_chktex(chktex: str, source: Path) -> tuple[int, str]:
    """Runs ChkTeX; returns (finding count, first findings for the report)."""
    cmd = [chktex, "-q", *CHKTEx_ARGS, source.name]
    proc = subprocess.run(
        cmd, cwd=source.parent, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    count = len(re.findall(r"Warning \d+ ", out))
    sample = "\n".join(line for line in out.splitlines() if line.strip())[:600]
    return count, sample


def build(pdflatex: str, source: Path) -> Path:
    """Two pdflatex passes into build/tex/; returns the produced PDF."""
    out_dir = source.parent / "build" / "tex"
    out_dir.mkdir(parents=True, exist_ok=True)
    for pass_no in (1, 2):
        proc = subprocess.run(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={out_dir.as_posix()}",
                source.name,
            ],
            cwd=source.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").splitlines()[-25:])
            raise SystemExit(f"FAIL: pdflatex pass {pass_no} (exit {proc.returncode}):\n{tail}")
    pdf = out_dir / (source.stem + ".pdf")
    if not pdf.exists():
        raise SystemExit(f"FAIL: {pdf} was not produced")
    return pdf


def verify_pdf(pdf: Path) -> tuple[int, str]:
    """Reads the PDF header facts: (page count, 'WxH mm' description).

    pdflatex compresses object streams, so the page objects are inflated
    with zlib before the /MediaBox and /Type /Page facts are read.
    """
    import zlib

    data = pdf.read_bytes()
    blobs = [data]
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        with contextlib.suppress(zlib.error):
            blobs.append(zlib.decompress(m.group(1)))
    haystack = b"\n".join(blobs)
    boxes = re.findall(
        rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", haystack
    )
    if not boxes:
        raise SystemExit("FAIL: no /MediaBox found in the PDF")
    x0, y0, x1, y1 = (float(v) for v in boxes[0])
    width_mm = (float(x1) - float(x0)) * 25.4 / 72
    height_mm = (float(y1) - float(y0)) * 25.4 / 72
    pages = len(re.findall(rb"/Type\s*/Page[^s]", haystack))
    if abs(width_mm - 210.0) > 1.0 or abs(height_mm - 297.0) > 1.0:
        raise SystemExit(f"FAIL: page size {width_mm:.1f} x {height_mm:.1f} mm is not A4")
    if pages < 1:
        raise SystemExit("FAIL: page count could not be read")
    return pages, f"{width_mm:.1f} x {height_mm:.1f} mm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="Technical_Plan.tex")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_file():
        raise SystemExit(f"FAIL: {source} not found (run from the folder root)")
    folder = source.parent
    text = source.read_text(encoding="utf-8")

    print("# bitnet report build (tools/build_report.py)")

    findings = lint_structural(text, folder)
    print(f"lint (structural): {len(findings)} finding(s)")
    for item in findings:
        print(f"  {item}")
    if findings:
        raise SystemExit("FAIL: structural lint")

    chktex = find_tool("chktex")
    if chktex:
        count, sample = lint_chktex(chktex, source)
        print(f"lint (chktex): {count} finding(s), zero required")
        if count > CHKTEx_BASELINE:
            print(sample)
            raise SystemExit("FAIL: chktex reports findings")
    else:
        print("lint (chktex): NOT FOUND - install TinyTeX + `tlmgr install chktex`")
        raise SystemExit("FAIL: chktex unavailable (it is part of the gate)")

    pdflatex = find_tool("pdflatex", "xelatex")
    if not pdflatex:
        raise SystemExit(
            "FAIL: pdflatex not found (install a LaTeX distribution, "
            "e.g. TinyTeX: https://yihui.org/tinytex/)"
        )
    print(f"build: {pdflatex}")
    pdf = build(pdflatex, source)

    pages, size = verify_pdf(pdf)
    print(f"verify: {pages} pages, {size} (A4)")

    target = folder / (source.stem + ".pdf")
    shutil.copyfile(pdf, target)
    print(f"RENDER: OK ({pages} pages, A4) -> {target}")


if __name__ == "__main__":
    main()
