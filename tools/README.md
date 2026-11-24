# Python tooling

Python is the tooling language beside C++: the export of the real-weight
sample of the test harness and the report build. The proof of concept
itself is C++ and Slang.
Linted by ruff, typed by mypy (strict), tested by pytest - configuration in
`pyproject.toml`.

| Tool | Purpose |
| --- | --- |
| `export_weights.py` | exports one layer of the bf16 checkpoint as a `BITNET1` container for `poc_tests --tensors` (default output `models/real_weights.bitnet1`) |
| `build_report.py` | lints `Technical_Plan.tex` (structural checks + ChkTeX, zero findings) and rebuilds `Technical_Plan.pdf` next to it (needs `pdflatex`/`chktex`, e.g. TinyTeX) |
| `tests/` | pytest suite of the tools above |

Each tool documents its options (`--help`).

The container it writes is consumed by `poc_tests --tensors`; provenance
and scope: [data/README.md](../data/README.md). Generated model files:
[models/README.md](../models/README.md).
