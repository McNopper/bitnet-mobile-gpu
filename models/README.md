# models/

Real-weight inputs for the test harness. Everything in this directory is
generated or downloaded, never shipped: the files are large, so `.gitignore`
keeps only this README as part of the folder.

## What appears here

| Path | What it is |
| --- | --- |
| `models/real_weights.bitnet1` | BITNET1 container with one real weight matrix of the model — the argument of `poc_tests --tensors` |
| `models/real_weights.bitnet1.provenance.json` | hub commit, tensor name and shape the container was exported from |
| `models/bitnet-b1.58-2B-4T-bf16/model.safetensors` | the bf16 master checkpoint (~4.8 GB) the exporter reads; reused on every later run |
| `models/*.gguf` | official packed models (llama.cpp CPU cross-check material); not consumed by this folder's tools |

## Create the container

```powershell
python tools/export_weights.py
```

The first run downloads the checkpoint into `models/` (~4.8 GB, one time),
ternarizes `model.layers.12.mlp.down_proj.weight` (2560 × 6912) exactly like
BitNet b1.58 training-time quantization, and writes
`models/real_weights.bitnet1` (~17.7 MB) plus its provenance record.
Options: `--safetensors <file>` to read a local checkpoint instead of
downloading, `--tensor <name>` for a different matrix, `--out <path>` for a
different destination. Requires Python 3.12+ with `numpy` and
`huggingface_hub` (declared in `pyproject.toml`).

## Run it

```powershell
# desktop, after pwsh scripts/build-windows.ps1
build\windows\poc_tests.exe --spv-dir build\windows\spv --tensors models\real_weights.bitnet1

# Android device, after pwsh scripts/build-android.ps1
pwsh scripts/run-on-device.ps1 -Tensors models/real_weights.bitnet1
```

The container feeds the real-weight lanes of the harness: one layer of the
model (2560 × 6912), 8 tokens, GPU output required to equal the CPU oracle
exactly like every other lane. The export itself is validated against the
reference quantizer by `tools/tests/test_export_weights.py`, which skips
when the checkpoint is not present. The container is optional: without
`--tensors` the harness runs its deterministic lanes only.
