# data/ — sample vector, model budget, real-model validation

| File | What it is |
| --- | --- |
| `golden_vector.txt` | **sample input and expected output** of the test harness: a hand-computed known-answer vector every kernel is checked against |
| `model-budget.txt` | parameter, storage and bytes-per-token budget of the model |
| `README.md` | this file: how validation works against real model weights |

## Real-model testing

How the PoC is validated against an actual trained BitNet model, beyond
random tensors.

## What exists on Hugging Face

| Repo | Content | Used here for |
| --- | --- | --- |
| [`microsoft/bitnet-b1.58-2B-4T-bf16`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16) | bf16 master weights (`model.safetensors`, 4,825,679,400 B) | **ground-truth weights - the test source** (local copy pinned to hub commit `276681394656abdadb8e80e5b2c3db5e5d7fcaff`, file sha256 `529637ff…5ae1`, verified against the hub etag) |
| [`microsoft/bitnet-b1.58-2B-4T-gguf`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) | `ggml-model-i2_s.gguf` (I2_S, bitnet.cpp format) | not used: I2_S is not an upstream llama.cpp type (issue #12997) |
| [`microsoft/bitnet-b1.58-2B-4T`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) | transformers checkpoint | reference implementation of the quantizers (`transformers/integrations/bitnet.py`) |

## Ecosystem facts the tests rely on

1. **Upstream llama.cpp has ternary types and float-activation GPU
   kernels** (verified at `e0949d09b130`, 2026-11-12): `TQ1_0`/`TQ2_0`
   (PR #8151), the BitNet architecture (PR #7931), Vulkan
   `mul_mat_vec_tq{1,2}_0.comp` and Metal `TQ2_0` kernels.
2. **bitnet.cpp's GPU kernel is CUDA-only** (`gpu/` of `microsoft/BitNet`,
   <https://github.com/microsoft/BitNet/tree/0b341e582afbf9e1011f24744b554c96a3477eb5/gpu>);
   the mobile GPU paths in llama.cpp itself are Vulkan and Metal.
3. **Reference quantizers** (Hugging Face `integrations/bitnet.py`):
   weights `scale = 1/mean(|W|)` over the whole tensor,
   `round().clamp(-1, 1)`; activations `scale = 127/max(|x|)` per token
   vector, `round().clamp(-128, 127)`, `torch.round` (half to even).

## Test path: bf16 master weights → BITNET1 container

`tools/export_weights.py` reads one weight matrix from the local
safetensors file, ternarizes it exactly like the reference quantizer
(γ = mean(|W|) over the WHOLE tensor, q = clip(round(W/γ), -1, 1)) and
writes a `BITNET1` container (magic, n, k, int8 ternary weights, fp32 γ
stored as n equal per-row floats: the container has one scale slot per
row).

- Source tensor: `model.layers.12.mlp.down_proj.weight`, shape
  2560 × 6912 (n = 2560 outputs, k = 6912 inputs) - the largest-K
  projection of a layer.
- Measured properties, recorded in the provenance sidecar written next
  to the container by `tools/export_weights.py`: non-zero fraction
  0.5785, γ = 2.4586.
- `pytest` (`tools/tests/test_export_weights.py`) checks that the export
  equals the reference formula on the real tensor element for element
  (the test needs the local checkpoint and is skipped without it).

`poc_tests --tensors <file>` then runs two tests on these weights:

1. **Integer kernel test** - shape m = 8 × 2560 × 6912 with seeded int8
   activations through every core kernel; GPU output must equal the CPU
   oracle exactly.
2. **BitLinear forward test** (`[bitlinear]`) - float activations in,
   float outputs out: per-token absmax quantization with the reference
   formula (127/max|x| with the 1e-5 floor, round half to even, clamp to
   [-128, 127]) → integer kernel → epilogue y = acc · γ / s. The epilogue
   is one shared function applied to the GPU and the CPU accumulators, so
   the 8 × 2560 = 20,480 float outputs are compared with exact equality.

## Scope of the claim

**Only the weights are model data.** Activations are synthetic (seeded
int8 in the kernel test, a deterministic float pattern in [-1, 1] in the
BitLinear test). The tests therefore establish: the kernels are exact on a real
ternary weight distribution, and the quantize → multiply → rescale chain
of one BitLinear layer is exact given its input. They do not establish
anything about real activation statistics or end-to-end model quality;
that is the perplexity/KL gate of the integration plan (report §6).

Local model copies live in the git-ignored `models/` folder.
`tools/export_weights.py` reads
`models/bitnet-b1.58-2B-4T-bf16/model.safetensors` and downloads the file
(4.8 GB) when it is missing.

## Acceptance

- [x] BITNET1 container built from the real layer; provenance sidecar
      pins hub commit and file hash; container bytes are reproducible
      (sha256 `48c1b712…581a` across regenerations).
- [x] `poc_tests --tensors`: both tests pass on the RTX 5070 (validation
      layer ON, 0 messages) and on Mali-G57, Mali-G52 and Adreno 740.
- [x] End-to-end model load: the `TQ2_0` GGUF converted from the same
      checkpoint loads and runs in llama.cpp on the CPUs of the Mali-G57
      tablet and of the Quest 3; its
      perplexity against the reference implementation is in report §2.2.
