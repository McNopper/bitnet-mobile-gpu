# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Norbert Nopper

"""Export a real BitNet weight matrix into the BitNet packed-test format.

Downloads the bf16 master weights of microsoft/bitnet-b1.58-2B-4T-bf16,
ternarizes a chosen weight matrix exactly like BitNet b1.58 training-time
quantization (absmean scaling with ONE scale per tensor:
gamma = clamp(mean(|W|), min=1e-5) over the whole matrix,
q = clip(round(W / gamma), -1, 1)),
and writes a "BITNET1" container that `poc_tests --tensors <file>` consumes
for bit-exact real-data validation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import struct
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

# safetensors container parsing is implemented below (numpy has no bf16 dtype,
# and we avoid a torch dependency just to read master weights).

MAGIC = b"TETH"
VERSION = 1
DEFAULT_REPO = "microsoft/bitnet-b1.58-2B-4T-bf16"
DEFAULT_TENSOR = "model.layers.12.mlp.down_proj.weight"  # [2560, 6912]


def ternarize_absmean(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BitNet b1.58 training-time weight quantization (reference semantics).

    gamma = clamp(mean(|W|), min=1e-5) over the WHOLE tensor (arXiv
    2402.17764; the HF reference code uses w.abs().mean().clamp_(min=1e-5)
    per tensor, not per row); q = clip(round(W / gamma), -1, 1). The bf16
    master weights are stored
    as (ternary * gamma) with tiny bf16 rounding, so round() reproduces
    the exact ternary values by construction. The container keeps the
    historical per-row gamma layout (n floats, all equal) so the format
    version and every reader stay unchanged.
    """
    # HF WeightQuant clamps the mean at 1e-5, which also defines the
    # all-zero case. Clip the rounded ratio BEFORE the int8 cast so large
    # finite ratios cannot wrap around into the ternary alphabet.
    gamma_scalar = max(float(np.mean(np.abs(w))), 1e-5)
    q = np.clip(np.rint(w / gamma_scalar), -1, 1).astype(np.int8)
    gamma = np.full(w.shape[0], gamma_scalar, dtype=np.float32)
    return q, gamma


def write_container(path: Path, weights: np.ndarray, gamma: np.ndarray) -> None:
    n, k = weights.shape
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, n, k))
        f.write(weights.astype(np.int8).tobytes(order="C"))
        f.write(gamma.astype(np.float32).tobytes(order="C"))


def load_weight_matrix(path: Path, tensor_name: str) -> np.ndarray:
    """Minimal safetensors reader (header JSON + raw tensor bytes).

    Supports BF16/F16/F32. BF16 -> fp32 by left-shifting the 16-bit payload
    into the top half of an fp32 bit pattern (bf16 is truncated fp32).
    """
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        if tensor_name not in header:
            raise SystemExit(
                f"tensor '{tensor_name}' not found. hint: try a name like "
                "'model.layers.12.mlp.down_proj.weight'"
            )
        meta = header[tensor_name]
        data_start = 8 + header_len
        f.seek(data_start + meta["data_offsets"][0])
        nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
        raw = f.read(nbytes)
        if len(raw) != nbytes:
            raise SystemExit("short read on tensor data")

    dtype = meta["dtype"]
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2")
        w = (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)
    elif dtype == "F16":
        w = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "F32":
        w = np.frombuffer(raw, dtype="<f4")
    else:
        raise SystemExit(f"unsupported dtype {dtype} for {tensor_name}")

    shape = meta["shape"]
    if len(shape) != 2:
        raise SystemExit(f"expected 2D tensor, got shape {shape}")
    return np.ascontiguousarray(w.reshape(shape))


def local_download_revision(model_file: Path) -> tuple[str, str]:
    """Hub provenance of a local model copy: (commit sha, file etag).

    `hf_hub_download(..., local_dir=...)` records both next to the file
    (`.cache/huggingface/download/<name>.metadata`: line 1 = commit sha,
    line 2 = etag; for an LFS file the etag is its sha256). Returns
    ("local-file", "") when the copy carries no such record.
    """
    meta = model_file.parent / ".cache" / "huggingface" / "download" / f"{model_file.name}.metadata"
    if not meta.exists():
        return "local-file", ""
    lines = meta.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or not lines[0].strip():
        return "local-file", ""
    return lines[0].strip(), lines[1].strip()


def write_provenance(path: Path, info: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--safetensors", type=Path, default=None, help="local model.safetensors (skips download)"
    )
    parser.add_argument("--tensor", default=DEFAULT_TENSOR)
    parser.add_argument("--out", type=Path, default=Path("models/real_weights.bitnet1"))
    args = parser.parse_args()

    revision = "local-file"
    if args.safetensors is not None:
        local = args.safetensors
    else:
        # Resolution order: local models/ copy first, then download into
        # models/ (hf cache makes repeats cheap).
        repo_dir = args.repo.split("/")[-1]
        local_model = Path("models") / repo_dir / "model.safetensors"
        if local_model.exists():
            print(f"using local weights: {local_model}")
            local = local_model
        else:
            print(
                f"downloading {args.repo}/model.safetensors (~4.8 GB, one-time) "
                f"into {local_model.parent}/ ..."
            )
            # Pinned commit for the provenance record ("unpinned" when the
            # hub API returns no sha).
            revision = HfApi().model_info(args.repo).sha or "unpinned"
            local = Path(
                hf_hub_download(
                    args.repo,
                    "model.safetensors",
                    local_dir=str(local_model.parent),
                    revision=revision,
                )
            )

    # Pin the provenance to the hub commit the local copy was downloaded
    # from (recorded by huggingface_hub next to the file), when available.
    recorded_revision, file_etag = local_download_revision(local)
    if revision == "local-file":
        revision = recorded_revision

    w = load_weight_matrix(local, args.tensor)
    q, gamma = ternarize_absmean(w)

    nonzeros = float(np.mean(q != 0))
    print(f"tensor {args.tensor}: shape {w.shape}")
    print(
        f"ternary stats: non-zero fraction {nonzeros:.3f} "
        f"(BitNet b1.58 typically ~0.40-0.55), gamma mean {float(np.mean(gamma)):.6g}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_container(args.out, q, gamma)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    provenance = {
        "repo": args.repo,
        "revision": revision,
        "file": str(local),
        "file_etag_sha256": file_etag,
        "tensor": args.tensor,
        "shape": list(w.shape),
        "ternarization": (
            "gamma=clamp(mean(|W|),min=1e-5) over the whole tensor (one scale; "
            "stored as n equal per-row floats); q=clip(round(W/gamma),-1,1)"
        ),
        "nonzero_fraction": nonzeros,
        "gamma_mean": float(np.mean(gamma)),
        "container_version": VERSION,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": "weights are model data; harness activations remain synthetic",
    }
    prov_path = args.out.with_suffix(args.out.suffix + ".provenance.json")
    write_provenance(prov_path, provenance)
    print(f"wrote {prov_path}")


if __name__ == "__main__":
    main()
