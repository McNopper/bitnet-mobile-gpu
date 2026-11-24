# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Norbert Nopper

# Unit tests for the Python tooling (pytest). Kernel-vs-oracle exactness is
# proven per device by the C++ GPU harness (poc_tests); these tests cover the
# Python-side logic in isolation.

import struct
from pathlib import Path

import numpy as np
import pytest

from export_weights import (
    load_weight_matrix,
    local_download_revision,
    ternarize_absmean,
    write_container,
)


def test_local_download_revision(tmp_path: Path) -> None:
    # huggingface_hub records commit sha + etag next to a local_dir download;
    # a copy without that record stays "local-file".
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"")
    assert local_download_revision(model) == ("local-file", "")
    meta_dir = tmp_path / ".cache" / "huggingface" / "download"
    meta_dir.mkdir(parents=True)
    (meta_dir / "model.safetensors.metadata").write_text(
        "abc123\nfeedbeef\n1.0\n", encoding="utf-8"
    )
    assert local_download_revision(model) == ("abc123", "feedbeef")


def test_ternarize_absmean_known() -> None:
    # Per-TENSOR gamma (BitNet b1.58 reference semantics): gamma =
    # mean(|W|) over the whole matrix = (2+0+2+1+1+1)/6 = 7/6;
    # round(2/(7/6)) = round(1.714) = 2 -> clip 1; round(1/(7/6)) = 1.
    w = np.array([[2.0, 0.0, -2.0], [1.0, 1.0, -1.0]], dtype=np.float32)
    q, gamma = ternarize_absmean(w)
    assert q.dtype == np.int8
    assert q.shape == (2, 3)
    assert q[0].tolist() == [1, 0, -1]
    assert q[1].tolist() == [1, 1, -1]
    assert bool(np.all(gamma == pytest.approx(7.0 / 6.0, abs=1e-6)))  # all-equal layout
    assert gamma.shape == (2,)


def test_ternarize_absmean_clips_outliers() -> None:
    # Values beyond +/-gamma clip to +/-1 (documented approximation for
    # post-hoc ternarization of latent weights).
    w = np.array([[10.0, 0.0, 0.0, -10.0]], dtype=np.float32)
    q, _ = ternarize_absmean(w)
    assert q.tolist() == [[1, 0, 0, -1]]


def test_ternarize_absmean_range() -> None:
    rng = np.random.default_rng(1)
    w = rng.standard_normal((16, 64)).astype(np.float32) * 3.0
    q, gamma = ternarize_absmean(w)
    assert set(np.unique(q)).issubset({-1, 0, 1})
    assert gamma.shape == (16,)
    assert bool(np.all(gamma == gamma[0]))  # per-tensor: one scalar, n-equal layout


def test_ternarize_clips_before_int8_cast() -> None:
    # gamma = 1000/256 = 3.90625, so the rounded ratio is 256. Casting
    # before clipping wraps that to int8 zero; the reference clips to +1.
    w = np.zeros((1, 256), dtype=np.float32)
    w[0, 0] = 1000.0
    q, _ = ternarize_absmean(w)
    assert q[0, 0] == 1


def test_ternarize_all_zero_uses_reference_floor() -> None:
    w = np.zeros((2, 4), dtype=np.float32)
    q, gamma = ternarize_absmean(w)
    assert np.array_equal(q, np.zeros_like(w, dtype=np.int8))
    assert np.all(gamma == np.float32(1e-5))


REAL_CHECKPOINT = Path("models") / "bitnet-b1.58-2B-4T-bf16" / "model.safetensors"


@pytest.mark.skipif(not REAL_CHECKPOINT.exists(), reason="local checkpoint not downloaded")
def test_ternarize_matches_reference_quantizer_on_real_slice() -> None:
    # The export equals the HF reference quantizer
    # (per-tensor absmean, w.abs().mean()) on a slice of the REAL tensor.
    w = load_weight_matrix(REAL_CHECKPOINT, "model.layers.12.mlp.down_proj.weight")
    ref_gamma = float(np.mean(np.abs(w)))  # the reference line, verbatim
    ref_q = np.clip(np.rint(w / ref_gamma), -1, 1).astype(np.int8)
    q, gamma = ternarize_absmean(w)
    assert float(gamma[0]) == pytest.approx(ref_gamma, abs=1e-6)
    assert np.array_equal(q, ref_q)
    assert bool(np.all(gamma > 0))


def test_container_roundtrip(tmp_path: Path) -> None:
    weights = np.array([[1, 0, -1], [0, 1, 1]], dtype=np.int8)
    gamma = np.array([0.25, 1.5], dtype=np.float32)
    out = tmp_path / "test.bitnet1"
    write_container(out, weights, gamma)

    raw = out.read_bytes()
    assert raw[:4] == b"TETH"
    version, n, k = struct.unpack("<III", raw[4:16])
    assert version == 1
    assert (n, k) == (2, 3)
    data_start = 16
    weights_back = np.frombuffer(raw[data_start : data_start + n * k], dtype=np.int8).reshape(n, k)
    off = data_start + n * k
    gamma_back = np.frombuffer(raw[off : off + n * 4], dtype="<f4")
    assert weights_back.tolist() == weights.tolist()
    assert gamma_back.tolist() == [0.25, 1.5]


def test_container_shape_is_recorded(tmp_path: Path) -> None:
    weights = np.zeros((3, 5), dtype=np.int8)
    gamma = np.ones(3, dtype=np.float32)
    out = tmp_path / "shape.bitnet1"
    write_container(out, weights, gamma)
    raw = out.read_bytes()
    _, n, k = struct.unpack("<III", raw[4:16])
    assert (n, k) == (3, 5)
    assert len(raw) == 16 + n * k + n * 4
