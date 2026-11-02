import numpy as np
import pytest

from lgmtp.traces import SCALAR_FIELDS, TOPK, TOPK_FIELDS, Trace, load_traces, save_traces, summary


def make_trace(i, n, rng):
    f = {}
    for k in SCALAR_FIELDS:
        f[k] = rng.integers(0, 50, n) if k in ("target_ids", "draft_argmax") else rng.random(n).astype(np.float32)
    for k in TOPK_FIELDS:
        f[k] = rng.integers(0, 50, (n, TOPK)) if k.endswith("ids") else rng.random((n, TOPK)).astype(np.float32)
    return Trace(id=i, n=n, **f)


def test_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    tr = [make_trace(i, n, rng) for i, n in enumerate([5, 1, 9, 3])]
    save_traces(tmp_path / "t.npz", tr, {"split": "dev", "hindi": "नमस्ते"})
    back, meta = load_traces(tmp_path / "t.npz")
    assert meta == {"split": "dev", "hindi": "नमस्ते"}
    assert [t.id for t in back] == [0, 1, 2, 3] and [t.n for t in back] == [5, 1, 9, 3]
    for a, b in zip(tr, back):
        for k in SCALAR_FIELDS + TOPK_FIELDS:
            np.testing.assert_allclose(getattr(a, k), getattr(b, k), rtol=1e-6)
        assert np.array_equal(a.match, b.match)
        assert b.draft_topk_ids.shape == (b.n, TOPK)


def test_match_derived_and_length_check():
    rng = np.random.default_rng(1)
    t = make_trace(0, 4, rng)
    t2 = Trace(**{**{k: getattr(t, k) for k in SCALAR_FIELDS + TOPK_FIELDS}, "id": 0, "n": 4,
                  "draft_argmax": t.target_ids.copy()})
    assert t2.match.all()
    with pytest.raises(ValueError):
        Trace(**{**{k: getattr(t, k) for k in SCALAR_FIELDS + TOPK_FIELDS}, "id": 0, "n": 5})


def test_summary():
    rng = np.random.default_rng(2)
    t = make_trace(0, 6, rng)
    t.match = np.array([1, 1, 0, 1, 0, 0], bool)
    t.H = np.array([0, 0, 2, 0, 2, 2], np.float32)
    S = summary([t])
    assert S["agreement"] == 0.5 and S["mean_H_match"] == 0 and S["mean_H_mismatch"] == 2
