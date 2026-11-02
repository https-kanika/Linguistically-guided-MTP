import numpy as np
import pytest

from lgmtp.traces import SCALAR_FIELDS, TOPK, TOPK_FIELDS, Trace


def make_trace(match, sid=0, rng=None, H=None, **over):
    """Synthetic trace: target ids 0..n-1 mod 50, draft argmax equal where match else different."""
    rng = rng or np.random.default_rng(sid)
    m = np.asarray(match, bool)
    n = len(m)
    tgt = (np.arange(n) * 7 + 3) % 50
    drf = np.where(m, tgt, (tgt + 1) % 50)
    f = dict(target_ids=tgt, draft_argmax=drf)
    for k in SCALAR_FIELDS:
        if k not in f:
            f[k] = rng.random(n).astype(np.float32)
    f["H"] = np.zeros(n, np.float32) if H is None else np.asarray(H, np.float32)
    for k in TOPK_FIELDS:
        f[k] = rng.integers(0, 50, (n, TOPK)) if k.endswith("ids") else rng.random((n, TOPK)).astype(np.float32)
    f.update(over)
    return Trace(id=sid, n=n, **f)


@pytest.fixture
def mk():
    return make_trace
