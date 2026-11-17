import json

import numpy as np

from lgmtp.collect import load_records, save_topk


def test_load_records_reads_arrays_once(tmp_path):
    """Week-1 Kaggle OOM: per-record NpzFile access kept one full array copy per record alive."""
    rng = np.random.default_rng(0)
    recs, tk = [], []
    for i in range(50):
        n = int(rng.integers(1, 30))
        recs.append(dict(id=i, out_ids=rng.integers(0, 100, n).tolist(), out_text="x"))
        tk.append((rng.integers(0, 100, (n, 20)).astype(np.int32), rng.normal(size=(n, 20)).astype(np.float16)))
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(r) for r in recs[::-1]), encoding="utf-8")   # order differs
    save_topk(tmp_path / "s_topk.npz", recs, tk)
    out = load_records(tmp_path / "s.jsonl", tmp_path / "s_topk.npz")
    assert len(out) == 50
    by = {r["id"]: r for r in out}
    for i, (a, b) in enumerate(tk):
        assert (by[i]["topk_ids"] == a).all() and np.allclose(by[i]["topk_logp"], b.astype(np.float32))
    bases = {id(r["topk_ids"].base) for r in out}
    assert len(bases) == 1, "every record must view one shared array, not its own full copy"
