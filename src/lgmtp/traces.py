"""Trace schema (one target sentence), compressed ``.npz`` save/load, gold-label attachment.

Index convention: index ``t`` refers to target token ``y[t]``; every draft feature at
``t`` is a function of ``q_t``, the draft distribution over ``y[t]`` computed teacher-forced on
``prompt + y[:t]``.

Storage: per-position arrays of all sentences are concatenated into flat arrays and stored with
an ``offsets`` array (sentence i occupies ``offsets[i]:offsets[i+1]``). ``meta`` is stored as a
JSON string.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

# per-position 1-D arrays saved to disk (in this order)
SCALAR_FIELDS = ("target_ids", "draft_argmax", "H", "top1", "top2", "p_target",
                 "p_space", "p_punct", "p_eos", "p_newword", "p_mark", "p_cont",
                 "target_logp_of_draft_argmax")
# per-position (n, K) arrays
TOPK_FIELDS = ("draft_topk_ids", "draft_topk_logp", "target_topk_ids", "target_topk_logp")
INT_FIELDS = {"target_ids", "draft_argmax", "draft_topk_ids", "target_topk_ids"}
TOPK = 20


@dataclass
class Trace:
    """One sentence. All arrays have first dimension ``n``."""
    id: int
    n: int
    target_ids: np.ndarray
    draft_argmax: np.ndarray
    H: np.ndarray            # draft entropy (nats)
    top1: np.ndarray         # draft prob of its argmax
    top2: np.ndarray         # draft prob of its 2nd-best token
    p_target: np.ndarray     # q_t(y[t])
    p_space: np.ndarray
    p_punct: np.ndarray
    p_eos: np.ndarray
    p_newword: np.ndarray    # p_space + p_punct + p_eos
    p_mark: np.ndarray
    p_cont: np.ndarray
    draft_topk_ids: np.ndarray
    draft_topk_logp: np.ndarray
    target_topk_ids: np.ndarray
    target_topk_logp: np.ndarray
    target_logp_of_draft_argmax: np.ndarray
    match: np.ndarray = None  # derived: draft_argmax == target_ids
    labels: dict = field(default_factory=dict, repr=False)  # gold labels (analysis only, not saved)

    def __post_init__(self):
        self.match = np.asarray(self.draft_argmax) == np.asarray(self.target_ids)
        for f in SCALAR_FIELDS + TOPK_FIELDS:
            a = getattr(self, f)
            if len(a) != self.n:
                raise ValueError(f"trace {self.id}: {f} has length {len(a)} != n={self.n}")


def save_traces(path, traces: list[Trace], meta: dict | None = None) -> None:
    """Save traces as one compressed ``.npz`` (flat arrays + offsets)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lens = np.array([t.n for t in traces], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    out = dict(ids=np.array([t.id for t in traces], dtype=np.int64), offsets=offsets,
               meta=np.array(json.dumps(meta or {}, ensure_ascii=False)))
    for f in SCALAR_FIELDS + TOPK_FIELDS:
        dt = np.int32 if f in INT_FIELDS else np.float32
        parts = [np.asarray(getattr(t, f), dtype=dt) for t in traces]
        if parts:
            out[f] = np.concatenate(parts, axis=0)
        else:
            out[f] = np.zeros((0, TOPK) if f in TOPK_FIELDS else (0,), dtype=dt)
    np.savez_compressed(path, **out)


def load_traces(path) -> tuple[list[Trace], dict]:
    """Inverse of ``save_traces``."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    offsets, ids = z["offsets"], z["ids"]
    flat = {f: z[f] for f in SCALAR_FIELDS + TOPK_FIELDS}
    traces = []
    for i, sid in enumerate(ids):
        a, b = int(offsets[i]), int(offsets[i + 1])
        traces.append(Trace(id=int(sid), n=b - a, **{f: flat[f][a:b] for f in flat}))
    return traces, meta


def summary(traces: list[Trace]) -> dict:
    """Positions, agreement rate, mean draft entropy when match / mismatch."""
    m = np.concatenate([t.match for t in traces])
    H = np.concatenate([t.H for t in traces])
    return dict(sentences=len(traces), positions=int(len(m)), agreement=float(m.mean()),
                mean_H=float(H.mean()), mean_H_match=float(H[m].mean()) if m.any() else float("nan"),
                mean_H_mismatch=float(H[~m].mean()) if (~m).any() else float("nan"))


def attach_word_labels(traces: list[Trace], tok) -> None:
    """Attach gold word/akshara labels from the *target* tokens (fixes B5). Analysis only."""
    from .boundaries import token_labels
    from .vocab import special_ids, token_bytes_list
    from .boundaries import _valid_utf8_prefix_len
    spec = special_ids(tok)
    for t in traces:
        tb = token_bytes_list(tok, t.target_ids)
        sp = [int(i) in spec for i in t.target_ids]
        t.labels.update(token_labels(tb, special=sp))
        full = b"".join(b for b, s in zip(tb, sp) if not s)
        t.labels["text"] = full[:_valid_utf8_prefix_len(full)].decode("utf-8")  # text the char spans refer to


def trace_field_names() -> list[str]:
    return [f.name for f in fields(Trace)]
