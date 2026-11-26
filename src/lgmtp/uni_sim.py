"""Exact trace simulation of the universal head on any target (W1), plus training-free baselines.

For greedy decoding the head's draft at committed position p depends only on out_ids[:p] and q_t for
t < p, which are the same however p was reached, so drafts can be precomputed for every position and
the round-by-round loop below is exact up to the cost model (same argument as Phase 1).

Round at p: draft g tokens (the head's units converted to the *target's* tokens, see ``convert``),
accept the leading matches a, emit min(a+1, n-p) tokens, cost = c_head + verify(g).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import bootstrap_ci, paired_bootstrap
from .uni_features import TokTable, position_targets, seq_text, utf8_offsets
from .unichar import UnitVocab, segment_units


# --------------------------------------------------------------------------- #
# cost model
# --------------------------------------------------------------------------- #

class Verify:
    """verify(g) in target-step units from a Phase-1 style timing dict; linear extrapolation past the
    largest measured g (np.interp would flatten it and flatter long drafts)."""

    def __init__(self, timing: dict):
        tt = timing["t_target"]
        gs = sorted(int(g) for g in timing["t_target_verify"])
        vs = [max(1.0, timing["t_target_verify"][str(g)] / tt) for g in gs]
        self.gs, self.vs = gs, vs
        tail = [(g, v) for g, v in zip(gs, vs) if g >= max(gs) - 4]
        self.slope = max(0.0, float(np.polyfit([g for g, _ in tail], [v for _, v in tail], 1)[0])) if len(tail) > 1 else 0.0

    def __call__(self, g: int) -> float:
        if g <= 0:
            return 1.0
        if g <= self.gs[-1]:
            return float(np.interp(g, self.gs, self.vs))
        return self.vs[-1] + self.slope * (g - self.gs[-1])

    def as_dict(self):
        return {str(g): round(self(g), 4) for g in range(0, 17)}


# --------------------------------------------------------------------------- #
# sequences of one target
# --------------------------------------------------------------------------- #

@dataclass
class SimSeq:
    id: int
    out_ids: np.ndarray
    n: int
    b_tok: np.ndarray
    out_bytes: bytes
    char_start: np.ndarray     # bool per byte offset (0..len): is a char boundary
    ps_byte: np.ndarray        # partial-unit start byte at each position p
    pend: np.ndarray
    eos_id: int | None
    prompt_ids: list | None
    topk_ids: np.ndarray | None


def build_simseqs(records: list[dict], table: TokTable, vocab: UnitVocab) -> list[SimSeq]:
    out = []
    for r in records:
        ids = np.asarray(r["out_ids"], np.int64)
        st = seq_text(table, ids)
        ps_b, _, _, _ = position_targets(st, vocab, bool(r.get("eos_end", True)))
        cs = np.zeros(len(st.out_bytes) + 1, bool)
        cs[st.coff] = True
        eos = int(ids[-1]) if len(ids) and table.is_eos[int(ids[-1])] else (table.eos_ids[0] if table.eos_ids else None)
        out.append(SimSeq(int(r["id"]), ids, len(ids), st.b_tok, st.out_bytes, cs, ps_b, st.pend, eos,
                          r.get("prompt_ids"), np.asarray(r["topk_ids"]) if r.get("topk_ids") is not None else None))
    return out


def draft_levels(vocab: UnitVocab, w_row, partial_len: int) -> list[tuple[bytes, bool]]:
    """Draft bytes for levels 1..k (level j = first j units); stops at <eos> (flag) or a missing unit."""
    levels, acc = [], b""
    for i, w in enumerate(w_row):
        w = int(w)
        if w < 0:
            break
        if w == vocab.eos_id:
            levels.append((acc, True))
            break
        ub = vocab.bytes[w]
        acc = acc + (ub[partial_len:] if i == 0 else ub)
        levels.append((acc, False))
    return levels


def _anchor(seq: SimSeq, p: int) -> int:
    """Largest token index q <= p that starts at or before the partial unit, on a char boundary."""
    q = p
    while q > 0 and (seq.b_tok[q] > seq.ps_byte[p] or not seq.char_start[seq.b_tok[q]]):
        q -= 1
    return q


def _level_prefixes(toks: list, tok_bytes, levels: list, eos_id) -> list:
    """Level j = the longest prefix of ``toks`` (tokenization of the longest draft) whose bytes end
    inside level j's text. A shorter level therefore never ends with a token that the following
    predicted unit would have merged into (Llama-3 glues a word-final vowel sign to the punctuation
    after it: "ी।" is one token). Only the longest level can end on such a token."""
    cum, c = [], 0
    for t in toks:
        c += len(tok_bytes[t]) if t < len(tok_bytes) else 0
        cum.append(c)
    out = []
    for db, eos in levels:
        B = len(db)
        n = 0
        while n < len(cum) and cum[n] <= B:
            n += 1
        d = list(toks[:n])
        if eos and eos_id is not None:
            d.append(eos_id)
        out.append(d)
    return out


def convert_all(seqs: list[SimSeq], W: np.ndarray, vocab: UnitVocab, encode, tok_bytes, gmax: int = 16):
    """Target-token drafts for every position and level. ``tok_bytes`` = ``TokTable.bytes``.

    Per position: retokenise (committed text from the start of the current word) + (all predicted
    units) once with the target tokenizer; if that reproduces the committed tokens, the remaining
    tokens are the draft (``anchored``); otherwise, when the committed text ends on a char boundary,
    the draft text is tokenised on its own (``fallback``, non-canonical); else no draft (``failed``).
    Returns (drafts, stats) with drafts[s][p] = list over levels 0..L of token-id lists (level 0 = [])."""
    jobs, meta = [], []
    off = 0
    for s, seq in enumerate(seqs):
        for p in range(seq.n):
            plen = int(seq.b_tok[p] - seq.ps_byte[p])
            levels = draft_levels(vocab, W[off + p], plen)
            if levels and levels[-1][0]:
                q = _anchor(seq, p)
                try:
                    txt = (seq.out_bytes[seq.b_tok[q]:seq.b_tok[p]] + levels[-1][0]).decode("utf-8")
                except UnicodeDecodeError:
                    txt = None
                meta.append((s, p, q, levels, txt))
                jobs.append(txt if txt is not None else "")
            elif levels:                                   # only <eos> / empty completions
                drafts_e = [([seq.eos_id] if eos and seq.eos_id is not None else []) for _, eos in levels]
                meta.append((s, p, None, levels, drafts_e))
        off += seq.n
    enc = iter(encode(jobs) if jobs else [])
    drafts = [[[[]] for _ in range(seq.n)] for seq in seqs]
    stats = dict(positions=0, levels=0, anchored=0, fallback=0, failed=0, empty=0)
    fb_jobs, fb_meta = [], []
    for s, p, q, levels, txt in meta:
        seq = seqs[s]
        stats["positions"] += 1
        stats["levels"] += len(levels)
        if q is None:
            drafts[s][p] = [[]] + [d[:gmax] for d in txt]
            stats["empty"] += 1
            continue
        ids = next(enc)
        if txt is not None and list(ids[:p - q]) == seq.out_ids[q:p].tolist():
            drafts[s][p] = [[]] + [d[:gmax] for d in _level_prefixes(list(ids[p - q:]), tok_bytes, levels, seq.eos_id)]
            stats["anchored"] += 1
            continue
        stats["failed"] += 1
        drafts[s][p] = [[] for _ in range(len(levels) + 1)]
        if seq.pend[p] == 0:
            try:
                fb_jobs.append(levels[-1][0].decode("utf-8"))
                fb_meta.append((s, p, levels))
            except UnicodeDecodeError:
                pass
    for (s, p, levels), ids in zip(fb_meta, encode(fb_jobs) if fb_jobs else []):
        seq = seqs[s]
        drafts[s][p] = [[]] + [d[:gmax] for d in _level_prefixes(list(ids), tok_bytes, levels, seq.eos_id)]
        stats["fallback"] += 1
        stats["failed"] -= 1
    return drafts, stats


# --------------------------------------------------------------------------- #
# simulation
# --------------------------------------------------------------------------- #

def choose_level(probs, n_levels: int, tau: float) -> int:
    """Largest level j <= n_levels whose cumulative unit probability is >= tau (0 if none)."""
    c, best = 1.0, 0
    for j in range(n_levels):
        c *= float(probs[j])
        if c >= tau:
            best = j + 1
        else:
            break
    return best


def simulate(seqs: list[SimSeq], drafts, P: np.ndarray, tau: float, c_head: float, verify, gmax: int = 16,
             cap_to_end: bool = True, oracle_level: bool = False) -> dict:
    ns = len(seqs)
    em, co = np.zeros(ns), np.zeros(ns)
    R = dict(rounds=0, drafted=0, accepted=0)
    off = 0
    for s, seq in enumerate(seqs):
        p = 0
        y = seq.out_ids
        while p < seq.n:
            lv_all = drafts[s][p]
            if oracle_level:
                best = (-1, 0)
                for lv in range(len(lv_all)):
                    d = lv_all[lv][: (seq.n - p) if cap_to_end else gmax]
                    a = _accept(d, y, p)
                    if a > best[0]:
                        best = (a, lv)
                lv = best[1]
            else:
                lv = choose_level(P[off + p], len(lv_all) - 1, tau)
            d = lv_all[lv][:gmax]
            if cap_to_end:
                d = d[: seq.n - p]
            a = _accept(d, y, p)
            gained = min(a + 1, seq.n - p)
            em[s] += gained
            co[s] += c_head + verify(len(d))
            R["rounds"] += 1; R["drafted"] += len(d); R["accepted"] += a
            p += gained
        off += seq.n
    R.update(sent_emitted=em, sent_cost=co, speedup=float(em.sum() / co.sum()),
             tokens_per_round=float(em.sum() / max(R["rounds"], 1)),
             mean_draft_len=R["drafted"] / max(R["rounds"], 1),
             acceptance=R["accepted"] / max(R["drafted"], 1))
    return R


def _accept(d, y, p) -> int:
    a = 0
    for i, t in enumerate(d):
        if p + i < len(y) and int(y[p + i]) == int(t):
            a += 1
        else:
            break
    return a


def tune_tau(seqs, drafts, P, taus, c_head, verify, gmax=16) -> tuple[float, list]:
    rows = [(float(simulate(seqs, drafts, P, t, c_head, verify, gmax)["speedup"]), float(t)) for t in taus]
    best = max(rows)[1]
    return best, [dict(tau=t, speedup=s) for s, t in rows]


def summary_row(name: str, R: dict, n_boot: int = 1000, seed: int = 0, ref: dict | None = None, **extra) -> dict:
    ci = bootstrap_ci(R["sent_emitted"], R["sent_cost"], n_boot, seed)
    row = dict(method=name, speedup=ci["point"], ci_lo=ci["lo"], ci_hi=ci["hi"],
               tokens_per_round=R["tokens_per_round"], mean_draft_len=R["mean_draft_len"],
               acceptance=R["acceptance"], rounds=R["rounds"], **extra)
    if ref is not None:
        pb = paired_bootstrap(R["sent_emitted"], R["sent_cost"], ref["sent_emitted"], ref["sent_cost"], n_boot, seed)
        row.update(diff_vs_ref=pb["diff"], diff_lo=pb["lo"], diff_hi=pb["hi"], p_better=pb["p_better"])
    return row


# --------------------------------------------------------------------------- #
# training-free baselines (token level, draft cost ~0)
# --------------------------------------------------------------------------- #

def _lookup(ctx: list, key: tuple, max_len: int) -> list:
    """Continuation after the most recent earlier occurrence of ``key`` in ``ctx`` (excluding the end)."""
    k = len(key)
    for i in range(len(ctx) - k - 1, -1, -1):
        if tuple(ctx[i:i + k]) == key:
            return ctx[i + k:i + k + max_len]
    return []


class _NgramIndex:
    """Latest start of every n-gram of a growing context that has at least one following token
    (same answer as ``_lookup``, in O(1) per query)."""

    def __init__(self, ctx: list, ns):
        self.ctx, self.ns = list(ctx), tuple(ns)
        self.idx = {n: {} for n in self.ns}
        for n in self.ns:
            for i in range(len(self.ctx) - n):
                self.idx[n][tuple(self.ctx[i:i + n])] = i

    def lookup(self, key: tuple, max_len: int) -> list:
        i = self.idx[len(key)].get(tuple(key))
        return [] if i is None else self.ctx[i + len(key):i + len(key) + max_len]

    def append(self, t) -> None:
        for n in self.ns:
            if len(self.ctx) >= n:
                self.idx[n][tuple(self.ctx[-n:])] = len(self.ctx) - n
        self.ctx.append(t)


def pld_drafts(seqs: list[SimSeq], max_len: int = 8, ngram: tuple = (3, 2, 1)):
    """Prompt-lookup decoding: match the last n committed tokens in prompt + committed output."""
    out = []
    for seq in seqs:
        ix = _NgramIndex(list(seq.prompt_ids or []), ngram)
        rows = []
        for t in seq.out_ids.tolist():
            d = []
            for n in ngram:
                if len(ix.ctx) > n:
                    d = ix.lookup(tuple(ix.ctx[-n:]), max_len)
                    if d:
                        break
            rows.append([[], d])
            ix.append(t)
        out.append(rows)
    return out


def logitspec_drafts(seqs: list[SimSeq], max_len: int = 8):
    """LogitSpec-style chain: guess y[p] as the best candidate of q_{p-1} other than the committed token,
    then extend by looking up (y[p-1], guess) in prompt + committed output."""
    out = []
    for seq in seqs:
        y = seq.out_ids.tolist()
        ix = _NgramIndex(list(seq.prompt_ids or []), (2,))
        if y:
            ix.append(y[0])
        rows = [[[], []]]
        for p in range(1, seq.n):
            cands = [int(v) for v in seq.topk_ids[p - 1] if int(v) != y[p - 1]]
            if cands:
                g = cands[0]
                rows.append([[], [g] + ix.lookup((y[p - 1], g), max_len - 1)])
            else:
                rows.append([[], []])
            ix.append(y[p])
        out.append(rows)
    return out


def next_next_coverage(seqs: list[SimSeq], ks=(1, 2, 5, 10, 20)) -> dict:
    """Fraction of positions p >= 1 where y[p] is among the top-k of q_{p-1} (LogitSpec's observation)."""
    hits = {k: 0 for k in ks}
    n = 0
    for seq in seqs:
        y = seq.out_ids
        for p in range(1, seq.n):
            row = seq.topk_ids[p - 1]
            for k in ks:
                hits[k] += int(y[p]) in set(int(v) for v in row[:k])
            n += 1
    return {f"top{k}": hits[k] / max(n, 1) for k in ks} | dict(positions=n)


def ones_like_probs(drafts) -> np.ndarray:
    n = sum(len(r) for r in drafts)
    return np.ones((n, 1), np.float32)


# --------------------------------------------------------------------------- #
# cross-LLM agreement on word units (teacher forced)
# --------------------------------------------------------------------------- #

def unit_agreement(text: str, ids: list, argmax: list, table: TokTable) -> dict:
    """Model B teacher-forced on ``text`` (tokenised by B as ``ids``; ``argmax[t]`` = B's greedy token
    at t). Counts per unit: first token correct (unit start), later tokens correct (continuation),
    whole unit correct, whole unit correct given its start was correct."""
    bl = table.seq_bytes(ids)
    starts = np.zeros(len(bl) + 1, np.int64)
    starts[1:] = np.cumsum([len(b) for b in bl])
    _, coff = utf8_offsets(text.encode("utf-8"))
    spans = segment_units(text)
    ubytes = [(int(coff[a]), int(coff[b])) for a, b in spans]
    ok = [int(a) == int(t) for a, t in zip(argmax, ids)]
    tok_unit = np.searchsorted([u[0] for u in ubytes], starts[:-1], side="right") - 1
    res = dict(units=0, start_ok=0, cont_tokens=0, cont_ok=0, unit_ok=0, unit_ok_given_start=0, start_ok_units=0)
    for u in range(len(ubytes)):
        toks = np.flatnonzero(tok_unit == u)
        if not len(toks):
            continue          # unit inside a token that started earlier (merged across units)
        res["units"] += 1
        s_ok = ok[toks[0]]
        all_ok = all(ok[t] for t in toks)
        res["start_ok"] += s_ok
        res["cont_tokens"] += len(toks) - 1
        res["cont_ok"] += sum(ok[t] for t in toks[1:])
        res["unit_ok"] += all_ok
        if s_ok:
            res["start_ok_units"] += 1
            res["unit_ok_given_start"] += all_ok
    return res


def agreement_table(counts: dict) -> dict:
    c = counts
    return dict(units=c["units"], p_start=c["start_ok"] / max(c["units"], 1),
                p_cont_token=c["cont_ok"] / max(c["cont_tokens"], 1),
                p_unit=c["unit_ok"] / max(c["units"], 1),
                p_unit_given_start=c["unit_ok_given_start"] / max(c["start_ok_units"], 1))
