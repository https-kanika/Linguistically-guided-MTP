"""Exact teacher-forced simulation of the "token-tax refund" drafters (PLAN_wordheads.md §3).

Same protocol as Phase 1: greedy, lossless speculative decoding of the 8B outputs stored in the
Phase-1 traces. A round at committed position ``p`` drafts ``g`` tokens, the 8B accepts the leading
matches ``a`` and emits ``min(a + 1, n - p)`` tokens; the round costs drafting + ``verify(g)``, all in
units of one 8B decode step. Speed-up = emitted tokens / cost (ratio of sums over sentences).

Drafters (``mode``):
  entropy   the 1B drafts every token, stop rule ``1 - sqrt(beta H) < lam`` (Phase 1).
  freecont  as ``entropy`` but drafted word-continuation tokens cost ``kappa * c`` (ceiling).
  noisy     as ``freecont`` with each continuation token kept right only with probability ``rho``.
  dict      the 1B drafts word starts; a frequency lexicon completes the word (optionally gated by
            the top word's share of the matching words' frequency, ``tau``).
  heads     the 1B drafts a token; Medusa heads on the same hidden state propose up to K following
            tokens (stop at a word-initial proposal, at log-prob < log ``tau_h``, or for V1/V2 at
            continuation probability < ``tau_c``); the 1B then reads them in one call.

Approximation shared with Phase 1: stopping decisions after the first wrong draft token use the
teacher-forced (true-path) features. Accepted tokens are exact.
"""
from __future__ import annotations

import bisect
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .policies import entropy_fires

GMAX = 8
BETA = 4.0
LAM_GRID = (-1.3, -1.1, -0.9, -0.8, -0.683, -0.6, -0.5, -0.4, -0.3)
VARIANTS = ("v0", "v1", "v2")
DEV_RE = re.compile(r"[ऀ-ॿ‌‍]+")


# --------------------------------------------------------------------------- cost model
@dataclass
class Cost:
    c: float                       # one 1B call / one 8B decode step
    verify: object                 # verify(g) -> cost of the 8B pass over g drafted tokens
    r_q: dict | None = None        # 1B call reading q new tokens, relative to q = 1 (measured)
    head_rel_cv: float = 0.0177    # all K heads, continuation vocabulary, relative to a 1B call
    head_rel_full: float = 0.114   # all K heads, full vocabulary

    def r(self, q: int) -> float:
        """Cost of a 1B call that reads q new tokens, relative to q = 1. Measured values below 1 are
        timing noise and are clipped to 1; beyond the measured range the last value is used."""
        if q <= 1 or not self.r_q:
            return 1.0
        ks = sorted(self.r_q)
        k = max([x for x in ks if x <= q] or [ks[0]])
        return max(1.0, float(self.r_q[k]))


def load_cost(timing_path, heads_run_info=None) -> Cost:
    from .simulator import load_timing, make_verify
    timing = load_timing(timing_path)
    cost = Cost(c=float(timing["c"]), verify=make_verify(timing))
    if heads_run_info:
        tm = json.loads(Path(heads_run_info).read_text())["timing"]
        cost.r_q = {int(k): float(v) for k, v in tm["r_q"].items()}
        cost.head_rel_cv = float(tm["head_rel_cv"])
        cost.head_rel_full = float(tm["head_rel_full"])
    return cost


# --------------------------------------------------------------------------- data
@dataclass
class Heads:
    """Head predictions of one variant for one sentence (rows aligned with the trace positions)."""
    ids: np.ndarray       # [n, K, TK] full-vocabulary top-k
    logp: np.ndarray
    cv_id: np.ndarray     # [n, K] argmax within the continuation vocabulary (renormalised log-prob)
    cv_logp: np.ndarray
    cont: np.ndarray      # [n, K] P(y[t+1..t+k] all continue the word of y[t])


def attach_heads(traces, npz_path, check_agreement: float = 0.97) -> dict:
    """Attach ``t.heads[v]`` to every trace from ``heads_{split}.npz``; returns alignment info."""
    z = dict(np.load(npz_path))        # decompress every array once (NpzFile re-reads per access)
    off, ids = z["offsets"], z["ids"]
    by_id = {int(s): i for i, s in enumerate(ids)}
    agree = tot = 0
    vmax = int(max(z["cont_vocab"].max(), max(int(t.target_ids.max()) for t in traces),
                   int(z["draft_argmax"].max()), int(z["v0_ids"].max()))) + 1
    in_cv = np.zeros(vmax, bool)
    in_cv[z["cont_vocab"]] = True
    for t in traces:
        t._in_cv = in_cv
        i = by_id[t.id]
        a, b = int(off[i]), int(off[i + 1])
        if b - a != t.n:
            raise ValueError(f"sentence {t.id}: heads have {b - a} rows, trace has {t.n}")
        agree += int((z["draft_argmax"][a:b] == t.draft_argmax).sum())
        tot += t.n
        t.heads = {v: Heads(z[f"{v}_ids"][a:b], z[f"{v}_logp"][a:b].astype(np.float32),
                            z[f"{v}_cv_id"][a:b], z[f"{v}_cv_logp"][a:b].astype(np.float32),
                            z[f"{v}_cont"][a:b].astype(np.float32)) for v in VARIANTS}
    rate = agree / max(tot, 1)
    if rate < check_agreement:
        raise ValueError(f"heads rows misaligned with traces: draft argmax agreement {rate:.4f}")
    return dict(agreement=rate, positions=tot, cont_vocab=z["cont_vocab"])


def devanagari_mask(tok, vocab_size: int | None = None) -> np.ndarray:
    """mask[id] = token text contains a Devanagari character or is a bare UTF-8 fragment
    (fragments of Llama-3 Hindi tokens are pieces of Devanagari characters)."""
    from .vocab import token_bytes_list
    V = vocab_size or len(tok)
    out = np.zeros(V, bool)
    for i, b in enumerate(token_bytes_list(tok, range(V))):
        s = b.decode("utf-8", errors="ignore")
        out[i] = bool(DEV_RE.search(s)) or (len(b) > 0 and not s)
    return out


# --------------------------------------------------------------------------- lexicon (dictionary drafter)
class Lexicon:
    """Word frequencies (dev outputs + word list). ``complete`` returns the continuation tokens of the
    most frequent word consistent with the current word prefix ([] = no proposal)."""

    def __init__(self, freq: Counter, tok):
        self.freq, self.tok = freq, tok
        self.words = sorted(freq)
        self._wt, self._cache = {}, {}

    @classmethod
    def build(cls, tok, output_jsonl, wordlist_csv=None):
        freq = Counter()
        for line in open(output_jsonl, encoding="utf-8"):
            freq.update(DEV_RE.findall(json.loads(line)["out_text"]))
        if wordlist_csv and Path(wordlist_csv).exists():
            for i, line in enumerate(open(wordlist_csv, encoding="utf-8")):
                w = line.split(",")[0].strip()
                if i and w and w not in freq and DEV_RE.fullmatch(w):
                    freq[w] = 1
        return cls(freq, tok)

    def tbytes(self, i: int) -> bytes:
        tb = self.__dict__.setdefault("_tb", {})
        if i not in tb:
            from .vocab import token_bytes_list
            tb[i] = token_bytes_list(self.tok, [i])[0]
        return tb[i]

    def is_prefix(self, ids) -> bool:
        """Is the text of these tokens (one word, possibly cut mid-character) the start of a lexicon word?
        Words that are not Devanagari cannot be judged and pass."""
        key = tuple(ids)
        pc = self.__dict__.setdefault("_pc", {})
        if key not in pc:
            txt = b"".join(self.tbytes(i) for i in key).decode("utf-8", errors="ignore").strip()
            if not txt or not DEV_RE.fullmatch(txt):
                pc[key] = True
            else:
                k = bisect.bisect_left(self.words, txt)
                pc[key] = k < len(self.words) and self.words[k].startswith(txt)
        return pc[key]

    def wtoks(self, w):
        if w not in self._wt:
            self._wt[w] = self.tok.encode(" " + w, add_special_tokens=False)
        return self._wt[w]

    def candidates(self, prefix_ids: tuple, prefix_txt: str):
        """[(word, continuation token ids)] for lexicon words whose tokenisation extends prefix_ids."""
        key = (prefix_ids, prefix_txt)
        if key not in self._cache:
            lo = bisect.bisect_left(self.words, prefix_txt)
            hi = bisect.bisect_left(self.words, prefix_txt + "￿")
            k, cons = len(prefix_ids), []
            for w in self.words[lo:hi]:
                if w == prefix_txt:
                    cons.append((w, ()))
                    continue
                t = self.wtoks(w)
                if len(t) > k and tuple(t[:k]) == prefix_ids:
                    cons.append((w, tuple(t[k:])))
            self._cache[key] = cons
        return self._cache[key]

    def complete(self, prefix_ids, prefix_txt, tau: float = 0.0, min_freq: int = 1):
        cons = self.candidates(tuple(prefix_ids), prefix_txt)
        if not cons:
            return ()
        w, c = max(cons, key=lambda x: self.freq[x[0]])
        tot = sum(self.freq[x] for x, _ in cons) or 1
        if self.freq[w] / tot < tau or self.freq[w] < min_freq:
            return ()
        return c


def word_prefix(t, p, tok):
    """Token ids and text of the current Devanagari word ending at position p-1 (text '' if none)."""
    wi = t.labels["word_initial"]
    if p == 0:
        return (), ""
    j = p - 1
    while j > 0 and not wi[j]:
        j -= 1
    ids = tuple(int(x) for x in t.target_ids[j:p])
    cache = t.labels.setdefault("_wp", {})
    if (j, p) not in cache:
        txt = tok.decode(list(ids)).strip()
        cache[(j, p)] = txt if txt and DEV_RE.fullmatch(txt) else ""
    return ids, cache[(j, p)]


# --------------------------------------------------------------------------- one sentence
def _round_entropy(t, p, cost, lam, beta, kappa=None, correct=None):
    n, H, wi = t.n, t.H, t.labels.get("word_initial")
    m = t.match if correct is None else correct
    g, c_draft, stop = 0, 0.0, False
    cap = min(GMAX, n - p)
    while g < cap and not stop:
        pos = p + g
        cheap = kappa is not None and not wi[pos]
        c_draft += cost.c * (kappa if cheap else 1.0)
        stop = entropy_fires(H[pos], lam, beta)
        g += 1
    ok = 0
    while ok < g and m[p + ok]:
        ok += 1
    return g, ok, c_draft


def _round_heads(t, p, cost, lam, beta, v, src, tau_h, tau_c, gate, lex=None):
    """1B drafts a token; heads propose the following tokens; repeat until the entropy rule fires.
    src: "cv"   argmax within the continuation vocabulary (cheap heads);
         "full" full-vocabulary argmax, a non-continuation token ends the proposal;
         "lexfilter"  as "cv", but stop once the word stops being a prefix of a lexicon word;
         "lexrerank"  full-vocabulary top-k: take the best continuation token that keeps the word a
                      lexicon prefix (needs full-vocabulary heads, so it pays head_rel_full)."""
    n, H, m, y = t.n, t.H, t.match, t.target_ids
    hd = t.heads[v]
    K = hd.cv_id.shape[1]
    log_tau = math.log(tau_h) if tau_h > 0 else -math.inf
    head_cost = cost.c * (cost.head_rel_cv if src in ("cv", "lexfilter") else cost.head_rel_full)
    wi = t.labels.get("word_initial")
    in_cv = t._in_cv
    cap = min(GMAX, n - p)
    g, ok, on, pending, c_draft = 0, 0, True, 0, 0.0
    while g < cap:
        pos = p + g
        c_draft += cost.c * cost.r(1 + pending)       # 1B call drafts y[pos]
        pending = 0
        if on and m[pos]:
            ok += 1
        else:
            on = False
        g += 1
        if entropy_fires(H[pos], lam, beta) or g >= cap:
            break
        if gate is not None and not gate[int(t.draft_argmax[pos])]:
            continue
        c_draft += head_cost                           # heads on h_pos propose y[pos+1..pos+K]
        props = []
        word = None
        if src.startswith("lex"):                      # current word (true path) up to and including y[pos]
            j0 = pos
            while j0 > 0 and not wi[j0]:
                j0 -= 1
            word = [int(z) for z in y[j0:pos + 1]]
        for j in range(min(K, cap - g)):
            if src in ("cv", "lexfilter"):
                x, lp = int(hd.cv_id[pos, j]), float(hd.cv_logp[pos, j])
                if src == "lexfilter" and not lex.is_prefix(word + props + [x]):
                    break
            elif src == "full":
                x, lp = int(hd.ids[pos, j, 0]), float(hd.logp[pos, j, 0])
                if not in_cv[x]:
                    break                              # head predicts a new word
            else:                                      # lexrerank
                x = None
                for r in range(hd.ids.shape[2]):
                    cand = int(hd.ids[pos, j, r])
                    if not in_cv[cand]:
                        if r == 0:
                            break                      # most likely: a new word starts
                        continue
                    if lex.is_prefix(word + props + [cand]):
                        x, lp = cand, float(hd.logp[pos, j, r])
                        break
                if x is None:
                    break
            if lp < log_tau or (tau_c is not None and hd.cont[pos, j] < tau_c):
                break
            props.append(x)
        for j, x in enumerate(props):
            if on and x == int(y[pos + 1 + j]):
                ok += 1
            else:
                on = False
        g += len(props)
        pending = len(props)
    return g, ok, c_draft


def _round_dict(t, p, cost, lam, beta, lex, tok, tau, min_freq):
    n, H, m, y = t.n, t.H, t.match, t.target_ids
    cap = min(GMAX, n - p)
    g, ok, on, pending, c_draft, stop = 0, 0, True, 0, 0.0, False
    while g < cap and not stop:
        pos = p + g
        comp = ()
        if on:                                         # off-path: no prefix to complete
            pid, ptxt = word_prefix(t, pos, tok)
            if ptxt:
                comp = lex.complete(pid, ptxt, tau, min_freq)
        if comp:
            comp = comp[:cap - g]
            for x in comp:
                if on and int(y[p + g]) == x:
                    ok += 1
                else:
                    on = False
                g += 1
            pending += len(comp)
            continue
        c_draft += cost.c * cost.r(1 + pending)
        pending = 0
        if on and m[pos]:
            ok += 1
        else:
            on = False
        stop = entropy_fires(H[pos], lam, beta)
        g += 1
    return g, ok, c_draft


def simulate(traces, mode: str, cost: Cost, lam: float = -0.683, beta: float = BETA, **kw):
    """Returns (speed-up, emitted per sentence, cost per sentence, stats)."""
    rng = np.random.default_rng(kw.get("seed", 0))
    E, C = np.zeros(len(traces)), np.zeros(len(traces))
    st = Counter()
    for i, t in enumerate(traces):
        correct = None
        if mode == "noisy":
            wi = t.labels["word_initial"]
            correct = t.match & (wi | (rng.random(t.n) < kw["rho"]))
        p = 0
        while p < t.n:
            if mode == "entropy":
                g, ok, cd = _round_entropy(t, p, cost, lam, beta)
            elif mode in ("freecont", "noisy"):
                g, ok, cd = _round_entropy(t, p, cost, lam, beta, kappa=kw.get("kappa", 0.05), correct=correct)
            elif mode == "heads":
                g, ok, cd = _round_heads(t, p, cost, lam, beta, kw["variant"], kw.get("src", "cv"),
                                         kw.get("tau_h", 0.0), kw.get("tau_c"), kw.get("gate"), kw.get("lex"))
            elif mode == "dict":
                g, ok, cd = _round_dict(t, p, cost, lam, beta, kw["lex"], kw["tok"], kw.get("tau", 0.0),
                                        kw.get("min_freq", 1))
            else:
                raise ValueError(mode)
            gained = min(ok + 1, t.n - p)
            E[i] += gained
            C[i] += cd + cost.verify(g)
            st["rounds"] += 1
            st["drafted"] += g
            st["accepted"] += ok
            p += gained
    return E.sum() / C.sum(), E, C, dict(st)


# --------------------------------------------------------------------------- statistics
def boot_ci(E, C, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(E), (B, len(E)))
    s = E[idx].sum(1) / C[idx].sum(1)
    return [float(x) for x in np.percentile(s, [2.5, 97.5])]


def paired_ci(a, b, B=1000, seed=0):
    """CI of speed-up(a) - speed-up(b) with the same sentence resamples. a, b = (E, C)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a[0]), (B, len(a[0])))
    d = a[0][idx].sum(1) / a[1][idx].sum(1) - b[0][idx].sum(1) / b[1][idx].sum(1)
    return [float(x) for x in np.percentile(d, [2.5, 97.5])]


def tune(dev, grid: list[dict], mode: str, cost: Cost, **fixed):
    """Best kwargs on dev (ties -> first). Returns (best, [(kwargs, dev speed-up)])."""
    scored = [(g, simulate(dev, mode, cost, **{**fixed, **g})[0]) for g in grid]
    best = max(scored, key=lambda x: x[1])[0]
    return best, scored


# --------------------------------------------------------------------------- head diagnostics
def head_accuracy(traces, variant: str, src: str = "cv") -> dict:
    """Along the 8B outputs: share of word-continuation targets y[t+k] (all of y[t+1..t+k] continue
    the word of y[t]) that head k gets right, vs the 1B's own accuracy on those tokens (teacher forced).
    Also rho = P(head right | 1B right) on the first continuation token."""
    K = traces[0].heads[variant].cv_id.shape[1]
    hit, b1, n = np.zeros(K), np.zeros(K), np.zeros(K)
    both = b1_first = 0
    for t in traces:
        hd, wi, y, m = t.heads[variant], t.labels["word_initial"], t.target_ids, t.match
        pred = hd.cv_id if src == "cv" else hd.ids[:, :, 0]
        for pos in range(t.n):
            for k in range(1, K + 1):
                q = pos + k
                if q >= t.n or wi[pos + 1:q + 1].any():
                    break
                n[k - 1] += 1
                hit[k - 1] += pred[pos, k - 1] == y[q]
                b1[k - 1] += m[q]
                if k == 1 and m[q]:
                    b1_first += 1
                    both += pred[pos, 0] == y[q]
    return dict(n=n.astype(int).tolist(), head_acc=(hit / np.maximum(n, 1)).round(4).tolist(),
                draft_acc=(b1 / np.maximum(n, 1)).round(4).tolist(), rho_first=round(both / max(b1_first, 1), 4))
