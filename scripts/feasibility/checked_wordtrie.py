"""Exact-simulation test of the training-free idea: a word trie proposes the rest of the current word (and, in one
variant, the next word); the 1B CHECKS the proposal inside the draft call it makes anyway (one forward over the
pending tokens gives its own prediction after every one of them). The draft chain is always the 1B's own greedy chain.

A 1B call at position pos (known chain y[:pos]) reads the last token plus the fill F. It yields the leading F_j that
equal the 1B's own argmax, plus one more 1B token. Off the 8B's path (after a wrong 1B token) no fill is used, so every
further token costs a full call (conservative; those tokens are lost anyway).
Verification, entropy rule, cost model and g <= 8 are the ones of refund_sim.
  keep=False: the chain stops at the first token where the entropy rule fires (same chain as Entropy, fewer calls).
  keep=True : tokens a call has already produced are free, so all are kept; the rule only decides about another call.
Tuned on dev, reported on devtest. Writes results/checked/wordtrie.csv.
  "oracle": the rest of the word the 1B itself will write (a bound for word-completion proposals).

    python scripts/feasibility/checked_wordtrie.py      (needs ../week1/data/sources.json and data/traces_v2)"""
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from transformers import AutoTokenizer
from lgmtp import refund_sim as R
from lgmtp.traces import attach_word_labels, load_traces

t0 = time.time()
D = ROOT / "data/traces_v2"
tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
cost = R.load_cost(D / "timing.json", None)
tr = {}
for s in ("dev", "devtest"):
    tr[s] = load_traces(D / f"traces_{s}.npz")[0]
    attach_word_labels(tr[s], tok)

S = json.load(open(ROOT.parent / "week1" / "data" / "sources.json", encoding="utf-8"))
mono = [x[1] if isinstance(x, list) else x for x in S["mono"]][:190000]
devout = [json.loads(l)["out_text"] for l in open(D / "target_outputs_dev.jsonl", encoding="utf-8")]
PUN = "।,.?!;:\"'()[]{}–-—“”‘’"


def words_of(sents):
    for s in sents:
        yield [w.strip(PUN) for w in s.split()]


def build(sents):
    uni_c, bi_c = collections.Counter(), collections.Counter()
    for ws in words_of(sents):
        prev = "<s>"
        for w in ws:
            if w:
                uni_c[w] += 1
                bi_c[(prev, w)] += 1
                prev = w
    types = list(uni_c)
    T = dict(zip(types, map(tuple, tok([" " + w for w in types], add_special_tokens=False)["input_ids"])))
    uni, big, nxt = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter), {}
    for w, c in uni_c.items():
        tt = T[w]
        for j in range(1, len(tt) + 1):
            uni[tt[:j]][tt] += c
    for (a, w), c in bi_c.items():
        if c >= 2:
            tt = T[w]
            for j in range(1, len(tt) + 1):
                big[(a, tt[:j])][tt] += c
            if c >= 3 and c > nxt.get(a, (None, 0))[1]:
                nxt[a] = (w, c)
    U = {p: c.most_common(1)[0][0] for p, c in uni.items()}
    B = {k: c.most_common(1)[0][0] for k, c in big.items() if sum(c.values()) >= 3}
    N = {a: T[w] for a, (w, _) in nxt.items()}
    return U, B, N, {v: k for k, v in T.items()}


U, B, N, INV = build(mono)
print(f"[{time.time() - t0:.0f}s] trie: {len(U)} prefixes, {len(B)} with previous word, {len(N)} next-word entries", flush=True)


def prep(t):
    """Per position pos: (ids of the current word up to pos-1, text of the previous word)."""
    if "_cw" in t.labels:
        return t.labels["_cw"]
    wi, y, out = t.labels["word_initial"], t.target_ids, []
    start, prev = 0, "<s>"
    out.append(((), "<s>"))
    for p in range(1, t.n + 1):
        if p - 1 > 0 and wi[p - 1]:
            prev = tok.decode([int(x) for x in y[start:p - 1]]).strip().strip(PUN) or prev
            start = p - 1
        out.append((tuple(int(x) for x in y[start:p]), prev))
    t.labels["_cw"] = out
    return out


def fill(t, pos, variant):
    pre, prev = prep(t)[pos]
    if not pre:
        return ()
    full = B.get((prev, pre)) if variant != "unigram" else None
    if full is None:
        full = U.get(pre)
    F = full[len(pre):] if full is not None and len(full) > len(pre) else ()
    if variant == "next":
        if full is not None:                        # word known (complete or completed): propose the likely next word
            w = INV.get(tuple(full) if F else pre)
            if w in N:
                F = tuple(F) + N[w]
    return F


def oracle_fill(t, pos):
    """Upper bound for word completion: the rest of the word the 1B itself will write."""
    wi, m, F, q = t.labels["word_initial"], t.match, [], pos
    while q < t.n and not (wi[q] and q >= pos) and q - pos < 8:
        F.append(int(t.draft_argmax[q]))
        if not m[q]:
            break
        q += 1
    return tuple(F)


def simulate(traces, lam, variant, keep, beta=4.0, maxfill=8):
    E, C, st = np.zeros(len(traces)), np.zeros(len(traces)), collections.Counter()
    for i, t in enumerate(traces):
        n, H, m, a = t.n, t.H, t.match, t.draft_argmax
        p = 0
        while p < n:
            cap = min(R.GMAX, n - p)
            g, on, ok, cd, stop = 0, True, 0, 0.0, False
            while g < cap and not stop:
                pos = p + g
                F = ()
                if on and variant != "none":
                    F = (oracle_fill(t, pos) if variant == "oracle" else fill(t, pos, variant))[:min(maxfill, cap - g - 1)]
                cd += cost.c * cost.r(1 + len(F))
                st["calls"] += 1
                st["fill_read"] += len(F)
                for j in range(len(F) + 1):
                    q = pos + j
                    if on and m[q]:
                        ok += 1
                    else:
                        on = False
                    g += 1
                    fires = R.entropy_fires(H[q], lam, beta)
                    if j < len(F) and int(a[q]) == F[j] and on:     # the 1B agrees with the fill: next token is free
                        st["fill_ok"] += 1
                        if fires and not keep:
                            stop = True
                            break
                        continue
                    stop = fires
                    break
            gained = min(ok + 1, n - p)
            E[i] += gained
            C[i] += cd + cost.verify(g)
            st["rounds"] += 1
            st["drafted"] += g
            p += gained
    return E.sum() / C.sum(), E, C, st


base = R.simulate(tr["devtest"], "entropy", cost, lam=-0.683)
chk = simulate(tr["devtest"], -0.683, "none", False)
assert abs(base[0] - chk[0]) < 1e-12, (base[0], chk[0])          # same code path as the published baseline
print(f"Entropy {base[0]:.4f}; no-fill check {chk[0]:.4f}")
rows = {}
for variant in ("unigram", "context", "next", "oracle"):
    for keep in (False, True):
        sc = [(simulate(tr["dev"], l, variant, keep)[0], l) for l in R.LAM_GRID]
        d, lam = max(sc)
        r = simulate(tr["devtest"], lam, variant, keep)
        lo, hi = R.boot_ci(r[1], r[2])
        dl, dh = R.paired_ci((r[1], r[2]), (base[1], base[2]))
        st = r[3]
        rows[(variant, keep)] = dict(proposal=variant, keep=keep, lam=lam, dev=round(d, 4), speedup=round(r[0], 4),
                                     lo=round(lo, 4), hi=round(hi, 4), d_entropy_lo=round(dl, 4), d_entropy_hi=round(dh, 4),
                                     fill_accepted_by_1B=round(st["fill_ok"] / max(1, st["fill_read"]), 3))
        print(f"[{time.time() - t0:4.0f}s] {variant:8s} keep={keep!s:5s} lam={lam:6.3f} dev {d:.4f} devtest {r[0]:.4f} [{lo:.4f}, {hi:.4f}] "
              f"diff vs Entropy [{dl:+.4f}, {dh:+.4f}] | 1B tokens per call {st['drafted'] / st['calls']:.2f}, tokens/round {r[1].sum() / st['rounds']:.2f}, "
              f"fill accepted by 1B {st['fill_ok'] / max(1, st['fill_read']):.0%}", flush=True)

import csv  # noqa: E402
with open(ROOT / "results" / "checked" / "wordtrie.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, list(next(iter(rows.values()))))
    w.writeheader()
    w.writerows(rows.values())
