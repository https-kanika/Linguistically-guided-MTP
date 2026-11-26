"""Checked drafting: a cheap source proposes tokens, the 1B checks them inside a draft call it makes anyway.

One 1B call reads the pending token plus a proposal F and gives its own argmax after every token read, so it
yields the leading tokens of F it agrees with plus one more of its own. Every drafted token is therefore the
1B's own greedy token (the draft chain does not change, only the number of 1B calls), and the 8B verifies the
draft as usual. The idea is the vertical cascade of Cascade Speculative Drafting with an n-gram at the bottom;
here it is applied to this project's draft, entropy rule and word-completion heads.

Proposals: the heads' continuation tokens (stop at probability < ``tau_h``), extended by the n-gram up to ``L``
tokens. The entropy rule decides only whether another 1B call is made; tokens a call has produced are kept.
The live counterpart is ``heads_loop.HeadsDecoder`` (method kind "checked").
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .policies import entropy_fires
from .refund_sim import BETA, GMAX


class NGram:
    """Most frequent next token after the longest known context of up to ``order`` tokens (contexts seen at
    least twice). Stored as sorted integer keys, looked up by bisection."""

    def __init__(self, vals, tab: dict):
        self.vals = np.asarray(vals)                       # token of each compact id; id 0 separates sentences
        self.tab = {int(k): v for k, v in tab.items()}     # k -> (sorted context keys, next tokens)
        self.order, self.V = max(self.tab), len(self.vals)
        self.ix = {int(v): i for i, v in enumerate(self.vals) if v >= 0}

    @classmethod
    def build(cls, seqs, order: int = 4) -> "NGram":
        flat = np.concatenate([np.asarray(list(s) + [-1], dtype=np.int64) for s in seqs])
        vals, inv = np.unique(flat, return_inverse=True)
        assert vals[0] == -1 and len(vals) ** order < 2 ** 63, "contexts of this order do not fit into 63 bits"
        tab, n = {}, len(inv)
        for k in range(1, order + 1):
            m = max(n - k, 0)
            key, ok = np.zeros(m, dtype=np.uint64), np.ones(m, dtype=bool)
            for j in range(k):
                c = inv[j:j + m].astype(np.uint64)
                ok &= c > 0
                key = key * np.uint64(len(vals)) + c
            nx = inv[k:k + m]
            ok &= nx > 0
            key, nx = key[ok], nx[ok].astype(np.uint64)
            if not len(key):
                tab[k] = (np.zeros(0, np.uint64), np.zeros(0, np.int32))
                continue
            o = np.lexsort((nx, key))
            key, nx = key[o], nx[o]
            st = np.flatnonzero(np.r_[True, (key[1:] != key[:-1]) | (nx[1:] != nx[:-1])])
            cnt = np.diff(np.r_[st, len(key)])
            key, nx = key[st], nx[st]
            o = np.lexsort((-cnt, key))                    # ties: smallest token id
            key, nx, cnt = key[o], nx[o], cnt[o]
            first = np.r_[True, key[1:] != key[:-1]]
            keep = np.add.reduceat(cnt, np.flatnonzero(first)) >= 2
            tab[k] = (key[first][keep], vals[nx[first][keep].astype(np.int64)].astype(np.int32))
        return cls(vals, tab)

    def save(self, path) -> None:
        np.savez_compressed(path, vals=self.vals, **{f"k{k}": a for k, (a, _) in self.tab.items()},
                            **{f"n{k}": b for k, (_, b) in self.tab.items()})

    @classmethod
    def load(cls, path) -> "NGram":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["vals"], {int(f[1:]): (z[f], z["n" + f[1:]]) for f in z.files if f[0] == "k"})

    def next(self, ctx):
        ids = [self.ix.get(int(c)) for c in ctx[-self.order:]]
        for k in range(len(ids), 0, -1):
            if None in ids[-k:]:
                continue
            key = 0
            for c in ids[-k:]:
                key = key * self.V + c
            ks, nx = self.tab[k]
            i = int(np.searchsorted(ks, np.uint64(key)))
            if i < len(ks) and int(ks[i]) == key:
                return int(nx[i])
        return None

    def chain(self, ctx, L: int) -> list[int]:
        ctx, out = [int(c) for c in ctx[-self.order:]], []
        while len(out) < L:
            x = self.next(ctx)
            if x is None:
                break
            out.append(x)
            ctx.append(x)
        return out


def proposal(ctx, head_tokens, ngram, L: int, room: int) -> list[int]:
    """Tokens to append after ``ctx`` (output so far): the heads' tokens, extended by the n-gram to ``L``."""
    F = list(head_tokens)
    if ngram is not None and len(F) < L:
        F += ngram.chain([int(x) for x in ctx[-ngram.order:]] + F, L - len(F))
    return F[:max(room, 0)]


# --------------------------------------------------------------------------- exact simulation
def round_checked(t, p, cost, lam, beta=BETA, variant=None, tau_h=0.0, ngram=None, L=6, gate=None, memo=None):
    """(drafted, accepted by the 8B, draft cost, 1B calls). After a wrong 1B token (off the 8B's path, no
    traces) nothing is proposed and every further token costs a call; those tokens are lost anyway."""
    n, H, m, a, y = t.n, t.H, t.match, t.draft_argmax, t.target_ids
    hd = t.heads[variant] if variant else None
    log_tau = math.log(tau_h) if tau_h > 0 else -math.inf
    cap = min(GMAX, n - p)
    g, ok, on, cd, calls = 0, 0, True, 0.0, 0
    while g < cap:
        pos = p + g
        F = []
        if on and cap - g > 1:
            head = []
            if hd is not None and calls and (gate is None or gate[int(a[pos - 1])]):
                for j in range(hd.cv_id.shape[1]):         # heads ran on the state that drafted pos - 1
                    if float(hd.cv_logp[pos - 1, j]) < log_tau:
                        break
                    head.append(int(hd.cv_id[pos - 1, j]))
            if memo is not None and not head:
                if pos not in memo:
                    memo[pos] = proposal(y[:pos], [], ngram, L, L)
                F = memo[pos][:cap - g - 1]
            else:
                F = proposal(y[:pos], head, ngram, L, cap - g - 1)
        cd += cost.c * cost.r(1 + len(F))
        calls += 1
        j = 0
        while True:
            q = pos + j
            if on and m[q]:
                ok += 1
            else:
                on = False
            g += 1
            if j < len(F) and on and int(a[q]) == F[j]:    # the 1B agrees: the next token is free
                j += 1
                continue
            break
        if entropy_fires(H[pos + j], lam, beta) or g >= cap:
            break
        if hd is not None:
            cd += cost.c * cost.head_rel_cv
    return g, ok, cd, calls


def simulate_checked(traces, cost, lam, beta=BETA, **kw):
    """Same conventions and return value as ``refund_sim.simulate``."""
    E, C, st = np.zeros(len(traces)), np.zeros(len(traces)), Counter()
    ng, L = kw.get("ngram"), kw.get("L", 6)
    for i, t in enumerate(traces):
        memo = t.labels.setdefault("_ng", {}).setdefault((ng, L), {})
        p = 0
        while p < t.n:
            g, ok, cd, calls = round_checked(t, p, cost, lam, beta, memo=memo, **kw)
            gained = min(ok + 1, t.n - p)
            E[i] += gained
            C[i] += cd + cost.verify(g)
            for k, v in (("rounds", 1), ("drafted", g), ("accepted", ok), ("calls", calls)):
                st[k] += v
            p += gained
    return E.sum() / C.sum(), E, C, dict(st)
