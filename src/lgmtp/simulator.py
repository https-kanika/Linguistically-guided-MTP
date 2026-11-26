"""Trace simulator for greedy speculative decoding (fixes B3, B8).

Ported from ``notebooks/pre_interim/day2-day3-llama8b-kaggle.ipynb`` (cells 14-17), with:
  1. the ``Policy`` interface ``stop(view, p, g)`` called with g = 1..cap-1 after each drafted token
     (dynamic policies always draft >= 1 token; NoDraft / FixedK(0) / Oracle decide g = 0 via
     ``initial_stop(view, p)``);
  2. a causality guard (``TraceView``);
  3. lookahead cost: a lookahead-1 policy that stops by itself pays one extra draft forward ``c``;
  4. per-sentence emitted / cost arrays and round records;
  5. measured verification cost as the default.

Exactness for greedy decoding: within a round the draft's own tokens coincide with the target's up
to the first mismatch, and everything after it is rejected, so accepted = leading matches.
Only the stopping decision after a mismatch can differ from a real loop (teacher-forcing
approximation).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .traces import Trace

GAMMA_MAX = 8

# trace fields a policy may read through ``view.feat`` (functions of q_t only)
DRAFT_FEATURES = {"H", "top1", "top2", "p_space", "p_punct", "p_eos", "p_newword", "p_mark", "p_cont",
                  "draft_argmax", "draft_topk_ids", "draft_topk_logp"}
# never readable by non-oracle policies
GOLD_FIELDS = {"match", "target_ids", "p_target", "target_topk_ids", "target_topk_logp",
               "target_logp_of_draft_argmax"}


class CausalityError(RuntimeError):
    """A policy read information a real decoder would not have at that moment."""


class TraceView:
    """Read-only, causality-checked view of one trace for policies.

    The simulator sets ``limit = p + g - 1 + policy.lookahead`` and ``p`` before each call.
      feat(name, idx): draft feature at index idx (idx <= limit).
      token(idx):      committed target token for idx < p, the draft's own token for p <= idx <= p+g-1
                       (idx <= limit; for lookahead policies idx = p+g gives the draft argmax at p+g).
      gold(name, idx): any field incl. match / target ids / labels; oracle policies only.
    """

    __slots__ = ("trace", "limit", "p", "is_oracle", "n", "_labels")

    def __init__(self, trace: Trace, is_oracle: bool = False):
        self.trace, self.is_oracle, self.n = trace, is_oracle, trace.n
        self.limit = -1
        self.p = 0
        self._labels = trace.labels

    def _check(self, idx: int) -> None:
        if idx > self.limit and not self.is_oracle:
            raise CausalityError(f"read index {idx} > limit {self.limit} (p={self.p})")
        if idx < 0 or idx >= self.n:
            raise IndexError(idx)

    def feat(self, name: str, idx: int):
        if name not in DRAFT_FEATURES and not self.is_oracle:
            raise CausalityError(f"policy may not read '{name}' via feat()")
        self._check(idx)
        return getattr(self.trace, name)[idx]

    def token(self, idx: int) -> int:
        self._check(idx)
        if idx < self.p:
            return int(self.trace.target_ids[idx])
        return int(self.trace.draft_argmax[idx])

    def gold(self, name: str, idx: int):
        if not self.is_oracle:
            raise CausalityError(f"non-oracle policy read gold field '{name}'")
        if name in self._labels:
            return self._labels[name][idx]
        return getattr(self.trace, name)[idx]


class Policy:
    name: str = "policy"
    lookahead: int = 0          # 0 or 1
    is_oracle: bool = False

    def initial_stop(self, view: TraceView, p: int) -> bool:
        """Decision at g = 0 (before drafting; view.limit = p - 1). Only NoDraft / FixedK(0) and the
        oracle can return True: every other dynamic policy drafts at least one token."""
        return False

    def stop(self, view: TraceView, p: int, g: int) -> bool:  # g >= 1
        raise NotImplementedError

    def reset(self, view: TraceView) -> None:
        """Called once per sentence (policies with per-sentence caches)."""

    def params(self) -> dict:
        return {}

    def __repr__(self):
        ps = ", ".join(f"{k}={v}" for k, v in self.params().items())
        return f"{self.name}({ps})"


# --------------------------------------------------------------------------- #
# cost model
# --------------------------------------------------------------------------- #

def load_timing(path) -> dict:
    return json.loads(Path(path).read_text())


class MeasuredVerify:
    """Measured verification cost for g drafted tokens (g+1 target tokens) relative to one decode
    step, linearly interpolated, clipped below at 1 (dips are timing noise). verify(0) = 1.
    A class (not a closure) so it can be sent to worker processes."""

    def __init__(self, timing: dict):
        self.gs = sorted(int(g) for g in timing["t_target_verify"])
        self.vs = [max(1.0, timing["t_target_verify"][str(g)] / timing["t_target"]) for g in self.gs]
        self.table = {0: 1.0}
        for g in range(1, max(self.gs) + 1):
            self.table[g] = float(np.interp(g, self.gs, self.vs))

    def __call__(self, g: int) -> float:
        return self.table[g] if g in self.table else float(np.interp(g, self.gs, self.vs))


def make_verify(timing: dict) -> MeasuredVerify:
    return MeasuredVerify(timing)


def unit_verify(g: int) -> float:
    return 1.0


def formula_speedup(alpha: float, gamma: int, c: float) -> float:
    """f = (1 - a^(g+1)) / ((1 - a)(g c + 1)), i.i.d. acceptance assumption (legacy)."""
    exp_tokens = (gamma + 1) if alpha >= 1 else (1 - alpha ** (gamma + 1)) / (1 - alpha)
    return exp_tokens / (gamma * c + 1)


# --------------------------------------------------------------------------- #
# simulator
# --------------------------------------------------------------------------- #

@dataclass
class SimResult:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    rejections: int = 0
    emitted: int = 0
    cost: float = 0.0
    lookahead_charges: int = 0
    policy_stops: int = 0
    draft_lens: list = field(default_factory=list)
    sent_ids: np.ndarray = None
    sent_emitted: np.ndarray = None
    sent_cost: np.ndarray = None
    records: list = field(default_factory=list)   # (sentence_id, p, g, accepted, stopped_by_policy)


def simulate(traces: Sequence[Trace], policy: Policy, gamma_max: int = GAMMA_MAX, c: float = 0.334,
             verify: Callable[[int], float] | None = None, charge_lookahead: bool = True,
             record_rounds: bool = True) -> SimResult:
    """Simulate greedy speculative decoding of every trace with ``policy``.

    One round at committed position p: draft g tokens (cap = min(gamma_max, n - p)), accept the
    leading matches a, emit min(a + 1, n - p) tokens, cost g*c + verify(g)
    (+ c if the policy has lookahead 1, stopped by itself before the cap, and charge_lookahead).
    """
    verify = verify or unit_verify
    R = SimResult()
    ns = len(traces)
    R.sent_ids = np.array([t.id for t in traces])
    R.sent_emitted = np.zeros(ns, dtype=np.int64)
    R.sent_cost = np.zeros(ns, dtype=np.float64)
    la = int(policy.lookahead)
    for si, tr in enumerate(traces):
        n, match = tr.n, tr.match
        view = TraceView(tr, policy.is_oracle)
        policy.reset(view)
        p = 0
        s_emit, s_cost = 0, 0.0
        while p < n:
            r = n - p
            cap = min(gamma_max, r)
            view.p = p
            g = 0
            by_policy = False
            view.limit = p - 1
            if cap > 0 and not policy.initial_stop(view, p):
                g = 1
                while g < cap:
                    view.limit = p + g - 1 + la
                    if policy.stop(view, p, g):
                        by_policy = True
                        break
                    g += 1
            elif cap > 0:
                by_policy = True
            a = 0
            while a < g and match[p + a]:
                a += 1
            gained = min(a + 1, r)
            cost = g * c + verify(g)
            if la and by_policy and g > 0 and charge_lookahead:
                cost += c
                R.lookahead_charges += 1
            R.rounds += 1
            R.drafted += g
            R.accepted += a
            R.rejections += int(a < g)
            R.emitted += gained
            R.cost += cost
            R.policy_stops += int(by_policy)
            R.draft_lens.append(g)
            if record_rounds:
                R.records.append((tr.id, p, g, a, by_policy))
            s_emit += gained
            s_cost += cost
            p += gained
        R.sent_emitted[si] = s_emit
        R.sent_cost[si] = s_cost
    return R


def summarize(res: SimResult) -> dict:
    """Legacy summary metrics."""
    return dict(
        acceptance_rate=res.accepted / max(res.drafted, 1),                    # accepted / drafted
        alpha_hat=res.accepted / max(res.accepted + res.rejections, 1),        # per-token accept prob (geometric MLE)
        tokens_per_verif=res.emitted / max(res.rounds, 1),
        mean_draft_len=res.drafted / max(res.rounds, 1),
        speedup=res.emitted / res.cost if res.cost else float("nan"),
    )


# --------------------------------------------------------------------------- #
# legacy protocol (sanity check against the PDF numbers)
# --------------------------------------------------------------------------- #

def legacy_protocol(traces: Sequence[Trace], c: float, seed: int = 0) -> dict:
    """Reproduce the old evaluation: random half split of devtest (default_rng(0).permutation),
    entropy grid lam = linspace(-1.5, 0.95, 40), beta in {0.5, 1, 2}, unit verify, test on the 2nd half.
    Expected (PDF): fixed k=2 ~ 1.36, entropy ~ 1.45 (within ~0.03 after regeneration)."""
    from .policies import Entropy, FixedK, Oracle
    idx = np.random.default_rng(seed).permutation(len(traces))
    dev = [traces[i] for i in idx[: len(idx) // 2]]
    test = [traces[i] for i in idx[len(idx) // 2:]]
    ks = range(2, 9)
    best_k = max(ks, key=lambda k: summarize(simulate(dev, FixedK(k), k, c, record_rounds=False))["speedup"])
    rows = []
    for beta in (0.5, 1.0, 2.0):
        for lam in np.round(np.linspace(-1.5, 0.95, 40), 3):
            m = summarize(simulate(dev, Entropy(float(lam), beta), GAMMA_MAX, c, record_rounds=False))
            rows.append((m["speedup"], float(lam), beta))
    _, lam, beta = max(rows)
    fixed_all = {k: summarize(simulate(traces, FixedK(k), k, c, record_rounds=False)) for k in ks}
    return dict(
        best_k_dev=best_k,
        fixed_test=summarize(simulate(test, FixedK(best_k), best_k, c, record_rounds=False)),
        fixed_k2_all=fixed_all[2],
        entropy_params=dict(lam=lam, beta=beta),
        entropy_test=summarize(simulate(test, Entropy(lam, beta), GAMMA_MAX, c, record_rounds=False)),
        oracle_test=summarize(simulate(test, Oracle(), GAMMA_MAX, c, record_rounds=False)),
        expected=dict(fixed_k2=1.361, entropy=1.452, oracle=1.905),
    )
