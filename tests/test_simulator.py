import numpy as np
import pytest

from lgmtp.policies import Entropy, FixedK, NoDraft, Oracle, WordBoundarySoft
from lgmtp.simulator import (CausalityError, Policy, make_verify, simulate, summarize, unit_verify)


# --- legacy sanity checks ---------------------------------------------------- #
def test_all_match_k4(mk):
    r = simulate([mk([1] * 10)], FixedK(4), 4, 0.0)
    assert r.rounds == 2 and r.emitted == 10


def test_no_match(mk):
    r = simulate([mk([0] * 6)], FixedK(3), 3, 0.0)
    assert r.rounds == 6 and r.accepted == 0 and r.emitted == 6


def test_nodraft_speedup_1(mk):
    r = simulate([mk([1] * 7)], NoDraft(), 8, 0.3)
    assert abs(summarize(r)["speedup"] - 1) < 1e-9 and r.drafted == 0


def test_oracle_never_rejects(mk):
    rng = np.random.default_rng(0)
    trs = [mk(rng.random(50) < 0.7, sid=i) for i in range(20)]
    r = simulate(trs, Oracle(), 8, 0.3)
    assert r.rejections == 0 and r.emitted == sum(t.n for t in trs)


def test_every_policy_emits_n(mk):
    rng = np.random.default_rng(1)
    trs = [mk(rng.random(37) < 0.6, sid=i, H=rng.random(37) * 3) for i in range(10)]
    n = sum(t.n for t in trs)
    for pol in [FixedK(1), FixedK(5), Entropy(0.5), Entropy(-0.3, 2.0), NoDraft(), Oracle(),
                WordBoundarySoft(0.5, 2)]:
        r = simulate(trs, pol, 8, 0.3)
        assert r.emitted == n == r.sent_emitted.sum()
        assert r.cost == pytest.approx(r.sent_cost.sum())


def test_legacy_entropy_semantics(mk):
    # H high at index 1 -> entropy (lam=0.5, beta=1) stops after drafting 2 tokens in the first round
    H = [0, 4, 0, 0, 0, 0, 0, 0, 0, 0]
    r = simulate([mk([1] * 10, H=H)], Entropy(0.5, 1.0), 8, 0.0)
    assert r.records[0][2] == 2


# --- lookahead cost ---------------------------------------------------------- #
def test_lookahead_charge_exactly_c(mk):
    c = 0.3
    trs = [mk([1] * 10), mk([1, 0, 1, 1, 0, 1, 1, 1], sid=1)]
    pol = WordBoundarySoft(tau=-1.0, min_g=2)       # always stops by itself at g = 2
    ch = simulate(trs, pol, 8, c, charge_lookahead=True)
    un = simulate(trs, pol, 8, c, charge_lookahead=False)
    stopped = sum(1 for rec in ch.records if rec[4] and rec[2] > 0)
    assert ch.lookahead_charges == stopped > 0
    assert ch.cost - un.cost == pytest.approx(c * stopped)
    assert ch.emitted == un.emitted
    # rounds that hit the cap (cap < 2 at the end of a sentence) are not charged
    capped = [rec for rec in ch.records if not rec[4]]
    assert all(rec[2] < 2 for rec in capped)


def test_causal_policy_never_charged(mk):
    r = simulate([mk([1] * 10, H=[5] * 10)], Entropy(0.5), 8, 0.3)
    assert r.lookahead_charges == 0


# --- causality guard --------------------------------------------------------- #
class PeekH(Policy):
    name = "PeekH"
    def stop(self, view, p, g):
        return view.feat("H", p + g) > 1     # reads q_{p+g} without declaring lookahead


class PeekToken(Policy):
    name = "PeekToken"
    def stop(self, view, p, g):
        return view.token(p + g) == 0


class PeekMatch(Policy):
    name = "PeekMatch"
    lookahead = 1
    def stop(self, view, p, g):
        return not view.gold("match", p + g)


class PeekTarget(Policy):
    name = "PeekTarget"
    lookahead = 1
    def stop(self, view, p, g):
        return view.feat("target_ids", p) == 0


class LookTwo(Policy):
    name = "LookTwo"
    lookahead = 1
    def stop(self, view, p, g):
        return view.feat("p_newword", p + g + 1) > 0.5


@pytest.mark.parametrize("pol", [PeekH(), PeekToken(), PeekMatch(), PeekTarget(), LookTwo()])
def test_cheating_policy_raises(mk, pol):
    with pytest.raises(CausalityError):
        simulate([mk([1] * 20)], pol, 8, 0.3)


def test_token_view_committed_vs_drafted(mk):
    seen = []

    class Rec(Policy):
        name = "Rec"
        def stop(self, view, p, g):
            seen.append((p, g, [view.token(i) for i in range(p + g)]))
            return g >= 2

    t = mk([1, 0, 1, 1, 1, 1])
    simulate([t], Rec(), 8, 0.0)
    for p, g, toks in seen:
        assert toks[:p] == list(t.target_ids[:p])            # committed: target
        assert toks[p:] == list(t.draft_argmax[p:p + g])     # drafted: draft's own tokens


# --- measured verify cost ---------------------------------------------------- #
TIMING = dict(t_target=1.0, t_target_verify={"1": 1.2, "2": 1.5, "4": 2.0, "8": 0.5})


def test_make_verify_interp_and_clip():
    v = make_verify(TIMING)
    assert v(0) == 1.0 and v(1) == pytest.approx(1.2) and v(2) == pytest.approx(1.5)
    assert v(3) == pytest.approx(1.75)
    assert v(8) == 1.0                                       # clipped at one decode step
    assert v(6) == pytest.approx(np.interp(6, [4, 8], [2.0, 1.0]))


def test_measured_verify_hand_computed(mk):
    v = make_verify(TIMING)
    # 6 tokens all match, k=2, c=0.5: rounds (g=2, emit 3) x 2; cost = 2 * (2*0.5 + 1.5) = 5
    r = simulate([mk([1] * 6)], FixedK(2), 2, 0.5, verify=v)
    assert r.rounds == 2 and r.cost == pytest.approx(5.0) and summarize(r)["speedup"] == pytest.approx(6 / 5)
    # with a mismatch at index 1: round1 g=2 a=1 emit 2 (p=2); round2 g=2 a=2 emit 3 (p=5);
    # round3 cap=1 g=1 a=1 emit 1 -> cost 2*(1+1.5) + (0.5+1.2) = 6.7
    r = simulate([mk([1, 0, 1, 1, 1, 1])], FixedK(2), 2, 0.5, verify=v)
    assert [rec[2:4] for rec in r.records] == [(2, 1), (2, 2), (1, 1)]
    assert r.cost == pytest.approx(6.7)
    assert unit_verify(5) == 1.0


# --- port equivalence: legacy simulate (verbatim from day2-day3 notebook cell 16) ----------- #
def _legacy_simulate(traces, policy, gamma_max, c, verify=None):
    rounds = drafted = accepted = rejections = emitted = 0
    cost = 0.0
    draft_lens = []
    for s in traces:
        n, match = s["n"], s["match"]
        p = 0
        while p < n:
            r = n - p
            cap = min(gamma_max, r)
            g = 0
            while g < cap and not policy(s, p, g):
                g += 1
            a = 0
            while a < g and match[p + a]:
                a += 1
            gained = min(a + 1, r)
            rounds += 1; drafted += g; accepted += a
            rejections += int(a < g); emitted += gained
            cost += g * c + (verify(g) if verify else 1.0)
            draft_lens.append(g)
            p += gained
    return dict(rounds=rounds, drafted=drafted, accepted=accepted, rejections=rejections,
                emitted=emitted, cost=cost)


def test_port_matches_legacy(mk):
    import math
    rng = np.random.default_rng(7)
    trs = [mk(rng.random(int(rng.integers(5, 80))) < 0.7, sid=i, H=rng.random(80)[:0] if False else None)
           for i in range(30)]
    for t in trs:
        t.H = (rng.random(t.n) * 3).astype(np.float32)
    leg = [dict(n=t.n, match=t.match, H=t.H) for t in trs]
    v = make_verify(TIMING)
    cases = [
        (FixedK(3), lambda s, p, g: g >= 3, 3),
        (FixedK(8), lambda s, p, g: g >= 8, 8),
        (Entropy(0.3, 2.0), lambda s, p, g: g != 0 and 1 - math.sqrt(2.0 * max(s["H"][p + g - 1], 0)) < 0.3, 8),
        (Oracle(), lambda s, p, g: not s["match"][p + g], 8),
        (NoDraft(), lambda s, p, g: True, 8),
    ]
    for pol, legpol, cap in cases:
        for verify in (None, v):
            new = simulate(trs, pol, cap, 0.334, verify=verify)
            old = _legacy_simulate(leg, legpol, cap, 0.334, verify)
            for k in ("rounds", "drafted", "accepted", "rejections", "emitted"):
                assert getattr(new, k) == old[k], (pol, k)
            assert new.cost == pytest.approx(old["cost"])
