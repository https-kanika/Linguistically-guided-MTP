import numpy as np
import pytest

from lgmtp import refund_sim as R
from lgmtp.refund_sim import Cost, Heads


def unit_cost(c=0.3, **kw):
    return Cost(c=c, verify=lambda g: 1.0, head_rel_cv=0.0, head_rel_full=0.0, **kw)


WI = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0], bool)   # words of 3, 2, 1, 4, 2 tokens


def with_words(mk, match=None, wi=WI):
    t = mk(np.ones(len(wi), bool) if match is None else match)
    t.labels["word_initial"] = wi
    return t


def perfect_heads(t, K=3, wrong_at=None):
    """Heads that propose exactly y[t+1..t+k] and know where the word ends."""
    n, y, wi = t.n, t.target_ids, t.labels["word_initial"]
    cv_id = np.zeros((n, K), np.int32)
    cont = np.zeros((n, K), np.float32)
    for pos in range(n):
        for j in range(K):
            q = pos + 1 + j
            if q < n:
                cv_id[pos, j] = y[q]
                cont[pos, j] = float(not wi[pos + 1:q + 1].any())
    if wrong_at is not None:
        cv_id[wrong_at] = (cv_id[wrong_at] + 1) % 50
    ids = np.repeat(cv_id[:, :, None], 2, 2)
    lp = np.zeros((n, K, 2), np.float32)
    t.heads = {v: Heads(ids, lp, cv_id, np.zeros((n, K), np.float32), cont) for v in R.VARIANTS}
    t._in_cv = np.ones(64, bool)
    return t


def test_entropy_never_fires_drafts_gmax(mk):
    t = with_words(mk, wi=np.ones(8, bool))
    s, E, C, st = R.simulate([t], "entropy", unit_cost(), lam=-0.683)
    assert E[0] == 8 and C[0] == pytest.approx(8 * 0.3 + 1.0)
    assert st["rounds"] == 1


def test_perfect_heads_equal_free_continuations(mk):
    """With perfect heads costing nothing, the heads drafter matches the kappa = 0 ceiling within one
    round. (Across rounds the ceiling is looser: a round that starts mid-word still needs a 1B call.)"""
    t = perfect_heads(with_words(mk, wi=WI[:8]))
    free = R.simulate([t], "freecont", unit_cost(), kappa=0.0)
    heads = R.simulate([t], "heads", unit_cost(), variant="v1", src="cv", tau_c=0.5)
    assert heads[1][0] == free[1][0]
    assert heads[2][0] == pytest.approx(free[2][0])
    assert heads[3]["drafted"] == free[3]["drafted"]


def test_wrong_head_token_ends_acceptance(mk):
    t = perfect_heads(with_words(mk), wrong_at=0)      # head 1 at position 0 proposes a wrong y[1]
    g, ok, _ = R._round_heads(t, 0, unit_cost(), -0.683, 4.0, "v1", "cv", 0.0, 0.5, None)
    assert ok == 1 and g > 1                            # only the 1B's y[0] is accepted


def test_head_confidence_gate_blocks_proposals(mk):
    t = perfect_heads(with_words(mk))
    for v in R.VARIANTS:
        t.heads[v].cv_logp[:] = np.log(0.2)
    g, ok, cd = R._round_heads(t, 0, unit_cost(), -0.683, 4.0, "v1", "cv", 0.5, None, None)
    # no proposals pass tau_h = 0.5: the 1B drafts all 8 tokens itself
    assert g == 8 and ok == 8 and cd == pytest.approx(8 * 0.3)


def test_noisy_rho_one_equals_freecont(mk):
    t = with_words(mk)
    a = R.simulate([t], "noisy", unit_cost(), rho=1.0, kappa=0.05)
    b = R.simulate([t], "freecont", unit_cost(), kappa=0.05)
    assert a[0] == pytest.approx(b[0])


def test_r_q_clipped_and_extended():
    c = unit_cost(r_q={1: 1.0, 2: 0.95, 4: 1.1})
    assert c.r(1) == 1.0 and c.r(2) == 1.0 and c.r(3) == 1.0 and c.r(4) == 1.1 and c.r(8) == 1.1


def test_paired_ci_zero_for_identical():
    E, C = np.arange(1, 11.0), np.arange(2, 12.0)
    assert R.paired_ci((E, C), (E, C)) == [0.0, 0.0]
