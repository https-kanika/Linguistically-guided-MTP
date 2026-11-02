import numpy as np
import pytest

from lgmtp.boundaries import token_labels
from lgmtp.morph import PrefixBoundaryModel
from lgmtp.policies import (AksharaSafeEntropy, Entropy, FixedK, GoldWordOracle, Hybrid, MorphEntropy,
                            MorphOnly, NoDraft, Oracle, TokenTable, WordBoundaryHard, WordBoundarySoft,
                            current_word_prefix)
from lgmtp.simulator import TraceView, simulate

PIECES = [" राम", "पुर", " में", "।", " सी", "ता", "ा", "्", "क", " है", ",", " गढ", "ने", "ं", "ि"]
V = 50


def table():
    tb = [PIECES[i % len(PIECES)].encode() for i in range(V)]
    ws = np.array([b[:1] in (b" ",) or b in ("।".encode(), b",") for b in tb])
    return TokenTable(tb, ws)


def prefix_model():
    return PrefixBoundaryModel(min_count=1).fit([("रामपुर", {6}, 1.0), ("सीता", {2}, 1.0), ("गढने", {4}, 2.0)])


def synth(mk, n=60, sid=0):
    rng = np.random.default_rng(sid)
    t = mk(rng.random(n) < 0.65, sid=sid, rng=rng, H=rng.random(n) * 3)
    t.p_newword = rng.random(n).astype(np.float32)
    t.p_mark = rng.random(n).astype(np.float32)
    tb = table()
    t.labels.update(token_labels([tb.token_bytes[i] for i in t.target_ids]))
    return t


def all_policies():
    tb, pm = table(), prefix_model()
    newword = tb.word_start.copy()
    return [
        NoDraft(), FixedK(3), Entropy(0.2, 1.0),
        WordBoundarySoft(0.5, 1), WordBoundarySoft(0.3, 3), WordBoundaryHard(1, newword), WordBoundaryHard(2, newword),
        AksharaSafeEntropy(0.2, 1.0, 0.3, 2), Hybrid(0.2, 1.0, 0.4, 0), Hybrid(0.2, 2.0, 0.6, 3),
        MorphEntropy(0.2, 1.0, 0.3, False, 0.5, 2, pm, tb), MorphEntropy(0.2, 1.0, 0.3, True, 0.5, 1, pm, tb),
        MorphOnly(0.3, 1, pm, tb),
    ]


@pytest.mark.parametrize("pol", all_policies(), ids=lambda p: repr(p)[:60])
def test_non_oracle_policies_are_causal(mk, pol):
    assert not pol.is_oracle
    trs = [synth(mk, 60, s) for s in range(5)]
    for charge in (True, False):
        r = simulate(trs, pol, 8, 0.33, charge_lookahead=charge)   # raises CausalityError if not causal
        assert r.emitted == sum(t.n for t in trs)
    if pol.lookahead == 0:
        assert r.lookahead_charges == 0


def test_oracles_run_and_bound(mk):
    trs = [synth(mk, 60, s) for s in range(5)]
    o = simulate(trs, Oracle(), 8, 0.33)
    assert o.rejections == 0
    for mg in (1, 2, 3):
        r = simulate(trs, GoldWordOracle(mg), 8, 0.33)
        assert r.emitted == sum(t.n for t in trs)


def test_extend_semantics(mk):
    # entropy fires at g=1 everywhere (H large); Hybrid with an impossible word condition stops at g0+max_ext
    n = 30
    t = mk([1] * n, H=[9.0] * n)
    t.p_newword = np.zeros(n, np.float32)
    for ext in (0, 1, 3):
        r = simulate([t], Hybrid(0.5, 1.0, 0.9, ext), 8, 0.0)
        assert all(rec[2] == min(1 + ext, n - rec[1]) for rec in r.records)


def test_current_word_prefix(mk):
    tb = table()
    ids = {p: i for i, p in enumerate(PIECES)}
    # tokens: " राम" "पुर" " में" -> drafted from p=1
    t = mk([1, 1, 1])
    t.target_ids = np.array([ids[" राम"], ids["पुर"], ids[" में"]])
    t.draft_argmax = t.target_ids.copy()
    v = TraceView(t)
    v.p, v.limit = 1, 2
    assert current_word_prefix(v, 1, 1, tb) == "रामपुर"     # committed " राम" + drafted "पुर"
    assert current_word_prefix(v, 1, 2, tb) == "में"         # last drafted token starts a new word
