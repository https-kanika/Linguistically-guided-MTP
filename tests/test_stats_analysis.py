import numpy as np
import pytest

from lgmtp.analysis import point_biserial, position_frame, standard_correlations, table_by, token_type
from lgmtp.boundaries import token_labels
from lgmtp.stats import bootstrap_ci, paired_bootstrap


def test_bootstrap_ci_contains_point_and_ratio_of_sums():
    rng = np.random.default_rng(0)
    e = rng.integers(20, 80, 200)
    c = e / rng.uniform(1.2, 1.6, 200)
    r = bootstrap_ci(e, c, n_boot=500)
    assert r["point"] == pytest.approx(e.sum() / c.sum())
    assert r["lo"] < r["point"] < r["hi"]


def test_paired_bootstrap_identical_and_shifted():
    rng = np.random.default_rng(1)
    e = rng.integers(20, 80, 100).astype(float)
    c = e / 1.4
    same = paired_bootstrap(e, c, e, c, n_boot=200)
    assert same["diff"] == 0 and same["p_better"] == 0
    better = paired_bootstrap(e, c * 0.9, e, c, n_boot=200)
    assert better["diff"] > 0 and better["p_better"] == 1.0 and better["lo"] > 0
    with pytest.raises(ValueError):
        paired_bootstrap(e, c, e[:-1], c[:-1])


def test_point_biserial_sign():
    x = np.array([0.1, 0.2, 2.0, 2.2])
    assert point_biserial(x, [0, 0, 1, 1]) > 0.9
    assert point_biserial(x, [1, 1, 0, 0]) < -0.9
    assert np.isnan(point_biserial(x, [1, 1, 1, 1]))


def test_token_type():
    assert token_type(" राम") == "devanagari" and token_type("19") == "digit" and token_type("०") == "digit"
    assert token_type(" COVID") == "latin" and token_type("।") == "punct" and token_type(" ") == "space/special"


def test_position_frame_and_tables(mk):
    toks = [" भा", "रत", " एक", " दे", "श", "।"]
    text = "".join(toks)
    t = mk([1, 0, 1, 1, 0, 1], H=[0.1, 2.0, 0.2, 0.3, 1.5, 0.1])
    t.labels.update(token_labels([s.encode() for s in toks]))
    df = position_frame([t], {t.id: text})
    assert list(df.pos_in_word) == [0, 1, 0, 0, 1, 0]
    assert list(df.token_type) == ["devanagari"] * 5 + ["punct"]
    tb = table_by(df, "pos_bucket").set_index("pos_bucket")
    assert tb.loc["1", "match_rate"] == 0 and tb.loc["0", "match_rate"] == 1
    cor = standard_correlations(df, n_boot=50).set_index("pair")
    assert cor.loc["H vs word_initial", "r"] < 0          # signed: entropy is lower word-initially here


def _synthetic_frame(effect: float, n_sent=150, n=40, seed=0):
    """H drives match; word_initial lowers match by `effect` beyond H (or not at all)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sent):
        wi = rng.random(n) < 0.4
        H = rng.gamma(2, 0.5, n) + 0.5 * wi            # word-initial tokens have higher entropy
        p = 1 / (1 + np.exp(-(2.0 - 1.5 * H - effect * wi)))
        rows.append(pd.DataFrame(dict(sid=s, H=H, top1=np.exp(-H), word_initial=wi,
                                      morph_boundary_after=rng.random(n) < 0.4, match=rng.random(n) < p)))
    return pd.concat(rows, ignore_index=True)


def test_conditional_detects_effect_beyond_entropy():
    from lgmtp.analysis import add_morph_start, conditional_by_entropy, logit_beyond_entropy
    for effect, expect_sig in ((1.5, True), (0.0, False)):
        df = add_morph_start(_synthetic_frame(effect))
        tab, s = conditional_by_entropy(df, "word_initial", n_boot=100)
        assert s["raw_diff"] < 0                                   # confounded by H in both cases
        assert (s["hi"] < 0) == expect_sig, (effect, s)
        lg = logit_beyond_entropy(df, ling_cols=("word_initial",), n_boot=30).set_index("quantity")
        d = lg.loc["delta AUC (full - base)"]
        assert (d.lo > 0) == expect_sig, (effect, d.to_dict())


def test_add_morph_start_shift():
    import pandas as pd
    from lgmtp.analysis import add_morph_start
    df = pd.DataFrame(dict(sid=[0, 0, 0, 1, 1], morph_boundary_after=[False, True, False, False, False]))
    assert list(add_morph_start(df).morph_start) == [True, False, True, True, False]
