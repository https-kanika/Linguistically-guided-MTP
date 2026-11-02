import json

import numpy as np

from lgmtp.policies import MorphEntropy, MorphOnly
from lgmtp.tuning import (ENTROPY_LAMS, FAMILIES, Context, evaluate, grid, lam_neighbourhood, make_policy,
                          tune_all, write_selected)
from test_policies_causality import prefix_model, synth, table


def test_lam_neighbourhood():
    assert len(lam_neighbourhood(ENTROPY_LAMS[20])) == 11
    assert lam_neighbourhood(ENTROPY_LAMS[0]) == ENTROPY_LAMS[:6]
    assert lam_neighbourhood(ENTROPY_LAMS[-1]) == ENTROPY_LAMS[-6:]


def test_grid_sizes():
    ctx = Context(entropy_best=dict(lam=ENTROPY_LAMS[20], beta=1.0))
    assert len(grid("Entropy", ctx)) == 160 and len(grid("FixedK", ctx)) == 8
    assert len(grid("WordBoundarySoft", ctx)) == 57 and len(grid("MorphOnly", ctx)) == 57
    assert len(grid("AksharaSafeEntropy", ctx)) == 44 * 12
    assert len(grid("Hybrid", ctx)) == 44 * 16
    assert len(grid("MorphEntropy", ctx)) == 44 * 6 * 4 * 3


def test_tune_and_evaluate_end_to_end(mk, tmp_path):
    from lgmtp.morph import token_morph_labels, TieredLabeler, GoldLexicon
    tb = table()
    dev = [synth(mk, 50, s) for s in range(6)]
    test = [synth(mk, 50, 100 + s) for s in range(6)]
    for t in test:  # add morph labels so the morph-boundary column is filled
        text = b"".join(tb.token_bytes[i] for i in t.target_ids).decode()
        t.labels.update(token_morph_labels(t.labels, text, TieredLabeler(GoldLexicon(), None)))
    ctx = Context(newword_mask=tb.word_start, prefix_model=prefix_model(), table=tb)
    sel = tune_all(dev, 0.33, lambda g: 1.0, ctx, out_dir=tmp_path, progress=False, n_jobs=1)
    assert set(sel) == set(FAMILIES)
    assert (tmp_path / "grid_MorphEntropy.csv").exists()
    write_selected(sel, tmp_path / "sel.json")
    sel2 = json.loads((tmp_path / "sel.json").read_text(encoding="utf-8"))["selected"]
    for fam, params in sel2.items():
        make_policy(fam, params, ctx)          # JSON round trip rebuilds every policy
    df, sims = evaluate(test, sel2, 0.33, lambda g: 1.0, ctx, n_boot=50)
    assert len(df) == len(FAMILIES)
    row = df.set_index("policy")
    assert row.loc["Oracle", "speedup_charged"] >= row.loc["FixedK", "speedup_charged"]
    assert abs(row.loc["Entropy", "diff_vs_entropy"]) < 1e-12
    assert row.loc["NoDraft", "speedup_charged"] == 1.0
    assert np.isfinite(row.loc["FixedK", "frac_ending_at_morph_boundary"])
    lk = row[row.lookahead == 1]
    assert (lk.speedup_uncharged >= lk.speedup_charged - 1e-12).all()


def test_parallel_grid_equals_serial(mk):
    import copy
    from lgmtp.simulator import make_verify
    from lgmtp.tuning import run_grid
    tb = table()
    dev = [synth(mk, 40, s) for s in range(4)]
    v = make_verify(dict(t_target=1.0, t_target_verify={"1": 1.1, "8": 1.4}))
    ctx = Context(newword_mask=tb.word_start, prefix_model=prefix_model(), table=tb,
                  entropy_best=dict(lam=ENTROPY_LAMS[20], beta=1.0))
    for fam in ("Entropy", "MorphOnly"):
        a = run_grid(fam, dev, 0.33, v, ctx, n_jobs=1)
        b = run_grid(fam, copy.deepcopy(dev), 0.33, v, ctx, n_jobs=2, parallel_min=1)
        assert a.equals(b), fam


def test_prefix_cache_does_not_change_results(mk):
    import copy
    from lgmtp.simulator import simulate
    tb, pm = table(), prefix_model()
    trs = [synth(mk, 60, s) for s in range(5)]
    fresh = copy.deepcopy(trs)
    pols = [MorphEntropy(0.2, 1.0, 0.3, True, 0.5, 1, pm, tb), MorphOnly(0.3, 1, pm, tb),
            MorphEntropy(0.0, 2.0, 0.2, False, 0.5, 2, pm, tb)]
    for pol in pols:
        r1 = simulate(trs, pol, 8, 0.33)          # fills / uses the cache
        r2 = simulate(trs, pol, 8, 0.33)          # warm cache
        r3 = simulate(copy.deepcopy(fresh), pol, 8, 0.33)   # no cache
        assert r1.records == r2.records == r3.records
