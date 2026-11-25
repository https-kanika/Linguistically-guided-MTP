import numpy as np

from fake_tok import FakeTok
from lgmtp import uni_sim as U
from lgmtp.uni_features import TokTable, featurize_records, make_encoder, position_targets, seq_text
from lgmtp.unichar import UnitVocab, unit_strings

TEXTS = ["भारत में आज नीति लागू हुई।", "आज बारिश हुई और नदी भर गई।"]
MERGES = ["भारत", " में", " आज", "ति", " हुई", "।", " और"]
TIMING = dict(t_target=0.07, t_draft=0.0226, c=0.3231,
              t_target_verify={str(g): 0.07 * v for g, v in zip(range(1, 9),
                                                               (1.020, 1.025, 1.034, 1.043, 1.060, 1.063, 1.066, 1.074))})


def setup(merges=MERGES):
    tok = FakeTok(merges)
    V = UnitVocab(sum((unit_strings(t) for t in TEXTS), []))
    table = TokTable(tok, [tok.eos_token_id])
    recs = []
    for i, t in enumerate(TEXTS):
        ids = tok.encode_one(t) + [tok.eos_token_id]
        n = len(ids)
        tk = np.tile(np.arange(20), (n, 1))
        tk[:, 0] = ids
        recs.append(dict(id=i, prompt_text="", prompt_ids=[1, 2, 3], out_ids=ids, eos_end=True,
                         topk_ids=tk, topk_logp=np.full((n, 20), -3.0, np.float32)))
    return tok, V, table, recs


def truth_W(fs):
    W = fs.tgt.copy()
    W[W < 0] = -1
    for i in range(1, W.shape[1]):
        W[:, i] = np.where(W[:, i - 1] < 0, -1, W[:, i])
    return W


def test_verify_extrapolates():
    v = U.Verify(TIMING)
    assert v(0) == 1.0 and abs(v(1) - 1.02) < 1e-9 and abs(v(8) - 1.074) < 1e-9
    assert v(12) > v(8) and v(16) > v(12)


def test_truth_units_convert_exactly():
    """Drafting the true next units must reproduce the true target tokens (exact conversion)."""
    for merges in (MERGES, []):                       # [] = pure bytes: many mid-char cuts
        tok, V, table, recs = setup(merges)
        fs = featurize_records(recs, table, V)
        seqs = U.build_simseqs(recs, table, V)
        drafts, st = U.convert_all(seqs, truth_W(fs), V, make_encoder(tok), table.bytes)
        assert st["failed"] == 0, st
        for s, seq in enumerate(seqs):
            for p in range(seq.n):
                for d in drafts[s][p][1:]:
                    assert d == seq.out_ids[p:p + len(d)].tolist(), (merges, s, p, d)


def test_simulate_perfect_and_empty_drafts():
    tok, V, table, recs = setup()
    fs = featurize_records(recs, table, V)
    seqs = U.build_simseqs(recs, table, V)
    W = truth_W(fs)
    drafts, _ = U.convert_all(seqs, W, V, make_encoder(tok), table.bytes)
    v = U.Verify(TIMING)
    R = U.simulate(seqs, drafts, np.ones_like(W, np.float32), 0.0, 0.05, v)
    n = sum(s.n for s in seqs)
    assert R["sent_emitted"].sum() == n
    assert R["tokens_per_round"] > 2 and R["speedup"] > 1.5
    assert R["acceptance"] == 1.0
    Rn = U.simulate(seqs, drafts, np.zeros_like(W, np.float32), 0.5, 0.05, v)   # threshold never met
    assert Rn["mean_draft_len"] == 0 and abs(Rn["speedup"] - 1 / 1.05) < 1e-9
    Ro = U.simulate(seqs, drafts, np.zeros_like(W, np.float32), 0.5, 0.05, v, oracle_level=True)
    assert Ro["speedup"] >= R["speedup"] - 1e-9


def test_choose_level():
    assert U.choose_level([0.9, 0.8, 0.5], 3, 0.5) == 2
    assert U.choose_level([0.4, 0.9], 2, 0.5) == 0
    assert U.choose_level([0.9, 0.9, 0.9], 1, 0.0) == 1


def test_baselines_and_coverage():
    tok, V, table, recs = setup()
    seqs = U.build_simseqs(recs, table, V)
    for dr in (U.pld_drafts(seqs), U.logitspec_drafts(seqs)):
        assert len(dr) == len(seqs) and all(len(r) == s.n for r, s in zip(dr, seqs))
        R = U.simulate(seqs, dr, U.ones_like_probs(dr), 0.0, 0.0, U.Verify(TIMING))
        assert R["sent_emitted"].sum() == sum(s.n for s in seqs)
    cov = U.next_next_coverage(seqs)
    assert 0 <= cov["top1"] <= cov["top20"] <= 1


def test_unit_agreement_self_is_perfect():
    tok, V, table, recs = setup()
    t = TEXTS[0]
    ids = tok.encode_one(t)
    c = U.unit_agreement(t, ids, ids, table)
    a = U.agreement_table(c)
    assert a["p_start"] == 1 and a["p_unit"] == 1 and a["p_cont_token"] == 1
    bad = list(ids)
    bad[0] = 0
    a2 = U.agreement_table(U.unit_agreement(t, ids, bad, table))
    assert a2["p_start"] < 1


def test_ngram_index_matches_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(20):
        prompt = rng.integers(0, 5, rng.integers(0, 12)).tolist()
        out = rng.integers(0, 5, 30).tolist()
        ix = U._NgramIndex(prompt, (1, 2, 3))
        for t in out:
            for n in (1, 2, 3):
                if len(ix.ctx) > n:
                    key = tuple(ix.ctx[-n:])
                    assert ix.lookup(key, 4) == U._lookup(ix.ctx, key, 4)
            for key in [(0, 1), (2,), (3, 3, 3), (4, 0)]:
                assert ix.lookup(key, 4) == U._lookup(ix.ctx, key, 4)
            ix.append(t)
