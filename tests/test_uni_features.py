import numpy as np
import pytest

from fake_tok import FakeTok
from lgmtp.uni_features import (TokTable, featurize_records, featurize_texts, make_batch, position_targets, seq_text,
                                utf8_offsets)
from lgmtp.unichar import EOS, PEND, UnitVocab, unit_strings

TEXT = "भारत में आज नीति लागू हुई।"
MERGES = ["भारत", " में", " आज", " नी", "ति", " लागू", " हुई", "।"]


def setup():
    tok = FakeTok(MERGES)
    V = UnitVocab(unit_strings(TEXT) + [" भारतीय", " नीतियाँ"])
    table = TokTable(tok, [tok.eos_token_id])
    return tok, V, table


def record(tok, text=TEXT, seed=0, eos=True):
    ids = tok.encode_one(text) + ([tok.eos_token_id] if eos else [])
    rng = np.random.default_rng(seed)
    n = len(ids)
    tk = rng.integers(0, len(tok), (n, 20))
    tk[:, 0] = ids
    lp = np.sort(rng.normal(-3, 1, (n, 20)))[:, ::-1].astype(np.float32)
    lp[:, 0] = -0.1
    return dict(id=seed, prompt_text="Translate: India adopted the policy today.", out_ids=ids, eos_end=eos,
                topk_ids=tk, topk_logp=lp)


def test_utf8_offsets():
    t, o = utf8_offsets("aक".encode())
    assert t == "aक" and o.tolist() == [0, 1, 4]
    t, o = utf8_offsets("aक".encode()[:-1])            # truncated last char is left out
    assert t == "a" and o.tolist() == [0, 1]
    t, o = utf8_offsets(b"a\xffb")
    assert t == "a�b" and o.tolist() == [0, 1, 2, 3]


def test_targets_mid_word_and_mid_char():
    tok, V, table = setup()
    # " हुई" split as " हु" + bytes: force mid-char cuts with a tokenizer lacking the merge
    tok2 = FakeTok([m for m in MERGES if m != " हुई"])
    table2 = TokTable(tok2, [tok2.eos_token_id])
    ids = tok2.encode_one(TEXT) + [tok2.eos_token_id]
    st = seq_text(table2, ids)
    ps, lo, hi, tg = position_targets(st, V, True)
    hu = V.id(" हुई")
    rows = [p for p in range(len(ids)) if tg[p, 0] == hu]
    assert rows, "positions inside ' हुई' must target it"
    for p in rows:
        assert lo[p] <= hu < hi[p]
        assert st.out_bytes[ps[p]:].startswith(" हुई".encode())
    assert any(st.pend[p] > 0 for p in rows)               # some cuts fall inside a UTF-8 char
    last = len(ids) - 1                                     # before eos: current unit '।' then <eos>
    assert tg[last, 0] == V.id("।") and tg[last, 1] == V.eos_id and tg[last, 2] == -100
    assert tg[0, 0] == V.id("भारत") and tg[0, 1] == V.id(" में")


def test_causality_of_batch_inputs():
    """Inputs at position p are identical for two outputs that agree on out_ids[:p] and topk[:p]."""
    tok, V, table = setup()
    a = record(tok, TEXT, 0)
    b_text = "भारत में आज बारिश हुई और"
    b = record(tok, b_text, 0)
    ids_a, ids_b = a["out_ids"], b["out_ids"]
    p = next(i for i in range(min(len(ids_a), len(ids_b))) if ids_a[i] != ids_b[i])
    b["topk_ids"][:p], b["topk_logp"][:p] = a["topk_ids"][:p], a["topk_logp"][:p]
    rng = np.random.default_rng(5)
    b["topk_ids"][p:] = rng.integers(0, len(tok), b["topk_ids"][p:].shape)
    fa = featurize_records([a], table, V)
    fb = featurize_records([b], table, V)
    for q in range(p + 1):
        ba, bb = make_batch(fa, [q]), make_batch(fb, [q])
        for k in ba:
            if k in ("tgt", "lo", "hi"):
                continue
            np.testing.assert_array_equal(ba[k], bb[k], err_msg=f"{k} differs at p={q} (first diff {p})")


def test_candidate_features():
    tok, V, table = setup()
    r = record(tok)
    # make the top-2 candidate at token 0 a longer string that extends 'भारत' -> ' भारतीय'-like continuation
    fs = featurize_records([r], table, V)
    b = make_batch(fs, np.arange(fs.n_pos))
    assert b["cand_u"].shape == (fs.n_pos, 60, 8) and b["cand_f"].shape[-1] == 10
    assert not b["cand_mask"][0].any()                       # p = 0: no distribution yet
    assert b["cand_mask"][1, :20].any() and not b["cand_mask"][1, 20:].any()
    # lag-1 argmax candidate is the committed token itself: consistent, does not extend
    f = b["cand_f"][1, 0]
    assert f[-2] == 1 and f[-1] == 0
    eos_row = fs.n_pos - 1
    assert (b["ctx_u"][eos_row] != 0).sum() > 0


def test_eos_candidate_and_pend_marker():
    tok, V, table = setup()
    tok2 = FakeTok([m for m in MERGES if m != " हुई"])
    table2 = TokTable(tok2, [tok2.eos_token_id])
    r = record(tok2)
    r["topk_ids"][-2, 1] = tok2.eos_token_id
    fs = featurize_records([r], table2, V)
    assert fs.cand_u[len(r["out_ids"]) - 2, 1, 0] == EOS
    b = make_batch(fs, np.arange(fs.n_pos))
    assert (b["ctx_u"][:, -1] == PEND).any()


def test_featurize_texts():
    _, V, _ = setup()
    fs = featurize_texts([TEXT, "", "भारत"], V, cuts_per_text=5, seed=1)
    assert fs.n_pos > 0 and not fs.has_cands
    b = make_batch(fs, np.arange(fs.n_pos))
    assert "cand_u" not in b and b["ctx_u"].shape[0] == fs.n_pos


def test_save_load_concat(tmp_path):
    from lgmtp.uni_features import FeatureSet, concat
    tok, V, table = setup()
    fs = featurize_records([record(tok, TEXT, 0), record(tok, "आज बारिश हुई।", 1)], table, V)
    fs.save(tmp_path / "f.npz")
    g = FeatureSet.load(tmp_path / "f.npz")
    np.testing.assert_array_equal(g.cand_u, fs.cand_u)
    h = concat([fs, g])
    assert h.n_pos == 2 * fs.n_pos
    b1 = make_batch(fs, [fs.n_pos - 1])
    b2 = make_batch(h, [2 * fs.n_pos - 1])
    for k in b1:
        np.testing.assert_array_equal(b1[k], b2[k])


def test_inputs_do_not_depend_on_token_ids():
    """Same strings under a permuted vocabulary give identical head inputs (no token id reaches the head)."""
    tok1, V, _ = setup()
    tok2 = FakeTok(list(reversed(MERGES)))
    assert tok1.vocab != tok2.vocab
    r1 = record(tok1)
    remap = {i: tok2.vocab.index(t) for i, t in enumerate(tok1.vocab)}
    r2 = dict(r1, out_ids=[remap[i] for i in r1["out_ids"]],
              topk_ids=np.vectorize(remap.get)(r1["topk_ids"]))
    f1 = featurize_records([r1], TokTable(tok1, [tok1.eos_token_id]), V)
    f2 = featurize_records([r2], TokTable(tok2, [tok2.eos_token_id]), V)
    b1, b2 = make_batch(f1, np.arange(f1.n_pos)), make_batch(f2, np.arange(f2.n_pos))
    assert set(b1) == set(b2)
    for k in b1:
        np.testing.assert_array_equal(b1[k], b2[k], err_msg=k)
