"""Real heads loop: losslessness, equality with the exact simulator, and the notebook's input handling."""
import json
import math
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from fake_tok import FakeTok  # noqa: E402

from lgmtp import heads_loop as L  # noqa: E402
from lgmtp import refund_sim as R  # noqa: E402
from lgmtp.boundaries import token_labels  # noqa: E402
from lgmtp.refund_sim import Cost, Heads  # noqa: E402
from lgmtp.vocab import token_bytes_list  # noqa: E402
from lgmtp.wordheads import lm_logits, make_heads  # noqa: E402

TOK = FakeTok(merges=["भा", "रत", " भा", "में", "।"])   # the danda is one token piece, as in Llama-3
V = len(TOK)
NO_EOS = [V + 10]
CV = list(range(V))[::-1]                # toy "continuation vocabulary" (reversed: checks the id mapping)


def tiny(seed, layers=2):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    m = LlamaForCausalLM(LlamaConfig(vocab_size=V, hidden_size=32, intermediate_size=48, num_hidden_layers=layers,
                                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                                     tie_word_embeddings=False))
    with torch.no_grad():
        m.lm_head.weight.mul_(120.0)          # peaked distributions: the entropy rule fires only sometimes
    return m.float().eval()


def rand_heads(seed, K=3, d=32):
    torch.manual_seed(seed)
    h = make_heads(d, K)
    for b in h.blocks:
        torch.nn.init.normal_(b.weight, std=0.3)
    torch.nn.init.normal_(h.cont.weight, std=0.5)
    return h.eval()


def greedy(model, prompt, n):
    """Plain greedy decoding without a cache."""
    seq = list(prompt)
    with torch.no_grad():
        for _ in range(n):
            seq.append(int(model(torch.tensor([seq])).logits[0, -1].argmax()))
    return seq[len(prompt):]


@pytest.fixture(scope="module")
def target():
    return tiny(0)


@pytest.fixture(scope="module")
def good_heads(target):
    """Heads fitted to the toy target's outputs, so that some proposals are right (and some are not)."""
    torch.manual_seed(5)
    h = make_heads(32, 3)
    torch.nn.init.normal_(h.cont.weight, std=0.5)
    opt = torch.optim.Adam(h.parameters(), lr=0.02)
    hs, ys = [], []
    for prompt in PROMPTS:
        y = greedy(target, prompt, 63)
        with torch.no_grad():
            hs.append(target.model(torch.tensor([list(prompt) + y])).last_hidden_state[0, len(prompt) - 1:-1][:60])
        ys.append(torch.tensor([[y[t + k] for k in (1, 2, 3)] for t in range(60)]))
    hs, ys = torch.cat(hs), torch.cat(ys)
    for _ in range(60):
        loss = sum(torch.nn.functional.cross_entropy(lm_logits(h.z(hs, k), target.lm_head.weight.detach()), ys[:, k - 1])
                   for k in (1, 2, 3))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return h.eval()


class HashLexicon:
    """Arbitrary but deterministic rule: exercises the filter without needing real words."""

    def is_prefix(self, ids):
        return sum(ids) % 4 != 0


PROMPTS = [[3, 17, 5, 9], [40, 41, 42], [7]]
LAM = -0.5
METHODS = [L.Method("NoDraft", "nodraft"), L.Method("Entropy", "entropy", lam=LAM),
           L.Method("V1", "heads", lam=LAM, tau_h=0.0, tau_c=None),
           L.Method("V1c", "heads", lam=LAM, tau_h=0.2, tau_c=0.4),
           L.Method("V1+lex", "heads", lam=LAM, tau_h=0.0, tau_c=None, lexicon=True)]


@pytest.mark.parametrize("noise", [None, 0.0, 2.0])
def test_lossless_for_every_method(noise, target, good_heads):
    """Output = the target's greedy output, with an unrelated draft (None), with draft = target (0) and with a
    noisy copy (mixed acceptance). Exercises every cache-crop path, with head proposals kept and rejected."""
    draft = tiny(1) if noise is None else target
    dec = L.HeadsDecoder(target, draft, TOK, NO_EOS, "cpu", {"v1": good_heads}, HashLexicon(),
                         np.ones(V, bool), CV, noise=noise or 0.0)
    props, ok, g, acc = 0, 0, 0, 0
    for prompt in PROMPTS:
        ref = greedy(target, prompt, 40)
        for m in METHODS:
            r = dec.decode(prompt, m, 40)
            assert r.out_ids == ref, (m.name, prompt)
            assert all(x["g"] <= 8 and 0 <= x["accepted"] <= x["g"] for x in r.rounds)
            if m.kind == "nodraft":
                assert len(r.rounds) == 40
            if noise == 0.0 and m.name == "Entropy":
                assert all(x["accepted"] == x["g"] for x in r.rounds[:-1])   # the draft is the target
            for x in r.rounds:
                props, ok, g, acc = props + x["props"], ok + x["props_ok"], g + x["g"], acc + x["accepted"]
    assert props > 20 and acc < g                                       # heads used, rejections happened
    if noise == 0.0:
        assert 0 < ok < props                                           # right and wrong proposals


def teacher_forced(draft, heads, prompt, y, noise):
    """Draft argmax / entropy and head predictions along y, i.e. what notebook 07 dumps."""
    with torch.no_grad():
        h = draft.model(torch.tensor([list(prompt) + list(y)])).last_hidden_state[0, len(prompt) - 1:-1]
        lg = lm_logits(h, draft.lm_head.weight)
        if noise:
            ctx = (list(prompt) + list(y))[len(prompt) - 1:-1]
            lg = lg + noise * torch.stack([torch.randn(V, generator=torch.Generator().manual_seed(1_000_003 + int(t)))
                                           for t in ctx])
        lp = torch.log_softmax(lg, -1)
        z = [torch.log_softmax(lm_logits(heads.z(h, k), draft.lm_head.weight[torch.tensor(CV)]), -1).max(-1)
             for k in (1, 2, 3)]
        cv_id = np.stack([np.array(CV)[x.indices.numpy()] for x in z], 1)
        cv_lp = np.stack([x.values.numpy() for x in z], 1)
        cont = torch.sigmoid(heads.cont(h)).numpy()
    return lp.argmax(-1).numpy(), -(lp.exp() * lp).sum(-1).numpy(), Heads(None, None, cv_id, cv_lp, cont)


@pytest.mark.parametrize("m", [x for x in METHODS if x.kind != "nodraft"], ids=lambda x: x.name)
def test_rounds_equal_the_exact_simulator(mk, m, target, good_heads):
    """The live loop commits exactly the tokens per round that refund_sim predicts from teacher-forced
    features: same round starts, same accepted tokens. This is the claim the paper's numbers rest on."""
    heads, noise, n = good_heads, 2.0, 60
    dec = L.HeadsDecoder(target, target, TOK, NO_EOS, "cpu", {"v1": heads}, HashLexicon(), np.ones(V, bool), CV,
                         noise=noise)
    n_props, n_ok, gs = 0, 0, set()
    for si, prompt in enumerate(PROMPTS):
        live = dec.decode(prompt, m, n)
        y = live.out_ids
        argmax, H, hd = teacher_forced(target, heads, prompt, y, noise)
        t = mk(np.ones(n, bool), sid=si, target_ids=np.array(y), draft_argmax=argmax, H=H)
        t.labels["word_initial"] = np.array([dec.words.is_initial(y, j) for j in range(n)])
        t.heads, t._in_cv = {"v1": hd}, np.ones(V, bool)
        p, sim = 0, []
        while p < n:
            if m.kind == "entropy":
                g, ok, _ = R._round_entropy(t, p, Cost(0.3, lambda g: 1.0), m.lam, m.beta)
            else:
                g, ok, _ = R._round_heads(t, p, Cost(0.3, lambda g: 1.0), m.lam, m.beta, "v1",
                                          "lexfilter" if m.lexicon else "cv", m.tau_h, m.tau_c, np.ones(V, bool),
                                          HashLexicon())
            sim.append((p, g, ok))
            p += min(ok + 1, n - p)
        assert [(x["p"], x["accepted"]) for x in live.rounds] == [(p, ok) for p, _, ok in sim]
        # rounds without a rejection have no off-path features, so their draft lengths agree too
        for x, (_, g, ok) in zip(live.rounds, sim):
            if x["accepted"] == x["g"]:
                assert x["g"] == g == ok
        n_props += sum(x["props"] for x in live.rounds)
        n_ok += sum(x["props_ok"] for x in live.rounds)
        gs |= {x["g"] for x in live.rounds}
    assert len(gs) > 3                                # the entropy rule fires at different lengths
    assert (n_props > 10 and 0 < n_ok < n_props) if m.kind == "heads" else n_props == 0


def test_stops_drafting_at_eos(target, good_heads):
    """No token is drafted after an end-of-text token, and the output ends there."""
    ref = greedy(target, PROMPTS[0], 40)
    eos = next(t for i, t in enumerate(ref) if i >= 12 and t not in ref[:i])   # first seen mid-output
    cut = ref.index(eos) + 1
    heads = {"v1": good_heads}
    for m in METHODS:
        dec = L.HeadsDecoder(target, target, TOK, [eos], "cpu", heads, HashLexicon(), np.ones(V, bool), CV)
        r = dec.decode(PROMPTS[0], m, 40)
        assert r.out_ids == ref[:cut]
        if m.name == "Entropy":                       # draft = target, so it drafts the eos itself
            assert r.rounds[-1]["p"] + r.rounds[-1]["g"] <= cut


def test_heads_respect_cap_gate_and_thresholds(target):
    heads = {"v1": rand_heads(2)}
    m = L.Method("V1", "heads", lam=-1e9)             # entropy rule never fires
    dec = L.HeadsDecoder(target, target, TOK, NO_EOS, "cpu", heads, None, np.ones(V, bool), CV)
    r = dec.decode(PROMPTS[0], m, 40)
    assert max(x["g"] for x in r.rounds) == 8 and sum(x["props"] for x in r.rounds) > 0
    off = L.HeadsDecoder(target, target, TOK, NO_EOS, "cpu", heads, None, np.zeros(V, bool), CV)
    assert sum(x["props"] for x in off.decode(PROMPTS[0], m, 40).rounds) == 0
    n = lambda **kw: sum(x["props"] for x in dec.decode(PROMPTS[0], L.Method("V1", "heads", lam=-1e9, **kw), 40).rounds)  # noqa: E731
    assert n(tau_c=1.1) == 0 and n(tau_h=1.0 + 1e-6) == 0
    assert 0 < n(tau_h=0.9) < n(tau_h=0.0)


TEXTS = ["भारत में बहुत भाषाएँ हैं।", "उसने कहा, \"ठीक है\" और 2024 में गया।", "IIT बॉम्बे (मुंबई) अच्छा है। फिर?", "दुःख-सुख क्षेत्रज्ञ"]


@pytest.mark.parametrize("text", TEXTS)
def test_word_tracker_matches_gold_labels(text):
    ids = TOK.encode_one(text)
    gold = token_labels(token_bytes_list(TOK, ids))["word_initial"]
    w = L.WordTracker(TOK)
    for j in range(len(ids)):
        assert w.is_initial(ids[: j + 1], j) == gold[j], (j, text)      # causal: sees tokens <= j only
        j0 = max(i for i in range(j + 1) if gold[i] or i == 0)
        assert w.word(ids[: j + 1]) == ids[j0:j + 1]


def test_word_tracker_survives_invalid_utf8():
    w = L.WordTracker(TOK)
    ids = [0xE0, 0x41, 0xA4, 0xFF, 0x20, 0xE0]
    assert [w.is_initial(ids, j) for j in range(len(ids))][0] is True
    assert w.word(ids) == [0xE0]                      # after the space


def _heads_dir(d, n_train, variants=("v0", "v1")):
    d.mkdir(parents=True)
    for v in variants:
        torch.save(rand_heads(0, d=4).state_dict(), d / f"wordheads_{v}.pt")
    (d / "run_info.json").write_text(json.dumps(dict(cfg=dict(N_TRAIN=n_train))))


def test_find_heads_prefers_most_training_data_and_reads_zips(tmp_path):
    _heads_dir(tmp_path / "in" / "a" / "wordheads", 60000)
    _heads_dir(tmp_path / "src", 180000)
    (tmp_path / "in" / "b").mkdir()
    with zipfile.ZipFile(tmp_path / "in" / "b" / "wordheads.zip", "w") as z:
        for f in (tmp_path / "src").iterdir():
            z.write(f, "wordheads/" + f.name)
    best = L.find_heads([tmp_path / "in"], ["v1"], tmp_path / "work", lambda *_: None)
    assert L._n_train(best) == 180000 and (best / "wordheads_v1.pt").exists()
    with pytest.raises(FileNotFoundError):
        L.find_heads([tmp_path / "in"], ["v2"], tmp_path / "work", lambda *_: None)


def test_tuned_params_from_result_tables():
    root = L.Path(__file__).resolve().parents[1] / "results"
    p = L.tuned_params(root / "wordheads_long" / "main_table.csv")
    assert (p["V1+lex"]["lam"], p["V1+lex"]["tau_h"], p["V1+lex"]["tau_c"]) == (-1.1, 0.4, 0.3)
    assert p["V0"]["tau_c"] is None and p["Entropy"]["lam"] == -0.683
    ms = L.build_methods(["NoDraft", "Entropy", "V1", "V1+lex"], p)
    assert [m.kind for m in ms] == ["nodraft", "entropy", "heads", "heads"] and ms[3].lexicon and not ms[2].lexicon


def test_read_records_drops_cut_off_lines(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"a": 1}\n{"a": 2}\n{"a": ', encoding="utf-8")
    assert L.read_records(f) == [{"a": 1}, {"a": 2}]
    assert L.read_records(tmp_path / "missing.jsonl") == []


def test_summary_numbers():
    def rec(method, i, rep, wall):
        rd = dict(p=0, g=4, accepted=2, calls=2, props=2, props_ok=1, t_draft=0.1, t_verify=0.1)
        return dict(method=method, id=i, rep=rep, wall=wall, prefill=0.5, n_out=3, lossless=True, rounds=[rd])
    res = [rec("NoDraft", i, r, 4.5 + r) for i in (1, 2) for r in range(3)]
    res += [rec("Entropy", i, r, 2.5 + r) for i in (1, 2) for r in range(3)]
    res += [rec("V1", i, r, 2.0 + r) for i in (1, 2) for r in range(3)] + [rec("V1", 3, 0, 99.0)]
    rows = {r["method"]: r for r in L.summarize_loop(res, ["NoDraft", "Entropy", "V1"], {"V1": {"sim": 1.5}})}
    assert rows["NoDraft"]["speedup_wall"] == pytest.approx(1.0)
    assert rows["Entropy"]["speedup_wall"] == pytest.approx(5.5 / 3.5)      # medians over the 3 repeats
    assert rows["V1"]["speedup_wall"] == pytest.approx(5.5 / 3.0)           # sentence 3 has no baseline: ignored
    assert rows["V1"]["speedup_decode_only"] == pytest.approx(5.0 / 2.5)
    assert rows["V1"]["diff_vs_entropy"] == pytest.approx(5.5 / 3.0 - 5.5 / 3.5)
    assert rows["V1"]["head_tokens_accepted"] == 0.5 and rows["V1"]["sim_devtest"] == 1.5
    assert "| V1 |" in L.summary_md(list(rows.values()), dict(heads_tag="180k"))
    assert L.summarize_loop(res, ["NoDraft", "V2"]) == []


def test_entropy_rule_threshold_used_live(target):
    """lam = 1 fires at every positive entropy: exactly one token is drafted per round."""
    dec = L.HeadsDecoder(target, target, TOK, NO_EOS, "cpu", {"v1": rand_heads(2)}, None, np.ones(V, bool), CV)
    r = dec.decode(PROMPTS[0], L.Method("E", "entropy", lam=1.0), 20)
    assert all(x["g"] == 1 and x["calls"] == 1 for x in r.rounds[:-1])
    assert not math.isnan(r.wall)


# ------------------------------------------------------------------ checked drafting
from lgmtp import checked as C  # noqa: E402

CHECKED = [L.Method("NGram", "checked", lam=LAM, variant=None, ngram=True, L=6),
           L.Method("V1+check", "checked", lam=LAM, variant="v1", tau_h=0.2),
           L.Method("NGram+V1", "checked", lam=LAM, variant="v1", tau_h=0.2, ngram=True, L=7)]


@pytest.fixture(scope="module")
def ngram(target):
    """Table built from the toy target's outputs, so that some proposals are right and some are not."""
    return C.NGram.build([greedy(target, p, 60)[:45] for p in PROMPTS] + [list(range(30))] * 2, order=2)


@pytest.mark.parametrize("noise", [None, 0.0, 2.0])
def test_checked_drafting_is_lossless_and_keeps_the_draft_chain(noise, target, good_heads, ngram):
    """Output = the target's greedy output, and every drafted token is the draft's own greedy token: with the
    entropy rule off, checked methods draft exactly what plain drafting does, in fewer 1B calls."""
    draft = tiny(1) if noise is None else target
    dec = L.HeadsDecoder(target, draft, TOK, NO_EOS, "cpu", {"v1": good_heads}, None, np.ones(V, bool), CV,
                         noise=noise or 0.0, ngram=ngram)
    read, agreed = 0, 0
    for prompt in PROMPTS:
        ref = greedy(target, prompt, 40)
        for m in CHECKED:
            r = dec.decode(prompt, m, 40)
            assert r.out_ids == ref, (m.name, prompt)
            assert all(1 <= x["g"] <= 8 and x["g"] == x["calls"] + x["props_ok"] for x in r.rounds if x["calls"])
            read, agreed = read + sum(x["props"] for x in r.rounds), agreed + sum(x["props_ok"] for x in r.rounds)
            free = L.Method(m.name, "checked", lam=-1e9, variant=m.variant, tau_h=m.tau_h, ngram=m.ngram, L=m.L)
            a = dec.decode(prompt, free, 40)
            b = dec.decode(prompt, L.Method("E", "entropy", lam=-1e9), 40)
            assert [(x["p"], x["g"], x["accepted"]) for x in a.rounds] == [(x["p"], x["g"], x["accepted"]) for x in b.rounds]
            assert sum(x["calls"] for x in a.rounds) <= sum(x["calls"] for x in b.rounds)
    assert read > 20 and (agreed > 0 or noise is None) and agreed < read


@pytest.mark.parametrize("m", CHECKED, ids=lambda x: x.name)
def test_checked_rounds_equal_the_exact_simulator(mk, m, target, good_heads, ngram):
    noise, n = 2.0, 60
    dec = L.HeadsDecoder(target, target, TOK, NO_EOS, "cpu", {"v1": good_heads}, None, np.ones(V, bool), CV,
                         noise=noise, ngram=ngram)
    cost, agreed, full = Cost(0.3, lambda g: 1.0), 0, 0
    for si, prompt in enumerate(PROMPTS):
        live = dec.decode(prompt, m, n)
        argmax, H, hd = teacher_forced(target, good_heads, prompt, live.out_ids, noise)
        t = mk(np.ones(n, bool), sid=si, target_ids=np.array(live.out_ids), draft_argmax=argmax, H=H)
        t.heads = {"v1": hd}
        p, sim = 0, []
        while p < n:
            memo = {}
            g, ok, _, calls = C.round_checked(t, p, cost, m.lam, m.beta, m.variant, m.tau_h,
                                              ngram if m.ngram else None, m.L, np.ones(V, bool), memo)
            sim.append((p, g, ok, calls))
            p += min(ok + 1, n - p)
        assert [(x["p"], x["accepted"]) for x in live.rounds] == [(p, ok) for p, _, ok, _ in sim]
        for x, (_, g, ok, calls) in zip(live.rounds, sim):
            if x["accepted"] == x["g"]:                  # never left the target's path
                assert (x["g"], x["calls"]) == (g, calls)
                full += 1
        agreed += sum(x["props_ok"] for x in live.rounds)
    assert agreed > 5 and full > 5


def test_checked_stops_at_eos(target, good_heads, ngram):
    ref = greedy(target, PROMPTS[0], 40)
    eos = next(t for i, t in enumerate(ref) if i >= 12 and t not in ref[:i])
    cut = ref.index(eos) + 1
    for m in CHECKED:
        dec = L.HeadsDecoder(target, target, TOK, [eos], "cpu", {"v1": good_heads}, None, np.ones(V, bool), CV, ngram=ngram)
        r = dec.decode(PROMPTS[0], m, 40)
        assert r.out_ids == ref[:cut] and r.rounds[-1]["p"] + r.rounds[-1]["g"] <= cut


def test_ngram_matches_a_plain_counter(tmp_path):
    from collections import Counter, defaultdict
    rng = np.random.default_rng(0)
    seqs = [list(rng.integers(0, 12, 40) * 1000 + 7) for _ in range(30)] + [[5], []]
    ng = C.NGram.build(seqs, order=3)
    ng.save(tmp_path / "n.npz")
    ng2 = C.NGram.load(tmp_path / "n.npz")
    cnt = {k: defaultdict(Counter) for k in (1, 2, 3)}
    for s in seqs:
        for k in cnt:
            for i in range(len(s) - k):
                cnt[k][tuple(s[i:i + k])][s[i + k]] += 1

    def slow(ctx):
        for k in (3, 2, 1):
            c = cnt[k].get(tuple(ctx[-k:])) if len(ctx) >= k else None
            if c and sum(c.values()) >= 2:
                return min(c, key=lambda x: (-c[x], x))
        return None

    for _ in range(300):
        ctx = list(rng.integers(0, 13, rng.integers(0, 6)) * 1000 + 7)
        assert ng.next(ctx) == ng2.next(ctx) == slow(ctx), ctx
    assert ng.next([5]) is None and ng.next([]) is None and ng.next([999]) is None
    assert C.proposal([7], [1, 2], None, 6, 1) == [1] and C.proposal([7], [], None, 6, 5) == []
    assert len(C.proposal([7, 1007], [], ng, 6, 4)) == 4


def test_checked_params_and_methods():
    root = L.Path(__file__).resolve().parents[1] / "results" / "checked"
    for tag, p in json.loads((root / "params.json").read_text()).items():
        ms = L.build_methods(list(L.CHECKED_ROWS), p)
        assert all(m.kind == "checked" and (m.ngram or m.variant) for m in ms)
        assert L.variant_of("NGram") is None and L.variant_of("NGram+V0") == "v0" and L.variant_of("V1+lex") == "v1"
