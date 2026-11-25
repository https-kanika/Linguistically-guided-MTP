"""CPU smoke test of the Kaggle notebook code paths (PLAN rule 2a).

    python scripts/smoke_cpu.py --part 01      # notebook 01: generation, teacher-forced passes, timing, zip
    python scripts/smoke_cpu.py --part 03      # notebook 03: real speculative loop (added in step S5)
    python scripts/smoke_cpu.py --part w1      # notebook 05: week-1 universal head, all stages, tiny models

Uses meta-llama/Llama-3.2-1B-Instruct as both target and draft (fp32, loaded once), 3 sentences per
split, MAX_NEW=24. Outputs go to data/smoke/ (gitignored).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lgmtp import gpu_traces  # noqa: E402
from lgmtp.traces import load_traces, summary  # noqa: E402

SMOKE_ROOT = ROOT / "data" / "smoke"


def part01() -> None:
    out_root = SMOKE_ROOT / "01"
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ("dev", "devtest"):
        t0 = time.time()
        cfg = dict(SPLIT=split, N=3, BATCH=2, MAX_NEW=24, SMOKE=True, OUT_ROOT=str(out_root),
                   GIT_HASH="smoke", TIMING=dict(steps=3, reps=1, max_g=8))
        res = gpu_traces.notebook01_main(cfg)

        # unzip as the user would, then load
        dest = out_root / "unzipped"
        with zipfile.ZipFile(res["zip"]) as z:
            names = sorted(z.namelist())
            z.extractall(dest)
        print("zip contents:", names)
        assert names == sorted([f"traces_{split}.npz", f"target_outputs_{split}.jsonl", "timing.json",
                                f"run_info_{split}.json"]), names
        traces, meta = load_traces(dest / f"traces_{split}.npz")
        S = summary(traces)
        print("loaded:", S)
        assert len(traces) == 3 and meta["split"] == split
        for t in traces:
            assert t.n <= 24 and t.draft_topk_ids.shape == (t.n, 20)
            assert np.all((t.p_newword >= 0) & (t.p_newword <= 1 + 1e-4))
            np.testing.assert_allclose(t.p_newword, t.p_space + t.p_punct + t.p_eos, atol=1e-5)
            assert np.all(t.top1 >= t.top2)
        # draft == target: the teacher-forced argmax reproduces greedy generation
        assert S["agreement"] >= 0.95, S
        assert res["target_check"]["rate"] >= 0.95, res["target_check"]["rate"]
        # resume: a second call must not regenerate or recompute anything
        t1 = time.time()
        gpu_traces.notebook01_main(cfg)
        print(f"{split}: first run {t1 - t0:.0f}s, resumed run {time.time() - t1:.0f}s")

    # timing was measured once and reused by the second split
    import json
    td = json.loads((out_root / "traces_v2_dev" / "timing.json").read_text())
    tt = json.loads((out_root / "traces_v2_devtest" / "timing.json").read_text())
    assert td == tt, "devtest should reuse the timing measured in the dev run"

    # RESUME_FROM: a fresh session (empty OUT_ROOT) continues from an earlier run's outputs
    fresh = SMOKE_ROOT / "01_fresh"
    if fresh.exists():
        shutil.rmtree(fresh)
    t2 = time.time()
    res = gpu_traces.notebook01_main(dict(SPLIT="dev", N=3, BATCH=2, MAX_NEW=24, SMOKE=True,
                                          OUT_ROOT=str(fresh), GIT_HASH="smoke", RESUME_FROM=str(out_root),
                                          TIMING=dict(steps=3, reps=1, max_g=8)))
    assert Path(res["zip"]).exists()
    print(f"RESUME_FROM run: {time.time() - t2:.0f}s")
    print("\nSMOKE 01 PASSED")


SMOKE_SELECTED = {
    "NoDraft": {}, "FixedK": {"k": 3}, "Entropy": {"lam": 0.2, "beta": 1.0},
    "WordBoundarySoft": {"tau": 0.5, "min_g": 1},
    "Hybrid": {"lam": 0.2, "beta": 1.0, "tau_w": 0.4, "max_ext": 2},
    "MorphEntropy_ext1": {"lam": 0.2, "beta": 1.0, "tau_morph": 0.3, "use_akshara": True, "tau_m": 0.5,
                                 "max_ext": 1},
}


def part03(selected_path: str | None = None, noise: float = 0.0) -> None:
    """noise > 0: the draft gets deterministic logit noise (spec_loop.NoisyDraft), so drafts are rejected
    and the cache-crop / rejection path of the real loop runs (draft == target never rejects)."""
    import json
    import pickle
    from lgmtp import spec_loop
    from lgmtp.gpu_traces import load_flores_pairs, read_jsonl
    from lgmtp.morph import GoldLexicon, PrefixBoundaryModel, TieredLabeler, build_prefix_model, load_morphtok
    from lgmtp.simulator import simulate
    from lgmtp.tuning import Context, make_policy

    out_root = SMOKE_ROOT / ("03_noisy" if noise else "03")
    if out_root.exists():
        shutil.rmtree(out_root)
    if selected_path:   # exercise the real artifacts that notebook 03 will embed
        selected = json.loads(Path(selected_path).read_text(encoding="utf-8"))["selected"]
        pm = PrefixBoundaryModel.load(ROOT / "results" / "phase1" / "prefix_model.pkl.gz")
    else:
        selected = SMOKE_SELECTED
        df, _ = load_morphtok(ROOT / "data" / "morphtok" / "hindi.csv")
        lex = GoldLexicon.from_dataframe(df)
        pm = build_prefix_model(lex, TieredLabeler(lex, None), [], min_count=5)
    state = dict(min_count=pm.min_count, pos={k: dict(v) for k, v in pm.pos.items()},
                 tot={k: dict(v) for k, v in pm.tot.items()}, g_pos=pm.g_pos, g_tot=pm.g_tot, sources=pm.sources)
    ids = [r[0] for r in load_flores_pairs("devtest", 3)]
    out_root.mkdir(parents=True, exist_ok=True)
    pm.save(out_root / "prefix_model.pkl.gz")   # used by the notebook 04 dry run
    t0 = time.time()
    res = spec_loop.notebook03_main(dict(SMOKE=True, OUT_ROOT=str(out_root), N_SENT=3, REPEATS=2, MAX_NEW=24,
                                         SELECTED=selected, PREFIX_MODEL_STATE=state, IDS=ids, GIT_HASH="smoke",
                                         SMOKE_DRAFT_NOISE=noise))
    print(f"part 03 run: {time.time() - t0:.0f}s")
    with zipfile.ZipFile(res["zip"]) as z:
        names = sorted(z.namelist())
        z.extractall(out_root / "unzipped")
    print("zip contents:", names)
    assert res["n_lossless"] == res["n_results"] == 3 * 2 * len(spec_loop.LIVE_FAMILIES), res
    results = read_jsonl(out_root / "unzipped" / "loop_results.jsonl")
    rounds = [rd for r in results for rd in r["rounds"]]
    n_rej = sum(rd["accepted"] < rd["g"] for rd in rounds)
    if noise:
        # rejections happened, so caches were cropped, and every output is still identical to greedy
        print(f"rounds with a rejection: {n_rej} / {len(rounds)}")
        assert n_rej > 0, "noise too small: no rejection, crop path not exercised"
    else:
        # draft == target: every drafted token is accepted
        assert n_rej == 0, "draft == target must never reject"

    # simulator vs live on the teacher-forced traces of the same outputs (notebook 04 in miniature)
    traces, _ = load_traces(out_root / "unzipped" / "traces_real100.npz")
    from lgmtp.policies import TokenTable
    from lgmtp.gpu_traces import load_tokenizer
    from lgmtp.vocab import build_masks
    tok = load_tokenizer("meta-llama/Llama-3.2-1B-Instruct")
    eos_ids = json.loads((out_root / "unzipped" / "traces_meta.json").read_text())["eos_ids"]         if (out_root / "unzipped" / "traces_meta.json").exists() else load_traces(out_root / "unzipped" / "traces_real100.npz")[1]["eos_ids"]
    masks = build_masks(tok, eos_ids)
    ctx = Context(newword_mask=masks.starts_newword, prefix_model=pm, table=TokenTable.from_tokenizer(tok, masks))
    by = {t.id: t for t in traces}
    for fam in spec_loop.LIVE_FAMILIES:
        pol = make_policy(fam, selected.get(fam, {}), ctx)
        cap = selected[fam]["k"] if fam == "FixedK" else 8
        sim = simulate(traces, pol, cap, 0.3)
        sim_g = {}
        for sid, p, g, a, _ in sim.records:
            sim_g.setdefault(sid, []).append(g)
        live = [r for r in results if r["policy"] == fam and r["rep"] == 0]
        agree = []
        for r in live:
            lg = [rd["g"] for rd in r["rounds"]]
            sg = sim_g[r["id"]]
            # identical except the last round: the simulator caps at n - p, the live loop at max_new - p.
            # With a noisy draft, dynamic policies may legitimately differ after a mismatch (teacher-forcing
            # approximation); fixed-length policies must still agree exactly.
            if not noise or fam in ("NoDraft", "FixedK"):
                assert lg[:-1] == sg[:-1], (fam, r["id"], lg, sg)
            agree.append(lg[:-1] == sg[:-1])
        print(f"{fam:18s} live vs simulated draft lengths identical in {sum(agree)}/{len(agree)} sentences")
    print("\nSMOKE 03 PASSED")


# --------------------------------------------------------------------------- #
# week 1 (notebook 05): every stage with tiny ungated models on CPU
# --------------------------------------------------------------------------- #

W1_HI = ["भारत में आज नई नीति लागू हुई।", "सरकार ने कहा कि योजना अगले साल शुरू होगी।",
         "बारिश के कारण कई सड़कें बंद हो गईं।", "उन्होंने बताया कि बच्चे स्कूल जा रहे हैं।",
         "यह फ़ैसला सभी के लिए महत्वपूर्ण है।", "वैज्ञानिकों ने एक नई खोज की घोषणा की।",
         "शहर में यातायात की समस्या बढ़ रही है।", "किसानों को फसल का उचित मूल्य मिलना चाहिए।",
         "मैच के बाद खिलाड़ियों ने दर्शकों का धन्यवाद किया।", "नदी का जल स्तर तेज़ी से बढ़ा है।"]
W1_EN = ["The government announced a new policy today.", "Heavy rain closed several roads in the city.",
         "Scientists have discovered a new species of frog.", "The children are going to school by bus.",
         "Farmers should get a fair price for their crops.", "The river level rose quickly after the storm.",
         "The players thanked the crowd after the match.", "This decision is important for everyone.",
         "Traffic problems are growing in the capital.", "The plan will start next year, officials said.",
         "He said the bridge would be repaired soon."]


def part_w1() -> None:
    import json
    import random
    from lgmtp import collect as C
    from lgmtp.traces import Trace, save_traces
    from lgmtp.week1 import phase1_constants, week1_main

    out_root = SMOKE_ROOT / "w1"
    if out_root.exists():
        shutil.rmtree(out_root)
    rng = random.Random(0)
    mono = []
    for _ in range(60):                      # tiny synthetic Hindi corpus (vocab + text-only pretraining)
        mono.append(" ".join(rng.sample(W1_HI, 3)))
    rows = [[i, s] for i, s in enumerate(W1_EN)]
    sources = dict(dev=rows[:4], devtest=rows[4:7], extra=[[1_000_000 + i, s] for i, s in rows[7:]], mono=mono + W1_HI)
    models = [
        dict(key="smol135", names=["HuggingFaceTB/SmolLM2-135M-Instruct"], dtype="fp32", family="smol"),
        dict(key="qwen0.5", names=["Qwen/Qwen2.5-0.5B-Instruct"], dtype="fp32", family="qwen"),
        dict(key="smol135base", names=["HuggingFaceTB/SmolLM2-135M-Instruct"], dtype="fp32", family="llama3",
             prompt_mode="base"),
    ]
    # fake "Phase-1 8B traces" built from Qwen-0.5B outputs (exercises the attached-input path of S4)
    root = out_root / "week1"
    qd = root / "data" / "qwen0.5"
    res, info, _ = C.collect_model(models[1], {s: [tuple(r) for r in sources[s]] for s in ("dev", "extra", "devtest")},
                                   qd, 2, 24, "cpu", 20)
    p1 = out_root / "phase1_input" / "traces_v2_devtest"
    p1.mkdir(parents=True)
    t_t = 0.0706
    for split in ("dev", "devtest"):
        traces = []
        for r in res[split]:
            n = len(r["out_ids"])
            z = np.zeros(n, np.float32)
            ids = np.asarray(r["out_ids"], np.int64)
            traces.append(Trace(id=r["id"], n=n, target_ids=ids, draft_argmax=ids, H=z, top1=z + 1, top2=z, p_target=z + 1,
                                p_space=z, p_punct=z, p_eos=z, p_newword=z, p_mark=z, p_cont=z,
                                draft_topk_ids=r["topk_ids"], draft_topk_logp=r["topk_logp"],
                                target_topk_ids=r["topk_ids"], target_topk_logp=r["topk_logp"],
                                target_logp_of_draft_argmax=z))
        save_traces(p1 / f"traces_{split}.npz", traces, dict(split=split, eos_ids=info["eos_ids"]))
        with open(p1 / f"target_outputs_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in res[split]:
                f.write(json.dumps(dict(id=r["id"], src=r["src"], prompt_ids=r["prompt_ids"], out_ids=r["out_ids"]),
                                   ensure_ascii=False) + "\n")
    (p1 / "timing.json").write_text(json.dumps(dict(t_target=t_t, t_draft=t_t * 0.3231, c=0.3231,
                                                    t_target_verify={str(g): t_t * (1 + 0.01 * g) for g in range(1, 9)})))
    cfg = dict(OUT_ROOT=str(out_root), SMOKE=True, GIT_HASH="smoke", MODELS=models, SOURCES=sources,
               MAX_NEW=24, GEN_BATCH=2, VOCAB_MIN_COUNT=1, ARCH=dict(d=64, layers=1, nhead=2),
               PRE_STEPS=20, FT_STEPS=20, INMODEL_STEPS=10, BATCH=16, N_AGREE=3, N_BOOT=50, TAUS=[0.0, 0.1, 0.5],
               PHASE1_DIR=str(out_root / "phase1_input"), PHASE1_SEARCH=[], TARGET8B=["Qwen/Qwen2.5-0.5B-Instruct"],
               COLLECT_8B_IF_MISSING=False, C_SENS=[0.0, 0.05], **phase1_constants(ROOT / "results" / "phase1"))
    t0 = time.time()
    out = week1_main(cfg)
    print(f"week1 smoke run: {time.time() - t0:.0f}s, verdict {out['verdict']['decision']}")
    rd = root / "results"
    for f in ("collection_summary.csv", "agreement.csv", "lomo.csv", "lomo_sim.csv", "l8.csv", "l8_heads.csv",
              "l8_tau.csv", "l8_sens.csv", "conv.csv", "nn_cov.csv", "verdict.json", "SUMMARY.md", "timing_head.json",
              "vocab_coverage.csv", "run_info.json"):
        assert (rd / f).exists(), f
    import pandas as pd
    l8 = pd.read_csv(rd / "l8.csv")
    print(l8[["method", "speedup", "ci_lo", "ci_hi", "tokens_per_round"]].to_string())
    meths = set(l8["method"])
    for m in ("Entropy (Phase 1, 1B draft)", "Universal head, zero-shot (all small LLMs)",
              "Text-only head, zero-shot (ablation: no distribution input)", "In-model head (8B dev only)",
              "Truth units (upper bound of this head design)"):
        assert m in meths, m
    ent = l8.set_index("method").loc["Entropy (Phase 1, 1B draft)", "speedup"]
    assert ent > 1.0                                    # fake traces: draft == target, every draft accepted
    truth = l8.set_index("method").loc["Truth units (upper bound of this head design)"]
    # tiny models write little real Hindi (low vocabulary coverage), so few positions have a draft;
    # the drafts that exist must be accepted (only a longest level's final token may be rejected)
    assert truth["acceptance"] > 0.85, truth["acceptance"]
    conv = pd.read_csv(rd / "conv.csv")
    print(conv.to_string())
    tr = conv[(conv.method == "truth_units")]
    assert (tr["failed"] <= 0.02 * tr["positions"]).all(), "truth-unit drafts should convert exactly"
    with zipfile.ZipFile(out["results_zip"]) as z:
        assert any(n.endswith("SUMMARY.md") for n in z.namelist())
    # resume: a second call reuses everything
    t1 = time.time()
    week1_main(cfg)
    print(f"resume run: {time.time() - t1:.0f}s")

    # Kaggle-style resume: a fresh session whose only input is a crashed run that had finished S1, S2 and one model,
    # re-uploaded as a Dataset that stayed a .zip and has no week1/ folder inside (the hardest layout)
    inp = out_root / "kaggle_input" / "old-week1-output"
    inp.mkdir(parents=True, exist_ok=True)
    keep = [root / "data" / "sources.json", root / "data" / "unit_vocab.json", root / "results" / "log.txt",
            *[f for f in (root / "data" / "smol135").rglob("*") if f.is_file()]]
    with zipfile.ZipFile(inp / "week1_output.zip", "w") as z:
        for f in keep:
            z.write(f, arcname=str(f.relative_to(root)))
    fresh = out_root / "fresh_session"
    t2 = time.time()
    week1_main(dict(cfg, OUT_ROOT=str(fresh), RESUME_FROM="auto", PHASE1_SEARCH=[str(out_root / "kaggle_input")]))
    new = (fresh / "week1" / "results" / "log.txt").read_text(encoding="utf-8").split("resume: copied")[-1]
    assert "S1 sources: reused" in new and "S2 vocab: reused" in new, "sources / vocab must be reused"
    assert "[smol135] split" not in new, "a finished model must not be regenerated"
    assert "[qwen0.5] split dev" in new, "unfinished models must be collected"
    assert (fresh / "week1_results.zip").exists()
    print(f"auto-resume run: {time.time() - t2:.0f}s")

    # a crash is logged with its traceback and the partial results are still zipped
    crash = out_root / "crash_session"
    try:
        week1_main(dict(cfg, OUT_ROOT=str(crash), MODELS=[dict(key="nope", names=["no-such-org/no-such-model"])]))
        raise AssertionError("expected a failure")
    except RuntimeError as e:
        assert "only 0 models collected" in str(e)
    clog = (crash / "week1" / "results" / "log.txt").read_text(encoding="utf-8")
    assert "FAILED: RuntimeError" in clog and "Traceback" in clog and (crash / "week1_results.zip").exists()
    print("week1 smoke test passed")


def part_dt() -> None:
    """Notebook 06: draft-model teacher-forced pass over a week-1 run's target outputs, timing, zip.
    The 'target' outputs are generated here by Qwen2.5-0.5B and the draft is the same model, so the draft
    argmax must equal the target token almost everywhere: an exact check of the row alignment."""
    import json
    from lgmtp import collect as C
    from lgmtp.drafter_tf import drafter_main
    out_root = SMOKE_ROOT / "dt"
    if out_root.exists():
        shutil.rmtree(out_root)
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    spec = dict(key="q", names=[name], dtype="fp32")
    m, tok, _ = C.load_model(spec, "cpu")
    rows = [(i, s) for i, s in enumerate(["The weather is nice today.", "India has many languages.",
                                          "He went to the market to buy vegetables.", "Water boils at 100 degrees."])]
    run = out_root / "kaggle_input" / "mtp-final-boss" / "week1"
    d = run / "data" / "llama3-8b"
    d.mkdir(parents=True)
    (run / "data" / "sources.json").write_text("{}")
    eos = C.model_eos_ids(m, tok, spec)
    for split, rr in (("dev", rows[:2]), ("devtest", rows[2:])):
        C.generate(m, tok, spec, rr, d / f"{split}.jsonl", 2, 16, "cpu", set(eos), print)
    (d / "timing.json").write_text(json.dumps(dict(t_target=0.07)))
    del m
    cfg = dict(DRAFT=[name], TARGET_TOKENIZER=[name], SEARCH=[str(out_root / "kaggle_input")],
               OUT_ROOT=str(out_root / "working"), TIME_Q=[1, 4], TIME_CTX=[32], TIME_STEPS=2, TIME_REPS=1, SMOKE=True)
    t0 = time.time()
    s = drafter_main(cfg)
    print(json.dumps(s["splits"], indent=1))
    for split, v in s["splits"].items():
        assert v["positions"] > 0 and v["draft_target_agreement"] > 0.97, f"{split}: rows misaligned? {v}"
    with np.load(out_root / "working" / "drafter_traces" / "devtest.npz") as z:
        assert z["d_topk_ids"].shape == (len(z["target_ids"]), 20) and len(z["d_H"]) == len(z["target_ids"])
        assert z["offsets"][-1] == len(z["target_ids"])
    c = s["timing"]["c"]["32"]
    assert set(c) == {"1", "4"} and all(v > 0 for v in c.values())
    with zipfile.ZipFile(out_root / "working" / "drafter_traces.zip") as z:
        names = set(z.namelist())
    assert {"dev.npz", "devtest.npz", "devtest.jsonl", "timing.json", "summary.json"} <= names, names
    t1 = time.time()
    drafter_main(cfg)                                   # resume: everything reused
    log = (out_root / "working" / "drafter_traces" / "log.txt").read_text(encoding="utf-8")
    assert log.count("dev: reused") == 1 and "timing: reused" in log
    print(f"drafter smoke test passed ({t1 - t0:.0f}s, resume {time.time() - t1:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["01", "03", "w1", "dt"], required=True)
    ap.add_argument("--selected", default=None, help="selected_params.json (part 03; default: smoke params)")
    ap.add_argument("--noise", type=float, default=0.0, help="part 03: draft logit noise (exercise rejections)")
    a = ap.parse_args()
    if a.part == "01":
        part01()
    elif a.part == "w1":
        part_w1()
    elif a.part == "dt":
        part_dt()
    else:
        part03(a.selected, a.noise)


if __name__ == "__main__":
    main()
