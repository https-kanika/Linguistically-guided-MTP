"""CPU smoke test of the Kaggle notebook code paths (PLAN rule 2a).

    python scripts/smoke_cpu.py --part 01      # notebook 01: generation, teacher-forced passes, timing, zip
    python scripts/smoke_cpu.py --part 03      # notebook 03: real speculative loop (added in step S5)

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


def part03(selected_path: str | None = None) -> None:
    import json
    import pickle
    from lgmtp import spec_loop
    from lgmtp.gpu_traces import load_flores_pairs, read_jsonl
    from lgmtp.morph import GoldLexicon, PrefixBoundaryModel, TieredLabeler, build_prefix_model, load_morphtok
    from lgmtp.simulator import simulate
    from lgmtp.tuning import Context, make_policy

    out_root = SMOKE_ROOT / "03"
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
                                         SELECTED=selected, PREFIX_MODEL_STATE=state, IDS=ids, GIT_HASH="smoke"))
    print(f"part 03 run: {time.time() - t0:.0f}s")
    with zipfile.ZipFile(res["zip"]) as z:
        names = sorted(z.namelist())
        z.extractall(out_root / "unzipped")
    print("zip contents:", names)
    assert res["n_lossless"] == res["n_results"] == 3 * 2 * len(spec_loop.LIVE_FAMILIES), res
    results = read_jsonl(out_root / "unzipped" / "loop_results.jsonl")
    # draft == target: every drafted token is accepted
    for r in results:
        assert all(rd["accepted"] == rd["g"] for rd in r["rounds"]), (r["policy"], r["rounds"])

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
        for r in live:
            lg = [rd["g"] for rd in r["rounds"]]
            sg = sim_g[r["id"]]
            # identical except the last round: the simulator caps at n - p, the live loop at max_new - p
            assert lg[:-1] == sg[:-1], (fam, r["id"], lg, sg)
        print(f"{fam:18s} live vs simulated draft lengths agree (sentences: {len(live)})")
    print("\nSMOKE 03 PASSED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["01", "03"], required=True)
    ap.add_argument("--selected", default=None, help="selected_params.json (part 03; default: smoke params)")
    a = ap.parse_args()
    if a.part == "01":
        part01()
    else:
        part03(a.selected)


if __name__ == "__main__":
    main()
