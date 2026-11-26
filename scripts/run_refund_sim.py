"""Main table for the word-completion heads: every drafter tuned on dev,
reported on devtest with bootstrap CIs and paired CIs vs Entropy.

    python scripts/run_refund_sim.py [--quick]

Needs data/traces_v2/ (Phase-1 traces), data/wordheads/ (Kaggle notebook 07 output), ../hindi.csv.
Writes results/wordheads/{sim_results.json, main_table.csv, head_accuracy.json, tuning_dev.csv}.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lgmtp import refund_sim as R                      # noqa: E402
from lgmtp.traces import attach_word_labels, load_traces  # noqa: E402

D, H, OUT = ROOT / "data/traces_v2", ROOT / "data/wordheads", ROOT / "results/wordheads"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small grids (minutes)")
    ap.add_argument("--no-dict", action="store_true", help="skip the (slow) dictionary drafters")
    ap.add_argument("--heads-dir", default=str(H), help="Kaggle output with heads_{dev,devtest}.npz")
    ap.add_argument("--out", default=str(OUT), help="output directory")
    a = ap.parse_args()
    H_, OUT_ = Path(a.heads_dir), Path(a.out)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    cost = R.load_cost(D / "timing.json", H_ / "run_info.json")
    tr, align = {}, {}
    for s in ("dev", "devtest"):
        tr[s] = load_traces(D / f"traces_{s}.npz")[0]
        attach_word_labels(tr[s], tok)
        align[s] = R.attach_heads(tr[s], H_ / f"heads_{s}.npz")["agreement"]
    gate = R.devanagari_mask(tok, len(tr["dev"][0]._in_cv))
    lams = (-0.9, -0.683, -0.5) if a.quick else R.LAM_GRID
    OUT_.mkdir(parents=True, exist_ok=True)
    tuning, rows = [], {}
    t0 = time.time()

    def log(*x):
        print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)

    def run(name, mode, grid, **fixed):
        best, scored = R.tune(tr["dev"], grid, mode, cost, **fixed)
        for g, s in scored:
            tuning.append(dict(method=name, **{k: v for k, v in g.items()}, dev=round(s, 4)))
        res = R.simulate(tr["devtest"], mode, cost, **fixed, **best)
        rows[name] = dict(params=best, res=res)
        log(f"{name:40s} dev {max(s for _, s in scored):.3f}  devtest {res[0]:.3f}  {best}")
        return best

    def lam_then(name, mode, grid_inner, **fixed):
        """Coordinate search: inner grid at lam = -0.683, then lam for the best inner setting."""
        inner, scored = R.tune(tr["dev"], [dict(lam=-0.683, **g) for g in grid_inner], mode, cost, **fixed)
        for g, s in scored:
            tuning.append(dict(method=name, **g, dev=round(s, 4)))
        inner = {k: v for k, v in inner.items() if k != "lam"}
        return run(name, mode, [dict(lam=l, **inner) for l in lams], **fixed)

    # baselines and ceilings
    rows["Entropy (Phase 1, lam=-0.683)"] = dict(params=dict(lam=-0.683),
                                                  res=R.simulate(tr["devtest"], "entropy", cost, lam=-0.683))
    run("Entropy (lam tuned)", "entropy", [dict(lam=l) for l in lams])
    for kap in (0.0, 0.05):
        run(f"Ceiling: free continuations (kappa={kap})", "freecont", [dict(lam=l, kappa=kap) for l in lams])
    rho = {}
    for r in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
        best, _ = R.tune(tr["dev"], [dict(lam=l, rho=r, kappa=0.05) for l in (-0.9, -0.683, -0.5)], "noisy", cost)
        rho[r] = R.simulate(tr["devtest"], "noisy", cost, **best)[0]
        log(f"rho={r}: {rho[r]:.3f}")

    # dictionary drafters (training-free)
    if not a.no_dict:
        lex = R.Lexicon.build(tok, D / "target_outputs_dev.jsonl", ROOT.parent / "hindi.csv")
        run("Dictionary (every mid-word position)", "dict", [dict(lam=l) for l in lams], lex=lex, tok=tok)
        taus = (0.4, 0.6, 0.8) if a.quick else (0.3, 0.4, 0.5, 0.6, 0.8, 0.9)
        lam_then("Dictionary, confidence-gated", "dict", [dict(tau=t, min_freq=2) for t in taus], lex=lex, tok=tok)

    # heads
    acc = {f"{v}_{src}": R.head_accuracy(tr["devtest"], v, src) for v in R.VARIANTS for src in ("cv", "full")}
    for k, v in acc.items():
        log("head accuracy", k, v)
    ths = (0.0, 0.3, 0.5) if a.quick else (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    tcs = (None, 0.5) if a.quick else (None, 0.3, 0.4, 0.5, 0.6, 0.7)
    for v in R.VARIANTS:
        for src in ("cv", "full"):
            grid = [dict(tau_h=th, tau_c=tc) for th in ths for tc in ((None,) if v == "v0" else tcs)]
            lam_then(f"Heads {v.upper()} ({src} vocab)", "heads", grid, variant=v, src=src, gate=gate)
    # heads + lexicon (model context + linguistic knowledge)
    if not a.no_dict:
        for v in ("v0", "v1"):
            for src, nm in (("lexfilter", "cont. vocab + lexicon filter"), ("lexrerank", "full top-8 + lexicon re-rank")):
                grid = [dict(tau_h=th, tau_c=tc) for th in ths for tc in ((None,) if v == "v0" else tcs)]
                lam_then(f"Heads {v.upper()} ({nm})", "heads", grid, variant=v, src=src, gate=gate, lex=lex)

    # save
    base = rows["Entropy (Phase 1, lam=-0.683)"]["res"]
    table = []
    for name, r in rows.items():
        s, E, C, st = r["res"]
        table.append(dict(method=name, speedup=round(s, 4), ci=[round(x, 4) for x in R.boot_ci(E, C)],
                          diff_vs_entropy_ci=[round(x, 4) for x in R.paired_ci((E, C), (base[1], base[2]))],
                          tokens_per_round=round(E.sum() / st["rounds"], 3),
                          drafted_per_round=round(st["drafted"] / st["rounds"], 3),
                          params={k: v for k, v in r["params"].items()}))
    comparisons = {}
    for a_, b_ in [("Heads V1 (cv vocab)", "Heads V0 (cv vocab)"), ("Heads V1 (full vocab)", "Heads V0 (full vocab)"),
                   ("Heads V2 (cv vocab)", "Heads V1 (cv vocab)"), ("Heads V1 (cv vocab)", "Heads V0 (full vocab)"),
                   ("Heads V1 (cv vocab)", "Dictionary, confidence-gated"),
                   ("Heads V1 (cont. vocab + lexicon filter)", "Heads V1 (cv vocab)"),
                   ("Heads V1 (full top-8 + lexicon re-rank)", "Heads V1 (full vocab)"),
                   ("Heads V1 (full top-8 + lexicon re-rank)", "Heads V1 (cv vocab)"),
                   ("Heads V1 (cont. vocab + lexicon filter)", "Heads V0 (cont. vocab + lexicon filter)"),
                   ("Heads V0 (cont. vocab + lexicon filter)", "Heads V0 (cv vocab)")]:
        if a_ in rows and b_ in rows:
            ra, rb = rows[a_]["res"], rows[b_]["res"]
            comparisons[f"{a_} - {b_}"] = dict(diff=round(ra[0] - rb[0], 4),
                                               ci=[round(x, 4) for x in R.paired_ci((ra[1], ra[2]), (rb[1], rb[2]))])
            print("compare", a_, "-", b_, comparisons[f"{a_} - {b_}"])
    meta = dict(alignment=align, comparisons=comparisons, cost=dict(c=cost.c, r_q=cost.r_q, head_rel_cv=cost.head_rel_cv,
                                           head_rel_full=cost.head_rel_full), quick=a.quick)
    (OUT_ / "sim_results.json").write_text(json.dumps(dict(meta=meta, table=table, rho_curve=rho), indent=1))
    (OUT_ / "head_accuracy.json").write_text(json.dumps(acc, indent=1))
    with open(OUT_ / "main_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "speedup", "ci_lo", "ci_hi", "diff_lo", "diff_hi", "tokens_per_round", "params"])
        for r in table:
            w.writerow([r["method"], r["speedup"], *r["ci"], *r["diff_vs_entropy_ci"], r["tokens_per_round"],
                        json.dumps(r["params"])])
    keys = sorted({k for r in tuning for k in r})
    with open(OUT_ / "tuning_dev.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(tuning)
    for r in table:
        print(f"{r['method']:44s} {r['speedup']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  "
              f"diff [{r['diff_vs_entropy_ci'][0]:+.3f}, {r['diff_vs_entropy_ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
