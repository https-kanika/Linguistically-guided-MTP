"""Analyses and figures for checked drafting (paper section on the n-gram).

    python scripts/checked_analysis.py

Needs data/traces_v2, data/wordheads_long, data/ngram_hi.npz (scripts/run_checked_sim.py), ../hindi.csv,
results/checked/. Writes results/checked/{gain_by_share.csv, where_saved.json} and
paper/figures/{fig_checked_share.png, fig_checked_data_cost.png}.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lgmtp import refund_sim as R  # noqa: E402
from lgmtp.checked import NGram, simulate_checked  # noqa: E402
from lgmtp.traces import attach_word_labels, load_traces  # noqa: E402

COL = {"Entropy": "#7a7a7a", "V1": "#0072B2", "NGram": "#D55E00"}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
    D, H = ROOT / "data" / "traces_v2", ROOT / "data" / "wordheads_long"
    out, fig_dir = ROOT / "results" / "checked", ROOT / "paper" / "figures"
    tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
    cost = R.load_cost(D / "timing.json", H / "run_info.json")
    tr = load_traces(D / "traces_devtest.npz")[0]
    attach_word_labels(tr, tok)
    R.attach_heads(tr, H / "heads_devtest.npz")
    gate = R.devanagari_mask(tok, len(tr[0]._in_cv))
    lex = R.Lexicon.build(tok, D / "target_outputs_dev.jsonl", ROOT.parent / "hindi.csv")
    ng = NGram.load(ROOT / "data" / "ngram_hi.npz")
    P = json.loads((out / "params.json").read_text())["180k"]
    best = json.loads(next(r for r in csv.DictReader(open(ROOT / "results" / "wordheads_long" / "main_table.csv",
                                                          encoding="utf-8"))
                           if r["method"] == "Heads V1 (cont. vocab + lexicon filter)")["params"])

    e = R.simulate(tr, "entropy", cost, lam=-0.683)
    h = R.simulate(tr, "heads", cost, variant="v1", src="lexfilter", gate=gate, lex=lex, **best)
    tally = Counter()
    n = simulate_checked(tr, cost, P["NGram"]["lam"], ngram=ng, L=P["NGram"]["L"], tally=tally)
    print("re-simulated: Entropy %.4f, V1+lex %.4f, NGram %.4f" % (e[0], h[0], n[0]))

    # 1. where the saved 1B calls are: word continuations vs word starts
    wi = np.concatenate([t.labels["word_initial"] for t in tr])
    match = np.concatenate([t.match for t in tr])
    agreed = tally["agreed_continuation"] + tally["agreed_word_start"]
    where = dict(continuation_share_of_tokens=round(float(1 - wi.mean()), 4),
                 continuation_share_of_correct_1B_tokens=round(float((~wi & match).sum() / match.sum()), 4),
                 agreed_proposal_tokens=agreed, agreed_continuation=tally["agreed_continuation"],
                 agreed_word_start=tally["agreed_word_start"],
                 continuation_share_of_saved_calls=round(tally["agreed_continuation"] / agreed, 4))
    (out / "where_saved.json").write_text(json.dumps(where, indent=1), encoding="utf-8")
    print(json.dumps(where))

    # 2. gain over Entropy by the sentence's share of continuation tokens
    share = np.array([1 - t.labels["word_initial"].mean() for t in tr])
    base = e[1] / e[2]
    df = pd.DataFrame({"cont_share": share, "NGram": n[1] / n[2] / base - 1, "V1+lex": h[1] / h[2] / base - 1})
    df["bin"] = pd.qcut(share, 5)
    g = df.groupby("bin", observed=True).agg(sentences=("NGram", "size"), mean_cont_share=("cont_share", "mean"),
                                             gain_ngram=("NGram", "mean"), gain_v1_lex=("V1+lex", "mean"))
    rho = {m: round(df["cont_share"].corr(df[m], method="spearman"), 3) for m in ("NGram", "V1+lex")}
    g.to_csv(out / "gain_by_share.csv")
    (out / "gain_by_share_spearman.json").write_text(json.dumps(rho), encoding="utf-8")
    print(g.to_string(), "\nSpearman:", rho)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for m, col, key, lab in (("NGram", "NGram", "gain_ngram", "4-gram, checked"),
                             ("V1+lex", "V1", "gain_v1_lex", "V1 heads + lexicon (unchecked)")):
        ax.plot(100 * g.mean_cont_share, 100 * g[key], marker="o", ms=5, lw=2, color=COL[col],
                label=f"{lab} (Spearman {rho[m]:.2f})")
    ax.set_xlabel("word-continuation tokens in the sentence (%, quintile means)")
    ax.set_ylabel("gain over Entropy (%)")
    ax.set_ylim(0, 27)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_checked_share.png", bbox_inches="tight", dpi=200)

    # 3. data size and cost sensitivity (rows written by run_checked_sim.py)
    T = pd.read_csv(out / "main_table.csv")
    T = T[T.heads == "180k"].set_index("method")
    sizes = [(1, "1k"), (10, "10k"), (60, "60k"), (190, None)]
    ys = [T.loc["n-gram, checked" if lab is None else f"n-gram, checked (4-gram, {lab} sentences)"] for _, lab in sizes]
    ent, v1 = T.loc["Entropy", "speedup"], T.loc["Heads V1 + lexicon (unchecked)", "speedup"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
    xs = [x for x, _ in sizes]
    axes[0].errorbar(xs, [y.speedup for y in ys],
                     yerr=[[y.speedup - y.lo for y in ys], [y.hi - y.speedup for y in ys]],
                     marker="o", ms=5, lw=2, color=COL["NGram"], capsize=0)
    axes[0].set_xscale("log")
    axes[0].set_xticks(xs, ["1k", "10k", "60k", "190k"])
    axes[0].minorticks_off()
    axes[0].set_xlabel("n-gram training sentences (IITB Hindi)")
    axes[0].set_ylabel("simulated speed-up (devtest)")
    dear = [0, 5, 10, 20]
    sp = [T.loc["n-gram, checked", "speedup"]] + [
        T.loc[f"n-gram, checked, calls with a proposal {x}% dearer", "speedup"] for x in dear[1:]]
    axes[1].plot(dear, sp, marker="o", ms=5, lw=2, color=COL["NGram"])
    axes[1].set_xticks(dear, [f"+{x}%" for x in dear])
    axes[1].set_xlabel("extra cost of a 1B call that reads a proposal")
    for a in axes:
        a.axhline(v1, color=COL["V1"], ls="--", lw=1)
        a.axhline(ent, color=COL["Entropy"], ls="--", lw=1)
        a.set_ylim(1.40, 1.75)
    axes[1].text(20, v1 + 0.007, "V1 heads + lexicon", color=COL["V1"], fontsize=8, ha="right")
    axes[1].text(20, ent + 0.007, "Entropy", color=COL["Entropy"], fontsize=8, ha="right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_checked_data_cost.png", bbox_inches="tight", dpi=200)
    print("wrote figures")

    # 4. real GPU loop vs simulation on the same sentences, with the costs measured in that run
    loop = ROOT / "results" / "heads_loop"
    ids = set(json.loads((ROOT / "results" / "phase1" / "real_loop_ids.json").read_text())["ids"])
    sub = [t for t in tr if t.id in ids]
    real = {r["method"]: r for r in json.loads((loop / "summary.json").read_text(encoding="utf-8"))}
    methods = json.loads((loop / "run_info.json").read_text(encoding="utf-8"))["methods"]
    rows = []
    for tname, tpath in (("phase1", D / "timing.json"), ("this_run", loop / "timing.json")):
        c = R.load_cost(tpath, H / "run_info.json")
        for m, p in methods.items():
            if m == "NoDraft":
                continue
            if m == "Entropy":
                s = R.simulate(sub, "entropy", c, lam=p["lam"])
            elif m == "V1+lex":
                s = R.simulate(sub, "heads", c, variant="v1", src="lexfilter", gate=gate, lex=lex, lam=p["lam"],
                               tau_h=p["tau_h"], tau_c=p["tau_c"])
            else:
                s = simulate_checked(sub, c, p["lam"], variant=p["variant"], tau_h=p["tau_h"],
                                     ngram=ng if p["ngram"] else None, L=p["L"], gate=gate)
            r = real[m]
            rows.append(dict(timing=tname, c=round(c.c, 4), method=m, sim_same_sentences=round(s[0], 4),
                             real_decode_only=round(r["speedup_decode_only"], 4), real_wall=round(r["speedup_wall"], 4),
                             gap_decode_only=round(r["speedup_decode_only"] / s[0] - 1, 4),
                             sim_tokens_per_round=round(s[1].sum() / s[3]["rounds"], 3),
                             real_tokens_per_round=round(r["tokens_per_round"], 3),
                             sim_calls_per_round=round(s[3]["calls"] / s[3]["rounds"], 3) if "calls" in s[3] else None,
                             real_calls_per_round=round(r["draft_calls_per_round"], 3)))
    pd.DataFrame(rows).to_csv(loop / "sim_vs_real.csv", index=False)
    print(pd.DataFrame(rows).to_string())

    # 5. draft time per 1B call in the real loop, and V1 + lexicon vs Entropy (paired, simulated)
    per_call = {}
    for line in open(loop / "loop_results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        a = per_call.setdefault(r["method"], [0.0, 0])
        a[0] += sum(x["t_draft"] for x in r["rounds"])
        a[1] += sum(x["calls"] for x in r["rounds"])
    extra = dict(draft_ms_per_call={m: round(1000 * t / c, 2) for m, (t, c) in per_call.items() if c},
                 v1_lex_minus_entropy=[round(x, 4) for x in R.paired_ci((h[1], h[2]), (e[1], e[2]))])
    (out / "extra.json").write_text(json.dumps(extra, indent=1), encoding="utf-8")
    print(json.dumps(extra))


if __name__ == "__main__":
    main()
