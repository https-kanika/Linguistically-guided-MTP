"""Simulate checked drafting (src/lgmtp/checked.py): n-gram, heads, and both, each checked by the 1B.

    python scripts/run_checked_sim.py --sources ../week1/data/sources.json

Needs data/traces_v2, data/wordheads, data/wordheads_long and ../hindi.csv (baseline row). --sources is week 1's
sources.json ("mono" = IITB Hindi sentences); without it the same sentences are downloaded (collect.load_iitb).
Tuned on dev, reported on devtest. Writes data/ngram_hi.npz (table for notebook 09) and results/checked/.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lgmtp import refund_sim as R  # noqa: E402
from lgmtp.checked import NGram, simulate_checked  # noqa: E402
from lgmtp.traces import attach_word_labels, load_traces  # noqa: E402

LAMS = (-0.9, -0.8, -0.683, -0.6, -0.5, -0.4)
N_MONO = 190000
HEADS = {"60k": "wordheads", "180k": "wordheads_long"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=None)
    ap.add_argument("--hindi-csv", default=str(ROOT.parent / "hindi.csv"))
    ap.add_argument("--tokenizer", default="NousResearch/Meta-Llama-3-8B-Instruct")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    D, out, t0 = ROOT / "data" / "traces_v2", ROOT / "results" / "checked", time.time()
    out.mkdir(parents=True, exist_ok=True)

    if a.sources:
        S = json.load(open(a.sources, encoding="utf-8"))["mono"]
        mono = [x[1] if isinstance(x, list) else x for x in S][:N_MONO]
    else:
        from lgmtp.collect import load_iitb
        mono = load_iitb(3000, 200000, seed=0)[1][:N_MONO]
    enc = tok([" " + s for s in mono], add_special_tokens=False)["input_ids"]
    table = ROOT / "data" / "ngram_hi.npz"
    NGram.build(enc, 4).save(table)
    grams = {"4-gram, 190k sentences": NGram.load(table)}
    for n in (1000, 10000, 60000):
        grams[f"4-gram, {n // 1000}k sentences"] = NGram.build(enc[:n], 4)
    dev_ids = [json.loads(x)["out_ids"] for x in open(D / "target_outputs_dev.jsonl", encoding="utf-8")]
    grams["4-gram, 997 dev outputs of the 8B"] = NGram.build(dev_ids, 4)     # tuned on the same outputs
    ng = grams["4-gram, 190k sentences"]
    info = dict(sentences=len(mono), tokens=sum(map(len, enc)), order=ng.order, vocab=ng.V - 1,
                contexts={k: len(v[0]) for k, v in ng.tab.items()}, table_mb=round(table.stat().st_size / 1e6, 1),
                table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(), corpus="cfilt/iitb-english-hindi (week-1 mono)")
    print(json.dumps(info), flush=True)

    params, rows = {}, []
    lex = R.Lexicon.build(tok, D / "target_outputs_dev.jsonl", a.hindi_csv)
    for tag, sub in HEADS.items():
        H = ROOT / "data" / sub
        cost = R.load_cost(D / "timing.json", H / "run_info.json")
        tr = {}
        for s in ("dev", "devtest"):
            tr[s] = load_traces(D / f"traces_{s}.npz")[0]
            attach_word_labels(tr[s], tok)
            R.attach_heads(tr[s], H / f"heads_{s}.npz")
        gate = R.devanagari_mask(tok, len(tr["dev"][0]._in_cv))
        base = R.simulate(tr["devtest"], "entropy", cost, lam=-0.683)
        best = json.loads(next(r for r in csv.DictReader(open(ROOT / "results" / sub / "main_table.csv", encoding="utf-8"))
                               if r["method"] == "Heads V1 (cont. vocab + lexicon filter)")["params"])
        old = R.simulate(tr["devtest"], "heads", cost, variant="v1", src="lexfilter", gate=gate, lex=lex, **best)
        ref = {}
        params[tag] = {}

        def run(name, key, grid, **fixed):
            sc = [(simulate_checked(tr["dev"], cost, l, **fixed, **g)[0], l, g) for g in grid for l in LAMS]
            d, lam, g = max(sc, key=lambda z: z[0])
            r = simulate_checked(tr["devtest"], cost, lam, **fixed, **g)
            lo, hi = R.boot_ci(r[1], r[2])
            row = dict(heads=tag, method=name, speedup=round(r[0], 4), lo=round(lo, 4), hi=round(hi, 4),
                       dev=round(d, 4), params=json.dumps(dict(lam=lam, **g)),
                       tokens_per_round=round(r[1].sum() / r[3]["rounds"], 3), calls_per_round=round(r[3]["calls"] / r[3]["rounds"], 3))
            for nm, x in (("entropy", base), ("v1_lex", old), ("ngram", ref.get("NGram"))):
                if x is not None:
                    row[f"d_{nm}_lo"], row[f"d_{nm}_hi"] = (round(v, 4) for v in R.paired_ci((r[1], r[2]), (x[1], x[2])))
            rows.append(row)
            print(f"[{time.time() - t0:4.0f}s] {tag:4s} {name:42s} {r[0]:.4f} [{lo:.4f}, {hi:.4f}]  {row['params']}  "
                  f"calls/round {row['calls_per_round']}  vs n-gram [{row.get('d_ngram_lo', '')}, {row.get('d_ngram_hi', '')}]", flush=True)
            if key:
                ref[key] = r
                params[tag][key] = dict(lam=lam, L=g.get("L", 6), tau_h=g.get("tau_h", 0.0), sim=round(r[0], 4))

        for nm, r in (("Entropy", base), ("Heads V1 + lexicon (unchecked)", old)):
            lo, hi = R.boot_ci(r[1], r[2])
            rows.append(dict(heads=tag, method=nm, speedup=round(r[0], 4), lo=round(lo, 4), hi=round(hi, 4)))
            print(f"[{time.time() - t0:4.0f}s] {tag:4s} {nm:42s} {r[0]:.4f}", flush=True)
        taus = [dict(tau_h=x) for x in (0.0, 0.1, 0.3, 0.5)]
        Ls = [dict(L=x) for x in (4, 6, 7)]
        both = [dict(tau_h=x, L=6) for x in (0.3, 0.5, 0.7)]
        run("n-gram, checked", "NGram", Ls, ngram=ng)
        run("heads V0, checked", "V0+check", taus, variant="v0", gate=gate)
        run("heads V1, checked", "V1+check", taus, variant="v1", gate=gate)
        run("heads V0 + n-gram, checked", "NGram+V0", both, variant="v0", gate=gate, ngram=ng)
        run("heads V1 + n-gram, checked", "NGram+V1", both, variant="v1", gate=gate, ngram=ng)
        if tag == "180k":
            for nm, g2 in list(grams.items())[1:]:
                run(f"n-gram, checked ({nm})", None, Ls, ngram=g2)
            for x in (0.05, 0.10, 0.20):                     # every 1B call that reads a proposal costs x more
                c2 = R.Cost(cost.c, cost.verify, {1: 1.0, 2: 1.0 + x}, cost.head_rel_cv, cost.head_rel_full)
                r = simulate_checked(tr["devtest"], c2, params[tag]["NGram"]["lam"], ngram=ng, L=params[tag]["NGram"]["L"])
                rows.append(dict(heads=tag, method=f"n-gram, checked, calls with a proposal {x:.0%} dearer", speedup=round(r[0], 4)))
                print(f"     n-gram, calls with a proposal {x:.0%} dearer: {r[0]:.4f}", flush=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "main_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    (out / "params.json").write_text(json.dumps(params, indent=1), encoding="utf-8")
    (out / "ngram_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("wrote", out, table)


if __name__ == "__main__":
    main()
