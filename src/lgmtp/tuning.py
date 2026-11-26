"""Grid search on FLORES+ dev, evaluation on devtest (fixes B7, B8).

Every policy family gets a dev grid; selection = best dev speed-up with lookahead charged and the
measured verify cost. Grids:
  - "around the entropy best": lam = Entropy's dev-best lam +- 5 steps of Entropy's lam grid,
    beta in all of Entropy's betas.
  - MorphEntropy: tau_morph in {0.1..0.6}, use_akshara in {F, T}, tau_m in {0.3, 0.5} (only with
    use_akshara), max_ext in {0..3}. MorphOnly: tau_morph in linspace(0.05, 0.95, 19), min_g in {1,2,3}.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .boundaries import END, MIDAKS, MIDUTF8, WORD
from .policies import (AksharaSafeEntropy, Entropy, FixedK, GoldWordOracle, Hybrid, MorphEntropy, MorphOnly,
                       NoDraft, Oracle, TokenTable, WordBoundaryHard, WordBoundarySoft)
from .simulator import GAMMA_MAX, simulate, summarize, unit_verify
from .stats import bootstrap_ci, paired_bootstrap

ENTROPY_LAMS = [float(x) for x in np.round(np.linspace(-1.5, 0.95, 40), 3)]
ENTROPY_BETAS = [0.5, 1.0, 2.0, 4.0]
LAM_NBHD = 5

FAMILIES = ["NoDraft", "FixedK", "Entropy", "WordBoundarySoft", "WordBoundaryHard", "AksharaSafeEntropy",
            "Hybrid", "MorphEntropy", "MorphOnly", "Oracle", "GoldWordOracle",
            "Hybrid_ext1", "MorphEntropy_ext1"]
# Constrained variants "<Family>_ext1" (user decision 2026-11-02): same class and grid, restricted to max_ext >= 1 so the
# linguistic condition is actually consulted (max_ext = 0 means "stop whenever entropy fires").
CONSTRAINED = {"Hybrid_ext1": "Hybrid", "MorphEntropy_ext1": "MorphEntropy"}


def base_family(family: str) -> str:
    return CONSTRAINED.get(family, family)


def reduction_note(family: str, params: dict) -> str:
    """Human-readable note when a selected config is equivalent to a simpler policy."""
    fam = base_family(family)
    if fam == "MorphEntropy" and params.get("max_ext") == 0 and not params.get("use_akshara"):
        return "reduces to Entropy (max_ext=0: morph model never consulted)"
    if fam == "MorphEntropy" and params.get("max_ext") == 0:
        return "max_ext=0: stops whenever entropy fires; only adds a charged lookahead"
    if fam == "Hybrid" and params.get("max_ext") == 0:
        return "max_ext=0: Entropy + charged lookahead (word condition never changes the decision)"
    return ""
ORACLES = {"Oracle", "GoldWordOracle"}


@dataclass
class Context:
    """Everything policies need besides scalar params."""
    newword_mask: np.ndarray | None = None      # VocabMasks.starts_newword
    prefix_model: object = None                 # morph.PrefixBoundaryModel
    table: TokenTable | None = None
    entropy_best: dict = field(default_factory=dict)   # filled after tuning Entropy


def lam_neighbourhood(best_lam: float, k: int = LAM_NBHD) -> list[float]:
    i = int(np.argmin(np.abs(np.array(ENTROPY_LAMS) - best_lam)))
    return ENTROPY_LAMS[max(0, i - k): i + k + 1]


def grid(family: str, ctx: Context) -> list[dict]:
    if family in CONSTRAINED:
        return [p for p in grid(CONSTRAINED[family], ctx) if p["max_ext"] >= 1]
    lb = None
    if family in ("AksharaSafeEntropy", "Hybrid", "MorphEntropy"):
        if not ctx.entropy_best:
            raise RuntimeError("tune Entropy before " + family)
        lb = list(itertools.product(lam_neighbourhood(ctx.entropy_best["lam"]), ENTROPY_BETAS))
    if family == "NoDraft" or family == "Oracle":
        return [{}]
    if family == "FixedK":
        return [dict(k=k) for k in range(1, 9)]
    if family == "Entropy":
        return [dict(lam=l, beta=b) for b in ENTROPY_BETAS for l in ENTROPY_LAMS]
    if family == "WordBoundarySoft":
        return [dict(tau=float(t), min_g=m) for t in np.round(np.linspace(0.05, 0.95, 19), 3) for m in (1, 2, 3)]
    if family == "WordBoundaryHard":
        return [dict(min_g=m) for m in (1, 2, 3)]
    if family == "AksharaSafeEntropy":
        return [dict(lam=l, beta=b, tau_m=t, max_ext=e) for (l, b) in lb for t in (0.1, 0.3, 0.5, 0.7) for e in (1, 2, 3)]
    if family == "Hybrid":
        return [dict(lam=l, beta=b, tau_w=t, max_ext=e) for (l, b) in lb for t in (0.2, 0.4, 0.6, 0.8) for e in (0, 1, 2, 3)]
    if family == "MorphEntropy":
        out = []
        for (l, b) in lb:
            for tm in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                for e in (0, 1, 2, 3):
                    out.append(dict(lam=l, beta=b, tau_morph=tm, use_akshara=False, tau_m=0.5, max_ext=e))
                    for tau_m in (0.3, 0.5):
                        out.append(dict(lam=l, beta=b, tau_morph=tm, use_akshara=True, tau_m=tau_m, max_ext=e))
        return out
    if family == "MorphOnly":
        return [dict(tau_morph=float(t), min_g=m) for t in np.round(np.linspace(0.05, 0.95, 19), 3) for m in (1, 2, 3)]
    if family == "GoldWordOracle":
        return [dict(min_g=m) for m in (1, 2, 3)]
    raise KeyError(family)


def make_policy(family: str, params: dict, ctx: Context):
    family = base_family(family)
    p = dict(params)
    if family == "NoDraft":
        return NoDraft()
    if family == "FixedK":
        return FixedK(**p)
    if family == "Entropy":
        return Entropy(**p)
    if family == "WordBoundarySoft":
        return WordBoundarySoft(**p)
    if family == "WordBoundaryHard":
        return WordBoundaryHard(p["min_g"], ctx.newword_mask)
    if family == "AksharaSafeEntropy":
        return AksharaSafeEntropy(**p)
    if family == "Hybrid":
        return Hybrid(**p)
    if family == "MorphEntropy":
        return MorphEntropy(**p, prefix_model=ctx.prefix_model, table=ctx.table)
    if family == "MorphOnly":
        return MorphOnly(**p, prefix_model=ctx.prefix_model, table=ctx.table)
    if family == "Oracle":
        return Oracle()
    if family == "GoldWordOracle":
        return GoldWordOracle(**p)
    raise KeyError(family)


def cap_for(family: str, params: dict) -> int:
    """Fixed-k drafts exactly k (legacy: gamma_max = k); every dynamic policy uses GAMMA_MAX = 8."""
    return int(params["k"]) if family == "FixedK" else GAMMA_MAX


_W: dict = {}


def _worker_init(traces, c, verify, ctx, family):
    _W.update(traces=traces, c=c, verify=verify, ctx=ctx, family=family)


def _grid_rows(params_list):
    fam, traces, c, verify, ctx = _W["family"], _W["traces"], _W["c"], _W["verify"], _W["ctx"]
    rows = []
    for params in params_list:
        pol = make_policy(fam, params, ctx)
        r = simulate(traces, pol, cap_for(fam, params), c, verify, charge_lookahead=True, record_rounds=False)
        rows.append(dict(family=fam, **params, lookahead=pol.lookahead, **summarize(r)))
    return rows


def run_grid(family: str, traces, c: float, verify: Callable, ctx: Context, progress: bool = False,
             n_jobs: int | None = None, parallel_min: int = 100):
    """Simulate every config of ``family`` on ``traces``. Grids with >= parallel_min configs run in
    ``n_jobs`` worker processes (default: all cores but one); results are identical to serial."""
    import os
    import pandas as pd
    t0 = time.time()
    configs = grid(family, ctx)
    n_jobs = n_jobs if n_jobs is not None else max(1, (os.cpu_count() or 2) - 1)
    if n_jobs > 1 and len(configs) >= parallel_min:
        from concurrent.futures import ProcessPoolExecutor
        chunks = [configs[i::n_jobs * 4] for i in range(n_jobs * 4)]
        with ProcessPoolExecutor(n_jobs, initializer=_worker_init, initargs=(traces, c, verify, ctx, family)) as ex:
            parts = list(ex.map(_grid_rows, chunks))
        order = {json.dumps(p, sort_keys=True): i for i, p in enumerate(configs)}
        rows = sorted((r for part in parts for r in part),
                      key=lambda r: order[json.dumps({k: r[k] for k in configs[0]}, sort_keys=True)])
    else:
        _worker_init(traces, c, verify, ctx, family)
        rows = _grid_rows(configs)
    df = pd.DataFrame(rows)
    if progress:
        print(f"{family:20s} {len(df):5d} configs  {time.time() - t0:6.1f}s  best dev speed-up {df.speedup.max():.4f}")
    return df


def select(df) -> dict:
    """Params of the best dev speed-up (ties -> first in grid order)."""
    best = df.loc[df.speedup.idxmax()]
    drop = {"family", "lookahead", "acceptance_rate", "alpha_hat", "tokens_per_verif", "mean_draft_len", "speedup"}
    out = {}
    for k, v in best.items():
        if k in drop or (isinstance(v, float) and np.isnan(v)):
            continue
        out[k] = v.item() if hasattr(v, "item") else v
    if "k" in out or "min_g" in out or "max_ext" in out:
        for key in ("k", "min_g", "max_ext"):
            if key in out:
                out[key] = int(out[key])
    if "use_akshara" in out:
        out["use_akshara"] = bool(out["use_akshara"])
    return out


def tune_all(dev, c: float, verify: Callable, ctx: Context, out_dir=None, families=FAMILIES,
             progress: bool = True, n_jobs: int | None = None) -> dict:
    """Grid search every family on dev; saves grid_{family}.csv; returns {family: params}."""
    out_dir = Path(out_dir) if out_dir else None
    order = ["FixedK", "Entropy"] + [f for f in families if f not in ("FixedK", "Entropy")]
    selected = {}
    for fam in order:
        if fam not in families and fam not in ("FixedK", "Entropy"):
            continue
        df = run_grid(fam, dev, c, verify, ctx, progress, n_jobs=n_jobs)
        if out_dir:
            df.to_csv(out_dir / f"grid_{fam}.csv", index=False)
        selected[fam] = select(df)
        if fam == "Entropy":
            ctx.entropy_best = dict(selected[fam])
    return selected


# --------------------------------------------------------------------------- #
# evaluation on devtest
# --------------------------------------------------------------------------- #

def _ending_fracs(records, traces_by_id) -> dict:
    """Fractions of non-empty drafts whose last drafted index p+g-1 ends at a gold word boundary, inside
    an akshara (incl. mid-UTF-8), or at a gold morpheme boundary (labels of the target sequence)."""
    n = w = a = m = 0
    has_m = True
    for sid, p, g, acc, _ in records:
        if g == 0:
            continue
        L = traces_by_id[sid].labels
        cut = L["end_cut"][p + g - 1]
        n += 1
        w += cut in (WORD, END)
        a += cut in (MIDAKS, MIDUTF8)
        if "morph_boundary_after" in L:
            m += bool(L["morph_boundary_after"][p + g - 1])
        else:
            has_m = False
    return dict(frac_drafts_ending_at_gold_word_boundary=w / n if n else float("nan"),
                frac_ending_mid_akshara=a / n if n else float("nan"),
                frac_ending_at_morph_boundary=(m / n if n else float("nan")) if has_m else float("nan"))


def evaluate(test, selected: dict, c: float, verify: Callable, ctx: Context, n_boot: int = 1000, seed: int = 0):
    """Main table on devtest. Returns (DataFrame, per-policy SimResults charged)."""
    import pandas as pd
    by_id = {t.id: t for t in test}
    sims, rows = {}, []
    for fam, params in selected.items():
        pol = make_policy(fam, params, ctx)
        cap = cap_for(fam, params)
        ch = simulate(test, pol, cap, c, verify, charge_lookahead=True)
        un = simulate(test, pol, cap, c, verify, charge_lookahead=False, record_rounds=False)
        uv = simulate(test, pol, cap, c, unit_verify, charge_lookahead=True, record_rounds=False)
        sims[fam] = ch
        m = summarize(ch)
        ci = bootstrap_ci(ch.sent_emitted, ch.sent_cost, n_boot, seed)
        has_labels = all("end_cut" in t.labels for t in test)
        fr = _ending_fracs(ch.records, by_id) if has_labels else {}
        rows.append(dict(policy=fam, note=reduction_note(fam, params),
                         params=json.dumps(params, ensure_ascii=False), lookahead=pol.lookahead,
                         speedup_charged=m["speedup"], CI_low=ci["lo"], CI_high=ci["hi"],
                         speedup_uncharged=summarize(un)["speedup"], speedup_unit_verify=summarize(uv)["speedup"],
                         acceptance_rate=m["acceptance_rate"], alpha_hat=m["alpha_hat"],
                         tokens_per_verif=m["tokens_per_verif"], mean_draft_len=m["mean_draft_len"], **fr))
    df = pd.DataFrame(rows)
    if "Entropy" in sims:
        e = sims["Entropy"]
        d = [paired_bootstrap(sims[f].sent_emitted, sims[f].sent_cost, e.sent_emitted, e.sent_cost, n_boot, seed)
             for f in df.policy]
        df["diff_vs_entropy"] = [x["diff"] for x in d]
        df["diff_CI_low"] = [x["lo"] for x in d]
        df["diff_CI_high"] = [x["hi"] for x in d]
        df["p_better"] = [x["p_better"] for x in d]
    if "FixedK" in sims and "Oracle" in sims:
        fb = df.set_index("policy").loc["FixedK", "speedup_charged"]
        orc = df.set_index("policy").loc["Oracle", "speedup_charged"]
        df["frac_oracle_gap_closed"] = (df.speedup_charged - fb) / (orc - fb)
    return df, sims


def write_selected(selected: dict, path, extra: dict | None = None) -> None:
    out = dict(selected=selected, **(extra or {}))
    Path(path).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
