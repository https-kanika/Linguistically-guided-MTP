"""Where do drafts fail? (PLAN T8, fixes B6.)

Per-position table built from traces with gold labels attached (``traces.attach_word_labels`` and
``morph.token_morph_labels``). Reports descriptive tables and *signed* point-biserial correlations
with bootstrap CIs over sentences. No automatically generated interpretation text.
"""
from __future__ import annotations

import json
import unicodedata

import numpy as np

from .boundaries import END, MIDAKS, MIDUTF8, MIDWORD, WORD
from .stats import bootstrap_stat_by_sentence


def token_type(s: str) -> str:
    """Type of a token from its decoded text (leading whitespace ignored)."""
    s = s.lstrip()
    if not s:
        return "space/special"
    ch = s[0]
    if ch.isdigit():
        return "digit"
    if unicodedata.category(ch).startswith("P"):    # before the block check: danda U+0964 is in it
        return "punct"
    if "ऀ" <= ch <= "ॿ":
        return "devanagari"
    if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
        return "latin"
    return "other"


def position_frame(traces, texts: dict | None = None):
    """One row per target position. ``texts[id]`` = decoded target text (for token types); labels
    must be attached to each trace (end_cut, start_cut, pos_in_word, word_initial[, morph_*])."""
    import pandas as pd
    rows = []
    for t in traces:
        L = t.labels
        if "end_cut" not in L:
            raise ValueError(f"trace {t.id} has no gold labels attached")
        n = t.n
        d = dict(sid=np.full(n, t.id), t=np.arange(n), H=np.asarray(t.H, float), match=np.asarray(t.match, bool),
                 p_target=np.asarray(t.p_target, float), top1=np.asarray(t.top1, float),
                 pos_in_word=np.asarray(L["pos_in_word"]),
                 word_initial=np.asarray(L["word_initial"], bool), start_cut=np.asarray(L["start_cut"]),
                 end_cut=np.asarray(L["end_cut"]))
        if "morph_boundary_after" in L:
            d["morph_boundary_after"] = np.asarray(L["morph_boundary_after"], bool)
            d["morph_tier"] = np.asarray(L["morph_tier"])
        if texts is not None and t.id in texts:
            txt = texts[t.id]
            d["token_type"] = np.array([token_type(txt[a:b]) for a, b in L["char_span"]])
        rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True)
    df["pos_bucket"] = np.where(df.pos_in_word >= 4, "4+", df.pos_in_word.astype(str))
    df["start_kind"] = df.start_cut.map({WORD: "word-initial", END: "word-initial", MIDWORD: "mid-word",
                                         MIDAKS: "mid-akshara", MIDUTF8: "mid-utf8"})
    return df


def table_by(df, col: str):
    g = df.groupby(col, sort=True)
    out = g.agg(count=("match", "size"), mean_H=("H", "mean"), match_rate=("match", "mean"),
                mean_p_target=("p_target", "mean"))
    out["frac"] = out["count"] / out["count"].sum()
    return out.reset_index()


def point_biserial(x, y) -> float:
    """Signed point-biserial r between continuous x and binary y (= Pearson r)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if y.std() == 0 or x.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def correlation_with_ci(df, xcol: str, ycol_or_fn, n_boot: int = 1000, seed: int = 0) -> dict:
    """r(x, y) with a bootstrap CI that resamples sentences (positions within a sentence stay together)."""
    y = ycol_or_fn(df) if callable(ycol_or_fn) else df[ycol_or_fn]
    sub = df[[xcol, "sid"]].assign(_y=np.asarray(y, float))
    groups = [(g[xcol].to_numpy(), g["_y"].to_numpy()) for _, g in sub.groupby("sid", sort=False)]

    def fn(gs):
        return point_biserial(np.concatenate([a for a, _ in gs]), np.concatenate([b for _, b in gs]))
    r = bootstrap_stat_by_sentence(groups, fn, n_boot=n_boot, seed=seed)
    return dict(r=r["point"], lo=r["lo"], hi=r["hi"], n_positions=int(len(sub)), n_sentences=len(groups))


def standard_correlations(df, n_boot: int = 1000, seed: int = 0):
    import pandas as pd
    specs = [("H vs word_initial", "word_initial"),
             ("H vs end_cut==WORD", lambda d: (d.end_cut == WORD).astype(float)),
             ("H vs match", "match")]
    if "morph_boundary_after" in df:
        specs.append(("H vs morph_boundary_after", "morph_boundary_after"))
    rows = [dict(pair=name, **correlation_with_ci(df, "H", y, n_boot, seed)) for name, y in specs]
    return pd.DataFrame(rows)


def plot_by_position(df, path=None, title: str = ""):
    import matplotlib.pyplot as plt
    tb = table_by(df, "pos_bucket")
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(tb.pos_bucket, tb.match_rate, "o-", color="tab:blue", label="match rate")
    ax1.set_xlabel("position in word (tokens)"); ax1.set_ylabel("draft/target match rate", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(tb.pos_bucket, tb.mean_H, "s--", color="tab:red", label="mean entropy")
    ax2.set_ylabel("mean draft entropy (nats)", color="tab:red")
    ax1.set_title(title or "Match rate and entropy vs position in word")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
    return fig


# --------------------------------------------------------------------------- #
# T9: real loop vs simulator (notebook 04)
# --------------------------------------------------------------------------- #

def real_loop_tables(results: list[dict]):
    """From loop_results.jsonl records: per-policy wall-clock and live round statistics.

    Wall-clock per (policy, sentence) = median over repeats; speed-up = sum of NoDraft medians /
    sum of policy medians over the same sentences. Round statistics use repeat 0."""
    import pandas as pd
    df = pd.DataFrame([{k: r[k] for k in ("rep", "id", "policy", "wall", "n_out", "lossless")} for r in results])
    med = df.groupby(["policy", "id"]).agg(wall=("wall", "median"), n_out=("n_out", "first")).reset_index()
    base = med[med.policy == "NoDraft"].set_index("id").wall
    rows = []
    for pol, g in med.groupby("policy", sort=False):
        g = g.set_index("id")
        common = g.index.intersection(base.index)
        r0 = [r for r in results if r["policy"] == pol and r["rep"] == 0]
        rounds = [rd for r in r0 for rd in r["rounds"]]
        n_rounds = len(rounds)
        drafted = sum(rd["g"] for rd in rounds)
        acc = sum(rd["accepted"] for rd in rounds)
        rows.append(dict(policy=pol, params=json.dumps(r0[0]["params"], ensure_ascii=False) if r0 else "",
                         sentences=len(g), tokens=int(g.n_out.sum()), wall_s=float(g.wall.sum()),
                         tokens_per_s=float(g.n_out.sum() / g.wall.sum()),
                         speedup_wall=float(base.loc[common].sum() / g.wall.loc[common].sum()) if len(common) else float("nan"),
                         tokens_per_verif=sum(r["n_out"] for r in r0) / max(n_rounds, 1),
                         mean_draft_len=drafted / max(n_rounds, 1), acceptance_rate=acc / max(drafted, 1),
                         lossless_rate=float(df[df.policy == pol].lossless.mean()),
                         n_lossless=int(df[df.policy == pol].lossless.sum()), n_decodes=int((df.policy == pol).sum()),
                         extra_lookahead_rounds=sum(rd["extra_lookahead"] for rd in rounds)))
    return pd.DataFrame(rows)


def cost_model_table(results: list[dict], t_draft: float, t_target: float, verify) -> "object":
    """Measured mean draft / verify time per draft length g (repeat >= 0, all policies) vs the cost model
    (g * t_draft [+ t_draft if the extra lookahead forward was run] and verify(g) * t_target)."""
    import pandas as pd
    rows = [dict(g=rd["g"], extra=rd["extra_lookahead"], t_draft=rd["t_draft"], t_verify=rd["t_verify"])
            for r in results for rd in r["rounds"]]
    df = pd.DataFrame(rows)
    out = df.groupby("g").agg(rounds=("t_draft", "size"), t_draft_ms=("t_draft", lambda x: 1e3 * x.mean()),
                              t_verify_ms=("t_verify", lambda x: 1e3 * x.mean()),
                              frac_extra=("extra", "mean")).reset_index()
    out["model_t_draft_ms"] = 1e3 * t_draft * (out.g + out.frac_extra)
    out["model_t_verify_ms"] = [1e3 * t_target * verify(int(g)) for g in out.g]
    return out


def sim_vs_real(real_tab, sim_rows: list[dict], tol: float = 0.05):
    """Join live statistics with simulator predictions; relative errors, flag > tol."""
    import pandas as pd
    sim = pd.DataFrame(sim_rows)
    m = real_tab.merge(sim, on="policy", suffixes=("_real", "_sim"))
    for col in ("tokens_per_verif", "mean_draft_len"):
        m[f"relerr_{col}"] = (m[f"{col}_sim"] - m[f"{col}_real"]) / m[f"{col}_real"]
    for k in ("charged", "uncharged"):
        m[f"relerr_speedup_{k}"] = (m[f"speedup_sim_{k}"] - m["speedup_wall"]) / m["speedup_wall"]
    rel = [c for c in m.columns if c.startswith("relerr_")]
    m["flag_gt_5pct"] = (m[rel].abs() > tol).any(axis=1)
    return m


# --------------------------------------------------------------------------- #
# Does linguistic structure predict draft failure beyond confidence? (added 2026-11-02, user decision)
# --------------------------------------------------------------------------- #

def add_morph_start(df):
    """morph_start[t]: the cut *before* token t is a word or morpheme boundary (t = 0: True)."""
    if "morph_boundary_after" not in df:
        raise ValueError("morph labels missing")
    prev = df.groupby("sid", sort=False).morph_boundary_after.shift(1)
    df["morph_start"] = prev.fillna(True).astype(bool)
    return df


def conditional_by_entropy(df, ling: str, n_bins: int = 10, n_boot: int = 1000, seed: int = 0):
    """Match rate for ling=True vs False *within* entropy deciles, and the entropy-adjusted difference
    (count-weighted mean of within-decile differences, bins where both groups exist) with a
    sentence-bootstrap CI. Returns (per-bin table, summary dict)."""
    import pandas as pd
    d = df[["sid", "H", "match", ling]].copy()
    d["H_bin"] = pd.qcut(d.H.rank(method="first"), n_bins, labels=False)
    edges = d.groupby("H_bin").H.agg(["min", "max"])
    tab = (d.groupby(["H_bin", ling]).agg(count=("match", "size"), match_rate=("match", "mean"), mean_H=("H", "mean"))
           .reset_index())

    def adjusted(sub):
        g = sub.groupby(["H_bin", ling]).match.agg(["mean", "size"]).unstack(ling)
        g = g.dropna()
        if g.empty:
            return float("nan")
        diff = g[("mean", True)] - g[("mean", False)]
        w = g[("size", True)] + g[("size", False)]
        return float((diff * w).sum() / w.sum())

    groups = {s: g for s, g in d.groupby("sid", sort=False)}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    point = adjusted(d)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        boots.append(adjusted(pd.concat([groups[keys[i]] for i in pick], ignore_index=True)))
    boots = np.array([b for b in boots if np.isfinite(b)])
    raw = float(d[d[ling]].match.mean() - d[~d[ling]].match.mean())
    tab = tab.merge(edges.reset_index().rename(columns={"min": "H_min", "max": "H_max"}), on="H_bin")
    return tab, dict(ling=ling, raw_diff=raw, entropy_adjusted_diff=point,
                     lo=float(np.quantile(boots, 0.025)), hi=float(np.quantile(boots, 0.975)),
                     n_bins=n_bins, n_boot=len(boots))


def logit_beyond_entropy(df, ling_cols=("word_initial", "morph_start"), base_cols=("H", "top1"),
                         n_boot: int = 200, seed: int = 0):
    """Logistic regression of match on confidence features vs confidence + linguistic features.
    Features are standardised on the full split. Reports coefficients (full model), AUC of both
    models, delta AUC and delta mean log-likelihood per position, all with sentence-bootstrap CIs
    (models refit on each resample, evaluated in-sample)."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    cols = list(base_cols) + list(ling_cols)
    X = df[cols].astype(float).to_numpy()
    X = (X - X.mean(0)) / X.std(0)
    y = df["match"].astype(int).to_numpy()
    sid = df["sid"].to_numpy()
    nb = len(base_cols)

    def fit(idx):
        out = {}
        for name, k in (("base", nb), ("full", len(cols))):
            m = LogisticRegression(C=1e6, max_iter=1000).fit(X[idx, :k], y[idx])
            pr = m.predict_proba(X[idx, :k])[:, 1]
            out[name] = dict(auc=roc_auc_score(y[idx], pr),
                             ll=float(np.mean(y[idx] * np.log(pr) + (1 - y[idx]) * np.log(1 - pr))),
                             coef=m.coef_[0].tolist())
        return out

    point = fit(np.arange(len(y)))
    uniq = np.unique(sid)
    pos = {s: np.flatnonzero(sid == s) for s in uniq}
    rng = np.random.default_rng(seed)
    B = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq))
        B.append(fit(np.concatenate([pos[s] for s in pick])))

    def ci(vals):
        return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
    rows = []
    for j, c in enumerate(cols):
        lo, hi = ci([b["full"]["coef"][j] for b in B])
        rows.append(dict(quantity=f"coef_std[{c}] (full model)", value=point["full"]["coef"][j], lo=lo, hi=hi))
    for name in ("base", "full"):
        lo, hi = ci([b[name]["auc"] for b in B])
        rows.append(dict(quantity=f"AUC {name} ({'+'.join(cols[:nb] if name == 'base' else cols)})",
                         value=point[name]["auc"], lo=lo, hi=hi))
    lo, hi = ci([b["full"]["auc"] - b["base"]["auc"] for b in B])
    rows.append(dict(quantity="delta AUC (full - base)", value=point["full"]["auc"] - point["base"]["auc"], lo=lo, hi=hi))
    lo, hi = ci([b["full"]["ll"] - b["base"]["ll"] for b in B])
    rows.append(dict(quantity="delta mean log-lik per position (full - base)",
                     value=point["full"]["ll"] - point["base"]["ll"], lo=lo, hi=hi))
    out = pd.DataFrame(rows)
    out["n_positions"], out["n_sentences"], out["n_boot"] = len(y), len(uniq), n_boot
    return out
