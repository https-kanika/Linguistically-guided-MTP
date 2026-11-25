"""Week-1 kill test of the universal (LLM-independent) Hindi MTP head, end to end (Kaggle notebook 05).

Stages (each skips work whose output already exists, so a dead session can resume):
  S1 sources      FLORES+ dev/devtest English, IITB English (extra training sources) + IITB Hindi (vocab/pretrain)
  S2 vocab        unit vocabulary from public Hindi text only
  S3 collect      each small LLM: greedy outputs + teacher-forced top-20 on dev, extra, devtest
  S4 phase1       Phase-1 Llama-3-8B traces (attached Kaggle input) or a fresh collection
  S5 agree        cross-LLM agreement on word units (unit starts vs continuations)
  S6 features     tokenizer-independent features per model and split
  S7 pretrain     text-only pretraining on Hindi text; head latency -> c_head
  S8 folds        leave-one-model-out heads (zero-shot / text-only / in-model) on the small models
  S9 llama8b      zero-shot head on the 8B traces vs Entropy (Phase 1), in-model heads, baselines
  S10 report      CSV/JSON, SUMMARY.md with the go / no-go verdict, zips
"""
from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

from . import collect as C
from . import uni_sim as U
from .gpu_traces import read_jsonl, run_info
from .stats import paired_bootstrap
from .uni_features import FeatureSet, TokTable, featurize_records, featurize_texts, make_encoder, seq_text
from .unichar import CharCodec, UnitVocab
from .unihead import build_model, load_head, predict, save_head, time_head, train_head

DEFAULT_W1 = dict(
    OUT_ROOT="/kaggle/working", RUN_NAME="week1", SMOKE=False, GIT_HASH="unknown", SEED=0,
    MODELS=C.DEFAULT_MODELS, MIN_MODELS=3,
    N_DEV=None, N_DEVTEST=None, N_EXTRA=3000, MONO_N=200000,
    MAX_NEW=256, GEN_BATCH=32, TOPK=20, VAL_FRAC=0.04,
    VOCAB_MAX=50000, VOCAB_MIN_COUNT=3,
    ARCH=dict(d=256, layers=2, nhead=4), PRE_STEPS=3000, FT_STEPS=4000, INMODEL_STEPS=3000,
    BATCH=256, LR=1e-3, PRE_LR=2e-3,
    RUN_AGREE=True, N_AGREE=300, RUN_FOLDS=True, RUN_L8=True,
    PHASE1_DIR=None, PHASE1_SEARCH=["/kaggle/input"],
    TARGET8B=["meta-llama/Meta-Llama-3-8B-Instruct", "NousResearch/Meta-Llama-3-8B-Instruct"],
    COLLECT_8B_IF_MISSING=True, MAX_MEMORY={0: "8GiB", 1: "10GiB"},
    GMAX=16, TAUS=[0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    C_HEAD=None, C_SENS=[0.0, 0.02, 0.05, 0.1, 0.2], N_BOOT=1000,
    GO_SPEEDUP=1.6, GO_RECOVERY=0.7,
    PHASE1_SELECTED=None, PHASE1_MAIN=None, PHASE1_TIMING_FALLBACK=None,
    SOURCES=None,            # smoke tests: dict(dev=[(id, en)], devtest=[...], extra=[...], mono=[hi, ...])
    RESUME_FROM=None,
)


PHASE1_VERIFY_REPORT = (1.020, 1.025, 1.034, 1.043, 1.060, 1.063, 1.066, 1.074)   # REPORT.md section 3


def phase1_constants(results_dir) -> dict:
    """Phase-1 numbers embedded into the notebook at build time (results/phase1/ is not on Kaggle):
    selected Entropy / FixedK settings, the devtest main-table rows, and a timing fallback (REPORT.md:
    70.6 ms per 8B step, c = 0.3231, verify(1..8)) used only if the attached traces lack timing.json."""
    import csv
    rd = Path(results_dir)
    sel = json.loads((rd / "selected_params.json").read_text(encoding="utf-8"))
    main = {}
    with open(rd / "main_table.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["policy"] in ("Entropy", "FixedK", "Oracle", "MorphEntropy_ext1"):
                main[r["policy"]] = dict(speedup=float(r["speedup_charged"]), ci_lo=float(r["CI_low"]),
                                         ci_hi=float(r["CI_high"]), tokens_per_round=float(r["tokens_per_verif"]))
    t_t = 0.0706
    timing = dict(t_target=t_t, t_draft=t_t * float(sel["c"]), c=float(sel["c"]),
                  t_target_verify={str(g): t_t * v for g, v in zip(range(1, 9), PHASE1_VERIFY_REPORT)},
                  source="Phase-1 REPORT.md (fallback)")
    return dict(PHASE1_SELECTED={k: sel["selected"][k] for k in ("Entropy", "FixedK")}, PHASE1_MAIN=main,
                PHASE1_TIMING_FALLBACK=timing)


class Log:
    def __init__(self, path: Path):
        self.path = path
        self.t0 = time.time()

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        stamp = f"[{(time.time() - self.t0) / 60:6.1f}m] "
        print(stamp + msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(stamp + msg + "\n")


def _device():
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _mem() -> str:
    """Free RAM / GPU memory, for the log (a killed kernel leaves no traceback)."""
    parts = []
    try:
        import psutil
        vm = psutil.virtual_memory()
        parts.append(f"RAM free {vm.available / 2**30:.1f}/{vm.total / 2**30:.0f} GB")
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        if torch.cuda.is_available():
            f, t = torch.cuda.mem_get_info()
            parts.append(f"GPU0 free {f / 2**30:.1f}/{t / 2**30:.0f} GB")
    except Exception:  # noqa: BLE001
        pass
    return ", ".join(parts) or "memory n/a"


def _md_table(rows: list[dict], cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = fmt.get(c, "{:.3f}").format(v)
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _write_csv(path: Path, rows: list[dict]) -> None:
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# S1-S2: sources and vocabulary
# --------------------------------------------------------------------------- #

def stage_sources(cfg, root: Path, log) -> dict:
    p = root / "data" / "sources.json"
    if p.exists():
        S = json.loads(p.read_text(encoding="utf-8"))
        log(f"S1 sources: reused {p.name} (dev {len(S['dev'])}, devtest {len(S['devtest'])}, "
            f"extra {len(S['extra'])}, mono {len(S['mono'])})")
        return S
    if cfg["SOURCES"] is not None:
        S = {k: [list(x) if isinstance(x, (list, tuple)) else x for x in v] for k, v in cfg["SOURCES"].items()}
    else:
        dev = C.load_flores_english("dev", cfg["N_DEV"])
        devtest = C.load_flores_english("devtest", cfg["N_DEVTEST"])
        extra, mono = C.load_iitb(cfg["N_EXTRA"], cfg["MONO_N"], seed=cfg["SEED"])
        S = dict(dev=[list(r) for r in dev], devtest=[list(r) for r in devtest], extra=[list(r) for r in extra], mono=mono)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(S, ensure_ascii=False), encoding="utf-8")
    log(f"S1 sources: dev {len(S['dev'])}, devtest {len(S['devtest'])}, extra {len(S['extra'])}, mono {len(S['mono'])}")
    return S


def stage_vocab(cfg, root: Path, S: dict, log) -> UnitVocab:
    p = root / "data" / "unit_vocab.json"
    if p.exists():
        V = UnitVocab.load(p)
        log(f"S2 vocab: reused ({V.size} units)")
        return V
    V, st = UnitVocab.build(S["mono"], cfg["VOCAB_MAX"], cfg["VOCAB_MIN_COUNT"])
    V.save(p)
    (root / "results" / "vocab_stats.json").write_text(json.dumps(st, indent=1))
    log(f"S2 vocab: {st}")
    return V


# --------------------------------------------------------------------------- #
# S3: collection
# --------------------------------------------------------------------------- #

def _rows(S, split):
    return [(int(i), s) for i, s in S[split]]


def stage_collect(cfg, root: Path, S: dict, log, device) -> dict:
    out = {}
    summ = []
    for spec in cfg["MODELS"]:
        key = spec["key"]
        d = root / "data" / key
        log(f"S3 collect: {key} ({_mem()})")
        try:
            res, info, _ = C.collect_model(spec, {s: _rows(S, s) for s in ("dev", "extra", "devtest")}, d,
                                           cfg["GEN_BATCH"], cfg["MAX_NEW"], device, cfg["TOPK"], log)
            rows = []
            for s, recs in res.items():
                txt = "".join(r["out_text"] for r in recs)
                deva = sum("ऀ" <= ch <= "ॿ" for ch in txt) / max(sum(not ch.isspace() for ch in txt), 1)
                rows.append(dict(model=key, name=info["name"], split=s, n=len(recs), status="ok",
                                 mean_tokens=float(np.mean([len(r["out_ids"]) for r in recs])) if recs else 0.0,
                                 eos_rate=float(np.mean([r["eos_end"] for r in recs])) if recs else 0.0,
                                 hit_max=int(sum(len(r["out_ids"]) >= cfg["MAX_NEW"] for r in recs)),
                                 deva_frac=deva, tf_argmax=info.get("tf_argmax_agreement", {}).get(s, float("nan"))))
        except Exception as e:  # noqa: BLE001
            log(f"S3 collect: model {key} FAILED and is skipped: {type(e).__name__}: {str(e)[:300]}")
            summ.append(dict(model=key, status=f"failed: {type(e).__name__}"))
            continue
        out[key] = dict(records=res, info=info, spec=spec)
        summ.extend(rows)
        log(f"S3 collect: {key} done ({_mem()})")
    _write_csv(root / "results" / "collection_summary.csv", summ)
    log(f"S3 collect: {len(out)} models ok: {list(out)}")
    return out


# --------------------------------------------------------------------------- #
# S4: Phase-1 Llama-3-8B traces
# --------------------------------------------------------------------------- #

def _find(name: str, dirs) -> Path | None:
    for d in dirs:
        if d and Path(d).exists():
            hits = sorted(Path(d).rglob(name))
            if hits:
                return hits[0]
    return None


def find_phase1(cfg, root: Path, log) -> dict:
    dirs = [cfg["PHASE1_DIR"], cfg["RESUME_FROM"], *cfg["PHASE1_SEARCH"], str(root / "phase1")]
    found = {}
    for split in ("dev", "devtest"):
        npz = _find(f"traces_{split}.npz", dirs)
        if npz is None:
            z = _find(f"traces_v2_{split}.zip", dirs)
            if z is not None:
                dst = root / "phase1" / split
                dst.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(dst)
                npz = dst / f"traces_{split}.npz"
                log(f"S4: extracted {z}")
        if npz is not None and npz.exists():
            jl = npz.parent / f"target_outputs_{split}.jsonl"
            jl = jl if jl.exists() else _find(f"target_outputs_{split}.jsonl", dirs)
            tm = npz.parent / "timing.json"
            tm = tm if tm.exists() else _find("timing.json", [str(npz.parent.parent)] + dirs)
            found[split] = dict(npz=npz, jsonl=jl, timing=tm)
            log(f"S4: Phase-1 {split}: {npz} (outputs: {jl}, timing: {tm})")
    return found


def load_tokenizer_any(names):
    from transformers import AutoTokenizer
    last = None
    for n in names:
        try:
            return AutoTokenizer.from_pretrained(n), n
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"no tokenizer from {names}: {last}")


def stage_phase1(cfg, root: Path, S: dict, log, device) -> dict | None:
    from .traces import load_traces
    found = find_phase1(cfg, root, log)
    timing = None
    if "devtest" in found:
        L8 = dict(source="phase1", traces={}, records={})
        for split, f in found.items():
            traces, meta = load_traces(f["npz"])
            outs = {int(r["id"]): r for r in read_jsonl(f["jsonl"])} if f["jsonl"] else {}
            eos_ids = set(meta.get("eos_ids", []))
            recs = []
            for t in traces:
                o = outs.get(int(t.id), {})
                recs.append(dict(id=int(t.id), prompt_text=C.head_prompt_text(o["src"]) if "src" in o else "",
                                 prompt_ids=o.get("prompt_ids"), out_ids=np.asarray(t.target_ids, np.int64),
                                 topk_ids=t.target_topk_ids, topk_logp=t.target_topk_logp.astype(np.float32),
                                 eos_end=bool(len(t.target_ids)) and int(t.target_ids[-1]) in eos_ids))
            L8["traces"][split], L8["records"][split] = traces, recs
            L8["eos_ids"] = sorted(eos_ids)
            if f["timing"] and timing is None:
                timing = json.loads(Path(f["timing"]).read_text())
        if "dev" not in L8["records"]:
            log("S4: Phase-1 dev traces not found: in-model 8B heads and 8B-dev tuning are skipped")
    elif cfg["COLLECT_8B_IF_MISSING"]:
        log("S4: Phase-1 traces not attached -> collecting Llama-3-8B outputs here (slow, ~2-3 h on 2xT4)")
        spec = dict(key="llama3-8b", names=cfg["TARGET8B"], dtype="fp16", device_map="auto", family="llama3")
        d = root / "data" / "llama3-8b"
        res, info, mt = C.collect_model(spec, {s: _rows(S, s) for s in ("dev", "devtest")}, d, 16, cfg["MAX_NEW"],
                                        device, cfg["TOPK"], log, max_memory=cfg["MAX_MEMORY"], keep_loaded=True)
        tp = d / "timing.json"
        if not tp.exists() and mt is not None and mt[0] is not None:
            from .gpu_traces import time_fwd
            m = mt[0]
            t_t = time_fwd(m, 1, device=device)
            ver = {str(g): time_fwd(m, g + 1, device=device) for g in range(1, 9)}
            tp.write_text(json.dumps(dict(t_target=t_t, t_target_verify=ver, c=None, source="week1 fresh"), indent=1))
        if mt is not None:
            del mt
            C.free_model()
        timing = json.loads(tp.read_text()) if tp.exists() else None
        L8 = dict(source="fresh", traces={}, records=res, eos_ids=info["eos_ids"])
    else:
        log("S4: no Phase-1 traces and COLLECT_8B_IF_MISSING=False -> the Llama-3-8B fold is skipped")
        return None
    if timing is None:
        timing = cfg["PHASE1_TIMING_FALLBACK"]
        log("S4: timing.json not found; using the Phase-1 REPORT numbers embedded in the notebook")
    tok, name = load_tokenizer_any(cfg["TARGET8B"])
    L8.update(timing=timing, tok=tok, tok_name=name)
    log(f"S4: 8B data from {L8['source']}; splits {list(L8['records'])}; tokenizer {name}; "
        f"t_target {timing['t_target'] * 1000:.1f} ms")
    return L8


# --------------------------------------------------------------------------- #
# S5: cross-LLM agreement
# --------------------------------------------------------------------------- #

def stage_agree(cfg, root: Path, M: dict, L8, tables: dict, log, device) -> list:
    p = root / "results" / "agreement.csv"
    if p.exists():
        import pandas as pd
        log("S5 agree: reused")
        return pd.read_csv(p).to_dict("records")
    texts = {}
    for key, m in M.items():
        recs = m["records"]["devtest"][: cfg["N_AGREE"]]
        texts[key] = [(r["src"], seq_text(tables[key], r["out_ids"]).text) for r in recs]
    if L8 is not None and "devtest" in L8["records"]:
        tl = TokTable(L8["tok"], L8["eos_ids"])
        texts["llama3-8b"] = []
        for r in L8["records"]["devtest"][: cfg["N_AGREE"]]:
            src = r["prompt_text"].split("\n\n", 1)[-1] if r["prompt_text"] else ""
            texts["llama3-8b"].append((src, seq_text(tl, r["out_ids"]).text))
    rows = []
    for key, m in M.items():
        model, tok, _ = C.load_model(m["spec"], device)
        enc = make_encoder(tok)
        for a, items in texts.items():
            items = [(s, t) for s, t in items if t.strip()]
            if not items:
                continue
            ids = enc([t for _, t in items])
            recs = [dict(prompt_ids=tok(C.build_model_prompt(tok, m["spec"], s), add_special_tokens=False)["input_ids"],
                         out_ids=i) for (s, _), i in zip(items, ids)]
            am = C.teacher_forced_topk(model, recs, device, argmax_only=True)
            tot = {}
            for (s, t), i, a_ in zip(items, ids, am):
                for k_, v in U.unit_agreement(t, i, list(a_), tables[key]).items():
                    tot[k_] = tot.get(k_, 0) + v
            row = dict(text_from=a, model=key, same=(a == key), **U.agreement_table(tot))
            rows.append(row)
            log(f"S5 agree: {key} on {a}'s text: start {row['p_start']:.3f} cont {row['p_cont_token']:.3f} "
                f"unit {row['p_unit']:.3f} unit|start {row['p_unit_given_start']:.3f}")
        del model, tok
        C.free_model()
    _write_csv(p, rows)
    return rows


# --------------------------------------------------------------------------- #
# S6: features
# --------------------------------------------------------------------------- #

def _split_records(recs, frac, seed):
    rng = np.random.default_rng(seed)
    m = rng.random(len(recs)) < frac
    if not m.any() and len(recs) > 1:
        m[0] = True
    return [r for r, x in zip(recs, m) if not x], [r for r, x in zip(recs, m) if x]


def stage_features(cfg, root: Path, M: dict, L8, V: UnitVocab, S: dict, log):
    from transformers import AutoTokenizer
    fd = root / "features"
    fd.mkdir(exist_ok=True)
    codec = CharCodec()
    F, R, T = {}, {}, {}
    for key, m in M.items():
        info = m["info"]
        tok = AutoTokenizer.from_pretrained(info["name"])
        T[key] = dict(table=TokTable(tok, info["eos_ids"], info["vocab_size"], codec), enc=make_encoder(tok), tok=tok)
        tr, va = _split_records(m["records"]["dev"] + m["records"]["extra"], cfg["VAL_FRAC"], cfg["SEED"])
        R[key] = dict(train=tr, val=va, test=m["records"]["devtest"])
        F[key] = {}
        for part, recs in R[key].items():
            p = fd / f"{key}_{part}.npz"
            if p.exists():
                F[key][part] = FeatureSet.load(p)
            else:
                t0 = time.time()
                F[key][part] = featurize_records(recs, T[key]["table"], V, codec, meta=dict(model=key, part=part))
                F[key][part].save(p)
                log(f"S6 features {key}/{part}: {F[key][part].n_pos} positions ({time.time() - t0:.0f}s)")
    if L8 is not None:
        key = "llama3-8b"
        T[key] = dict(table=TokTable(L8["tok"], L8["eos_ids"], None, codec), enc=make_encoder(L8["tok"]), tok=L8["tok"])
        R[key], F[key] = {}, {}
        for split, part in (("dev", "train"), ("devtest", "test")):
            if split not in L8["records"]:
                continue
            R[key][part] = L8["records"][split]
            p = fd / f"{key}_{part}.npz"
            if p.exists():
                F[key][part] = FeatureSet.load(p)
            else:
                F[key][part] = featurize_records(R[key][part], T[key]["table"], V, codec, meta=dict(model=key, part=part))
                F[key][part].save(p)
                log(f"S6 features {key}/{part}: {F[key][part].n_pos} positions")
    p = fd / "mono.npz"
    if p.exists():
        mono = FeatureSet.load(p)
    else:
        mono = featurize_texts(S["mono"], V, codec, seed=cfg["SEED"])
        mono.save(p)
    log(f"S6 features mono: {mono.n_pos} positions")
    # coverage of the vocabulary on each model's test outputs
    cov = []
    for key in R:
        if "test" in R[key]:
            texts = [seq_text(T[key]["table"], r["out_ids"]).text for r in R[key]["test"]]
            cov.append(dict(model=key, unit_coverage=V.coverage(texts),
                            head1_target_ok=float((F[key]["test"].tgt[:, 0] != -100).mean()) if F[key]["test"].n_pos else 0.0))
    _write_csv(root / "results" / "vocab_coverage.csv", cov)
    return F, R, T, mono


# --------------------------------------------------------------------------- #
# S7-S9: heads
# --------------------------------------------------------------------------- #

def _train_or_load(path: Path, init: Path | None, V, cfg, sources, val, use_cands, steps, lr, device, log, tag):
    if path.exists():
        m, extra = load_head(path, device)
        log(f"  {tag}: reused {path.name}")
        return m, extra
    if init is not None and init.exists():
        m, _ = load_head(init, device)
    else:
        m = build_model(V.size, cfg["ARCH"]).to(device)
    log(f"  {tag}: training {steps} steps on {sum(len(i) for _, i in sources)} positions (cands={use_cands})")
    out = train_head(m, sources, steps, cfg["BATCH"], lr, device, use_cands, val=val, log=log, seed=cfg["SEED"],
                     log_every=max(steps // 10, 1))
    save_head(m, path, dict(tag=tag, use_cands=use_cands, val=out.get("val"), train_sec=out["train_sec"]))
    return m, dict(tag=tag, use_cands=use_cands, val=out.get("val"), train_sec=out["train_sec"])


def _pred(path: Path, model, fs, device, use_cands):
    if path.exists():
        z = np.load(path)
        return z["W"], z["P"]
    W, P = predict(model, fs, device, use_cands)
    np.savez(path, W=W, P=P)
    return W, P


def _truth_W(fs) -> np.ndarray:
    W = fs.tgt.copy()
    W[W < 0] = -1
    for i in range(1, W.shape[1]):          # a chain stops at the first missing unit
        W[:, i] = np.where(W[:, i - 1] < 0, -1, W[:, i])
    return W


def _head_acc(W, fs) -> dict:
    t = fs.tgt
    out = {}
    for i in range(t.shape[1]):
        m = t[:, i] != -100
        out[f"acc{i + 1}"] = float((W[m, i] == t[m, i]).mean()) if m.any() else float("nan")
    m = (t[:, 0] != -100)
    out["chain2"] = float(((W[:, 0] == t[:, 0]) & (W[:, 1] == t[:, 1]))[m].mean()) if m.any() else float("nan")
    return out


class Evaluator:
    """Converts head predictions on one target's sequences into that target's tokens and simulates."""

    def __init__(self, key, records, T, V, verify, c_head, cfg):
        self.key, self.V, self.verify, self.c_head, self.cfg = key, V, verify, c_head, cfg
        self.seqs = U.build_simseqs(records, T["table"], V)
        self.enc = T["enc"]
        self.table = T["table"]
        self._cache: dict = {}

    def drafts(self, W):
        """Converted drafts for predictions W (cached: the same W is simulated under many settings)."""
        import hashlib
        key = hashlib.md5(np.ascontiguousarray(W).tobytes()).hexdigest()
        if key not in self._cache:
            if len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = U.convert_all(self.seqs, W, self.V, self.enc, self.table.bytes, self.cfg["GMAX"])
        return self._cache[key]

    def run(self, W, P, tau, c_head=None, oracle_level=False):
        d, st = self.drafts(W)
        R = U.simulate(self.seqs, d, P, tau, self.c_head if c_head is None else c_head, self.verify,
                       self.cfg["GMAX"], oracle_level=oracle_level)
        return R, d, st

    def tau_grid(self, W, P):
        d, _ = self.drafts(W)
        return [dict(tau=t, speedup=U.simulate(self.seqs, d, P, t, self.c_head, self.verify, self.cfg["GMAX"])["speedup"])
                for t in self.cfg["TAUS"]]


_VAL_EV: dict = {}


def _val_tau(model, keys, F, R, T, V, verify, c_head, cfg, device, use_cands) -> float:
    """Draft-length threshold chosen on the *training* models' validation outputs (never the held-out one)."""
    grids = []
    for k in keys:
        if not F[k]["val"].n_pos:
            continue
        if k not in _VAL_EV:
            _VAL_EV[k] = Evaluator(k, R[k]["val"], T[k], V, verify, c_head, cfg)
        Wv, Pv = predict(model, F[k]["val"], device, use_cands)
        grids.append(_VAL_EV[k].tau_grid(Wv, Pv))
    return _pick_tau(grids) if grids else 0.0


def _pick_tau(grids: list[list[dict]]) -> float:
    taus = [g["tau"] for g in grids[0]]
    mean = [np.mean([gr[i]["speedup"] for gr in grids]) for i in range(len(taus))]
    return float(taus[int(np.argmax(mean))])


def stage_heads(cfg, root: Path, F, R, T, mono, V, L8, log, device) -> dict:
    hd, pd_ = root / "heads", root / "preds"
    hd.mkdir(exist_ok=True); pd_.mkdir(exist_ok=True)
    res = dict(lomo=[], lomo_sim=[], l8=[], l8_tau=[], l8_sens=[], conv=[], nn_cov=[], timing={})
    # ---- S7 pretrain + timing
    pre = hd / "pretrain_text.pt"
    tr, va = mono.split_by_seq(0.01, cfg["SEED"])
    m_pre, _ = _train_or_load(pre, None, V, cfg, [(mono, tr)], [(mono, va)], False, cfg["PRE_STEPS"], cfg["PRE_LR"],
                              device, log, "S7 pretrain (text only, Hindi corpus)")
    timing = cfg["PHASE1_TIMING_FALLBACK"] if L8 is None else L8["timing"]
    verify = U.Verify(timing)
    any_fs = next(F[k]["test"] for k in F if "test" in F[k] and F[k]["test"].n_pos)
    th = time_head(m_pre, any_fs, device, use_cands=True) if device != "cpu" or cfg["SMOKE"] else {}
    tt = time_head(m_pre, any_fs, device, use_cands=False) if device != "cpu" or cfg["SMOKE"] else {}
    c_meas = (th["t_head_gpu"] + th["t_inputs_cpu"]) / timing["t_target"] if th else 0.05
    c_head = float(cfg["C_HEAD"]) if cfg["C_HEAD"] is not None else c_meas
    res["timing"] = dict(head_with_cands=th, head_text_only=tt, t_target=timing["t_target"], c_head_measured=c_meas,
                         c_head_used=c_head, verify=verify.as_dict(),
                         note="c_head = (GPU forward + chained argmax + CPU input building) / 8B decode step")
    log(f"S7 head latency: {th.get('t_head_gpu', 0) * 1000:.2f} ms GPU + {th.get('t_inputs_cpu', 0) * 1000:.2f} ms CPU "
        f"-> c_head = {c_meas:.4f} (used {c_head:.4f}); params {th.get('params')}")
    small = [k for k in F if k != "llama3-8b"]

    def sources(keys, part="train"):
        return [(F[k][part], F[k][part].trainable()) for k in keys if part in F[k] and F[k][part].n_pos]

    # ---- S8 leave-one-model-out on the small models
    if cfg["RUN_FOLDS"] and len(small) >= 3:
        for h in small:
            others = [k for k in small if k != h]
            ev = Evaluator(h, R[h]["test"], T[h], V, verify, c_head, cfg)
            variants = [("zeroshot", others, True, pre, cfg["FT_STEPS"]),
                        ("textonly", others, False, pre, cfg["FT_STEPS"]),
                        ("inmodel", [h], True, pre, cfg["INMODEL_STEPS"])]
            preds = {}
            for name, keys, uc, init, steps in variants:
                path = hd / f"lomo_{h}_{name}.pt"
                m, ex = _train_or_load(path, init, V, cfg, sources(keys), sources(keys, "val"), uc, steps, cfg["LR"],
                                       device, log, f"S8 fold {h} / {name}")
                W, P = _pred(pd_ / f"lomo_{h}_{name}_test.npz", m, F[h]["test"], device, uc)
                tau = _val_tau(m, keys, F, R, T, V, verify, c_head, cfg, device, uc)
                preds[name] = (W, P, tau)
                res["lomo"].append(dict(heldout=h, variant=name, train_models="+".join(keys), tau=tau,
                                        **_head_acc(W, F[h]["test"]), **{f"val_{k}": v for k, v in (ex.get("val") or {}).items()}))
                del m
            truthW = _truth_W(F[h]["test"])
            for name, (W, P, tau) in list(preds.items()) + [("truth_units", (truthW, np.ones_like(truthW, np.float32), 0.0))]:
                Rr, _, st = ev.run(W, P, tau)
                res["lomo_sim"].append(dict(heldout=h, method=name, tau=tau, speedup_8b_cost=Rr["speedup"],
                                            tokens_per_round=Rr["tokens_per_round"], mean_draft_len=Rr["mean_draft_len"],
                                            acceptance=Rr["acceptance"]))
                res["conv"].append(dict(target=h, method=name, **st))
            for bname, dr in (("pld", U.pld_drafts(ev.seqs)), ("logitspec", U.logitspec_drafts(ev.seqs))):
                Rb = U.simulate(ev.seqs, dr, U.ones_like_probs(dr), 0.0, 0.0, verify, cfg["GMAX"])
                res["lomo_sim"].append(dict(heldout=h, method=bname, tau=0.0, speedup_8b_cost=Rb["speedup"],
                                            tokens_per_round=Rb["tokens_per_round"], mean_draft_len=Rb["mean_draft_len"],
                                            acceptance=Rb["acceptance"]))
            res["nn_cov"].append(dict(target=h, **U.next_next_coverage(ev.seqs)))
            log(f"S8 fold {h}: " + "; ".join(f"{r['method']} {r['speedup_8b_cost']:.3f}x ({r['tokens_per_round']:.2f} tok/round)"
                                            for r in res["lomo_sim"] if r["heldout"] == h))
    elif cfg["RUN_FOLDS"]:
        log(f"S8: only {len(small)} small models loaded (< 3): leave-one-model-out skipped")

    # ---- S9 Llama-3-8B
    if cfg["RUN_L8"] and L8 is not None and "llama3-8b" in F and "test" in F["llama3-8b"]:
        res.update(stage_l8(cfg, root, F, R, T, V, L8, small, sources, pre, verify, c_head, log, device))
    return res


def stage_l8(cfg, root, F, R, T, V, L8, small, sources, pre, verify, c_head, log, device) -> dict:
    hd, pd_ = root / "heads", root / "preds"
    key = "llama3-8b"
    out = dict(l8=[], l8_tau=[], l8_sens=[], conv=[], nn_cov=[], l8_heads=[])
    ev = Evaluator(key, R[key]["test"], T[key], V, verify, c_head, cfg)
    ev_dev = Evaluator(key, R[key]["train"], T[key], V, verify, c_head, cfg) if "train" in R[key] else None
    families = {s["key"]: s.get("family", "") for s in cfg["MODELS"]}
    nollama = [k for k in small if families.get(k) != "llama3"]
    variants = [("zeroshot_all", small, True, pre, cfg["FT_STEPS"], "train"),
                ("textonly_all", small, False, pre, cfg["FT_STEPS"], "train")]
    if len(nollama) >= 2 and len(nollama) < len(small):
        variants.append(("zeroshot_nollama", nollama, True, pre, cfg["FT_STEPS"], "train"))
    preds = {}
    for name, keys, uc, init, steps, part in variants:
        path = hd / f"l8_{name}.pt"
        m, ex = _train_or_load(path, init, V, cfg, sources(keys), sources(keys, "val"), uc, steps, cfg["LR"], device, log,
                               f"S9 {name}")
        W, P = _pred(pd_ / f"l8_{name}_test.npz", m, F[key]["test"], device, uc)
        tau_zs = _val_tau(m, keys, F, R, T, V, verify, c_head, cfg, device, uc)
        tau_dev = None
        if ev_dev is not None:
            Wd, Pd = _pred(pd_ / f"l8_{name}_dev.npz", m, F[key]["train"], device, uc)
            tau_dev = _pick_tau([ev_dev.tau_grid(Wd, Pd)])
        preds[name] = dict(W=W, P=P, tau=tau_zs, tau_dev=tau_dev, use_cands=uc, train=keys)
        out["l8_heads"].append(dict(variant=name, train_models="+".join(keys), tau_zeroshot=tau_zs, tau_8b_dev=tau_dev,
                                    **_head_acc(W, F[key]["test"])))
        del m
    # in-model references: need 8B dev traces; trained on 90% of dev, threshold chosen on the other 10%
    if "train" in F[key] and F[key]["train"].n_pos:
        fs8 = F[key]["train"]
        val_seq = np.random.default_rng(cfg["SEED"]).random(len(fs8.prompt_u)) < 0.1
        if not val_seq.any():
            val_seq[0] = True
        trn = fs8.trainable()
        tr8 = [(fs8, trn[~val_seq[fs8.pos_seq[trn]]])]
        va8 = [(fs8, trn[val_seq[fs8.pos_seq[trn]]])]
        va_pos = np.flatnonzero(val_seq[fs8.pos_seq])
        ev_dval = Evaluator(key, [r for r, v in zip(R[key]["train"], val_seq) if v], T[key], V, verify, c_head, cfg)
        for name, init, steps in (("inmodel_ft", hd / "l8_zeroshot_all.pt", cfg["INMODEL_STEPS"]),
                                  ("inmodel_own", pre, cfg["FT_STEPS"])):
            path = hd / f"l8_{name}.pt"
            m, ex = _train_or_load(path, init, V, cfg, tr8, va8, True, steps, cfg["LR"] * (0.5 if name == "inmodel_ft" else 1),
                                   device, log, f"S9 {name}")
            W, P = _pred(pd_ / f"l8_{name}_test.npz", m, F[key]["test"], device, True)
            Wv, Pv = predict(m, fs8, device, True, idx=va_pos)
            tau_dev = _pick_tau([ev_dval.tau_grid(Wv, Pv)])
            preds[name] = dict(W=W, P=P, tau=tau_dev, tau_dev=tau_dev, use_cands=True, train=[key])
            out["l8_heads"].append(dict(variant=name, train_models=key + " dev (90%)", tau_zeroshot=None, tau_8b_dev=tau_dev,
                                        **_head_acc(W, F[key]["test"])))
            del m
    # ---- baselines on the same 8B devtest sentences
    sims = {}
    ent = None
    if L8["source"] == "phase1" and "devtest" in L8["traces"] and cfg["PHASE1_SELECTED"]:
        from .policies import Entropy, FixedK
        from .simulator import GAMMA_MAX, MeasuredVerify, simulate as p1sim
        t = L8["timing"]
        c1 = t.get("c") or t["t_draft"] / t["t_target"]
        mv = MeasuredVerify(t)
        sel = cfg["PHASE1_SELECTED"]
        for nm, pol, gm in (("Entropy (Phase 1, 1B draft)", Entropy(**sel["Entropy"]), GAMMA_MAX),
                            ("FixedK (Phase 1, 1B draft)", FixedK(**sel["FixedK"]), sel["FixedK"]["k"])):
            r1 = p1sim(L8["traces"]["devtest"], pol, gm, c1, mv, record_rounds=False)
            em, co = r1.sent_emitted.astype(float), r1.sent_cost
            sims[nm] = dict(sent_emitted=em, sent_cost=co, speedup=float(em.sum() / co.sum()),
                            tokens_per_round=r1.emitted / max(r1.rounds, 1), mean_draft_len=r1.drafted / max(r1.rounds, 1),
                            acceptance=r1.accepted / max(r1.drafted, 1), rounds=r1.rounds)
        ent = sims["Entropy (Phase 1, 1B draft)"]
    for bname, dr in (("Prompt lookup (training-free)", U.pld_drafts(ev.seqs)),
                      ("LogitSpec-style chain (training-free)", U.logitspec_drafts(ev.seqs))):
        sims[bname] = U.simulate(ev.seqs, dr, U.ones_like_probs(dr), 0.0, 0.0, verify, cfg["GMAX"])
    out["nn_cov"].append(dict(target=key, **U.next_next_coverage(ev.seqs)))
    truthW = _truth_W(F[key]["test"])
    sims["Truth units (upper bound of this head design)"], _, st = ev.run(truthW, np.ones_like(truthW, np.float32), 0.0)
    out["conv"].append(dict(target=key, method="truth_units", **st))
    for name, pr in preds.items():
        R_, _, st = ev.run(pr["W"], pr["P"], pr["tau"])
        label = {"zeroshot_all": "Universal head, zero-shot (all small LLMs)",
                 "zeroshot_nollama": "Universal head, zero-shot (no Llama-family data)",
                 "textonly_all": "Text-only head, zero-shot (ablation: no distribution input)",
                 "inmodel_ft": "In-model head (zero-shot head fine-tuned on 8B dev)",
                 "inmodel_own": "In-model head (8B dev only)"}[name]
        sims[label] = R_
        pr["label"] = label
        out["conv"].append(dict(target=key, method=name, **st))
        R0, _, _ = ev.run(pr["W"], pr["P"], 0.0)
        sims[label + " [tau=0]"] = R0
        Ro, _, _ = ev.run(pr["W"], pr["P"], 0.0, oracle_level=True)
        sims[label + " [oracle length, not a method]"] = Ro
        for t_row in ev.tau_grid(pr["W"], pr["P"]):
            out["l8_tau"].append(dict(variant=name, **t_row))
        for cs in cfg["C_SENS"]:
            Rs, _, _ = ev.run(pr["W"], pr["P"], pr["tau"], c_head=cs)
            out["l8_sens"].append(dict(variant=name, c_head=cs, speedup=Rs["speedup"]))
    ref = ent if ent is not None else None
    for nm, R_ in sims.items():
        tau = next((p["tau"] for p in preds.values() if p.get("label") == nm), None)
        out["l8"].append(U.summary_row(nm, R_, cfg["N_BOOT"], cfg["SEED"], ref=ref if nm != "Entropy (Phase 1, 1B draft)" else None,
                                       tau=tau, c_head=c_head if nm.startswith(("Universal", "Text-only", "In-model", "Truth")) else None))
    # dist vs text-only paired comparison (the novelty signal)
    if "zeroshot_all" in preds and "textonly_all" in preds:
        a, b = sims[preds["zeroshot_all"]["label"]], sims[preds["textonly_all"]["label"]]
        out["dist_vs_text"] = paired_bootstrap(a["sent_emitted"], a["sent_cost"], b["sent_emitted"], b["sent_cost"],
                                               cfg["N_BOOT"], cfg["SEED"])
    out["preds_meta"] = {k: dict(tau=v["tau"], tau_dev=v["tau_dev"], train=v["train"]) for k, v in preds.items()}
    out["phase1_reference"] = cfg["PHASE1_MAIN"]
    log("S9 Llama-3-8B devtest:\n" + _md_table(out["l8"], ["method", "speedup", "ci_lo", "ci_hi", "tokens_per_round"]))
    return out


# --------------------------------------------------------------------------- #
# S10: verdict and report
# --------------------------------------------------------------------------- #

def verdict(cfg, res: dict) -> dict:
    rows = {r["method"]: r for r in res.get("l8", [])}
    zs = rows.get("Universal head, zero-shot (all small LLMs)")
    if zs is None:
        return dict(decision="INCOMPLETE", reason="the Llama-3-8B fold did not run (attach the Phase-1 traces or allow collection)")
    ent = rows.get("Entropy (Phase 1, 1B draft)")
    ent_s = ent["speedup"] if ent else (cfg["PHASE1_MAIN"] or {}).get("Entropy", {}).get("speedup")
    inm = rows.get("In-model head (zero-shot head fine-tuned on 8B dev)") or rows.get("In-model head (8B dev only)")
    txt = rows.get("Text-only head, zero-shot (ablation: no distribution input)")
    rec = (zs["speedup"] - 1) / (inm["speedup"] - 1) if inm and inm["speedup"] > 1 else None
    beats = zs.get("diff_lo", -1) > 0 if ent else (ent_s is not None and zs["speedup"] > ent_s)
    dist_helps = res.get("dist_vs_text") and res["dist_vs_text"]["lo"] > 0
    if zs["speedup"] >= cfg["GO_SPEEDUP"] and (rec is None or rec >= cfg["GO_RECOVERY"]):
        dec = "GO"
    elif beats and (rec is None or rec >= cfg["GO_RECOVERY"]):
        dec = "WEAK GO"
    else:
        dec = "NO GO"
    return dict(decision=dec, zeroshot_speedup=zs["speedup"], zeroshot_ci=[zs["ci_lo"], zs["ci_hi"]],
                entropy_speedup=ent_s, beats_entropy_ci=bool(beats), inmodel_speedup=inm["speedup"] if inm else None,
                recovery=rec, textonly_speedup=txt["speedup"] if txt else None,
                distribution_input_helps=bool(dist_helps), dist_vs_text=res.get("dist_vs_text"),
                criteria=dict(go_speedup=cfg["GO_SPEEDUP"], go_recovery=cfg["GO_RECOVERY"]))


def write_report(cfg, root: Path, res: dict, agree: list, log) -> dict:
    rd = root / "results"
    for name in ("lomo", "lomo_sim", "l8", "l8_tau", "l8_sens", "conv", "nn_cov", "l8_heads"):
        if res.get(name):
            _write_csv(rd / f"{name}.csv", res[name])
    (rd / "timing_head.json").write_text(json.dumps(res.get("timing", {}), indent=1, default=float))
    V = verdict(cfg, res)
    (rd / "verdict.json").write_text(json.dumps(V, indent=1, default=float))
    L = [f"# Week-1 kill test: universal Hindi MTP head\n", f"Build `{cfg['GIT_HASH']}`. Generated by `lgmtp.week1`.\n",
         f"## Verdict: **{V['decision']}**\n", "```", json.dumps(V, indent=1, default=float), "```\n"]
    if res.get("l8"):
        L += ["## Llama-3-8B, FLORES devtest (exact trace simulation)\n",
              _md_table(res["l8"], ["method", "speedup", "ci_lo", "ci_hi", "tokens_per_round", "mean_draft_len",
                                    "acceptance", "diff_vs_ref", "diff_lo", "diff_hi", "tau"]), "",
              "Paired differences are against Entropy (Phase 1, 1B draft) on the same sentences. "
              "Rows marked [oracle length] pick the best draft length with hindsight and are diagnostics only.\n"]
    if res.get("l8_heads"):
        L += ["## Head accuracy on 8B devtest (teacher forced, per unit)\n",
              _md_table(res["l8_heads"], ["variant", "train_models", "acc1", "acc2", "acc3", "chain2", "tau_zeroshot", "tau_8b_dev"]), ""]
    if res.get("lomo_sim"):
        L += ["## Leave-one-model-out on the small LLMs (speed-up under the 8B cost model)\n",
              _md_table(res["lomo_sim"], ["heldout", "method", "speedup_8b_cost", "tokens_per_round", "mean_draft_len", "acceptance", "tau"]), ""]
    if res.get("lomo"):
        L += ["## Leave-one-model-out head accuracy\n",
              _md_table(res["lomo"], ["heldout", "variant", "train_models", "acc1", "acc2", "acc3", "chain2"]), ""]
    if agree:
        L += ["## Cross-LLM agreement on word units (teacher forced on another model's output)\n",
              _md_table(agree, ["text_from", "model", "p_start", "p_cont_token", "p_unit", "p_unit_given_start"]), ""]
    if res.get("nn_cov"):
        L += ["## Next-next token in the top-k of the last distribution (LogitSpec observation)\n",
              _md_table(res["nn_cov"], ["target", "top1", "top2", "top5", "top10", "top20"]), ""]
    if res.get("conv"):
        L += ["## Draft conversion to target tokens\n",
              _md_table(res["conv"], ["target", "method", "positions", "levels", "anchored", "fallback", "failed", "empty"]), ""]
    t = res.get("timing", {})
    if t:
        L += ["## Head cost\n", "```", json.dumps(t, indent=1, default=float), "```"]
    (rd / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    log(f"S10 verdict: {V['decision']}")
    return V


def _resume_copy(run: Path, root: Path, log):
    """Copy an earlier run folder (the one holding data/sources.json) into `root`, never overwriting."""
    n_new = n_had = 0
    for f in sorted(Path(run).rglob("*")):
        if f.is_file():
            dst = root / f.relative_to(run)
            if dst.exists():
                n_had += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n_new += 1
    models = sorted(p.name for p in (Path(run) / "data").iterdir() if p.is_dir())
    log(f"resume: copied {n_new} files ({n_had} already present) from {run}; model folders: {models or 'none'}")


def _unzip_safe(z: zipfile.ZipFile, dst: Path) -> None:
    """extractall that also accepts backslash separators (Windows PowerShell's Compress-Archive)."""
    for info in z.infolist():
        name = info.filename.replace("\\", "/")
        parts = [p for p in name.split("/") if p not in ("", ".")]
        if name.endswith("/") or not parts or ".." in parts or ":" in parts[0]:
            continue
        out = dst.joinpath(*parts)
        out.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as fi, open(out, "wb") as fo:
            shutil.copyfileobj(fi, fo)


def locate_run(paths, run_name: str) -> Path | None:
    """An earlier run among `paths` (folders searched recursively, or .zip files): the folder holding
    data/sources.json, whatever it is called. Handles a notebook output attached as input, a downloaded
    output re-uploaded as a Dataset (unzipped by Kaggle or kept as a .zip) and zips without the run folder.
    A zip is extracted to a temp folder. Prefers a folder named `run_name`, then the shallowest."""
    import tempfile

    def is_hit(n: str) -> bool:
        n = n.replace("\\", "/")
        return n == "data/sources.json" or n.endswith("/data/sources.json")

    for d in paths:
        if not d or not Path(d).exists():
            continue
        d = Path(d)
        if d.is_dir():
            runs = [p.parent.parent for p in d.rglob("sources.json") if p.parent.name == "data"]
            if runs:
                return min(runs, key=lambda r: (r.name != run_name, len(r.parts), str(r)))
            zips = sorted(d.rglob("*.zip"))
        else:
            zips = [d] if d.suffix.lower() == ".zip" else []
        for zp in zips:
            try:
                with zipfile.ZipFile(zp) as z:
                    hits = [n.replace("\\", "/") for n in z.namelist() if is_hit(n)]
                    if not hits:
                        continue
                    tmp = Path(tempfile.mkdtemp(prefix="lgmtp_resume_"))
                    _unzip_safe(z, tmp)
            except (zipfile.BadZipFile, OSError):
                continue
            best = min(hits, key=lambda n: (n.split("/")[-3:-2] != [run_name], n.count("/"), n))
            return tmp / Path(best).parent.parent
    return None


def week1_main(cfg: dict) -> dict:
    cfg = {**DEFAULT_W1, **cfg}
    root = Path(cfg["OUT_ROOT"]) / cfg["RUN_NAME"]
    for sub in ("data", "features", "heads", "preds", "results"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    log = Log(root / "results" / "log.txt")
    src = cfg["RESUME_FROM"]
    if src:
        where = cfg["PHASE1_SEARCH"] if src == "auto" else [src]
        run = locate_run(where, root.name)
        if run is None:
            if src != "auto":
                raise FileNotFoundError(f"RESUME_FROM={src!r}: no earlier run (data/sources.json, as a folder or "
                                        f"inside a .zip) found there")
            log(f"resume: RESUME_FROM='auto' found no earlier run (data/sources.json) in {where}; starting fresh")
        elif run.resolve() != root.resolve():
            _resume_copy(run, root, log)
    try:
        return _week1_run(cfg, root, log)
    except BaseException as e:
        import traceback
        log(f"FAILED: {type(e).__name__}: {e} ({_mem()})\n" + traceback.format_exc())
        try:
            _zip_tree(root / "results", Path(cfg["OUT_ROOT"]) / "week1_results.zip", root)
            log("partial results zipped to week1_results.zip; attach this version's output and rerun to resume")
        except Exception:  # noqa: BLE001
            pass
        raise


def _week1_run(cfg: dict, root: Path, log) -> dict:
    device = "cpu" if cfg["SMOKE"] else _device()
    log(f"week1 start | device {device} | build {cfg['GIT_HASH']}")
    (root / "results" / "config.json").write_text(json.dumps({k: v for k, v in cfg.items() if k != "SOURCES"},
                                                             indent=1, default=str))
    S = stage_sources(cfg, root, log)
    V = stage_vocab(cfg, root, S, log)
    M = stage_collect(cfg, root, S, log, device)
    if len(M) < min(cfg["MIN_MODELS"], len(cfg["MODELS"])):
        raise RuntimeError(f"only {len(M)} models collected (need {cfg['MIN_MODELS']}); see collection_summary.csv")
    L8 = stage_phase1(cfg, root, S, log, device) if cfg["RUN_L8"] else None
    F, R, T, mono = stage_features(cfg, root, M, L8, V, S, log)
    agree = stage_agree(cfg, root, M, L8, {k: T[k]["table"] for k in M}, log, device) if cfg["RUN_AGREE"] else []
    res = stage_heads(cfg, root, F, R, T, mono, V, L8, log, device)
    Vd = write_report(cfg, root, res, agree, log)
    info = run_info(cfg["GIT_HASH"], models={k: m["info"] for k, m in M.items()}, l8_source=(L8 or {}).get("source"),
                    verdict=Vd["decision"])
    (root / "results" / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    z1 = Path(cfg["OUT_ROOT"]) / "week1_results.zip"
    _zip_tree(root / "results", z1, root)
    z2 = Path(cfg["OUT_ROOT"]) / "week1_data.zip"
    _zip_tree([root / "data", root / "heads"], z2, root)
    log("=" * 70)
    log(f"DONE. Verdict: {Vd['decision']}")
    log(f"Download {z1.name} (results, ~small) and unzip into results/week1/ in the repo.")
    log(f"{z2.name} holds the collected outputs + top-20 and the trained heads (for later weeks); keep it on Kaggle "
        f"or download into data/week1/.")
    log("=" * 70)
    return dict(verdict=Vd, results_zip=str(z1), data_zip=str(z2), res=res)


def _zip_tree(srcs, zip_path: Path, base: Path) -> None:
    srcs = srcs if isinstance(srcs, list) else [srcs]
    tmp = zip_path.with_name(zip_path.name + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for s in srcs:
            for f in sorted(Path(s).rglob("*")):
                if f.is_file():
                    z.write(f, arcname=str(f.relative_to(base)))
    tmp.replace(zip_path)
