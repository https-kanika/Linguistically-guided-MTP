"""Target generation, teacher-forced draft/target passes and timing (Kaggle notebook 01).

Ported from the legacy Part A (``notebooks/pre_interim/day2-day3-llama8b-kaggle.ipynb`` cells 4-11).
Everything here also runs on CPU (``scripts/smoke_cpu.py``) with a small model in fp32.

Outputs (in ``out_dir``) for one split:
  target_outputs_{split}.jsonl   id, src, ref, prompt_ids, out_ids, out_text   (append + resume)
  _draft_tf_{split}.pkl          partial draft teacher-forced features         (resume)
  _target_tf_{split}.pkl         partial target teacher-forced features        (resume)
  traces_{split}.npz             final traces (``traces.save_traces``)
  timing.json                    t_draft, t_target, c, verify(g)               (skipped if present)
  run_info_{split}.json
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import platform
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .traces import TOPK, Trace, save_traces, summary
from .vocab import VocabMasks, build_masks

PROMPT_TEMPLATE = "Translate the following English sentence into Hindi. Output only the Hindi translation.\n\n{s}"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def _torch():
    import torch
    return torch


def sync_all() -> None:
    torch = _torch()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


def gpu_report(tag: str = "") -> None:
    torch = _torch()
    if torch.cuda.is_available():
        print(tag, " | ".join(f"GPU{i}: {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB allocated"
                              for i in range(torch.cuda.device_count())))


def from_pretrained(name: str, dtype, **kw):
    """``AutoModelForCausalLM.from_pretrained`` across transformers versions (dtype vs torch_dtype)."""
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw).eval()
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kw).eval()


def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_target(name: str, dtype, max_memory: dict | None = None, device_map="auto"):
    """Target model, layer-split over the available GPUs (``device_map='auto'``) or on CPU."""
    torch = _torch()
    if torch.cuda.is_available():
        m = from_pretrained(name, dtype, device_map=device_map, max_memory=max_memory,
                            low_cpu_mem_usage=True)
        print("modules per device:", dict(Counter(m.hf_device_map.values())))
    else:
        m = from_pretrained(name, dtype)
    return m


def load_draft(name: str, dtype, device: str):
    return from_pretrained(name, dtype).to(device)


def get_eos_ids(model, tok) -> list[int]:
    """Stop ids: generation-config eos plus ``<|eot_id|>`` (Llama-3 instruct end of turn)."""
    eos = model.generation_config.eos_token_id
    ids = set(eos if isinstance(eos, list) else [eos])
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tok.unk_token_id:
        ids.add(eot)
    return sorted(int(i) for i in ids if i is not None)


def check_draft_target_compat(draft, target, tok_d, tok_t, probe: str) -> None:
    """Draft and target must share a vocabulary (legacy asserts, kept)."""
    assert draft.config.vocab_size == target.config.vocab_size, "vocab sizes differ"
    probe = probe + " नमस्ते दुनिया. Hello world!"
    assert (tok_d(probe, add_special_tokens=False).input_ids
            == tok_t(probe, add_special_tokens=False).input_ids), "tokenizers disagree"
    print("draft/target tokenizers match")


# --------------------------------------------------------------------------- #
# data + prompts
# --------------------------------------------------------------------------- #

def load_flores_pairs(split: str, n: int | None = None):
    """FLORES+ eng_Latn -> hin_Deva rows ``(id, src, ref)`` for ``split`` in {dev, devtest}."""
    from datasets import load_dataset
    assert split in ("dev", "devtest"), split
    en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split=split)
    hi = load_dataset("openlanguagedata/flores_plus", "hin_Deva", split=split)
    hi_by_id = {r["id"]: r["text"] for r in hi}
    rows = [(int(r["id"]), r["text"], hi_by_id[r["id"]]) for r in en]
    return rows[:n] if n else rows


def build_prompt(tok, s: str) -> str:
    msgs = [{"role": "user", "content": PROMPT_TEMPLATE.format(s=s)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


# --------------------------------------------------------------------------- #
# 1. batched greedy target generation (resumable)
# --------------------------------------------------------------------------- #

def generate_targets(target, tok, rows, out_jsonl, batch: int, max_new: int, eos_ids,
                     device: str, progress: bool = True) -> list[dict]:
    """Greedy, batched (left padding) target outputs, appended to ``out_jsonl`` after every batch.

    Resumes if ``out_jsonl`` holds a prefix of ``rows`` (by id); otherwise starts over.
    """
    torch = _torch()
    from tqdm.auto import tqdm
    out_jsonl = Path(out_jsonl)
    ids = [r[0] for r in rows]
    prompts = [build_prompt(tok, r[1]) for r in rows]
    records = read_jsonl(out_jsonl)
    if [r["id"] for r in records] != ids[:len(records)]:
        print("existing output does not match this run; starting over")
        records = []
    print(f"resuming: {len(records)} / {len(rows)} sentences already done")
    eos_set = set(eos_ids)
    with open(out_jsonl, "a" if records else "w", encoding="utf-8") as f:
        rng = range(len(records), len(prompts), batch)
        for i in (tqdm(rng, desc="generate") if progress else rng):
            enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            with torch.no_grad():
                out = target.generate(**enc, do_sample=False, temperature=None, top_p=None,
                                      max_new_tokens=max_new, eos_token_id=list(eos_set),
                                      pad_token_id=tok.pad_token_id)
            gen = out[:, enc.input_ids.shape[1]:].tolist()
            for j, g in enumerate(gen):
                cut = next((p + 1 for p, t in enumerate(g) if t in eos_set), len(g))  # keep eos
                g = g[:cut]
                pid = enc.input_ids[j][enc.attention_mask[j].bool()].tolist()      # unpadded prompt
                rid, src, ref = rows[i + j]
                rec = dict(id=rid, src=src, ref=ref, prompt_ids=pid, out_ids=g,
                           out_text=tok.decode(g, skip_special_tokens=True).strip())
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    n_hit = sum(len(r["out_ids"]) >= max_new for r in records)
    print(f"{len(records)} target outputs in {out_jsonl.name}; {n_hit} hit MAX_NEW={max_new}")
    return records


def generate_unbatched(model, tok, prompt_ids: list[int], max_new: int, eos_ids, device: str) -> list[int]:
    """Plain greedy generation for one prompt, no padding (reference for losslessness, B9)."""
    torch = _torch()
    x = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                             temperature=None, top_p=None, max_new_tokens=max_new,
                             eos_token_id=list(eos_ids), pad_token_id=tok.pad_token_id)
    g = out[0, len(prompt_ids):].tolist()
    eos_set = set(eos_ids)
    cut = next((p + 1 for p, t in enumerate(g) if t in eos_set), len(g))
    return g[:cut]


# --------------------------------------------------------------------------- #
# 2./3. teacher-forced passes
# --------------------------------------------------------------------------- #

def _mask_matrix(masks: VocabMasks, vocab_size: int, device):
    """(V, 5) float32 matrix: columns space, punct, eos, mark, cont."""
    torch = _torch()
    cols = [masks.padded("starts_space", vocab_size), masks.padded("starts_punct", vocab_size),
            np.concatenate([masks.is_eos, np.zeros(vocab_size - masks.size, bool)]),
            masks.padded("starts_mark", vocab_size), masks.padded("starts_cont", vocab_size)]
    return torch.tensor(np.stack(cols, 1).astype(np.float32), device=device)


def _logits_for_output(model, prompt_ids, out_ids, device):
    """Logits (float32) predicting each output token: row t is q_t (conditioned on prompt + y[:t])."""
    torch = _torch()
    x = torch.tensor([list(prompt_ids) + list(out_ids)], device=device)
    with torch.no_grad():
        lg = model(x).logits[0, len(prompt_ids) - 1:-1]
    return lg.float()


def draft_features(draft, rec: dict, M, device: str, topk: int = TOPK) -> dict[str, np.ndarray]:
    """Per-position draft features for one record (all float32 on host)."""
    torch = _torch()
    lg = _logits_for_output(draft, rec["prompt_ids"], rec["out_ids"], device)
    lp = torch.log_softmax(lg, -1)
    p = lp.exp()
    H = -(p * lp).sum(-1)
    t2 = p.topk(2, -1).values
    tk = lp.topk(topk, -1)
    y = torch.tensor(rec["out_ids"], device=lg.device)
    P = p @ M.to(lg.device)                       # (n, 5): space, punct, eos, mark, cont
    f = dict(draft_argmax=lp.argmax(-1), H=H, top1=t2[:, 0], top2=t2[:, 1],
             p_target=p.gather(1, y[:, None]).squeeze(1),
             p_space=P[:, 0], p_punct=P[:, 1], p_eos=P[:, 2], p_newword=P[:, 0] + P[:, 1] + P[:, 2],
             p_mark=P[:, 3], p_cont=P[:, 4], draft_topk_ids=tk.indices, draft_topk_logp=tk.values)
    return {k: v.cpu().numpy() for k, v in f.items()}


def target_features(target, rec: dict, draft_argmax: np.ndarray, device: str,
                    topk: int = TOPK) -> dict[str, np.ndarray]:
    torch = _torch()
    lg = _logits_for_output(target, rec["prompt_ids"], rec["out_ids"], device)
    lp = torch.log_softmax(lg, -1)
    tk = lp.topk(topk, -1)
    da = torch.tensor(np.asarray(draft_argmax), device=lg.device, dtype=torch.long)
    f = dict(target_topk_ids=tk.indices, target_topk_logp=tk.values,
             target_logp_of_draft_argmax=lp.gather(1, da[:, None]).squeeze(1))
    return {k: v.cpu().numpy() for k, v in f.items()}


def _resumable_pass(records, fn, partial_path, desc: str, save_every: int = 50, progress=True):
    """Run ``fn(rec)`` for every record, pickling partial results so a dead session can resume."""
    from tqdm.auto import tqdm
    partial_path = Path(partial_path)
    done: dict = {}
    if partial_path.exists():
        with open(partial_path, "rb") as fh:
            done = pickle.load(fh)
        print(f"{desc}: resuming with {len(done)} / {len(records)} done")
    it = tqdm(records, desc=desc) if progress else records
    since = 0
    for rec in it:
        if rec["id"] in done:
            continue
        done[rec["id"]] = fn(rec)
        since += 1
        if since >= save_every:
            _atomic_pickle(done, partial_path)
            since = 0
    _atomic_pickle(done, partial_path)
    return done


def _atomic_pickle(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def run_draft_pass(draft, records, masks: VocabMasks, out_dir, split: str, device: str, progress=True):
    vocab = draft.get_output_embeddings().weight.shape[0]
    M = _mask_matrix(masks, vocab, device)
    return _resumable_pass(records, lambda r: draft_features(draft, r, M, device),
                           Path(out_dir) / f"_draft_tf_{split}.pkl", "draft teacher-forced", progress=progress)


def run_target_pass(target, records, draft_feats: dict, out_dir, split: str, device: str, progress=True):
    return _resumable_pass(records, lambda r: target_features(target, r, draft_feats[r["id"]]["draft_argmax"], device),
                           Path(out_dir) / f"_target_tf_{split}.pkl", "target teacher-forced", progress=progress)


def assemble_traces(records, draft_feats: dict, target_feats: dict) -> list[Trace]:
    out = []
    for r in records:
        d, t = draft_feats[r["id"]], target_feats[r["id"]]
        out.append(Trace(id=int(r["id"]), n=len(r["out_ids"]), target_ids=np.asarray(r["out_ids"], np.int32),
                         **d, **t))
    return out


def target_argmax_check(traces: list[Trace]) -> dict:
    """Rate at which the unpadded teacher-forced target argmax equals the generated y[t]."""
    agree, diffs, total = 0, [], 0
    for t in traces:
        am = t.target_topk_ids[:, 0]
        eq = am == t.target_ids
        agree += int(eq.sum())
        total += t.n
        for pos in np.flatnonzero(~eq):
            lp = t.target_topk_logp[pos]
            diffs.append(dict(id=t.id, pos=int(pos), y=int(t.target_ids[pos]), argmax=int(am[pos]),
                              margin=float(lp[0] - lp[1])))
    return dict(rate=agree / max(total, 1), positions=total, n_diff=len(diffs), diffs=diffs)


# --------------------------------------------------------------------------- #
# 4. timing
# --------------------------------------------------------------------------- #

def time_fwd(model, q: int = 1, ctx: int = 150, steps: int = 60, reps: int = 3, device: str = "cuda:0") -> float:
    """Median latency (s) of a forward with ``q`` new tokens on top of a KV cache of ``ctx`` tokens."""
    torch = _torch()
    ts = []
    with torch.no_grad():
        for _ in range(reps):
            x = torch.randint(1000, 20000, (1, ctx), device=device)
            o = model(x, use_cache=True)
            pkv = o.past_key_values
            for s in range(steps + 10):
                inp = torch.randint(1000, 20000, (1, q), device=device)
                sync_all()
                t = time.perf_counter()
                o = model(inp, past_key_values=pkv, use_cache=True)
                sync_all()
                dt = time.perf_counter() - t
                pkv = o.past_key_values
                if s >= 10:
                    ts.append(dt)
    return float(np.median(ts))


def run_timing(draft, target, out_path, device: str, setup: dict, steps: int = 60, reps: int = 3,
               max_g: int = 8) -> dict:
    """Legacy timing (c and verify(g), g=1..max_g) plus draft q=1 at ctx 150 and 400."""
    out_path = Path(out_path)
    if out_path.exists():
        print(f"{out_path.name} exists, timing skipped")
        return json.loads(out_path.read_text())
    t_d = time_fwd(draft, 1, steps=steps, reps=reps, device=device)
    t_t = time_fwd(target, 1, steps=steps, reps=reps, device=device)
    print(f"draft decode step: {t_d * 1000:.2f} ms | target decode step: {t_t * 1000:.2f} ms | c = {t_d / t_t:.3f}")
    ver = {g: time_fwd(target, g + 1, steps=steps, reps=reps, device=device) for g in range(1, max_g + 1)}
    print({g: f"{v * 1000:.1f} ms" for g, v in ver.items()})
    extra = {str(ctx): time_fwd(draft, 1, ctx=ctx, steps=steps, reps=reps, device=device) for ctx in (150, 400)}
    T = dict(t_draft=t_d, t_target=t_t, c=t_d / t_t, t_target_verify={str(g): v for g, v in ver.items()},
             t_draft_by_ctx=extra, setup=setup)
    out_path.write_text(json.dumps(T, indent=1))
    return T


# --------------------------------------------------------------------------- #
# run info + zip
# --------------------------------------------------------------------------- #

def run_info(git_hash: str, **extra) -> dict:
    torch = _torch()
    import transformers
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    return dict(git_hash=git_hash, torch=torch.__version__, transformers=transformers.__version__,
                python=platform.python_version(), gpus=gpus, time=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)


def zip_dir(src_dir, zip_path, include=None) -> str:
    """Zip files of ``src_dir`` (only names in ``include`` if given; flat, no folder) to ``zip_path``.
    Never deletes or moves anything in ``src_dir``. Returns the zip path."""
    import zipfile
    src_dir, zip_path = Path(src_dir), Path(zip_path)
    names = list(include) if include is not None else sorted(p.name for p in src_dir.iterdir() if p.is_file())
    missing = [n for n in names if not (src_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"cannot zip, missing in {src_dir}: {missing}")
    tmp = zip_path.with_name(zip_path.name + ".part")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(src_dir / n, arcname=n)
    os.replace(tmp, zip_path)
    return str(zip_path)


def free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# full pipeline for one split (used by notebook 01 and the smoke test)
# --------------------------------------------------------------------------- #

def run_split(*, split: str, out_dir, target, draft, tok, tok_d, n: int | None, batch: int,
              max_new: int, device: str, git_hash: str, timing_kw: dict | None = None,
              setup_extra: dict | None = None, rows=None, progress: bool = True,
              zip_name: str | None = None) -> dict:
    """Trace collection for one split. Returns a dict with paths and summaries.

    timing_kw: passed to ``run_timing`` (steps, reps, max_g); setup_extra: added to timing.json setup
    (e.g. max_memory, GPU names).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = rows if rows is not None else load_flores_pairs(split, n)
    eos_ids = get_eos_ids(target, tok)
    print(f"{split}: {len(rows)} sentences; eos ids {eos_ids}")

    records = generate_targets(target, tok, rows, out_dir / f"target_outputs_{split}.jsonl",
                               batch, max_new, eos_ids, device, progress=progress)
    check_draft_target_compat(draft, target, tok_d, tok, records[0]["out_text"])

    masks = build_masks(tok, eos_ids)
    dfe = run_draft_pass(draft, records, masks, out_dir, split, device, progress=progress)
    tfe = run_target_pass(target, records, dfe, out_dir, split, device, progress=progress)
    traces = assemble_traces(records, dfe, tfe)
    chk = target_argmax_check(traces)
    print(f"target teacher-forced argmax == y[t]: {chk['rate']:.4f} ({chk['n_diff']} positions differ)")
    for d in chk["diffs"][:20]:
        print("   ", d)
    if chk["rate"] < 0.995:
        print("WARNING: target argmax agreement below 0.995")

    import transformers
    meta = dict(split=split, target=getattr(target.config, "_name_or_path", ""),
                draft=getattr(draft.config, "_name_or_path", ""), max_new=max_new, batch=batch,
                eos_ids=eos_ids, git_hash=git_hash, transformers=transformers.__version__,
                target_argmax_check={k: v for k, v in chk.items() if k != "diffs"},
                target_argmax_diffs=chk["diffs"][:500])
    npz = out_dir / f"traces_{split}.npz"
    save_traces(npz, traces, meta)
    S = summary(traces)
    print(f"positions: {S['positions']} | agreement: {S['agreement']:.3f} | mean H: {S['mean_H']:.3f} "
          f"(match {S['mean_H_match']:.3f}, mismatch {S['mean_H_mismatch']:.3f})")

    setup = dict(target=meta["target"], draft=meta["draft"], dtype=str(next(target.parameters()).dtype),
                 device=device, **(setup_extra or {}))
    T = run_timing(draft, target, out_dir / "timing.json", device, setup, **(timing_kw or {}))

    info = run_info(git_hash, split=split, n_sentences=len(rows), summary=S,
                    target=meta["target"], draft=meta["draft"])
    (out_dir / f"run_info_{split}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    zip_name = zip_name or f"traces_v2_{split}"
    zp = zip_dir(out_dir, out_dir.parent / f"{zip_name}.zip",
                 include=[npz.name, f"target_outputs_{split}.jsonl", "timing.json", f"run_info_{split}.json"])
    return dict(zip=zp, npz=str(npz), summary=S, timing=T, target_check=chk)


# --------------------------------------------------------------------------- #
# notebook 01 entry point (also called by scripts/smoke_cpu.py)
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG_01 = dict(
    SPLIT="dev",                                    # "dev" or "devtest": one split per run
    N=None,                                         # None = all sentences of the split
    BATCH=16, MAX_NEW=256,
    SMOKE=False,                                    # True: CPU, fp32, one model as target+draft
    TARGET="meta-llama/Meta-Llama-3-8B-Instruct",
    DRAFT="meta-llama/Llama-3.2-1B-Instruct",
    MAX_MEMORY={0: "8GiB", 1: "10GiB"},
    OUT_ROOT="/kaggle/working",
    GIT_HASH="unknown",
    TIMING=dict(steps=60, reps=3, max_g=8),
    RESUME_FROM=None,   # folder with partial outputs of an earlier (dead) run, e.g. /kaggle/input/<...>
)


def _prepare_out_dir(out_dir: Path, out_root: Path, resume_from) -> None:
    """Copy partial outputs from ``resume_from`` (searched recursively) and reuse a timing.json
    measured by another split's run in this session."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if resume_from:
        src = Path(resume_from)
        for f in src.rglob("*"):
            if f.is_file() and f.parent.name == out_dir.name and not (out_dir / f.name).exists():
                shutil.copy2(f, out_dir / f.name)
                print("resume: copied", f)
    if not (out_dir / "timing.json").exists():
        for other in sorted(out_root.glob("traces_v2_*/timing.json")):
            shutil.copy2(other, out_dir / "timing.json")
            print("reusing timing from", other)
            break


def notebook01_main(cfg: dict) -> dict:
    """Run trace collection for ``cfg['SPLIT']``. Writes ``{OUT_ROOT}/traces_v2_{split}/`` and
    ``{OUT_ROOT}/traces_v2_{split}.zip``. With ``SMOKE=True`` the draft model is used as both
    target and draft (fp32, CPU, loaded once) and only 3 sentences are processed."""
    torch = _torch()
    cfg = {**DEFAULT_CONFIG_01, **cfg}
    split = cfg["SPLIT"]
    out_dir = Path(cfg["OUT_ROOT"]) / f"traces_v2_{split}"
    _prepare_out_dir(out_dir, Path(cfg["OUT_ROOT"]), cfg["RESUME_FROM"])
    if cfg["SMOKE"]:
        device, dtype = "cpu", torch.float32
        tok = load_tokenizer(cfg["DRAFT"])
        draft = target = load_draft(cfg["DRAFT"], dtype, device)
        tok_d = tok
        n = cfg["N"] or 3
        setup_extra = dict(smoke=True)
    else:
        assert torch.cuda.device_count() >= 2, "Need 2 GPUs: Settings > Accelerator > GPU T4 x2"
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU{i}: {p.name}, {p.total_memory / 2**30:.1f} GiB")
        device, dtype = "cuda:0", torch.float16
        tok = load_tokenizer(cfg["TARGET"])
        tok_d = load_tokenizer(cfg["DRAFT"])
        target = load_target(cfg["TARGET"], dtype, max_memory=cfg["MAX_MEMORY"])
        gpu_report("after loading target:")
        draft = load_draft(cfg["DRAFT"], dtype, device)
        gpu_report("after loading draft:")
        n = cfg["N"]
        setup_extra = dict(max_memory={str(k): v for k, v in cfg["MAX_MEMORY"].items()},
                           gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    res = run_split(split=split, out_dir=out_dir, target=target, draft=draft, tok=tok, tok_d=tok_d,
                    n=n, batch=cfg["BATCH"], max_new=cfg["MAX_NEW"], device=device,
                    git_hash=cfg["GIT_HASH"], timing_kw=cfg["TIMING"], setup_extra=setup_extra)
    print("\n" + "=" * 70)
    print(f"DONE. Download  {Path(res['zip']).name}  from the notebook's Output tab")
    print(f"and unzip it into  data/traces_v2/  in the repo (files land directly in that folder).")
    print("=" * 70)
    return res
