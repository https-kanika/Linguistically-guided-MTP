"""Week-2 data for the token-tax-refund simulation: the draft model's (Llama-3.2-1B) teacher-forced
predictions along the target's (Llama-3-8B) greedy outputs, and draft-model call latency for q new tokens.

Input: an earlier week-1 run (attached notebook output, as a folder or a zip) with
``data/llama3-8b/{split}.jsonl`` and ``data/llama3-8b/timing.json``.

Output: ``/kaggle/working/drafter_traces.zip`` containing, per split,
  ``{split}.npz``   rec_ids, offsets, target_ids (the 8B greedy tokens, flat), d_topk_ids [n,K] int32,
                    d_topk_logp [n,K] float16, d_H [n] float32 (entropy of the full draft distribution, nats);
                    row t of a record is the draft distribution over target token t given prompt + target[:t]
  ``{split}.jsonl`` the 8B records (copied, so the zip is self-contained)
plus ``timing.json`` (t_target, t_draft[ctx][q], c[ctx][q] = t_draft / t_target) and ``summary.json``.
Resumable: finished splits and the timing are reused when the output folder already holds them.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import collect as C
from .gpu_traces import read_jsonl, time_fwd
from .week1 import Log, _zip_tree, locate_run

DEFAULT_DT = dict(
    DRAFT=["meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"],
    TARGET_TOKENIZER=["meta-llama/Meta-Llama-3-8B-Instruct", "NousResearch/Meta-Llama-3-8B-Instruct"],
    RUN_NAME="week1", TARGET_KEY="llama3-8b", SEARCH=["/kaggle/input"], OUT_ROOT="/kaggle/working",
    SPLITS=["dev", "devtest"], K=20, TIME_Q=[1, 2, 3, 4, 5, 6, 8, 12, 16], TIME_CTX=[150, 400],
    TIME_STEPS=60, TIME_REPS=3, T_TARGET_FALLBACK=0.0706,
)


def _flatten(recs, res):
    lens = np.array([len(r["out_ids"]) for r in recs], np.int64)
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    tgt = np.concatenate([np.asarray(r["out_ids"], np.int64) for r in recs]) if len(recs) else np.zeros(0, np.int64)
    ids = np.concatenate([x[0] for x in res], 0)
    lp = np.concatenate([x[1] for x in res], 0)
    H = np.concatenate([x[2] for x in res], 0)
    return dict(rec_ids=np.array([int(r["id"]) for r in recs], np.int64), offsets=offs, target_ids=tgt.astype(np.int32),
                d_topk_ids=ids, d_topk_logp=lp, d_H=H)


def _check_tokenizers(cfg, tok, recs, log):
    """The draft must share the target's vocabulary (Llama-3 family): compare decodings of real outputs."""
    tt, name = None, None
    for n in cfg["TARGET_TOKENIZER"]:
        try:
            from transformers import AutoTokenizer
            tt, name = AutoTokenizer.from_pretrained(n), n
            break
        except Exception as e:  # noqa: BLE001
            log(f"  target tokenizer {n} not loadable: {type(e).__name__}")
    if tt is None:
        log("  target tokenizer not loadable; vocabulary check skipped")
        return
    for r in recs[:20]:
        a = tok.decode(r["out_ids"], skip_special_tokens=False)
        b = tt.decode(r["out_ids"], skip_special_tokens=False)
        if a != b:
            raise ValueError(f"draft and target tokenizers decode record {r['id']} differently: not the same vocabulary")
    log(f"  vocabulary check ok against {name}")


def drafter_main(cfg: dict) -> dict:
    torch = C._torch()
    cfg = {**DEFAULT_DT, **cfg}
    out = Path(cfg["OUT_ROOT"]) / "drafter_traces"
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "log.txt")
    run = locate_run(cfg["SEARCH"], cfg["RUN_NAME"])
    if run is None:
        raise FileNotFoundError("no week-1 run found in the inputs: attach the mtp-final-boss output (Add Input -> Your Work)")
    src = run / "data" / cfg["TARGET_KEY"]
    missing = [s for s in cfg["SPLITS"] if not (src / f"{s}.jsonl").exists()]
    if missing:
        raise FileNotFoundError(f"{src} has no {missing} outputs; the week-1 run must have collected the 8B outputs")
    log(f"target outputs: {src}")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tok, name = C.load_model(dict(key="draft", names=cfg["DRAFT"], dtype="fp16"), device)
    model.eval()
    summ = dict(draft=name, target_dir=str(src), device=device, splits={})
    for split in cfg["SPLITS"]:
        recs = read_jsonl(src / f"{split}.jsonl")
        shutil.copy2(src / f"{split}.jsonl", out / f"{split}.jsonl")
        p = out / f"{split}.npz"
        if p.exists():
            log(f"{split}: reused {p.name}")
        else:
            if split == cfg["SPLITS"][0]:
                _check_tokenizers(cfg, tok, recs, log)
            vmax = max((max(r["out_ids"]) for r in recs if r["out_ids"]), default=0)
            if vmax >= model.get_output_embeddings().weight.shape[0]:
                raise ValueError(f"target token id {vmax} outside the draft vocabulary")
            t0 = time.time()
            res = C.teacher_forced_topk(model, recs, device, k=cfg["K"], log=log, with_entropy=True)
            d = _flatten(recs, res)
            tmp = out / f"{split}.part.npz"
            np.savez(tmp, **d)
            tmp.replace(p)
            log(f"{split}: {len(recs)} outputs, {len(d['target_ids'])} positions ({time.time() - t0:.0f}s)")
        with np.load(p) as z:
            agree = float((z["d_topk_ids"][:, 0] == z["target_ids"]).mean()) if len(z["target_ids"]) else float("nan")
            n = int(len(z["target_ids"]))
            Hm = float(z["d_H"].mean()) if n else float("nan")
        summ["splits"][split] = dict(outputs=len(recs), positions=n, draft_target_agreement=agree, mean_entropy=Hm)
        log(f"{split}: draft argmax == target token at {agree:.3f} of positions (Phase 1: 0.761); mean entropy {Hm:.3f}")
    tp = out / "timing.json"
    if tp.exists():
        log("timing: reused")
    elif device == "cpu" and not cfg.get("SMOKE"):
        log("timing: skipped on CPU")
    else:
        tsrc = src / "timing.json"
        t_target = json.loads(tsrc.read_text())["t_target"] if tsrc.exists() else cfg["T_TARGET_FALLBACK"]
        t_d = {str(ctx): {str(q): time_fwd(model, q, ctx=ctx, steps=cfg["TIME_STEPS"], reps=cfg["TIME_REPS"], device=device)
                          for q in cfg["TIME_Q"]} for ctx in cfg["TIME_CTX"]}
        c = {ctx: {q: v / t_target for q, v in row.items()} for ctx, row in t_d.items()}
        tp.write_text(json.dumps(dict(t_target=t_target, t_target_source=str(tsrc) if tsrc.exists() else "fallback",
                                      t_draft=t_d, c=c, gpu=torch.cuda.get_device_name(0) if device != "cpu" else "cpu",
                                      note="c[ctx][q] = draft forward with q new tokens on a KV cache of ctx tokens / 8B step"),
                                 indent=1))
        ctx0 = str(cfg["TIME_CTX"][0])
        log("timing: c(q) at ctx " + ctx0 + ": " + ", ".join(f"q={q}: {v:.3f}" for q, v in c[ctx0].items())
            + " (Phase 1: c = 0.323 for q=1)")
    if tp.exists():
        summ["timing"] = json.loads(tp.read_text())
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    _zip_tree(out, Path(cfg["OUT_ROOT"]) / "drafter_traces.zip", out)
    log("wrote drafter_traces.zip: download it and unzip into E:/MTP/drafter_traces")
    return summ
