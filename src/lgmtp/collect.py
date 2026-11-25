"""Data collection for W1: several LLMs' greedy Hindi outputs with their top-K distributions (GPU).

Per (model, split) it writes ``{split}.jsonl`` (id, src, prompt_text, prompt_ids, out_ids, out_text,
eos_end) and ``{split}_topk.npz`` (flat top-K ids / log-probs with offsets), both resumable. The top-K
comes from a separate teacher-forced pass over prompt + output (unpadded semantics via attention mask
and explicit position ids), like the Phase-1 traces, so small-model data and the Phase-1 8B traces are
built the same way.
"""
from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np

from .gpu_traces import PROMPT_TEMPLATE, from_pretrained, read_jsonl

BASE_DEMO = ("English: The weather is very pleasant today.\nHindi: आज मौसम बहुत सुहावना है।\n\n")

DEFAULT_MODELS = [
    dict(key="qwen2.5-1.5b", names=["Qwen/Qwen2.5-1.5B-Instruct"], dtype="fp16", family="qwen"),
    dict(key="gemma3-1b", names=["google/gemma-3-1b-it", "unsloth/gemma-3-1b-it"], dtype="fp32", family="gemma"),
    dict(key="llama3.2-1b", names=["meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"],
         dtype="fp16", family="llama3"),
    dict(key="sarvam1", names=["sarvamai/sarvam-1"], dtype="fp16", family="sarvam", prompt_mode="base"),
]
OPTIONAL_MODELS = [
    dict(key="qwen3-1.7b", names=["Qwen/Qwen3-1.7B"], dtype="fp16", family="qwen",
         chat_kwargs=dict(enable_thinking=False)),
]


def _torch():
    import torch
    return torch


def head_prompt_text(src: str) -> str:
    """The prompt text the universal head reads: the same for every model (model-independent)."""
    return PROMPT_TEMPLATE.format(s=src)


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #

def load_flores_english(split: str, n: int | None = None) -> list[tuple[int, str]]:
    """(id, English sentence) of FLORES+ ``split``. Falls back to the ungated mteb/flores parquet (whose
    values carry one extra leading quote, removed here); its Hindi column is corrupted and is not used."""
    from datasets import load_dataset
    try:
        en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split=split)
        rows = [(int(r["id"]), r["text"]) for r in en]
        src = "openlanguagedata/flores_plus"
    except Exception as e:  # noqa: BLE001
        print(f"flores_plus unavailable ({type(e).__name__}: {e}); using mteb/flores English column")
        ds = load_dataset("mteb/flores", split=split)
        rows = []
        for i, r in enumerate(ds):
            t = r["eng_Latn"]
            if t.startswith('"') and t.count('"') % 2 == 1:
                t = t[1:]
            rows.append((i, t))
        src = "mteb/flores"
    print(f"FLORES {split}: {len(rows)} sentences from {src}")
    return rows[:n] if n else rows


def load_iitb(n_en: int, n_hi: int, seed: int = 0, min_words: int = 8, max_words: int = 40):
    """Extra English sources (for generation) and Hindi sentences (unit vocabulary + text-only
    pretraining) from the IITB English-Hindi corpus, disjoint sentence pairs, length-filtered."""
    from datasets import load_dataset
    ds = load_dataset("cfilt/iitb-english-hindi", split="train")
    tr = ds["translation"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tr))
    en, hi = [], []
    for i in order:
        e, h = tr[i]["en"], tr[i]["hi"]
        ne, nh = len(e.split()), len(h.split())
        if not (min_words <= ne <= max_words and min_words <= nh <= max_words * 2):
            continue
        if len(en) < n_en and sum(ch.isalpha() for ch in e) > 0.6 * len(e):
            en.append((1_000_000 + int(i), e))
        elif len(hi) < n_hi and sum("ऀ" <= ch <= "ॿ" for ch in h) > 0.5 * len(h.replace(" ", "")):
            hi.append(h)
        if len(en) >= n_en and len(hi) >= n_hi:
            break
    print(f"IITB: {len(en)} English sources, {len(hi)} Hindi sentences")
    return en, hi


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #

def load_model(spec: dict, device: str = "cuda:0", max_memory=None):
    """Load the first loadable name of ``spec['names']``. Returns (model, tok, name) or raises."""
    torch = _torch()
    from transformers import AutoTokenizer
    dt = dict(fp16=torch.float16, fp32=torch.float32, bf16=torch.bfloat16)[spec.get("dtype", "fp16")]
    if device == "cpu":
        dt = torch.float32
    last = None
    for name in spec["names"]:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            if spec.get("device_map") == "auto" and device != "cpu":
                m = from_pretrained(name, dt, device_map="auto", max_memory=max_memory, low_cpu_mem_usage=True)
            else:
                m = from_pretrained(name, dt).to(device)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            print(f"loaded {name} ({spec.get('dtype', 'fp16')} on {device})")
            return m, tok, name
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  could not load {name}: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError(f"no loadable model for {spec['key']}: {last}")


def model_eos_ids(model, tok, spec: dict) -> list[int]:
    ids = set()
    eos = model.generation_config.eos_token_id
    for e in (eos if isinstance(eos, (list, tuple)) else [eos]):
        if e is not None:
            ids.add(int(e))
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for t in ("<|eot_id|>", "<end_of_turn>", "<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id and i >= 0:
            ids.add(i)
    return sorted(ids)


def build_model_prompt(tok, spec: dict, src: str) -> str:
    if spec.get("prompt_mode") != "base" and getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": PROMPT_TEMPLATE.format(s=src)}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           **spec.get("chat_kwargs", {}))
        except Exception as e:  # noqa: BLE001
            print(f"chat template failed ({e}); using base prompt")
    bos = tok.bos_token or ""
    return f"{bos}Translate the following English sentence into Hindi.\n\n{BASE_DEMO}English: {src}\nHindi:"


def _base_cut(ids: list, tok_bytes, eos_id: int) -> tuple[list, bool]:
    """Base-model outputs: stop at the first newline after some text; the newline becomes the model's eos."""
    acc = b""
    for j, t in enumerate(ids):
        b = tok_bytes[j]
        if b"\n" in b and acc.strip():
            return ids[:j] + [eos_id], True
        acc += b
    return ids, False


# --------------------------------------------------------------------------- #
# generation + teacher-forced top-K
# --------------------------------------------------------------------------- #

def generate(model, tok, spec: dict, rows: list, out_jsonl, batch: int, max_new: int, device: str,
             eos_ids: list[int], log=print) -> list[dict]:
    """Greedy batched generation (left padding), appended to ``out_jsonl`` after every batch, resumable."""
    torch = _torch()
    from .boundaries import token_to_bytes
    out_jsonl = Path(out_jsonl)
    done = {r["id"]: r for r in read_jsonl(out_jsonl)}
    todo = [r for r in rows if r[0] not in done]
    log(f"  generate: {len(done)} done, {len(todo)} to go")
    tok.padding_side = "left"
    order = sorted(range(len(todo)), key=lambda i: len(todo[i][1]))
    base = spec.get("prompt_mode") == "base" or not getattr(tok, "chat_template", None)
    eos_set = set(eos_ids)
    t0 = time.time()
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for bi in range(0, len(order), batch):
            chunk = [todo[i] for i in order[bi:bi + batch]]
            prompts = [build_model_prompt(tok, spec, s) for _, s in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                out = model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None,
                                     repetition_penalty=1.0, max_new_tokens=max_new,
                                     eos_token_id=sorted(eos_set), pad_token_id=tok.pad_token_id)
            gen = out[:, enc.input_ids.shape[1]:].tolist()
            for j, g in enumerate(gen):
                cut = next((p + 1 for p, t in enumerate(g) if t in eos_set), len(g))
                g = g[:cut]
                eos_end = bool(g) and g[-1] in eos_set
                if base:
                    core = g[:-1] if eos_end else g
                    tb = [token_to_bytes(t) for t in tok.convert_ids_to_tokens(core)]
                    cut_ids, was_cut = _base_cut(core, tb, sorted(eos_set)[0])
                    if was_cut:
                        g, eos_end = cut_ids, True
                pid = enc.input_ids[j][enc.attention_mask[j].bool()].tolist()
                rid, src = chunk[j]
                rec = dict(id=int(rid), src=src, prompt_text=head_prompt_text(src), prompt_ids=pid, out_ids=g,
                           out_text=tok.decode(g, skip_special_tokens=True).strip(), eos_end=eos_end)
                done[rid] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if bi == 0:
                el = time.time() - t0
                log(f"  first batch {el:.1f}s -> ESTIMATE ~{el * len(order) / max(len(chunk), 1) / 60:.1f} min for this split")
    recs = [done[r[0]] for r in rows if r[0] in done]
    nhit = sum(len(r["out_ids"]) >= max_new for r in recs)
    log(f"  {len(recs)} outputs; {nhit} hit MAX_NEW; mean len {np.mean([len(r['out_ids']) for r in recs]):.1f} tokens")
    return recs


def _logits_budget_rows(V: int, L: int, bytes_per: int, budget: float = 2.5e9) -> int:
    """Rows per teacher-forced batch so that the kept logits (model dtype + a float32 copy) fit."""
    return max(1, int(budget // max(V * L * (bytes_per + 4), 1)))


def teacher_forced_topk(model, recs: list[dict], device: str, k: int = 20, log=print,
                        ids_key: str = "out_ids", argmax_only: bool = False, with_entropy: bool = False):
    """Top-k log-probs of q_t for every output position t of every record (prompt_ids + out ids),
    batched with left padding, explicit position ids and only the output rows of the logits.
    Returns list of (ids [n,k] int32, logp [n,k] float16), or with ``with_entropy`` list of
    (ids, logp, H [n] float32, the full-distribution entropy in nats), or list of argmax arrays."""
    torch = _torch()
    V = model.get_output_embeddings().weight.shape[0]
    bpe = 4 if next(model.parameters()).dtype == torch.float32 else 2
    order = sorted(range(len(recs)), key=lambda i: len(recs[i]["prompt_ids"]) + len(recs[i][ids_key]))
    res = [None] * len(recs)
    i = 0
    while i < len(order):
        L_peek = max(len(recs[c][ids_key]) for c in order[i:i + 32]) + 1
        rows = _logits_budget_rows(V, L_peek, bpe)
        chunk = order[i:i + min(rows, 32)]
        L_out = max(len(recs[c][ids_key]) for c in chunk)
        seqs = [list(recs[c]["prompt_ids"]) + list(recs[c][ids_key]) for c in chunk]
        Lmax = max(len(s) for s in seqs)
        x = torch.full((len(seqs), Lmax), 0, dtype=torch.long)
        am = torch.zeros((len(seqs), Lmax), dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, Lmax - len(s):] = torch.tensor(s)
            am[j, Lmax - len(s):] = 1
        pos = (am.cumsum(1) - 1).clamp(min=0)
        x, am, pos = x.to(device), am.to(device), pos.to(device)
        keep = L_out + 1
        with torch.no_grad():
            try:
                lg = model(input_ids=x, attention_mask=am, position_ids=pos, logits_to_keep=keep).logits
            except TypeError:
                lg = model(input_ids=x, attention_mask=am, position_ids=pos).logits[:, -keep:]
            lg = lg.float()
            for j, c in enumerate(chunk):
                n = len(recs[c][ids_key])
                rowlg = lg[j, keep - 1 - n: keep - 1]           # q_t for t = 0..n-1
                if argmax_only:
                    res[c] = rowlg.argmax(-1).cpu().numpy().astype(np.int32)
                else:
                    lp = torch.log_softmax(rowlg, -1)
                    tk = lp.topk(k, -1)
                    res[c] = (tk.indices.cpu().numpy().astype(np.int32), tk.values.cpu().numpy().astype(np.float16))
                    if with_entropy:
                        H = -(lp.exp() * lp).sum(-1)
                        res[c] = res[c] + (H.cpu().numpy().astype(np.float32),)
        del lg
        i += len(chunk)
    return res


def save_topk(path, recs: list[dict], tk: list) -> None:
    offs = np.zeros(len(recs) + 1, np.int64)
    offs[1:] = np.cumsum([len(r["out_ids"]) for r in recs])
    ids = np.concatenate([a for a, _ in tk]) if tk else np.zeros((0, 20), np.int32)
    lp = np.concatenate([b for _, b in tk]) if tk else np.zeros((0, 20), np.float16)
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, rec_ids=np.array([r["id"] for r in recs], np.int64), offsets=offs, topk_ids=ids, topk_logp=lp)
    os.replace(tmp, path)


def load_records(jsonl, topk_npz) -> list[dict]:
    recs = read_jsonl(jsonl)
    # Read every array exactly once: each NpzFile[key] access decompresses the whole array again, and
    # a per-record slice of it keeps that full copy alive (records x array size: the week-1 Kaggle OOM).
    with np.load(topk_npz) as z:
        rec_ids, offs = z["rec_ids"], z["offsets"]
        ids_all, lp_all = z["topk_ids"], z["topk_logp"]
    by = {int(r): i for i, r in enumerate(rec_ids)}
    out = []
    for r in recs:
        i = by.get(int(r["id"]))
        if i is None:
            continue
        a, b = int(offs[i]), int(offs[i + 1])
        if b - a != len(r["out_ids"]):
            continue
        r["topk_ids"], r["topk_logp"] = ids_all[a:b], lp_all[a:b].astype(np.float32)
        out.append(r)
    return out


def collect_model(spec: dict, splits: dict, out_dir, batch: int, max_new: int, device: str, k: int = 20,
                  log=print, max_memory=None, keep_loaded: bool = False):
    """Generate + teacher-forced top-K for every split in ``splits`` ({name: rows}). Skips finished
    splits. Returns dict(name -> records) plus model info; the model is freed unless keep_loaded."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info_p = out_dir / "model_info.json"
    need = [s for s in splits if not (out_dir / f"{s}_topk.npz").exists()]
    model = tok = None
    if need:
        model, tok, name = load_model(spec, device, max_memory)
        eos_ids = model_eos_ids(model, tok, spec)
        info = dict(key=spec["key"], name=name, eos_ids=eos_ids, vocab_size=int(model.get_output_embeddings().weight.shape[0]),
                    len_tok=len(tok), dtype=spec.get("dtype", "fp16"), prompt_mode=spec.get("prompt_mode", "chat"),
                    family=spec.get("family", ""))
        info_p.write_text(json.dumps(info, indent=1))
        for s in need:
            t0 = time.time()
            log(f"[{spec['key']}] split {s}: {len(splits[s])} sources")
            recs = generate(model, tok, spec, splits[s], out_dir / f"{s}.jsonl", batch, max_new, device, eos_ids, log)
            tk = teacher_forced_topk(model, recs, device, k, log)
            chk = np.mean([float((a[:, 0] == np.asarray(r["out_ids"])).mean()) for r, (a, _) in zip(recs, tk) if len(r["out_ids"])])
            log(f"  teacher-forced argmax == generated token: {chk:.4f}  ({time.time() - t0:.0f}s)")
            save_topk(out_dir / f"{s}_topk.npz", recs, tk)
            info.setdefault("tf_argmax_agreement", {})[s] = float(chk)
            info_p.write_text(json.dumps(info, indent=1))
    info = json.loads(info_p.read_text())
    res = {s: load_records(out_dir / f"{s}.jsonl", out_dir / f"{s}_topk.npz") for s in splits}
    if keep_loaded:
        return res, info, (model, tok)
    del model, tok
    free_model()
    return res, info, None


def free_model(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
