"""Word-completion heads on the draft model (PLAN_wordheads.md).

Position convention (same as the Phase-1 traces): for an output ``y[0..n-1]`` after prompt ``P``,
``h_t`` is the draft's last (post-norm) hidden state at input index ``len(P) + t - 1``; the frozen LM
head maps it to the draft distribution over ``y[t]`` (row t of a trace).

* head k (k = 1..K) on ``h_t`` predicts ``y[t+k]``: Medusa-1 residual block ``z = h + SiLU(W_k h)``
  (``W_k`` zero-initialised) followed by the frozen, shared LM head;
* the continuation head ``Linear(d, K)`` predicts ``c_k[t] = 1`` iff ``y[t+1..t+k]`` all continue the
  word of ``y[t]`` (none of them is word-initial).

Variants trained side by side on the same draft forward passes (same data, same order):
  v0  plain MTP:     sum_k w_k CE_k over all positions
  v1  word-guided:   CE_k only where c_k = 1, plus LAMBDA_CONT * BCE(continuation head)
  v2  adaptive:      v1's K+1 losses with uncertainty weighting  sum_i exp(-s_i) L_i + s_i
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np

VARIANTS = ("v0", "v1", "v2")

DEFAULT_WH = dict(
    DRAFT=["meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"],
    K=3, HEAD_W=0.8, LAMBDA_CONT=0.5,
    TRAIN_DATASET="cfilt/iitb-english-hindi", N_TRAIN=60000, N_VAL=500, SEED=0,
    MIN_HI_CHARS=15, MAX_HI_CHARS=400, MIN_WORDS=6, MAX_LEN=320,
    BATCH=32, LR=1e-3, WD=0.0, WARMUP=100, POS_PER_HEAD=768, EVAL_EVERY=300, MAX_STEPS=None,
    DUMP_TOPK=8, TIME_Q=[1, 2, 3, 4, 5, 6, 8], TIME_CTX=150, TIME_STEPS=40, TIME_REPS=3,
    OUT_ROOT="/kaggle/working", SEARCH=["/kaggle/input"], SMOKE=False,
)


def _torch():
    import torch
    return torch


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #

def word_initial_flags(tok, ids) -> np.ndarray:
    """Gold 'token starts a new word' flags of an output sequence (same labeller as Phase 1)."""
    from .boundaries import token_labels
    from .vocab import special_ids, token_bytes_list
    spec = special_ids(tok)
    tb = token_bytes_list(tok, ids)
    return np.asarray(token_labels(tb, special=[int(i) in spec for i in ids])["word_initial"], bool)


def continuation_targets(word_initial: np.ndarray, K: int) -> np.ndarray:
    """c[t, k-1] = 1 iff y[t+1..t+k] exist and none is word-initial (all continue y[t]'s word)."""
    n = len(word_initial)
    c = np.zeros((n, K), bool)
    for t in range(n):
        ok = True
        for k in range(1, K + 1):
            ok = ok and (t + k < n) and not word_initial[t + k]
            c[t, k - 1] = ok
    return c


def continuation_vocab(tok) -> np.ndarray:
    """Ids of tokens that can only continue a Devanagari word: non-empty, no leading space, and every
    byte decodes to Devanagari (U+0900-U+097F), ZWJ/ZWNJ, or is part of an incomplete UTF-8 sequence
    of such a character. Inside a word these are the only possible next tokens, so a word-completion
    head can score this small vocabulary instead of the full one."""
    from .vocab import special_ids, token_bytes_list
    spec = special_ids(tok)
    n = len(tok)
    tb = token_bytes_list(tok, list(range(n)))
    keep = []
    for i, b in enumerate(tb):
        if i in spec or not b or b[:1] in (b" ", b"\n", b"\t"):
            continue
        txt = b.decode("utf-8", errors="ignore")
        if not txt and all(x >= 0x80 for x in b):        # pure UTF-8 fragment (part of a character)
            keep.append(i)
            continue
        if txt and all("ऀ" <= ch <= "ॿ" or ch in "‌‍" for ch in txt):
            keep.append(i)
    return np.asarray(keep, np.int64)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

def make_heads(d: int, K: int):
    torch = _torch()
    nn = torch.nn

    class WordHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.K = K
            self.blocks = nn.ModuleList([nn.Linear(d, d) for _ in range(K)])
            for b in self.blocks:
                nn.init.zeros_(b.weight)
                nn.init.zeros_(b.bias)
            self.cont = nn.Linear(d, K)
            self.log_var = nn.Parameter(torch.zeros(K + 1))   # used by v2 only

        def z(self, h, k: int):                     # k = 1..K
            return h + torch.nn.functional.silu(self.blocks[k - 1](h))

    return WordHeads()


def lm_logits(z, W):
    """Logits of the frozen LM head (weight W, [V, d], half precision) for hidden rows z (float32)."""
    return (z.to(W.dtype) @ W.t()).float()


def get_backbone_and_lm_weight(model):
    base = getattr(model, "model", None) or model.get_decoder()
    return base, model.get_output_embeddings().weight


def hidden_states(model, input_ids, attention_mask):
    """Last post-norm hidden states [B, T, d] (what the LM head reads)."""
    torch = _torch()
    base, _ = get_backbone_and_lm_weight(model)
    with torch.no_grad():
        return base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def encode_pair(tok, src: str, hi: str, eos_id: int) -> tuple[list[int], list[int]]:
    from .gpu_traces import build_prompt
    p = tok(build_prompt(tok, src), add_special_tokens=False).input_ids
    y = tok(hi, add_special_tokens=False).input_ids + [eos_id]
    return p, y


def make_example(tok, prompt_ids, out_ids, K: int) -> dict:
    wi = word_initial_flags(tok, out_ids)
    return dict(prompt_ids=list(prompt_ids), out_ids=list(out_ids),
                word_initial=wi, cont=continuation_targets(wi, K))


def _deva_frac(s: str) -> float:
    letters = [ch for ch in s if ch.isalpha()]
    return sum("ऀ" <= ch <= "ॿ" for ch in letters) / max(1, len(letters))


_UI_CHARS = set(r"%_<>/\{}[]|=@#~^*")


def keep_pair(en: str, hi: str, cfg: dict) -> bool:
    """Sentence-like pairs only: IITB contains many software-UI strings (GNOME/KDE localisation)."""
    if not (cfg["MIN_HI_CHARS"] <= len(hi) <= cfg["MAX_HI_CHARS"]):
        return False
    if len(en.split()) < cfg["MIN_WORDS"] or len(hi.split()) < cfg["MIN_WORDS"] - 1:
        return False
    if en[-1:] not in ".?!\"'" or any(ch in _UI_CHARS for ch in en):
        return False
    return _deva_frac(hi) >= 0.9


def load_train_pairs(cfg: dict, log=print) -> list[tuple[str, str]]:
    """N_TRAIN + N_VAL IITB en-hi pairs, sampled uniformly at random (seeded) from the whole train split
    and filtered with ``keep_pair``."""
    if cfg.get("TRAIN_PAIRS") is not None:
        return list(cfg["TRAIN_PAIRS"])
    from datasets import load_dataset
    ds = load_dataset(cfg["TRAIN_DATASET"], split="train")
    need = cfg["N_TRAIN"] + cfg["N_VAL"]
    order = np.random.default_rng(cfg["SEED"]).permutation(len(ds))
    out, seen = [], 0
    for start in range(0, len(order), 20000):
        rows = ds.select(order[start:start + 20000].tolist())["translation"]
        for r in rows:
            seen += 1
            en, hi = r["en"].strip(), r["hi"].strip()
            if keep_pair(en, hi, cfg):
                out.append((en, hi))
                if len(out) >= need:
                    break
        if len(out) >= need:
            break
    log(f"train pairs: {len(out)} kept of {seen} sampled (split size {len(ds)})")
    return out


def collate(batch: list[dict], pad_id: int, device):
    """Right-padded inputs + index tensors of the output rows h_t (t = 0..n-1) for every example."""
    torch = _torch()
    L = max(len(b["prompt_ids"]) + len(b["out_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(batch), L), dtype=torch.long)
    rows_b, rows_i, tgt, tvalid, cont = [], [], [], [], []
    for bi, b in enumerate(batch):
        seq = b["prompt_ids"] + b["out_ids"]
        ids[bi, :len(seq)] = torch.tensor(seq)
        am[bi, :len(seq)] = 1
        P, n = len(b["prompt_ids"]), len(b["out_ids"])
        y = np.asarray(b["out_ids"], np.int64)
        K = b["cont"].shape[1]
        for t in range(n):
            rows_b.append(bi)
            rows_i.append(P + t - 1)
            tk = np.array([y[t + k] if t + k < n else -100 for k in range(1, K + 1)], np.int64)
            tgt.append(tk)
            tvalid.append(tk >= 0)
            cont.append(b["cont"][t])
    T = lambda a, dt: torch.tensor(np.asarray(a), dtype=dt, device=device)
    return dict(ids=ids.to(device), am=am.to(device), rows_b=T(rows_b, torch.long), rows_i=T(rows_i, torch.long),
                tgt=T(tgt, torch.long), tvalid=T(tvalid, torch.bool), cont=T(cont, torch.bool))


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #

def variant_losses(name: str, heads, h, B, W, K: int, head_w: float, lam_cont: float,
                   pos_per_head: int, gen) -> tuple:
    """Total loss and per-part float values for one variant on hidden rows h [N, d] (float32)."""
    torch = _torch()
    F = torch.nn.functional
    parts, logs = [], {}
    for k in range(1, K + 1):
        valid = B["tvalid"][:, k - 1]
        if name in ("v1", "v2"):
            valid = valid & B["cont"][:, k - 1]
        idx = valid.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            parts.append(h.sum() * 0.0)
            continue
        if len(idx) > pos_per_head:
            idx = idx[torch.randperm(len(idx), generator=gen, device="cpu")[:pos_per_head].to(idx.device)]
        lg = lm_logits(heads.z(h[idx], k), W)
        ce = F.cross_entropy(lg, B["tgt"][idx, k - 1])
        parts.append(ce)
        logs[f"ce{k}"] = ce.item()
    if name == "v0":
        total = sum(head_w ** (k - 1) * parts[k - 1] for k in range(1, K + 1))
        return total, logs
    # continuation head on all positions where y[t+1] exists
    m = B["tvalid"][:, 0]
    bce = F.binary_cross_entropy_with_logits(heads.cont(h[m]), B["cont"][m].float())
    logs["bce"] = bce.item()
    if name == "v1":
        total = sum(head_w ** (k - 1) * parts[k - 1] for k in range(1, K + 1)) + lam_cont * bce
    else:
        allp = parts + [bce]
        total = sum(torch.exp(-heads.log_var[i]) * allp[i] + heads.log_var[i] for i in range(len(allp)))
        logs["w"] = [round(float(torch.exp(-s.detach())), 3) for s in heads.log_var]
    return total, logs


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #

def head_accuracy(heads, h, B, W, K: int) -> dict:
    """Top-1 accuracy per head on all valid positions and on in-word (c_k = 1) positions."""
    torch = _torch()
    out = {}
    with torch.no_grad():
        for k in range(1, K + 1):
            v = B["tvalid"][:, k - 1]
            if not v.any():
                continue
            pred = lm_logits(heads.z(h[v], k), W).argmax(-1)
            hit = pred == B["tgt"][v, k - 1]
            cw = B["cont"][v, k - 1]
            out[f"acc{k}_all"] = float(hit.float().mean())
            out[f"acc{k}_inword"] = float(hit[cw].float().mean()) if cw.any() else float("nan")
        m = B["tvalid"][:, 0]
        if m.any():
            pc = torch.sigmoid(heads.cont(h[m])) > 0.5
            out["cont_acc"] = float((pc == B["cont"][m]).float().mean())
    return out


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def train_heads(model, tok, train: list[dict], val: list[dict], cfg: dict, device, log=print) -> tuple:
    torch = _torch()
    K = cfg["K"]
    base, W = get_backbone_and_lm_weight(model)
    W = W.detach()
    d = W.shape[1]
    torch.manual_seed(cfg["SEED"])
    heads = {v: make_heads(d, K).to(device) for v in VARIANTS}
    opt = {v: torch.optim.AdamW(heads[v].parameters(), lr=cfg["LR"], weight_decay=cfg["WD"]) for v in VARIANTS}
    order = sorted(range(len(train)), key=lambda i: len(train[i]["prompt_ids"]) + len(train[i]["out_ids"]))
    batches = [order[i:i + cfg["BATCH"]] for i in range(0, len(order), cfg["BATCH"])]
    random.Random(cfg["SEED"]).shuffle(batches)
    total = len(batches) if not cfg.get("MAX_STEPS") else min(len(batches), cfg["MAX_STEPS"])
    sched = {v: torch.optim.lr_scheduler.LambdaLR(
        opt[v], lambda s: min(1.0, (s + 1) / cfg["WARMUP"]) * max(0.05, 0.5 * (1 + math.cos(math.pi * s / total))))
        for v in VARIANTS}
    gen = torch.Generator().manual_seed(cfg["SEED"])
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    hist, t0 = [], time.time()
    val_b = [collate(val[i:i + cfg["BATCH"]], pad, device) for i in range(0, len(val), cfg["BATCH"])]

    def evaluate(step):
        acc = {v: [] for v in VARIANTS}
        for B in val_b:
            h = hidden_states(model, B["ids"], B["am"])[B["rows_b"], B["rows_i"]].float()
            for v in VARIANTS:
                acc[v].append(head_accuracy(heads[v], h, B, W, K))
        res = {v: {k: float(np.nanmean([a[k] for a in acc[v] if k in a])) for k in acc[v][0]} for v in VARIANTS}
        hist.append(dict(step=step, val=res, t=round(time.time() - t0, 1)))
        log(f"[val step {step}] " + " | ".join(
            f"{v}: in-word acc " + "/".join(f"{res[v].get(f'acc{k}_inword', float('nan')):.3f}" for k in range(1, K + 1))
            for v in VARIANTS))
        return res

    for step, bidx in enumerate(batches[:total]):
        B = collate([train[i] for i in bidx], pad, device)
        h = hidden_states(model, B["ids"], B["am"])[B["rows_b"], B["rows_i"]].float()
        logs = {}
        for v in VARIANTS:
            loss, lg = variant_losses(v, heads[v], h, B, W, K, cfg["HEAD_W"], cfg["LAMBDA_CONT"],
                                      cfg["POS_PER_HEAD"], gen)
            opt[v].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads[v].parameters(), 1.0)
            opt[v].step()
            sched[v].step()
            logs[v] = lg
        if step % 50 == 0:
            log(f"step {step}/{total} {time.time() - t0:.0f}s " + json.dumps(
                {v: {k: (round(x, 3) if isinstance(x, float) else x) for k, x in logs[v].items()} for v in VARIANTS}))
        if cfg["EVAL_EVERY"] and step % cfg["EVAL_EVERY"] == 0 and step:
            evaluate(step)
    final = evaluate(total)
    return heads, dict(history=hist, final_val=final, steps=total, seconds=round(time.time() - t0, 1))


# --------------------------------------------------------------------------- #
# dump along the target's outputs (for the exact trace simulation)
# --------------------------------------------------------------------------- #

def dump_predictions(model, tok, heads: dict, records: list[dict], cfg: dict, device) -> dict:
    """Per output position t of every record: the draft argmax for y[t] (alignment check against the
    Phase-1 traces), and per variant the heads' top-k ids / log-probs for y[t+1..t+K] and the
    continuation-head probabilities. Flat arrays + offsets, like the Phase-1 traces."""
    torch = _torch()
    K, TK = cfg["K"], cfg["DUMP_TOPK"]
    _, W = get_backbone_and_lm_weight(model)
    W = W.detach()
    cv = torch.tensor(continuation_vocab(tok), device=W.device)
    Wcv = W[cv]
    out = {"ids": [], "lens": [], "draft_argmax": []}
    for v in VARIANTS:
        out[f"{v}_ids"], out[f"{v}_logp"], out[f"{v}_cont"] = [], [], []
        out[f"{v}_cv_id"], out[f"{v}_cv_logp"] = [], []
    with torch.no_grad():
        for r in records:
            P, y = list(r["prompt_ids"]), list(r["out_ids"])
            x = torch.tensor([P + y], device=device)
            h = hidden_states(model, x, torch.ones_like(x))[0, len(P) - 1:len(P) - 1 + len(y)].float()
            out["ids"].append(int(r["id"]))
            out["lens"].append(len(y))
            out["draft_argmax"].append(lm_logits(h, W).argmax(-1).cpu().numpy().astype(np.int32))
            for v in VARIANTS:
                ids_k, lp_k, cid_k, clp_k = [], [], [], []
                for k in range(1, K + 1):
                    z = heads[v].z(h, k)
                    lp = torch.log_softmax(lm_logits(z, W), -1)
                    tk = lp.topk(TK, -1)
                    ids_k.append(tk.indices.cpu().numpy().astype(np.int32))
                    lp_k.append(tk.values.cpu().numpy().astype(np.float16))
                    lc = torch.log_softmax(lm_logits(z, Wcv), -1).max(-1)   # restricted to continuation tokens
                    cid_k.append(cv[lc.indices].cpu().numpy().astype(np.int32))
                    clp_k.append(lc.values.cpu().numpy().astype(np.float16))
                out[f"{v}_ids"].append(np.stack(ids_k, 1))      # [n, K, TK]
                out[f"{v}_logp"].append(np.stack(lp_k, 1))
                out[f"{v}_cv_id"].append(np.stack(cid_k, 1))    # [n, K]
                out[f"{v}_cv_logp"].append(np.stack(clp_k, 1))
                out[f"{v}_cont"].append(torch.sigmoid(heads[v].cont(h)).cpu().numpy().astype(np.float16))
    lens = np.asarray(out.pop("lens"), np.int64)
    res = dict(ids=np.asarray(out.pop("ids"), np.int64), offsets=np.concatenate([[0], np.cumsum(lens)]),
               cont_vocab=cv.cpu().numpy())
    for k, v in out.items():
        res[k] = np.concatenate(v, 0)
    return res


def time_heads(model, tok, heads, cfg: dict, device) -> dict:
    """Draft latency for q new tokens (KV cache of TIME_CTX) and the heads' latency (all K heads in one
    batched matmul; full vocabulary and continuation vocabulary), relative to the draft's q = 1 call."""
    from .gpu_traces import sync_all, time_fwd
    torch = _torch()
    tq = {q: time_fwd(model, q=q, ctx=cfg["TIME_CTX"], steps=cfg["TIME_STEPS"], reps=cfg["TIME_REPS"], device=device)
          for q in cfg["TIME_Q"]}
    _, W = get_backbone_and_lm_weight(model)
    Wcv = W[torch.tensor(continuation_vocab(tok), device=W.device)]
    h = torch.randn(1, W.shape[1], device=device)

    def run(Wm):
        ts = []
        with torch.no_grad():
            for s in range(cfg["TIME_STEPS"] + 10):
                sync_all()
                t = time.perf_counter()
                z = torch.cat([heads.z(h, k) for k in range(1, cfg["K"] + 1)], 0)
                lm_logits(z, Wm).argmax(-1)
                torch.sigmoid(heads.cont(h))
                sync_all()
                if s >= 10:
                    ts.append(time.perf_counter() - t)
        return float(np.median(ts))

    t_full, t_cv = run(W), run(Wcv)
    return dict(t_draft={str(q): v for q, v in tq.items()},
                r_q={str(q): v / tq[1] for q, v in tq.items()},
                t_heads_full=t_full, head_rel_full=t_full / tq[1],
                t_heads_cv=t_cv, head_rel_cv=t_cv / tq[1], cont_vocab_size=int(Wcv.shape[0]),
                ctx=cfg["TIME_CTX"])


# --------------------------------------------------------------------------- #
# notebook entry point
# --------------------------------------------------------------------------- #

def _find_heads(cfg: dict) -> Path | None:
    for root in cfg["SEARCH"]:
        p = Path(root)
        if p.exists():
            for f in p.rglob("wordheads_v0.pt"):
                return f.parent
    return None


def wordheads_main(cfg_in: dict, target_records: dict) -> dict:
    """Train v0/v1/v2 heads on the frozen draft, dump predictions along ``target_records``
    ({split: [records with id, prompt_ids, out_ids]}), time the draft and heads, zip everything."""
    torch = _torch()
    from . import gpu_traces as G
    cfg = dict(DEFAULT_WH)
    cfg.update(cfg_in)
    out = Path(cfg["OUT_ROOT"]) / "wordheads"
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model, name = None, None
    for nm in cfg["DRAFT"]:
        try:
            model = G.from_pretrained(nm, dtype).to(device)
            name = nm
            break
        except Exception as e:  # noqa: BLE001
            log(f"could not load {nm}: {e}")
    assert model is not None, "no draft model could be loaded"
    for p in model.parameters():
        p.requires_grad_(False)
    tok = G.load_tokenizer(name)
    tok.padding_side = "right"
    eos = G.get_eos_ids(model, tok)
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    eos_id = eot if (eot is not None and eot != tok.unk_token_id and eot in eos) else eos[0]
    log(f"draft {name} on {device}; eos {eos_id}")

    K = cfg["K"]
    prev = _find_heads(cfg)
    info = dict(draft=name, cfg={k: v for k, v in cfg.items() if k != "TRAIN_PAIRS"})
    if prev is not None:
        d = get_backbone_and_lm_weight(model)[1].shape[1]
        heads = {}
        for v in VARIANTS:
            heads[v] = make_heads(d, K).to(device)
            heads[v].load_state_dict({k: t.float() for k, t in torch.load(prev / f"wordheads_{v}.pt").items()})
        log(f"heads: reused from {prev}")
        info["train"] = json.loads((prev / "train_info.json").read_text()) if (prev / "train_info.json").exists() else {}
    else:
        pairs = load_train_pairs(cfg, log)
        rnd = random.Random(cfg["SEED"])
        rnd.shuffle(pairs)
        ex = []
        for en, hi in pairs:
            p, y = encode_pair(tok, en, hi, eos_id)
            if len(p) + len(y) <= cfg["MAX_LEN"]:
                ex.append(make_example(tok, p, y, K))
        val, train = ex[:cfg["N_VAL"]], ex[cfg["N_VAL"]:]
        wi = np.concatenate([e["word_initial"] for e in train])
        log(f"examples: train {len(train)} val {len(val)}; output tokens {len(wi)}; continuation share {1 - wi.mean():.3f}")
        heads, tinfo = train_heads(model, tok, train, val, cfg, device, log)
        tinfo["continuation_share"] = float(1 - wi.mean())
        info["train"] = tinfo
        for v in VARIANTS:
            torch.save({k: t.half() for k, t in heads[v].state_dict().items()}, out / f"wordheads_{v}.pt")
        (out / "train_info.json").write_text(json.dumps(tinfo, indent=1))
    for v in VARIANTS:
        heads[v].eval()

    info["dump"] = {}
    for split, recs in target_records.items():
        res = dump_predictions(model, tok, heads, recs, cfg, device)
        np.savez_compressed(out / f"heads_{split}.npz", **res)
        info["dump"][split] = dict(records=len(recs), positions=int(res["offsets"][-1]))
        log(f"dump {split}: {info['dump'][split]}")
    info["timing"] = time_heads(model, tok, heads["v1"], cfg, device)
    log("timing: " + json.dumps(info["timing"]))
    info["run"] = G.run_info(cfg.get("BUILD", "nobuild"), draft=name)
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    zp = G.zip_dir(out, Path(cfg["OUT_ROOT"]) / "wordheads.zip")
    log(f"wrote {zp}")
    logf.close()
    return info
