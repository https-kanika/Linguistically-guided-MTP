"""The universal head: one small network that drafts the next k word units for any LLM (W1).

Input (all text, see ``uni_features``): prompt chars, committed output chars, and for each of the last
M target distributions its top-K candidate strings with probabilities. Output: k unit distributions in
one pass. Head 1 is restricted to units that extend the partial unit being written (``lo..hi``), heads
2..k are chained on the previous unit's embedding (teacher forcing in training, argmax at inference).
The text-only ablation is the same network without the candidate tokens.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .uni_features import K_CAND, K_HEADS, L_OUT, L_PROMPT, M_LAGS, N_FEAT, S_CHARS, FeatureSet, make_batch
from .unichar import N_UNITS, PAD, SCRIPTS


def _torch():
    import torch
    return torch


DEFAULT_ARCH = dict(d=256, layers=2, nhead=4, k=K_HEADS, S=S_CHARS, M=M_LAGS, K=K_CAND, F=N_FEAT,
                    L_max=max(L_PROMPT, L_OUT) + 2, dropout=0.1)


def build_model(n_vocab: int, arch: dict | None = None):
    torch = _torch()
    nn = torch.nn
    a = {**DEFAULT_ARCH, **(arch or {})}

    class UniHead(nn.Module):
        def __init__(self):
            super().__init__()
            d = a["d"]
            self.arch, self.n_vocab = a, n_vocab
            self.char = nn.Embedding(N_UNITS, d, padding_idx=PAD)
            self.script = nn.Embedding(len(SCRIPTS), d)
            self.ctype = nn.Embedding(6, d)
            self.pos = nn.Embedding(a["L_max"], d)
            self.spos = nn.Embedding(a["S"], d)
            self.epos = nn.Embedding(a["S"], d)
            self.cfeat = nn.Linear(a["F"], d)
            self.lag = nn.Embedding(a["M"] + 1, d)
            self.rank = nn.Embedding(a["K"], d)
            self.cand_type = nn.Parameter(torch.zeros(d))
            self.queries = nn.Parameter(torch.randn(a["k"], d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, a["nhead"], 4 * d, a["dropout"], batch_first=True,
                                               norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, a["layers"], enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.unit = nn.Embedding(n_vocab + 1, d)          # row n_vocab = unknown previous unit
            nn.init.normal_(self.unit.weight, std=0.02)
            self.out_bias = nn.Parameter(torch.zeros(n_vocab))
            self.chain = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
                                        for _ in range(a["k"] - 1)])
            self.proj = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
                                       for _ in range(a["k"])])

        def encode(self, b: dict):
            x = (self.char(b["ctx_u"]) + self.script(b["ctx_s"]) + self.ctype(b["ctx_t"]) + self.pos(b["ctx_p"]))
            pad = b["ctx_u"] == PAD
            parts, masks = [x], [pad]
            if "cand_u" in b:
                cu, eu = b["cand_u"], b["ext_u"]
                S = cu.shape[-1]
                sp = torch.arange(S, device=cu.device)
                cm = (cu != PAD).unsqueeze(-1).float()
                ce = ((self.char(cu) + self.script(b["cand_s"]) + self.spos(sp)) * cm).sum(2) / cm.sum(2).clamp(min=1)
                em = (eu != PAD).unsqueeze(-1).float()
                ee = ((self.char(eu) + self.epos(sp)) * em).sum(2) / em.sum(2).clamp(min=1)
                v = ce + ee + self.cfeat(b["cand_f"]) + self.lag(b["cand_lag"]) + self.rank(b["cand_rank"]) + self.cand_type
                parts.append(v)
                masks.append(~b["cand_mask"])
            B = x.shape[0]
            q = self.queries.unsqueeze(0).expand(B, -1, -1) + self.ctype.weight[5]
            parts.append(q)
            masks.append(torch.zeros(B, q.shape[1], dtype=torch.bool, device=x.device))
            h = self.enc(torch.cat(parts, 1), src_key_padding_mask=torch.cat(masks, 1))
            return self.norm(h[:, -q.shape[1]:])

        def head_logits(self, hq, i: int, prev=None):
            x = hq[:, i]
            if i > 0:
                x = x + self.chain[i - 1](torch.cat([x, self.unit(prev)], -1))
            h = self.proj[i](x)
            return h @ self.unit.weight[: self.n_vocab].t() + self.out_bias

    return UniHead()


def range_mask(lo, hi, V: int):
    torch = _torch()
    ar = torch.arange(V, device=lo.device)
    return (ar[None] >= lo[:, None]) & (ar[None] < hi[:, None])


def to_device(batch: dict, device) -> dict:
    torch = _torch()
    out = {}
    for k, v in batch.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        out[k] = t.to(device, non_blocking=True)
    if "cand_mask" in out:
        out["cand_mask"] = out["cand_mask"].bool()
    return out


def loss_fn(model, b: dict):
    """Sum of per-head cross entropies (ignore -100); head 1 restricted to its prefix range."""
    torch = _torch()
    F = torch.nn.functional
    V = model.n_vocab
    hq = model.encode(b)
    tgt = b["tgt"]
    total, stats = 0.0, {}
    prev = None
    for i in range(tgt.shape[1]):
        lg = model.head_logits(hq, i, prev).float()
        if i == 0:
            lg = lg.masked_fill(~range_mask(b["lo"], b["hi"], V), -1e4)
        t = tgt[:, i]
        m = t != -100
        if m.any():
            li = F.cross_entropy(lg, t, ignore_index=-100)
            total = total + li
            stats[f"loss{i + 1}"] = float(li.detach())
            stats[f"acc{i + 1}"] = float((lg.argmax(-1)[m] == t[m]).float().mean())
        prev = torch.where(m, t, torch.full_like(t, V))
    return total, stats


def train_head(model, sources: list, steps: int, batch_size: int, lr: float, device, use_cands: bool,
               val=None, log_every: int = 200, seed: int = 0, warmup: int = 200, log=print) -> dict:
    """``sources``: list of (FeatureSet, position index array). Batches sample positions uniformly over the
    union. ``val``: optional list of (FeatureSet, idx) evaluated at the end. AdamW, cosine decay, fp16
    autocast on GPU."""
    torch = _torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(device).train()
    sizes = np.array([len(ix) for _, ix in sources], float)
    if sizes.sum() == 0:
        raise ValueError("no training positions")
    probs = sizes / sizes.sum()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / max(steps, 1))))
    amp = device != "cpu" and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    hist, t0, run = [], time.time(), {}
    for step in range(steps):
        counts = rng.multinomial(batch_size, probs)
        parts = []
        for (fs, ix), cnt in zip(sources, counts):
            if cnt:
                parts.append((fs, ix[rng.integers(0, len(ix), cnt)]))
        loss = 0.0
        opt.zero_grad(set_to_none=True)
        stats_acc = {}
        for fs, ii in parts:
            b = to_device(make_batch(fs, ii, use_cands=use_cands), device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                l, st = loss_fn(model, b)
            w = len(ii) / batch_size
            scaler.scale(l * w).backward()
            for k, v in st.items():
                stats_acc[k] = stats_acc.get(k, 0.0) + v * w
            loss += float(l.detach()) * w
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        for k, v in stats_acc.items():
            run[k] = 0.98 * run.get(k, v) + 0.02 * v
        if (step + 1) % log_every == 0 or step == steps - 1:
            rec = dict(step=step + 1, loss=loss, sec=round(time.time() - t0, 1), **{k: round(v, 4) for k, v in run.items()})
            hist.append(rec)
            log(f"  step {step + 1}/{steps} loss {loss:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in run.items())
                + f" ({rec['sec']:.0f}s)")
    out = dict(history=hist, train_sec=time.time() - t0)
    if val:
        out["val"] = evaluate_heads(model, val, device, use_cands)
        log("  val: " + json.dumps(out["val"]))
    return out


def evaluate_heads(model, sets: list, device, use_cands: bool, batch_size: int = 1024) -> dict:
    """Teacher-forced per-head accuracy and loss on (FeatureSet, idx) pairs."""
    torch = _torch()
    model.eval()
    agg, n = {}, {}
    with torch.no_grad():
        for fs, ix in sets:
            for s in range(0, len(ix), batch_size):
                b = to_device(make_batch(fs, ix[s:s + batch_size], use_cands=use_cands), device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
                    _, st = loss_fn(model, b)
                w = len(ix[s:s + batch_size])
                for k, v in st.items():
                    agg[k] = agg.get(k, 0.0) + v * w
                    n[k] = n.get(k, 0) + w
    model.train()
    return {k: round(agg[k] / n[k], 4) for k in agg}


def predict(model, fs: FeatureSet, device, use_cands: bool, idx=None, batch_size: int = 1024):
    """Greedy chained prediction at every position: units ``w`` (N, k; -1 = none) and their
    probabilities ``pw`` (head 1 renormalised over its prefix range)."""
    torch = _torch()
    model.eval()
    idx = np.arange(fs.n_pos) if idx is None else np.asarray(idx)
    V, k = model.n_vocab, model.arch["k"]
    W = np.full((len(idx), k), -1, np.int32)
    P = np.zeros((len(idx), k), np.float32)
    eos = V - 1
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            ii = idx[s:s + batch_size]
            b = to_device(make_batch(fs, ii, use_cands=use_cands), device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
                hq = model.encode(b)
                prev, alive = None, (b["hi"] > b["lo"])
                for i in range(k):
                    lg = model.head_logits(hq, i, prev).float()
                    if i == 0:
                        lg = lg.masked_fill(~range_mask(b["lo"], b["hi"], V), -1e4)
                    pr = torch.softmax(lg, -1)
                    p, w = pr.max(-1)
                    W[s:s + len(ii), i] = torch.where(alive, w, torch.full_like(w, -1)).cpu().numpy()
                    P[s:s + len(ii), i] = torch.where(alive, p, torch.zeros_like(p)).cpu().numpy()
                    alive = alive & (w != eos)
                    prev = w
    return W, P


def time_head(model, fs: FeatureSet, device, use_cands: bool, n: int = 200, warmup: int = 20) -> dict:
    """Batch-1 latency of one drafting call: GPU forward + chained argmax (inputs already on the device),
    and separately the CPU time to build the inputs. Median over ``n`` positions."""
    torch = _torch()
    model.eval()
    rng = np.random.default_rng(0)
    ix = rng.integers(0, fs.n_pos, n + warmup)
    gpu, cpu = [], []
    V, k = model.n_vocab, model.arch["k"]
    sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else (lambda: None)
    with torch.no_grad():
        for j, i in enumerate(ix):
            t0 = time.perf_counter()
            nb = make_batch(fs, [i], use_cands=use_cands)
            t1 = time.perf_counter()
            b = to_device(nb, device)
            sync()
            t2 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
                hq = model.encode(b)
                prev = None
                for h in range(k):
                    lg = model.head_logits(hq, h, prev)
                    if h == 0:
                        lg = lg.masked_fill(~range_mask(b["lo"], b["hi"], V), -1e4)
                    prev = lg.argmax(-1)
            _ = prev.cpu()
            sync()
            t3 = time.perf_counter()
            if j >= warmup:
                cpu.append(t1 - t0)
                gpu.append(t3 - t2)
    return dict(t_head_gpu=float(np.median(gpu)), t_inputs_cpu=float(np.median(cpu)), n=n, device=str(device),
                params=int(sum(p.numel() for p in model.parameters())))


def save_head(model, path, extra: dict | None = None) -> None:
    torch = _torch()
    torch.save(dict(state=model.state_dict(), arch=model.arch, n_vocab=model.n_vocab, extra=extra or {}), path)


def load_head(path, device="cpu"):
    torch = _torch()
    d = torch.load(path, map_location=device, weights_only=False)
    m = build_model(d["n_vocab"], d["arch"])
    m.load_state_dict(d["state"])
    return m.to(device), d.get("extra", {})
