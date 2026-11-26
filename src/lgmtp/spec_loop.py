"""Real greedy draft-and-verify loop, batch size 1, KV caches for both models.

The *same* policy classes as the simulator run live: ``LiveView`` exposes ``feat`` / ``token`` with
the simulator's index convention (absolute output positions) and the same causality limit
(``p + g - 1 + lookahead``). Draft features are computed from the draft's actual logits
(conditioned on its own drafted tokens).

Round at committed output length p (pending token = last committed token, not yet in the caches):
  1. draft: feed every committed token missing from the draft cache, then draft greedily; after
     each drafted token ask the policy (g >= 1). A lookahead-1 policy needs q_{p+g}, i.e. one more
     draft forward; if the policy continues, that forward produced the next drafted token.
  2. verify: one target forward on [pending, d_1..d_g]; accept the leading matches a; the target
     argmax at position a is the correction / bonus token.
  3. crop both caches to the committed length minus one (``crop_to``; transformers 5.22 ``crop``
     takes the number of tokens to *remove*).
Output = committed tokens truncated after the first eos or at ``max_new``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from .simulator import GAMMA_MAX, CausalityError, Policy

FEATURE_NAMES = ("H", "top1", "top2", "p_space", "p_punct", "p_eos", "p_newword", "p_mark", "p_cont")


def _torch():
    import torch
    return torch


def crop_to(cache, length: int) -> None:
    """Crop a DynamicCache to ``length`` tokens (no-op if already shorter).

    transformers 5.22: ``DynamicCache.crop(n)`` removes ``-n`` tokens for negative n and rejects positive n
    (older versions took the final length). The result is checked so an API change fails loudly
    instead of silently corrupting the cache."""
    cur = cache.get_seq_length()
    if cur > length:
        cache.crop(-(cur - length))
        got = cache.get_seq_length()
        if got != length:
            raise RuntimeError(f"cache crop failed: expected length {length}, got {got} (transformers API change?)")


class NoisyDraft:
    """CPU smoke-test helper: wraps a causal LM and adds deterministic Gaussian noise to its logits,
    seeded by the token that precedes each predicted position. Teacher-forced and live calls therefore
    see the same draft distribution for the same context, while draft and target now disagree, so the
    rejection / cache-crop path of the real loop is exercised. Not used on Kaggle."""

    def __init__(self, model, sigma: float = 3.0, seed: int = 1234):
        self.model, self.sigma, self.seed = model, float(sigma), int(seed)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, input_ids=None, **kw):
        torch = _torch()
        out = self.model(input_ids=input_ids, **kw)
        V = out.logits.shape[-1]
        prev = input_ids[0].tolist()
        noise = torch.stack([torch.randn(V, generator=torch.Generator().manual_seed(self.seed * 1_000_003 + int(t)))
                             for t in prev]).to(out.logits.device, out.logits.dtype)
        out.logits = out.logits + self.sigma * noise[None]
        return out


def _sync():
    torch = _torch()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


class LiveView:
    """Causality-checked view of the live decoding state (same interface as ``TraceView``)."""

    def __init__(self):
        self.out: list[int] = []        # committed output tokens
        self.draft: dict[int, int] = {}  # absolute idx -> drafted token (this round, incl. lookahead argmax)
        self.feats: dict[int, dict] = {}  # absolute idx -> features of q_idx
        self.p = 0
        self.limit = -1
        self.is_oracle = False
        self.n = 10 ** 9

    def _check(self, idx):
        if idx > self.limit:
            raise CausalityError(f"live read index {idx} > limit {self.limit} (p={self.p})")

    def feat(self, name, idx):
        self._check(idx)
        if name == "draft_argmax":
            return self.draft[idx]
        if idx not in self.feats:
            raise CausalityError(f"feature {name}[{idx}] not computed live (committed by the target)")
        return self.feats[idx][name]

    def token(self, idx):
        self._check(idx)
        return self.out[idx] if idx < self.p else self.draft[idx]

    def gold(self, name, idx):
        raise CausalityError("oracle policies cannot run live")


@dataclass
class LoopResult:
    out_ids: list
    rounds: list = field(default_factory=list)  # dicts: p, g, accepted, extra_lookahead, t_draft, t_verify
    wall: float = 0.0
    n_draft_forwards: int = 0
    n_target_forwards: int = 0


class SpecDecoder:
    def __init__(self, target, draft, masks, eos_ids, device: str = "cuda:0", gamma_max: int = GAMMA_MAX):
        torch = _torch()
        from .gpu_traces import _mask_matrix
        self.target, self.draft, self.device = target, draft, device
        self.eos = set(int(i) for i in eos_ids)
        self.gamma_max = gamma_max
        V = draft.get_output_embeddings().weight.shape[0]
        self.M = _mask_matrix(masks, V, device)            # (V, 5): space, punct, eos, mark, cont
        self.torch = torch

    # ---- model calls ---------------------------------------------------- #
    def _fwd(self, model, ids, cache):
        torch = self.torch
        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            return model(input_ids=x, past_key_values=cache, use_cache=True).logits[0].float()

    def _features(self, logits_row) -> tuple[int, dict]:
        torch = self.torch
        lp = torch.log_softmax(logits_row, -1)
        p = lp.exp()
        top = p.topk(2)
        P = p @ self.M.to(p.device)
        H = -(p * lp).sum()
        vals = torch.stack([H, top.values[0], top.values[1], P[0], P[1], P[2], P[0] + P[1] + P[2], P[3],
                            P[4]]).cpu().tolist()
        f = dict(zip(FEATURE_NAMES, vals))
        return int(top.indices[0].item()), f

    # ---- decoding --------------------------------------------------------- #
    def decode(self, prompt_ids: list[int], policy: Policy, max_new: int, record: bool = True) -> LoopResult:
        from transformers import DynamicCache
        if policy.is_oracle:
            raise ValueError("oracle policies cannot run live")
        tc, dc = DynamicCache(), DynamicCache()
        la = int(policy.lookahead)
        res = LoopResult(out_ids=[])
        _sync()
        t_start = time.perf_counter()
        # prefill: target on prompt[:-1], draft on prompt[:-1]; pending token = prompt[-1]
        seq = list(prompt_ids)
        if len(seq) > 1:
            self._fwd(self.target, seq[:-1], tc)
            self._fwd(self.draft, seq[:-1], dc)
            res.n_target_forwards += 1
            res.n_draft_forwards += 1
        view = LiveView()
        policy.reset(view)
        n_prompt = len(prompt_ids)
        done = False
        while not done:
            p = len(seq) - n_prompt          # committed output length
            view.p, view.out = p, seq[n_prompt:]
            view.draft, view.feats = {}, {}
            cap = min(self.gamma_max, max_new - p)
            _sync()
            t0 = time.perf_counter()
            drafted: list[int] = []
            extra = False
            g = 0
            if cap > 0 and not policy.initial_stop(view, p):
                # draft forward on all committed tokens missing from the draft cache -> q_p
                missing = seq[dc.get_seq_length():]
                lg = self._fwd(self.draft, missing, dc)[-1]
                res.n_draft_forwards += 1
                tok, f = self._features(lg)
                view.draft[p], view.feats[p] = tok, f
                drafted.append(tok)
                g = 1
                while g < cap:
                    if la:   # q_{p+g}: feed the last drafted token
                        lg = self._fwd(self.draft, [drafted[-1]], dc)[-1]
                        res.n_draft_forwards += 1
                        tok, f = self._features(lg)
                        view.draft[p + g], view.feats[p + g] = tok, f
                    view.limit = p + g - 1 + la
                    if policy.stop(view, p, g):
                        extra = bool(la)
                        break
                    if not la:
                        lg = self._fwd(self.draft, [drafted[-1]], dc)[-1]
                        res.n_draft_forwards += 1
                        tok, f = self._features(lg)
                        view.draft[p + g], view.feats[p + g] = tok, f
                    drafted.append(view.draft[p + g])
                    g += 1
            _sync()
            t1 = time.perf_counter()
            # verify [pending] + drafted
            lg = self._fwd(self.target, [seq[-1]] + drafted, tc)
            res.n_target_forwards += 1
            am = lg.argmax(-1).tolist()
            a = 0
            while a < g and am[a] == drafted[a]:
                a += 1
            new = drafted[:a] + [am[a]]
            _sync()
            t2 = time.perf_counter()
            seq += new
            L = len(seq)
            crop_to(tc, L - 1)
            crop_to(dc, L - 1)
            if record:
                res.rounds.append(dict(p=p, g=g, accepted=a, extra_lookahead=extra, t_draft=t1 - t0, t_verify=t2 - t1))
            out = seq[n_prompt:]
            if any(t in self.eos for t in new) or len(out) >= max_new:
                done = True
        _sync()
        res.wall = time.perf_counter() - t_start
        out = seq[n_prompt:]
        cut = next((i + 1 for i, t in enumerate(out) if t in self.eos), len(out))
        res.out_ids = out[:min(cut, max_new)]
        return res


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def target_margin_at(target, prompt_ids, prefix_out, device) -> dict:
    """Target top-1 vs top-2 logit margin for the next token after prompt + prefix_out (fp16 near-tie check)."""
    torch = _torch()
    x = torch.tensor([list(prompt_ids) + list(prefix_out)], device=device)
    with torch.no_grad():
        lg = target(x).logits[0, -1].float()
    v, i = lg.topk(2)
    return dict(top1=int(i[0]), top2=int(i[1]), margin=float(v[0] - v[1]))


# --------------------------------------------------------------------------- #
# notebook 03 entry point (also called by scripts/smoke_cpu.py --part 03)
# --------------------------------------------------------------------------- #

# MorphEntropy runs in its constrained form: the selected config has max_ext=0 and is identical to Entropy
# (user decision 2026-11-02).
LIVE_FAMILIES = ["NoDraft", "FixedK", "Entropy", "WordBoundarySoft", "Hybrid", "MorphEntropy_ext1"]

DEFAULT_CONFIG_03 = dict(
    N_SENT=100, REPEATS=3, MAX_NEW=256, SMOKE=False,
    TARGET="meta-llama/Meta-Llama-3-8B-Instruct",
    DRAFT="meta-llama/Llama-3.2-1B-Instruct",
    MAX_MEMORY={0: "8GiB", 1: "10GiB"},
    OUT_ROOT="/kaggle/working",
    GIT_HASH="unknown",
    POLICIES=LIVE_FAMILIES,
    SELECTED=None,          # {family: params} from results/phase1/selected_params.json
    PREFIX_MODEL_STATE=None,  # PrefixBoundaryModel state dict (decoded from the embedded gzip+base64)
    IDS=None,               # devtest sentence ids (results/phase1/real_loop_ids.json)
)


def _append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def notebook03_main(cfg: dict) -> dict:
    """Real-loop run on the GPU (or CPU with SMOKE=True: draft model as both target and draft)."""
    import shutil
    from pathlib import Path
    from . import gpu_traces as G
    from .morph import PrefixBoundaryModel
    from .policies import TokenTable
    from .traces import save_traces
    from .tuning import Context, make_policy
    from .vocab import build_masks
    torch = _torch()
    cfg = {**DEFAULT_CONFIG_03, **cfg}
    out_dir = Path(cfg["OUT_ROOT"]) / "real_loop"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- models ---- #
    if cfg["SMOKE"]:
        device, dtype = "cpu", torch.float32
        tok = G.load_tokenizer(cfg["DRAFT"])
        draft = target = G.load_draft(cfg["DRAFT"], dtype, device)
        if cfg.get("SMOKE_DRAFT_NOISE"):          # exercise rejections + cache cropping on CPU
            draft = NoisyDraft(target, cfg["SMOKE_DRAFT_NOISE"])
    else:
        assert torch.cuda.device_count() >= 2, "Need 2 GPUs: Settings > Accelerator > GPU T4 x2"
        device, dtype = "cuda:0", torch.float16
        tok = G.load_tokenizer(cfg["TARGET"])
        target = G.load_target(cfg["TARGET"], dtype, max_memory=cfg["MAX_MEMORY"])
        draft = G.load_draft(cfg["DRAFT"], dtype, device)
        G.check_draft_target_compat(draft, target, G.load_tokenizer(cfg["DRAFT"]), tok, "परीक्षण")
    eos_ids = G.get_eos_ids(target, tok)
    masks = build_masks(tok, eos_ids)
    ctx = Context(newword_mask=masks.starts_newword, table=TokenTable.from_tokenizer(tok, masks),
                  prefix_model=PrefixBoundaryModel.from_state(cfg["PREFIX_MODEL_STATE"]))
    sel = cfg["SELECTED"]
    policies = {f: make_policy(f, sel.get(f, {}), ctx) for f in cfg["POLICIES"]}
    print("policies:", policies)
    dec = SpecDecoder(target, draft, masks, eos_ids, device)

    # ---- sentences ---- #
    rows = {r[0]: r for r in G.load_flores_pairs("devtest")}
    ids = list(cfg["IDS"])[: cfg["N_SENT"]]
    prompts = {i: tok(G.build_prompt(tok, rows[i][1]), add_special_tokens=False)["input_ids"] for i in ids}

    # ---- unbatched greedy reference (B9) ---- #
    ref_path = out_dir / "greedy_unbatched.jsonl"
    refs = {r["id"]: r for r in G.read_jsonl(ref_path)}
    for i in ids:
        if i not in refs:
            o = G.generate_unbatched(target, tok, prompts[i], cfg["MAX_NEW"], eos_ids, device)
            refs[i] = dict(id=i, src=rows[i][1], ref=rows[i][2], prompt_ids=prompts[i], out_ids=o,
                           out_text=tok.decode(o, skip_special_tokens=True).strip())
            _append_jsonl(ref_path, refs[i])
    print(f"unbatched greedy references: {len(refs)}")

    # ---- warm-up ---- #
    for f, pol in policies.items():
        dec.decode(prompts[ids[0]], pol, min(32, cfg["MAX_NEW"]), record=False)

    # ---- timed runs (resumable) ---- #
    res_path = out_dir / "loop_results.jsonl"
    done = {(r["rep"], r["id"], r["policy"]) for r in G.read_jsonl(res_path)}
    t_first = time.perf_counter()
    n_new = 0
    for rep in range(cfg["REPEATS"]):
        for si, i in enumerate(ids):
            for f, pol in policies.items():
                if (rep, i, f) in done:
                    continue
                r = dec.decode(prompts[i], pol, cfg["MAX_NEW"])
                lossless = r.out_ids == refs[i]["out_ids"]
                _append_jsonl(res_path, dict(rep=rep, id=i, policy=f, params=sel.get(f, {}), wall=r.wall,
                                             n_out=len(r.out_ids), out_ids=r.out_ids, lossless=lossless,
                                             first_divergence=None if lossless else first_divergence(r.out_ids, refs[i]["out_ids"]),
                                             n_draft_forwards=r.n_draft_forwards,
                                             n_target_forwards=r.n_target_forwards, rounds=r.rounds))
                n_new += 1
            if rep == 0 and si == 4 and n_new:
                per = (time.perf_counter() - t_first) / 5
                est = per * len(ids) * cfg["REPEATS"] / 3600
                print(f"ESTIMATE: {per:.1f}s per sentence (all policies) -> about {est:.1f} h for "
                      f"N_SENT={len(ids)} x REPEATS={cfg['REPEATS']}. Lower N_SENT / REPEATS if that is too long.")
        print(f"repeat {rep} done")

    # ---- losslessness details ---- #
    results = G.read_jsonl(res_path)
    div = []
    for r in results:
        if r["rep"] == 0 and not r["lossless"]:
            pos = r["first_divergence"]
            ref = refs[r["id"]]["out_ids"]
            m = target_margin_at(target, prompts[r["id"]], ref[:pos], device) if pos is not None and pos < len(ref) else {}
            div.append(dict(id=r["id"], policy=r["policy"], pos=pos, ref_token=ref[pos] if pos is not None and pos < len(ref) else None,
                            loop_token=r["out_ids"][pos] if pos is not None and pos < len(r["out_ids"]) else None, **m))
    (out_dir / "divergences.json").write_text(json.dumps(div, indent=1))
    n_loss = sum(r["lossless"] for r in results)
    print(f"losslessness: {n_loss} / {len(results)} decodes identical to unbatched greedy; {len(div)} divergences logged (rep 0)")

    # ---- teacher-forced traces of the unbatched greedy outputs (for notebook 04) ---- #
    recs = [refs[i] for i in ids]
    dfe = G.run_draft_pass(draft, recs, masks, out_dir, "real100", device)
    tfe = G.run_target_pass(target, recs, dfe, out_dir, "real100", device)
    traces = G.assemble_traces(recs, dfe, tfe)
    save_traces(out_dir / "traces_real100.npz", traces, dict(split="devtest-real100", eos_ids=eos_ids,
                                                              git_hash=cfg["GIT_HASH"], max_new=cfg["MAX_NEW"]))

    info = G.run_info(cfg["GIT_HASH"], n_sent=len(ids), repeats=cfg["REPEATS"], policies=list(policies),
                      selected=sel, smoke=cfg["SMOKE"], target=cfg["TARGET"] if not cfg["SMOKE"] else cfg["DRAFT"],
                      draft=cfg["DRAFT"], max_memory={str(k): v for k, v in cfg["MAX_MEMORY"].items()})
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    zp = G.zip_dir(out_dir, Path(cfg["OUT_ROOT"]) / "real_loop.zip",
                   include=["greedy_unbatched.jsonl", "loop_results.jsonl", "divergences.json",
                            "traces_real100.npz", "run_info.json"])
    print("\n" + "=" * 70)
    print("DONE. Download  real_loop.zip  from the notebook's Output tab")
    print("and unzip it into  data/real_loop/  in the repo (files land directly in that folder).")
    print("=" * 70)
    return dict(zip=zp, n_lossless=n_loss, n_results=len(results), divergences=div)
