"""Real greedy speculative-decoding loop with word-completion heads on the draft (batch size 1, KV caches).

Live counterpart of ``refund_sim._round_heads``: the 1B drafts a token; unless the entropy rule fires, the
heads on the *same* hidden state propose up to K continuation tokens (argmax within the continuation
vocabulary; a proposal chain stops at head probability < ``tau_h``, continuation probability < ``tau_c``,
or when the word stops being a lexicon prefix); the next 1B call reads them in one forward. The 8B verifies
everything, so the output is the 8B's own greedy output.

Checked methods (kind "checked", ``checked.py``): an n-gram and/or the heads propose tokens and the 1B checks
them inside its next call, so every drafted token is the 1B's own greedy token.

All methods (NoDraft, Entropy, heads, checked) run through the same code, so differences between them come from the
drafter alone. Differences from the Phase-1 loop (``spec_loop.SpecDecoder``), applied to every method:
  * drafting stops at an end-of-text token (Phase 1 kept drafting past it, 3.3% of Entropy's time);
  * NoDraft never calls the draft model (Phase 1 prefilled its cache anyway);
  * the draft reads the prompt in its first draft call instead of in a separate prefill call;
  * LM-head logits are computed only for the position that is needed.

Files written to ``OUT_ROOT/heads_loop`` (zipped to ``heads_loop.zip``): greedy_unbatched.jsonl,
loop_results.jsonl (one line per decode, with every round), timing.json, summary.json, SUMMARY.md, run_info.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .checked import NGram, proposal
from .policies import entropy_fires
from .refund_sim import BETA, DEV_RE, GMAX, Lexicon, devanagari_mask
from .spec_loop import crop_to, first_divergence
from .vocab import ZW, first_char, special_ids, token_bytes_list
from .wordheads import continuation_vocab, get_backbone_and_lm_weight, lm_logits, make_heads

PAPER_LEXICON_TYPES = 72797      # refund_sim.Lexicon.build (feasibility.json: 72907, which also counts non-Devanagari rows)
PAPER_CONT_VOCAB = 1406          # paper, §4


def _torch():
    import torch
    return torch


def _sync():
    torch = _torch()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


# --------------------------------------------------------------------------- #
# causal word starts
# --------------------------------------------------------------------------- #

def _last_char(b: bytes) -> str:
    """Last complete character of ``b`` ('' if ``b`` ends inside a character)."""
    for k in range(1, min(4, len(b)) + 1):
        if not 0x80 <= b[-k] <= 0xBF:                # ASCII or lead byte of the last character
            try:
                return b[-k:].decode("utf-8")
            except UnicodeDecodeError:
                return ""
    return ""


def _separator(ch: str) -> bool:
    return bool(ch) and (ch.isspace() or unicodedata.category(ch).startswith("P"))


class WordTracker:
    """Which token starts the current word, from token bytes of the text so far only. Agrees with the gold
    ``word_initial`` labels the simulator uses (``boundaries.token_labels``), but never looks ahead and
    never fails on invalid UTF-8 (drafts can contain it)."""

    def __init__(self, tok):
        self.tok, self._b, self.spec = tok, {}, special_ids(tok)

    def bytes(self, i: int) -> bytes:
        i = int(i)
        if i not in self._b:
            self._b[i] = token_bytes_list(self.tok, [i])[0]
        return self._b[i]

    def is_initial(self, ids, j: int) -> bool:
        if j <= 0 or int(ids[j]) in self.spec or int(ids[j - 1]) in self.spec:
            return True
        b = self.bytes(ids[j])
        if not b or 0x80 <= b[0] <= 0xBF:            # empty, or continues a character
            return False
        ch = first_char(b)
        if _separator(ch):
            return True
        if ch and (unicodedata.category(ch).startswith("M") or ch in ZW):
            return False                             # combining mark: attaches to the previous character
        return _separator(_last_char(b"".join(self.bytes(x) for x in ids[max(0, j - 4):j])))

    def word(self, ids) -> list[int]:
        """Tokens of the word that the last token of ``ids`` belongs to."""
        j = len(ids) - 1
        while j > 0 and not self.is_initial(ids, j):
            j -= 1
        return [int(x) for x in ids[j:]]


# --------------------------------------------------------------------------- #
# decoder
# --------------------------------------------------------------------------- #

@dataclass
class Method:
    """kind: "nodraft" | "entropy" | "heads" | "checked" (``variant`` None: no heads)."""
    name: str
    kind: str
    lam: float = -0.683
    beta: float = BETA
    variant: str | None = "v1"
    tau_h: float = 0.0
    tau_c: float | None = None
    lexicon: bool = False
    ngram: bool = False
    L: int = 6                       # checked: proposal length

    def params(self) -> dict:
        if self.kind == "nodraft":
            return {}
        d = dict(lam=self.lam, beta=self.beta)
        if self.kind == "heads":
            d.update(variant=self.variant, tau_h=self.tau_h, tau_c=self.tau_c, lexicon=self.lexicon)
        if self.kind == "checked":
            d.update(variant=self.variant, tau_h=self.tau_h, ngram=self.ngram, L=self.L)
        return d


@dataclass
class HeadsResult:
    out_ids: list
    rounds: list = field(default_factory=list)  # p, g, accepted, calls, props, props_ok, t_draft, t_verify
    wall: float = 0.0
    prefill: float = 0.0                        # target prefill (the draft reads the prompt in its first call)


class HeadsDecoder:
    def __init__(self, target, draft, tok, eos_ids, device: str = "cuda:0", heads: dict | None = None,
                 lexicon=None, gate=None, cont_vocab=None, gamma_max: int = GMAX, noise: float = 0.0, ngram=None):
        """``heads``: {variant: module}; ``gate``: bool array, token may be continued by the heads
        (``devanagari_mask``); ``noise``: CPU tests only, Gaussian logit noise on the draft (seeded by the
        last token read) so drafts get rejected even when draft and target are the same model."""
        torch = _torch()
        self.torch, self.target, self.draft, self.device = torch, target, draft, device
        self.base, W = get_backbone_and_lm_weight(draft)
        self.W = W.detach()
        self.heads, self.lexicon, self.gate, self.ngram = heads or {}, lexicon, gate, ngram
        self.words = WordTracker(tok)
        self.eos = set(int(i) for i in eos_ids)
        self.gamma_max, self.noise = gamma_max, float(noise)
        if self.heads:
            self.cv = [int(i) for i in (continuation_vocab(tok) if cont_vocab is None else cont_vocab)]
            self.Wcv = self.W[torch.tensor(self.cv, device=self.W.device)]
            self.K = next(iter(self.heads.values())).K

    # ---- model calls ---------------------------------------------------- #
    def _draft_call(self, ids: list[int], cache, heads=None, F=()):
        """Feed ``ids`` and the proposal ``F`` to the draft. Returns (n, token, entropy, props): the draft's own
        greedy continuation is F[:n] + [token]; entropy and props [(token, log p, p_cont)] * K (None without
        heads) belong to ``token``. Everything crosses to the CPU in one transfer."""
        torch = self.torch
        fed = list(ids) + list(F)
        x = torch.tensor([fed], device=self.device)
        with torch.no_grad():
            h = self.base(input_ids=x, past_key_values=cache, use_cache=True).last_hidden_state[0, -1 - len(F):]
            lg = lm_logits(h, self.W)
            if self.noise:
                lg = lg + self.noise * torch.stack([
                    torch.randn(lg.shape[1], generator=torch.Generator().manual_seed(1_000_003 + int(t)))
                    for t in fed[-1 - len(F):]]).to(lg.device)
            parts = []
            if F:
                am = lg.argmax(-1)
                n = (am[:-1] == x[0, -len(F):]).long().cumprod(0).sum()      # tokens of F the draft agrees with
                h, lg = h[n][None], lg[n]
                parts.append(n[None].float())
            else:
                lg = lg[0]
            lp = torch.log_softmax(lg, -1)
            parts += [lp.argmax(-1)[None].float(), -(lp.exp() * lp).sum(-1)[None]]
            if heads is not None:
                hf = h.float()
                z = torch.cat([heads.z(hf, k) for k in range(1, self.K + 1)], 0)
                top = torch.log_softmax(lm_logits(z, self.Wcv), -1).max(-1)
                parts += [top.indices.float(), top.values, torch.sigmoid(heads.cont(hf))[0]]
            v = torch.cat(parts).tolist()            # ids < 2^24 are exact in float32
        n = int(v.pop(0)) if F else 0
        if heads is None:
            return n, int(v[0]), v[1], None
        K = self.K
        props = [(self.cv[int(v[2 + k])], v[2 + K + k], v[2 + 2 * K + k]) for k in range(K)]
        return n, int(v[0]), v[1], props

    def _verify(self, ids: list[int], cache) -> list[int]:
        x = self.torch.tensor([ids], device=self.device)
        with self.torch.no_grad():
            return self.target(input_ids=x, past_key_values=cache, use_cache=True).logits[0].argmax(-1).tolist()

    # ---- drafting one round ---------------------------------------------- #
    def _propose(self, m: Method, text: list[int], props, room: int) -> list[int]:
        """Head proposals to append after ``text`` (output so far, ending in the 1B's drafted token)."""
        if room <= 0 or (self.gate is not None and not self.gate[text[-1]]):
            return []
        word = self.words.word(text) if m.lexicon else None
        log_tau = math.log(m.tau_h) if m.tau_h > 0 else -math.inf
        out: list[int] = []
        for x, lp, pc in props[:room]:
            if m.lexicon and not self.lexicon.is_prefix(word + out + [x]):
                break
            if lp < log_tau or (m.tau_c is not None and pc < m.tau_c):
                break
            out.append(x)
        return out

    def _checked_round(self, m: Method, seq: list[int], n_prompt: int, cache, cap: int):
        """Returns (drafted, proposed tokens read, proposed tokens the 1B agreed with, 1B calls)."""
        heads = self.heads[m.variant] if m.variant else None
        drafted: list[int] = []
        props, calls, read, agreed = None, 0, 0, 0
        while len(drafted) < cap:
            cur = seq + drafted
            text = cur[n_prompt:]
            room = cap - len(drafted) - 1
            head = self._propose(m, text, props, room) if props else []
            F = proposal(text, head, self.ngram if m.ngram else None, m.L, room)
            n, tok, H, props = self._draft_call(cur[cache.get_seq_length():], cache, heads, F)
            if n < len(F):
                crop_to(cache, len(cur) + n)          # forget the rejected part of the proposal
            calls, read, agreed = calls + 1, read + len(F), agreed + n
            new = F[:n] + [tok]
            end = next((i + 1 for i, t in enumerate(new) if t in self.eos), None)
            drafted += new[:end]
            if end is not None or entropy_fires(H, m.lam, m.beta):
                break
        return drafted, read, agreed, calls

    def _draft_round(self, m: Method, seq: list[int], n_prompt: int, cache, cap: int):
        drafted: list[int] = []
        from_head: list[bool] = []
        calls = 0
        heads = self.heads[m.variant] if m.kind == "heads" else None
        while len(drafted) < cap:
            cur = seq + drafted
            _, tok, H, props = self._draft_call(cur[cache.get_seq_length():], cache, heads)
            calls += 1
            drafted.append(tok)
            from_head.append(False)
            if tok in self.eos or entropy_fires(H, m.lam, m.beta) or len(drafted) >= cap:
                break
            if heads is not None:
                extra = self._propose(m, seq[n_prompt:] + drafted, props, cap - len(drafted))
                drafted += extra
                from_head += [True] * len(extra)
        return drafted, from_head, calls

    # ---- decoding --------------------------------------------------------- #
    def decode(self, prompt_ids: list[int], m: Method, max_new: int) -> HeadsResult:
        from transformers import DynamicCache
        torch = self.torch
        tc, dc = DynamicCache(), DynamicCache()
        res = HeadsResult(out_ids=[])
        seq, n_prompt = list(prompt_ids), len(prompt_ids)
        _sync()
        t_start = time.perf_counter()
        if len(seq) > 1:                              # target prefill; the pending token is prompt[-1]
            with torch.no_grad():
                self.target(input_ids=torch.tensor([seq[:-1]], device=self.device), past_key_values=tc,
                            use_cache=True, logits_to_keep=1)
        _sync()
        res.prefill = time.perf_counter() - t_start
        while True:
            p = len(seq) - n_prompt
            t0 = time.perf_counter()
            drafted, from_head, calls = ([], [], 0)
            read = agreed = 0
            cap = min(self.gamma_max, max_new - p)
            if m.kind == "checked" and cap > 0:
                drafted, read, agreed, calls = self._checked_round(m, seq, n_prompt, dc, cap)
            elif m.kind != "nodraft" and cap > 0:
                drafted, from_head, calls = self._draft_round(m, seq, n_prompt, dc, cap)
            _sync()
            t1 = time.perf_counter()
            am = self._verify([seq[-1]] + drafted, tc)
            a = 0
            while a < len(drafted) and am[a] == drafted[a]:
                a += 1
            new = drafted[:a] + [am[a]]
            _sync()
            t2 = time.perf_counter()
            seq += new
            crop_to(tc, len(seq) - 1)
            if m.kind != "nodraft":
                crop_to(dc, len(seq) - 1)
            res.rounds.append(dict(p=p, g=len(drafted), accepted=a, calls=calls, props=sum(from_head) + read,
                                   props_ok=sum(from_head[:a]) + agreed, t_draft=t1 - t0, t_verify=t2 - t1))
            if any(t in self.eos for t in new) or len(seq) - n_prompt >= max_new:
                break
        _sync()
        res.wall = time.perf_counter() - t_start
        out = seq[n_prompt:]
        cut = next((i + 1 for i, t in enumerate(out) if t in self.eos), len(out))
        res.out_ids = out[:min(cut, max_new)]
        return res


# --------------------------------------------------------------------------- #
# tuned parameters (built into the notebook) and methods
# --------------------------------------------------------------------------- #

TABLE_ROWS = {"V0": ("Heads V0 (cv vocab)", "v0", False), "V1": ("Heads V1 (cv vocab)", "v1", False),
              "V2": ("Heads V2 (cv vocab)", "v2", False),
              "V0+lex": ("Heads V0 (cont. vocab + lexicon filter)", "v0", True),
              "V1+lex": ("Heads V1 (cont. vocab + lexicon filter)", "v1", True)}


# name: (heads variant, n-gram); parameters from results/checked/params.json (scripts/run_checked_sim.py)
CHECKED_ROWS = {"NGram": (None, True), "V0+check": ("v0", False), "V1+check": ("v1", False),
                "NGram+V0": ("v0", True), "NGram+V1": ("v1", True)}


def variant_of(name: str) -> str | None:
    return TABLE_ROWS[name][1] if name in TABLE_ROWS else CHECKED_ROWS.get(name, (None,))[0]


def tuned_params(main_table_csv) -> dict:
    """{method: {lam, tau_h, tau_c, sim}} from a main_table.csv of ``scripts/run_refund_sim.py`` (dev-tuned;
    ``sim`` is the simulated devtest speed-up, for reference)."""
    rows = {r["method"]: r for r in csv.DictReader(open(main_table_csv, encoding="utf-8"))}
    ent = rows["Entropy (Phase 1, lam=-0.683)"]
    out = {"Entropy": dict(lam=json.loads(ent["params"])["lam"], sim=float(ent["speedup"]))}
    for name, (row, _, _) in TABLE_ROWS.items():
        p = json.loads(rows[row]["params"])
        out[name] = dict(lam=p["lam"], tau_h=p["tau_h"], tau_c=p["tau_c"], sim=float(rows[row]["speedup"]))
    return out


def build_methods(names, params: dict) -> list[Method]:
    out = []
    for n in names:
        if n == "NoDraft":
            out.append(Method(n, "nodraft"))
        elif n == "Entropy":
            out.append(Method(n, "entropy", lam=params[n]["lam"]))
        elif n in CHECKED_ROWS:
            (v, ng), p = CHECKED_ROWS[n], params[n]
            out.append(Method(n, "checked", lam=p["lam"], variant=v, tau_h=p["tau_h"], ngram=ng, L=p["L"]))
        else:
            _, v, lex = TABLE_ROWS[n]
            p = params[n]
            out.append(Method(n, "heads", lam=p["lam"], variant=v, tau_h=p["tau_h"], tau_c=p["tau_c"], lexicon=lex))
    return out


# --------------------------------------------------------------------------- #
# inputs: heads, word list, earlier run
# --------------------------------------------------------------------------- #

def _n_train(d: Path) -> int:
    """Training pairs of a heads folder (run_info.json, else estimated from the step count)."""
    try:
        return int(json.loads((d / "run_info.json").read_text(encoding="utf-8"))["cfg"]["N_TRAIN"])
    except Exception:  # noqa: BLE001
        try:
            return 32 * int(json.loads((d / "train_info.json").read_text(encoding="utf-8"))["steps"])
        except Exception:  # noqa: BLE001
            return 0


def find_heads(search, variants, work: Path, log=print) -> Path:
    """Folder with ``wordheads_<v>.pt`` for every variant; with several, the one trained on most pairs.
    Zips holding such files are unpacked into ``work``."""
    need = [f"wordheads_{v}.pt" for v in variants]
    keep = set(need) | {"run_info.json", "train_info.json"}
    cands: list[Path] = []
    for root in map(Path, search):
        if not root.exists():
            continue
        for f in sorted(root.rglob(need[0])):
            if all((f.parent / n).exists() for n in need):
                cands.append(f.parent)
        for i, zp in enumerate(sorted([root] if root.is_file() else root.rglob("*.zip"))):
            try:
                with zipfile.ZipFile(zp) as z:
                    members = {Path(n.replace("\\", "/")).name: n for n in z.namelist()}
                    if not set(need) <= set(members):
                        continue
                    dst = Path(work) / f"{zp.stem}_{i}"
                    dst.mkdir(parents=True, exist_ok=True)
                    for name in keep & set(members):
                        (dst / name).write_bytes(z.read(members[name]))
                    cands.append(dst)
            except (zipfile.BadZipFile, OSError):
                continue
    if not cands:
        raise FileNotFoundError(f"no {need} under {list(map(str, search))}: attach the output of notebook 07b "
                                "(wordheads.zip, or the notebook itself via Add Input -> Your Work)")
    best = max(cands, key=_n_train)
    for c in cands:
        log(f"  heads found: {c} (N_TRAIN = {_n_train(c)}){'  <- used' if c is best else ''}")
    return best


def find_file(search, name: str) -> Path | None:
    for root in map(Path, search):
        if root.exists():
            for f in sorted(root.rglob(name)):
                if f.is_file():
                    return f
    return None


def build_lexicon(tok, dev_out_ids, wordlist_csv) -> Lexicon:
    """Same words as ``refund_sim.Lexicon.build``: words of the 8B's dev outputs + the word list."""
    freq = Counter()
    for ids in dev_out_ids:
        freq.update(DEV_RE.findall(tok.decode(list(ids), skip_special_tokens=True).strip()))
    if wordlist_csv:
        for i, line in enumerate(open(wordlist_csv, encoding="utf-8")):
            w = line.split(",")[0].strip()
            if i and w and w not in freq and DEV_RE.fullmatch(w):
                freq[w] = 1
    return Lexicon(freq, tok)


def read_records(path) -> list[dict]:
    """jsonl records; lines cut off by a killed run are dropped."""
    out = []
    if Path(path).exists():
        for line in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def resume_files(search, out_dir: Path, log=print) -> None:
    """Copy the finished decodes of an earlier run (a ``heads_loop`` folder or zip in the inputs)."""
    names = ("greedy_unbatched.jsonl", "loop_results.jsonl", "timing.json")
    for root in map(Path, search):
        if not root.exists():
            continue
        src = next((f.parent for f in sorted(root.rglob("loop_results.jsonl")) if f.parent.name == "heads_loop"), None)
        zp = None if src else next((z for z in sorted(root.rglob("heads_loop.zip"))), None)
        for n in names:
            dst = out_dir / n
            if dst.exists():
                continue
            try:
                if src and (src / n).exists():
                    dst.write_bytes((src / n).read_bytes())
                elif zp:
                    with zipfile.ZipFile(zp) as z:
                        dst.write_bytes(z.read(n))
                else:
                    continue
                log(f"  resume: {n} from {src or zp}")
            except (KeyError, zipfile.BadZipFile):
                continue


def _append(path, rec) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #

def summarize_loop(results: list[dict], names: list[str], sim: dict | None = None, n_boot: int = 1000) -> list[dict]:
    """Per method: wall-clock speed-up over NoDraft (ratio of sums of per-sentence medians over repeats, as in
    Phase 1), the same without the prefill, bootstrap CI over sentences, paired difference to Entropy, and
    round statistics. Only sentences with at least one decode of every method count."""
    from .stats import bootstrap_ci, paired_bootstrap
    by: dict = {}
    for r in results:
        by.setdefault(r["method"], {}).setdefault(r["id"], []).append(r)
    ids = sorted(set.intersection(*[set(by.get(n, {})) for n in names])) if names else []
    if not ids or "NoDraft" not in names:
        return []
    W = {n: np.array([np.median([r["wall"] for r in by[n][i]]) for i in ids]) for n in names}
    D = {n: np.array([np.median([r["wall"] - r["prefill"] for r in by[n][i]]) for i in ids]) for n in names}
    N = {n: np.array([by[n][i][0]["n_out"] for i in ids]) for n in names}
    rows = []
    for n in names:
        ci = bootstrap_ci(W["NoDraft"], W[n], n_boot=n_boot)
        cd = bootstrap_ci(D["NoDraft"], D[n], n_boot=n_boot)
        rs = [r for i in ids for r in by[n][i]]
        rd = [x for r in rs for x in r["rounds"]]
        s = lambda k: sum(x[k] for x in rd)  # noqa: E731
        row = dict(method=n, sentences=len(ids), repeats=max(len(by[n][i]) for i in ids),
                   speedup_wall=ci["point"], lo=ci["lo"], hi=ci["hi"],
                   speedup_decode_only=cd["point"], decode_lo=cd["lo"], decode_hi=cd["hi"],
                   speedup_tokens_per_s=float((N[n].sum() / W[n].sum()) / (N["NoDraft"].sum() / W["NoDraft"].sum())),
                   lossless=float(np.mean([r["lossless"] for r in rs])),
                   tokens_per_round=sum(r["n_out"] for r in rs) / max(len(rd), 1),
                   drafted_per_round=s("g") / max(len(rd), 1), draft_calls_per_round=s("calls") / max(len(rd), 1),
                   head_tokens_per_round=s("props") / max(len(rd), 1),
                   head_tokens_accepted=s("props_ok") / max(s("props"), 1) if s("props") else None,
                   sim_devtest=(sim or {}).get(n, {}).get("sim"))
        if "Entropy" in names and n not in ("Entropy", "NoDraft"):
            d = paired_bootstrap(W["NoDraft"], W[n], W["NoDraft"], W["Entropy"], n_boot=n_boot)
            row.update(diff_vs_entropy=d["diff"], diff_lo=d["lo"], diff_hi=d["hi"])
        rows.append(row)
    return rows


def summary_md(rows: list[dict], info: dict) -> str:
    f = lambda x, k=3: "" if x is None else f"{x:.{k}f}"  # noqa: E731
    L = ["# Real GPU loop with word-completion heads", "",
         f"{rows[0]['sentences']} devtest sentences, {rows[0]['repeats']} repeats, "
         f"heads: {info.get('heads_tag')} pairs. Wall-clock, NoDraft = 1. `sim` is the simulated speed-up on all "
         "1,012 devtest sentences (simulation has no prefill, so compare it with `decode only`).", "",
         "| method | wall-clock [95% CI] | decode only | Δ vs Entropy (wall) [95% CI] | sim | lossless | tokens/round | "
         "1B calls/round | proposed tokens/round | proposed tokens accepted |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = (f"{r['diff_vs_entropy']:+.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}]" if "diff_vs_entropy" in r else "")
        L.append(f"| {r['method']} | {f(r['speedup_wall'])} [{f(r['lo'])}, {f(r['hi'])}] | {f(r['speedup_decode_only'])} | {d} "
                 f"| {f(r['sim_devtest'])} | {f(r['lossless'], 2)} | {f(r['tokens_per_round'])} | {f(r['draft_calls_per_round'])} "
                 f"| {f(r['head_tokens_per_round'])} | {f(r['head_tokens_accepted'])} |")
    L += ["", "`lossless`: share of decodes identical to the 8B's unbatched greedy output. NoDraft is also below 1 "
          "(fp16 near-ties), so compare with NoDraft, not with 1. Phase 1: 0.95–0.96 for every method.", "",
          "`proposed tokens`: for V0/V1/V2 (and +lex) the heads' tokens, accepted by the 8B; for the checked methods "
          "(V*+check, NGram*) tokens read in a 1B call, accepted by the 1B."]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# notebook 09 entry point (also scripts/smoke_cpu.py --part wl)
# --------------------------------------------------------------------------- #

DEFAULT_WL = dict(
    METHODS=["NoDraft", "Entropy", "V1+lex", "V1+check", "NGram", "NGram+V0"],
    N_SENT=100, REPEATS=3, MAX_NEW=256,
    TARGET=["meta-llama/Meta-Llama-3-8B-Instruct", "NousResearch/Meta-Llama-3-8B-Instruct"],
    DRAFT=["meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"],
    MAX_MEMORY={0: "8GiB", 1: "10GiB"},
    HEADS_DIR=None, WORDLIST=None, NGRAM=None, PARAMS=None,
    SEARCH=["/kaggle/input"], OUT_ROOT="/kaggle/working", BUILD="unknown",
    TIMING=dict(steps=60, reps=3, max_g=8), SMOKE=False, SMOKE_NOISE=0.0,
)


def _first(names, load, log):
    """``load(name)`` for the first name that can be loaded (gated repos fall back to mirrors)."""
    names = [names] if isinstance(names, str) else list(names)
    for i, n in enumerate(names):
        try:
            return load(n), n
        except (OSError, ValueError) as e:
            if i == len(names) - 1:
                raise
            log(f"  could not load {n} ({type(e).__name__}); trying {names[i + 1]}")


def heads_loop_main(cfg_in: dict, records: list[dict], dev_out_ids: list) -> dict:
    """``records``: [{id, prompt_ids}] of the sentences to decode; ``dev_out_ids``: the 8B's dev outputs (lexicon)."""
    torch = _torch()
    from . import gpu_traces as G
    cfg = {**DEFAULT_WL, **cfg_in}
    out_dir = Path(cfg["OUT_ROOT"]) / "heads_loop"
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(str(msg) + "\n")
        logf.flush()

    names = list(cfg["METHODS"])
    assert names[0] == "NoDraft", "METHODS must start with NoDraft (the baseline)"
    unknown = [n for n in names if n not in ("NoDraft", "Entropy", *TABLE_ROWS, *CHECKED_ROWS)]
    assert not unknown, f"unknown methods {unknown}; choose from NoDraft, Entropy, {[*TABLE_ROWS, *CHECKED_ROWS]}"

    # ---- inputs: fail before loading the 8B ---- #
    variants = sorted({variant_of(n) for n in names} - {None})
    use_lex = any(TABLE_ROWS[n][2] for n in names if n in TABLE_ROWS)
    heads_dir, n_train, tag = None, 0, None
    if variants:
        heads_dir = find_heads([cfg["HEADS_DIR"]] if cfg["HEADS_DIR"] else cfg["SEARCH"], variants,
                               Path(cfg["OUT_ROOT"]) / "_heads", log)
        n_train = _n_train(heads_dir)
        tag = "180k" if n_train >= 120000 else "60k"
        log(f"heads: {heads_dir} -> parameters tuned for the {tag} heads")
        if n_train == 0 and not cfg["SMOKE"]:
            log("WARNING: the heads folder has no run_info.json / train_info.json, so 60k parameters are assumed")
    params = cfg["PARAMS"][tag or "180k"]
    methods = build_methods(names, params)
    wordlist = None
    if use_lex:
        wordlist = Path(cfg["WORDLIST"]) if cfg["WORDLIST"] else find_file(cfg["SEARCH"], "hindi.csv")
        if wordlist is None or not wordlist.exists():
            raise FileNotFoundError("hindi.csv (the Hindi word list) not found: upload it as a Kaggle dataset and attach it, or "
                                    "remove the '+lex' methods from METHODS")
    ng_path = None
    if any(m.ngram for m in methods):
        ng_path = Path(cfg["NGRAM"]) if cfg["NGRAM"] else find_file(cfg["SEARCH"], "ngram_hi.npz")
        if ng_path is None or not ng_path.exists():
            raise FileNotFoundError("ngram_hi.npz (the Hindi n-gram table) not found: upload it as a Kaggle dataset and attach it, "
                                    "or remove the 'NGram' methods from METHODS")
    log(f"methods: {[(m.name, m.params()) for m in methods]}")

    # ---- models ---- #
    if cfg["SMOKE"]:
        device, dtype = "cpu", torch.float32
        draft, draft_name = _first(cfg["DRAFT"], lambda n: G.load_draft(n, dtype, device), log)
        target, target_name, tok = draft, draft_name, G.load_tokenizer(draft_name)
    else:
        assert torch.cuda.device_count() >= 2, "Need 2 GPUs: Settings > Accelerator > GPU T4 x2"
        device, dtype = "cuda:0", torch.float16
        target, target_name = _first(cfg["TARGET"], lambda n: G.load_target(n, dtype, max_memory=cfg["MAX_MEMORY"]), log)
        tok = G.load_tokenizer(target_name)
        draft, draft_name = _first(cfg["DRAFT"], lambda n: G.load_draft(n, dtype, device), log)
        G.check_draft_target_compat(draft, target, G.load_tokenizer(draft_name), tok, "परीक्षण")
    eos_ids = G.get_eos_ids(target, tok)
    d = get_backbone_and_lm_weight(draft)[1].shape[1]
    heads = {}
    for v in variants:
        sd = {k: t.float() for k, t in torch.load(heads_dir / f"wordheads_{v}.pt", map_location="cpu").items()}
        hd = make_heads(d, sum(k.startswith("blocks.") and k.endswith(".weight") for k in sd))
        hd.load_state_dict(sd)
        heads[v] = hd.to(device).eval()
    cv = continuation_vocab(tok) if variants else None
    gate = devanagari_mask(tok) if variants else None
    lexicon = build_lexicon(tok, dev_out_ids, wordlist) if use_lex else None
    ngram = NGram.load(ng_path) if ng_path else None
    if ngram is not None:
        log(f"n-gram table: {ng_path}, order {ngram.order}, contexts {[len(a) for a, _ in ngram.tab.values()]}")
    if not cfg["SMOKE"]:
        if cv is not None and len(cv) != PAPER_CONT_VOCAB:
            log(f"WARNING: continuation vocabulary has {len(cv)} tokens, the paper says {PAPER_CONT_VOCAB}")
        if lexicon is not None:
            n_lex = len(lexicon.words)
            log(f"lexicon: {n_lex} words from {wordlist} + dev outputs (paper: {PAPER_LEXICON_TYPES})")
            if n_lex != PAPER_LEXICON_TYPES:
                log("WARNING: the lexicon size differs from the paper's; is this the same hindi.csv?")
    dec = HeadsDecoder(target, draft, tok, eos_ids, device, heads, lexicon, gate, cv, noise=cfg["SMOKE_NOISE"],
                       ngram=ngram)

    recs = list(records)[: cfg["N_SENT"]]
    resume_files(cfg["SEARCH"], out_dir, log)

    # ---- timing of the model calls in this session (cost model) ---- #
    if not (out_dir / "timing.json").exists():
        tm = cfg["TIMING"]
        G.run_timing(draft, target, out_dir / "timing.json", device, dict(build=cfg["BUILD"]), steps=tm["steps"],
                     reps=tm["reps"], max_g=tm["max_g"])

    # ---- unbatched greedy references ---- #
    ref_path = out_dir / "greedy_unbatched.jsonl"
    kept = read_records(ref_path)
    ref_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    refs = {r["id"]: r for r in kept}
    for r in recs:
        if r["id"] not in refs:
            o = G.generate_unbatched(target, tok, r["prompt_ids"], cfg["MAX_NEW"], eos_ids, device)
            refs[r["id"]] = dict(id=r["id"], out_ids=o, out_text=tok.decode(o, skip_special_tokens=True).strip())
            _append(ref_path, refs[r["id"]])
    log(f"greedy references: {len(refs)}")

    # ---- timed runs (resumable; decodes with other parameters or heads are ignored) ---- #
    ng_tag = hashlib.sha256(Path(ng_path).read_bytes()).hexdigest()[:12] if ng_path else None

    def run_key(m):
        d = dict(params=m.params(), heads=n_train if m.kind == "heads" or (m.kind == "checked" and m.variant) else None)
        if m.ngram:
            d["ngram"] = ng_tag
        return json.dumps(d, sort_keys=True)

    key = {m.name: run_key(m) for m in methods}
    res_path = out_dir / "loop_results.jsonl"
    ids = {r["id"] for r in recs}
    results = [r for r in read_records(res_path) if key.get(r["method"]) == r["key"] and r["id"] in ids
               and r["rep"] < cfg["REPEATS"]]
    res_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    done = {(r["rep"], r["id"], r["method"]) for r in results}
    if done:
        log(f"resume: {len(done)} decodes reused")
    for m in methods:                                 # warm-up
        dec.decode(recs[0]["prompt_ids"], m, min(32, cfg["MAX_NEW"]))
    t0, n_new, n_sent = time.perf_counter(), 0, 0
    for rep in range(cfg["REPEATS"]):
        for si, r in enumerate(recs):
            order = methods[(rep + si) % len(methods):] + methods[: (rep + si) % len(methods)]
            fresh = False
            for m in order:
                if (rep, r["id"], m.name) in done:
                    continue
                x = dec.decode(r["prompt_ids"], m, cfg["MAX_NEW"])
                ref = refs[r["id"]]["out_ids"]
                rec = dict(rep=rep, id=r["id"], method=m.name, key=key[m.name], wall=x.wall, prefill=x.prefill,
                           n_out=len(x.out_ids), lossless=x.out_ids == ref,
                           first_divergence=first_divergence(x.out_ids, ref), out_ids=x.out_ids, rounds=x.rounds)
                _append(res_path, rec)
                results.append(rec)
                n_new, fresh = n_new + 1, True
            n_sent += fresh
            if fresh and n_sent == 5:
                per = (time.perf_counter() - t0) / 5
                log(f"ESTIMATE: {per:.1f} s per sentence (all methods) -> about "
                    f"{per * len(recs) * cfg['REPEATS'] / 3600:.1f} h for N_SENT={len(recs)} x REPEATS={cfg['REPEATS']}")
        log(f"repeat {rep} done ({(time.perf_counter() - t0) / 60:.0f} min)")

    rows = summarize_loop(results, names, params)
    info = G.run_info(cfg["BUILD"], methods={m.name: m.params() for m in methods}, n_sent=len(recs),
                      repeats=cfg["REPEATS"], heads_dir=str(heads_dir), heads_n_train=n_train, heads_tag=tag,
                      lexicon_words=len(lexicon.words) if lexicon else None, wordlist=str(wordlist),
                      ngram=str(ng_path) if ng_path else None, ngram_sha256=ng_tag,
                      target=target_name, draft=draft_name, smoke=cfg["SMOKE"],
                      cont_vocab=len(cv) if cv is not None else None, new_decodes=n_new)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    md = summary_md(rows, info)
    (out_dir / "SUMMARY.md").write_text(md, encoding="utf-8")
    log("\n" + md)
    logf.close()
    zp = G.zip_dir(out_dir, Path(cfg["OUT_ROOT"]) / "heads_loop.zip")
    print("DONE. Download heads_loop.zip from the Output tab and unzip it into  results/heads_loop/  in the repo.")
    return dict(zip=zp, rows=rows, info=info)
