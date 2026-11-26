"""Stopping policies. Each reads only through ``simulator.TraceView``.

Causal (lookahead 0) policies read draft features at indices <= p+g-1. Lookahead-1 policies also read
index p+g (the next draft distribution q_{p+g}, one extra draft forward; the simulator charges it
when the policy stops by itself). Oracles (``is_oracle``) may read gold fields; they are upper bounds.

"Extend" semantics (AksharaSafeEntropy, Hybrid, MorphEntropy): let g0 be the first g at which the
entropy rule fired in this round (recomputed from H[p..p+g-1], so there is no hidden state). At every
g >= g0 the policy stops if its linguistic condition holds at g, or once g - g0 >= max_ext.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .boundaries import END, WORD
from .simulator import Policy, TraceView


def entropy_fires(H: float, lam: float, beta: float) -> bool:
    """AdaEDL/SVIP-style rule: stop when 1 - sqrt(beta * H) < lam."""
    return 1.0 - math.sqrt(beta * max(float(H), 0.0)) < lam


def first_fire(view: TraceView, p: int, g: int, lam: float, beta: float) -> int | None:
    """Smallest g' in 1..g at which the entropy rule fired (reads H[p..p+g-1] only)."""
    for k in range(1, g + 1):
        if entropy_fires(view.feat("H", p + k - 1), lam, beta):
            return k
    return None


class NoDraft(Policy):
    name = "NoDraft"

    def initial_stop(self, view, p):
        return True

    def stop(self, view, p, g):
        return True


class FixedK(Policy):
    name = "FixedK"

    def __init__(self, k: int):
        self.k = int(k)

    def initial_stop(self, view, p):
        return self.k <= 0

    def stop(self, view, p, g):
        return g >= self.k

    def params(self):
        return dict(k=self.k)


class Entropy(Policy):
    name = "Entropy"

    def __init__(self, lam: float, beta: float = 1.0):
        self.lam, self.beta = float(lam), float(beta)

    def stop(self, view, p, g):
        return entropy_fires(view.feat("H", p + g - 1), self.lam, self.beta)

    def params(self):
        return dict(lam=self.lam, beta=self.beta)


class WordBoundarySoft(Policy):
    """Stop when g >= min_g and the next draft distribution puts > tau mass on new-word tokens."""
    name = "WordBoundarySoft"
    lookahead = 1

    def __init__(self, tau: float, min_g: int = 1):
        self.tau, self.min_g = float(tau), int(min_g)

    def stop(self, view, p, g):
        return g >= self.min_g and view.feat("p_newword", p + g) > self.tau

    def params(self):
        return dict(tau=self.tau, min_g=self.min_g)


class WordBoundaryHard(Policy):
    """Corrected legacy word-boundary policy: stop when g >= min_g and the draft's next token
    (argmax of q_{p+g}) starts with space / punctuation or is eos. Needs the vocabulary mask."""
    name = "WordBoundaryHard"
    lookahead = 1

    def __init__(self, min_g: int, newword_mask: np.ndarray):
        self.min_g = int(min_g)
        self.mask = np.asarray(newword_mask, dtype=bool)

    def stop(self, view, p, g):
        if g < self.min_g:
            return False
        t = view.token(p + g)
        return bool(t < len(self.mask) and self.mask[t])

    def params(self):
        return dict(min_g=self.min_g)


class _EntropyExtend(Policy):
    """Entropy gate + linguistic condition at the lookahead position, with up to max_ext extensions."""
    lookahead = 1

    def __init__(self, lam: float, beta: float, max_ext: int):
        self.lam, self.beta, self.max_ext = float(lam), float(beta), int(max_ext)

    def cond(self, view: TraceView, p: int, g: int) -> bool:
        raise NotImplementedError

    def stop(self, view, p, g):
        g0 = first_fire(view, p, g, self.lam, self.beta)
        if g0 is None:
            return False
        return self.cond(view, p, g) or (g - g0) >= self.max_ext


class AksharaSafeEntropy(_EntropyExtend):
    """Entropy rule, but only stop where the next token is unlikely to be a combining mark
    (p_mark[p+g] < tau_m), i.e. the last akshara is probably complete."""
    name = "AksharaSafeEntropy"

    def __init__(self, lam: float, beta: float, tau_m: float, max_ext: int):
        super().__init__(lam, beta, max_ext)
        self.tau_m = float(tau_m)

    def cond(self, view, p, g):
        return view.feat("p_mark", p + g) < self.tau_m

    def params(self):
        return dict(lam=self.lam, beta=self.beta, tau_m=self.tau_m, max_ext=self.max_ext)


class Hybrid(_EntropyExtend):
    """Entropy rule, but only stop where the current word is probably complete (p_newword[p+g] > tau_w)."""
    name = "Hybrid"

    def __init__(self, lam: float, beta: float, tau_w: float, max_ext: int):
        super().__init__(lam, beta, max_ext)
        self.tau_w = float(tau_w)

    def cond(self, view, p, g):
        return view.feat("p_newword", p + g) > self.tau_w

    def params(self):
        return dict(lam=self.lam, beta=self.beta, tau_w=self.tau_w, max_ext=self.max_ext)


class Oracle(Policy):
    """Upper bound: drafts exactly the leading run of matches (can draft 0), never rejects."""
    name = "Oracle"
    is_oracle = True

    def initial_stop(self, view, p):
        return not view.gold("match", p)

    def stop(self, view, p, g):
        return not view.gold("match", p + g)


class GoldWordOracle(Policy):
    """Upper bound for word-boundary stopping: stop (g >= min_g) when the *gold* cut after token
    p+g-1 of the target is a word boundary. Free lookahead; needs ``attach_word_labels``."""
    name = "GoldWordOracle"
    is_oracle = True

    def __init__(self, min_g: int = 1):
        self.min_g = int(min_g)

    def stop(self, view, p, g):
        return g >= self.min_g and view.gold("end_cut", p + g - 1) in (WORD, END)

    def params(self):
        return dict(min_g=self.min_g)


# --------------------------------------------------------------------------- #
# T6: morpheme policies
# --------------------------------------------------------------------------- #

@dataclass
class TokenTable:
    """Per-vocabulary-id raw bytes and a word-start mask (space / punctuation / special)."""
    token_bytes: list
    word_start: np.ndarray

    @classmethod
    def from_tokenizer(cls, tok, masks=None) -> "TokenTable":
        from .vocab import build_masks, token_bytes_list
        masks = masks or build_masks(tok)
        tb = token_bytes_list(tok, range(len(tok)))
        return cls(tb, masks.starts_space | masks.starts_punct | masks.is_special)


def _complete_utf8(b: bytes) -> str:
    for cut in range(len(b), max(len(b) - 4, -1), -1):
        try:
            return b[:cut].decode("utf-8")
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="ignore")


def current_word_prefix(view: TraceView, p: int, g: int, table: TokenTable, max_back: int = 32) -> str:
    """Text of the current word after drafting g tokens: committed target tokens before p plus drafted
    tokens p..p+g-1, scanning back to the last token that starts a word (that token's leading space
    is stripped). Only complete UTF-8 characters are decoded."""
    end = p + g - 1
    parts = []
    lo = max(0, end - max_back)
    for idx in range(end, lo - 1, -1):
        t = view.token(idx)
        b = table.token_bytes[t] if t < len(table.token_bytes) else b""
        parts.append(b)
        if t < len(table.word_start) and table.word_start[t]:
            break
    raw = b"".join(reversed(parts)).lstrip(b" \n\t")
    return _complete_utf8(raw)


def morph_prob(view, p: int, g: int, pm, table: TokenTable) -> float:
    """pm(current word prefix) after drafting g tokens from p.

    In the simulator the value at (trace, p, g) is the same for every config of every morph policy, so it
    is cached on the trace object (keyed by the prefix model's unique ``uid``). It is computed the first time through the
    causality-checked view, and (p, g) fixes exactly what the view allows, so reuse cannot leak
    information. Live views (no ``trace``) are never cached."""
    tr = getattr(view, "trace", None)
    if tr is None:
        return pm.prob(current_word_prefix(view, p, g, table))
    cache = tr.__dict__.setdefault("_pm_cache", {})
    key = (getattr(pm, "uid", id(pm)), p, g)
    v = cache.get(key)
    if v is None:
        v = cache[key] = pm.prob(current_word_prefix(view, p, g, table))
    return v


class MorphEntropy(_EntropyExtend):
    """Entropy rule; stop where the causal prefix model predicts a morpheme boundary (pm > tau_morph).
    If use_akshara (lookahead 1): the condition also holds when the word is complete
    (p_newword[p+g] > 0.5), and additionally requires p_mark[p+g] < tau_m.
    Without use_akshara the policy is causal (lookahead 0). Extends up to max_ext tokens."""
    name = "MorphEntropy"

    def __init__(self, lam: float, beta: float, tau_morph: float, use_akshara: bool, tau_m: float,
                 max_ext: int, prefix_model, table: TokenTable):
        super().__init__(lam, beta, max_ext)
        self.tau_morph, self.use_akshara, self.tau_m = float(tau_morph), bool(use_akshara), float(tau_m)
        self.pm, self.table = prefix_model, table
        self.lookahead = 1 if self.use_akshara else 0

    def cond(self, view, p, g):
        ok = morph_prob(view, p, g, self.pm, self.table) > self.tau_morph
        if self.use_akshara:
            ok = (ok or view.feat("p_newword", p + g) > 0.5) and view.feat("p_mark", p + g) < self.tau_m
        return ok

    def params(self):
        return dict(lam=self.lam, beta=self.beta, tau_morph=self.tau_morph, use_akshara=self.use_akshara,
                    tau_m=self.tau_m, max_ext=self.max_ext)


class MorphOnly(Policy):
    """Pure morpheme policy (no entropy, causal): stop when g >= min_g and pm > tau_morph."""
    name = "MorphOnly"

    def __init__(self, tau_morph: float, min_g: int, prefix_model, table: TokenTable):
        self.tau_morph, self.min_g = float(tau_morph), int(min_g)
        self.pm, self.table = prefix_model, table

    def stop(self, view, p, g):
        return g >= self.min_g and morph_prob(view, p, g, self.pm, self.table) > self.tau_morph

    def params(self):
        return dict(tau_morph=self.tau_morph, min_g=self.min_g)
