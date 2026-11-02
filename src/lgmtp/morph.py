"""Morpheme labelling (PLAN T3, fixes B1): MorphTok loader, gold lexicon, CRF segmenter,
tiered labeller, causal prefix-boundary model, per-token gold morph labels.

Conventions
- All strings are NFC.
- A word's morpheme boundaries are *internal* char offsets ``0 < b < len(word)``.
- MorphTok segmentations include sandhi splitting; only entries whose pieces re-join to the word
  are used. Words with several segmentations get the **union** of their boundaries.
- Gold boundaries that fall inside an akshara (e.g. ``निकाल+ेगी``) are **snapped left** to the start
  of that akshara (``निका|लेगी``); a boundary that would snap to 0 is dropped. (User decision.)
- Words for labelling target text are maximal runs of characters that are neither whitespace nor
  punctuation (consistent with ``boundaries.is_word_boundary``).
"""
from __future__ import annotations

import functools
import gzip
import hashlib
import pickle
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .boundaries import END, MIDWORD, WORD, akshara_units

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def is_devanagari_word(w: str) -> bool:
    return bool(DEVANAGARI_RE.search(w))


def akshara_boundaries(word: str) -> set[int]:
    """Internal akshara-unit boundaries of ``word``."""
    return {e for _, e in akshara_units(word)[:-1]}


def snap_to_akshara(word: str, bounds: Iterable[int]) -> tuple[set[int], int]:
    """Snap each boundary inside an akshara left to the akshara start. Returns (bounds, n_snapped)."""
    units = akshara_units(word)
    out, snapped = set(), 0
    for b in bounds:
        for s, e in units:
            if s < b < e:
                snapped += 1
                b = s
                break
        if 0 < b < len(word):
            out.add(b)
    return out, snapped


def seg_boundaries(seg: str) -> set[int]:
    out, pos = set(), 0
    for p in seg.split("+")[:-1]:
        pos += len(p)
        out.add(pos)
    return out


def iter_words(text: str) -> list[tuple[int, int]]:
    """(start, end) char spans of words: maximal runs of non-space, non-punctuation chars."""
    spans, i, n = [], 0, len(text)
    while i < n:
        if text[i].isspace() or unicodedata.category(text[i]).startswith("P"):
            i += 1
            continue
        j = i
        while j < n and not (text[j].isspace() or unicodedata.category(text[j]).startswith("P")):
            j += 1
        spans.append((i, j))
        i = j
    return spans


# --------------------------------------------------------------------------- #
# 1. MorphTok loader
# --------------------------------------------------------------------------- #

def load_morphtok(path) -> tuple["object", dict]:
    """Read MorphTok csv (columns word, segmentation, language), NFC-normalise, mark entries whose
    pieces re-join to the word. Returns (DataFrame with column ``kept``, info dict)."""
    import pandas as pd
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = ["word", "segmentation", "language"]
    if list(df.columns) != expected:
        raise ValueError(f"unexpected MorphTok columns {list(df.columns)}; expected {expected}")
    df["word"] = df["word"].map(nfc)
    df["segmentation"] = df["segmentation"].map(nfc)
    df["kept"] = df["segmentation"].str.replace("+", "", regex=False) == df["word"]
    dropped = df[~df["kept"]]
    info = dict(path=str(path.name), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rows=int(len(df)), unique_words=int(df["word"].nunique()),
                kept_rows=int(df["kept"].sum()), dropped_rows=int((~df["kept"]).sum()),
                dropped_examples=[f"{w} -> {s}" for w, s in zip(dropped["word"][:10], dropped["segmentation"][:10])],
                languages=dict(Counter(df["language"])))
    return df, info


# --------------------------------------------------------------------------- #
# 2. Gold lexicon
# --------------------------------------------------------------------------- #

@dataclass
class GoldLexicon:
    bounds: dict[str, frozenset] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    @classmethod
    def from_dataframe(cls, df) -> "GoldLexicon":
        raw: dict[str, set] = defaultdict(set)
        nseg: Counter = Counter()
        for w, s in zip(df.loc[df["kept"], "word"], df.loc[df["kept"], "segmentation"]):
            raw[w] |= seg_boundaries(s)
            nseg[w] += 1
        out, snapped, total = {}, 0, 0
        for w, b in raw.items():
            total += len(b)
            sb, k = snap_to_akshara(w, b)
            snapped += k
            out[w] = frozenset(sb)
        stats = dict(words=len(out), words_with_multiple_segmentations=sum(1 for v in nseg.values() if v > 1),
                     raw_internal_boundaries=total, boundaries_inside_akshara_snapped=snapped,
                     final_internal_boundaries=sum(len(v) for v in out.values()))
        return cls(out, stats)

    def __contains__(self, w: str) -> bool:
        return w in self.bounds

    def get(self, w: str):
        return self.bounds.get(w)


# --------------------------------------------------------------------------- #
# 3. CRF segmenter
# --------------------------------------------------------------------------- #

def _char_features(word: str, i: int, aks_start: set[int], aks_end: set[int], pos_in_aks: list[int]) -> dict:
    ch = word[i]
    n = len(word)
    f = {
        "bias": 1.0, "c": ch, "cat": unicodedata.category(ch),
        "aks_start": i in aks_start, "aks_end": (i + 1) in aks_end, "pos_in_aks": pos_in_aks[i],
        "len": min(n, 15), "from_start": min(i, 8), "from_end": min(n - 1 - i, 8),
    }
    for d in range(-3, 4):  # char n-grams in a ±3 window
        j = i + d
        f[f"c[{d}]"] = word[j] if 0 <= j < n else ("<s>" if j < 0 else "</s>")
    for a, b in [(-2, 0), (-1, 1), (0, 2), (1, 3), (-3, 0), (0, 3)]:
        f[f"ng[{a}:{b}]"] = "".join(word[j] if 0 <= j < n else "_" for j in range(i + a, i + b + 1))
    return f


def word2features(word: str) -> list[dict]:
    units = akshara_units(word)
    aks_start = {s for s, _ in units}
    aks_end = {e for _, e in units}
    pos_in_aks = [0] * len(word)
    for s, e in units:
        for k in range(s, e):
            pos_in_aks[k] = min(k - s, 4)
    return [_char_features(word, i, aks_start, aks_end, pos_in_aks) for i in range(len(word))]


def word2labels(word: str, bounds: Iterable[int]) -> list[str]:
    """Tag of char i is 'B' if there is a boundary after it (offset i+1)."""
    b = set(bounds)
    return ["B" if (i + 1) in b else "O" for i in range(len(word))]


class CRFSegmenter:
    """Character-level CRF predicting 'boundary after this char'; predictions are post-processed so
    boundaries only fall on akshara boundaries."""

    def __init__(self, c1: float = 0.1, c2: float = 0.05, max_iterations: int = 150):
        import sklearn_crfsuite
        self.crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iterations,
                                        all_possible_transitions=True)

    def fit(self, words: Sequence[str], bounds: Sequence[Iterable[int]]) -> "CRFSegmenter":
        self.crf.fit([word2features(w) for w in words], [word2labels(w, b) for w, b in zip(words, bounds)])
        return self

    def predict_raw(self, word: str) -> set[int]:
        tags = self.crf.predict_single(word2features(word))
        return {i + 1 for i, t in enumerate(tags) if t == "B" and i + 1 < len(word)}

    def predict(self, word: str) -> set[int]:
        if not word:
            return set()
        return self.predict_raw(word) & akshara_boundaries(word)

    def predict_many(self, words: Sequence[str]) -> list[set[int]]:
        tags = self.crf.predict([word2features(w) for w in words])
        out = []
        for w, ts in zip(words, tags):
            raw = {i + 1 for i, t in enumerate(ts) if t == "B" and i + 1 < len(w)}
            out.append(raw & akshara_boundaries(w))
        return out


def split_by_type(words: Sequence[str], frac_test: float = 0.1, seed: int = 0) -> tuple[list[str], list[str]]:
    ws = sorted(set(words))
    random.Random(seed).shuffle(ws)
    k = int(round(len(ws) * frac_test))
    return ws[k:], ws[:k]


def boundary_prf(gold: Sequence[set], pred: Sequence[set]) -> dict:
    tp = sum(len(g & p) for g, p in zip(gold, pred))
    np_ = sum(len(p) for p in pred)
    ng = sum(len(g) for g in gold)
    P = tp / np_ if np_ else float("nan")
    R = tp / ng if ng else float("nan")
    F = 2 * P * R / (P + R) if np_ and ng and (P + R) else float("nan")
    exact = float(np.mean([g == p for g, p in zip(gold, pred)])) if gold else float("nan")
    return dict(precision=P, recall=R, f1=F, tp=tp, n_pred=np_, n_gold=ng, word_exact_match=exact)


def train_and_eval_crf(lex: GoldLexicon, seed: int = 0, **crf_kw) -> tuple[CRFSegmenter, dict]:
    """Train on 90% of word types, evaluate boundary P/R/F1 on the held-out 10%, then return a model
    trained on the 90% split (the held-out metrics describe exactly the saved model)."""
    import time
    train, test = split_by_type(list(lex.bounds), 0.1, seed)
    t0 = time.time()
    seg = CRFSegmenter(**crf_kw).fit(train, [lex.bounds[w] for w in train])
    t_fit = time.time() - t0
    gold = [set(lex.bounds[w]) for w in test]
    pred = seg.predict_many(test)
    raw = [seg.predict_raw(w) for w in test]
    m = boundary_prf(gold, pred)
    m_raw = boundary_prf(gold, raw)
    # baseline: no boundaries at all / every akshara boundary
    m_all = boundary_prf(gold, [akshara_boundaries(w) for w in test])
    info = dict(seed=seed, n_train_types=len(train), n_test_types=len(test), fit_seconds=t_fit,
                crf_params=dict(c1=seg.crf.c1, c2=seg.crf.c2, max_iterations=seg.crf.max_iterations),
                heldout=m, heldout_before_akshara_postprocess=m_raw,
                baseline_every_akshara_boundary=m_all,
                n_raw_predictions_removed_by_postprocess=sum(len(r - p) for r, p in zip(raw, pred)))
    return seg, info


# --------------------------------------------------------------------------- #
# 4. Tiered labeller
# --------------------------------------------------------------------------- #

class TieredLabeler:
    """label(word) -> (boundaries, tier) with tiers gold -> crf -> none (none: non-Devanagari)."""

    def __init__(self, lex: GoldLexicon, crf: CRFSegmenter | None):
        self.lex, self.crf = lex, crf
        self._cache: dict[str, tuple[frozenset, str]] = {}

    def label(self, word: str) -> tuple[frozenset, str]:
        word = nfc(word)
        r = self._cache.get(word)
        if r is not None:
            return r
        if word in self.lex:
            r = (self.lex.get(word), "gold")
        elif is_devanagari_word(word) and self.crf is not None:
            r = (frozenset(self.crf.predict(word)), "crf")
        else:
            r = (frozenset(), "none")
        self._cache[word] = r
        return r

    def label_text(self, text: str) -> list[tuple[int, int, frozenset, str]]:
        return [(s, e, *self.label(text[s:e])) for s, e in iter_words(text)]

    def coverage(self, texts: Iterable[str]) -> dict:
        """Per-tier coverage over word tokens and word types."""
        tok, types = Counter(), {}
        for t in texts:
            for s, e, _, tier in self.label_text(nfc(t)):
                tok[tier] += 1
                types[t[s:e]] = tier
        typ = Counter(types.values())
        nt, ny = sum(tok.values()), sum(typ.values())
        return dict(word_tokens=nt, word_types=ny,
                    tokens={k: dict(n=tok[k], frac=tok[k] / nt if nt else float("nan")) for k in ("gold", "crf", "none")},
                    types={k: dict(n=typ[k], frac=typ[k] / ny if ny else float("nan")) for k in ("gold", "crf", "none")})


# --------------------------------------------------------------------------- #
# 5. Causal prefix-boundary model
# --------------------------------------------------------------------------- #

class PrefixBoundaryModel:
    """P(internal morpheme boundary right after ``prefix`` | prefix) with backoff:
    full prefix -> last 3 aksharas -> last 2 -> last 1 -> global rate.
    A level is used only if its count >= ``min_count``. Built from akshara-aligned proper prefixes
    of labelled words (MorphTok + dev target outputs only; never devtest)."""

    LEVELS = ("full", "last3", "last2", "last1")

    def __init__(self, min_count: int = 5):
        import uuid
        self.uid = uuid.uuid4().hex   # identifies this model in per-trace caches (never reused)
        self.min_count = min_count
        self.pos: dict[str, Counter] = {k: Counter() for k in self.LEVELS}
        self.tot: dict[str, Counter] = {k: Counter() for k in self.LEVELS}
        self.g_pos = 0.0
        self.g_tot = 0.0
        self.sources: dict = {}

    @staticmethod
    def _keys(prefix: str) -> dict[str, str]:
        units = akshara_units(prefix)
        u = [prefix[s:e] for s, e in units]
        return {"full": prefix, "last3": "".join(u[-3:]), "last2": "".join(u[-2:]), "last1": "".join(u[-1:])}

    def add_word(self, word: str, bounds: Iterable[int], weight: float = 1.0) -> None:
        b = set(bounds)
        for c in sorted(akshara_boundaries(word)):          # proper, akshara-aligned prefixes
            y = c in b
            for lvl, key in self._keys(word[:c]).items():
                self.tot[lvl][key] += weight
                if y:
                    self.pos[lvl][key] += weight
            self.g_tot += weight
            self.g_pos += weight * y

    def fit(self, weighted_words: Iterable[tuple[str, Iterable[int], float]], sources: dict | None = None):
        for w, b, wt in weighted_words:
            self.add_word(w, b, wt)
        self.sources = sources or {}
        self.prob.cache_clear()
        import uuid
        self.uid = uuid.uuid4().hex   # contents changed: invalidate per-trace caches
        return self

    @property
    def global_rate(self) -> float:
        return self.g_pos / self.g_tot if self.g_tot else 0.0

    def level_used(self, prefix: str) -> str:
        if not prefix:
            return "global"
        for lvl, key in self._keys(nfc(prefix)).items():
            if self.tot[lvl][key] >= self.min_count:
                return lvl
        return "global"

    @functools.lru_cache(maxsize=1_000_000)
    def prob(self, prefix: str) -> float:
        if not prefix:
            return self.global_rate
        for lvl, key in self._keys(nfc(prefix)).items():
            n = self.tot[lvl].get(key, 0)
            if n >= self.min_count:
                return float(self.pos[lvl].get(key, 0) / n)
        return self.global_rate

    def __hash__(self):  # lru_cache on a method keys on self
        return id(self)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(min_count=self.min_count, pos={k: dict(v) for k, v in self.pos.items()},
                     tot={k: dict(v) for k, v in self.tot.items()}, g_pos=self.g_pos, g_tot=self.g_tot,
                     sources=self.sources)
        with gzip.open(path, "wb") as fh:
            pickle.dump(state, fh, protocol=4)

    @classmethod
    def load(cls, path) -> "PrefixBoundaryModel":
        with gzip.open(path, "rb") as fh:
            s = pickle.load(fh)
        return cls.from_state(s)

    @classmethod
    def from_state(cls, s: dict) -> "PrefixBoundaryModel":
        m = cls(s["min_count"])
        m.pos = {k: Counter(v) for k, v in s["pos"].items()}
        m.tot = {k: Counter(v) for k, v in s["tot"].items()}
        m.g_pos, m.g_tot, m.sources = s["g_pos"], s["g_tot"], s.get("sources", {})
        return m


def build_prefix_model(lex: GoldLexicon, labeler: TieredLabeler, dev_texts: Iterable[str],
                       min_count: int = 5) -> PrefixBoundaryModel:
    """MorphTok words (weight 1 each) + every Devanagari word of the dev target outputs labelled by
    the tiered labeller, weighted by its frequency in dev. Never pass devtest texts here."""
    freq = Counter()
    for t in dev_texts:
        t = nfc(t)
        for s, e in iter_words(t):
            freq[t[s:e]] += 1
    items = [(w, b, 1.0) for w, b in lex.bounds.items() if is_devanagari_word(w)]
    tiers = Counter()
    for w, c in freq.items():
        b, tier = labeler.label(w)
        if tier == "none":
            continue
        tiers[tier] += c
        items.append((w, b, float(c)))
    src = dict(morphtok_words=len(lex.bounds), dev_word_types=len(freq), dev_word_tokens_by_tier=dict(tiers),
               min_count=min_count)
    return PrefixBoundaryModel(min_count).fit(items, src)


# --------------------------------------------------------------------------- #
# 6. Per-token gold morph labels (analysis only)
# --------------------------------------------------------------------------- #

def token_morph_labels(word_labels: dict, text: str, labeler: TieredLabeler) -> dict[str, np.ndarray]:
    """``word_labels`` is the output of ``boundaries.token_labels`` for the same sequence and
    ``text`` its decoded text. Returns
      morph_boundary_after[t]: the cut after t is a word boundary/end or an internal morpheme boundary
      morph_tier[t]: tier ('gold'/'crf'/'none') of the word containing token t ('none' outside words)."""
    end_cut, spans = word_labels["end_cut"], word_labels["char_span"]
    n = len(end_cut)
    words = labeler.label_text(text)
    char_word = np.full(len(text) + 1, -1, dtype=np.int64)
    for k, (s, e, _, _) in enumerate(words):
        char_word[s:e] = k
    after = np.zeros(n, dtype=bool)
    tier = np.full(n, "none", dtype=object)
    for t in range(n):
        cs, ce = int(spans[t][0]), int(spans[t][1])
        # tier: first word char inside the token's span
        for c in range(cs, max(ce, cs + 1)):
            if c < len(text) and char_word[c] >= 0:
                tier[t] = words[char_word[c]][3]
                break
        if end_cut[t] in (WORD, END):
            after[t] = True
            continue
        if end_cut[t] != MIDWORD:              # mid-akshara / mid-UTF-8 cuts are never morph boundaries
            continue
        k = char_word[ce - 1] if 0 < ce <= len(text) else -1
        if k >= 0 and char_word[ce] == k:      # cut strictly inside word k
            s = words[k][0]
            after[t] = (ce - s) in words[k][2]
    return dict(morph_boundary_after=after, morph_tier=tier)


def attach_morph_labels(traces, labeler: TieredLabeler) -> None:
    """Add morph_boundary_after / morph_tier to traces that already have word labels (and ``text``)."""
    for t in traces:
        t.labels.update(token_morph_labels(t.labels, t.labels["text"], labeler))
