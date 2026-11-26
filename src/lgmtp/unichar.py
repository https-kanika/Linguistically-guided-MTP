"""Tokenizer-independent text layer for the universal MTP head.

Three pieces, none of which depends on any LLM tokenizer:

1. ``CharCodec``: characters -> (unit id, script id). The nine major Brahmic scripts share one Unicode
   layout (each block is 128 code points with the same offsets for the same letters), so a char in any
   of them maps to ``BRAHMIC_BASE + (codepoint - block_start)`` plus a script id. A head trained on
   Devanagari therefore sees Bengali / Gujarati / ... letters as the *same* units.
2. ``segment_units``: text -> "units" = optional leading spaces + one word (letters, marks, digits,
   ZWJ/ZWNJ) or one punctuation/symbol char; newline runs are separate units. Units concatenate back to
   the text exactly. Leading spaces live inside the unit (" के"), which matches how every
   space-splitting tokenizer (Llama-3, Qwen, Gemma, ...) attaches the space to the following word.
3. ``UnitVocab``: the head's output vocabulary, built from public Hindi text (never from an LLM), sorted
   by UTF-8 bytes so that "all units that start with these bytes" is one contiguous id range
   (``prefix_range``). The last id is the end-of-output unit ``<eos>``.
"""
from __future__ import annotations

import bisect
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import regex as _re   # supports \p{..}; installed with transformers
except ImportError:  # pragma: no cover
    _re = None

# --------------------------------------------------------------------------- #
# 1. character codec
# --------------------------------------------------------------------------- #

SCRIPTS = ("none", "deva", "beng", "guru", "gujr", "orya", "taml", "telu", "knda", "mlym")
BLOCK_START = {"deva": 0x0900, "beng": 0x0980, "guru": 0x0A00, "gujr": 0x0A80, "orya": 0x0B00,
               "taml": 0x0B80, "telu": 0x0C00, "knda": 0x0C80, "mlym": 0x0D00}
SCRIPT_ID = {s: i for i, s in enumerate(SCRIPTS)}

PAD, UNK, BOS, EOS, SEP, PEND, NL = 0, 1, 2, 3, 4, 5, 6
N_SPECIAL = 7
ASCII_BASE = N_SPECIAL                       # printable ASCII 32..126 -> 7..101
BRAHMIC_BASE = ASCII_BASE + 95               # 102..229: offset within a Brahmic block
EXTRA_CHARS = ("‌‍ \t“”‘’–—…«»·•°₹€£×÷।॥"
               "¡¿§¶©®™µ‘’‚„′″")
EXTRA_BASE = BRAHMIC_BASE + 128
N_UNITS = EXTRA_BASE + len(EXTRA_CHARS)
_EXTRA = {ch: EXTRA_BASE + i for i, ch in enumerate(EXTRA_CHARS)}
_BLOCKS = sorted((v, k) for k, v in BLOCK_START.items())


def char_unit(ch: str) -> tuple[int, int]:
    """(unit id, script id) of one character. Danda / double danda (U+0964/5) are used by several
    scripts and get one shared unit."""
    if ch == "\n" or ch == "\r":
        return NL, 0
    if ch in _EXTRA:
        return _EXTRA[ch], 0
    cp = ord(ch)
    if 32 <= cp <= 126:
        return ASCII_BASE + cp - 32, 0
    if 0x0900 <= cp < 0x0D80:
        start, script = _BLOCKS[bisect.bisect_right([b for b, _ in _BLOCKS], cp) - 1]
        return BRAHMIC_BASE + cp - start, SCRIPT_ID[script]
    return UNK, 0


class CharCodec:
    """Vectorised text -> unit ids with a per-codepoint cache."""

    n_units = N_UNITS
    n_scripts = len(SCRIPTS)

    def __init__(self):
        self._cache: dict[str, tuple[int, int]] = {}

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        u = np.empty(len(text), np.int16)
        s = np.empty(len(text), np.int8)
        cache = self._cache
        for i, ch in enumerate(text):
            r = cache.get(ch)
            if r is None:
                r = cache[ch] = char_unit(ch)
            u[i], s[i] = r
        return u, s

    @staticmethod
    def decode(units: Sequence[int], script: str = "deva") -> str:
        """Inverse of ``encode`` for one target script (unknown / special units are dropped)."""
        out, start = [], BLOCK_START.get(script, 0x0900)
        inv_extra = {v: k for k, v in _EXTRA.items()}
        for u in units:
            u = int(u)
            if u == NL:
                out.append("\n")
            elif ASCII_BASE <= u < BRAHMIC_BASE:
                out.append(chr(u - ASCII_BASE + 32))
            elif BRAHMIC_BASE <= u < EXTRA_BASE:
                off = u - BRAHMIC_BASE
                out.append("।॥"[off - 0x64] if off in (0x64, 0x65) else chr(start + off))
            elif u in inv_extra:
                out.append(inv_extra[u])
        return "".join(out)


def transliterate(text: str, to_script: str) -> str:
    """Map every Brahmic char to the same offset in ``to_script`` (the unified-layout identity the codec
    relies on). Used in tests and for the zero-shot-script experiments of later weeks."""
    start = BLOCK_START[to_script]
    out = []
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp < 0x0D80 and ch not in "।॥":
            b = _BLOCKS[bisect.bisect_right([x for x, _ in _BLOCKS], cp) - 1][0]
            out.append(chr(start + cp - b))
        else:
            out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------- #
# 2. units
# --------------------------------------------------------------------------- #

ZW = "‌‍"
if _re is not None:
    UNIT_RE = _re.compile(r"[\r\n]+|[ \t ]*(?:[\p{L}\p{M}\p{N}‌‍]+|[^\s\p{L}\p{M}\p{N}])"
                          r"|[ \t ]+|.", _re.DOTALL)


def is_word_char(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LMN" or ch in ZW


def segment_units(text: str) -> list[tuple[int, int]]:
    """Char spans (start, end) of the units of ``text``; they tile the text exactly."""
    if _re is None:  # pragma: no cover
        raise ImportError("the 'regex' package is required")
    return [m.span() for m in UNIT_RE.finditer(text)]


def unit_strings(text: str) -> list[str]:
    return [text[a:b] for a, b in segment_units(text)]


def continues_unit(ch: str) -> bool:
    """Inference rule for a pending (incomplete) UTF-8 char: it continues the last unit if the last
    complete char is a word char or a space/tab (a word can follow spaces in the same unit)."""
    return is_word_char(ch) or ch in " \t "


# --------------------------------------------------------------------------- #
# 3. unit vocabulary
# --------------------------------------------------------------------------- #

EOS_UNIT = "<eos>"


class UnitVocab:
    """Sorted-by-bytes unit list; id ``len(units)`` is ``<eos>``. ``size`` includes ``<eos>``."""

    def __init__(self, units: Iterable[str]):
        uniq = sorted(set(u for u in units if u and u != EOS_UNIT), key=lambda s: s.encode("utf-8"))
        self.units: list[str] = uniq
        self.bytes: list[bytes] = [u.encode("utf-8") for u in uniq]
        self.index: dict[str, int] = {u: i for i, u in enumerate(uniq)}
        self.eos_id = len(uniq)
        self.size = len(uniq) + 1

    def id(self, unit: str) -> int:
        """Unit id, ``-100`` (ignore index) if out of vocabulary."""
        return self.index.get(unit, -100)

    def prefix_range(self, prefix: bytes) -> tuple[int, int]:
        """Ids [lo, hi) of all units whose UTF-8 bytes start with ``prefix`` (0xFF never occurs in UTF-8)."""
        lo = bisect.bisect_left(self.bytes, prefix)
        hi = bisect.bisect_left(self.bytes, prefix + b"\xff")
        return lo, hi

    def unit_bytes(self, i: int) -> bytes:
        return b"" if i == self.eos_id else self.bytes[i]

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(dict(units=self.units), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "UnitVocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["units"])

    @classmethod
    def build(cls, texts: Iterable[str], max_size: int = 50000, min_count: int = 2) -> tuple["UnitVocab", dict]:
        cnt: Counter = Counter()
        n_texts = 0
        for t in texts:
            cnt.update(unit_strings(t))
            n_texts += 1
        keep = [u for u, c in cnt.most_common(max_size) if c >= min_count]
        total = sum(cnt.values())
        covered = sum(cnt[u] for u in keep)
        V = cls(keep)
        return V, dict(texts=n_texts, unit_tokens=total, unit_types=len(cnt), vocab=V.size,
                       train_coverage=covered / max(total, 1))

    def coverage(self, texts: Iterable[str]) -> float:
        n = hit = 0
        for t in texts:
            for u in unit_strings(t):
                n += 1
                hit += u in self.index
        return hit / max(n, 1)
