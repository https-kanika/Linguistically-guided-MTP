"""Boolean masks over the full vocabulary, built from token *bytes* (``boundaries.token_to_bytes``).

Masks (all shape ``(len(tok),)``, dtype bool):

- ``starts_space``: first byte in ``b" \\n\\t"``.
- ``starts_punct``: first decoded char has Unicode category ``P*`` (``।``, ``,``, ``.``, ``"``, ...).
- ``starts_mark``: first decoded char has category ``M*`` (matra, virama, nukta, anusvara,
  candrabindu, visarga, ...) or is ZWJ / ZWNJ.
- ``starts_cont``: first byte is a UTF-8 continuation byte (0x80-0xBF): the token starts mid-character.
  Such a token is never ``starts_mark`` / ``starts_punct``.
- ``is_special``: special / control tokens (all registered special tokens and eos ids).

A token whose bytes start with a lead byte but do not contain a complete first character
(e.g. ``b"\\xe0\\xa4"``) has an unknown first char and is in none of space/punct/mark/cont.
Special tokens are only ``is_special``.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .boundaries import token_to_bytes

ZW = "‌‍"
MASK_NAMES = ("starts_space", "starts_punct", "starts_mark", "starts_cont", "is_special")


def first_char(b: bytes) -> str:
    """First complete character of ``b`` ('' if none). Continuation-byte starts give ''."""
    if not b or 0x80 <= b[0] <= 0xBF:
        return ""
    return b[:4].decode("utf-8", errors="ignore")[:1]


def classify_token_bytes(b: bytes) -> dict[str, bool]:
    """Mask flags for one non-special token given its raw bytes."""
    out = dict(starts_space=False, starts_punct=False, starts_mark=False, starts_cont=False,
               is_special=False)
    if not b:
        return out
    if b[0] in b" \n\t":
        out["starts_space"] = True
        return out
    if 0x80 <= b[0] <= 0xBF:
        out["starts_cont"] = True
        return out
    ch = first_char(b)
    if not ch:
        return out
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        out["starts_punct"] = True
    elif cat.startswith("M") or ch in ZW:
        out["starts_mark"] = True
    return out


@dataclass
class VocabMasks:
    starts_space: np.ndarray
    starts_punct: np.ndarray
    starts_mark: np.ndarray
    starts_cont: np.ndarray
    is_special: np.ndarray
    eos_ids: list[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.starts_space)

    @property
    def is_eos(self) -> np.ndarray:
        m = np.zeros(self.size, dtype=bool)
        m[[i for i in self.eos_ids if i < self.size]] = True
        return m

    @property
    def starts_newword(self) -> np.ndarray:
        """Tokens that end the current word if they come next: space, punctuation or eos."""
        return self.starts_space | self.starts_punct | self.is_eos

    def padded(self, name: str, size: int) -> np.ndarray:
        """Mask padded with False to ``size`` (model logits can be wider than ``len(tok)``)."""
        m = getattr(self, name)
        if size < len(m):
            raise ValueError("size smaller than vocabulary")
        return np.concatenate([m, np.zeros(size - len(m), dtype=bool)])

    def as_dict(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in MASK_NAMES}


def special_ids(tok) -> set[int]:
    """All special/control token ids of a Hugging Face tokenizer."""
    ids = set(getattr(tok, "all_special_ids", []) or [])
    for i, at in (getattr(tok, "added_tokens_decoder", {}) or {}).items():
        if getattr(at, "special", False):
            ids.add(int(i))
    return ids


def build_masks(tok, eos_ids: Iterable[int] = ()) -> VocabMasks:
    """Build all masks for a Hugging Face tokenizer (size ``len(tok)``)."""
    n = len(tok)
    eos_ids = sorted(set(int(i) for i in eos_ids))
    spec = special_ids(tok) | set(eos_ids)
    toks = tok.convert_ids_to_tokens(list(range(n)))
    arrs = {k: np.zeros(n, dtype=bool) for k in MASK_NAMES}
    for i, t in enumerate(toks):
        if i in spec or t is None:
            arrs["is_special"][i] = True
            continue
        for k, v in classify_token_bytes(token_to_bytes(t)).items():
            if v:
                arrs[k][i] = True
    return VocabMasks(**arrs, eos_ids=eos_ids)


def token_bytes_list(tok, ids) -> list[bytes]:
    """Raw bytes of each id (special tokens give their literal text bytes)."""
    return [token_to_bytes(t) for t in tok.convert_ids_to_tokens([int(i) for i in ids])]
