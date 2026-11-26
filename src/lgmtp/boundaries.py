"""Token bytes, Devanagari akshara units, word boundaries and per-token cut labels.

Extracted from ``notebooks/pre_interim/test1_hindi_boundaries.ipynb`` (cell 2). The logic of the
legacy functions is unchanged; only type hints / docstrings were added, plus ``token_labels``.

Fixes bug B4 (never decode single byte-level tokens): everything works on token *bytes* and
byte offsets, and text is only decoded for whole sequences.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from itertools import accumulate
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# token -> bytes
# --------------------------------------------------------------------------- #


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte<->unicode table (used by Llama-3 and Qwen byte-level BPE)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_DECODER: dict[str, int] = {v: k for k, v in _bytes_to_unicode().items()}
_HEX_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def token_to_bytes(tok: str) -> bytes:
    """Raw bytes of a vocabulary token string (as returned by ``convert_ids_to_tokens``)."""
    m = _HEX_RE.match(tok)
    if m:  # SentencePiece byte-fallback token
        return bytes([int(m.group(1), 16)])
    if all(ch in BYTE_DECODER for ch in tok):  # GPT-2 style byte-level BPE
        return bytes(BYTE_DECODER[ch] for ch in tok)
    return tok.replace("▁", " ").encode("utf-8")  # SentencePiece unicode piece


class HFTok:
    """Thin wrapper: text -> list of token byte-strings."""

    def __init__(self, name_or_tokenizer):
        if isinstance(name_or_tokenizer, str):
            from transformers import AutoTokenizer
            self.name = name_or_tokenizer
            self.tok = AutoTokenizer.from_pretrained(name_or_tokenizer)
        else:
            self.name = "custom"
            self.tok = name_or_tokenizer

    def to_bytes(self, text: str) -> list[bytes]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return [token_to_bytes(t) for t in self.tok.convert_ids_to_tokens(ids)]

    def ids_to_bytes(self, ids: Sequence[int]) -> list[bytes]:
        """Byte-strings for a list of token ids (no decoding of individual tokens)."""
        return [token_to_bytes(t) for t in self.tok.convert_ids_to_tokens(list(ids))]


QWEN2_PRETOK_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
                      r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def legacy_qwen2_pretok_tokenizer(name: str = "Qwen/Qwen3.5-0.8B"):
    """``name``'s tokenizer.json with its split regex replaced by the older Qwen2 one (``\\p{L}+``,
    no ``\\p{M}``), which splits before every combining mark.

    Only used to explain the legacy Test 1 Qwen number (3.33 tokens/word): the legacy run loaded
    the tokenizer this way. The model's real tokenizer (``[\\p{L}\\p{M}]+``) gives ~2.16.
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    j = json.load(open(hf_hub_download(name, "tokenizer.json"), encoding="utf-8"))
    j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = QWEN2_PRETOK_REGEX
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json.dumps(j)))


# --------------------------------------------------------------------------- #
# akshara + word boundaries
# --------------------------------------------------------------------------- #

_C = "क-हक़-य़ॸ-ॿ"      # consonants
_VIRAMA = "्"
_NUKTA = "़"
_ZW = "‌‍"                                 # ZWNJ / ZWJ
_IVOW = "ऄ-औॠॡॲ-ॷ"     # independent vowels
_MATRA = "ऺऻा-ौॎॏॕ-ॗॢॣ"
_MOD = "ऀ-ः॑-॔"                  # candrabindu, anusvara, visarga, stress marks

_AKSHARA_RE = re.compile(
    rf"[{_C}]{_NUKTA}?(?:{_VIRAMA}[{_ZW}]?[{_C}]{_NUKTA}?)*(?:{_VIRAMA}[{_ZW}]?|[{_MATRA}]*)[{_MOD}]*"
    rf"|[{_IVOW}]{_NUKTA}?[{_MATRA}]*[{_MOD}]*"
)


def akshara_units(text: str) -> list[tuple[int, int]]:
    """Return list of (start, end) char spans. Non-Devanagari chars are single units
    with any following combining marks / ZW joiners attached."""
    units, i, n = [], 0, len(text)
    while i < n:
        m = _AKSHARA_RE.match(text, i)
        if m and m.end() > i:
            j = m.end()
        else:
            j = i + 1
            while j < n and (unicodedata.category(text[j]).startswith("M") or text[j] in "‌‍"):
                j += 1
        units.append((i, j))
        i = j
    return units


def is_word_boundary(text: str, c: int) -> bool:
    """True if char offset ``c`` is a word boundary (text edge, whitespace or punctuation)."""
    n = len(text)
    if c <= 0 or c >= n:
        return True
    a, b = text[c - 1], text[c]
    if a.isspace() or b.isspace():
        return True
    if unicodedata.category(a).startswith("P") or unicodedata.category(b).startswith("P"):
        return True
    return False


END, WORD, MIDWORD, MIDAKS, MIDUTF8 = "end", "word_boundary", "mid_word", "mid_akshara", "mid_utf8"
CUT_CATEGORIES = (END, WORD, MIDWORD, MIDAKS, MIDUTF8)
TAG = {END: "E", WORD: "W", MIDWORD: "w", MIDAKS: "a", MIDUTF8: "u"}


def classify(b: int, total: int, byte2char: dict[int, int], text: str, aks: set[int]) -> str:
    """Category of a cut at byte offset ``b``.

    byte2char maps byte offsets that start a char (plus ``total``) to char offsets;
    aks is the set of char offsets that are akshara-unit boundaries.
    """
    if b >= total:
        return END
    c = byte2char.get(b)
    if c is None:
        return MIDUTF8
    if c not in aks:
        return MIDAKS
    return WORD if is_word_boundary(text, c) else MIDWORD


def sentence_features(tb: Sequence[bytes]) -> dict | None:
    """tb: list of token byte-strings. Returns None if bytes are not valid UTF-8."""
    full = b"".join(tb)
    try:
        text = full.decode("utf-8")
    except UnicodeDecodeError:
        return None
    byte2char, pos = {}, 0
    for ci, ch in enumerate(text):
        byte2char[pos] = ci
        pos += len(ch.encode("utf-8"))
    byte2char[pos] = len(text)
    units = akshara_units(text)
    aks = {0} | {e for _, e in units}
    ends = list(accumulate(len(b) for b in tb))
    starts = [0] + ends[:-1]
    total = len(full)
    return {
        "text": text,
        "units": units,
        "cut": [classify(e, total, byte2char, text, aks) for e in ends],
        "start": [classify(s, total, byte2char, text, aks) for s in starts],
    }


# --------------------------------------------------------------------------- #
# per-token gold labels (analysis only; never read by policies)
# --------------------------------------------------------------------------- #


def _valid_utf8_prefix_len(full: bytes) -> int:
    """Length of the longest prefix of ``full`` that is valid UTF-8, allowing only a
    truncated character at the very end (e.g. generation stopped at MAX_NEW)."""
    for cut in range(len(full), max(len(full) - 4, -1), -1):
        try:
            full[:cut].decode("utf-8")
            return cut
        except UnicodeDecodeError:
            continue
    raise ValueError("token bytes are not valid UTF-8 (invalid sequence before the end)")


def token_labels(token_bytes: Sequence[bytes],
                 special: Sequence[bool] | None = None) -> dict[str, np.ndarray]:
    """Gold labels for each token t of a target sequence (analysis only, never read by policies).

    end_cut[t]      : category of the cut AFTER token t (END/WORD/MIDWORD/MIDAKS/MIDUTF8)
    start_cut[t]    : category of the cut BEFORE token t
    word_initial[t] : token t starts a new word (start_cut in {WORD, END})
    pos_in_word[t]  : 0 for word-initial tokens, then 1, 2, ...
    word_id[t]      : index of the word the token belongs to
    char_span[t]    : (start_char, end_char) in the decoded text, shape (n, 2)

    ``special[t]`` marks special/control tokens (e.g. ``<|eot_id|>``). They are stripped before
    labelling and get ``END`` for both cuts, ``word_initial=True``, ``pos_in_word=0``, a fresh
    ``word_id`` and an empty char span at the end of the text. If ``special`` is None, no token is
    treated as special.

    A trailing incomplete UTF-8 character (output truncated mid-character) is tolerated: cuts at
    its start or inside it are ``MIDUTF8`` (what follows is unknown). Invalid bytes anywhere else
    raise ``ValueError``.
    """
    n = len(token_bytes)
    special = [False] * n if special is None else [bool(s) for s in special]
    if len(special) != n:
        raise ValueError("special must have the same length as token_bytes")
    keep = [t for t in range(n) if not special[t]]
    tb = [token_bytes[t] for t in keep]

    full = b"".join(tb)
    valid = _valid_utf8_prefix_len(full)
    text = full[:valid].decode("utf-8")
    full_total = len(full)

    byte2char, pos = {}, 0
    char_of_byte = np.zeros(full_total + 1, dtype=np.int64)  # char containing byte b
    for ci, ch in enumerate(text):
        byte2char[pos] = ci
        nb = len(ch.encode("utf-8"))
        char_of_byte[pos:pos + nb] = ci
        pos += nb
    byte2char[pos] = len(text)
    char_of_byte[pos:] = len(text)  # trailing partial char maps past the text
    aks = {0} | {e for _, e in akshara_units(text)}

    def cat(b: int) -> str:
        if b >= full_total:
            return END
        if b >= valid:  # inside the trailing truncated char
            return MIDUTF8
        return classify(b, valid, byte2char, text, aks)

    ends = list(accumulate(len(b) for b in tb))
    starts = [0] + ends[:-1]

    end_cut = np.full(n, END, dtype=object)
    start_cut = np.full(n, END, dtype=object)
    char_span = np.full((n, 2), len(text), dtype=np.int64)
    for j, t in enumerate(keep):
        s, e = starts[j], ends[j]
        end_cut[t] = cat(e)
        start_cut[t] = cat(s)
        cs = int(char_of_byte[min(s, full_total)])
        ce = int(char_of_byte[e - 1]) + 1 if e > s else cs
        char_span[t] = (cs, min(ce, len(text)) if e <= valid else len(text))

    word_initial = np.array([sc in (WORD, END) for sc in start_cut], dtype=bool)
    pos_in_word = np.zeros(n, dtype=np.int64)
    word_id = np.zeros(n, dtype=np.int64)
    wid = -1
    for t in range(n):
        if word_initial[t] or t == 0:
            wid += 1
            pos_in_word[t] = 0
        else:
            pos_in_word[t] = pos_in_word[t - 1] + 1
        word_id[t] = wid
    return dict(end_cut=end_cut, start_cut=start_cut, word_initial=word_initial,
                pos_in_word=pos_in_word, word_id=word_id, char_span=char_span)


# --------------------------------------------------------------------------- #
# corpus analysis
# --------------------------------------------------------------------------- #


def analyze(token_bytes_per_sentence: Iterable[Sequence[bytes]], ks: Sequence[int]):
    """Corpus statistics of token cuts and of fixed-k draft windows. Returns (tot, per_k)."""
    tot = Counter()
    per_k = {k: Counter() for k in ks}
    for tb in token_bytes_per_sentence:
        f = sentence_features(tb)
        if f is None:
            tot["skipped_invalid_utf8"] += 1
            continue
        n = len(tb)
        if n < 2:
            tot["skipped_short"] += 1
            continue
        text, cut, start = f["text"], f["cut"], f["start"]
        tot["sentences"] += 1
        tot["tokens"] += n
        tot["words"] += len(text.split())
        tot["chars"] += len(text)
        tot["aksharas"] += len(f["units"])
        tot["starts_mid_utf8"] += sum(1 for s in start if s == MIDUTF8)
        tot["starts_mid_akshara"] += sum(1 for s in start if s == MIDAKS)
        for j in range(n - 1):  # internal cuts only (a token follows)
            tot["cuts"] += 1
            tot["cut_" + cut[j]] += 1

        wb = [c in (WORD, END) for c in cut]
        pw = [0] + list(accumulate(1 if x else 0 for x in wb))
        last_wb, lw = [], -1
        for j in range(n):
            if wb[j]:
                lw = j
            last_wb.append(lw)
        next_wb, nw = [0] * n, n - 1
        for j in range(n - 1, -1, -1):
            if wb[j]:
                nw = j
            next_wb[j] = nw

        for k in ks:
            c = per_k[k]
            for s in range(0, n - k):
                e = s + k - 1
                cat = cut[e]
                c["windows"] += 1
                c[cat] += 1
                done = pw[e + 1] - pw[s]
                c["completed_words"] += done
                if done == 0:
                    c["zero_word"] += 1
                if cat not in (WORD, END):
                    c["mid_n"] += 1
                    c["fwd_sum"] += next_wb[e] - e
                    c["back_sum"] += e - last_wb[e]
    return tot, per_k


def pct(a: float, b: float) -> float:
    return 100.0 * a / b if b else float("nan")


def summarize(name: str, tot: Counter, per_k: dict, ks: Sequence[int]) -> dict:
    """Turn ``analyze`` counters into the Test 1 summary dict."""
    S = {
        "tokenizer": name,
        "sentences": tot["sentences"],
        "tokens": tot["tokens"],
        "tokens_per_word": tot["tokens"] / max(tot["words"], 1),
        "tokens_per_akshara": tot["tokens"] / max(tot["aksharas"], 1),
        "tokens_per_char": tot["tokens"] / max(tot["chars"], 1),
        "pct_tokens_start_mid_akshara": pct(tot["starts_mid_akshara"], tot["tokens"]),
        "pct_tokens_start_mid_utf8": pct(tot["starts_mid_utf8"], tot["tokens"]),
        "pct_cuts_word_boundary": pct(tot["cut_" + WORD] + tot["cut_" + END], tot["cuts"]),
        "pct_cuts_mid_word": pct(tot["cut_" + MIDWORD], tot["cuts"]),
        "pct_cuts_mid_akshara": pct(tot["cut_" + MIDAKS], tot["cuts"]),
        "pct_cuts_mid_utf8": pct(tot["cut_" + MIDUTF8], tot["cuts"]),
        "skipped_invalid_utf8": tot["skipped_invalid_utf8"],
        "per_k": [],
    }
    for k in ks:
        c = per_k[k]
        w = c["windows"]
        S["per_k"].append({
            "k": k,
            "windows": w,
            "pct_end_at_word_boundary": pct(c[WORD] + c[END], w),
            "pct_end_mid_word": pct(c[MIDWORD], w),
            "pct_end_mid_akshara": pct(c[MIDAKS], w),
            "pct_end_mid_utf8": pct(c[MIDUTF8], w),
            "pct_end_anywhere_inside_word": 100.0 - pct(c[WORD] + c[END], w) if w else float("nan"),
            "mean_words_completed": c["completed_words"] / w if w else float("nan"),
            "pct_drafts_completing_zero_words": pct(c["zero_word"], w),
            "mean_tokens_forward_to_boundary": c["fwd_sum"] / c["mid_n"] if c["mid_n"] else float("nan"),
            "mean_tokens_back_to_boundary": c["back_sum"] / c["mid_n"] if c["mid_n"] else float("nan"),
        })
    return S


def print_summary(S: dict) -> None:
    print(f"\n=== {S['tokenizer']} ===")
    print(f"sentences={S['sentences']}  tokens={S['tokens']}  "
          f"tokens/word={S['tokens_per_word']:.2f}  tokens/akshara={S['tokens_per_akshara']:.2f}  "
          f"tokens/char={S['tokens_per_char']:.2f}")
    print(f"tokens that START mid-akshara: {S['pct_tokens_start_mid_akshara']:.1f}%   "
          f"mid-UTF-8: {S['pct_tokens_start_mid_utf8']:.1f}%")
    print(f"all token cuts -> word boundary {S['pct_cuts_word_boundary']:.1f}% | mid-word {S['pct_cuts_mid_word']:.1f}% | "
          f"mid-akshara {S['pct_cuts_mid_akshara']:.1f}% | mid-UTF-8 {S['pct_cuts_mid_utf8']:.1f}%")
    if S["skipped_invalid_utf8"]:
        print(f"WARNING: skipped {S['skipped_invalid_utf8']} sentences (token bytes not valid UTF-8)")
    hdr = ("k", "@word", "mid-word", "mid-aks", "mid-utf8", "inside-any", "words/draft", "0-word%", "fwd", "back")
    print("\n" + " ".join(f"{h:>11}" for h in hdr))
    for r in S["per_k"]:
        print(" ".join(f"{v:>11}" for v in (
            r["k"], f"{r['pct_end_at_word_boundary']:.1f}", f"{r['pct_end_mid_word']:.1f}",
            f"{r['pct_end_mid_akshara']:.1f}", f"{r['pct_end_mid_utf8']:.1f}",
            f"{r['pct_end_anywhere_inside_word']:.1f}", f"{r['mean_words_completed']:.2f}",
            f"{r['pct_drafts_completing_zero_words']:.1f}",
            f"{r['mean_tokens_forward_to_boundary']:.2f}", f"{r['mean_tokens_back_to_boundary']:.2f}")))
    print("(columns 2-6 are % of k-token drafts; fwd/back = avg tokens to move to the nearest word boundary "
          "when the draft ends inside a word)")


# --------------------------------------------------------------------------- #
# examples for slides
# --------------------------------------------------------------------------- #


def _show(b: bytes) -> str:
    try:
        return b.decode("utf-8").replace(" ", "␣")  # visible space
    except UnicodeDecodeError:
        return "<" + b.hex() + ">"


def render_example(tb: Sequence[bytes], k: int) -> str:
    """Render a tokenised sentence with fixed-k draft ends tagged by cut category."""
    f = sentence_features(tb)
    if f is None:
        return "(invalid utf-8)"
    out = []
    for i, b in enumerate(tb):
        out.append(_show(b))
        if i < len(tb) - 1:
            out.append(f" ‖[{TAG[f['cut'][i]]}] " if (i + 1) % k == 0 else " | ")
    return "".join(out)


EXAMPLE_LEGEND = ("Legend: | token boundary,  ‖ end of a fixed-k draft,  tag = what the draft ends on: "
                  "W=word boundary, w=inside word, a=inside akshara, u=inside UTF-8 char, ␣ = space\n")

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

DEMO = [
    "भारत एक विशाल और विविधताओं से भरा देश है।",
    "विद्यार्थियों ने विद्यालय में सूर्योदय के समय प्रार्थना की।",
    "मौसम विभाग ने कल भारी बारिश की चेतावनी जारी की है।",
    "वैज्ञानिकों का मानना है कि जलवायु परिवर्तन का प्रभाव कृषि पर पड़ेगा।",
    "उसने अपने दोस्तों को खाने पर बुलाया और सबने मिलकर खाना खाया।",
]


def load_sentences(path: str, field: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            for line in fh:
                if line.strip():
                    out.append(json.loads(line)[field])
        else:
            out = [ln.rstrip("\n") for ln in fh]
    return [unicodedata.normalize("NFC", s.strip()) for s in out if s.strip()]


def load_flores_hindi(split: str = "devtest") -> list[str]:
    """FLORES+ hin_Deva sentences (NFC, stripped), as used in Test 1."""
    from datasets import load_dataset
    ds = load_dataset("openlanguagedata/flores_plus", "hin_Deva", split=split)
    return [unicodedata.normalize("NFC", s.strip()) for s in ds["text"] if s.strip()]
