import os

import numpy as np
import pytest

from lgmtp.boundaries import _bytes_to_unicode
from lgmtp.vocab import build_masks, classify_token_bytes


def flags(b):
    return {k for k, v in classify_token_bytes(b).items() if v}


def test_classify_bytes():
    assert flags(b" the") == {"starts_space"}
    assert flags(b"\nx") == {"starts_space"}
    assert flags(b",") == {"starts_punct"}
    assert flags("।".encode()) == {"starts_punct"}
    assert flags("ा".encode()) == {"starts_mark"}          # matra
    assert flags("्ष".encode()) == {"starts_mark"}         # virama
    assert flags("़".encode()) == {"starts_mark"}          # nukta
    assert flags("ं".encode()) == {"starts_mark"}          # anusvara
    assert flags("‍".encode()) == {"starts_mark"}     # ZWJ
    assert flags(b"\xa4\xbe") == {"starts_cont"}           # tail of a matra: cont, not mark
    assert flags(b"\xe0\xa4") == set()                     # lead bytes only: first char unknown
    assert flags("क".encode()) == set()
    assert flags(b"abc") == set()


def _llama_tok():
    if not (os.environ.get("HF_TOKEN") or os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))):
        pytest.skip("no HF token")
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Llama tokenizer unavailable: {e}")


def _id(tok, raw: bytes):
    enc = _bytes_to_unicode()
    i = tok.convert_tokens_to_ids("".join(enc[b] for b in raw))
    assert i is not None and i != tok.unk_token_id, raw
    return i


def test_llama_masks():
    tok = _llama_tok()
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    m = build_masks(tok, eos_ids=[eot])
    assert m.size == len(tok)
    assert m.starts_space[_id(tok, b" the")]
    assert m.starts_punct[_id(tok, b",")]
    assert m.starts_punct[_id(tok, "।".encode())]
    assert m.starts_mark[_id(tok, "ा".encode())]
    assert m.starts_mark[_id(tok, "्".encode())]
    assert not m.starts_mark[_id(tok, "क".encode())]
    assert m.is_special[eot] and m.is_eos[eot] and m.starts_newword[eot]
    assert not (m.starts_space[eot] or m.starts_punct[eot])
    # all special tokens of Llama-3 (reserved range) are special
    assert m.is_special[128000:].all()
    # some tokens start mid-character, and they are never marks
    assert m.starts_cont.any() and not (m.starts_cont & m.starts_mark).any()
    # categories are mutually exclusive
    tot = m.starts_space.astype(int) + m.starts_punct + m.starts_mark + m.starts_cont + m.is_special
    assert tot.max() == 1
