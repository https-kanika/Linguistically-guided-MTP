import os

import numpy as np
import pytest

from lgmtp.boundaries import (END, MIDAKS, MIDUTF8, MIDWORD, WORD, akshara_units, classify,
                              is_word_boundary, sentence_features, token_labels, token_to_bytes,
                              _bytes_to_unicode)


def units(w):
    return [w[a:b] for a, b in akshara_units(w)]


# --- Test 1 cell-3 asserts -------------------------------------------------- #
@pytest.mark.parametrize("word,exp", [
    ("क्षमा", ["क्ष", "मा"]),
    ("विद्यालय", ["वि", "द्या", "ल", "य"]),
    ("ज्ञान", ["ज्ञा", "न"]),
    ("श्री", ["श्री"]),
    ("है।", ["है", "।"]),
])
def test_test1_asserts(word, exp):
    assert units(word) == exp


def test_nukta():
    assert units("ज़िंदगी") == ["ज़िं", "द", "गी"]
    assert units("फ़ोन") == ["फ़ो", "न"]


def test_zwj_zwnj():
    # half-form via virama + ZWJ stays inside one akshara with the following consonant
    assert units("क्‍ष") == ["क्‍ष"]
    # ZWNJ after virama likewise
    assert units("क्‌ष") == ["क्‌ष"]
    # a ZW joiner after a non-Devanagari char attaches to it
    assert units("a‍b") == ["a‍", "b"]


def test_mixed_latin_digit_devanagari():
    s = "COVID-19 का टीका 2021 में"
    u = units(s)
    assert "".join(u) == s
    assert u[:8] == list("COVID-19")
    assert "का" in u and "टी" in u and "का" in u and "में" in u
    assert is_word_boundary(s, s.index(" "))
    assert is_word_boundary(s, s.index("-"))       # punctuation
    assert not is_word_boundary(s, 1)              # inside COVID


def test_word_boundary_edges():
    assert is_word_boundary("abc", 0) and is_word_boundary("abc", 3)
    assert not is_word_boundary("abc", 1)


def test_bytes_to_unicode_is_bijection():
    m = _bytes_to_unicode()
    assert len(m) == 256 and len(set(m.values())) == 256


def test_token_to_bytes():
    enc = _bytes_to_unicode()
    s = " नमस्ते".encode("utf-8")
    tok_str = "".join(enc[b] for b in s)
    assert token_to_bytes(tok_str) == s
    assert token_to_bytes("<0x0A>") == b"\n"
    assert token_to_bytes("▁hello") == b" hello"


def test_midutf8_split_character():
    # "क" is 3 bytes (e0 a4 95); split it across two byte-level tokens
    tb = [b"\xe0\xa4", b"\x95\xe0\xa4\xbe", b" \xe0\xa4\x95"]   # क|ा| क  -> "का क"
    f = sentence_features(tb)
    assert f["text"] == "का क"
    assert f["cut"] == [MIDUTF8, WORD, END]
    assert f["start"][1] == MIDUTF8
    lab = token_labels(tb)
    assert list(lab["end_cut"]) == [MIDUTF8, WORD, END]
    assert list(lab["start_cut"]) == [WORD, MIDUTF8, WORD]


def test_mid_akshara_and_mid_word():
    # क्ष|मा : cut after क्ष is an akshara boundary inside a word -> MIDWORD
    # क|्षमा : cut after क is inside the akshara क्ष -> MIDAKS
    tb = ["क्ष".encode(), "मा".encode()]
    assert sentence_features(tb)["cut"] == [MIDWORD, END]
    tb = ["क".encode(), "्षमा".encode()]
    assert sentence_features(tb)["cut"] == [MIDAKS, END]


def test_token_labels_basic_and_special():
    toks = [" भा", "रत", " एक", "।"]
    tb = [t.encode() for t in toks] + [b"<|eot_id|>"]
    lab = token_labels(tb, special=[False] * 4 + [True])
    assert list(lab["end_cut"]) == [MIDWORD, WORD, WORD, END, END]
    assert list(lab["start_cut"]) == [WORD, MIDWORD, WORD, WORD, END]
    assert list(lab["word_initial"]) == [True, False, True, True, True]
    assert list(lab["pos_in_word"]) == [0, 1, 0, 0, 0]
    assert list(lab["word_id"]) == [0, 0, 1, 2, 3]
    text = "".join(toks)
    spans = lab["char_span"]
    assert [text[a:b] for a, b in spans[:4]] == toks
    assert tuple(spans[4]) == (len(text), len(text))


def test_token_labels_truncated_last_char():
    tb = ["का".encode(), b"\xe0\xa4"]   # output cut mid-character
    lab = token_labels(tb)
    assert list(lab["end_cut"]) == [MIDUTF8, END]
    assert lab["start_cut"][1] == MIDUTF8


def test_token_labels_invalid_middle_raises():
    with pytest.raises(ValueError):
        token_labels([b"\xff\xfe\xfd\xfc\xfb", "का".encode()])


# --- gated tokenizer round trip -------------------------------------------- #
def _llama_tok():
    if not (os.environ.get("HF_TOKEN") or os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))):
        pytest.skip("no HF token")
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Llama tokenizer unavailable: {e}")


def test_llama_bytes_roundtrip():
    tok = _llama_tok()
    s = "विद्यार्थियों ने विद्यालय में प्रार्थना की। Hello 2024!"
    ids = tok(s, add_special_tokens=False)["input_ids"]
    tb = [token_to_bytes(t) for t in tok.convert_ids_to_tokens(ids)]
    assert b"".join(tb).decode("utf-8") == s
    lab = token_labels(tb)
    assert len(lab["end_cut"]) == len(ids) and lab["end_cut"][-1] == END
