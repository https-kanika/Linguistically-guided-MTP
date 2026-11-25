import numpy as np
import pytest

from lgmtp.unichar import (BRAHMIC_BASE, NL, CharCodec, UnitVocab, char_unit, segment_units, transliterate,
                           unit_strings)

HI = "भारत में आज, “नई” नीति लागू हुई।\nदूसरी पंक्ति  यहाँ है।"


def test_units_tile_text():
    for t in (HI, "", " ", "a  b", "x\n\n y", "क्षत्रिय।।", "‌क‍", "५ बजे 10:30 pm!"):
        assert "".join(unit_strings(t)) == t


def test_unit_shapes():
    u = unit_strings("भारत में आज, नीति।\nअब")
    assert u == ["भारत", " में", " आज", ",", " नीति", "।", "\n", "अब"]
    assert unit_strings("a  b") == ["a", "  b"]
    assert unit_strings("x \n") == ["x", " ", "\n"]


def test_codec_unifies_brahmic_scripts():
    c = CharCodec()
    deva, _ = c.encode("कमल")
    beng, sb = c.encode(transliterate("कमल", "beng"))
    assert transliterate("कमल", "beng") == "কমল"
    assert np.array_equal(deva, beng) and set(sb.tolist()) == {2}
    assert char_unit("\n")[0] == NL
    assert char_unit("क")[0] == BRAHMIC_BASE + 0x15
    assert char_unit("।") == char_unit("।")
    assert CharCodec.decode(deva, "deva") == "कमल"
    assert CharCodec.decode(deva, "gujr") == transliterate("कमल", "gujr")


def test_prefix_range_matches_bruteforce():
    V = UnitVocab(unit_strings(HI) + [" भार", " भारत", " भारतीय", "भा", "a", "ab", "abc", "b"])
    assert V.units[V.eos_id - 1] and V.size == len(V.units) + 1
    for pre in (b"", b" ", " भा".encode(), " भार".encode(), " भार".encode()[:-1], b"a", b"ab", b"z", "।".encode()):
        lo, hi = V.prefix_range(pre)
        brute = {i for i, b in enumerate(V.bytes) if b.startswith(pre)}
        assert set(range(lo, hi)) == brute, pre


def test_build_and_coverage():
    V, st = UnitVocab.build([HI, HI, "अलग वाक्य"], max_size=1000, min_count=2)
    assert st["vocab"] == V.size and 0 < st["train_coverage"] <= 1
    assert V.id(" में") >= 0 and V.id(" अलग") == -100
    assert V.coverage([HI]) == pytest.approx(1.0)
