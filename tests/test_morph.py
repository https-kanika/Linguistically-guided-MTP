import pandas as pd
import pytest

from lgmtp.boundaries import token_labels
from lgmtp.morph import (CRFSegmenter, GoldLexicon, PrefixBoundaryModel, TieredLabeler, akshara_boundaries,
                         boundary_prf, iter_words, seg_boundaries, snap_to_akshara, token_morph_labels)


def lex_from(rows):
    df = pd.DataFrame(rows, columns=["word", "segmentation"])
    df["language"] = "hi"
    df["kept"] = df["segmentation"].str.replace("+", "", regex=False) == df["word"]
    return GoldLexicon.from_dataframe(df)


def test_seg_boundaries_roundtrip():
    lex = lex_from([("भक्तभगवान", "भक्त+भगवान"), ("गढपुर", "गढ+पुर"), ("वर्चस्व", "वर्चस्व")])
    for w, seg in [("भक्तभगवान", "भक्त+भगवान"), ("गढपुर", "गढ+पुर"), ("वर्चस्व", "वर्चस्व")]:
        b = sorted(lex.get(w))
        pieces = [w[i:j] for i, j in zip([0] + b, b + [len(w)])]
        assert "+".join(pieces) == seg
    assert "नहीं" not in lex


def test_sandhi_rows_dropped_union_and_snap():
    lex = lex_from([
        ("दत्तेश", "दत्त+ईश"),          # sandhi: does not re-join -> dropped
        ("अक्षर", "अ+क्षर"), ("अक्षर", "अक्षर"),   # duplicate -> union
        ("निकालेगी", "निकाल+ेगी"),       # mid-akshara -> snapped left
    ])
    assert "दत्तेश" not in lex
    assert lex.get("अक्षर") == {1}
    assert lex.get("निकालेगी") == {len("निका")}
    assert lex.stats["boundaries_inside_akshara_snapped"] == 1


def test_snap_to_zero_is_dropped():
    # a boundary inside the first akshara snaps to 0 and disappears
    b, k = snap_to_akshara("कि", {1})
    assert b == set() and k == 1


def test_crf_only_akshara_boundaries():
    words = ["गढपुर", "रसदशा", "भक्तभगवान", "रामपुर", "सीतापुर", "पढ़ेगा", "लिखेगा", "खाएगा"] * 5
    bounds = [seg_boundaries(s) for s in ["गढ+पुर", "रस+दशा", "भक्त+भगवान", "राम+पुर", "सीता+पुर",
                                          "पढ़+ेगा", "लिख+ेगा", "खा+एगा"]] * 5
    seg = CRFSegmenter(max_iterations=30).fit(words, bounds)
    for w in ["क्षमापुर", "विद्यालयपुर", "पढ़ेगा", "ज़िंदगी", "श्रीनगर"] + words[:8]:
        assert seg.predict(w) <= akshara_boundaries(w)
    assert seg.predict("") == set()


def test_prf():
    m = boundary_prf([{1, 3}, set()], [{1}, {2}])
    assert m["tp"] == 1 and m["precision"] == 0.5 and m["recall"] == 0.5


def test_prefix_model_range_and_backoff():
    pm = PrefixBoundaryModel(min_count=2)
    data = [("रामपुर", {len("राम")}), ("सीतापुर", {len("सीता")}), ("गढपुर", {len("गढ")}), ("रामलाल", {len("राम")})]
    pm.fit([(w, b, 1.0) for w, b in data])
    for w, _ in data:
        for c in range(1, len(w) + 1):
            p = pm.prob(w[:c])
            assert 0.0 <= p <= 1.0
    # "राम" seen twice as a full prefix, both times followed by a boundary
    assert pm.level_used("राम") == "full" and pm.prob("राम") == 1.0
    # unseen prefix whose last akshara is seen enough -> backs off to last1/last2/last3
    assert pm.level_used("घनश्याम") in ("last1", "last2", "last3")
    # completely unseen material -> global rate
    assert pm.level_used("ज़ौ") == "global" and pm.prob("ज़ौ") == pytest.approx(pm.global_rate)
    assert pm.prob("") == pytest.approx(pm.global_rate)


def test_prefix_model_save_load(tmp_path):
    pm = PrefixBoundaryModel(min_count=1).fit([("रामपुर", {6}, 1.0)])
    pm.save(tmp_path / "p.pkl.gz")
    pm2 = PrefixBoundaryModel.load(tmp_path / "p.pkl.gz")
    assert pm2.prob("राम") == pm.prob("राम") and pm2.global_rate == pm.global_rate


def test_iter_words_and_token_morph_labels():
    text = "रामपुर में, COVID-19 है।"
    assert [text[s:e] for s, e in iter_words(text)] == ["रामपुर", "में", "COVID", "19", "है"]
    lex = lex_from([("रामपुर", "राम+पुर")])
    lab = TieredLabeler(lex, crf=None)
    toks = ["राम", "पुर", " में", ",", " COVID", "-", "19", " है", "।"]
    wl = token_labels([t.encode() for t in toks])
    m = token_morph_labels(wl, text, lab)
    assert list(m["morph_boundary_after"]) == [True, True, True, True, True, True, True, True, True]
    assert list(m["morph_tier"][:2]) == ["gold", "gold"]
    assert m["morph_tier"][4] == "none"
    # a mid-word cut that is not a morpheme boundary
    toks2 = ["रा", "मपुर"]
    m2 = token_morph_labels(token_labels([t.encode() for t in toks2]), "रामपुर", lab)
    assert list(m2["morph_boundary_after"]) == [False, True]


def test_tiered_labeler_tiers():
    lex = lex_from([("रामपुर", "राम+पुर")])
    seg = CRFSegmenter(max_iterations=5).fit(["रामपुर", "गढपुर"], [{6}, {4}])
    lab = TieredLabeler(lex, seg)
    assert lab.label("रामपुर")[1] == "gold"
    assert lab.label("सीतापुर")[1] == "crf"
    assert lab.label("COVID")[1] == "none" and lab.label("19") == (frozenset(), "none")
    cov = lab.coverage(["रामपुर सीतापुर COVID"])
    assert cov["word_tokens"] == 3 and cov["tokens"]["gold"]["n"] == 1
