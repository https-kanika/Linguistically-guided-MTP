import numpy as np
import torch

from lgmtp.wordheads import collate, continuation_targets, keep_pair, make_heads, variant_losses, DEFAULT_WH


def test_continuation_targets():
    # tokens: [W c c W c W]  (W = word-initial)
    wi = np.array([1, 0, 0, 1, 0, 1], bool)
    c = continuation_targets(wi, 3)
    assert c[0].tolist() == [True, True, False]     # y1, y2 continue word 0; y3 starts a new word
    assert c[1].tolist() == [True, False, False]
    assert c[2].tolist() == [False, False, False]
    assert c[3].tolist() == [True, False, False]
    assert c[4].tolist() == [False, False, False]
    assert c[5].tolist() == [False, False, False]   # nothing after the last token


def test_collate_rows_follow_trace_convention():
    # h_t sits at input index len(P) + t - 1 and head k at h_t targets y[t+k]
    ex = dict(prompt_ids=[1, 2, 3], out_ids=[10, 11, 12, 13], word_initial=np.array([1, 0, 1, 0], bool))
    ex["cont"] = continuation_targets(ex["word_initial"], 2)
    B = collate([ex], pad_id=0, device="cpu")
    assert B["rows_i"].tolist() == [2, 3, 4, 5]
    assert B["tgt"][:, 0].tolist() == [11, 12, 13, -100]
    assert B["tgt"][:, 1].tolist() == [12, 13, -100, -100]
    assert B["tvalid"][:, 1].tolist() == [True, True, False, False]
    assert B["cont"][:, 0].tolist() == [True, False, True, False]


def test_variant_losses_finite_and_v1_masks():
    torch.manual_seed(0)
    d, V, K = 8, 20, 2
    W = torch.randn(V, d)
    ex = dict(prompt_ids=[1, 2], out_ids=[3, 4, 5, 6, 7], word_initial=np.array([1, 0, 0, 1, 0], bool))
    ex["cont"] = continuation_targets(ex["word_initial"], K)
    B = collate([ex], pad_id=0, device="cpu")
    h = torch.randn(len(B["rows_i"]), d)
    gen = torch.Generator().manual_seed(0)
    for v in ("v0", "v1", "v2"):
        heads = make_heads(d, K)
        loss, logs = variant_losses(v, heads, h, B, W, K, 0.8, 0.5, 100, gen)
        assert torch.isfinite(loss)
        loss.backward()
        assert heads.blocks[0].weight.grad is not None
        if v != "v0":
            assert "bce" in logs


def test_keep_pair_drops_ui_strings():
    cfg = dict(DEFAULT_WH)
    assert keep_pair("The children are playing in the garden today.", "बच्चे आज बगीचे में खेल रहे हैं।", cfg)
    assert not keep_pair("Failed to parse project % s: % s", "परियोजना % s पार्स करने में विफल: % s", cfg)
    assert not keep_pair("Delete all stashes", "सभी स्टैश मिटाएँ", cfg)
