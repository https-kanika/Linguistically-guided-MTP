"""Exact conversion with real tokenizers on real Hindi (skips if a tokenizer is not available locally).

Drafting the true next units at every position must give the target's true tokens, for the Phase-1
target tokenizer (Llama-3) and for tokenizers of other families. Every draft level except the longest
must match exactly. The longest may differ in its final token only: the canonical tokenization can
merge the draft's last characters with text that follows it (Llama-3 glues a word-final vowel sign to
the next danda or comma; Qwen glues a question mark to the next newline), exactly as in a real
decoding loop, where that token is then rejected.
"""
import numpy as np
import pytest

from lgmtp import uni_sim as U
from lgmtp.uni_features import TokTable, featurize_records, make_batch, make_encoder
from lgmtp.unichar import UnitVocab, unit_strings

HINDI = [
    "भारत सरकार ने आज नई शिक्षा नीति की घोषणा की, जो अगले वर्ष से लागू होगी।",
    "डॉ. एहुद उर ने कहा कि \"हमारे पास अब 4 महीने के चूहे हैं\" जो पहले मधुमेह से ग्रस्त थे।",
    "वैज्ञानिकों के अनुसार, क्षेत्र में 2,500 से अधिक प्रजातियाँ पाई जाती हैं।",
    "उन्होंने बताया कि NASA का मिशन 2024 में शुरू हुआ था और यह सफल रहा।",
    "क्या आप जानते हैं कि हिमालय दुनिया की सबसे ऊँची पर्वत श्रृंखला है?\nहाँ, यह सच है।",
]
TOKENIZERS = ["NousResearch/Meta-Llama-3-8B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct"]


def _tok(name):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{name} not available locally: {e}")


@pytest.mark.parametrize("name", TOKENIZERS)
def test_truth_units_reproduce_true_tokens(name):
    tok = _tok(name)
    eos = tok.eos_token_id
    V = UnitVocab(sum((unit_strings(t) for t in HINDI), []))
    table = TokTable(tok, [eos])
    enc = make_encoder(tok)
    recs = []
    for i, t in enumerate(HINDI):
        ids = enc([t])[0] + [eos]
        n = len(ids)
        tk = np.tile(np.arange(20), (n, 1))
        tk[:, 0] = ids
        recs.append(dict(id=i, prompt_text="", out_ids=ids, eos_end=True, topk_ids=tk,
                         topk_logp=np.full((n, 20), -3.0, np.float32)))
    fs = featurize_records(recs, table, V)
    assert (fs.tgt[:, 0] != -100).mean() > 0.95, "head-1 targets should exist almost everywhere"
    W = fs.tgt.copy()
    W[W < 0] = -1
    for i in range(1, W.shape[1]):
        W[:, i] = np.where(W[:, i - 1] < 0, -1, W[:, i])
    seqs = U.build_simseqs(recs, table, V)
    drafts, st = U.convert_all(seqs, W, V, enc, table.bytes)
    bad, last_only, n_last = [], 0, 0
    for s, seq in enumerate(seqs):
        for p in range(seq.n):
            lv_all = drafts[s][p][1:]
            for lv, d in enumerate(lv_all, 1):
                true = seq.out_ids[p:p + len(d)].tolist()
                if lv < len(lv_all) and d != true:
                    bad.append((s, p, lv, "shorter level differs"))
                elif lv == len(lv_all):
                    n_last += 1
                    if d[:-1] != true[:len(d) - 1]:
                        bad.append((s, p, lv, "longest level differs before its last token"))
                    elif d != true:
                        last_only += 1
    assert not bad and st["failed"] == 0, (st, bad[:5])
    # Inherent, and free: a wrong final token is replaced by the target's correction token, so the
    # round emits as many tokens as without it. These sentences are punctuation-dense (Llama-3: ~9%).
    assert last_only <= 0.15 * n_last, (last_only, n_last)
    R = U.simulate(seqs, drafts, np.ones_like(W, np.float32), 0.0, 0.05, U.Verify(
        dict(t_target=1.0, t_target_verify={str(g): 1 + 0.01 * g for g in range(1, 9)})))
    assert R["acceptance"] > 0.97 and R["tokens_per_round"] > 2.5
    b = make_batch(fs, np.arange(fs.n_pos))
    assert b["cand_u"].shape[1] == 60
