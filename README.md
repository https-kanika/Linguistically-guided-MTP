# Linguistically-guided speculative decoding for Hindi

**Current direction: word-completion heads on the draft model ("a refund for the token tax").**
The detailed task plan is in [`PLAN_wordheads.md`](PLAN_wordheads.md). This README covers the
motivation, what we know so far, the related work, the code layout, and how to run everything.

---

## 1. Background in plain terms

**Speculative decoding.** A big model (Llama-3-8B-Instruct, the *target*) writes the translation, and
a small model (Llama-3.2-1B-Instruct, the *draft*) guesses a few tokens ahead. The 8B checks all the
guesses in one forward pass and keeps the longest correct prefix. The output is identical to running
the 8B alone (lossless). It is faster because one 8B pass can accept several tokens.

**The token tax.** Llama-3's tokenizer splits Hindi into many pieces: about **2.67 tokens per word**,
against about 1.3 for English. **59% of the tokens the 8B writes in Hindi only continue a word that has
already started.** Every one of those tokens costs a full draft-model call (0.32 of an 8B step).

**Phase 1 (done).** We tried to decide *when to stop drafting* with linguistic rules (word and morpheme
boundaries). None beat a plain entropy rule (stop when the 1B is uncertain): **1.452× [1.444, 1.461]**
on FLORES devtest. Report and results: `results/phase1/`.

**Friend's week 1 (branch `phase2-universal-head`, NO GO).** A universal word-level head that drafts
whole words reached 1.296× at best (text-only 1.287×, no-Llama 1.211×), below Entropy. It got the
first word right 66% of the time and later words 14% and 6%. Results: `results/week1/`.

## 2. The idea

The 1B is very accurate *inside* words: it matches the 8B on 81–86% of continuation tokens, against
66% at word starts. But it still pays full price for every continuation token.

So we let the 1B draft only the **first token of each word** with its normal LM head. Small
**multi-token heads** on the same hidden state propose the rest of the word almost for free. The 1B
then reads the completed tokens in one forward (about the cost of one token) and drafts the next word
start. The 8B verifies as usual, so decoding stays lossless.

```
word start  : 1B call -> "प्र"          (cost c)
inside word : heads   -> "ति" "योग" "िता" (cost ~0.02c, read back in the next 1B call)
next word   : 1B call -> " में"
8B verifies all drafted tokens in one pass
```

## 3. Evidence so far (CPU, exact simulation on Phase-1 traces)

All numbers are in [`results/wordheads/feasibility.json`](results/wordheads/feasibility.json) and come
from `scripts/feasibility/`. Everything was tuned on dev and reported on devtest (1,012 sentences), with
bootstrap 95% CIs.

| drafter | speed-up [95% CI] |
|---|---|
| Entropy, 1B drafts every token (Phase 1) | 1.452 [1.444, 1.461] |
| **Ceiling**: continuation tokens at 5% of a 1B call, as accurate as the 1B | **1.911 [1.89, 1.93]** |
| Dictionary completer on every mid-word position | 1.224 [1.217, 1.232] |
| **Dictionary completer, only when confident** (top word ≥ 60% of the frequency of matching words) | **1.470 [1.462, 1.480]**, +0.018 vs Entropy [+0.016, +0.021] |

**How accurate must a completer be?** The table below assumes the completer keeps a share ρ of the
1B's correct continuation tokens:

| ρ | 1.0 | 0.95 | 0.9 | 0.8 | **0.7** | 0.6 | 0.5 |
|---|---|---|---|---|---|---|---|
| speed-up | 1.91 | 1.81 | 1.72 | 1.57 | **1.45 (break-even)** | 1.36 | 1.28 |

**Context has to come from a model.** Whole-word completion accuracy over 38.7k mid-word positions:

| completer | accuracy |
|---|---|
| dictionary, prefix only | 31.3% |
| + previous word (bigram) + sentence cache | 34.9% |
| + the reference translation's words (oracle!) | 47.3% |
| **1B draft model** | **77.2%** |

Takeaways:
- There is a large, measurable refund available: up to +0.46× over Entropy.
- Dictionaries can't reach it, even with oracle context. Gating them gives a small, real gain, which
  makes it the training-free baseline.
- The heads must keep at least 70% of the 1B's correct continuation tokens to help.

## 3b. Head results (Kaggle run 07 + `scripts/run_refund_sim.py`)

Numbers from `results/wordheads/main_table.csv` (full tuning grid on dev, reported on devtest), with 95%
bootstrap CIs and paired CIs vs Entropy.

| drafter | speed-up [95% CI] | diff vs Entropy |
|---|---|---|
| Entropy (Phase 1) | 1.452 [1.444, 1.461] | — |
| Dictionary, every mid-word position | 1.226 [1.218, 1.234] | [−0.235, −0.217] |
| Dictionary, confidence-gated | 1.471 [1.462, 1.480] | [+0.017, +0.022] |
| Heads V0, plain MTP (continuation vocab) | 1.497 [1.487, 1.508] | [+0.042, +0.048] |
| Heads V0, plain MTP (full vocab) | 1.496 [1.485, 1.506] | [+0.040, +0.047] |
| **Heads V1, word-guided (continuation vocab)** | **1.514 [1.503, 1.525]** | **[+0.057, +0.066]** |
| Heads V2, adaptive weights (continuation vocab) | 1.505 [1.496, 1.516] | [+0.050, +0.057] |
| Ceiling: continuations at 5% of a 1B call | 1.911 [1.894, 1.931] | [+0.449, +0.470] |

Paired differences (devtest):
- V1 − V0, same (continuation) vocab: +0.016 [+0.012, +0.020].
- V1 − best V0: +0.018 [+0.014, +0.022].
- V2 − V1: −0.008 [−0.011, −0.005].
- V1 − gated dictionary: +0.043 [+0.038, +0.047].

Head accuracy on word-continuation tokens along the 8B outputs (`head_accuracy.json`), heads 1/2/3:

| | head 1 | head 2 | head 3 |
|---|---|---|---|
| V0 | 52.4% | 38.9% | 29.2% |
| V1 | 54.4% | 45.4% | 39.0% |
| 1B draft model itself | 83.0% | 86.2% | 85.7% |

Measured on a T4: the continuation-vocab heads cost 0.018 of a 1B call. Reading q ≤ 8 tokens costs about
the same as reading one token (r(q) ≈ 1).

The heads keep only about 62% of the 1B's correct first continuation tokens, below the 70% break-even of
the ρ curve. They still win because they propose only when confident (τ_h ≈ 0.5), and the ρ curve
assumes a completer that always proposes.

## 4. Method

Position convention (same as the Phase-1 traces): `h_t` is the 1B's last hidden state that predicts
output token `y[t]`.

- **Heads.** K = 3 Medusa-1 heads: `z_k = h_t + SiLU(W_k h_t)`, with `W_k` zero-initialised, followed by
  the frozen, shared 1B LM head. Head k predicts `y[t+k]`. At inference the LM head is restricted to the
  **continuation vocabulary**: 1,406 Llama-3 tokens with no leading space, which cover 96.3% of the
  8B's continuation tokens. This makes a head cost about 0.015–0.02 of a 1B call instead of 0.25.
- **Continuation head.** `Linear(d, K)`, which predicts whether `y[t+1..t+k]` all continue the current word.
- **Training.** The 1B stays frozen. Teacher forcing on 60k filtered IITB en→hi pairs with the Phase-1
  prompt. One 1B forward per batch trains all three variants on identical data:

| variant | loss |
|---|---|
| **V0** plain MTP | Σ_k 0.8^(k−1) · CE_k at every position |
| **V1** word-guided | CE_k only inside the word (`c_k = 1`), + 0.5 · BCE(continuation head) |
| **V2** adaptive | V1's losses with uncertainty weighting `Σ exp(−s_i) L_i + s_i` |

- **Drafting (simulated exactly on the 8B traces).**
  - At a word start, the 1B drafts the first token.
  - The heads then propose the rest of the word. They stop at a word-initial proposal, at low
    confidence (τ_h), or (V1/V2) when the continuation probability is below τ_c.
  - The Phase-1 entropy rule ends the round. All thresholds are tuned on dev.

## 5. Related work (what we found and how we differ)

| paper | what it does | relation to us |
|---|---|---|
| **"No Refund for the Token Tax"** (Yang, 2026, alphaxiv 2610.*) | Speculative decoding in 21 languages: high-fertility languages get no *acceptance-length* refund, and acceptance within token types is below English | Our framing. They measure acceptance only; we show the refund is in *drafting cost* (ceiling 1.91× vs 1.45×) |
| **DictSpec** (UNLP 2026) | Static n-gram / dictionary drafter for Ukrainian and Crimean Tatar; up to 1.65× fewer verification steps, 1.76× with prompt-local n-grams | Closest training-free method. Our dictionary rows reproduce the idea for Hindi and show its limit (31–47% word accuracy) |
| **Naidu** (arXiv 2611.12464) | Phrase/word heads on the target's hidden state, Indic languages on low-fertility Gemma (Hindi 2.09×) | High-fertility tokenizers were "largely unrun". We put heads on the *draft*, under Llama-3's 2.67 tokens/word |
| **Paudel et al.** (arXiv 2605.30580) | N-gram drafters win mainly because drafting is cheap | Supports the "drafting cost is the lever" argument |
| **Medusa / Medusa-2, Hydra, MLP Speculator** | Extra decoding heads on the target, or sequential heads | Our heads are Medusa-1 style, but on the 1B draft and trained only inside words |
| **REST / CREST, SuffixDecoding** | Retrieval / suffix-tree drafting, hybrids with model drafting | Hybrid switching; ours switches on word structure |
| **ReSpec, Talon, METASD** | Adaptive switching between drafters by confidence | Our gate is linguistic (word start vs inside) plus confidence |

The gap we fill: nobody has measured how much of the token tax is *recoverable through cheaper drafting
inside words*. Nobody has tested whether word-aware training of MTP heads (V1) beats plain MTP (V0) in a
high-fertility language.

## 6. Repository layout (current work)

```
PLAN_wordheads.md            active plan (3 coding days, decision points)
src/lgmtp/wordheads.py       heads, word labels, losses, IITB data, training, prediction dump, timing
src/lgmtp/{traces,simulator,policies,boundaries}.py   Phase-1 trace loading, cost model, entropy rule
scripts/build_kaggle_notebooks.py   builds self-contained Kaggle notebooks (`wh` -> 07_wordheads.ipynb)
scripts/smoke_cpu.py         CPU smoke tests (`--part wh` runs the real 1B for 4 steps)
scripts/feasibility/         the CPU feasibility simulations behind section 3
  refund_bound.py            Entropy, ceiling (free continuations), dictionary completer
  gated.py                   confidence-gated dictionary
  noisy.py                   rho curve (break-even)
  diag.py, ctx.py            whole-word accuracy: dictionary (+context) vs 1B
kaggle/07_wordheads.ipynb    GPU run: train V0/V1/V2, dump predictions along the 8B outputs, time heads
tests/test_wordheads.py      label/collate/loss/data-filter tests
src/lgmtp/refund_sim.py      exact simulator: entropy, ceiling, rho curve, dictionary, heads
scripts/run_refund_sim.py    tunes on dev, writes results/wordheads/{main_table.csv,sim_results.json,...}
results/wordheads/           feasibility.json, main_table.csv, sim_results.json, head_accuracy.json, tuning_dev.csv
```

`data/` is git-ignored. The simulations need `data/traces_v2/` (Phase-1 traces) and `../hindi.csv`
(the Hindi word list, one folder above the repo).

## 7. How to run

Setup (Windows, from the repo root):
```bash
python -m venv .venv && .venv/Scripts/pip install -e . pytest
.venv/Scripts/python -m pytest -q                         # 115 passed, 3 skipped
.venv/Scripts/python scripts/smoke_cpu.py --part wh       # ~minutes, downloads Llama-3.2-1B (HF token needed)
```

Feasibility numbers (CPU; each takes a few minutes, `ctx.py` takes longer):
```bash
.venv/Scripts/python scripts/feasibility/refund_bound.py
.venv/Scripts/python scripts/feasibility/gated.py
.venv/Scripts/python scripts/feasibility/noisy.py
.venv/Scripts/python scripts/feasibility/diag.py
.venv/Scripts/python scripts/feasibility/ctx.py
```

GPU run (Kaggle, one T4, about 1–1.5 h):
1. Rebuild if code changed: `.venv/Scripts/python scripts/build_kaggle_notebooks.py wh`.
2. Upload `kaggle/07_wordheads.ipynb` to Kaggle. Settings: GPU T4 ×1, Internet on, and add the secret
   `HF_TOKEN`, which needs access to `meta-llama/Llama-3.2-1B-Instruct`. The notebook falls back to
   `unsloth/Llama-3.2-1B-Instruct`.
3. Run all. Download `/kaggle/working/wordheads.zip` and unzip it into `data/wordheads/`. It contains
   `wordheads_{v0,v1,v2}.pt`, `heads_{dev,devtest}.npz`, `train_info.json`, `run_info.json`, `log.txt`.
4. Sanity check (done locally by `refund_sim.py`): `draft_argmax` in `heads_devtest.npz` should match the
   Phase-1 traces' `draft_argmax` on ≥ 99% of positions.

## 8. Status and next steps

| day | task | status |
|---|---|---|
| 1 | `wordheads.py`, Kaggle notebook 07, smoke test, tests | **done** (smoke passes, row alignment 0.996 on CPU) |
| 2 | `src/lgmtp/refund_sim.py` + tests, `scripts/run_refund_sim.py` (main table) | **done** (Kaggle run done, heads beat Entropy) |
| 3 | `notebooks/08_wordheads_analysis.ipynb` (built by `scripts/build_analysis_notebook.py`): main table, ρ curve with the heads placed on it, head accuracy, training curves, gain by sentence, token-tax table; figures `results/wordheads/fig_*.png` | **done** |

Decision points:
- If the heads keep less than 70% of the 1B's correct continuation tokens, there is no speed-up. The
  paper is then a negative-but-quantified result: the 1.91× ceiling, the context finding, the break-even
  curve, and how far MTP heads get.
- If V1 ≈ V0, word guidance doesn't help MTP heads. We report that.
- Out of scope: a real GPU decoding loop with heads. The simulation is exact for acceptance and was
  within 3–8% of the real loop in Phase 1. Listed as a limitation.
