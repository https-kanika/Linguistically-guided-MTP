# PLAN: Word-completion heads on the draft model ("a refund for the token tax")

Replaces `PLAN_mtp.md` as the active plan. Three days of coding, then analysis and write-up.
Rules from earlier plans still apply:
- GPU work only in self-contained Kaggle notebooks built by `scripts/build_kaggle_notebooks.py`, smoke-tested on CPU first (`scripts/smoke_cpu.py`).
- Logic lives in `src/lgmtp/`.
- Every number comes from a file under `results/`.
- Anything fitted or tuned uses dev (or training data); devtest is only for reported numbers.

---

## 1. Idea in one paragraph

Hindi costs 2.67 Llama-3 tokens per word, and **59% of the tokens the 8B writes continue a word**.
In Phase 1 the 1B draft model paid full price for every drafted token (c = 0.323 of an 8B step),
even inside words, where it is accurate (matches the 8B on 81–86% of continuation tokens, 66% at word
starts). We let the 1B draft only the **first token of each word** with its normal LM head, and let
small **multi-token heads** on the same forward pass propose the rest of the word. The 1B then
reads the completed tokens in one forward (about the cost of one token) and drafts the next word start.
The 8B verifies as usual, so outputs are identical to greedy decoding (lossless).

## 2. Evidence already in hand (CPU, Phase-1 devtest traces; `results/wordheads/feasibility.json`)

| drafter | speed-up [95% CI] |
|---|---|
| Entropy, 1B drafts every token (Phase 1) | 1.452 [1.444, 1.461] |
| Upper bound: continuation tokens drafted at 5% of a 1B call, as accurately as the 1B | 1.91 [1.89, 1.93] |
| Context-free dictionary completer | 1.224 [1.217, 1.232] |

Speed-up when a completer keeps only a share ρ of the 1B's correct continuation tokens:
ρ = 1.0 → 1.91, 0.9 → 1.72, 0.8 → 1.57, **0.7 → 1.45 (break-even)**, 0.6 → 1.36.

Whole-word completion accuracy (devtest, 38.7k mid-word positions):

| completer | accuracy |
|---|---|
| prefix only | 31.3% |
| + previous word / sentence cache | 34.9% |
| + the reference translation's words (oracle) | 47.3% |
| **1B draft model** | **77.2%** |

So context from a lexicon is not enough. The context must come from a model, and the 1B's hidden
state already carries it.

Related work to cite and position against:
- "No Refund for the Token Tax" (2610.*): no *acceptance* refund in high-fertility languages. We target *drafting cost*.
- DictSpec (UNLP 2026): dictionary drafter, Ukrainian.
- Naidu 2611.12464: phrase heads on the target, low-fertility Indic only.
- Paudel et al. 2605.30580: n-gram drafters win through cheap drafting.
- Medusa / Hydra: heads on the target.
- ReSpec / SuffixDecoding hybrids: switching decided by confidence, not linguistics.

## 3. Method

Position convention (as in the Phase-1 traces): `h_t` is the 1B's last hidden state that predicts
output token `y[t]`.
- Head k (k = 1..K, K = 3) on `h_t` predicts `y[t+k]`, with a Medusa-1 residual block (`z = h + SiLU(W_k h)`, `W_k` zero-initialised) and the frozen, shared LM head.
- A continuation head `Linear(d, K)` predicts `c_k[t]` = 1 if `y[t+1..t+k]` all continue the word of `y[t]`.

Training: the 1B is frozen. Teacher forcing on IITB en→hi references with the Phase-1 prompt.
One 1B forward feeds all variants (same data, same order):

| variant | loss |
|---|---|
| V0 plain MTP | Σ_k 0.8^(k-1) CE_k over all positions |
| V1 word-guided | CE_k only where `c_k = 1` (inside the word), + 0.5 · BCE(continuation head) |
| V2 adaptive | V1's K+1 losses with uncertainty weighting `Σ exp(-s_i) L_i + s_i` |

Drafting (simulated exactly on the 8B traces): round at committed position p.
1. At a word start, the 1B drafts `y[t]` (cost c·r(q), where q is the number of new tokens it reads).
2. If the token starts a Devanagari word, heads 1..K propose `y[t+1..]`. Stop at the first proposal that is word-initial, below confidence τ_h, or, for V1/V2, has continuation probability < τ_c. Cost: c_head.
3. The 1B continues after the proposed tokens.

The entropy rule (Phase 1) decides when the round ends, checked on 1B tokens. τ, λ are tuned on dev.

Rows of the main table:
- Entropy;
- upper bound;
- dictionary, both plain and confidence-gated;
- V0, V1, V2;
- V1 with lexicon constraint (head top-8 filtered to tokens that continue a real Hindi word);
- oracle-ρ curve with every method placed on it.

## 4. Tasks (3 coding days)

| day | task | accept if |
|---|---|---|
| 1 | `src/lgmtp/wordheads.py` (heads, labels, losses, training, dump along 8B outputs, timing); builder `wh` → `kaggle/07_wordheads.ipynb` (8B outputs embedded); `smoke_cpu.py --part wh` with Qwen2.5-0.5B | smoke passes; dump rows align (1B argmax equals Phase-1 `draft_argmax` on ≥ 99% of positions on Kaggle) |
| 2 | `src/lgmtp/refund_sim.py` + tests (reproduces Entropy 1.452 at full cost; upper bound; dictionary; heads) | tests pass |
| 3 | `notebooks/08_wordheads_analysis.ipynb`: main table with paired CIs, ρ curve, per-position accuracy V0 vs V1, tokenizer "token tax" table | all numbers in `results/wordheads/` |

Kaggle budget: one run of about 1–1.5 h on one T4.

## 5. Decision points

- **Heads' retained share ρ < 0.7:** no speed-up over Entropy. The paper still has the ceiling (1.91×), the context finding, the break-even curve, and how far MTP heads get.
- **V1 ≈ V0:** linguistic guidance does not help MTP heads. Report it, as for Phase 1.
- **Not done in 3 days:** real GPU loop with heads (the simulation is exact for acceptance and was within 3–8% of the real loop in Phase 1). List it as a limitation.
