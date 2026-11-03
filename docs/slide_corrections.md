# Slide / script corrections (PLAN T10)

Suggested replacement wording for the presentation, based on the Phase 1 re-run. Your slides have
not been edited. Every number below comes from a file in `results/phase1/` (named next to it). Setup
for all speed-ups: 8B target / 1B draft on 2× T4, tuned on FLORES+ **dev**, reported on FLORES+
**devtest** (1012 sentences), measured verification cost, `c = 0.323`. 95% CIs are sentence
bootstraps (1000 resamples).

Real-loop numbers (T9) come from Kaggle notebook 03 → notebook 04 (`results/phase1/REPORT.md` §9).

---

## Slide 2: motivation

**Remove:** "When a draft ends mid-word or mid-morpheme, acceptance drops."
T8 does not support this as a causal statement. Verification is token-level, so ending mid-word
cannot by itself reduce acceptance. What the data shows is *where* drafts fail, and that confidence
already explains it.

**Replace with (measured):**
- "The Llama-3 tokenizer splits Hindi into 2.67 tokens per word; 54.3% of token cuts fall *inside an
  akshara* and a fixed 4-token draft ends inside a word 60.6% of the time."
  (`test1_regression.csv`, `test1_summary.json`)
- "Drafts fail mostly at the *start* of words: the draft matches the target on 66% of word-initial
  tokens vs 81–86% of later tokens in a word, and draft entropy is 2–3× higher there
  (1.52 vs 0.52–0.74 nats)." (`analysis_devtest_by_pos_bucket.csv`)
- "But draft entropy already captures this: at equal entropy, word-initial tokens are *not* harder
  (+1.4 points match rate [+0.8, +2.0], entropy deciles), and adding word/morpheme position to
  entropy does not improve failure prediction (ΔAUC −0.0001 [−0.0004, 0.0000])."
  (`analysis_conditional_summary.csv`, `analysis_conditional_devtest_logit.csv`)

## Slide 3: method (hybrid / linguistic policies)

**Old description** (hybrid = "akshara/word boundary stop"): it did not match the code that ran.
The old `HybridPolicy` had no entropy term, and its akshara check read the *next* token for free
(bugs B2, B3).

**Replace with:**
- "**Hybrid**: stop when the entropy rule fires (1 − √(βH) < λ) *and* the draft predicts the word
  is complete (P(next token starts a new word) > τ_w); otherwise extend by at most `max_ext` tokens."
- "**AksharaSafeEntropy**: same, but the condition is that the next token is unlikely to be a
  combining mark (P(mark) < τ_m), i.e. the last akshara is probably complete."
- "**MorphEntropy**: same, with a causal morpheme-boundary model P(boundary | word prefix)
  built from MorphTok + dev outputs."
- "Word/akshara completeness depends on the *next* token, so these policies need one extra draft
  forward (lookahead). We charge its cost (`c`) when the policy stops by itself, and also report
  the uncharged bound."
- "All policies are tuned on dev with grids of comparable size and evaluated once on devtest."

## Slide 5: results

**Remove:** "Morpheme result is a coverage failure: MorphTok covers only 41%."
The old "morpheme lookup" row never used MorphTok: it was an 11-word placeholder dictionary (bug B1).
The old word-boundary and hybrid rows are also invalid (B2, B3).

**Replace the results table with** (`main_table.csv`):

| policy | speed-up [95% CI] | Δ vs Entropy |
|---|---|---|
| Oracle (upper bound, cap 8) | 1.904 [1.895, 1.914] | +0.452 |
| **Entropy** (λ=−0.683, β=4) | **1.452 [1.444, 1.461]** | 0 |
| MorphEntropy, morph model consulted (max_ext ≥ 1) | 1.427 [1.418, 1.437] | −0.025 |
| Fixed k=2 (best fixed) | 1.357 [1.350, 1.366] | −0.094 |
| Gold word-boundary oracle | 1.353 [1.345, 1.361] | −0.099 |
| MorphOnly | 1.343 [1.335, 1.353] | −0.108 |
| Hybrid (lookahead charged / uncharged) | 1.253 / 1.424 | −0.199 |
| AksharaSafeEntropy (charged / uncharged) | 1.239 / 1.430 | −0.212 |
| WordBoundarySoft (charged / uncharged) | 1.161 / 1.341 | −0.291 |

Suggested headline: "**No linguistic stopping rule beats the entropy rule.** Even stopping at the
*true* word boundaries (an oracle) gives only 1.35×. With the morpheme model, coverage is
67% of word tokens from the MorphTok lexicon and 31% from the CRF segmenter (held-out boundary F1
0.93)." (`morph_coverage.json`, `morph_crf_eval.json`)

Real loop (wall clock, 100 devtest sentences × 3 repeats, 2× T4): "On real hardware, Entropy gives
**1.356× [1.328, 1.381]** over plain greedy decoding and is still the best policy. MorphEntropy is
1.323× (Δ −0.033 [−0.044, −0.022]), fixed k=2 1.304×, Hybrid 1.145× and WordBoundarySoft 1.096×.
The simulator ranks the policies in the same order. In absolute terms it is 3–8% optimistic (Entropy
predicted 1.440×)." (`real_loop_ci.csv`, `sim_vs_real.csv`, `sim_vs_real_decomposition.csv`)

If a slide shows the real-loop table, add: "285–288 of 300 decodes are token-identical to unbatched
greedy decoding. The no-draft baseline is too (288/300), so the remaining divergences are fp16
near-ties (top-2 logit gap ≤ 0.016), not speculation errors." (`real_loop_lossless.csv`,
`real_loop_divergences.csv`)

Do **not** say "lossless in 100% of decodes".

## Any slide quoting the Qwen tokenizer

**Remove:** "Qwen3.5-0.8B: 3.33 tokens/word". That number came from an older transformers
loading path that applied the Qwen2 pre-tokenizer regex (it splits before every matra). With the
model's own `tokenizer.json`, Qwen3.5-0.8B gives **2.16 tokens/word**. Swapping in the old regex
reproduces 3.33 exactly. (`test1_regression.csv`, `test1_summary.json`)

## Numbers that are unchanged (still valid)

The old fixed-k / entropy / oracle numbers were re-checked with the old protocol on the new traces
(random half of devtest, unit verify): fixed k=2 1.378 (old 1.361), entropy 1.473 (old 1.452),
oracle 1.934 (old 1.905), all within 0.03. (`legacy_protocol.json`) For new slides, quote the
dev→devtest numbers above, not these.

## Caveats to say out loud

- Speed-ups are simulated from teacher-forced traces. Accepted tokens are exact; only the stopping
  decision after a mismatch is approximated. On the GPU, that approximation costs only 0.2–1.4%
  of speed-up, and tokens per verification match within 0.4%. The simulator is still 3–8% too high
  overall, mostly because our live loop keeps drafting past the end-of-sentence token in the last
  round (≈5 wasted draft tokens per sentence for Entropy). That is an implementation inefficiency
  that hits every drafting policy, not a ranking issue. Fixed, it would put Entropy at about 1.40×.
  (`sim_vs_real_decomposition.csv`)
- `c` was measured with the 8B target split over two T4s, so each target step includes a GPU-to-GPU
  hop. On a single large GPU, `c` and the speed-ups would differ.
- Lookahead policies are shown with the extra draft forward charged (conservative) and uncharged
  (optimistic). Even uncharged, none beats Entropy.
