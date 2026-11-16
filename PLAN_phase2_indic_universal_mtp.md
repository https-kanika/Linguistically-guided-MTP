# PLAN: Phase 2 (pivot): one multi-token drafting head for every LLM, for Hindi and other Indian languages

Written for Claude Code, in the same format as `PLAN_phase1_bugfix.md` and `PLAN_phase2.md`. Read the
whole file before changing anything. Phase 1 is finished (`results/phase1/REPORT.md`). This plan
replaces the research direction of `PLAN_phase2.md`; the parts of that plan that are kept are listed in §6.

---

## 0. Working rules

All rules of `PLAN_phase1_bugfix.md` §0 and `PLAN_phase2.md` §0 still apply: GPU work only in
self-contained Kaggle notebooks built by `scripts/build_kaggle_notebooks.py` and smoke-tested on CPU
first (`scripts/smoke_cpu.py`); logic in `src/lgmtp/`; every number from a file under `results/`;
dev (or training data) for anything fitted or tuned, devtest only for reported numbers; ask the user
about decisions this plan does not cover.

Additional rules for this pivot:

1. **The head never sees token ids.** Its inputs are characters and strings with probabilities
   (`uni_features`), its outputs are word units from a vocabulary built from public Hindi text
   (`unichar.UnitVocab`). Token ids appear only in the final, exact conversion to the target's
   tokens (`uni_sim.convert_all`). A test must fail if a token id reaches the network.
2. **Zero-shot means zero-shot.** For a held-out target, none of its data may be used for training,
   for the unit vocabulary, or for choosing thresholds. The draft-length threshold τ of a zero-shot
   head is chosen on the training models' validation outputs (`week1._val_tau`). Llama-3.2-1B shares
   training data and tokenizer with Llama-3-8B, so every 8B result is also reported for a head
   trained without Llama-family data (`zeroshot_nollama`).
3. **Causality.** At committed position p the head reads `out_ids[:p]` (as text) and the target
   distributions `q_t` for t < p only. Enforced by construction and by
   `tests/test_uni_features.py::test_causality_of_batch_inputs`.
4. **Exactness of the simulation.** Drafts are precomputed for every position; this is exact for
   greedy decoding because the head's input at p does not depend on how p was reached. Conversion
   of the true next units must reproduce the true tokens (`tests/test_uni_sim.py`,
   `conv.csv` row `truth_units`, failed ≈ 0).
5. **Same cost model as Phase 1.** One round costs `c_head + verify(g)` target steps and emits
   `accepted + 1` tokens. `verify(g)` comes from the Phase-1 `timing.json` (linear extrapolation past
   g = 8). `c_head` is measured on the T4 (GPU forward + chained argmax + CPU input building) and divided
   by the measured 8B step.

---

## 1. The idea

**Problem.** Multi-token prediction heads (Medusa, EAGLE, Naidu 2026's phrase heads) are trained per
target model on its hidden states. Indian-language text costs 2–12 tokens per word depending on the
tokenizer (Llama-3 Hindi 2.67, Qwen3 Telugu 11.7), so drafting whole words is where the speed is, but
every new model needs a new head.

**Method.** Draft in *word space*, from the only interface every LLM shares: text and the next-token
distribution. At every round the head reads the committed text, the prompt, and the target's last
M = 3 top-20 distributions with every candidate token written out as a string relative to the
committed text (its probability, whether it is consistent with what was committed, the characters by
which it would extend it, and the char-prefix mass of all candidates that start with it). It outputs
the next k = 3 word units in one pass (head 1 completes the word being written, restricted to units that
extend the committed part; heads 2–3 are chained). The units are converted to the target's own tokens
exactly (retokenise from the start of the current word and check that the committed tokens are
reproduced), and the target verifies them losslessly as usual. Characters of the nine major Brahmic
scripts share one code (the Unicode blocks have the same layout), so one head can later read any of
them.

**Why it should work (Phase-1 evidence).** At equal entropy, word-continuation tokens of Llama-3.2-1B
agree with Llama-3-8B 81–86% of the time, word starts only 66%: completing a word is nearly
model-independent, choosing the next word is where model-specific information (the distribution)
is needed. A perfect drafter of g = 8 tokens is capped at 2.46× by the 1B draft cost; a head with
c ≈ 0.05 that gets about 1.6 tokens accepted per target pass already ties Entropy's 1.452×, and
3 tokens per pass would give about 2.7×.

**What is new** (prior-art search, Nov 2026; absence of evidence is weak evidence):
- per-target heads on hidden states (Medusa, EAGLE-3, Naidu 2611.12464 phrase heads, Gemma MTP) need
  one head per model; Osprey (2609.09338) pretrains target-agnostically but still adapts per target;
- LogitSpec (2507.01449) and ECHO (2609.17241) use logits, but training-free and token-level;
- heterogeneous-vocabulary speculative decoding uses a separate small LM, not the target's distribution;
- Indic work (TokenLift, UniSpec, DictSpec, Yang's EAGLE-3 heads) is per model or retrieval-based.
Not found anywhere: a learned drafter conditioned on the target's distribution as strings, applied
zero-shot to unseen LLMs, or a script-unified Indic drafter.

**Claims the paper wants to make** (each is a W-task below):
1. One head trained on small LLMs drafts zero-shot for an unseen 8B model and beats its tuned
   1B-draft baseline (W1, W2).
2. The distribution input matters: the text-only ablation is clearly worse (W1).
3. Speed-up grows with tokenizer fertility, roughly speed-up ≈ (W·T + 1)/(verify + c) with W words
   accepted per pass and T tokens per word (W3).
4. One head serves several Indian languages and scripts, including scripts it was not trained on (W4).

---

## 2. Repository additions (done in this step)

```
src/lgmtp/unichar.py        Brahmic-unified char codec, word units, sorted unit vocabulary (prefix ranges)
src/lgmtp/uni_features.py   token tables, per-position unit targets, candidate strings, FeatureSet, make_batch
src/lgmtp/unihead.py        the head (PyTorch), training, chained prediction, latency timing
src/lgmtp/uni_sim.py        exact conversion to target tokens, trace simulation, PLD / LogitSpec baselines,
                            next-next coverage, cross-LLM unit agreement
src/lgmtp/collect.py        small-LLM zoo with fallbacks, greedy generation, teacher-forced top-20
src/lgmtp/week1.py          stages S1–S10, verdict, report, zips
tests/test_unichar.py, tests/test_uni_features.py, tests/test_uni_sim.py, tests/fake_tok.py
scripts/smoke_cpu.py --part w1                     CPU smoke test of every stage with tiny models
scripts/build_kaggle_notebooks.py w1  ->  kaggle/05_week1_universal_head.ipynb
results/week1/                                     unzip week1_results.zip here
```

---

## 3. Tasks

### W1. Kill test (Kaggle notebook 05, one run, ~3–4 h on 2×T4)

Inputs: the four small LLMs (`collect.DEFAULT_MODELS`: Qwen2.5-1.5B-Instruct, Gemma-3-1B-it in fp32,
Llama-3.2-1B-Instruct, Sarvam-1 with a one-shot base prompt), FLORES+ dev/devtest English, 3000 IITB
English sentences, 200k IITB Hindi sentences, and **the Phase-1 notebook-01 outputs attached as input**
(`traces_{dev,devtest}.npz`, `target_outputs_*.jsonl`, `timing.json`; or `traces_v2_{dev,devtest}.zip`
uploaded together as one private Dataset if Kaggle allows only one notebook version as input).
`devtest` is needed for the main result, `dev` for the in-model reference heads (recovery criterion).

Stages and acceptance criteria (all written by `lgmtp.week1.week1_main`):

| stage | what | accept if |
|---|---|---|
| S1–S2 | sources; unit vocabulary from IITB Hindi only | `vocab_stats.json`; `vocab_coverage.csv` unit coverage ≥ 0.9 on every model's devtest outputs |
| S3 | generation + teacher-forced top-20 per model and split | `collection_summary.csv`: ≥ 3 models ok, `tf_argmax` ≥ 0.99, `deva_frac` ≥ 0.8 (else drop that model and say so) |
| S4 | Phase-1 8B traces (or fresh collection) | log says `8B data from phase1` |
| S5 | cross-LLM agreement (teacher forced on another model's output) | `agreement.csv`; self rows ≈ 1.0 (sanity) |
| S6 | features | per-model positions logged; `head1_target_ok` ≥ 0.85 |
| S7 | text-only pretraining; head latency | `timing_head.json` with `c_head_measured` |
| S8 | leave-one-model-out (zero-shot / text-only / in-model + PLD, LogitSpec, truth units) | `lomo.csv`, `lomo_sim.csv` |
| S9 | 8B: zero-shot (all / no-Llama), text-only, in-model (fine-tuned / 8B-dev only), Entropy and FixedK recomputed on the same traces, PLD, LogitSpec, truth units, τ grid, c sensitivity | `l8.csv` with bootstrap CIs and paired differences vs Entropy; `conv.csv` truth-unit `failed` ≤ 1% |
| S10 | verdict | `verdict.json`, `SUMMARY.md` |

**Decision rule (implemented in `week1.verdict`):** with S = speed-up of the zero-shot head (all small
LLMs) on 8B devtest, S_in = the in-model head, recovery = (S − 1)/(S_in − 1):
- **GO**: S ≥ 1.6 and recovery ≥ 0.7;
- **WEAK GO**: S beats Entropy with the paired CI above 0, and recovery ≥ 0.7;
- **NO GO** otherwise.
Report separately whether the distribution input helps (zero-shot vs text-only, paired CI above 0) and
the no-Llama result. A GO with "distribution does not help" is still a usable result (a universal
word-level drafter), but the paper's main novelty then weakens; discuss with the user before W2.

**Reading a NO GO** (all diagnostics are already in the tables):
- `Truth units` speed-up low → the unit/conversion design caps the method (fix units or k first);
- truth units high but heads low, `acc1` high and `acc2/acc3` low → next-word prediction is the
  bottleneck: give the head more context or a tree of top-2 units (W5) before giving up;
- in-model high, zero-shot low → transfer fails: more source models, or a short per-target calibration
  (then the claim becomes "minutes of adaptation", compare with Naidu's training cost);
- oracle-length rows far above the τ rows → draft-length selection is the problem, not drafting.

### W2. Real loop and losslessness on the 8B (after a GO; ~3 Kaggle h)

1. P0 of `PLAN_phase2.md` (EOS overrun fix) first, behind a flag.
2. Add a `UniversalHeadDrafter` to `spec_loop.py`: per round, build the head's input from the committed
   text and the cached distributions, draft units, convert, verify; reuse the existing token-by-token
   losslessness check against plain greedy decoding.
3. 100 devtest sentences (the Phase-1 real-loop ids), 3 repeats, Entropy and the head on the same run.
   **Accept if** outputs are identical to greedy for every sentence and real tokens per round are within
   3% of the simulation; otherwise report the decomposition as in Phase 1 §10.3.

### W3. More targets and the fertility law (~4 Kaggle h)

Targets: Qwen3-4B (or 8B), Gemma-3-4B, Sarvam-1 as a target, Llama-3-8B. For each: traces on FLORES
devtest (notebook-01 code path), zero-shot simulation, and the measured tokens per Hindi word T.
Fit speed-up against W·T and test the formula. Baseline for each target: a same-family small draft
with Entropy tuned on dev where one exists, otherwise prompt lookup.

### W4. Languages and scripts (~4 Kaggle h)

Marathi (Devanagari), Bengali, Tamil, Telugu with FLORES; train on some, hold out Gujarati / Odia /
Kannada scripts entirely (zero-shot script transfer through the unified codec). The unit vocabulary
becomes per language (built from public text of that language, rule 2); report OOV rates. Tamil has no
aspirated stops and Urdu / romanised Hindi fall outside the codec: state these limits.

### W5. Tree drafting and write-up

Top-2 units per head as a small tree verified in one pass (verify cost measured for tree sizes); then
the paper. The Phase-1 finding (linguistic boundaries add nothing beyond confidence for *stopping*)
becomes the motivation section: boundaries matter for *what* to draft, not *when* to stop.

---

## 4. Checkpoints

- **C1 (after W1):** GO / WEAK GO / NO GO as defined above. On NO GO, stop and discuss with the user,
  bringing the diagnosis from the tables.
- **C2 (after W2):** the real loop is lossless and within 3% of the simulation; else fix before W3.
- **C3 (after W3):** the fertility relation holds across ≥ 3 targets (paper claim 3) or is reported as
  not holding.
- **C4 (after W4):** zero-shot to an unseen script works (claim 4) or the paper is scoped to
  "every LLM, Hindi and Marathi".

---

## 5. Risks

- Per-target heads are cheap (Naidu trains one in minutes). The value must rest on zero-shot use
  (new models on day one, languages without per-model data) and on the science (what transfers).
- The head reads the prompt; on translation it is partly a tiny translation model. W3/W4 should add
  one non-translation task (Indic chat or summarisation) so the claim is not translation-specific.
- Script unification is approximate (see W4). The vocabulary is closed; OOV words end the draft.
- The field moves fast: re-run the prior-art search before writing.

---

## 6. What happens to `PLAN_phase2.md`

Kept: P0 (EOS fix, now in W2), P5 (Marathi, now in W4), P7.1 fertility scan (now W3). P2 (failure
taxonomy) becomes a diagnostic of W1's errors. P4 (tree bounds) is absorbed into W5. Dropped: P3
(learned stopping controller), P6, P8, P9.

---

## 7. Budget

| week | GPU (Kaggle h) | CPU / analysis |
|---|---|---|
| 1 | W1 ≈ 3–4 | review `results/week1/`, decide at C1 |
| 2 | W2 ≈ 3 | `UniversalHeadDrafter`, smoke tests |
| 3 | W3 ≈ 4 | fertility analysis |
| 4 | W4 ≈ 4 | scripts, write-up start |

---

## 8. Decisions to ask the user

1. After W1: how to proceed on WEAK GO or on "distribution does not help".
2. W3: which larger targets fit the Kaggle budget (Qwen3-8B needs both T4s).
3. W4: which languages, and whether to add a non-translation task.
4. Whether Sarvam-1's base-model prompt is acceptable or should be replaced by an instruct model.
