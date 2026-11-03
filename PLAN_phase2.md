# PLAN: Phase 2: from "do boundaries help?" to a paper (CS769)

Written for Claude Code, in the same format as `PLAN_phase1_bugfix.md`. Read the whole file before
changing anything. Phase 1 is finished; its results are in `results/phase1/REPORT.md`, and this plan
builds on them.

---

## 0. Working rules

All working rules from `PLAN_phase1_bugfix.md` §0 still apply. In short:

- GPU work is done in self-contained Kaggle notebooks built by `scripts/build_kaggle_notebooks.py`,
  smoke-tested on CPU first. Use `scripts/smoke_cpu.py`, including the `--noise` mode, so rejections
  and cache cropping are exercised.
- Logic lives in `src/lgmtp/`, and notebooks stay thin.
- Causality is enforced by `TraceView` and tests.
- Every number comes from a file under `results/`.
- Ask the user about any decision this plan does not cover.

Additional rules for Phase 2:

1. **Simulate first, use the GPU last.** Phase 1 showed the simulator is accurate on this setup:
   tokens per verification are within 0.4% of the real loop, and the teacher-forcing error is 0.2–1.4%
   of speed-up. Every new policy and controller is therefore developed and selected on CPU from
   traces. A GPU real-loop run happens only once, at the end (P10).
2. **Dev for everything you fit or tune, devtest only for the final numbers.** This includes
   classifier training *and* threshold selection. Use cross-fitting inside dev (§P3) so the
   threshold is never tuned on predictions for data the classifier was trained on.
3. **Comparable tuning budgets.** Every new policy or controller gets a dev grid of a size comparable
   to Entropy's, and is compared with Entropy by a paired bootstrap on devtest (1000 resamples).
4. **Language and model are parameters, not forks.** Code that is Hindi- or Llama-specific gets a
   `lang` / `model_pair` argument, not a copy. Results go to `results/phase2/<lang>_<pair>/`.
5. **Keep Phase 1 results reproducible.** Do not change defaults in ways that alter
   `results/phase1/`. Where a fix changes behaviour (e.g. P0.1), put it behind a flag whose default
   keeps the old behaviour for re-running Phase 1, and turn it on explicitly in Phase 2 configs.

---

## 1. Where Phase 1 left us

All numbers below are FLORES+ en→hi devtest, Llama-3-8B target with a Llama-3.2-1B draft, c = 0.323.

| finding | number | source |
|---|---|---|
| Entropy is the best policy (simulated) | 1.452× [1.444, 1.461] | `main_table.csv` |
| Best fixed draft length (k=2) / oracle | 1.357× / 1.904× | `main_table.csv` |
| Share of the fixed-k → oracle gap that Entropy closes | ≈17% | `main_table.csv` |
| Best linguistic policy (MorphEntropy, morph model consulted) | 1.427× (Δ −0.025) | `main_table.csv` |
| Stopping at the true word boundaries (oracle) | 1.353× | `main_table.csv` |
| Word-initial tokens at equal entropy | +1.4 points match rate [+0.8, +2.0] | `analysis_conditional_summary.csv` |
| ΔAUC from adding word/morpheme position to entropy | −0.0001 [−0.0004, 0.0000] | `analysis_conditional_devtest_logit.csv` |
| Real loop, Entropy (100 sentences, wall clock) | 1.356× [1.328, 1.381]; same ranking as the simulator | `real_loop_ci.csv` |
| Simulator optimism, and its main cause | 3–8%; mostly the end-of-sentence overrun (≈3.3% for Entropy) | `sim_vs_real_decomposition.csv` |
| Qwen3.5-0.8B fertility | 2.16 tokens/word (*lower* than Llama's 2.67) | `test1_summary.json` |

**What this means for the plan.**

- **The original Phase 2 question is already answered for Hindi, at the token level.** Linguistic
  position adds nothing to draft confidence. The A vs A+B ablation is still worth doing as a
  *stopping* experiment, but expect a null result for Hindi.
- **The morphology claim can only be settled in Marathi**, where morpheme boundaries differ from word
  boundaries much more than in Hindi. Marathi is required, not optional.
- **There is still a lot of room above Entropy** (the oracle is at 1.90×), so a better controller
  based on confidence and context is a plausible positive result whatever happens with linguistics.
- **Linguistic position may help decide *where to branch*** (tree drafting) rather than where to
  stop: drafts fail at high-entropy, word-initial tokens.
- **Paper framing.** The default is a characterization paper: "draft confidence already accounts for
  word, akshara and morpheme structure in lossless Indic speculative decoding". It is strengthened by
  Marathi, an English control, a second tokenizer and the validated simulator, plus whatever the
  controller and tree experiments give. Switch to a positive paper only at checkpoint C2 (§4).

---

## 2. Repository additions

```
src/lgmtp/
  controller.py        # P3: feature builder (causal), classifiers, cross-fitting, ControllerPolicy
  oracle_gap.py        # P1: early/late stopping decomposition
  taxonomy.py          # P2: mismatch classification
  tree_sim.py          # P4: tree / branching estimates from traces
  sampling_sim.py      # P8: speculative sampling estimates from top-k log-probs
  langs.py             # P5/P6: per-language config (FLORES code, script regex, word rules, morph source)
scripts/
  fertility_scan.py    # P7: tokenizer fertility + shared-tokenizer check for candidate pairs
notebooks/
  05_oracle_gap_taxonomy.ipynb
  06_controller.ipynb
  07_tree_sampling.ipynb
  08_crosslingual.ipynb
kaggle/
  01_traces_v2.ipynb   # rebuilt with LANG / PAIR / TASK config
  03_real_loop.ipynb   # rebuilt with the EOS fix and the controller (P10)
results/phase2/<lang>_<pair>/...
docs/paper_outline.md  # P11
```

---

## 3. Tasks

Tasks are grouped into three tiers. Tier A needs no GPU and comes first. Each task lists its
**owner** (K = Kanika, R = Rakshit) and **acceptance criteria**.

### Tier A: CPU only, on the existing Hindi traces

#### P0. Housekeeping (R)

1. **End-of-sentence fix in `spec_loop.py`.** Stop drafting in a round once the draft has emitted
   an end-of-sentence token, and also cap the draft at `n_remaining = max_new - p`. This is behind
   `stop_at_draft_eos=True`, default False for Phase 1 reproducibility. The simulator already behaves
   this way.
   - *Accept:* a unit test where the draft emits EOS mid-round gives `g` = position of EOS + 1.
   - *Accept:* `smoke_cpu.py --part 03 --noise 2` is still lossless.
   - *Accept:* re-running notebook 04's decomposition on smoke output shows zero EOS overrun.
2. **Parametrize language and pair.**
   - Add `langs.py` with, per language:
     - FLORES+ code;
     - source/target direction and prompt;
     - akshara regex (Devanagari covers Hindi and Marathi);
     - word-boundary rule;
     - morpheme source.
   - Add `LANG`, `PAIR` and `TASK` config keys to notebook 01, and make `boundaries` / `morph` take
     `lang`.
   - *Accept:* the Hindi defaults reproduce `results/phase1/traces_summary.json` exactly on the
     existing traces.
   - *Accept:* the tests pass.

#### P1. Where does Entropy lose to the oracle? (R)

For every devtest round of the selected Entropy policy, compare with the oracle at the same
committed position:

- **stopped too early:** Entropy stopped while the next draft token would have been accepted. Count
  the accepted tokens lost.
- **stopped too late:** tokens drafted after the first mismatch. Count the draft cost wasted.

Convert both into speed-up units, so that early + late + interaction = oracle − Entropy. Break the
result down by the word position, token type and entropy decile of the stopping position.

- *Accept:* the decomposition adds up to the oracle gap within 0.001.
- *Accept:* `results/phase2/hi_llama/oracle_gap.csv` and a stacked-bar figure are written.
- *Decision this informs:* if most of the gap is "too early" at low-entropy positions, a context
  feature (P3, group C) is the most promising lever; if it is "too late", better confidence
  calibration is.

#### P2. Failure taxonomy of mismatches (K)

For every devtest position with `match == 0`, record:

- the rank of the target token in the draft's top-20 (1 = near miss … >20 = miss);
- the target and draft token strings (byte-safe);
- word position and token type.

Rule-based classes:

- **inflection variant:** same word stem, different suffix or matra, using the akshara and
  morph-label tools;
- **numeral or punctuation;**
- **Latin script or transliterated named entity;**
- **space or segmentation difference:** the same text split into different tokens;
- **other lexical choice.**

Hand-check a random sample of 200 mismatches (K) to estimate how accurate the rules are.

- *Accept:* `taxonomy.csv` with class shares and the target's rank within the draft top-k by class.
- *Accept:* agreement between the rules and the hand labels is reported.
- *Accept:* a short written conclusion in the notebook.
- *Why:* it tells you whether failures are "draft knows the right token but ranks it second" (which
  motivates P4 branching) or "draft has no idea".

#### P3. Learned stopping controller (R, with K for group B features)

**Target.** Predict `match[t]` at each draft position t, using only features available when token t
has been drafted. Those are the draft distribution at t and earlier drafted positions in the round,
plus the committed history. No lookahead: no `p_*` read at t+1.

**Feature groups**:

- **A, confidence:** H, top-1, margin (top-1 − top-2), log top-1.
- **B, linguistic:**
  - p_newword, p_mark, p_space, p_punct;
  - position in the current word (from drafted bytes);
  - word-initial flag;
  - prefix-model morpheme probability (as used in the Phase 1 policies).
- **C, context:**
  - acceptance rate so far in this sentence;
  - length of the last round and whether it was fully accepted;
  - position in the draft round (g);
  - relative position in the sentence;
  - running product of predicted acceptance.

**Models**:

- logistic regression, with standardized features and an L2 penalty tuned by cross-validation;
- a gradient-boosted tree model (sklearn `HistGradientBoostingClassifier`);
- optionally a 2-layer MLP.

**Stopping rule:** keep drafting while ∏ P̂(accept) over the round ≥ θ (the SpecDec++-style
cumulative rule). Cap at 8.

**Fitting protocol (no leakage):**

1. Split dev sentences into 5 folds. Train on 4 folds and predict the 5th, giving out-of-fold
   predictions for all of dev.
2. Tune θ (a grid of 40 values) on dev using those out-of-fold predictions in the simulator.
3. Refit the classifier on all of dev, then evaluate once on devtest with the selected θ.

Implement the controller as a `Policy` subclass (`ControllerPolicy`) so that it runs unchanged in the
simulator and in the real loop. Features are read only through `TraceView` / `LiveView`.

**Headline ablation table** (devtest):

| features | model | speed-up [CI] | Δ vs Entropy [CI] | Δ vs A [CI] | oracle-gap closed | AUC |
|---|---|---|---|---|---|---|
| A, A+B, A+C, A+B+C | LR, GBT | | | | | |

- *Accept:*
  - the causality guard raises if any feature reads beyond `p+g-1`;
  - a test checks that out-of-fold predictions never use their own fold;
  - `results/phase2/hi_llama/controller_*.csv` is written;
  - the feature importances / coefficients are reported.
- *Read first:* DISCO (Mamou et al., 2024, "Dynamic speculation lookahead") and SpecDec++ (Huang et
  al., 2024). Cite them, and state what is new: script- and morphology-aware features, and the
  multilingual comparison.

#### P4. Branching at uncertain positions: a feasibility estimate (R)

**Question:** if the draft proposes its top-b tokens (b = 2 or 4) at positions selected by a rule,
how many more tokens would the target accept? Candidate rules:

- **entropy rule:** branch when H > h;
- **linguistic rule:** branch when the drafted token is word-initial;
- **both.**

**What traces can and cannot give.** The traces contain the draft top-20 and the target argmax at
every position, conditioned on the target prefix. At a branching position t, "the target token is
among the draft's top-b" is exact. Draft continuations *after* a non-top-1 branch were never
computed, so:

- **lower bound:** a successful branch adds one token, and the round ends there;
- **upper bound:** after a successful branch, the draft continues as it does in the teacher-forced
  trace.

Report both bounds. The cost model for tree verification needs the verify cost for a tree of
`n` nodes: assume verify(n) from the chain timing as a first approximation, and flag it.

- *Accept:* `tree_bounds.csv` with speed-up bounds for each rule and b.
- *Accept:* a go / no-go note. **Go** (continue to a real tree loop in Phase 3 / future work) only if
  the lower bound beats Entropy's chain speed-up by more than its CI width.
- *Accept:* a test of the bound computation on a hand-built trace.

### Tier B: new traces on GPU, about 6–10 Kaggle hours in total

#### P5. Marathi (K labels, R traces)

1. **Traces.** FLORES+ en→mar (`mar_Deva`), dev + devtest, Llama-3-8B / 3.2-1B, through rebuilt
   notebook 01 with `LANG=mar`. Check first, on CPU, what share of Llama-3's Marathi outputs are
   actually Marathi rather than Hindi or English. Use script and word-list checks on 20 sentences
   produced with the 1B model as a stand-in, and ask the user if the share is low.
2. **Morpheme labels.**
   - Check whether MorphTok (or another resource) has Marathi segmentations, and **ask the user**
     which source to use.
   - Fallback order: an existing segmented resource → a CRF trained on it → Morfessor trained on
     Marathi monolingual text (e.g. IndicCorp / the FLORES dev source side). Report coverage and
     held-out boundary F1 per tier, as for Hindi.
   - Also report how often a morpheme boundary is *not* a word boundary, in Hindi and in Marathi.
     This is the number that justifies Marathi.
3. **Rerun everything on Marathi:** the Phase 1 tuning and evaluation (`tune_all` / `evaluate`), the
   conditional analysis (§8.3 of the Phase 1 report), P1, P2 and P3.

- *Accept:* `results/phase2/mar_llama/` contains the same set of files as `results/phase1/`, plus the
  P1–P3 outputs.
- *Accept:* a Hindi-vs-Marathi comparison table in `08_crosslingual.ipynb`.

#### P6. English control (K)

Same pipeline with English as the *target* language. **Ask the user** which direction:

- hi→en (same FLORES sentences, recommended);
- another source language →en.

Akshara features are undefined for English (p_mark ≈ 0). Word and morpheme features use a simple
English segmenter (Morfessor, or a suffix list), with its accuracy noted.

- *Accept:* `results/phase2/en_llama/` with the main table, the conditional analysis and the P3
  ablation.

#### P7. A second model pair / tokenizer (R)

1. **`scripts/fertility_scan.py`** (CPU). For candidate families with a shared tokenizer, run the
   Test 1 metrics on FLORES Hindi and Marathi: tokens/word, % of cuts inside an akshara, % of cuts
   at word boundaries. Candidates, all to be checked:
   - Gemma-3 1B with a 4B target (fits on one T4);
   - Qwen3 0.6B with a 4B or 8B target;
   - an Indic-specific family if a matched draft/target pair exists.
   Verify that draft and target share a tokenizer, and record the license and gating for each.
2. **Ask the user** to pick the pair. Recommend the one whose fertility and %-mid-akshara differ most
   from Llama's, preferring a target that fits on one GPU. That avoids the layer split across two
   T4s and checks whether c = 0.323 drives the Phase 1 conclusions.
3. **Traces + timing** for Hindi (and Marathi if the budget allows), then the Phase 1 evaluation,
   the conditional analysis and the P3 ablation.

- *Accept:* `results/phase2/hi_<pair>/`.
- *Accept:* a fertility-vs-effect table: does the linguistic ΔAUC or Δspeed-up change with
  fertility?

### Tier C: extensions, done if time allows

#### P8. Speculative sampling at temperature 0.7 (R)

The traces store the draft and target top-20 log-probs. At T = 0.7, compute the expected acceptance
Σ_x min(q(x), p(x)) restricted to the top-20 union, and bound the probability mass outside it.

- *Accept:* `sampling_estimates.csv` with expected tokens per verification for Entropy and the
  controller, with truncation bounds.
- *Accept:* a note on whether the ranking of policies changes compared with greedy.
- For exact numbers, add `temperature` to `SpecDecoder` and include it in P10 (only if the user
  approves the GPU time).

#### P9. One open-ended task (K)

Hindi summarization, e.g. XL-Sum Hindi (200 dev / 200 test articles, truncated). Run notebook 01
with `TASK=sum`, then the main table and the conditional analysis.

- *Accept:* `results/phase2/hi_llama_sum/`.

#### P10. Final real-loop run (R, GPU, once)

Rebuild notebook 03 with:

- `stop_at_draft_eos=True`;
- the policies NoDraft, FixedK, Entropy, the best controller (A or A+C), and the best controller with
  B features.

Run 100 devtest sentences × 3 repeats for Hindi. Add Marathi if P5 found anything interesting.
Notebook 04 reports wall-clock speed-up, losslessness and simulated-vs-real.

- *Accept:* the simulated-vs-real gap is ≤ 3% for every policy after the EOS fix. If not, explain it
  with the decomposition.
- *Accept:* paired CIs for controller − Entropy.

#### P11. Write-up (K + R)

`docs/paper_outline.md`:

- research question;
- setup;
- validated simulator (Phase 1 §9);
- where drafts fail (position, taxonomy);
- conditional analysis (Hindi, Marathi, English, second tokenizer);
- controller ablation;
- tree / sampling (if done);
- limitations.

Every figure comes from a `results/` file. Updated slides follow `docs/slide_corrections.md`.

---

## 4. Checkpoints

- **C1 (after Tier A):**
  - P3 shows the controller beats Entropy (A or A+C, paired CI above 0): the paper has a
    constructive result.
  - P3 shows B adds nothing in Hindi: expected; go on to P5.
- **C2 (after P5 Marathi):**
  - A+B beats A in Marathi with a paired CI above 0 (and not in Hindi or English): **positive
    paper** about morphology-aware drafting for agglutinative Indic languages.
  - Otherwise: **characterization paper**, "confidence subsumes linguistic structure", across
    languages and tokenizers, with the controller and tree results as the constructive part.
- **C3 (after P4):** tree go / no-go, as defined in P4.
- **If any real-loop result disagrees with the simulator by more than 3% after the EOS fix:** report
  the real numbers and the decomposition, and do not tune further on the simulator for that setting.

---

## 5. Order and budget

| week | Kanika | Rakshit | GPU (Kaggle h) |
|---|---|---|---|
| 1 | P2 taxonomy; Marathi morph resource search (P5.2) | P0, P1, start P3 | 0 |
| 2 | P5.2 labels + Marathi checks; P6 setup | P3 ablation; P4 bounds; P7.1 fertility scan | Marathi traces ≈ 2 |
| 3 | P5.3 Marathi analysis; P6 English | P7 second pair | English ≈ 2, second pair ≈ 3 |
| 4 | P11 write-up; P9 if time | P10 real loop; P8 if time | real loop ≈ 3 |

---

## 6. Decisions to ask the user before starting the task concerned

1. P5: the source of Marathi morpheme segmentations, and what to do if Llama-3's Marathi output is
   poor.
2. P6: the English-control direction (recommended: hi→en).
3. P7: which second model pair, after the fertility scan.
4. P8 / P10: whether to spend GPU time on exact sampling in the real loop.
5. P4: whether to build a real tree loop if the estimate says "go" (probably beyond this course's
   timeline).
