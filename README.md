# A Refund for the Token Tax: speculative decoding for Hindi

CS769 course project, IIT Bombay.

**One-line summary.** A Hindi word costs Llama-3 about 2.67 tokens, and the draft model pays a full
forward pass for every one of them. We first tried to make drafting *stop* at linguistic boundaries,
which did not help. We then made drafting *cheaper*: a Hindi 4-gram proposes the rest of each word and
the draft model checks those proposals inside the call it makes anyway. This gives **1.620× wall-clock
speed-up on real GPUs, against 1.384× for the standard entropy rule**. The output is identical to the
8B model's, and nothing is trained.

Setup throughout: **Llama-3-8B-Instruct target, Llama-3.2-1B-Instruct draft**, English→Hindi
translation of FLORES+ (tuned on `dev`, 997 sentences; reported on `devtest`, 1,012 sentences), greedy
decoding (lossless), Kaggle 2× T4 GPUs.

| | |
|---|---|
| Paper | [`paper/main.pdf`](paper/main.pdf) (LaTeX source in [`paper/`](paper/)) |
| Final presentation | [`docs/presentation/CS769_final_presentation.pdf`](docs/presentation/CS769_final_presentation.pdf) (and `.pptx`) |
| Pre-interim work (proposal, first experiments, interim results) | [`docs/pre_interim/README.md`](docs/pre_interim/README.md) |
| Phase 1 report (fair re-run of all stopping rules) | [`results/phase1/REPORT.md`](results/phase1/REPORT.md) |
| Final GPU results | [`results/heads_loop/SUMMARY.md`](results/heads_loop/SUMMARY.md) |

---

## Contents

1. [Headline results](#1-headline-results)
2. [Project timeline](#2-project-timeline)
3. [Background and method](#3-background-and-method)
4. [Results by stage](#4-results-by-stage)
5. [Repository layout](#5-repository-layout)
6. [Data and large files](#6-data-and-large-files)
7. [How to reproduce](#7-how-to-reproduce)
8. [Limitations](#8-limitations)
9. [References](#9-references)

---

## 1. Headline results

Wall-clock speed-up over the 8B decoding alone. Real GPU loop: 2× T4, 100 devtest sentences × 3
repeats, batch size 1. Simulated: exact trace simulation on all 1,012 devtest sentences. 95% bootstrap
confidence intervals over sentences.

| system | simulated [95% CI] | real GPU [95% CI] |
|---|---|---|
| Entropy rule (stop drafting when the 1B is unsure; baseline) | 1.452 [1.444, 1.461] | 1.384 [1.359, 1.408] |
| Word-completion heads (V1) + word list, unchecked | 1.542 [1.530, 1.555] | 1.469 [1.433, 1.504] |
| Word-completion heads (V1), checked by the 1B | 1.572 [1.560, 1.586] | 1.505 [1.467, 1.539] |
| **Hindi 4-gram, checked by the 1B** | **1.701 [1.687, 1.716]** | **1.620 [1.580, 1.657]** |
| 4-gram + V0 heads, checked | 1.708 [1.694, 1.724] | 1.620 [1.581, 1.657] |
| Bound: the 1B's own word completion, checked | 1.857 [1.841, 1.875] | — |
| Ceiling: continuation tokens at 5% of a 1B call | 1.911 [1.894, 1.931] | — |

- The checked 4-gram beats the best unchecked heads by **+0.151 [+0.129, +0.174]** on real GPUs.
- The simulator agrees with the real loop within 1% on the same sentences, for every method.
- 96% of outputs are token-identical to plain 8B decoding, the same rate as plain decoding against
  itself. The differences are fp16 near-ties, not speculative decoding.
- 77% of the 1B calls the 4-gram saves are on word-continuation tokens, which are 59% of all tokens:
  the gain sits on the token tax.

## 2. Project timeline

| stage | question | answer | where |
|---|---|---|---|
| **Before the interim** | | | |
| Proposal | Should drafts stop at word, morpheme or akshara boundaries instead of after a fixed k tokens? | — | [`docs/pre_interim/`](docs/pre_interim/README.md) |
| Tokenizer analysis (Test 1) | Do fixed windows cut through Hindi words? | Yes: 2.67 tokens/word, 54.3% of token cuts inside an akshara, 60.6% of 4-token drafts end mid-word | `notebooks/pre_interim/test1_hindi_boundaries.ipynb` |
| Draft traces, 3B then 8B target | How cheap must the draft be? | With a 3B target a 1B call costs 0.61 of a target step and even the oracle gets 1.40×; with the 8B it costs 0.33 | `notebooks/pre_interim/day2-day3-llama8b-kaggle.ipynb` |
| Simulator and stopping rules | Which stopping rule is fastest? | Interim: entropy rule 1.452×, linguistic rules below it | `notebooks/pre_interim/lama8b-kaggle-test.ipynb` |
| **After the interim** | | | |
| Audit and rebuild (Phase 1) | Were the interim numbers right? | Three linguistic rows were broken (e.g. the "morpheme" rule never used MorphTok). Pipeline rebuilt, all outputs regenerated | [`results/phase1/REPORT.md`](results/phase1/REPORT.md) §3 |
| Fair re-run + real GPU loop | Does any linguistic stopping rule beat entropy? | No. Best: 1.427× vs 1.452×. Entropy already explains where drafts fail (ΔAUC −0.0001). Real GPUs: entropy 1.356×, same ranking | `results/phase1/` |
| Universal word-level head | Can one head draft whole words for any LLM? | No-go: 1.296× against a 1.6× target | `results/week1/` |
| Token-tax analysis | Where is the remaining speed-up? | In the cost of drafting: the 1B is 81–86% right inside words but pays a full call per token. Ceiling 1.911× | `results/wordheads/feasibility.json` |
| Word-completion heads | Can cheap heads finish the word? | Yes, a little: 1.542× simulated, 1.469× real GPU. Unchecked guesses are often wrong | `results/wordheads/`, `results/wordheads_long/` |
| Checked drafting | What if the 1B checks cheap guesses? | **1.701× simulated, 1.620× on real GPUs**, with a plain 4-gram and no training | `results/checked/`, `results/heads_loop/` |

## 3. Background and method

**Speculative decoding.** The 1B *draft* model guesses a few tokens ahead, one forward pass per token.
The 8B *target* checks all of them in one forward pass and keeps the longest agreeing prefix, plus its own
next token. With greedy decoding the output is exactly what the 8B would write alone.

**Cost model.** Times are in units of one 8B decoding step. One 1B call costs **c = 0.323**; one 8B check
of g ≤ 8 drafted tokens costs **v(g) = 1.020–1.074**; a 1B call that reads up to 8 proposed tokens costs
about the same as one that reads one token. Speed-up over a set of rounds is

```
S = Σ (accepted + 1) / Σ (C_draft + v(g))
```

**Traces and the exact simulator.** On GPU we record, once, the 8B's greedy outputs and the 1B's
teacher-forced predictions along them (top-20 distributions, entropy). A CPU simulator then replays any
drafting method over these traces. Accepted tokens are exact; the only approximation is where a policy
stops after a wrong guess, which the real loop shows is small. A causality guard raises an error if a
policy reads anything a real decoder would not know at that point.

**The token tax.** Llama-3 splits Hindi into 2.67 tokens per word (English: 1.24), and 59% of the tokens
the 8B writes only continue a word that has already started. The 1B agrees with the 8B on 66% of
word-initial tokens but 81–86% of continuation tokens.

**Methods compared.**

| method | how it drafts |
|---|---|
| Fixed k | always k tokens |
| Entropy rule (AdaEDL / SVIP style) | stop when 1 − √(β·H) < λ, H = the 1B's entropy |
| Linguistic stopping rules | entropy or fixed rules plus word / akshara / morpheme conditions (MorphTok lexicon + CRF segmenter, prefix model) |
| Universal word-level head | a 14.6M-parameter head that reads an LLM's next-token distribution and drafts whole word units, trained on four small LLMs and used zero-shot on the 8B |
| Word-completion heads | three Medusa-1 heads `z_k = h + SiLU(W_k h)` on the frozen 1B, over a 1,406-token continuation vocabulary (0.018 of a 1B call). V0: plain multi-token loss; V1: loss only inside words + "does the word go on?" head; V2: V1 with uncertainty weighting. Optional word-list filter |
| **Checked drafting** | a cheap proposer (Hindi 4-gram from 190k IITB sentences, word trie, or the heads) proposes up to 7 tokens; the 1B reads them in its next call, keeps the prefix it agrees with plus its own next token; the 8B verifies as usual. This is the vertical cascade of Cascade Speculative Drafting (Chen et al., 2024), applied to the token tax |

## 4. Results by stage

### 4.1 Pre-interim (as presented at the interim)

Full write-up, including what was later corrected: [`docs/pre_interim/README.md`](docs/pre_interim/README.md).

| policy (8B target, interim protocol) | speed-up |
|---|---|
| fixed k = 2 | 1.361 |
| **entropy rule** | **1.452** |
| word boundary* | 1.335 |
| morpheme lookup* | 1.066 |
| hybrid akshara* | 1.287 |
| oracle (cap 8) | 1.905 |

\* rows later found to be invalid (see 4.2).

### 4.2 Phase 1: audit, fair re-run, real GPU loop — [`results/phase1/`](results/phase1/)

Every policy tuned on dev with comparable grids, look-ahead charged as one 1B call, reported once on
devtest. Report: [`results/phase1/REPORT.md`](results/phase1/REPORT.md).

| policy | simulated [95% CI] |
|---|---|
| Oracle (upper bound) | 1.904 [1.895, 1.914] |
| **Entropy** | **1.452 [1.444, 1.461]** |
| Entropy + morpheme rule (morph model consulted) | 1.427 [1.418, 1.437] |
| Fixed k = 2 | 1.357 [1.350, 1.366] |
| True word boundaries (oracle) | 1.353 [1.345, 1.361] |
| Morpheme rule alone | 1.343 [1.335, 1.353] |
| Entropy + word check (Hybrid) | 1.253 [1.243, 1.264] |
| Akshara-safe entropy | 1.239 [1.231, 1.249] |
| Stop at word end (soft / hard) | 1.161 / 1.144 |

- At equal entropy, word-initial tokens are *not* harder (+1.4 points [+0.8, +2.0]); adding word or
  morpheme position to a confidence-based failure predictor changes AUC by −0.0001.
- Real GPU loop (100 sentences): entropy 1.356×, entropy + morpheme 1.323×, fixed k = 2 1.304×; same
  ranking as the simulator, which was 3–8% optimistic in that loop (mainly because the loop drafted past
  the end of the text; fixed in the final loop).

### 4.3 Universal word-level head — [`results/week1/`](results/week1/)

Verdict **NO GO** ([`SUMMARY.md`](results/week1/SUMMARY.md)): zero-shot 1.296× [1.287, 1.306],
fine-tuned on the 8B 1.311×, text-only ablation 1.287×, without Llama-family data 1.211×, all below the
entropy rule (1.452×) and the 1.6× go criterion. Word-unit accuracy: 66% for the next unit, 14% for the
one after. Guessing the next word is hard; finishing the current one is easy.

### 4.4 Token-tax analysis and word-completion heads — [`results/wordheads/`](results/wordheads/), [`results/wordheads_long/`](results/wordheads_long/)

| drafter (unchecked) | 60k training pairs | 180k pairs [95% CI] |
|---|---|---|
| Entropy (baseline) | 1.452 | 1.452 |
| Dictionary on every mid-word position | 1.226 | |
| Dictionary, confidence-gated | 1.471 | |
| Heads V0 | 1.497 | 1.510 [1.500, 1.522] |
| Heads V1 (word-guided) | 1.514 | 1.518 [1.508, 1.530] |
| Heads V2 (adaptive weights) | 1.505 | 1.517 [1.507, 1.529] |
| **Heads V1 + word-list filter** | 1.533 | **1.542 [1.530, 1.555]** |
| Ceiling: continuations at 5% of a 1B call | 1.911 | |

- Whole-word completion accuracy: dictionary 31%, + bigram + cache 35%, + reference words 47%, the 1B 77%.
  Context has to come from a model.
- Head accuracy on continuation tokens (180k): V1 57 / 50 / 44%, V0 56 / 43 / 33%, the 1B 83–86%.
- 3× more training data adds only +0.010 (V1 + word list). V1 − V0 = +0.008 [+0.004, +0.012].
- Figures: `results/wordheads/fig_*.png`; analysis notebook `notebooks/08_wordheads_analysis.ipynb`.

### 4.5 Checked drafting — [`results/checked/`](results/checked/)

| checked proposer | simulated [95% CI] |
|---|---|
| V0 heads | 1.568 [1.556, 1.581] |
| V1 heads | 1.572 [1.560, 1.586] |
| Word trie + previous and next word | 1.659 [1.646, 1.673] |
| **Hindi 4-gram** | **1.701 [1.687, 1.716]** |
| 4-gram + V0 heads | 1.708 [1.694, 1.724] |

- 4-gram training data: 1k sentences 1.615, 10k 1.668, 60k 1.689, 190k 1.701; only the 8B's 997 dev
  outputs: 1.660.
- Robust to overhead: if a 1B call that reads proposals were 5 / 10 / 20% dearer, 1.677 / 1.654 / 1.610.
- The gain grows with the share of continuation tokens in a sentence (Spearman 0.21).
- Checked drafting recovers 54% of the gap between entropy and the ceiling.

### 4.6 Final real GPU loop — [`results/heads_loop/`](results/heads_loop/)

See section 1 and [`SUMMARY.md`](results/heads_loop/SUMMARY.md). Per-decode logs:
`loop_results.jsonl`; paired differences: `paired_wall.json`; simulator vs real: `sim_vs_real.csv`.

## 5. Repository layout

```
README.md                      this file
pyproject.toml                 package `lgmtp` (pip install -e .)
paper/                         paper: main.tex, sec_checked.tex, sec_real.tex, refs.bib, figures/, main.pdf
docs/
  presentation/                final slides (.pptx with embedded fonts, .pdf)
  pre_interim/                 proposal, plan, first experiments, interim results, interim slide corrections
src/lgmtp/                     all logic (imported by scripts, notebooks and the generated Kaggle notebooks)
  boundaries.py                word / akshara / UTF-8 boundaries on byte spans (tokenizer analysis)
  vocab.py, traces.py          token classes; trace loading, gold labels on the target's text
  gpu_traces.py                GPU: 8B greedy outputs, teacher-forced draft/target traces, timing
  simulator.py                 exact trace simulator, cost model, causality guard (TraceView)
  policies.py                  stopping policies (fixed k, entropy, word / akshara / morpheme, hybrids, oracles)
  morph.py                     MorphTok loader, gold lexicon, CRF segmenter, causal prefix model
  tuning.py, stats.py          dev grids, devtest evaluation, sentence bootstrap and paired CIs
  analysis.py                  where drafts fail; entropy-controlled comparisons; failure prediction (AUC)
  spec_loop.py                 Phase 1 real draft-and-verify loop with KV caches
  week1.py, unihead.py, unichar.py, uni_features.py, uni_sim.py, collect.py
                               universal word-level head: data collection over 4 LLMs, model, simulator, verdict
  drafter_tf.py                1B teacher-forced along the 8B outputs, multi-token timing
  wordheads.py                 word-completion heads: labels, losses, IITB data, training, prediction dump
  refund_sim.py                exact simulator for refund drafters (ceiling, dictionary, heads, word list)
  checked.py                   checked drafting: token n-gram table and exact simulator
  heads_loop.py                final real GPU loop (NoDraft, Entropy, heads, word list, checked methods)
scripts/
  download_morphtok.py         put MorphTok hindi.csv into data/morphtok/ and record its provenance
  build_morph.py               gold morpheme lexicon + CRF segmenter (results/phase1/morph_*)
  make_report.py               writes results/phase1/REPORT.md from the result files
  build_kaggle_notebooks.py    generates the self-contained notebooks in kaggle/ from src/lgmtp
  smoke_cpu.py                 CPU smoke tests of the Kaggle code paths
  feasibility/                 refund_bound.py (ceiling, dictionary), gated.py, noisy.py (rho curve),
                               diag.py / ctx.py (whole-word accuracy), checked_wordtrie.py
  run_refund_sim.py            word-completion heads main table (results/wordheads*, --heads-dir/--out)
  build_analysis_notebook.py   builds notebooks/08_wordheads_analysis.ipynb
  run_checked_sim.py           builds the 4-gram table, tunes checked drafting (results/checked/)
  checked_analysis.py          saved calls by token type, gain by continuation share, data/cost curves
kaggle/                        GPU notebooks, as run (generated; outputs kept)
  01_traces_v2.ipynb           8B outputs + traces + timing (dev, devtest)
  03_real_loop.ipynb           Phase 1 real loop
  05_week1_universal_head.ipynb   universal word-level head, kill test
  06_drafter_traces.ipynb      1B traces + multi-token timing for the refund simulation
  07_wordheads.ipynb           train heads V0/V1/V2 (60k pairs), dump predictions, timing
  07b_wordheads_long.ipynb     same with 180k pairs
  09_heads_loop.ipynb          final real GPU loop
notebooks/                     CPU analysis notebooks, as run
  00_tokenizer_analysis.ipynb  tokenizer analysis (re-run of Test 1)
  02_policies_v2.ipynb         labels, tuning on dev, evaluation on devtest, failure analysis
  04_real_loop_analysis.ipynb  Phase 1 real loop: losslessness, simulator vs real
  08_wordheads_analysis.ipynb  word-completion heads: figures and tables
  pre_interim/                 the three original notebooks behind the interim results, unchanged
results/
  phase1/                      REPORT.md, main_table.csv, grid_*.csv, analysis_*, morph_*, real_loop_*, plots
  week1/                       universal head: SUMMARY.md, verdict.json, l8*.csv, lomo*.csv, agreement.csv
  wordheads/                   feasibility.json, main_table.csv, head_accuracy.*, rho_curve.csv, token_tax.csv, fig_*.png
  wordheads_long/              180k-pair heads: main_table.csv, paired_vs_60k.json, head_accuracy.json
  checked/                     main_table.csv, wordtrie.csv, gain_by_share.csv, where_saved.json, ngram_info.json
  heads_loop/                  final GPU loop: SUMMARY.md, summary.json, paired_wall.json, sim_vs_real.csv, logs
tests/                         unit tests (pytest): boundaries, causality of every policy, simulators,
                               losslessness and round-by-round equality of the real loops, statistics
```

Each Kaggle notebook embeds the exact `src/lgmtp` code it ran (its first cell prints the git hash). To
change a notebook, edit `src/lgmtp/` and rebuild with `scripts/build_kaggle_notebooks.py`.

## 6. Data and large files

`data/` is git-ignored. Everything the CPU scripts need is either in `results/` (tracked) or in the
shared Google Drive folder:

**Google Drive: [data folder](https://drive.google.com/drive/folders/13jKZOti66G2qEAWVm30rxmwf8d_kgEif?usp=sharing)** — download and unzip into `data/` so the paths below exist.

| path | size | what | produced by |
|---|---|---|---|
| `data/traces_v2/` | 32 MB | 8B outputs and 1B/8B teacher-forced traces for dev and devtest, timing | `kaggle/01_traces_v2.ipynb` |
| `data/real_loop/` | 10 MB | Phase 1 real-loop logs | `kaggle/03_real_loop.ipynb` |
| `data/wordheads/` | 107 MB | heads V0/V1/V2 (60k pairs) and their predictions along the 8B outputs | `kaggle/07_wordheads.ipynb` |
| `data/wordheads_long/` | 107 MB | same with 180k pairs | `kaggle/07b_wordheads_long.ipynb` |
| `data/ngram_hi.npz` | 6 MB | the Hindi 4-gram table | `scripts/run_checked_sim.py` |
| `data/week1/` (`sources.json`, per-model traces) | | universal-head data collection | `kaggle/05_week1_universal_head.ipynb` |

External inputs:
- **MorphTok** `hindi.csv` (Hindi morpheme segmentations, `karthika95/morphTok` on Hugging Face):
  `python scripts/download_morphtok.py --hf`. Several scripts also read it as `../hindi.csv`, one folder
  above the repository; copy it there.
- **FLORES+** (`openlanguagedata/flores_plus`, gated) and the **Llama** models (gated) need a Hugging Face
  token with access. The IITB English–Hindi corpus is downloaded automatically.

## 7. How to reproduce

### 7.1 Setup and tests

```bash
python -m venv .venv
.venv/Scripts/pip install -e . pytest        # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m pytest -q             # 151 passed, 3 skipped (the skipped ones need gated tokenizers)
```

Python 3.11. CPU-only PyTorch is enough for everything outside Kaggle.

### 7.2 GPU runs (Kaggle)

Upload the notebook, set the accelerator, turn Internet on, add the secret `HF_TOKEN`, then
*Save & Run All*. Each notebook's first cell lists its inputs and outputs.

| order | notebook | GPU, approx. time | output → put into |
|---|---|---|---|
| 1 | `kaggle/01_traces_v2.ipynb` (run with `SPLIT="dev"`, then `"devtest"`) | 2× T4 | `data/traces_v2/` |
| 2 | `kaggle/03_real_loop.ipynb` | 2× T4 | `data/real_loop/` |
| 3 | `kaggle/05_week1_universal_head.ipynb` | 2× T4 (resumable) | `results/week1/`, `data/week1/` |
| 4 | `kaggle/06_drafter_traces.ipynb` (input: output of 3) | 1× T4, ~20 min | refund simulation inputs |
| 5 | `kaggle/07_wordheads.ipynb` / `07b_wordheads_long.ipynb` | 1× T4, ~1 h / ~2.5 h | `data/wordheads/`, `data/wordheads_long/` |
| 6 | `kaggle/09_heads_loop.ipynb` (dataset with the 180k heads, `hindi.csv`, `ngram_hi.npz`) | 2× T4, ~2 h | `results/heads_loop/` |

### 7.3 CPU analysis (from the repository root)

```bash
# Tokenizer analysis
jupyter nbconvert --execute --to notebook --inplace notebooks/00_tokenizer_analysis.ipynb

# Phase 1: morphology, tuning, evaluation, real-loop analysis, report
python scripts/download_morphtok.py
python scripts/build_morph.py
jupyter nbconvert --execute --to notebook --inplace notebooks/02_policies_v2.ipynb
jupyter nbconvert --execute --to notebook --inplace notebooks/04_real_loop_analysis.ipynb
python scripts/make_report.py

# Token-tax analysis and word-completion heads
python scripts/feasibility/refund_bound.py
python scripts/feasibility/gated.py
python scripts/feasibility/noisy.py
python scripts/feasibility/diag.py
python scripts/feasibility/ctx.py
python scripts/run_refund_sim.py
python scripts/run_refund_sim.py --heads-dir data/wordheads_long --out results/wordheads_long
python scripts/build_analysis_notebook.py      # then execute notebooks/08_wordheads_analysis.ipynb

# Checked drafting (sources.json = the IITB Hindi sentences collected by notebook 05;
# without --sources they are downloaded)
python scripts/run_checked_sim.py --sources data/week1/data/sources.json
python scripts/checked_analysis.py
python scripts/feasibility/checked_wordtrie.py
```

Every script writes into `results/`, and every number in the paper and the slides comes from a file
there.

## 8. Limitations

- One language pair (English→Hindi), one model pair (Llama-3 8B / 3.2 1B), one domain (FLORES+).
- Real-GPU numbers are on 100 sentences at batch size 1, on 2× T4 (the 8B is split across two GPUs, so
  every target step includes a transfer). Absolute speed-ups would differ on one large GPU.
- No English control yet, so we cannot say how much larger the gain is for Hindi than for a low-tax
  language.
- The checking mechanism itself is known (cascaded drafting); the contribution is the token-tax analysis
  and showing where the gain comes from.
- 31% of morpheme labels come from the CRF segmenter, whose accuracy on unseen stems is probably below
  its held-out F1 (0.927).

## 9. References

Main related work (full list in [`paper/refs.bib`](paper/refs.bib)):

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023.
- Chen et al., *Accelerating LLM Decoding with Speculative Sampling*, 2023.
- Cai et al., *Medusa*, ICML 2024; Ankner et al., *Hydra*, 2024.
- Chen et al., *Cascade Speculative Drafting for Even Faster LLM Inference*, NeurIPS 2024;
  Spector and Ré, *Staged Speculative Decoding*, 2023.
- He et al., *REST*, NAACL 2024; Oliaro et al., *SuffixDecoding*, 2024.
- Yang, *No Refund for the Token Tax: Speculative Decoding Across 21 Languages*, 2026.
- Paudel et al., *Speculative Decoding and the Curse of Multilinguality*, 2026.
- Syvokon, *Dictionary-Based Speculative Decoding for Non-Latin-Script Languages*, UNLP 2026.
- Naidu, *Phrase-Level Speculative Decoding on a Frozen Target*, 2026.
- Kunchukuttan et al., *The IIT Bombay English-Hindi Parallel Corpus*, LREC 2018; NLLB Team, FLORES, 2022.
