# PLAN: Bug fixes + Phase 1 rebuild — Linguistically Guided Multi-Token Prediction (CS769)

This document is written for Claude Code. It contains the full project context, a precise list of bugs found in the existing notebooks, the target repository layout, and step-by-step tasks with acceptance criteria. Read the whole file before changing anything.

---

## 0. Working rules for Claude Code

1. **No GPU locally; the user runs GPU notebooks on Kaggle by hand.** Everything that needs a GPU (8B target generation, teacher-forced draft/target passes, timing, the real speculative loop) is delivered as a **single self-contained `.ipynb`** that the user uploads manually to Kaggle (Accelerator: GPU T4 ×2, Internet: on, Secret `HF_TOKEN`), runs with *Save Version → Save & Run All (Commit)*, and whose output zip the user downloads and unzips into the repo. Therefore every GPU notebook must:
   - (a) contain all the `lgmtp` code it needs **inlined**. Generate these notebooks with `scripts/build_kaggle_notebooks.py`, which writes the current `src/lgmtp/*.py` files into the notebook's first cell (each module written to `/kaggle/working/lgmtp/` and then imported). `src/lgmtp/` stays the single source of truth; never hand-edit the generated notebooks. The user should not have to upload any Kaggle Dataset unless an embedded artifact is too large (see T9).
   - (b) write all outputs under `/kaggle/working/<name>/` and zip them at the end (one zip per split / run);
   - (c) be resumable if the session dies (append-and-resume, like the legacy generation cell), and allow running one split per session via a config flag (Kaggle sessions are limited to ~12 h and the weekly GPU quota is limited);
   - (d) end by printing exactly which zip to download and which local folder to unzip it into.
   CPU parts (boundary labelling, MorphTok download and segmenter, simulator, tuning, statistics, tests, analysis) run locally.
2. **No old data is available.** The user no longer has MorphTok `hindi.csv` or any previous Kaggle outputs (traces, timing, target outputs). The only record of old results is the numbers in §1.3 (from the user's PDF report). Everything is regenerated from scratch in this phase; MorphTok is re-downloaded from Hugging Face (T3).
2a. **Smoke-test GPU code on CPU before handing it over.** Kaggle GPU hours are scarce, so `scripts/smoke_cpu.py` must run the exact code paths of notebooks 01 and 03 locally on CPU with `SMOKE=True`: `meta-llama/Llama-3.2-1B-Instruct` used as *both* target and draft (fp32, loaded once), 3 sentences per split, `MAX_NEW=24`. It must finish in a few minutes and exercise generation, both teacher-forced passes, feature extraction, saving/loading, zipping, and the real loop with every policy (with draft = target, every draft token is accepted, which is itself a useful test). Only build the Kaggle notebook after the smoke test passes.
3. **Put logic in `.py` modules, keep notebooks thin.** Notebooks should only import from `src/lgmtp/`, call functions, and plot. Editing notebook JSON is error-prone; avoid large logic inside cells.
4. **Do not delete or edit the old notebooks.** Move them to `notebooks/legacy/` with `git mv` and leave their contents unchanged. They are the record of what was run.
5. **Do not invent data formats.** Where a file format is unknown (notably MorphTok `hindi.csv`), inspect the file first and print its columns and a few rows. If the format is ambiguous, stop and ask the user.
6. **Causality is the most important correctness property** of this project (see §3). Every stopping policy must only read information that a real decoder would have at that moment. This is enforced in code (Task T4, causality guard) and in tests.
7. Every numeric claim must come from a script/notebook output saved under `results/`. No hand-copied numbers.
8. Write unit tests (pytest) for every module in `src/lgmtp/` that does not need a GPU. Tests that need the gated Llama tokenizer must `pytest.skip` cleanly if `HF_TOKEN` is not set.
9. When unsure about a design decision not covered here, ask the user rather than guessing.

---

## 1. Project context

### 1.1 What the project is

Team: Kanika and Rakshit (IIT Bombay, CS769 course project). Goal: an AA-grade project that can become a workshop/ICON-level paper.

**Problem.** Speculative decoding (and multi-token prediction in general) uses a small *draft* model to propose several tokens, which a large *target* model then verifies in one forward pass. Verification is lossless: the output is identical to greedy target decoding. Speed-up depends on (a) how many drafted tokens the target accepts and (b) how much drafting costs. Standard methods draft a **fixed** number of tokens `k`, regardless of whether the draft ends in the middle of a word, a morpheme, or an *akshara* (Devanagari grapheme cluster: consonant + halant + consonant + matra, e.g. क्ष, ज्ञ, द्या).

For Hindi (and other Indic languages), byte-level BPE tokenizers cut through words and aksharas constantly. Our measured Test 1 numbers for the Llama-3 tokenizer on FLORES+ Hindi devtest: 2.67 tokens/word; only 38.6% of token cuts are word boundaries; 54.3% of cuts are *inside an akshara*; a k=4 draft ends inside a word ~61% of the time.

**Hypothesis.** An *adaptive stopping controller* that decides when to stop drafting using (1) draft-model confidence (entropy) and (2) linguistic structure (word / morpheme / akshara boundaries) gives better speed-up than fixed-k, and linguistic signals add information beyond confidence.

**Reframed scientific question (what we actually want to answer).** *Does linguistic structure predict where drafts fail, beyond what model confidence already captures?* Because verification is lossless and token-level, ending a draft mid-word cannot hurt acceptance by itself; boundaries can only help if they carry information about where draft and target disagree. Either answer (yes / no) is publishable if the experiment is rigorous, so **correctness and fairness of comparison matter more than getting a high number.**

### 1.2 Setup

- Target model: `meta-llama/Meta-Llama-3-8B-Instruct` (fp16, layer-split across 2× T4 with `device_map="auto"`, `max_memory={0:"8GiB",1:"10GiB"}`).
- Draft model: `meta-llama/Llama-3.2-1B-Instruct` (fp16, on `cuda:0`). Draft and target share the Llama-3 tokenizer (asserted in code; keep the assertion).
- Task: English → Hindi translation on FLORES+ (`openlanguagedata/flores_plus`, configs `eng_Latn`, `hin_Deva`). Prompt: chat template, user message `"Translate the following English sentence into Hindi. Output only the Hindi translation.\n\n{s}"`. Greedy decoding, `MAX_NEW=256`, stop on eos ids + `<|eot_id|>`.
- Cost model: one round with `g` drafted tokens costs `g*c + verify(g)` target-step units, where `c = t_draft / t_target` (measured ≈ 0.334 on 2× T4) and `verify(g)` is measured verification cost for g+1 tokens relative to one decode step (≈1). It emits `accepted + 1` tokens (correction or bonus token from target). Speed-up = tokens emitted / total cost. Plain decoding = 1.0.
- Policies compared: fixed-k, entropy-only (AdaEDL/SVIP-style rule: stop when `1 - sqrt(beta*H) < lam`), word-boundary, morpheme-boundary, hybrid (entropy + boundary), and an oracle (stops exactly before the first mismatch; upper bound).

### 1.3 Current (pre-fix) results, 8B target, devtest random half as test

| policy | speed-up | acceptance | mean draft len |
|---|---|---|---|
| fixed k=2 (best fixed) | 1.361 | 0.651 | 1.98 |
| entropy λ=-0.37 β=2.0 | 1.452 | 0.684 | 2.44 |
| word boundary | 1.335 | 0.623 | 2.09 |
| morpheme lookup | 1.066 | 0.374 | 6.57 |
| hybrid akshara | 1.287 | 0.690 | 1.21 |
| oracle (cap 8) | 1.905 | 1.000 | 2.63 |

**The three linguistic rows are invalid** because of the bugs in §2. The fixed-k, entropy, and oracle rows are believed correct (simulator logic is sound for greedy decoding), but will be recomputed on the new splits.

### 1.4 Existing files (all go to `notebooks/legacy/`, unchanged)

| file | what it contains | status |
|---|---|---|
| `test1_hindi_boundaries.ipynb` | Tokenizer analysis (no GPU). Cell 2 has **correct** byte-level token→bytes mapping (`token_to_bytes`, `HFTok`), a **correct** akshara regex (`akshara_units`, handles nukta, ZWJ/ZWNJ, independent vowels, modifiers), `is_word_boundary`, `sentence_features`, `analyze`, `summarize`. Cell 3 has akshara sanity asserts. | Logic is good. **Extract into `src/lgmtp/boundaries.py`** (Task T1). |
| `day2-day3-llama8b-kaggle.ipynb` | Part A (8B target outputs, 1B draft traces, timing) + Part B (simulator, fixed-k, entropy sweep, oracle). Header says "MorphTok is skipped". | Logic OK. Simulator moves to `src/lgmtp/simulator.py`. |
| `lama8b-kaggle-test.ipynb` | Same as above plus **Day 4 cells 39–47**: `add_boundary_flags`, `WordBoundaryPolicy`, `LookupPolicy`, `HybridPolicy`, correlation, final summary. | **Cells 39–47 contain the bugs in §2.** Do not reuse them. |

(Uploaded filenames may carry a `__1_` suffix; use whatever names exist in the repo.)

An earlier 3B-target run (`Llama-3.2-3B-Instruct`, c≈0.61) reported MorphTok coverage of 41% of word types and 71% of word tokens on model outputs. That code is not available; coverage is recomputed in T3.

### 1.5 Workflow and handoffs between Claude Code (local) and the user (Kaggle)

| step | where | who | does what | produces → goes where |
|---|---|---|---|---|
| S1 | local | Claude Code | T0, T1, T2 code (`vocab.py`, `traces.py`), T3 steps 1–4 (download MorphTok, gold lexicon, CRF segmenter + held-out eval), T4, T5, T6 code, all unit tests | `src/lgmtp/`, `tests/`, `results/phase1/morph_crf_eval.json` |
| S2 | local | Claude Code | `scripts/smoke_cpu.py` for notebook 01 code path; then `scripts/build_kaggle_notebooks.py` → `kaggle/01_traces_v2.ipynb` | notebook for the user |
| S3 | Kaggle | **user** | upload `kaggle/01_traces_v2.ipynb`, run with `SPLIT="dev"`, then again with `SPLIT="devtest"` (same session if time allows) | `traces_v2_dev.zip`, `traces_v2_devtest.zip` (each includes `timing.json`) → unzip into `data/traces_v2/` |
| S4 | local | Claude Code | T3 steps 5–6 (coverage on outputs, prefix model from dev), T7 tuning + evaluation, T8 analysis, legacy-comparison check (T4) | `results/phase1/*`, `results/phase1/selected_params.json`, `results/phase1/prefix_model.pkl.gz` |
| S5 | local | Claude Code | smoke test notebook 03 code path; build `kaggle/03_real_loop.ipynb` with `selected_params.json` and the prefix model embedded (base64 + gzip) | notebook for the user |
| S6 | Kaggle | **user** | upload and run `kaggle/03_real_loop.ipynb` | `real_loop.zip` → unzip into `data/real_loop/` |
| S7 | local | Claude Code | T9 analysis part (losslessness summary, simulator-vs-real comparison), T10, `results/phase1/README.md` | final results |

At each **user** step, stop and tell the user exactly what to upload, which config flags to set, roughly how long it should take, and what to download. Do not proceed to the next local step until the user confirms the files are in place.

---

## 2. Bugs found (all in `lama8b-kaggle-test.ipynb`, cells 39–47)

### B1 — "Morpheme lookup" never used MorphTok (critical)
`MORPH_DICT = {"है","और","में","की","से","को","का","कि","यह","एक","तो"}` is an 11-word placeholder. `LookupPolicy` only stops when the word-so-far exactly equals one of these function words, so it almost never stops and runs to the cap → mean draft 6.57, speed-up 1.07×. The presentation script attributes this to "MorphTok covers only 41% of word types", which describes something that did not run.
**Fix:** real tiered morpheme labeller + causal morpheme policy (Tasks T3, T6).

### B2 — Hybrid policy has no entropy gate and an impossible akshara check (critical)
`HybridPolicy` only reads `is_hybrid_end`; there is no entropy term despite the script saying so. `is_hybrid_end[i]` is True when the word-so-far fully matches the akshara regex. But whether the last akshara is complete depends on the **next** token (e.g. "क" looks complete until "्ष" arrives; 54% of Llama tokens start mid-akshara). So it is True almost everywhere → stops after ~1 token (mean draft 1.21). Also, this notebook's regex lacks nukta (U+093C) and other cases that the Test 1 regex handles.
**Fix:** akshara completeness via one-token lookahead feature `p_mark` (Task T2/T5), entropy gate restored (Task T5), delete the notebook regex and use `boundaries.akshara_units`.

### B3 — Word-boundary policy reads an undrafted token for free (critical)
`is_word_end[i]` is computed from `draft_argmax[i+1]`. When the policy has drafted up to position `p+g-1` and checks `is_word_end[p+g-1]`, it is reading token `p+g`, which a real decoder has not computed yet. Obtaining it costs one more draft forward (≈0.33 of a target step) that the simulator never charges. The entropy policy is causal (uses `H[p+g-1]`, the distribution that drafted the last token), so the comparison is biased in favour of word-boundary.
**Fix:** explicit lookahead declaration + cost charging + causality guard in the simulator (Task T4).

### B4 — Per-token decoding of byte-level tokens
`tokenizer.decode([id])` on a single byte-level BPE token can yield `\ufffd` for partial UTF-8 pieces, corrupting boundary flags. **Fix:** use `boundaries.token_to_bytes` and byte offsets (Task T1).

### B5 — Boundaries computed on draft tokens instead of target text
Ground-truth linguistic labels (for analysis) must come from the **target output text**, since that is the committed prefix. **Fix:** Task T2 labels from `target_ids`.

### B6 — Wrong interpretation of the entropy–boundary correlation
`if correlation > 0.3` ignores the sign; r = −0.273 prints "Low correlation. The hybrid boundary policy intercepts unique states that pure entropy thresholding misses", which does not follow. With ~33k positions p≈0 is meaningless. **Fix:** report signed point-biserial r, and a table of mean entropy / match rate by position-in-word (Task T8). Remove the auto-generated "insight" text.

### B7 — Unequal tuning effort
Entropy got a 40×3 grid on dev; boundary policies got no hyper-parameters and no tuning. **Fix:** every policy gets a dev grid (Task T7).

### B8 — Inconsistent evaluation sets
Fixed-k table is on all traces; policy table on the random test half; entropy tuned on the other half of devtest. **Fix:** tune on FLORES+ `dev`, report everything on FLORES+ `devtest` (Task T0/T7), with bootstrap CIs.

### B9 — Batched/padded target generation vs. losslessness
Target outputs were generated in batches of 16 with left padding in fp16. Unbatched greedy can differ slightly. Not a bug for the simulator, but the real loop (T9) must compare against **unbatched** greedy target output.

---

## 3. Causality model (read carefully — this defines what a policy may see)

Index convention for a sentence with target output tokens `y[0..n-1]`:
- The **draft distribution at index t**, `q_t`, is the draft model's next-token distribution that predicts `y[t]`, computed teacher-forced on `prompt + y[:t]`. All per-position draft features at index `t` (`H[t]`, `top1[t]`, `p_space[t]`, …) are functions of `q_t`.
- A round starts at committed position `p`. After drafting `g ≥ 1` tokens, the draft has computed `q_p, …, q_{p+g-1}` (g forward passes).
- **Causal policy (lookahead = 0):** may read draft features at indices `≤ p+g-1`, the draft's own tokens at indices `≤ p+g-1`, and committed target tokens at indices `< p`. Example: entropy policy reads `H[p+g-1]`.
- **Lookahead policy (lookahead = 1):** may additionally read draft features at index `p+g` (the distribution of the *next* token). This requires one extra draft forward pass. If the policy stops at `g` (rather than hitting the cap), charge **one extra `c`** for that round. If it continues, that forward pass is the one used to draft token `p+g`, so no extra charge.
- Word-end and akshara-completeness after token `p+g-1` are inherently lookahead quantities: they depend on whether `y[p+g]` starts with a space/punctuation or with a combining mark.
- Never read `match[...]`, `target_ids[≥p]`, or any gold label inside a policy (except the explicitly named oracle policies).

**Teacher-forcing approximation (document it in results):** in the trace, `q_t` is conditioned on target tokens, whereas in a real loop it is conditioned on the draft's own tokens. They coincide up to the first mismatch in a round, so the number of accepted tokens is exact. Only the stopping decision (and therefore draft cost) after a mismatch can differ. Task T9 measures how much this matters.

**Bounds to report for lookahead policies:** speed-up with lookahead charged (`charge_lookahead=True`, conservative) and uncharged (`False`, optimistic; in a real loop the extra forward's KV entry can partially be reused when all drafts are accepted). The real loop (T9) decides which is closer.

---

## 4. Target repository layout

```
repo/
  PLAN_phase1_bugfix.md          # this file
  pyproject.toml                 # package "lgmtp", src layout; deps: numpy, pandas, scipy, matplotlib, transformers, datasets, sklearn-crfsuite, pytest
  src/lgmtp/
    __init__.py
    boundaries.py   # T1: token bytes, akshara units, word boundaries, per-token labels
    vocab.py        # T2: vocabulary masks (starts_space, starts_punct, starts_mark, starts_cont_byte)
    morph.py        # T3: MorphTok loader, CRF segmenter, tiered labeller, causal prefix-boundary model
    traces.py       # T2: trace schema, save/load (.npz), label attachment
    simulator.py    # T4: simulate(), summarize(), per-sentence outputs, causality guard, lookahead cost
    policies.py     # T5/T6: FixedK, Entropy, WordBoundary(soft/hard), AksharaSafeEntropy, Hybrid, Morph, Oracle, GoldBoundaryOracle, NoDraft
    tuning.py       # T7: grid search on dev, select by speed-up
    stats.py        # T7: bootstrap CI, paired bootstrap difference
    analysis.py     # T8: entropy/match by position-in-word, signed correlations
    gpu_traces.py   # T2: generation + teacher-forced passes + timing (used by Kaggle notebook 01 and the smoke test)
    spec_loop.py    # T9: real draft-and-verify loop (used by Kaggle notebook 03 and the smoke test)
  scripts/
    download_morphtok.py        # T3: fetch hindi.csv (and marathi.csv for later phases) from Hugging Face
    smoke_cpu.py                # rule 2a: CPU smoke test of notebook 01 and 03 code paths
    build_kaggle_notebooks.py   # rule 1(a): inline src/lgmtp into self-contained Kaggle notebooks
  kaggle/                       # GENERATED notebooks the user uploads (do not hand-edit)
    01_traces_v2.ipynb
    03_real_loop.ipynb
  tests/
    test_boundaries.py
    test_vocab.py
    test_morph.py
    test_simulator.py
    test_policies_causality.py
  notebooks/
    legacy/                              # the three old notebooks, unchanged
    00_tokenizer_analysis.ipynb          # thin rewrite of Test 1 using boundaries.py (CPU)
    02_policies_v2.ipynb                 # CPU: labels, tuning, test evaluation, plots
    04_real_loop_analysis.ipynb          # CPU: losslessness summary + simulator-vs-real comparison from data/real_loop/
  data/
    morphtok/hindi.csv                   # downloaded by scripts/download_morphtok.py (gitignored)
    traces_v2/                           # unzipped outputs of Kaggle notebook 01 (gitignored)
    real_loop/                           # unzipped outputs of Kaggle notebook 03 (gitignored)
  results/phase1/                        # all CSV/JSON/PNG outputs
  docs/slide_corrections.md              # T10
```

**Getting `lgmtp` onto Kaggle (`scripts/build_kaggle_notebooks.py`):** the generated notebook's first code cell writes each needed module from `src/lgmtp/` into `/kaggle/working/lgmtp/` (as Python string literals or `%%writefile` cells), adds `/kaggle/working` to `sys.path`, and imports it. The second cell is a clearly marked **CONFIG** cell (`SPLIT`, `N`, `BATCH`, `MAX_NEW`, `SMOKE=False`, model names). Remaining cells only call functions from `lgmtp.gpu_traces` / `lgmtp.spec_loop`. Embedded artifacts (T9) are written as gzip+base64 strings and decoded at runtime; if their total size exceeds ~5 MB, instead write them to `kaggle/artifacts/`, tell the user to upload that folder as a private Kaggle Dataset, and read from `/kaggle/input/...`. The build script must record the git commit hash in the notebook so results can be traced to code. Add `data/` and `kaggle/*.zip` to `.gitignore`.

---

## 5. Tasks (do them in this order)

### T0 — Scaffolding
- Create the layout above, `pyproject.toml`, `.gitignore`, `git mv` old notebooks to `notebooks/legacy/`.
- Local Hugging Face access: the user must `huggingface-cli login` (or set `HF_TOKEN`) with access to the gated `meta-llama/Llama-3.2-1B-Instruct` and `openlanguagedata/flores_plus`. If access fails, tell the user rather than working around it.
- Splits: **FLORES+ `dev` (997 sentences) = tuning set**, **FLORES+ `devtest` (1012) = test set**. Never tune on devtest.
- Acceptance: `pip install -e .` works; `pytest` runs (possibly with skips).

### T1 — `boundaries.py` (extract from Test 1, cell 2)
Move these verbatim (then add type hints/docstrings): `_bytes_to_unicode`, `BYTE_DECODER`, `token_to_bytes`, `HFTok`, the Devanagari character classes, `_AKSHARA_RE`, `akshara_units`, `is_word_boundary`, cut categories (`END, WORD, MIDWORD, MIDAKS, MIDUTF8`), `classify`, `sentence_features`, `analyze`, `summarize`, `render_example`.

Add:
```python
def token_labels(token_bytes: list[bytes]) -> dict[str, np.ndarray]:
    """Gold labels for each token t of a target sequence (analysis only, never read by policies).
    end_cut[t]      : category of the cut AFTER token t (END/WORD/MIDWORD/MIDAKS/MIDUTF8)
    start_cut[t]    : category of the cut BEFORE token t
    word_initial[t] : token t starts a new word (start_cut in {WORD, END})
    pos_in_word[t]  : 0 for word-initial tokens, then 1, 2, ...
    word_id[t]      : index of the word the token belongs to
    char_span[t]    : (start_char, end_char) in the decoded text
    """
```
Note: Llama target sequences end with `<|eot_id|>`; handle special tokens by stripping them before labelling and padding labels back to length `n` (label them `END`).

`notebooks/00_tokenizer_analysis.ipynb` reruns Test 1 via the module. **Acceptance (regression):** Llama-3.2-1B tokenizer on FLORES+ hin devtest reproduces tokens/word 2.67, word-boundary cuts 38.6%, mid-akshara cuts 54.3%, k=4 inside-word 60.6% (±0.1). Qwen/Qwen3.5-0.8B reproduces 3.33 tokens/word.

Tests (`test_boundaries.py`): the Test 1 cell-3 asserts (`क्षमा→[क्ष,मा]`, `विद्यालय→[वि,द्या,ल,य]`, `ज्ञान→[ज्ञा,न]`, `श्री→[श्री]`, `है।→[है,।]`), plus nukta (`ज़िंदगी`, `फ़ोन`), ZWJ, a mixed Latin/digit/Devanagari string, and a byte-level token sequence that splits a 3-byte Devanagari character (must produce `MIDUTF8`).

### T2 — `vocab.py`, `traces.py`, `gpu_traces.py`, and Kaggle notebook `kaggle/01_traces_v2.ipynb` (GPU, run by the user)

**`vocab.py`** — boolean masks over the full vocabulary (`len(tok)`), built with `token_to_bytes`:
- `starts_space`: first byte in `b" \n\t"`.
- `starts_punct`: first decoded char has Unicode category `P*` (includes `।`, `,`, `.`, `"` etc.).
- `starts_mark`: first decoded char has category `M*` (matra, halant/virama U+094D, nukta U+093C, anusvara, candrabindu, visarga) or is ZWJ/ZWNJ.
- `starts_cont`: first byte is a UTF-8 continuation byte (0x80–0xBF), i.e. token starts mid-character.
- `is_special`: special/control tokens; eos ids.
- If bytes are not decodable (partial UTF-8 at the end), decode with `errors="ignore"` only to get the first char; a token whose bytes *start* with a continuation byte is `starts_cont`, not `starts_mark`.
Tests: hand-pick a few real Llama tokens (skip if tokenizer unavailable) and check masks.

**`gpu_traces.py` + Kaggle notebook 01 (2× T4)** — port the legacy Part A code (loading, prompting, batched generation with resume, timing) into functions in `gpu_traces.py`; the notebook only calls them. Config flag `SPLIT ∈ {"dev","devtest"}`; run once per split. Timing (step 4) runs in whichever session runs first and is skipped if `timing.json` already exists in the output folder. Estimated runtime per split: generation of ~1000 sentences on 2× T4 plus two teacher-forced passes, roughly comparable to the legacy Part A run; print progress with tqdm and save after every batch. For the chosen split:
1. Generate 8B greedy target outputs (batched, as before) → `target_outputs_{split}.jsonl` (id, src, ref, prompt_ids, out_ids, out_text).
2. **Draft teacher-forced pass** (1B) over `prompt + out_ids`, per position `t`, from `q_t` (logits at `len(prompt)-1+t`), compute in float32:
   - `draft_argmax`, `H` (nats), `top1`, `top2`, `p_target = q_t(y[t])`,
   - `p_space = q_t(starts_space)`, `p_punct = q_t(starts_punct)`, `p_eos = q_t(eos ids)`, `p_newword = p_space + p_punct + p_eos`,
   - `p_mark = q_t(starts_mark)`, `p_cont = q_t(starts_cont)`,
   - `draft_topk_ids`, `draft_topk_logp` with k=20 (for sampling experiments in later phases).
3. **Target teacher-forced pass** (8B) over the same sequence: `target_topk_ids`, `target_topk_logp` (k=20), and `target_logp_of_draft_argmax`. Needed later for speculative-sampling simulation. Sanity check: target argmax at each position should equal `y[t]` for ≥ 99.5% of positions (fp16 + padding differences); report the rate and the positions that differ.
4. Timing as before (`t_draft`, `t_target`, `c`, `verify(g)` for g=1..8) → `timing.json`. Also time one extra configuration: draft forward with q=1 at context ≈ 150 and ≈ 400.
5. Save each split as compressed `.npz` with ragged arrays stored as flat arrays + `offsets` (do not use one giant JSON). Schema in `traces.py`:
   ```python
   @dataclass
   class Trace:   # one sentence
       id: int; n: int
       target_ids, draft_argmax, H, top1, top2, p_target,
       p_space, p_punct, p_eos, p_newword, p_mark, p_cont: np.ndarray
       draft_topk_ids, draft_topk_logp, target_topk_ids, target_topk_logp: np.ndarray  # (n, 20)
       match: np.ndarray  # derived: draft_argmax == target_ids
   def save_traces(path, traces, meta) / load_traces(path) -> list[Trace], meta
   ```
6. Zip to `/kaggle/working/traces_v2_{split}.zip` (traces `.npz`, `target_outputs_{split}.jsonl`, `timing.json`, a `run_info.json` with git hash, model names, transformers/torch versions, GPU names) and print the download instructions.

Keep the existing asserts: draft/target vocab sizes equal; tokenizers produce identical ids on a probe string containing Hindi and English.

**Acceptance:** smoke test passes on CPU before the notebook is built; after the user's run, both splits load locally with `load_traces`; summary printed (positions, agreement rate, mean H when match/mismatch); agreement on devtest within ~1–2% of the legacy 0.761 (from the PDF); `c` within ~10% of the legacy 0.334.

### T3 — `morph.py`: real morpheme labelling (fixes B1)

1. **Download and inspect.** `scripts/download_morphtok.py` downloads from the public Hugging Face dataset `karthika95/morphTok` (CC-BY-4.0), e.g. `huggingface_hub.hf_hub_download(repo_id="karthika95/morphTok", repo_type="dataset", filename="hindi.csv")`, into `data/morphtok/` (also fetch `marathi.csv` for later phases). Its README says each CSV has columns `word`, `segmentation` (morphemes joined with `+`), and `language`, but the files were re-uploaded recently, so **verify**: print columns, dtypes, row count, and 10 rows. Record the dataset revision (commit hash) in `results/phase1/morphtok_info.json`.
   Normalise all strings to NFC. The MorphTok segmentations include **sandhi splitting**, so the joined pieces may not spell the surface word exactly. For boundary labels, keep only entries where `"".join(segmentation.split("+")) == word`; report how many entries were kept and dropped (and show 10 dropped examples). If the format differs from the README, stop and ask the user.
2. `GoldLexicon`: `word -> set of internal boundary char offsets`.
3. `CRFSegmenter` (sklearn-crfsuite): character-level tagging "boundary after this char" with features = char identity, Unicode category, akshara position flags, char n-grams in a ±3 window, word length, position from start/end. Train on 90% of MorphTok words, evaluate boundary precision/recall/F1 on the held-out 10% (split by word type, fixed seed). **Post-process:** drop predicted boundaries that are not at akshara boundaries (`boundaries.akshara_units`). Save the model to `results/phase1/morph_crf.pkl` and metrics to `results/phase1/morph_crf_eval.json`.
4. `TieredLabeler.label(word) -> (boundaries: set[int], tier: str)` with tiers `gold` → `crf` → `none` (`none` only for non-Devanagari tokens such as digits/Latin). Report coverage on target outputs per tier (word tokens and word types), for dev and devtest.
5. `PrefixBoundaryModel` — the **causal** online signal. From all words in the labelled corpus (MorphTok words + all words in the FLORES `dev` target outputs labelled by the tiered labeller, weighted by frequency), for each akshara-aligned proper prefix of each word, count whether the prefix end is an internal morpheme boundary. Estimate `P(boundary | prefix)` with backoff: full prefix → last 3 aksharas of prefix → last 2 → last 1 → global rate (require a minimum count, e.g. 5, before trusting a level). Built **only from dev + MorphTok**, never devtest. Method: `prob(prefix: str) -> float`, cached with `functools.lru_cache`.
6. Per-token gold morph labels for analysis: `morph_boundary_after[t]` = True if the cut after token t is an internal morpheme boundary (from the tiered labeller) or a word boundary; plus `morph_tier[t]`.

Tests: gold lookup round-trips; CRF never outputs a boundary inside an akshara; prefix model returns values in [0,1] and backs off correctly on unseen prefixes.

### T4 — `simulator.py` (fixes B3, B8)

Port `simulate`, `summarize`, `formula_speedup`, `VERIFY` interpolation from the legacy notebook, then change:

1. **Policy interface:**
   ```python
   class Policy:
       name: str
       lookahead: int = 0          # 0 or 1
       def stop(self, view: "TraceView", p: int, g: int) -> bool: ...
   ```
   Policies are called with `g = 1 .. cap-1` after each drafted token (all dynamic policies draft at least one token; `FixedK` and `NoDraft` handle g=0 specially). Keep `gamma_max = 8` for every dynamic policy (same as the max fixed k).
2. **Causality guard:** `TraceView` wraps a `Trace` and exposes `view.feat(name, idx)` and `view.token(idx)`. The simulator sets `view.limit = p + g - 1 + policy.lookahead` before each call; any read with `idx > limit` raises `CausalityError`. Reading `match`, `target_ids[idx >= p]`, or gold labels raises unless the policy has `is_oracle = True`. Committed tokens (`idx < p`) are read from `target_ids`; drafted tokens (`p <= idx <= p+g-1`) are read from `draft_argmax`.
3. **Lookahead cost:** `cost += g*c + verify(g) + (c if policy.lookahead and stopped_by_policy and charge_lookahead else 0)`. `simulate(..., charge_lookahead: bool = True)`.
4. **Per-sentence outputs:** return arrays of per-sentence `emitted` and `cost` (for bootstrap), and the list of draft lengths and round records `(sentence_id, p, g, accepted)` for analysis.
5. Default `verify` = measured `VERIFY` (report the unit-cost version as a secondary column).

Tests (`test_simulator.py`): keep all legacy sanity checks (all-match k=4 → 2 rounds; no-match → every round emits 1; NoDraft → speed-up 1; oracle never rejects; every policy emits exactly n tokens). Add: lookahead charge adds exactly `c` per policy-stopped round; a deliberately cheating policy raises `CausalityError`; hand-computed small examples for the measured-verify cost path.

**Legacy comparison (no legacy traces exist, so this is a sanity check, not an exact regression):** once `traces_v2_devtest` is available (step S4), add a `legacy_protocol()` function that reproduces the old evaluation exactly: random half split of devtest with `np.random.default_rng(0).permutation(len(traces))`, entropy grid `lam = linspace(-1.5, 0.95, 40)`, `beta ∈ {0.5, 1, 2}`, unit verify cost, test on the second half. Expected: fixed k=2 ≈ 1.36 and entropy ≈ 1.45 (the PDF numbers). Because outputs are regenerated, agreement within ~0.03 is fine; a larger gap means something changed in generation or the simulator port and must be investigated before going further.

### T5 — Policies in `policies.py` (fixes B2, B3)

| class | lookahead | rule (stop when …) | grid (tuned on dev) |
|---|---|---|---|
| `NoDraft` | 0 | always | — |
| `FixedK(k)` | 0 | g ≥ k | k ∈ 1..8 |
| `Entropy(lam, beta)` | 0 | `1 - sqrt(beta*H[p+g-1]) < lam` | lam ∈ linspace(-1.5, 0.95, 40), beta ∈ {0.5, 1, 2, 4} |
| `WordBoundarySoft(tau, min_g)` | 1 | g ≥ min_g and `p_newword[p+g] > tau` | tau ∈ linspace(0.05, 0.95, 19), min_g ∈ {1, 2, 3} |
| `WordBoundaryHard(min_g)` | 1 | g ≥ min_g and `draft_argmax[p+g]` ∈ starts_space ∪ starts_punct ∪ eos (= corrected legacy policy) | min_g ∈ {1, 2, 3} |
| `AksharaSafeEntropy(lam, beta, tau_m, max_ext)` | 1 | entropy rule fires **and** `p_mark[p+g] < tau_m`; if entropy fired but akshara unsafe, keep drafting up to `max_ext` more tokens, then stop regardless | (lam, beta) around entropy best; tau_m ∈ {0.1, 0.3, 0.5, 0.7}; max_ext ∈ {1, 2, 3} |
| `Hybrid(lam, beta, tau_w, max_ext)` | 1 | entropy rule fires **and** `p_newword[p+g] > tau_w` (word complete); otherwise extend up to `max_ext` tokens then stop | lam, beta as above; tau_w ∈ {0.2, 0.4, 0.6, 0.8}; max_ext ∈ {0, 1, 2, 3} |
| `Oracle` | — | `not match[p+g]` (is_oracle) | — |
| `GoldWordOracle(min_g)` | — | stop at the gold word boundary (free lookahead, is_oracle) — labelled clearly as an upper bound for word-boundary stopping | min_g ∈ {1,2,3} |

All dynamic policies also stop at `gamma_max`. Tests (`test_policies_causality.py`): run every non-oracle policy over a synthetic trace with the causality guard on; no `CausalityError`.

### T6 — Morpheme policy (replaces `LookupPolicy`)

`MorphEntropy(lam, beta, tau_morph, use_akshara, tau_m, max_ext)`:
- At each step, build the current word prefix text from committed target tokens before `p` plus drafted tokens `p..p+g-1` (via `token_to_bytes`; decode only complete UTF-8; if the last drafted token starts a new word, the prefix restarts there).
- `pm = PrefixBoundaryModel.prob(prefix)` (causal, lookahead 0). If `use_akshara`, additionally require `p_mark[p+g] < tau_m` (then lookahead = 1).
- Stop when the entropy rule fires and (`pm > tau_morph` or word complete via `p_newword[p+g] > 0.5` when lookahead is enabled); otherwise extend up to `max_ext`.
- Also a pure variant without entropy: `MorphOnly(tau_morph, min_g)`.
- Cache prefix probabilities; the simulator over ~70k positions × grid must finish in reasonable time (target: full dev grid for all policies < 30 min on a laptop CPU; if slower, precompute per-position features where they do not depend on `p`, or use numba).

### T7 — Tuning and evaluation (`tuning.py`, `stats.py`, `02_policies_v2.ipynb`) (fixes B7, B8)

1. Load dev and devtest traces; attach gold labels (T1/T3) for analysis only.
2. For each policy family: grid search on **dev**, select by dev speed-up (lookahead charged). Save full grids to `results/phase1/grid_{policy}.csv`.
3. Evaluate the selected config of every policy on **devtest** with both `charge_lookahead=True/False`.
4. `stats.py`: bootstrap 95% CI over sentences (1000 resamples; speed-up = Σemitted / Σcost on the resample); paired bootstrap of the difference vs. `Entropy` (same resampled sentence indices), with the fraction of resamples where the difference > 0.
5. Main table `results/phase1/main_table.csv` columns: policy, params, lookahead, speedup_charged, CI_low, CI_high, speedup_uncharged, acceptance_rate, alpha_hat, tokens_per_verif, mean_draft_len, diff_vs_entropy, p_better, frac_drafts_ending_at_gold_word_boundary, frac_ending_mid_akshara, frac_ending_at_morph_boundary.
6. Also add: fraction of the oracle gap closed = (speedup − fixed_best) / (oracle − fixed_best).
7. Plots: speed-up with CI error bars per policy; acceptance vs. k; dev sweeps per policy; histogram of draft lengths per policy.
8. Write `results/phase1/selected_params.json` (policy class name → selected params) and `results/phase1/prefix_model.pkl.gz`; these are embedded into Kaggle notebook 03.

### T8 — Analysis (`analysis.py`) (fixes B6)

On devtest (and dev for comparison):
- Table: by `pos_in_word` (0, 1, 2, 3, 4+): count, mean H, match rate, mean `p_target`.
- Same by `start_cut` category (word-initial / mid-word / mid-akshara), by `morph_boundary_after`, and by token type (Devanagari, digit, Latin, punctuation).
- Signed point-biserial correlations of H with `word_initial`, with `end_cut == WORD`, and with `morph_boundary_after`; report r and the 95% CI by bootstrap over sentences. No auto-generated "insight" text.
- Save `results/phase1/analysis_*.csv` and one figure: match rate and mean entropy vs. position-in-word.

### T9 — Real draft-and-verify loop (`spec_loop.py`, Kaggle notebook `kaggle/03_real_loop.ipynb`, local `04_real_loop_analysis.ipynb`) (fixes B9, validates the simulator)

Split of work: the Kaggle notebook only **runs** things and saves raw logs; all comparisons and statistics happen locally in notebook 04. The Kaggle notebook embeds `results/phase1/selected_params.json` (best config of each policy from T7) and the prefix model needed by `MorphEntropy` (see §4 on embedded artifacts), and it pins the `transformers` version used in the smoke test.

1. Implement greedy speculative decoding, batch size 1, with KV caches for both models (`DynamicCache`; on rejection crop caches to the committed length — check the installed `transformers` version for the correct cache-crop API and pin the version in the notebook). Target sharded over 2 T4s, draft on cuda:0.
2. Policies run **live**: compute the same features as `vocab.py`/T2 from the draft's actual logits; morph prefix uses the actual drafted tokens. Implement lookahead policies the natural way: compute `q_{p+g}`, decide whether to sample from it.
3. Baselines in the same loop: plain greedy target decoding (no draft), with the same cache code, for wall-clock comparison.
4. Run on 100 devtest sentences (fixed list of ids saved to `results/phase1/real_loop_ids.json`) for: NoDraft, FixedK(best), Entropy(best), WordBoundarySoft(best), Hybrid(best), MorphEntropy(best).
5. **Losslessness check:** output ids must equal *unbatched* greedy target output for each sentence. Count mismatches; for each mismatch, log the first divergent position and the target logit margin there (fp16 near-ties are acceptable but must be reported).
6. **Data for simulator validation:** in the same Kaggle run, compute teacher-forced draft and target traces (same function as notebook 01) for the 100 *unbatched* greedy outputs and save them. Locally (notebook 04), run the simulator with each selected policy on these traces and compare per policy: tokens/verification, mean draft length, and predicted vs. measured speed-up. Report relative error; flag anything > 5% and explain it (teacher-forcing approximation, lookahead cost, cache overhead).
7. Report wall-clock tokens/s and speed-up vs. plain greedy, median over 3 runs, with warm-up. Also log per-round records (g, accepted, draft time, verify time) so the cost model `g*c + verify(g)` can be checked against measured time.
8. Kaggle output: `real_loop.zip` (per-round logs, per-sentence outputs, unbatched greedy outputs, traces for the 100 sentences, run_info.json). Local outputs: `results/phase1/real_loop.csv`, `results/phase1/sim_vs_real.csv`.
9. Budget: 6 policies + plain greedy × 100 sentences × 3 repeats on 2× T4 may be long. Add a config flag `REPEATS` (default 3) and `N_SENT` (default 100); print an estimated runtime after the first 5 sentences so the user can lower them if needed.

### T10 — Slide/script corrections (`docs/slide_corrections.md`)
Draft corrected wording for the user (do not edit their slides directly):
- Slide 2: remove "When a draft ends mid-word or mid-morpheme, acceptance drops" unless T8 supports it; replace with the measured statement.
- Slide 3: hybrid description must match the implemented policy (entropy gate + akshara/word lookahead).
- Slide 5: remove "morpheme result is a coverage failure: MorphTok covers only 41%" — the previous run used an 11-word placeholder. Replace with new numbers from T7, including CI and coverage per tier from T3.
- All headline numbers must come from `results/phase1/main_table.csv` (new splits: tuned on dev, tested on devtest), and real-loop numbers from T9.

---

## 6. Definition of done for this phase

- [ ] `pytest` passes locally (gated-tokenizer tests may skip without `HF_TOKEN`).
- [ ] Test 1 numbers reproduced via `boundaries.py` (T1 regression).
- [ ] CPU smoke tests pass for the code paths of both Kaggle notebooks; generated notebooks in `kaggle/` with the git hash recorded.
- [ ] Traces v2 for dev and devtest with all features, run by the user on Kaggle and loaded locally (T2).
- [ ] Legacy-protocol sanity check on new devtest traces lands near the PDF numbers (fixed k=2 ≈ 1.36, entropy ≈ 1.45), or the gap is explained (T4).
- [ ] MorphTok re-downloaded with revision recorded; tiered labeller with CRF held-out F1 and per-tier coverage reported (T3).
- [ ] Every policy causal (guard on), tuned on dev, evaluated on devtest with bootstrap CIs and paired differences vs. entropy (T5–T7).
- [ ] Position-in-word analysis tables and figure (T8).
- [ ] Real loop: losslessness verified, simulator validated within 5% or discrepancy explained (T9).
- [ ] `docs/slide_corrections.md` written (T10).
- [ ] A short `results/phase1/README.md` summarising what was run, with the main table and any caveats (teacher-forcing approximation, lookahead charged vs uncharged, 2× T4 layer split affecting `c`).

## 7. Out of scope for this phase (do not start yet)
Learned stopping classifier (Phase 2), Marathi/Tamil, English control, other tokenizers/model pairs, speculative sampling at temperature > 0, other tasks. The traces from T2 already store what those phases need.
