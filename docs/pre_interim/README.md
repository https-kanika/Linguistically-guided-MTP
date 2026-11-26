# Pre-interim work

This folder records what the project looked like up to the interim review: the problem statement, the
plan, the first experiments, and the results shown at the interim. The numbers here are the ones we had
**at the time**. Some of them were later found to be wrong (see [the audit](#what-changed-after-the-interim));
the corrected versions are in [`results/phase1/REPORT.md`](../../results/phase1/REPORT.md).

The notebooks that produced these numbers are kept unchanged in
[`notebooks/pre_interim/`](../../notebooks/pre_interim/):

| notebook | what it ran |
|---|---|
| `test1_hindi_boundaries.ipynb` | Test 1: how Llama-3.2 and Qwen3.5 tokenizers split Hindi, and where fixed-k drafts end (CPU) |
| `day2-day3-llama8b-kaggle.ipynb` | Day 2/3 on Kaggle (2× T4): 8B target outputs, 1B draft traces, timing, trace simulator, fixed-k / entropy / oracle and the linguistic policies |
| `lama8b-kaggle-test.ipynb` | The same evaluation as a cleaned-up end-to-end run, used for the interim slides |

[`interim_slide_corrections.md`](interim_slide_corrections.md) lists which statements on the interim
slides had to be corrected after the re-run, with the corrected numbers.

---

## 1. Problem statement (proposal)

Multi-token prediction and speculative decoding usually draft a **fixed number of tokens**, regardless of
whether they form a complete word, morpheme or written syllable. In morphologically rich languages written
in Indic scripts, a fixed window often cuts through words, morphemes and grapheme clusters (a matra, a
halant or a conjunct).

We proposed to make drafting **linguistically aware**: the draft model generates normal subword tokens, but
an adaptive stopping controller decides when to stop and send the draft for verification. Instead of
always proposing k tokens, it stops at meaningful positions:

- a complete word boundary;
- a morphologically valid boundary (e.g. root + suffix in Hindi, using MorphTok);
- a script-safe akshara boundary that does not split a matra, halant or conjunct;

combined with the draft model's confidence: verify early when entropy is high, keep going to the next
valid boundary when it is low, up to a maximum draft length.

Policies to compare: fixed window, entropy only, word-aware, morphology-aware, and hybrid
entropy + boundary. Metrics: acceptance rate, accepted tokens per verification, speed-up, and how often a
draft ends at a valid boundary.

## 2. Plan

| step | content |
|---|---|
| Test 1 (CPU) | Tokenize Hindi FLORES+ devtest (1,012 sentences) with the Llama-3.2 tokenizer; tokens per word; for k = 2..8, how often a fixed window ends mid-word, mid-akshara or mid-UTF-8 character. **Decision point:** if windows rarely end mid-word, reframe as a characterization study. |
| Day 2 | Target model translates English→Hindi FLORES+ (greedy); the 1B draft reads those outputs and we save per-position argmax and entropy; measure the cost ratio c = t_draft / t_target; build the MorphTok boundary lookup. |
| Day 3 | Trace simulator; fixed-k baseline (k = 2..8); entropy policy (AdaEDL/SVIP rule: stop when 1 − √(βH) < λ); oracle upper bound. |
| Day 4 | Word-boundary, morpheme-boundary and hybrid policies; correlation between entropy and word boundaries; check with a real draft-and-verify loop. |
| Day 5 | Full runs and plots: acceptance vs k, speed-up per policy, share of drafts ending at a boundary. |

## 3. Test 1: how tokenizers split Hindi

FLORES+ `hin_Deva` devtest, 1,012 sentences.

| | Llama-3.2-1B | Qwen3.5-0.8B (as measured then) |
|---|---|---|
| tokens | 68,511 | 85,415 |
| tokens per word | **2.67** | 3.33 |
| tokens that start mid-akshara | 53.5% | 45.8% |
| token cuts at a word boundary | 38.6% | 33.0% |
| token cuts inside an akshara | **54.3%** | 46.4% |
| k = 4 drafts that end inside a word | **60.6%** | 66.4% |
| k = 4 drafts that contain no complete word | 4.5% | 11.5% |

Drafts with no complete word, by k (Llama-3.2): 27.8% (k = 2), 11.9%, 4.5%, 1.5%, 0.4%, 0.1%, 0.0% (k = 8).

The decision point was passed: fixed windows do end mid-word most of the time.

*Later correction:* the Qwen3.5 numbers came from an outdated pre-tokenizer in the `transformers` version
we used. With the tokenizer's own rules Qwen3.5 needs 2.16 tokens per word (see
`results/phase1/REPORT.md` §5). The Llama numbers were reproduced exactly.

## 4. Day 2/3 with a 3B target

Llama-3.2-1B draft, Llama-3.2-3B target, FLORES+ devtest.

- Token-level draft/target agreement 79.5%; mean draft entropy 0.92 nats (0.61 when the draft is right,
  2.14 when it is wrong).
- Draft step 24.8 ms, target step 40.6 ms: **c = 0.61**.
- MorphTok coverage on the model outputs: 71% of word tokens, 41% of word types.

| policy (3B target) | acceptance | tokens / verification | speed-up |
|---|---|---|---|
| fixed k = 2 (best fixed) | 0.691 | 2.35 | 1.065 |
| entropy (λ = 0.32, β = 1) | 0.772 | 2.47 | 1.134 |
| oracle (cap 8) | 1.000 | 3.96 | 1.399 |

Fixed k got slower with every extra token (k = 8: 0.70×). With c = 0.61 the draft is too expensive for
speculation to pay much, so we moved to a larger target.

## 5. Day 2–5 with the 8B target (interim results)

Llama-3.2-1B draft, Meta-Llama-3-8B-Instruct target, fp16 on 2× T4.

- t_target = 70.6 ms, t_draft = 23.6 ms: **c = 0.33**.
- Token agreement 76.1%; entropy 0.66 nats when right, 2.09 when wrong.

| policy (8B target, as presented at the interim) | acceptance | tokens / verification | mean draft | speed-up |
|---|---|---|---|---|
| fixed k = 2 (best) | 0.651 | 2.26 | 1.98 | 1.361 |
| **entropy (λ = −0.37, β = 2)** | 0.684 | 2.63 | 2.44 | **1.452** |
| word boundary | 0.623 | 2.27 | 2.09 | 1.335 |
| morpheme lookup | 0.374 | 3.40 | 6.57 | 1.066 |
| hybrid akshara | 0.690 | 1.81 | 1.21 | 1.287 |
| oracle (cap 8) | 1.000 | 3.58 | 2.63 | 1.905 |
| oracle (cap 32) | 1.000 | 3.92 | 2.98 | 1.966 |

Entropy–word-boundary correlation (dev): r = −0.27.

**What we concluded at the interim:** entropy stopping is the best real policy; word boundary is about
equal to fixed k; the morpheme rule is weak because of MorphTok coverage; the oracle shows a lot of
headroom. Planned next steps: a learned stopping signal, AdaEDL-style dynamic thresholds, a better
morpheme lookup, and Marathi/Tamil.

## What changed after the interim

Before building on these results we audited the code. Three of the six rows did not measure what they
claimed:

| interim row | problem |
|---|---|
| morpheme lookup | never used MorphTok: it looked words up in an 11-word placeholder list, so it almost never stopped (mean draft 6.57) |
| hybrid akshara | had no entropy part; its akshara test almost always passed, so it stopped after about one token |
| word boundary | read the draft's next token without paying for the extra draft call |

Also, only the entropy rule had been tuned, tuning and reporting used two halves of devtest, and tokens
were decoded one by one (which breaks Devanagari characters split across byte-level tokens).

The code comments refer to these problems by ID:

| ID | problem in the interim code | fixed in |
|---|---|---|
| B1 | "Morpheme lookup" never used MorphTok (11-word placeholder list) | `src/lgmtp/morph.py`, `policies.py` |
| B2 | Hybrid policy had no entropy gate and an akshara check that could not work | `src/lgmtp/policies.py` |
| B3 | Word-boundary policy read an undrafted token for free | `src/lgmtp/simulator.py` (causality guard), `policies.py` |
| B4 | Byte-level tokens decoded one by one | `src/lgmtp/boundaries.py` |
| B5 | Boundaries computed on draft tokens instead of the target's text | `src/lgmtp/traces.py` |
| B6 | Entropy–boundary correlation read without its sign; p-values meaningless at this size | `src/lgmtp/analysis.py` |
| B7 | Only the entropy rule was tuned | `src/lgmtp/tuning.py` |
| B8 | Tuning and reporting on different halves of devtest | `src/lgmtp/tuning.py`, `simulator.py` |
| B9 | Batched, padded target generation, so losslessness was never checked | `src/lgmtp/gpu_traces.py`, `spec_loop.py` |

The pipeline was rebuilt and every GPU output regenerated (`results/phase1/`). The fixed-k, entropy and
oracle rows reproduced within 0.03. With fair tuning and charged look-ahead, **no linguistic stopping rule
beats the entropy rule** (best: 1.427× vs 1.452×), because draft entropy already captures where words
start. That result led to the second half of the project (see the main [README](../../README.md)).
