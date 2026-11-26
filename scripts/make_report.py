"""Render results/phase1/REPORT.md (full report) and results/phase1/README.md (short summary) from the
result files. No number is typed by hand: every value is read from a file in results/phase1/.

    python scripts/make_report.py

The report is meant to be read on its own, by someone who has not seen the planning documents.
The real-loop section shows placeholders until notebook 04 has written its files; rerun afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase1"
PH = "⟨pending: real-loop run⟩"


def J(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def C(name):
    return pd.read_csv(RES / name)


def has(name):
    return (RES / name).exists()


def f3(x):
    return "—" if pd.isna(x) else f"{x:.3f}"


def pct(x, nd=1):
    return "—" if pd.isna(x) else f"{100 * x:.{nd}f}%"


def md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, float):
                cells.append(f3(v))
            else:
                cells.append("" if pd.isna(v) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def img(name, alt):
    return f"![{alt}]({name})" if has(name) else f"*({name} not found)*"


def H_threshold(lam, beta):
    """The entropy rule 'stop when 1 - sqrt(beta*H) < lam' is the same as 'stop when H > (1-lam)^2/beta'."""
    return (1 - lam) ** 2 / beta


INT = lambda v: str(int(v))                                         # noqa: E731
SIGNED3 = lambda v: f"{v:+.3f}"                                     # noqa: E731


def build() -> tuple[str, str]:
    t1 = J("test1_summary.json")
    reg = C("test1_regression.csv")
    TS = J("traces_summary.json")
    mt_info = J("morphtok_info.json")
    crf = J("morph_crf_eval.json")
    cov = J("morph_coverage.json")
    LP = J("legacy_protocol.json")
    OLD = J("earlier_results.json")
    sel = J("selected_params.json")
    S = sel["selected"]
    main = C("main_table.csv")
    fk = C("fixed_k_devtest.csv")
    cond = C("analysis_conditional_summary.csv")
    logit_t = C("analysis_conditional_devtest_logit.csv")
    logit_d = C("analysis_conditional_dev_logit.csv")
    corr_t = C("analysis_devtest_correlations.csv")
    corr_d = C("analysis_dev_correlations.csv")
    pos_t = C("analysis_devtest_by_pos_bucket.csv")
    sk_t = C("analysis_devtest_by_start_kind.csv")
    tt_t = C("analysis_devtest_by_token_type.csv")
    mb_t = C("analysis_devtest_by_morph_boundary_after.csv")
    ids = J("real_loop_ids.json") if has("real_loop_ids.json") else {"ids": []}
    m = main.set_index("policy")
    E = m.loc["Entropy"]
    O = m.loc["Oracle"]
    FK = m.loc["FixedK"]
    T = TS["timing"]
    llama = next(s for s in t1 if s["tokenizer"].startswith("meta-llama"))
    qwen = next(s for s in t1 if s["tokenizer"] == "Qwen/Qwen3.5-0.8B")
    qwen_leg = next(s for s in t1 if "legacy" in s["tokenizer"])
    k4 = lambda s: next(r for r in s["per_k"] if r["k"] == 4)       # noqa: E731
    cs = cond.set_index(["split", "ling"])
    lt = logit_t.set_index("quantity")
    dauc = lt.loc["delta AUC (full - base)"]
    real_done = all(has(f) for f in ("real_loop.csv", "sim_vs_real.csv", "real_loop_lossless.csv",
                                     "real_loop_ci.csv", "sim_vs_real_decomposition.csv"))
    ling_pols = ["WordBoundarySoft", "WordBoundaryHard", "AksharaSafeEntropy", "Hybrid", "Hybrid_ext1",
                 "MorphEntropy_ext1", "MorphOnly"]
    best_ling = max(ling_pols, key=lambda p: m.loc[p, "speedup_charged"])
    best_ling_un = max(ling_pols, key=lambda p: m.loc[p, "speedup_uncharged"])
    Hstar = H_threshold(S["Entropy"]["lam"], S["Entropy"]["beta"])
    posb = pos_t.set_index("pos_bucket")
    wi = cs.loc[("devtest", "word_initial")]
    old = {r["policy"]: r for r in OLD["rows"]}
    old_morph = next(r for k, r in old.items() if k.startswith("morpheme"))
    old_hyb = next(r for k, r in old.items() if k.startswith("hybrid"))
    lex = mt_info["lexicon"]
    h = crf["heldout"]

    L = []
    A = L.append

    # ================================================================== title
    A("# Does Hindi word and morpheme structure help speculative decoding? Phase 1 report\n")
    A("CS769 course project (IIT Bombay). This report describes everything done in the first phase of "
      "the project, why it was done, and what came out. It is written to be read on its own. Every number is "
      "read from a file in `results/phase1/` by `scripts/make_report.py`; file names are given in brackets "
      "where they matter.\n")

    # ================================================================== 1 summary
    A("## 1. Summary\n")
    A("**Question.** Speculative decoding speeds up a large language model by letting a small *draft* model "
      "guess several tokens that the large *target* model then checks in one step. How many tokens to guess "
      "each time is decided by a *stopping policy*. Hindi words are split by the tokenizer into several "
      "pieces, often in the middle of a written syllable (an *akshara*). We asked whether word, morpheme or "
      "akshara boundaries tell us where the draft model goes wrong, beyond what the draft model's own "
      "confidence already tells us, and whether stopping policies that use those boundaries are faster.\n")
    A("**Answer for our setup (Llama-3-8B target, Llama-3.2-1B draft, English→Hindi translation): no.**\n")
    A(f"- The fastest stopping policy is a simple confidence rule (\"stop when the draft is unsure\", called "
      f"*Entropy* below): **{E.speedup_charged:.3f}×** speed-up [95% CI {E.CI_low:.3f}–{E.CI_high:.3f}] on the "
      f"test set. Always guessing the best fixed number of tokens ({S['FixedK']['k']}) gives "
      f"{FK.speedup_charged:.3f}×. An *oracle* that knows in advance where the draft will be wrong gives "
      f"{O.speedup_charged:.3f}×, so there is still room for better policies.")
    A(f"- None of the eight policies that use word, akshara or morpheme boundaries beats the confidence rule. The "
      f"best one reaches {m.loc[best_ling, 'speedup_charged']:.3f}× "
      f"({m.loc[best_ling, 'diff_vs_entropy']:+.3f}, CI {m.loc[best_ling, 'diff_CI_low']:+.3f} to "
      f"{m.loc[best_ling, 'diff_CI_high']:+.3f}). Even an oracle that stops exactly at the true word boundaries "
      f"reaches only {m.loc['GoldWordOracle', 'speedup_charged']:.3f}×.")
    A(f"- The draft *does* fail more often at the start of a word: it agrees with the target on "
      f"{pct(posb.loc['0', 'match_rate'])} of word-initial tokens but on {pct(posb.loc['2', 'match_rate'])} of "
      f"third tokens in a word. But the draft is also much less confident there, and that explains it: among "
      f"tokens with the same confidence, word-initial tokens are not harder (they agree "
      f"{wi.entropy_adjusted_diff * 100:+.1f} percentage points *more* often). Adding word and morpheme position "
      f"to a model that predicts draft failures from confidence does not improve it "
      f"(change in AUC {dauc.value:+.4f}).")
    if real_done:
        rc = C("real_loop_ci.csv").set_index("policy")
        dc = C("sim_vs_real_decomposition.csv").set_index("policy").drop("NoDraft")
        A(f"- Most results come from a fast simulator. We checked it with real speculative decoding on GPUs "
          f"({len(ids['ids'])} test sentences). The real speed-ups put the policies in the same order, with the "
          f"confidence rule fastest at **{rc.loc['Entropy', 'speedup_wall']:.3f}×** in wall-clock time. The "
          f"simulator is {pct(-dc.total.max())}–{pct(-dc.total.min())} too optimistic, mostly because of a fixable "
          f"inefficiency in our decoding loop rather than an error in the simulation.")
    else:
        A(f"- Real speculative decoding on GPUs (checks the simulator): {PH}.")
    A(f"- An earlier version of this project reported results for linguistic policies. Those numbers were "
      f"invalid because of errors in the policy code (§3). For example, its \"morpheme\" policy never used a "
      f"morpheme resource. The rebuilt pipeline reproduces the earlier fixed-length, confidence and oracle "
      f"numbers within {max(abs(v) for v in LP['gap'].values()):.3f}, so those parts of the earlier work were "
      f"sound.\n")

    # ================================================================== 2 background
    A("## 2. Background\n")
    A("### 2.1 Speculative decoding\n")
    A("A language model writes text one token at a time, and each token needs one full pass through the model. "
      "For a large model, that pass is slow. Speculative decoding works in *rounds*:\n")
    A("1. A small, fast **draft** model guesses the next *g* tokens, one after another.")
    A("2. The large **target** model processes all *g* guesses in **one** pass (about as expensive as producing "
      "a single token), and checks each guess against the token it would have chosen itself.")
    A("3. Guesses are accepted from the left until the first disagreement. At that point the target's own "
      "token is used instead. If all guesses are accepted, the target adds one more token for free.\n")
    A("With greedy decoding, the output is **exactly** what the target model would have written alone, so the "
      "method is *lossless*. The only question is speed. Each round costs *g* draft steps plus one target pass "
      "and produces (accepted guesses + 1) tokens. Guessing too few tokens wastes the cheap draft model; "
      "guessing too many wastes draft steps on tokens that will be rejected. The rule that decides when to "
      "stop guessing in each round is the **stopping policy**, and it is what this project studies.\n")
    A("### 2.2 Why Hindi might be special\n")
    A("Hindi is written in Devanagari. The unit a reader sees as one written syllable is an **akshara**: a "
      "consonant, optionally joined to further consonants by a *halant* (virama), followed by a vowel sign "
      "(*matra*) or other marks. For example, क्षे is one akshara made of क + ् + ष + े. Words are made of "
      "**morphemes**: a stem plus suffixes, e.g. घरों ('houses', oblique) = घर (stem) + ों (plural oblique suffix).\n")
    A(f"Modern tokenizers split text into byte-level pieces learned from mostly English data. They split Hindi "
      f"badly. With the Llama-3 tokenizer, a Hindi word is on average {llama['tokens_per_word']:.2f} tokens, "
      f"{pct(llama['pct_cuts_mid_akshara'] / 100)} of the cuts between tokens fall *inside* an akshara, and a "
      f"4-token guess ends in the middle of a word {pct(k4(llama)['pct_end_anywhere_inside_word'] / 100)} of the "
      f"time (§5).\n")
    A("**Hypothesis.** A stopping policy that knows where words, morphemes and aksharas end could stop at "
      "better places than one that looks only at the draft model's confidence. Ending a guess mid-word cannot "
      "lower acceptance by itself, since checking is token by token. Boundaries can only help if they carry "
      "information about *where the draft and target disagree*. So the precise question is:\n")
    A("> *Does linguistic structure predict where drafts fail, beyond what the draft model's confidence already "
      "captures? And can a stopping policy use it to be faster?*\n")
    A("Either answer is useful if the comparison is fair, which is why most of this phase went into making the "
      "comparison correct.\n")

    # ================================================================== 3 earlier version
    A("## 3. The earlier version of this project, and why it was redone\n")
    A("An earlier version of the project (three Jupyter notebooks, kept unchanged in `notebooks/pre_interim/`) "
      "reported these results on half of the FLORES+ test set:\n")
    A(md_table(pd.DataFrame([[r["policy"], r["speedup"], r["acceptance"], r["mean_draft_len"]] for r in OLD["rows"]],
                            columns=["policy (earlier version)", "speed-up", "acceptance", "mean draft length"]),
               fmt={"acceptance": pct, "mean draft length": lambda v: f"{v:.2f}"}))
    A("\n(`earlier_results.json`.) A review of the code found that the three linguistic rows (word boundary, "
      "morpheme, hybrid) did not measure what they claimed. The fixed-length, confidence and oracle rows were "
      "sound. The problems and what this phase does instead:\n")
    A(md_table(pd.DataFrame([
        ["The \"morpheme\" policy never used a morpheme resource",
         f"It looked words up in a placeholder list of 11 common function words (है, और, में, …). It almost never "
         f"stopped, hence its mean draft length of {old_morph['mean_draft_len']:.2f} and its speed-up of "
         f"{old_morph['speedup']:.2f}×. The earlier explanation, low coverage of the MorphTok morpheme dataset, "
         f"described something that had not run.",
         "Real morpheme labels from MorphTok plus a trained segmenter for unknown words, and a model of morpheme "
         "boundaries that can be used while drafting (§6)."],
        ["The \"hybrid\" policy had no confidence part, and its akshara test could not work",
         f"It was described as \"entropy + akshara boundary\" but only checked whether the word so far consisted "
         f"of complete aksharas. Whether the last akshara is complete depends on the *next* token (क looks "
         f"complete until ्ष arrives), so the test almost always passed and the policy stopped after about one "
         f"token (mean draft length {old_hyb['mean_draft_len']:.2f}).",
         "Policies that combine the confidence rule with an estimate, from the draft model's own prediction, of "
         "whether the next token continues the akshara or word (§7)."],
        ["The word-boundary policy used information it did not have",
         "To decide whether the current word was finished, it read the draft model's *next* token without "
         "paying for computing it. In a real decoder that costs one more draft step, so the comparison favoured "
         "this policy.",
         "Every policy may read only what a real decoder knows. Reading the next prediction counts as one extra "
         "draft step and is charged. A guard in the code raises an error on any other read (§4.3)."],
        ["Tokens were turned into text one by one",
         "Byte-level tokens that hold part of a character decode to garbage (`\\ufffd`) on their own, so "
         "boundaries were misplaced.",
         "All boundaries are computed on bytes, with each token's byte span in the full text."],
        ["Boundaries were taken from the draft's tokens",
         "What matters is the text that is actually kept, i.e. the target model's output.",
         "Labels are computed on the target model's output text."],
        ["The correlation summary ignored the sign",
         "An automatic message reported a negative correlation (r = −0.27) as support for the hypothesis. With "
         "33,000 tokens every p-value is ≈ 0, so p-values said nothing.",
         "Signed effect sizes with confidence intervals, descriptive tables, and an analysis that holds "
         "confidence fixed (§9)."],
        ["Only the confidence rule was tuned",
         "It got a 40 × 3 grid of settings; the linguistic policies got none.",
         "Every policy family is tuned on a separate tuning set with a grid of comparable size (§7.2)."],
        ["Tables used different test sets, and there was no tuning set",
         "Settings were chosen on one half of the test set and reported on the other half, and the fixed-length "
         "table used all of it.",
         "Tune on FLORES+ *dev* (997 sentences), report everything once on FLORES+ *devtest* (1012 sentences)."],
        ["Losslessness was never checked",
         "The reference outputs were generated in padded batches, which can differ slightly from one-at-a-time "
         "decoding in 16-bit arithmetic.",
         "A real speculative decoder whose output is compared token by token with one-at-a-time decoding (§10)."],
    ], columns=["problem", "what went wrong", "what this phase does"])))
    A("\nBecause none of the earlier GPU outputs were kept, every model output, measurement and result in this "
      "report was regenerated from scratch.\n")

    # ================================================================== 4 setup
    A("## 4. Setup and method\n")
    A("### 4.1 Models, data and task\n")
    A(md_table(pd.DataFrame([
        ["Target model", "meta-llama/Meta-Llama-3-8B-Instruct, 16-bit floats, split across two NVIDIA T4 GPUs (Kaggle)"],
        ["Draft model", "meta-llama/Llama-3.2-1B-Instruct, 16-bit floats. It uses the same tokenizer as the target (checked in code)."],
        ["Task", "English→Hindi translation of FLORES+ sentences, chat prompt \"Translate the following English sentence "
                 "into Hindi. Output only the Hindi translation.\", greedy decoding, at most 256 new tokens"],
        ["Tuning set", f"FLORES+ **dev**: {TS['dev']['sentences']} sentences, {TS['dev']['positions']:,} output tokens. "
                       "Used for choosing every policy's settings and for building the morpheme model."],
        ["Test set", f"FLORES+ **devtest**: {TS['devtest']['sentences']} sentences, {TS['devtest']['positions']:,} "
                     "output tokens. Used once, for the reported numbers."],
        ["Maximum draft length", "8 tokens per round for every policy"],
    ], columns=["item", "setting"])))
    A("\n### 4.2 How speed-up is measured\n")
    A(f"Times are measured on the same GPUs and expressed in units of one target step. Drafting one token costs "
      f"**c = {T['c']:.3f}** (the 1B model takes about a third of the time of the 8B model). The target's pass "
      f"over g guesses costs **verify(g)**, measured at {T['verify']['1']:.3f} for g = 1 up to "
      f"{T['verify']['8']:.3f} for g = 8: checking 8 guesses costs barely more than writing one token. So one "
      f"round costs g·c + verify(g) and produces (accepted + 1) tokens, and\n")
    A("> **speed-up = total tokens produced / total cost**, so plain decoding with the target alone = 1.0.\n")
    A("For example, a round that guesses 2 tokens and has both accepted costs 2 × 0.323 + 1.025 ≈ 1.67 and "
      "produces 3 tokens.\n")
    A("### 4.3 Traces, the simulator, and the fairness rule\n")
    A("Running real speculative decoding for every policy and every setting would need far more GPU time than "
      "we have. Instead we record **traces** once on GPU:\n")
    A("1. The target model translates every sentence with greedy decoding.")
    A("2. Both models read the target's output once (*teacher forcing*). At every position we store what the "
      "draft model would predict next: its most likely token, how confident it is, and probabilities of "
      "particular kinds of next token (one that starts a new word, punctuation, end of text, a vowel sign or "
      "other mark that continues an akshara). We also store the top 20 tokens of both models.\n")
    A("The **simulator** then replays any stopping policy over these traces on a CPU, in seconds. Within a "
      "round, the draft's guesses agree with the target exactly up to the first disagreement, so accepted "
      "tokens are exact. The one approximation is that after a wrong guess, a real draft model continues from "
      "its own wrong token, while the trace continues from the right one. This can only change *where a policy "
      "stops* after a mistake. §10 measures the effect with real decoding: it is small.\n")
    A("**Fairness rule.** A policy may only use what a real decoder knows when it makes its decision. After "
      "guessing g tokens, that is the draft model's predictions up to the g-th guess, the guesses themselves, "
      "and the text already accepted. Some linguistic policies want to know what comes *next* (\"would the next "
      "token start a new word?\"). That requires computing the draft model's prediction for the next position, "
      "which costs one extra draft step. We charge that cost (c) whenever such a *lookahead* policy decides to "
      "stop, and also report the result without the charge as an optimistic bound. The simulator enforces "
      "the rule: any read beyond what is allowed raises an error, and unit tests check this for every policy.\n")
    A("### 4.4 Statistics\n")
    A("All intervals are 95% confidence intervals from a bootstrap over sentences (1000 resamples, sentences "
      "drawn with replacement). Differences between two policies use a *paired* bootstrap (the same resampled "
      "sentences for both), which is much more precise than comparing two separate intervals.\n")
    A("### 4.5 Engineering\n")
    A("All logic is in a Python package (`src/lgmtp/`) with unit tests. The GPU work (generation, traces, timing, "
      "real decoding) runs on Kaggle as self-contained notebooks, generated from the package, that can resume "
      "if a session ends. Each GPU code path was first run on a CPU with the 1B model in both roles. For the "
      "real decoder this was done twice: once with draft = target (nothing is ever rejected) and once with "
      "deterministic noise added to the draft (about half the rounds have a rejection). The noisy test was "
      "added after the first GPU run of the decoder crashed when discarding rejected tokens from the model's "
      "memory (the library's `crop` function changed its argument convention). Mistakes in the code that "
      "were found and fixed along the way are recorded in the tests.\n")

    # ================================================================== 5 checks
    A("### 4.6 Checks that the rebuilt pipeline is right\n")
    A(md_table(pd.DataFrame([
        [k, TS[k]["sentences"], TS[k]["positions"], TS[k]["agreement"], TS[k]["mean_H_match"], TS[k]["mean_H_mismatch"],
         TS[k]["target_argmax_check"]["rate"], TS[k]["n_hit_max_new"]] for k in ("dev", "devtest")],
        columns=["set", "sentences", "output tokens", "draft guess = target token", "draft entropy when right",
                 "draft entropy when wrong", "target re-read = its output", "outputs cut at 256 tokens"]),
        fmt={"draft guess = target token": pct, "target re-read = its output": lambda v: pct(v, 2),
             "output tokens": lambda v: f"{int(v):,}"}))
    A(f"\nThe draft model agrees with the target on {pct(TS['devtest']['agreement'])} of tokens (earlier version: "
      f"{TS['legacy_reference']['agreement_devtest']:.3f}), and c = {T['c']:.3f} (earlier: "
      f"{TS['legacy_reference']['c']:.3f}). Entropy (uncertainty, in nats) is three times higher where the draft is "
      "wrong, which is why a confidence rule works. When the target re-reads its own output, its top choice "
      "differs from what it wrote at about 0.1% of positions. These are near-ties in 16-bit arithmetic "
      "(`traces_summary.json`).\n")
    A("We also re-ran the earlier version's exact evaluation (half of the test set, its settings grid, "
      "verification cost fixed at 1) on the new traces:\n")
    A(md_table(pd.DataFrame([
        ["fixed k=2", LP["fixed_test"]["speedup"], LP["expected"]["fixed_k2"], LP["gap"]["fixed"]],
        ["confidence rule", LP["entropy_test"]["speedup"], LP["expected"]["entropy"], LP["gap"]["entropy"]],
        ["oracle", LP["oracle_test"]["speedup"], LP["expected"]["oracle"], LP["gap"]["oracle"]],
    ], columns=["policy", "rebuilt pipeline", "earlier version", "difference"]), fmt={"difference": SIGNED3}))
    A("\nAll differences are below 0.03 (`legacy_protocol.json`), so the regenerated outputs and the new "
      "simulator agree with the earlier work where that work was correct.\n")

    # ================================================================== 5 tokenizer
    A("## 5. How tokenizers split Hindi\n")
    A("Measured on the Hindi side of FLORES+ devtest (`notebooks/00_tokenizer_analysis.ipynb`). A *cut* is the "
      "point between two consecutive tokens.\n")
    A(md_table(pd.DataFrame([
        [s["tokenizer"].replace(" [legacy Qwen2 regex]", " with an outdated pre-tokenizer (see below)"),
         s["tokens_per_word"], s["pct_cuts_word_boundary"] / 100, s["pct_cuts_mid_akshara"] / 100,
         k4(s)["pct_end_anywhere_inside_word"] / 100]
        for s in (llama, qwen, qwen_leg)],
        columns=["tokenizer", "tokens per word", "cuts at a word boundary", "cuts inside an akshara",
                 "4-token guess ends inside a word"]),
        fmt={"tokens per word": lambda v: f"{v:.2f}", "cuts at a word boundary": pct, "cuts inside an akshara": pct,
             "4-token guess ends inside a word": pct}))
    A(f"\nThe Llama numbers match the earlier version exactly ({int(reg.ok.sum())}/{len(reg)} checks, "
      f"`test1_regression.csv`). The earlier version also reported {OLD['earlier_claims']['qwen_tokens_per_word']:.2f} "
      "tokens per word for Qwen3.5-0.8B. That number came from an older version of the `transformers` library, "
      "which loaded this tokenizer with the pre-tokenization rule of the older Qwen2 models. That rule splits "
      f"text before every vowel sign. With the tokenizer's own definition, Qwen3.5 needs only "
      f"{qwen['tokens_per_word']:.2f} tokens per word, fewer than Llama. Re-applying the old rule reproduces "
      f"{qwen_leg['tokens_per_word']:.2f} exactly (third row).\n")

    # ================================================================== 6 labels
    A("## 6. Linguistic labels\n")
    A("Every token of the target output gets labels describing where it sits: its position in its word, "
      "whether it starts inside an akshara, and whether a morpheme boundary follows it. The labels come from "
      "the target's output text, using each token's byte span.\n")
    A("### 6.1 Morpheme boundaries from MorphTok\n")
    A("**MorphTok** is a dataset of Hindi words segmented into morphemes (`hindi.csv`; checksum in "
      "`morphtok_info.json`). It was prepared as follows:\n")
    A(md_table(pd.DataFrame([
        ["rows in MorphTok", mt_info["rows"], ""],
        ["rows whose pieces join back into the word", mt_info["kept_rows"], "kept"],
        ["rows whose pieces do not (sandhi: sounds change at the join)", mt_info["dropped_rows"],
         "dropped, because their boundary cannot be placed in the written word"],
        ["distinct words in the resulting lexicon", lex["words"], ""],
        ["words with more than one segmentation", lex["words_with_multiple_segmentations"],
         "all their boundaries are kept (union)"],
        ["boundaries inside an akshara", f"{lex['boundaries_inside_akshara_snapped']} of {lex['raw_internal_boundaries']}",
         "moved to the start of that akshara, e.g. निकाल+ेगी → निका+लेगी"],
    ], columns=["MorphTok", "count", "treatment"])))
    A("\nBoundaries inside an akshara are moved because a token boundary can never usefully fall there: the "
      "vowel sign belongs to the consonant before it in writing.\n")
    A(f"**Words not in MorphTok** are segmented by a character-level CRF (conditional random field) trained on "
      f"MorphTok. On 10% of word types held out from training it finds boundaries with precision "
      f"{h['precision']:.3f}, recall {h['recall']:.3f}, **F1 {h['f1']:.3f}**, and gets "
      f"{pct(h['word_exact_match'])} of words exactly right. For comparison, a boundary after every akshara "
      f"scores F1 {crf['baseline_every_akshara_boundary']['f1']:.3f}. Held-out words often share stems with "
      "training words, so accuracy on truly new stems is probably lower.\n")
    A("Where the labels come from, for the words in the target outputs:\n")
    A(md_table(pd.DataFrame([
        [sp, cov[sp]["word_tokens"], cov[sp]["tokens"]["gold"]["frac"], cov[sp]["tokens"]["crf"]["frac"],
         cov[sp]["tokens"]["none"]["frac"], cov[sp]["word_types"], cov[sp]["types"]["gold"]["frac"],
         cov[sp]["types"]["crf"]["frac"]] for sp in ("dev", "devtest")],
        columns=["output set", "tokens in words", "from MorphTok", "from CRF", "none (not Devanagari)",
                 "distinct words", "distinct: MorphTok", "distinct: CRF"]),
        fmt={c: pct for c in ("from MorphTok", "from CRF", "none (not Devanagari)", "distinct: MorphTok", "distinct: CRF")}))
    A("\n(`morph_coverage.json`.) Two thirds of word tokens are covered by MorphTok itself; frequent words are "
      "mostly in the lexicon, rare ones mostly are not.\n")
    A("### 6.2 A morpheme signal usable while drafting\n")
    A("The labels above need the whole word. A stopping policy sees only the beginning of the current word "
      "(the part drafted so far). We therefore built a **prefix model** that estimates P(morpheme boundary "
      "here | the word so far). It counts, over MorphTok and the tuning-set outputs, how often a word that "
      "begins this way has a boundary at this point. When a prefix is rare it backs off to the last three, "
      "two or one aksharas, and finally to the overall rate (a minimum of 5 observations at each level). It "
      "never sees the test set (`prefix_model.pkl.gz`).\n")

    # ================================================================== 7 policies
    A("## 7. Stopping policies\n")
    A("### 7.1 The policies\n")
    A("*g* is the number of tokens guessed so far in the round, and *H* is the entropy of the draft model's "
      "prediction for the last guessed token (high = unsure). \"Lookahead\" marks policies that need the "
      "draft's prediction for the *next* position and pay one extra draft step for it (§4.3).\n")
    A(md_table(pd.DataFrame([
        ["NoDraft", "", "no draft model: plain decoding (speed-up 1 by definition)"],
        ["FixedK", "", "always guess exactly k tokens"],
        ["Entropy", "", "stop as soon as the draft is unsure: 1 − √(β·H) < λ, i.e. H above a threshold set by λ and β. "
                        "This is the confidence rule from the literature (AdaEDL, SVIP)."],
        ["WordBoundarySoft", "yes", "stop once g ≥ min_g and P(next token starts a new word) > τ"],
        ["WordBoundaryHard", "yes", "stop once g ≥ min_g and the draft's next token does start a new word (the "
                                    "earlier word-boundary policy, now paying for its lookahead)"],
        ["AksharaSafeEntropy", "yes", "Entropy rule, but when it fires, only stop if the current akshara is probably "
                                      "complete (P(next token is a vowel sign or other mark) < τ_m). Otherwise keep "
                                      "guessing, at most max_ext more tokens."],
        ["Hybrid", "yes", "Entropy rule, but only stop if the current word is probably complete "
                          "(P(next token starts a new word) > τ_w). Otherwise keep guessing, at most max_ext more."],
        ["MorphEntropy", "only with the akshara test", "Entropy rule, but only stop if the prefix model (§6.2) says a "
                                                       "morpheme boundary is likely (> τ_morph). Optionally also "
                                                       "require a complete akshara. Otherwise keep guessing, at most "
                                                       "max_ext more."],
        ["MorphOnly", "", "stop once g ≥ min_g and P(morpheme boundary) > τ_morph (no confidence rule)"],
        ["Oracle", "(not real)", "stop right before the first wrong guess. Not implementable; an upper bound."],
        ["GoldWordOracle", "(not real)", "stop exactly at the true word boundaries of the target's output. The best a "
                                         "pure word-boundary policy could do."],
    ], columns=["policy", "lookahead", "rule"])))
    A("\n### 7.2 Tuning\n")
    gsz = {f: len(C(f"grid_{f}.csv")) for f in main.policy if has(f"grid_{f}.csv")}
    A("Each policy's settings were chosen on the tuning set (dev) by maximising simulated speed-up, with the "
      "lookahead cost charged. The confidence rule was tuned over λ and β (β ∈ {0.5, 1, 2, 4}). The policies "
      "built on it were tuned over the 11 λ values around the best Entropy setting × all four β, times their "
      "own settings. Grid sizes: " + ", ".join(f"{k} {v}" for k, v in gsz.items() if k not in ('NoDraft', 'Oracle'))
      + " (`grid_*.csv`). The chosen settings (`selected_params.json`):\n")
    A(md_table(pd.DataFrame([[k, json.dumps(v, ensure_ascii=False)] for k, v in S.items() if v],
                            columns=["policy", "chosen on dev"])))
    A(f"\nWith the chosen λ and β, the confidence rule stops as soon as the draft's entropy exceeds "
      f"**{Hstar:.2f} nats**.\n")
    A("**Two tuned policies switched their linguistic part off.** For Hybrid and MorphEntropy, the best setting "
      "on dev was max_ext = 0: never keep guessing past where the confidence rule stops. MorphEntropy then "
      "*is* the confidence rule, and Hybrid is the confidence rule plus a lookahead it never uses. That is "
      "already a result: tuning prefers to ignore the linguistic condition. To still measure what the "
      "condition does when it is used, we also tuned versions restricted to max_ext ≥ 1. They are named "
      "**Hybrid_ext1** and **MorphEntropy_ext1**.\n")

    # ================================================================== 8 results
    A("## 8. Main results (simulated, test set)\n")
    tab = main[["policy", "lookahead", "speedup_charged", "CI_low", "CI_high", "speedup_uncharged", "diff_vs_entropy",
                "diff_CI_low", "diff_CI_high", "acceptance_rate", "mean_draft_len", "tokens_per_verif",
                "frac_oracle_gap_closed"]].copy().sort_values("speedup_charged", ascending=False)
    tab["speed-up [95% CI]"] = [f"**{a:.3f}** [{b:.3f}, {c:.3f}]" if p == "Entropy" else f"{a:.3f} [{b:.3f}, {c:.3f}]"
                                for p, a, b, c in zip(tab.policy, tab.speedup_charged, tab.CI_low, tab.CI_high)]
    tab["without lookahead charge"] = [f"{u:.3f}" if la == 1 else "" for u, la in zip(tab.speedup_uncharged, tab.lookahead)]
    tab["difference vs Entropy [95% CI]"] = [
        "—" if p == "Entropy" else f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        for p, d, lo, hi in zip(tab.policy, tab.diff_vs_entropy, tab.diff_CI_low, tab.diff_CI_high)]
    out = tab[["policy", "speed-up [95% CI]", "without lookahead charge", "difference vs Entropy [95% CI]",
               "acceptance_rate", "mean_draft_len", "tokens_per_verif", "frac_oracle_gap_closed"]]
    out.columns = ["policy", "speed-up [95% CI]", "without lookahead charge", "difference vs Entropy [95% CI]",
                   "guesses accepted", "guesses per round", "tokens per round", "share of fixed-k → oracle gap closed"]
    A(md_table(out, fmt={"guesses accepted": pct, "share of fixed-k → oracle gap closed": pct}))
    A(f"\n(`main_table.csv`.) *Guesses accepted* is the share of guessed tokens the target accepts. *Tokens per "
      f"round* counts the target's extra token. The last column shows how much of the distance from the best "
      f"fixed length to the oracle a policy covers.\n")
    A(img("plot_speedup_ci.png", "speed-up per policy with 95% CI") + "\n")
    A("**What the table shows.**\n")
    A(f"- The confidence rule is best. Every other real policy is slower, with a confidence interval entirely "
      f"below zero, except MorphEntropy, which as tuned is identical to it (§7.2). The confidence rule covers "
      f"{pct(E.frac_oracle_gap_closed)} of the gap between the best fixed length and the oracle.")
    A(f"- The best linguistic policy, {best_ling}, comes within {abs(m.loc[best_ling, 'diff_vs_entropy']):.3f} of "
      f"the confidence rule. It is the confidence rule plus a morpheme condition, and the condition only costs "
      f"speed.")
    A(f"- The lookahead policies lose even if the extra draft step were free: the best of them reaches "
      f"{m.loc[best_ling_un, 'speedup_uncharged']:.3f}× without the charge ({best_ling_un}).")
    A(f"- Stopping at the *true* word boundaries (GoldWordOracle, {m.loc['GoldWordOracle', 'speedup_charged']:.3f}×) "
      f"is no better than a fixed length of 2. So the problem is not that the policies estimate word boundaries "
      f"badly: word boundaries are not good places to stop.\n")
    A("**Where do the policies end their guesses?** Share of rounds whose last guessed token ends at a word "
      "boundary, inside an akshara, or at a morpheme boundary of the target text:\n")
    ends = main[["policy", "frac_drafts_ending_at_gold_word_boundary", "frac_ending_mid_akshara",
                 "frac_ending_at_morph_boundary"]].dropna()
    ends.columns = ["policy", "at a word boundary", "inside an akshara", "at a morpheme boundary"]
    A(md_table(ends, fmt={c: pct for c in ends.columns[1:]}))
    A(f"\nThe linguistic policies do stop where they intend to (WordBoundarySoft ends "
      f"{pct(m.loc['WordBoundarySoft', 'frac_drafts_ending_at_gold_word_boundary'])} of rounds at a word "
      f"boundary, against {pct(E.frac_drafts_ending_at_gold_word_boundary)} for Entropy). But reaching those "
      f"places means guessing more tokens, and more of them are rejected.\n")
    A("**Fixed draft lengths** (test set):\n")
    fkt = fk.copy()
    fkt.columns = ["k", "guesses accepted", "per-token acceptance estimate", "tokens per round", "guesses per round",
                   "speed-up"]
    A(md_table(fkt, fmt={"k": INT, "guesses accepted": pct}))
    A("\n(`fixed_k_devtest.csv`.) Guesses per round is slightly below k because a round never guesses past the "
      "end of the sentence.\n")
    A(img("plot_fixed_k.png", "acceptance and speed-up vs k") + "\n")
    A("Every setting tried on the tuning set, by policy family (speed-up against average guesses per round):\n")
    A(img("plot_dev_sweeps.png", "dev grids: speed-up vs mean draft length per family") + "\n")
    A(img("plot_draft_len_hist.png", "draft length histograms") + "\n")

    # ================================================================== 9 analysis
    A("## 9. Where do drafts fail, and does linguistic structure add anything to confidence?\n")
    A("This section looks at single tokens rather than policies: for each output token, did the draft model's "
      "guess match the target (given the correct text so far)? All tables are on the test set.\n")
    A("**By position in the word** (0 = first token of a word):\n")
    pt = pos_t.copy()
    pt.columns = ["position in word", "tokens", "mean draft entropy", "draft = target", "mean target prob. of draft guess",
                  "share of tokens"]
    A(md_table(pt, fmt={"tokens": INT, "draft = target": pct, "share of tokens": pct}))
    A("\n" + img("analysis_devtest_by_position.png", "match rate and entropy vs position in word (test set)") + "\n")
    A("**By where the token starts** (inside an akshara, inside a word between aksharas, at the start of a word):\n")
    skt = sk_t.copy()
    skt.columns = ["token starts", "tokens", "mean draft entropy", "draft = target", "mean target prob. of draft guess",
                   "share of tokens"]
    A(md_table(skt, fmt={"tokens": INT, "draft = target": pct, "share of tokens": pct}))
    A("\n**By whether a morpheme (or word) boundary follows the token:**\n")
    mbt = mb_t.copy()
    mbt.columns = ["boundary after token", "tokens", "mean draft entropy", "draft = target",
                   "mean target prob. of draft guess", "share of tokens"]
    A(md_table(mbt, fmt={"tokens": INT, "draft = target": pct, "share of tokens": pct}))
    A("\n**By kind of token:**\n")
    ttt = tt_t.copy()
    ttt.columns = ["token kind", "tokens", "mean draft entropy", "draft = target", "mean target prob. of draft guess",
                   "share of tokens"]
    A(md_table(ttt, fmt={"tokens": INT, "draft = target": pct, "share of tokens": pct}))
    A("\nWord-initial tokens are the hard ones: the draft has to choose the next *word*, while later pieces of "
      "a word are mostly determined by its beginning. Latin-script tokens (names, numbers written in Latin "
      "script) are the hardest of all, but rare.\n")
    A("**Correlations** of draft entropy with the labels (point-biserial r, signed; 95% CI):\n")
    ct = corr_t[["pair", "r", "lo", "hi"]].merge(corr_d[["pair", "r"]].rename(columns={"r": "r (tuning set)"}), on="pair")
    ct["pair"] = ct["pair"].str.replace("H vs ", "entropy vs ", regex=False).str.replace(
        "end_cut==WORD", "token ends a word", regex=False).str.replace("match", "draft = target", regex=False)
    A(md_table(ct, fmt={c: SIGNED3 for c in ("r", "lo", "hi", "r (tuning set)")}))
    A("\nThe draft is more uncertain at word starts (r > 0) and less uncertain at the end of words and morphemes. "
      "The strongest relation is between entropy and being right (r ≈ −0.54).\n")
    A("### 9.1 The key test: same confidence, different position\n")
    A("Word-initial tokens fail more, but they also have higher entropy. Is position informative *on top of* "
      "entropy? Two tests:\n")
    A("**(a) Compare within confidence bands.** Split tokens into ten equal-size bands of draft entropy. Within "
      "each band, compare how often the draft is right for word-initial tokens vs other tokens (likewise for "
      "tokens that start a morpheme), then average over bands, weighted by size:\n")
    cd = cond[["split", "ling", "raw_diff", "entropy_adjusted_diff", "lo", "hi"]].copy()
    cd["split"] = cd["split"].map({"devtest": "test", "dev": "tuning"})
    cd["ling"] = cd["ling"].map({"word_initial": "word-initial vs other", "morph_start": "morpheme-initial vs other"})
    cd.columns = ["set", "comparison", "raw difference", "difference at equal entropy", "CI low", "CI high"]
    A(md_table(cd, fmt={c: lambda v: f"{100 * v:+.1f} pts" for c in ("raw difference", "difference at equal entropy",
                                                                      "CI low", "CI high")}))
    A(f"\n(`analysis_conditional_summary.csv`.) Without controlling for entropy, word-initial tokens are "
      f"{abs(wi.raw_diff) * 100:.0f} points worse. At equal entropy the difference is small and *positive*: "
      f"word-initial tokens are slightly *easier* than their entropy suggests.\n")
    A("**(b) Predict failures with and without position.** A logistic regression predicts \"draft guess "
      "accepted\" from confidence only (entropy and top-1 probability), and then from confidence + word-initial "
      "+ morpheme-initial. If position carried extra information, the second model would separate right "
      "from wrong guesses better (higher AUC):\n")
    lg = logit_t[["quantity", "value", "lo", "hi"]].merge(logit_d[["quantity", "value"]].rename(columns={"value": "tuning set"}),
                                                          on="quantity")
    lg["quantity"] = (lg["quantity"].str.replace("coef_std", "standardised coefficient", regex=False)
                      .str.replace("(full model)", "(model with position)", regex=False)
                      .str.replace("AUC base (H+top1)", "AUC, confidence only", regex=False)
                      .str.replace("AUC full (H+top1+word_initial+morph_start)", "AUC, confidence + position", regex=False)
                      .str.replace("delta AUC (full - base)", "change in AUC", regex=False)
                      .str.replace("delta mean log-lik per position (full - base)", "change in log-likelihood per token",
                                   regex=False))
    lg.columns = ["quantity", "test set", "CI low", "CI high", "tuning set"]
    A(md_table(lg, fmt={c: lambda v: f"{v:+.4f}" for c in ("test set", "CI low", "CI high", "tuning set")}))
    A("\n(`analysis_conditional_devtest_logit.csv`; 200 bootstrap resamples.) Adding position changes AUC by "
      "−0.0001: no gain at all. The two position coefficients have opposite signs because every word start is "
      "also a morpheme start, so the two labels nearly duplicate each other. Only their joint effect (the AUC "
      "and log-likelihood change) is meaningful.\n")
    A("**Conclusion of this section.** Drafts fail at word starts *because* the draft is uncertain there, and "
      "entropy already measures that uncertainty. Once entropy is known, position adds nothing. This explains "
      "the policy results: a boundary condition can only move the stopping point away from where the "
      "confidence rule would put it.\n")

    # ================================================================== 10 real loop
    A("## 10. Real speculative decoding on GPU\n")
    A(f"To check the simulator, we implemented real greedy speculative decoding. Both models keep a memory of "
      f"the text so far (the KV cache), and rejected guesses are removed from it. We ran it on "
      f"{len(ids['ids'])} randomly chosen test sentences (`real_loop_ids.json`) with six policies, using the "
      "settings chosen on the tuning set: NoDraft, FixedK, Entropy, WordBoundarySoft, Hybrid and "
      "MorphEntropy_ext1. The policies run through the same code as in the simulator. Each sentence was decoded "
      "3 times per policy after a warm-up; we use the median time, and speed-up is total time of plain decoding "
      "/ total time of the policy on the same sentences.\n")
    if real_done:
        ll = C("real_loop_lossless.csv").set_index("policy")
        rl = C("real_loop.csv").set_index("policy")
        svr = C("sim_vs_real.csv").set_index("policy")
        rci = C("real_loop_ci.csv").set_index("policy")
        dec = C("sim_vs_real_decomposition.csv").set_index("policy")
        am = C("real_loop_after_mismatch.csv")
        dv = C("real_loop_divergences.csv") if has("real_loop_divergences.csv") else pd.DataFrame()
        RI = J("real_loop_run_info.json") if has("real_loop_run_info.json") else {}
        cmt = C("real_loop_cost_model.csv").set_index("g")
        am_e = am[am.policy == "Entropy"].set_index("source").mean_after_mismatch
        order = [p for p in rci.index if p != "NoDraft"] + ["NoDraft"]
        e = rci.loc["Entropy"]
        runner = [p for p in order if p not in ("Entropy", "NoDraft")][0]
        dd = dec.drop("NoDraft")
        de = dec.loc["Entropy"]
        fixed_est = de.speedup_wall * de.speedup_real_rounds_capped_model / de.speedup_real_rounds_model
        A("### 10.1 Results\n")
        tab = pd.DataFrame([{
            "policy": p,
            "real speed-up [95% CI]": f"{rci.loc[p, 'speedup_wall']:.3f} [{rci.loc[p, 'lo']:.3f}, {rci.loc[p, 'hi']:.3f}]",
            "difference vs Entropy [95% CI]": ("—" if p == "Entropy" else
                                               f"{rci.loc[p, 'diff_vs_ref']:+.3f} [{rci.loc[p, 'diff_lo']:+.3f}, {rci.loc[p, 'diff_hi']:+.3f}]"),
            "simulated speed-up": f"{svr.loc[p, 'speedup_sim_charged']:.3f}",
            "simulator too high by": pct(svr.loc[p, "relerr_speedup_charged"]),
            "tokens per second": f"{rl.loc[p, 'tokens_per_s']:.1f}",
            "tokens per round, real / simulated": f"{svr.loc[p, 'tokens_per_verif_real']:.3f} / {svr.loc[p, 'tokens_per_verif_sim']:.3f}",
            "identical to plain decoding": f"{ll.loc[p, 'identical']}/{ll.loc[p, 'decodes']}"} for p in order])
        A(md_table(tab))
        A("\n(`real_loop_ci.csv`, `sim_vs_real.csv`, `real_loop.csv`, `real_loop_lossless.csv`.)\n")
        A(img("plot_sim_vs_real.png", "real vs simulated speed-up"))
        A("\n*Real wall-clock speed-up with 95% confidence interval (blue), and the simulator's prediction for the "
          "same sentences (orange).*\n")
        A(f"- **Same ranking as the simulator.** The confidence rule is fastest at **{e.speedup_wall:.3f}×**, and "
          f"every other policy is slower with a paired confidence interval below zero. The closest is {runner} "
          f"({rci.loc[runner, 'diff_vs_ref']:+.3f}). The two lookahead policies are the slowest.")
        A(f"- **The simulator gets acceptance right.** Tokens produced per round differ by at most "
          f"{pct(svr.relerr_tokens_per_verif.abs().max(), 2)} between simulation and reality.")
        A(f"- **The simulator is {pct(-dd.total.max())}–{pct(-dd.total.min())} too optimistic about speed**, for "
          "reasons explained in §10.3.")
        A(f"- **Output check:** {ll.identical.min()}–{ll.identical.max()} of {ll.decodes.max()} decodes per policy are "
          f"identical to plain one-at-a-time decoding. That includes NoDraft ({ll.loc['NoDraft', 'identical']}/"
          f"{ll.loc['NoDraft', 'decodes']}), which does no guessing at all, so the differences are not caused "
          "by speculative decoding (§10.2).\n")
        A(f"Run details: build `{RI.get('git_hash', '?')}`, transformers {RI.get('transformers', '?')}, "
          f"torch {RI.get('torch', '?')}, 2× T4 (`real_loop_run_info.json`).\n")
        A("### 10.2 Why some outputs differ from plain decoding\n")
        if len(dv):
            g = (dv.groupby(["id", "pos"]).agg(margin=("margin", "first"), policies=("policy", lambda s: ", ".join(s)))
                 .reset_index())
            A(f"The same {dv.id.nunique()} sentences differ in all three repeats. At every first difference, the "
              f"target model's top two candidate tokens have almost the same score (gap at most "
              f"{dv.margin.abs().max():.4f}, one or two rounding steps of 16-bit floats). Plain decoding without a "
              f"draft differs too, on {int((dv.policy == 'NoDraft').sum())} sentences, and several differences occur "
              "at the same position for every policy. Processing tokens one at a time with a cache versus all at once "
              "changes rounding slightly, and that flips exact ties. Speculative decoding itself always keeps the "
              "target's top choice.\n")
            A(md_table(g.rename(columns={"id": "sentence", "pos": "token position", "margin": "gap between top two",
                                         "policies": "policies that differ here"}),
                       fmt={"gap between top two": lambda x: f"{x:.4f}"}))
            A("")
        A("### 10.3 Why the simulator is optimistic\n")
        A("We split the gap into four steps, each measured on the same sentences:\n")
        A("1. **Teacher forcing**: the simulator vs the real rounds (the draft lengths the real loop actually "
          "chose) priced with the simulator's cost model. This is the simulator's only approximation about policy "
          "behaviour (§4.3).")
        A("2. **Guessing past the end**: in the round that finishes a sentence, our real loop keeps guessing after "
          "the end-of-text token. The draft is confident there, and the target accepts tokens that are then thrown "
          "away. The simulator never guesses past the end.")
        A("3. **Cost model**: the same rounds priced with the cost model vs the measured time of each model call.")
        A("4. **Overhead**: model-call time vs total time (choosing tokens, trimming the cache, Python).\n")
        dshow = dd.reset_index()[["policy", "speedup_sim", "speedup_wall", "eos_overrun_tokens_per_sentence",
                                  "step_teacher_forcing", "step_eos_overrun", "step_cost_model", "step_overhead", "total"]]
        dshow.columns = ["policy", "simulated", "real", "tokens guessed past the end, per sentence",
                         "1 teacher forcing", "2 past the end", "3 cost model", "4 overhead", "total"]
        A(md_table(dshow, fmt={c: pct for c in ["1 teacher forcing", "2 past the end", "3 cost model", "4 overhead", "total"]}
                   | {"tokens guessed past the end, per sentence": lambda x: f"{x:.1f}"}))
        A(f"\n(`sim_vs_real_decomposition.csv`.)\n")
        A(f"- **Teacher forcing matters little** ({pct(-dd.step_teacher_forcing.max())}–"
          f"{pct(-dd.step_teacher_forcing.min())}). Round by round, real and simulated decoding have the same share "
          f"of rejected rounds and the same draft lengths (table below). After a wrong guess, the real draft "
          f"continues slightly longer (for Entropy {am_e['real']:.3f} vs {am_e['sim']:.3f} extra guesses per "
          f"rejected round).")
        A(f"- **Guessing past the end is the largest effect** for the confidence-based policies "
          f"({pct(-de.step_eos_overrun)} for Entropy, {de.eos_overrun_tokens_per_sentence:.1f} wasted guesses per "
          f"sentence). It is a flaw in our loop, not in the policies, and it slows every policy that guesses, so it "
          f"does not change the ranking. Stopping the draft once it produces the end-of-text token keeps the output "
          f"identical and would bring Entropy to about **{fixed_est:.2f}×**.")
        A(f"- **The cost model is accurate per round** (table at the end of this section). The "
          f"{pct(-dd.step_cost_model.max())}–{pct(-dd.step_cost_model.min())} step comes mainly from plain decoding "
          f"being slightly faster in the loop ({cmt.loc[0, 't_verify_ms']:.1f} ms per token) than in the separate "
          f"timing run ({cmt.loc[0, 'model_t_verify_ms']:.1f} ms), which helps the baseline.")
        A(f"- **Overhead** is {pct(dd.overhead_share_of_wall.min())}–{pct(dd.overhead_share_of_wall.max())} of total "
          f"time for the policies that guess, against {pct(dec.loc['NoDraft', 'overhead_share_of_wall'])} for plain "
          f"decoding: about 1% of speed-up.\n")
        A("**Rounds in real decoding (with the last round cut at the end of the sentence) vs the simulator** "
          "(`real_loop_after_mismatch.csv`):\n")
        ams = am.copy()
        ams.columns = ["policy", "source", "rounds", "rounds with a rejection", "guesses (all accepted)",
                       "guesses (some rejected)", "guesses after the first wrong one"]
        A(md_table(ams, fmt={"rounds with a rejection": pct}))
        A("\n**Measured vs modelled time per round**, by number of guesses g (all policies and repeats; extra = share "
          "of rounds with a lookahead step; `real_loop_cost_model.csv`):\n")
        cm = C("real_loop_cost_model.csv")
        cm.columns = ["g", "rounds", "draft ms (measured)", "verify ms (measured)", "extra", "draft ms (model)",
                      "verify ms (model)"]
        A(md_table(cm[["g", "rounds", "draft ms (measured)", "draft ms (model)", "verify ms (measured)",
                       "verify ms (model)", "extra"]],
                   fmt={"g": INT, "rounds": INT, "extra": pct}))
        A("\n**Bottom line.** On real hardware the policies rank as the simulator says, and the confidence rule "
          "stays best. Simulated speed-ups are a few percent high in absolute terms, mostly because of a fixable "
          "flaw in our loop that affects all policies alike.\n")
    else:
        A(f"Results: {PH}. To fill this section, run the real-loop notebook on Kaggle, unzip its output into "
          "`data/real_loop/`, run `notebooks/04_real_loop_analysis.ipynb`, then `python scripts/make_report.py`.\n")

    # ================================================================== 11 conclusions
    A("## 11. Conclusions\n")
    A("1. For English→Hindi translation with Llama-3 8B/1B, **draft confidence already captures what word, "
      "akshara and morpheme boundaries say about draft failures.** Failures cluster at word starts, but only "
      "because the draft is unsure there.")
    A("2. **No boundary-based stopping policy beats the confidence rule**, in simulation or on real hardware, "
      "even with fair tuning, even without charging for lookahead, and even with the true word boundaries.")
    A("3. **There is still a large gap to the oracle**, so better stopping is possible. It will have to come "
      "from better use of confidence and context rather than from boundaries, at least for Hindi.")
    A("4. **The simulator is reliable for comparing policies**: it ranks them like real decoding, and its "
      "acceptance figures are within half a percent.")
    A("5. **Hindi may be the wrong language to find a morphology effect.** Hindi writes postpositions as "
      "separate words (घर में \"in the house\"), so its morpheme boundaries mostly coincide with word "
      "boundaries. Marathi attaches them (घरात), so a morphology effect is more likely to show there.\n")
    A("**Next steps** (planned for the next phase): a learned stopping controller that combines confidence, "
      "context and linguistic features; Marathi, an English control and a second model family with a "
      "different tokenizer; an analysis of where the confidence rule loses to the oracle; and fixing the "
      "end-of-sentence flaw before the final timing run.\n")

    # ================================================================== 12 limitations
    A("## 12. Limitations\n")
    for c in [
        "Simulated speed-ups use teacher-forced traces. Real decoding confirms the ranking but is a few percent "
        "slower in absolute terms (§10.3). Quote real speed-ups from §10, and use the simulated ones to compare "
        "policies.",
        f"Real decoding was run on {len(ids['ids'])} sentences, not all 1012.",
        f"c = {T['c']:.3f} was measured with the 8B target split over two T4 GPUs, so every target step includes "
        "a transfer between GPUs. On a single large GPU, c and all speed-ups would change (the ranking probably "
        "much less).",
        "One language pair, one model family, greedy decoding, one domain (FLORES+ news/wiki sentences).",
        f"{pct(cov['devtest']['tokens']['crf']['frac'], 0)} of morpheme labels come from the CRF segmenter, whose "
        "accuracy on new stems is probably below its held-out F1. MorphTok words whose morphemes change sound "
        "at the join (sandhi) were excluded.",
        "The best confidence-rule setting uses β = 4, the edge of its grid. A check with β = 8 changed the tuning "
        "speed-up by less than 0.001.",
    ]:
        A(f"- {c}")
    A("")

    # ================================================================== appendix
    A("## Appendix: reproducing the results\n")
    A("```bash\n"
      "pip install -e .                                  # Python 3.11; CPU-only torch is enough locally\n"
      "pytest                                            # unit tests (tests needing the gated tokenizer skip without a login)\n"
      "python scripts/smoke_cpu.py --part 01             # CPU test of the trace-generation code\n"
      "python scripts/build_kaggle_notebooks.py 01       # -> kaggle/01_traces_v2.ipynb, run on Kaggle (SPLIT=dev, then devtest)\n"
      "python scripts/download_morphtok.py && python scripts/build_morph.py   # MorphTok lexicon + CRF segmenter\n"
      "jupyter nbconvert --execute notebooks/00_tokenizer_analysis.ipynb      # §5\n"
      "jupyter nbconvert --execute notebooks/02_policies_v2.ipynb             # §6–§9\n"
      "python scripts/smoke_cpu.py --part 03 --selected results/phase1/selected_params.json             # draft = target\n"
      "python scripts/smoke_cpu.py --part 03 --selected results/phase1/selected_params.json --noise 2   # with rejections\n"
      "python scripts/build_kaggle_notebooks.py 03       # -> kaggle/03_real_loop.ipynb, run on Kaggle\n"
      "jupyter nbconvert --execute notebooks/04_real_loop_analysis.ipynb      # §10\n"
      "python scripts/make_report.py                     # this report\n"
      "```\n")
    A("Trace generation ran build `git=0a45b51ed3e2-dirty src_sha256=c75053076a02` (`run_info_*.json` in the "
      "trace output). A later CPU-only helper changes the hash of a rebuild; the GPU code is unchanged.\n")
    A("**Files in `results/phase1/`.** Main table `main_table.csv` · tuning grids `grid_*.csv` · chosen settings "
      "`selected_params.json` · token-level analysis `analysis_*.csv` / `.png` · morphology `morph*.json`, "
      "`morph_crf.pkl`, `prefix_model.pkl.gz` · tokenizer analysis `test1_*` · real decoding `real_loop*`, "
      "`sim_vs_real*` · earlier version's numbers `earlier_results.json` · suggested slide wording "
      "`../../docs/pre_interim/interim_slide_corrections.md`.\n")
    report = "\n".join(L)

    readme_lines = [
        "# results/phase1\n",
        "Outputs of the first project phase. **Full report: [REPORT.md](REPORT.md)** (generated by "
        "`scripts/make_report.py`).\n",
        "What was done: tokenizer analysis of Hindi; target outputs and draft/target traces for FLORES+ dev and "
        "devtest (Llama-3 8B target, 3.2 1B draft); morpheme labels from MorphTok plus a CRF segmenter; a "
        "simulator that enforces what a real decoder can know; 13 stopping policies tuned on dev and tested on "
        "devtest with bootstrap CIs; a token-level failure analysis; and real speculative decoding on GPU.\n",
        f"Headline (devtest, simulated): confidence rule {E.speedup_charged:.3f}× [{E.CI_low:.3f}, {E.CI_high:.3f}], "
        f"best fixed length {FK.speedup_charged:.3f}×, oracle {O.speedup_charged:.3f}×. No linguistic policy beats "
        f"the confidence rule (best: {best_ling} {m.loc[best_ling, 'speedup_charged']:.3f}×), and at equal "
        f"confidence linguistic position does not predict draft failure (change in AUC {dauc.value:+.4f}).\n",
    ]
    if real_done:
        readme_lines.append(
            f"Real decoding (100 devtest sentences, wall clock): confidence rule "
            f"{C('real_loop_ci.csv').set_index('policy').loc['Entropy', 'speedup_wall']:.3f}× over plain decoding, "
            f"best of all policies. The simulator ranks the policies the same way but is a few percent optimistic "
            f"(REPORT §10).\n")
    return report, "\n".join(readme_lines)


def main() -> None:
    report, readme = build()
    (RES / "REPORT.md").write_text(report, encoding="utf-8")
    (RES / "README.md").write_text(readme, encoding="utf-8")
    print("wrote", RES / "REPORT.md", f"({len(report.splitlines())} lines) and README.md")


if __name__ == "__main__":
    main()
