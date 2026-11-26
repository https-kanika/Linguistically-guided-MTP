# Real GPU loop with word-completion heads

100 devtest sentences, 3 repeats, heads: 180k pairs. Wall-clock, NoDraft = 1. `sim` is the simulated speed-up on all 1,012 devtest sentences (simulation has no prefill, so compare it with `decode only`).

| method | wall-clock [95% CI] | decode only | Δ vs Entropy (wall) [95% CI] | sim | lossless | tokens/round | 1B calls/round | proposed tokens/round | proposed tokens accepted |
|---|---|---|---|---|---|---|---|---|---|
| NoDraft | 1.000 [1.000, 1.000] | 1.000 |  |  | 0.96 | 1.000 | 0.000 | 0.000 |  |
| Entropy | 1.384 [1.359, 1.408] | 1.395 |  | 1.452 | 0.96 | 2.445 | 2.081 | 0.000 |  |
| V1+lex | 1.469 [1.433, 1.504] | 1.483 | +0.085 [+0.064, +0.105] | 1.542 | 0.96 | 2.629 | 2.100 | 0.617 | 0.639 |
| V1+check | 1.505 [1.467, 1.539] | 1.520 | +0.121 [+0.106, +0.135] | 1.572 | 0.96 | 2.548 | 1.844 | 2.287 | 0.220 |
| NGram | 1.620 [1.580, 1.657] | 1.640 | +0.236 [+0.218, +0.254] | 1.701 | 0.96 | 2.537 | 1.478 | 8.848 | 0.100 |
| NGram+V0 | 1.620 [1.581, 1.657] | 1.640 | +0.236 [+0.218, +0.254] | 1.708 | 0.96 | 2.537 | 1.449 | 7.771 | 0.119 |

`lossless`: share of decodes identical to the 8B's unbatched greedy output. NoDraft is also below 1 (fp16 near-ties), so compare with NoDraft, not with 1. Phase 1: 0.95–0.96 for every method.

`proposed tokens`: for V0/V1/V2 (and +lex) the heads' tokens, accepted by the 8B; for the checked methods (V*+check, NGram*) tokens read in a 1B call, accepted by the 1B.
