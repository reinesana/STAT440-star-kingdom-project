# Executed results — October 28, 2026

Independent implementation and local run completed. This is a model-based
experiment, not an official contest evaluation. Source data: 43,039 pipes,
14,113 unique leak events (2019–2026), 38,455 original iron pipes, and 26,237
initially unreplaced iron pipes. There are 8 missing materials and 15 missing
installation dates. Input and output SHA-256 hashes are in `outputs/report.json`.

## Deterministic construction, no future leaks

| Strategy | CSV circles | Cost | Finish year | Uncovered active iron |
|---|---:|---:|---:|---:|
| Single-pipe | 26,237 | $190,155,279.56 | 2046 | 0 |
| Pairwise | 20,537 | $181,563,237.62 | 2045 | 0 |
| Multi-pipe | 15,328 | $175,986,889.96 | 2044 | 0 |

Multi-pipe saves **$5,576,347.66 (3.07%)** versus pairwise and uses 5,209 fewer
circles. It selects 728 triples, 551 groups of four, 494 groups of five, and 854
groups of six: **2,627 groups of size 3+**. These are designed groups; incidental
containment and prior replacements can change the number replaced at execution.
All arms meet the supplied scripts' ordered annual budget and retirement rules.
They cover all active iron. They do not geometrically cover every historically
replaced iron pipe; see the report's separate original-iron coverage counts.

## Common-random-number simulation

30 trials per scenario, 3 scenarios, 3 strategies: **270 strategy-trial evaluations**.
Every evaluated strategy finished within the 2051 horizon. Scores below are
250 million minus modeled construction and repair costs; units are millions.

| Hazard multiplier | Pairwise mean score | Multi-pipe mean score | Paired multi-pipe gain | Approx. 95% Monte Carlo interval for gain |
|---|---:|---:|---:|---:|
| 0.5 | 41.233 | 44.679 | +3.446 | [+2.664, +4.228] |
| 1.0 | 33.872 | 35.345 | +1.473 | [+0.685, +2.261] |
| 2.0 | 27.462 | 25.745 | -1.717 | [-2.788, -0.647] |

At fitted hazards, multi-pipe's mean construction cost is $138.986M and repair
cost is $75.669M, versus pairwise's $140.338M and $75.790M. At doubled hazards,
the single-pipe arm has the best mean score ($29.227M), illustrating that more
grouping is not uniformly better. All scenarios use the same seed and shared
draws across strategies. The intervals reflect Monte Carlo noise conditional on
the fitted model, not predictive uncertainty or a guarantee about hidden scores.

## Risk checks and interpretation

| Chronological year | Observed annual leak rate | Predicted rate | Log loss | Brier score |
|---|---:|---:|---:|---:|
| 2025 | 5.530% | 4.292% | 0.19187 | 0.04930 |
| 2026 | 10.031% | 6.154% | 0.26437 | 0.08057 |

The model underpredicts both years, especially 2026. The fitted-hazard advantage
therefore is insufficient evidence of a robust improvement in the contest.
The multi-pipe submission is an experimental candidate with verified construction
savings, accompanied by a pairwise alternative and explicit stress results.
No official score or global optimum is claimed.

## Validation and reproducibility

Executed `python optimize.py --sims 30` successfully; measured run time was
104.13 seconds in the bundled Python environment. Six unit tests passed, including
40 random enclosing-disk comparisons against exhaustive boundary enumeration.
All submissions were re-read after serialization and checked for complete active
iron coverage and sequential cost/budget feasibility. A further 300 exported
circles (100 per arm) passed full-data brute-force containment checks independent
of the spatial index. `python verify_outputs.py` reproduces this spot audit.

The available formula correction agrees with the implemented area exponent.
The complete official assignment was absent, so additional contest constraints
and exact timing semantics remain unverified. See README for all working rules,
the distinctions from the supplied simulator, and reproduction instructions.

For GitHub: code, tests, this summary, documentation, original reference scripts,
and outputs are bundled in `stat440_github_handoff.zip`. Raw input data is excluded
from that archive; its original hashes and expected file placement are recorded.
Nothing has been published to GitHub.
