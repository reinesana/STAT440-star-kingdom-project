# Dynamic priority ordering from 09a — Rion

Submission: `12a_dynamic_net_benefit_per_cost.csv`. Official public score: **+$22.50M**, submitted by Rion on October 20, 2026 at 14:44 (as shown in the evaluator screenshot).

Keep exactly the same 38,455 circle records as 09a. Starting from historical replacement status, repeatedly choose the remaining circle with the highest current net benefit divided by current unrounded construction cost. Net benefit is the sum of p(leak in 2027) × (predicted repair cost − $200 × pipe length) over still-unreplaced contained pipes. After choosing a circle, mark its contained pipes replaced in the planning simulation and update all overlapping circles. Ties use Pipe ID. Polyurethane-only circles are placed immediately at no cost. No models are retrained; fixed 2027 predictions are reused.

The result is a fixed CSV, not a policy that can respond to actual evaluation leaks. Annual timing is validated after ordering; the selection heuristic does not optimize calendar years or account for unknown future leaks.

Predictions come from the same published leak and repair-cost ensembles used by 09a. Missing iron probabilities use the 09a audit fallback. Missing-input contributions for eight unknown-material pipes are recovered from 09a's published aggregate-benefit constraints where relevant; contributions outside every candidate circle are irrelevant. Identifiability for participating missing pipes is asserted. Reconstructed initial circle benefits match the 09a audit within $0.00001; the observed maximum error is approximately 2.91e-11. No missing probability is silently treated as a confirmed zero-risk forecast.

Validation reparses the exported CSV, checks the identical circle multiset, full original-iron endpoint coverage, and annual carryover budget through 2051. It includes all contained surface types, current non-polyurethane length, overlapping replacements, zero-cost skips and per-unit half-up cent rounding. It uses integer cents and stops at the first unaffordable unit. Reference 09a construction cost is reproduced exactly.

No-future-leak results:
- Uncovered original iron: 0
- Remaining original iron after execution: 0
- Construction cost: $151,529,540.50
- Reference 09a construction cost: $152,944,811.51
- Completion year: 2042, within the construction budget

These are construction checks, not official score forecasts. Compare evaluator results with reference 09a (-$6.79M) and the prior grouped-circle experiment (-$5.98M). Public gains do not guarantee private gains.

Run `python build_dynamic.py` with numpy and pandas installed. All inputs are under source/. Rion used Codex for implementation and validation.

## Public evaluation result

| Candidate | Displayed public score | Change versus 12a |
| --- | ---: | ---: |
| Reference 09a | -$6.79M | 12a improves by approximately $29.29M |
| Previous grouped-circle candidate 10a | -$5.98M | 12a improves by approximately $28.48M |
| 12a dynamic ordering | **+$22.50M** | — |

Differences use rounded displayed scores. The result measures the full public evaluation, including construction and repair costs. Their separate contributions cannot be inferred from the displayed score alone. Private performance is unknown.

## GitHub location

Place this entire folder under `submissions/2026-10-20_rion_dynamic/`. The submission file itself is `12a_dynamic_net_benefit_per_cost.csv`. Keep source/ alongside the script to reproduce the result. Rion used Codex to implement and validate the experiment using the team's existing data and predictions.
