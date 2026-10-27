# Corrected surface-cost formula — Rion experiments

Updated October 26, 2026. All official scores are pending.

## Formula

    cost = round_to_cents(100 + surface_rate * (pi * radius**2)**0.85 + 200 * unreplaced_length)

Only area is raised to 0.85. Previous code used `(surface_rate * area)**0.85`. Predictive models, probabilities, repair-cost predictions and other project rules are unchanged. Originals are preserved under source/. These files have not been uploaded to GitHub or the evaluator. Old public scores must not be assigned to rebuilt candidates.

## Rebuilt candidates

- 09a_corrected_reference.csv: supporting reference; rerank original circles using corrected initial net benefit / cost.
- 10a_corrected_nearby_pairs.csv: rerun grouping with the corrected reference order and pair costs.
- 11b_corrected_net_benefit_only.csv: original B-only ordering; no cost denominator, so CSV hash is unchanged. Budget validation uses the corrected formula.
- 11c_corrected_net_benefit_sqrt_cost.csv: rerank B / sqrt(max(C, 1)) with corrected initial costs.
- 12a_corrected_dynamic.csv: original circle geometry, dynamic overlap-updating priority with corrected current costs.
- 13a_corrected_grouped_dynamic.csv: newly grouped 10a geometry plus corrected dynamic ordering.

B sums p(leak 2027) × (predicted repair cost − 200 × pipe length) over relevant unreplaced contained pipes. Dynamic methods recompute B and cost after each planned replacement. This happens before submission and produces a fixed CSV, not a policy reacting to actual future leaks.

## Validation — no future leaks

| Candidate | Units | Construction cost | Finish | Uncovered iron |
| --- | ---: | ---: | ---: | ---: |
| 09a_corrected_reference.csv | 38,455 | $196,422,709.41 | 2046 | 0 |
| 11b_corrected_net_benefit_only.csv | 38,455 | $195,914,746.73 | 2046 | 0 |
| 11c_corrected_net_benefit_sqrt_cost.csv | 38,455 | $195,690,242.78 | 2046 | 0 |
| 10a_corrected_nearby_pairs.csv | 37,789 | $194,881,031.96 | 2046 | 0 |
| 12a_corrected_dynamic.csv | 38,455 | $194,098,428.46 | 2046 | 0 |
| 13a_corrected_grouped_dynamic.csv | 37,789 | $192,591,377.43 | 2046 | 0 |

All six candidates geometrically cover every known original iron pipe (38,455), leave zero original iron unreplaced, and finish by 2051 under the no-future-leak simulation. This is not an official score forecast or proof for every future leak path.

Costs count distinct surfaces from all contained pipes, including already-polyurethane pipes. Linear charges use only currently non-polyurethane pipe length, including copper/brass and conservative unknown-material treatment. Polyurethane-only units are skipped for free. Budget checks use half-up cent rounding, integer cents, $10M yearly allocation, carryover and stopping at the first unaffordable unit. Exported CSVs are reparsed for validation.

## Grouping and independent geometry check

Grouping retains the previous settings: pairs at most 100 positions apart, spatially close, at most 12 pipes in their original union, sequential pair savings above $100, and greedy disjoint pairing. Enclosing circles cover all endpoints of all pipes contained in the original pair. Every merged circle was independently checked against all pipe endpoints without the spatial grid; untouched circle geometry was preserved.

The corrected search merged 666 pairs, rather than the previous 656. New geometry and ranks differ because costs and the static reference order changed. 13a uses exactly the newly rebuilt 10a circles.

## Original unchanged CSVs evaluated with the corrected formula

| Original | No-future-leak construction cost | Finish |
| --- | ---: | ---: |
| 09a | $196,850,854.96 | 2046 |
| 10a | $195,702,089.40 | 2046 |
| 12a | $194,221,444.28 | 2046 |
| 13a | $193,162,059.49 | 2046 |

Lower construction cost does not guarantee a better official score because timing also changes repair costs. Compare original and rebuilt CSVs under the same evaluator version.

## Data provenance and reproduction

Source repository: https://github.com/reinesana/STAT440-star-kingdom-project

Inputs are preserved team snapshots under source/: pipes.csv, train.csv, original 09a CSV and audit, ensemble_stable repair-cost forecasts, full-inventory 2027 probabilities, and original Rion submissions. Original 09a SHA256 is verified. Missing iron probabilities retain audit fallback values. Relevant missing-input benefit contributions follow 12a's audit-constraint reconstruction; participating unknowns are checked for identifiability. Newly merged circles cannot introduce previously unconstrained missing-input pipes. Original aggregate benefits match the audit to numerical tolerance. Models are not retrained.

Requires numpy and pandas:

    python rebuild_corrected.py

Outputs include comparison.csv, per-candidate validation JSON with annual balances and hashes, grouping/order audit CSVs, and rebuild_checks.json with source hashes.

First suggested submissions: 13a_corrected_grouped_dynamic.csv and 12a_corrected_dynamic.csv. Keep original 13a as a reference and submit it under the same current evaluator if needed. Record the evaluation time and formula version.
