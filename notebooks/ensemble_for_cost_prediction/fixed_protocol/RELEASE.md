# 2026-10-30: selection-process reset

The user requested removing both public-score-driven adjustment and repeated
historical-validation-driven candidate selection. The old F168 report describes
comparison of 591 prediction recipes and 20 official work-order submissions.
That selection history remains attached to the old artifacts; it is not relabeled
as a clean test, and it does not by itself prove that future labels entered training.

This release chooses CMS002 and HFX001 by the new process policy, independently
of the numerical backtest results. It is a change in procedure and complexity,
not evidence of predictive improvement. It does not upload any new submission to
the official evaluator and does not optimize work-order geometry or priorities.

## Frozen before this run's metrics

- Specification/code freeze: `6a2da48`, committed before model fitting.
- This commit was pushed to `takito/fixed-protocol-2026-10-30` before metrics were computed.
- Exact source/protocol/data hashes: `artifacts/frozen_run.json`.
- Single setting: 200 depth-4 Ordered CatBoost trees, learning rate 0.05, L2=10.
- Six columns: four basic predictors, two missing-input flags.
- No search, stacking, cohort overwrite, early stopping or metric-based fallback.
- All final fits/predictions finish before report-year statistics are calculated.
- Backtest years are descriptive and were seen in prior work: no untouched holdout.

## Files

- `artifacts/descriptive_backtest.csv`: every prespecified report year, no shortlist.
- `artifacts/hazard_backtest.csv.gz` and `cost_backtest.csv`: row-level evidence.
- `artifacts/hazard.cbm`, `severity.cbm`: fitted 2027 models.
- `artifacts/predictions_2027.csv`: full inventory, annual predictions and eligibility/unknown-input flags.
- `artifacts/cost_predictions_2027.csv`: severity join interface.
- `artifacts/verification.json`: training cutoffs, counts, checks and artifact hashes.

Integrity gates are independent of metric values. Missing material is retained as
an unknown category, not declared leak-free. Long-horizon calibration and the
private planning score have not been established. Previously inspected data
cannot be made unseen by changing the code or choosing fewer features.

## Recorded descriptive results (no revision after viewing)

| Year | Brier | PR-AUC | Predicted / actual leaks | Repair-cost RMSE | Repair-cost MAE |
|---|---:|---:|---:|---:|---:|
| 2023 | 0.035725 | 0.413611 | 1,193.9 / 1,826 | $163,387.43 | $36,153.99 |
| 2024 | 0.037266 | 0.481582 | 1,344.7 / 1,955 | $98,754.28 | $17,757.51 |
| 2025 | 0.047712 | 0.212397 | 926.4 / 1,773 | $116,465.80 | $31,332.74 |
| 2026 | 0.084528 | 0.372386 | 1,081.9 / 3,038 | $129,901.15 | $26,430.31 |

**Material limitation:** the frozen hazard model underpredicts the annual number
of leaks in every report year, especially 2026 (about 1,082 versus 3,038).
The 2024–2026 mean repair-cost RMSE is $115,040.41. These results do not
establish an accuracy improvement. No settings were changed and no alternative
was selected after reading this table. Do not infer a future planning-score gain
from the process change. Old model comparisons have different eligibility scopes
and must be aligned row-for-row before reporting a numerical improvement.

The 2027 forecast has 43,039 inventory rows, of which 27,248 are eligible.
Seventeen eligible pipes have incomplete inputs (including unknown material);
retain their flags when using forecasts. Hazard training uses 292,137 pipe-years;
cost training uses all 14,113 supplied event records, ending in 2026.

Validation: four temporal/data-integrity tests passed; both saved models reproduce
predictions exactly after loading and after reversing the input row order.
The artifact verifier checks every saved output against its SHA256 manifest.
Git attributes preserve output bytes so newline conversion does not invalidate
release hashes. The current-model selector resolves CMS002 successfully.
