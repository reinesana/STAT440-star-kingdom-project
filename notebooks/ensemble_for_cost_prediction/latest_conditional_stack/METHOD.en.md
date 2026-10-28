# Latest conditional repair-cost stack: method and comparison

[日本語](METHOD.ja.md) · [Model and evidence](README.md)

## Recommendation and measured improvement

The current recommendation is **material × surface conditional convex stacking, optimized for squared error** (`conditional_material_surface_rmse`). This supersedes the earlier stable-ensemble and global-stack recommendation for this severity-prediction task. It predicts repair amount conditional on leakage, not failure probability or the final planning score.

On the same 6,766 events in 2024–2026, mean annual RMSE is **$112,943.12**, below every one of the **24 saved earlier repair-cost recipes** in the [previous comparison](https://github.com/reinesana/STAT440-star-kingdom-project/blob/8359301/docs/reports/2026-10-28_timing_winner/evidence/cost_internal_comparison.csv). The improvement is 0.77% over the earlier stable ensemble, 0.58% over the earlier 15-input optimized stack, and 0.21% over CatBoost. These are descriptive results on reused historical validation, not significant superiority or an untouched test result.

| Earlier repair-cost input / recipe | Mean annual RMSE | Mean annual MAE | New RMSE change |
|---|---:|---:|---:|
| Earlier stable ensemble | $113,822.99 | $22,667.17 | −0.77% |
| Earlier six-input optimized stack | $113,346.55 | $23,471.65 | −0.36% |
| Earlier 15-input optimized stack | $113,605.17 | $23,507.16 | −0.58% |
| CatBoost | $113,185.40 | $25,246.75 | −0.21% |
| Corrected log RF | $113,599.40 | $21,169.98 | −0.58% |
| Corrected log LightGBM (C002 specification) | $114,965.39 | $20,823.18 | −1.76% |
| Uncorrected log LightGBM (C024 specification) | $120,454.82 | $17,507.43 | −6.24% |
| **New recommended conditional stack** | **$112,943.12** | **$23,375.50** | — |

The earlier team pipeline used the stable severity export, including Shana's cohort-stage candidate 16b/17b pipeline: [build_candidate_16.py](../../../src/build_candidate_16.py) reads `ensemble_stable.csv`. These common-year comparisons support a narrower claim: the new amount predictor has lower mean RMSE than the saved earlier amount-model baselines used or compared in team work. They do **not** show that every original fitted model by Shana or every earlier replacement plan is worse. Archived four-year means are not compared directly with new three-year means.

## Why choose this rather than the mean-RMSE winner?

The newly tested three-input Ridge stack has lower mean RMSE, **$112,795.93**, so the conditional model is not the overall error minimum. The user selected it for a smaller observed worst-year deterioration relative to CatBoost while retaining a competitive mean.

| Method | Mean RMSE | 2024 RMSE vs CatBoost | 2025 | 2026 |
|---|---:|---:|---:|---:|
| **Conditional material × surface** | **$112,943.12** | **−0.22%** | **−1.02%** | **+0.53%** |
| Global optimized three-input blend | $113,047.50 | −0.01% | −1.18% | +0.76% |
| Three-input equal blend | $112,973.85 | +0.24% | −2.41% | +1.52% |
| Three-input Ridge | $112,795.93 | −2.51% | −0.89% | +1.83% |

This relative-error comparison helps account for changes in year difficulty; raw annual RMSE dispersion alone mixes model behavior with changes in observed repair costs. Three years are too few to establish future stability. Selection of the family after examining results is itself exploratory. Annual total bias is **+11.30%, +25.26%, +1.89%**, versus **+2.69%, +4.54%, −5.95%** for equal-three. The conditional stack is not the budget-calibration winner; its MAE also exceeds the earlier stable ensemble's.

## Exact model

Base inputs, in order, are Takito CatBoost, Anthony mean-corrected log RF, and Rion mean-corrected log LightGBM. Their own parameters and year-start features are preserved. Fit a global nonnegative weight vector summing to one by minimizing pooled squared dollar error on earlier OOF predictions. Fit a local vector by the same procedure for each material × surface category having at least 50 prior OOF events.

```text
effective_weight(group) = n/(n+tau) × local_weight(group)
                        + tau/(n+tau) × global_weight
prediction = sum(effective_weight × base_forecast)
```

Groups below 50 events and unseen/missing categories use global weights. Weights may be zero; retaining all three as available inputs does not force every group to use all three. The key is `material + " | " + surface`.

For 2027, global weights are **68.6270% CatBoost + 31.3730% corrected log RF + 0% corrected log LightGBM**. There are 18 eligible local groups, tau=50, and 8,592 training OOF events from 2023–2026. The locally verified effective-weight table contains the final group-specific coefficients; the global coefficients alone do not describe the model.

For 2024, fit weights on 2023 OOF; for 2025 on 2023–24; for 2026 on 2023–25. Candidate tau values are 0,50,200,1000,1e9. Select on mean annual earlier-forward RMSE only; use fixed tau=200 when no earlier validation exists. Selected tau is 200,0,50 for the three evaluation years and 50 for 2027. Final 2027 settings are never used retroactively. The tau-search minimum in the 2027 configuration is a selection statistic, not the honest nested recipe's reported $112,943.12.

## Shana's earlier manual probability correction and feature-based replacement

The user identified the earlier manually imposed installation-cohort probability correction as inappropriate for the current modeling policy and requested its removal from the active method. The archived [16b implementation](../../../src/build_candidate_16.py) replaces learned probabilities with 1963:0.12, 1964:0.50, 1965:0.60, 1966:0.30; [17b](../../../src/build_candidate_17.py) further sets 1964:0.55 and 1965:0.70. These are post-prediction probability overrides, not estimated repair-amount parameters. Historical files remain reproducible archives.

The current **F168 timing pipeline does not apply these overrides**. Instead, selected survival/hazard models learn from installation year/decade, age, material × installation cohort and **past-only** cohort leak fractions. Saved [2027 feature lists](https://github.com/reinesana/STAT440-star-kingdom-project/blob/8359301/docs/reports/2026-10-28_timing_winner/evidence/base_models_2027.json) confirm `lay_year`, `lay_decade`, `material_cohort`, `cohort_prior_rate` and/or `age` in the relevant bases. Not every base uses every feature. The correction's hard-coded probabilities were not inserted as target-like features.

Thus the team prefers the current feature-based approach as the more defensible, reproducible modeling procedure even though later displayed planning scores are below some older manually adjusted plans. It is **not established that removing this correction alone caused the lower score**: geometry, ordering, forecast inputs and evaluator versions also differ. Higher historical public scores do not validate the manual override, and removing it does not prove that every probability is now correct or calibrated.

This must be distinguished from the **new repair-cost stack**: its meta inputs are the three amount predictions and material/surface categories. Its base amount models include year-start age (and, in some models, past age-band cost summaries), but the stack itself is not the F168 cohort timing model. Log-to-dollar mean retransformation corrections in the RF/LightGBM amount inputs are retained; they are not the removed installation-year probability override.

## Evidence, publication scope and official evaluation

The source experiment passed independent event-level metric recomputation, upstream training/tuning cutoff checks, all-origin refits, future-label poisoning invariance, reordered queries and unknown-category fallback. At the user's explicit request, the final deployment was rebuilt: all three bases were freshly fitted on 14,113 events from 2019–2026 with fixed selected settings, and conditional weights were refitted on 8,592 temporal OOF events from 2023–2026. The reloaded fitted estimators and preprocessing reproduce all 27,240 2027 forecasts within numerical tolerance, including reversed query order.

[Actual fitted model](fitted_model_2027.joblib), [2027 forecasts](predictions_2027.csv), [training/deployment verification](evidence/deployment_verification.json), [group weights](model_2027.json), [comparison tables](evidence/mean_metrics_all_recipes.csv), [earlier 24-recipe comparison](evidence/prior_model_comparison.csv) and [run commands](README.md#実行方法--use) are published together. The artifact includes the actual three base estimators, preprocessing and log corrections; it is not weights alone. The base-forecast replay requires only the bundled predictions; actual base inference requires a compatible prepared feature CSV. Raw repair targets and prepared feature matrices are not published. `train.py` can rebuild from compatible local prepared features and OOF without a new hyperparameter search.

The previous benchmark and timing-feature lists linked above were already published at commit 8359301. Historical years have been inspected repeatedly, so this is reused-validation evidence, not an untouched final holdout. The selected model is a post-validation, user-approved stability-oriented choice. Old four-year averages are not compared to new three-year averages.

The current forecast supports 27,240 pipes. Older public exports include 28,926 rows under a different extrapolation policy; they remain unchanged. No new predictions are invented for the other 1,686 IDs.

There is **no official planning score for this new conditional amount stack**. Existing C024/C002/21a results are not reassigned to it. Combining it with timing forecasts and rebuilding order under the current construction-cost rules require a separate official same-version evaluation.
