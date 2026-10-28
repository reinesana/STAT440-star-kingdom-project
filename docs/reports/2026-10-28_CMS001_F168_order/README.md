# CMS001 + F168: forecasting models and the combined repair order

[日本語版 / Japanese](README.ja.md)

## Selected plan and deliverables

**02_OVERLAP_CREDIT combines CMS001 repair-cost forecasts, F168 failure-time forecasts, and a dynamic work-priority rule. Its official Public score is 73.71M.** The 2027 forecasting models had already been trained using data through 2026. This package reuses their frozen forecasts and regenerates the complete repair order; it does not claim that all base models were retrained during this publication step. No observed outcomes from 2027 onward are used in forecast training.

Failure timing and planned construction timing are different quantities. The original 38,455 work circles retain exactly the same centers and radii. Candidate generation does not inherit 12a's order, planned years, or old forecasts. Physical inventory and historical leak IDs are used for construction calculations and initial replacement status.

| File | Purpose |
|---|---|
| [submission_2027.csv](submission_2027.csv) | Headerless official submission: 38,455 circles in execution order |
| [work_units_ranked.csv](work_units_ranked.csv) | Circle ranks, geometry, no-future-leak construction years and costs |
| [active_iron_pipes_ranked.csv](active_iron_pipes_ranked.csv) | All 26,237 unreplaced target iron pipes, sorted by their first covering work unit |
| [all_original_iron_pipes.csv](all_original_iron_pipes.csv) | All 38,455 original iron pipes, including 12,218 already replaced by the end of 2026 |
| [reproduce.py](reproduce.py) | Rebuild the submitted order from the two frozen forecasts and the saved policy recipe |
| [recipe.json](recipe.json) | Coefficients and accepted order transformations needed for exact reproduction |
| [verification.json](verification.json) | Counts, cutoff, hashes, coverage and budget checks |

Already repaired pipes are not scheduled for replacement again, but their original circles remain in the coverage submission. Pipes replaced together in one circle are tied; Pipe ID breaks the tie in the pipe-level table and is not a separate physical construction sequence. Construction years and costs assume no future leaks. They are planning scenarios, not promises about actual evaluation-year execution.

## Official comparison

| Candidate | Public score, M | Change |
|---|---:|---|
| 01 FINAL_BLOCKS | 73.54 | Joint index, self-derived calendar and order moves |
| **02 OVERLAP_CREDIT** | **73.71** | Adds credit for redundant overlapping work |
| 03 PROFILE | 73.67 | Forecast-profile weights |
| 04 SAME12A_INDEX | 73.68 | Published 12a index with only the forecast inputs replaced |
| 05 S092 | 73.46 | Self-derived calendar refinement |
| 06 fine blocks | 72.85 | Smaller order moves |
| 07 inverse time | 73.38 | Inverse first-failure-time weighting |
| Old 12a, resubmitted to the current site | 69.89 | Reference with the same circle multiset |

The mapping for 01–04 and 05–07 comes from the user's stated submission order and screenshots. The 12a resubmission was verified in the browser. Rounded displayed scores give a **+3.82M Public gain over 12a**. The difference between 02 and 04 is only 0.03M.

02 is the best Public result among these seven fixed-circle candidates. This is not proof of global optimality, Private superiority, or superiority on every forecasting metric. Other team submissions with different geometry are outside this claim. Synthetic internal worlds sometimes reverse the ranking, and the earlier requirement for improvement in all three worlds was not met. The user explicitly chose official Public results as the adoption criterion. Repeated Public-driven selection can overfit that evaluation; Private performance remains unknown.

## Combining the forecasts

For pipe i, let C_i be CMS001's predicted repair cost conditional on a leak, S_i(k) be F168's k-year survival probability, F_i(k)=1-S_i(k), and L_i be straight-line length in metres. Eight unreplaced pipes missing from CMS001's supported forecast inventory use the median of CMS001's own predictions. Their IDs and the exact fallback value are in verification.json. No third cost forecast or historical target mean is added.

The final-stage pipe value is:

```text
r_i = (1 - 0.32273279945607375) * F_i(2)
      + 0.32273279945607375 * F_i(25)
C_adjusted_i = C_i / (max(L_i, 0.1) / median(L_live))**0.0030795309070914337
v_i = r_i**0.8300158667969959
      * max(C_adjusted_i - 0.9780439932142714 * 200 * L_i, 0)**0.878016813874991
```

These coefficients were selected by finite internal policy searches. They transform the ordering index; they do not overwrite the forecast files or apply the old cohort-specific probability overrides.

For circle j, sum v_i over its currently unreplaced contained pipes. Q_j is the sum of fixed/surface work charges of other unfinished circles whose remaining live pipes would all be replaced by selecting j. Such circles subsequently become free skips. Shared pipe-length charges are not added to Q_j. This is a heuristic credit: another future circle might also eliminate those charges, so Q_j is not an exact global counterfactual saving.

```text
priority_j = (sum(v_i over remaining contained pipes) + 0.01 * Q_j)
             / max(current_construction_cost_j, 1)**0.8490093178605346

current_construction_cost_j = 100
    + sum(unique contained surface rates) * area_j**0.85
    + 200 * sum(currently non-polyurethane contained lengths)
```

Only area is exponentiated in the actual current-rule construction cost. The priority denominator's exponent does not change the billing rule. After every selected circle, update replacement status, benefits, construction costs and redundancy credits. Credit can change two adjacency hops away, which the implementation includes. Fully replaced circles are skipped for free.

## Full construction of the seed order and its calendar

1. **CF150:** dynamically order circles using a joint index of CMS001 and F168's two-year and 25-year risks. Initial coefficients are recorded in `recipe.json → self_seed_recipe.joint_config`.
2. Test single-job insertions and block insertions/reversals, retaining internally improving changes. This starts from the two-model order, not from 12a.
3. Execute that order in a planning simulation with a 10M annual budget, carryover and head-of-line blocking. Derive its own planned years. S092 uses three-year priority bands and the final-stage v_i above to rebuild the order.
4. Apply the accepted `self_blocks` and `final_blocks` transformations to obtain FINAL_BLOCKS.
5. Derive planned years again from FINAL_BLOCKS. Final candidate 02 uses **one-year bands**. First prefer smaller `max(self_planned_year, current_planning_year)`; within that priority year, prefer higher priority_j. A pending unaffordable head job is first next year. Available budget can bring a later-planned job forward. These years are priority bands, not hard deadlines.
6. Reconstruct the entire order from the forecasts and saved recipe, then assert byte-for-byte agreement with the SHA-256 of the 73.71M submission.

The saved transformations are parameters of a policy selected using synthetic internal futures, not another repair-cost or failure predictor. This recipe reproduces the frozen 2027 deployment. It does not guarantee the same transformations are appropriate for a different inventory or new training data.

## Reproduction and training cutoff

From the repository root:

```bash
python -m pip install -r docs/reports/2026-10-28_CMS001_F168_order/requirements.txt
python docs/reports/2026-10-28_CMS001_F168_order/reproduce.py --data-dir data
```

The script reads the repository's existing raw data without duplicating it into this folder. `inputs/` contains the two frozen forecast outputs; `evidence/` contains model settings. Coverage, counts, completion year and input hashes are recorded in verification.json. The independent shared validator also checked the submission. Git attributes preserve published bytes across platforms.

**This command regenerates the order; it does not retrain the forecasting models.** Full base-model training follows the original model-specific training procedures and requires additional computation.

- **CMS001:** its three base models were refitted on 14,113 leak records from 2019–2026. Conditional stacking weights use 8,592 temporal out-of-fold records from 2023–2026.
- **F168:** base training uses data no later than the end of 2026. Meta-training uses 250,776 pipe-year out-of-fold observations from 2020–2026. The 2027–2051 survival curves include extrapolation and are not empirically validated 25 years into the future.

The appendices below consolidate the detailed explanations of both predictors in this document. **R029 in the timing appendix is an earlier timing-only ordering policy, not the combined 02 policy described above.** Historical statements in those source articles retain their original scope; the new joint Public result is reported above.

---

## Appendix A: CMS001 training, selection and validation

> Historical scope: this source article predates the combined 02 Public evaluation. Any statement that planning evaluation is pending refers to that publication stage. The joint CMS001 + F168 + 02 result is now 73.71M; it is not a standalone cost-forecast accuracy score.

# Latest conditional repair-cost stack: method and comparison

**Model: CMS001 — Material-Surface Cost Stack / 材質・地表面別コストスタック (v1.0.0).** [Current selection](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_model.json) · [Model registry](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/model_registry.json) · [How to switch](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/MODEL_MANAGEMENT.md)
[日本語](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/METHOD.ja.md) · [Model and evidence](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/README.md)

## Direct comparison with the shared Google Doc's team ensemble

The [shared Google Doc](https://docs.google.com/document/d/1N32Z_DC6Frw1OYwSNA9mIlXjgDPuKiOQXH4nJV2RZcQ/edit), read on 28 October 2026, identifies the selected team ensemble as **raw XGBoost + CatBoost + baseline raw LightGBM, each weighted 1/3**. This is a leakage-probability model, confirmed against its [selection record](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/leak_model_comparison/comparison_results/selection.json) and [saved OOF predictions](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/leak_model_comparison/comparison_results/selected_oof_predictions.csv).

**The latest timing model F168 outperformed this team ensemble in four-year mean Brier, mean PR-AUC and mean absolute annual leak-count bias on identical pipe/year observations.** Recompute both on the 132,121 common pipe-years in 2023–2026, with unique joins and matching labels. Mean annual Brier falls from **0.050400 to 0.045625 (9.47% improvement)**; PR-AUC rises from **0.343890 to 0.405169**; mean absolute annual count bias falls from **19.06% to 4.88%**.

| Common-row Brier (lower is better) | Team equal-three | F168 |
|---|---:|---:|
| 2023: 35,817 pipes | **0.035815** | 0.038870 |
| 2024: 33,993 pipes | 0.035003 | **0.033550** |
| 2025: 32,042 pipes | 0.048990 | **0.044183** |
| 2026: 30,269 pipes | 0.081792 | **0.065896** |
| Annual mean | 0.050400 | **0.045625** |

F168 wins Brier in 3 of 4 years. The earlier ensemble still wins 2023 Brier and mean log loss: **0.189145 versus F168 0.230070**, using the same probability clipping epsilon of 1e-7. This is not superiority in every year or metric. Coverage differs, so neither model's previously reported full-population average is mixed into this common-row comparison. [Annual results](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/evidence/shared_doc_timing_ensemble_comparison.csv) and [coverage/source checks](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/evidence/shared_doc_timing_comparison.json) are included. These are reused historical results, not proof of a better official planning score or a causal benefit of particular features.

This probability comparison concerns **F168**, separately from the conditional repair-amount stack below. The shared Doc's rule-based zero probability for native polyurethane or previously replaced pipes is also distinct from the old 16b/17b manual installation-year probability overrides.

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

The earlier team pipeline used the stable severity export, including Shana's cohort-stage candidate 16b/17b pipeline: [build_candidate_16.py](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/src/build_candidate_16.py) reads `ensemble_stable.csv`. These common-year comparisons support a narrower claim: the new amount predictor has lower mean RMSE than the saved earlier amount-model baselines used or compared in team work. They do **not** show that every original fitted model by Shana or every earlier replacement plan is worse. Archived four-year means are not compared directly with new three-year means.

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

The user identified the earlier manually imposed installation-cohort probability correction as inappropriate for the current modeling policy and requested its removal from the active method. The archived [16b implementation](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/src/build_candidate_16.py) replaces learned probabilities with 1963:0.12, 1964:0.50, 1965:0.60, 1966:0.30; [17b](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/src/build_candidate_17.py) further sets 1964:0.55 and 1965:0.70. These are post-prediction probability overrides, not estimated repair-amount parameters. Historical files remain reproducible archives.

The current **F168 timing pipeline does not apply these overrides**. Instead, selected survival/hazard models learn from installation year/decade, age, material × installation cohort and **past-only** cohort leak fractions. Saved [2027 feature lists](https://github.com/reinesana/STAT440-star-kingdom-project/blob/8359301/docs/reports/2026-10-28_timing_winner/evidence/base_models_2027.json) confirm `lay_year`, `lay_decade`, `material_cohort`, `cohort_prior_rate` and/or `age` in the relevant bases. Not every base uses every feature. The correction's hard-coded probabilities were not inserted as target-like features.

Thus the team prefers the current feature-based approach as the more defensible, reproducible modeling procedure even though later displayed planning scores are below some older manually adjusted plans. It is **not established that removing this correction alone caused the lower score**: geometry, ordering, forecast inputs and evaluator versions also differ. Higher historical public scores do not validate the manual override, and removing it does not prove that every probability is now correct or calibrated.

This must be distinguished from the **new repair-cost stack**: its meta inputs are the three amount predictions and material/surface categories. Its base amount models include year-start age (and, in some models, past age-band cost summaries), but the stack itself is not the F168 cohort timing model. Log-to-dollar mean retransformation corrections in the RF/LightGBM amount inputs are retained; they are not the removed installation-year probability override.

## Evidence, publication scope and official evaluation

The source experiment passed independent event-level metric recomputation, upstream training/tuning cutoff checks, all-origin refits, future-label poisoning invariance, reordered queries and unknown-category fallback. At the user's explicit request, the final deployment was rebuilt: all three bases were freshly fitted on 14,113 events from 2019–2026 with fixed selected settings, and conditional weights were refitted on 8,592 temporal OOF events from 2023–2026. The reloaded fitted estimators and preprocessing reproduce all 27,240 2027 forecasts within numerical tolerance, including reversed query order.

[Actual fitted model](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/fitted_model_2027.joblib), [2027 forecasts](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/predictions_2027.csv), [training/deployment verification](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/evidence/deployment_verification.json), [group weights](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/model_2027.json), [comparison tables](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/evidence/mean_metrics_all_recipes.csv), [earlier 24-recipe comparison](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/evidence/prior_model_comparison.csv) and [run commands](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/latest-model-articles-2026-10-28/notebooks/ensemble_for_cost_prediction/latest_conditional_stack/README.md#実行方法--use) are published together. The artifact includes the actual three base estimators, preprocessing and log corrections; it is not weights alone. The base-forecast replay requires only the bundled predictions; actual base inference requires a compatible prepared feature CSV. Raw repair targets and prepared feature matrices are not published. `train.py` can rebuild from compatible local prepared features and OOF without a new hyperparameter search.

The previous benchmark and timing-feature lists linked above were already published at commit 8359301. Historical years have been inspected repeatedly, so this is reused-validation evidence, not an untouched final holdout. The selected model is a post-validation, user-approved stability-oriented choice. Old four-year averages are not compared to new three-year averages.

The current forecast supports 27,240 pipes. Older public exports include 28,926 rows under a different extrapolation policy; they remain unchanged. No new predictions are invented for the other 1,686 IDs.

There is **no official planning score for this new conditional amount stack**. Existing C024/C002/21a results are not reassigned to it. Combining it with timing forecasts and rebuilding order under the current construction-cost rules require a separate official same-version evaluation.


---

## Appendix B: F168 training, selection and validation

# How the winning timing model works

[日本語](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/timing-winner-methods-2026-10-28/docs/reports/2026-10-28_timing_winner/METHOD.ja.md) · [Results and evidence](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/timing-winner-methods-2026-10-28/docs/reports/2026-10-28_timing_winner/README.md)

## 1. What was selected

**F168** predicts a 25-year survival curve for each pipe still at risk. **R029** converts those curves into a fixed sequence of circular work units. These are separate decisions: good probability predictions alone do not determine the best replacement order.

Among 591 forecast recipes, F168 had the lowest mean annual next-year Brier score, **0.04575249**, on the four reused temporal evaluation years. Among the 20 officially submitted candidates, R029 had the lowest official total cost, **$219,523,846.12**.

The selected model does not predict repair amounts. Repair-cost predictions and circle-radius optimization are outside this timing experiment.

## 2. Data and at-risk observations

The inputs are `pipes.csv` and historical leak records in `train.csv`, 2019–2026. Coordinates, straight-line lengths and neighborhood radii are in metres. One training landmark is a pipe observed at the start of a year.

- Originally polyurethane pipes and pipes that leaked before the landmark are outside the at-risk set: historical leaks imply replacement with polyurethane.
- A pipe can contribute several annual landmarks before its first leak. This is temporal landmark training, not a random row split.
- Features use inventory information and leak history strictly before the landmark year. Unknown material is retained as an explicit category; missing installation dates are retained and flagged.
- Training survival targets use only events visible by the training cutoff. Observed failures are placed at the midpoint of their recorded year; the remaining observations are administratively right-censored.

The feature families include material, surface, age, length/log-length and missing-date flags; installation cohorts; planar geometry; exact-endpoint adjacency proxies; 50/100/250 m inventory neighborhoods; and prior-only local leak history. Each base model uses its own selected subset. Exact endpoint coincidence is a geometric proxy, not proof of physical connectivity. The exact 2027 subsets are in [base_models_2027.json](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/timing-winner-methods-2026-10-28/docs/reports/2026-10-28_timing_winner/evidence/base_models_2027.json).

Numeric imputation/scaling and categorical transformations are fitted within the corresponding training split. CatBoost uses its categorical handling. Supervised feature ranking, where used, is also fitted on training data only.

## 3. Ten base survival models

| Model | Implemented approach |
|---|---|
| Elastic-net Cox | Penalized proportional-hazards model with a fitted baseline survival function |
| Weibull AFT | Penalized censored likelihood for log failure time with a Weibull distribution |
| Lognormal AFT | Penalized censored likelihood with a lognormal failure-time distribution |
| Spline survival | Annual logistic hazard with cubic age splines and material-specific age interactions |
| Random Survival Forest | Tree ensemble fitted to censored durations/events |
| XGBoost AFT | `survival:aft`, with right-censored upper bounds |
| CatBoost hazard | Annual leak classifier rolled forward into annual hazards |
| Neural MTLR | Discrete-time censored likelihood with cumulative-logit construction |
| DeepHit-style network | Discrete event-time probabilities, censored likelihood and ranking loss |
| Hierarchical hazard | Penalized MAP logistic hazard with grouped effects; not MCMC posterior inference |

The winning pool contains these ten models. **The team's ordinary XGBoost, CatBoost and LightGBM leak classifiers were included as other comparison candidates, but they are not extra inputs to F168.** In particular, the CatBoost hazard model here is not the same specification as the shared ordinary CatBoost classifier.

Existing, verified, earlier-year nested Optuna selections were reused where available. Missing Cox/forest exports were filled using 12-trial past-only searches. The 2020–2022 warmup bank uses prespecified defaults rather than borrowing the separately tuned 2027 settings. Search budgets differed across base families; this was not an equal-compute or globally exhaustive comparison.

## 4. Survival curves become comparable annual hazards

Let \(S_{im}(k)\) be model \(m\)'s probability that pipe \(i\) survives beyond forecast year \(k\), with \(S_{im}(0)=1\). The corresponding conditional annual hazard is

\[
h_{im}(k)=\operatorname{clip}\left(1-\frac{S_{im}(k)}{\max(S_{im}(k-1),10^{-12})},0,1\right).
\]

Some empirical models have support shorter than 25 years. After the last supported year, the implementation extends the last observed conditional hazard, capped at 0.95. This is an extrapolation assumption, not evidence that 25-year probabilities are calibrated. The 2027 Cox, survival forest, MTLR and DeepHit curves have eight years of empirical support before this extension.

## 5. Ridge stacking learned from past out-of-time predictions

For each historical year \(u\), each base model is fitted/tuned using years before \(u\). Its next-year probability \(1-S_{im}(1)\) becomes one of ten meta-features; the target is whether that pipe leaks during \(u\).

For deployment in 2027, the stack uses **250,776 at-risk pipe-year observations from 2020–2026**. These are out-of-time base predictions, not predictions fitted directly to their own labels.

The meta-model is `StandardScaler` followed by `Ridge`. With standardized columns \(z_m=(x_m-\mu_m)/s_m\), Ridge minimizes

\[
\sum_i(y_i-b-\boldsymbol\beta^\top\boldsymbol z_i)^2+\alpha\|\boldsymbol\beta\|_2^2.
\]

The intercept is not penalized. The coefficients may be negative and do not have to sum to one. Thus this is **not a convex weighted average**. A negative coefficient can correct a correlated forecast's contribution; it does not mean that a failure itself has negative probability, and coefficient size is not causal feature importance.

Optuna runs 12 TPE trials over \(\alpha\in[0.01,10000]\) on a log scale. The latest available past OOF year is the inner validation year. For the 2027 model, fit candidate stacks on 2020–2025 OOF and select alpha by clipped-probability Brier loss on 2026 OOF; then refit on 2020–2026. Scalers are fitted separately within each candidate's training data. The selected deployment alpha is **7308.877234445037**.

At prediction time, apply the same learned annual mapping to each future year's vector of ten conditional hazards:

\[
\hat h_i(k)=\operatorname{clip}\left(b+\sum_m\beta_m\frac{h_{im}(k)-\mu_m}{s_m},0,1\right),\qquad
\hat S_i(k)=\prod_{j=1}^{k}(1-\hat h_i(j)).
\]

Values are clipped and survival is made nonincreasing. Using the stored alpha and saved OOF bank, we reconstructed the deployed scaler/coefficients and obtained **zero maximum absolute difference** from the saved 2027 float32 survival curves. [stacking_2027.json](https://github.com/reinesana/STAT440-star-kingdom-project/blob/takito/timing-winner-methods-2026-10-28/docs/reports/2026-10-28_timing_winner/evidence/stacking_2027.json) contains the exact parameters. This checks the meta-model reconstruction, not an independent refit of every base estimator.

## 6. Temporal evaluation

| Split | Base-model history | Test year |
|---|---|---|
| 1 | 2019–2022 | 2023 |
| 2 | 2019–2023 | 2024 |
| 3 | 2019–2024 | 2025 |
| 4 | 2019–2025 | 2026 |

Meta-training and its inner validation also stop before each outer test year. For example, for test 2023, stack alpha selection uses 2020–2021 OOF for fitting and 2022 OOF for validation, followed by a 2020–2022 OOF refit.

| Test year | F168 Brier | PR-AUC |
|---|---:|---:|
| 2023 | 0.038892 | 0.326699 |
| 2024 | 0.033629 | 0.537127 |
| 2025 | 0.044469 | 0.324140 |
| 2026 | 0.066020 | 0.420636 |
| Annual mean | **0.045752** | — |

Brier evaluates next-year probability error; lower is better. The order's mean observed test-year leak-prevention recall under the annual construction budget was **0.548824**. DeepHit-based orders reached about **0.619952**, so R029 was not the best on this short-horizon ordering metric.

## 7. From probabilities to the R029 work sequence

### Common geometry

All timing candidates use the same 38,455 circles, one anchored to each original iron pipe. Center = endpoint midpoint; radius = half its straight-line length + \(10^{-6}\) m; output coordinates/radii are rounded to nine decimals. Both endpoints must be contained. Other live pipes fully contained by a circle are replaced too, so an anchored circle is not a guarantee that exactly one physical pipe is replaced.

### Timing and annual priority

Define cumulative failure probability \(F_i(k)=1-\hat S_i(k)\). R029 uses the constant threshold **q = 0.10**, not a binary leak-classification threshold:

\[
a_i=\min\{k:F_i(k)\ge0.10\}.
\]

A pipe that never crosses within 25 years receives an outside-horizon sentinel. For each planned year, earlier crossing deadlines come first; work already due or overdue shares the current deadline. Within a deadline group, prioritize the annual **first-failure probability mass** \(\hat S_i(k-1)-\hat S_i(k)\), descending. Remaining ties retain inventory row order.

This is more than sorting by a single predicted date. The annual priority lists are converted into one fixed CSV order through a construction-budget replay. Each year adds $10 million with rollover. The selected but unaffordable head unit stays first in the next year; it cannot be bypassed. Already completed/empty units are skipped. This planning replay assumes no future leaks; the official evaluator later executes the fixed CSV with its own events. There is no online rescheduling from unseen official events.

The corrected work cost is

\[
100+R\,A^{0.85}+200L_{\mathrm{live}},
\]

rounded to cents. Only area is exponentiated. \(R\) sums the distinct surface rates present in the circle, including surfaces on already-polyurethane pipes; \(L_{\mathrm{live}}\) includes currently non-polyurethane contained pipe lengths. A work unit with no live pipe costs zero.

### How 10% was chosen

For each forecast/fold, the constant family compares 10%, 20%, …, 90%. It minimizes forecast-implied expected first failures before the planned replacement, with a fixed small penalty for threshold changes in variable families. The deployed R029 selected 10% using its forecasts and inventory, without using outer test outcomes or public feedback. Earlier folds can select different constant thresholds; R029 names the **selection procedure**, not a claim that 10% was held fixed in every historical fold.

The larger fixed grid had 95,742 rows. The user then reduced the remaining adaptive experiment to nine distinct forecasters × four interpretable schedule families: constant, two segments, three equal segments and free segments. Previous partial broad-search outputs were preserved and excluded from this focused comparison. Twelve fixed-grid candidates and eight focused candidates were frozen from internal results before their own official evaluation.

## 8. Official result and interpretation

| Item | Amount |
|---|---:|
| Official CAPEX | $158,410,940.32 |
| Official OPEX | $61,112,905.80 |
| Total | **$219,523,846.12** |
| Public score = $250m − total | **$30,476,153.88** |
| Next best submitted total | $226,735,113.14 |
| Difference | **$7,211,267.02 lower** |

The no-future-leak planning construction cost is approximately **$194.81m**, finishing in **2046**. That is a feasibility control, not the official total: official leaks can replace pipes before planned works and change later live lengths and skipped units. Internal test-year-only dollar totals are also not 25-year total-cost estimates.

The evidence supports using F168/R029 as the current timing baseline for later cost-model or circle experiments. It does not establish why 10% is intrinsically optimal, that variable schedules always lose, or that Ridge caused the improvement by itself. Many recipes were compared on already examined historical years, and the public winner was selected among 20 submissions. Private-test improvement and freedom from overfitting remain unproven.
