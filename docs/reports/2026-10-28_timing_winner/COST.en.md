# Follow-up: repair-cost prediction and its interaction with timing

[日本語](COST.ja.md) · [Timing method](METHOD.en.md)

## Decision

Among the 18 additional submissions whose filename order the user confirmed, **C024 (`rion_lgb_log_mean_refit_plain`) has the highest displayed public score, $47.59m**. This implies approximately **$202.41m total cost**, about **$17.11m less** than the timing-only winner. Values come from the rounded public display, not downloaded annual reports; exact cents and exact ties are unverified.

The initial three submissions include another displayed $47.59m. Assigning it to C002 (mean-corrected log LightGBM) assumes the initial C001→C002→C003 order; the supplied screenshot confirms only the later 18. Therefore C024 is the confirmed additional-batch public leader; C002 is the provisional accuracy-oriented choice if that initial mapping is confirmed. The two should remain paired controls in the next experiment rather than declaring a verified exact tie.

## Cost model

These LightGBM variants predict repair amount **conditional on a leak**, using a `log1p(cost)` regression target. They are not failure probabilities. The selected existing 2027 specification uses 309 selected inventory/geometry/history features, 250 trees, learning rate 0.0345333, max depth 6, 35 leaves, minimum child samples 142 and `extra_trees=true`; settings were selected using earlier temporal validation and preserved here. The full existing configuration is [rion_lgb_log_mean_refit_2027.json](../../../notebooks/ensemble_for_cost_prediction/configs/rion_lgb_log_mean_refit_2027.json).

C002 uses a mean retransformation correction estimated from training residuals. **C024 uses the saved uncorrected variant**. Its exact existing implementation exponentiates the log1p prediction without the corrected branch's subtraction of one; do not relabel it as an exact conditional mean or silently change this historical candidate. Log predictions alone generally do not give the dollar-scale conditional mean.

## Internal evidence and overfitting limits

We compared 15 frozen input variants and nine ensemble recipes. Single-model outer years were 2023–2026; comparisons involving every stack use the same 6,766 events in 2024–2026 because the new families lack earlier OOF. Each meta-fit uses preceding OOF only. No new base Optuna search was run.

| Candidate | Common-year mean RMSE | Mean MAE |
|---|---:|---:|
| CatBoost | $113,185.40 | $25,246.75 |
| Previous six-model convex stack | $113,346.55 | $23,471.65 |
| C002 corrected log LightGBM | $114,965.39 | $20,823.18 |
| C024 uncorrected log LightGBM | $120,454.82 | $17,507.43 |

C024 wins its confirmed public batch but has worse internal RMSE. A better public ordering score is not proof of a better amount predictor or freedom from overfitting. Pairwise uncertainty intervals for several leading internal models include zero. Historical years have already been examined, public selection now covers 21 unique cost submissions, and private superiority remains unknown.

## What the public comparison actually changed

The timing forecast F168, 10% cumulative deadline, circle geometry and construction rules stayed fixed. Within deadline groups, the annual first-failure probability mass was multiplied by the fixed 2027 predicted repair amount. The resulting annual rankings were converted to one CSV using budget rollover and head-of-line blocking. This tests a **combined planning policy**, not cost prediction in isolation. Eight missing at-risk cost predictions use the same past-only smoothed surface mean across candidates. All plans cover the original iron and finish by 2046 in the no-future-leak feasibility replay.

The next comparison will vary the combination policy using cached forecasts, with internal screening and a small representative public set. C024 and provisionally C002 are cost inputs, not a reason to retune base models on public feedback.

[User-reported scores and mapping status](evidence/cost_public_scores.csv) · [All internal cost metrics](evidence/cost_internal_comparison.csv)
