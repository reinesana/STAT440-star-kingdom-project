# How the winning timing model works

[日本語](METHOD.ja.md) · [Results and evidence](README.md)

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

The feature families include material, surface, age, length/log-length and missing-date flags; installation cohorts; planar geometry; exact-endpoint adjacency proxies; 50/100/250 m inventory neighborhoods; and prior-only local leak history. Each base model uses its own selected subset. Exact endpoint coincidence is a geometric proxy, not proof of physical connectivity. The exact 2027 subsets are in [base_models_2027.json](evidence/base_models_2027.json).

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

Values are clipped and survival is made nonincreasing. Using the stored alpha and saved OOF bank, we reconstructed the deployed scaler/coefficients and obtained **zero maximum absolute difference** from the saved 2027 float32 survival curves. [stacking_2027.json](evidence/stacking_2027.json) contains the exact parameters. This checks the meta-model reconstruction, not an independent refit of every base estimator.

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
