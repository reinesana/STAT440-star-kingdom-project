# Ten survival models for failure-year prediction and future stacking

Owner: Takito Kobayashi. Experiment date: 2026-10-25.

This is a new experiment, separate from earlier repair-cost models. It predicts
the first future leak of a currently unfailed, non-polyurethane pipe **if it is
left in service without preventive replacement**. Pipe ID is a lookup key,
never a model predictor. Both early and late errors receive symmetric treatment.
No stacking model is fitted in this experiment; temporal OOF predictions are
exported for that next step.

## Models

1. Elastic Net Cox: penalized Coxnet with a fitted Breslow baseline.
2. Weibull AFT: censored parametric likelihood, L2-penalized coefficients.
3. Log-normal AFT: a different lifetime tail, same censored-likelihood treatment.
4. Spline survival: annual logistic hazard, material-specific cubic age splines.
5. Random Survival Forest: bootstrap survival trees, terminal-node survival curves.
6. XGBoost-AFT: native censored AFT, normal/logistic/extreme-value distributions.
7. CatBoost discrete-time hazard: annual conditional failure probabilities.
8. Neural MTLR: reverse-cumulative logits and censored discrete PMF likelihood.
9. DeepHit single-event: discrete PMF likelihood plus comparable-pair ranking loss.
10. Hierarchical hazard: annual logistic MAP with material, installation-cohort,
    nested material/cohort effects and separately tuned Gaussian shrinkage.
    This is a penalized hierarchical fit, not full Bayesian posterior sampling.

The two neural losses are implemented directly in `models.py` for inspectability.
DeepHit uses stochastic comparable pairs rather than a full quadratic pair matrix.
The spline model and hierarchical model are explicit implementations, not aliases
for a generic survival classifier.

## Temporal protocol

| Forecast year | Training outcomes | Inner validation years | Requested trials |
|---|---|---|---:|
| 2023 | 2019–2022 | 2021, 2022 | 40 |
| 2024 | 2019–2023 | 2022, 2023 | 40 |
| 2025 | 2019–2024 | 2023, 2024 | 40 |
| 2026 | 2019–2025 | 2024, 2025 | 40 |
| 2027 deployment | 2019–2026 | 2025, 2026 | 60 |

Every inner year is itself predicted from strictly earlier outcomes. Selection
minimizes the mean of the two validation origins' horizon-averaged Brier scores.
The earlier origin evaluates 1- and 2-year cumulative failure probabilities; the
later origin evaluates 1 year, so every label precedes the outer forecast year. Search
budgets are fixed in advance and no outer-year early stopping is used. Each
origin/model study is independent. Failed trials are recorded, never reported
as valid models. The budget counts attempted trials, so actual valid trial
counts must be checked in each selection file.

This follows the expanding-window design in the
[current shared experiment record](https://docs.google.com/document/d/1N32Z_DC6Frw1OYwSNA9mIlXjgDPuKiOQXH4nJV2RZcQ/edit).
Annual results use its 2023/2024/2025/2026/Mean layout. The three-model probability
ensemble previously used Brier score for probability selection; that motivates
using Brier scores here, with an additional 2-year check for timing distributions.
Outer annual tables retain the same next-year comparison format. F1 at 0.5 is
secondary, not the tuning target.

### Pooled landmark construction

At each historical January 1, create one row for every pipe not yet failed and
not originally polyurethane. Features use only inventory and outcomes strictly
before that origin. The target is remaining time from the landmark to first
failure, or administrative censoring at the current training cutoff. Annual
failures are placed at the midpoint of their year; no day-level precision is
claimed. The hazard models use the first-year outcome of each landmark.

A pipe can appear at multiple historical landmarks. These are correlated rows,
not independent new pipes; likelihoods are pooled-landmark working likelihoods.
We do not claim independent-row confidence intervals. Same-pipe presence in an
earlier training year is legitimate for predicting that inventory next year.
Pipe IDs are excluded from features. This also avoids pretending the 2019–2026
history observed the pipes' entire lifetimes since installation: predictions
are conditional on survival to each landmark, not unconditional lifetime fits.

### Features and optimization

Candidate groups: basic attributes; installation cohorts; coordinates and
orientation; exact-endpoint geometry; 50/100/250 m midpoint neighborhoods;
past-only nearby leaks and cohort leak fractions; forecast calendar year.
Coordinates and Euclidean lengths are metres. Endpoint adjacency is a proxy;
planar crossings do not prove physical connectivity. Historical inventory is
valid because every known installation date precedes 2019. Missing dates and
materials are retained, imputed or assigned explicit categories, and flagged
in prediction exports.

Each Optuna trial jointly chooses hyperparameters and basic/all/group-selected/
training-only ranked top-k features. Ranking is an ExtraTrees proposal trained
only on the corresponding training fold's first-year outcomes. Category
vocabularies, medians and scales are also fit inside that fold. Model-defining
age/material/cohort variables are mandatory for the spline/hierarchical models.
This is a finite search, not an exhaustive subset search or proof of optimality.
All search ranges are in `propose()` in `models.py`.

### Leakage checks

- Raw `pipes.csv` and `train.csv` are the only data inputs; no existing enriched
  cost table or future replacement flags are imported.
- Features at year t only use events before Jan 1 of t.
- Training targets only see events through that training cutoff.
- Deleting future events must not change features, risk sets or training labels.
- Selected features, parameters and prediction files are frozen before outer
  outcome reporting. `optimize()` never receives outer-year evaluation labels.
- Smoke tests exercise all ten survival interfaces separately from evaluation.
- Outcomes have been viewed in prior team work; this is a temporal backtest,
  not a newly untouched external holdout. OOF data still need a temporal split
  when training/evaluating a future stacker.

## Failure years and supported horizons

The exported year is the calendar year containing the 50% cumulative failure
crossing. q10/q90 are predictive distribution quantiles, **not confidence
intervals for fitted parameters**. Null quantiles mean the crossing was not
identified within the model's supported horizon.

Parametric AFT, XGBoost-AFT and annual hazard models can extrapolate to 25 years;
those forecasts depend on unverified tail/aging/calendar assumptions. Annual
hazard forecasts age the pipe and advance the scenario year, while keeping
observed neighborhood/cohort history frozen at the forecast origin. They do
not invent future neighbor failures or planned replacements.

Cox baseline, RSF, MTLR and DeepHit have only empirical time support through the
training follow-up length (at most 8 years). Their curves beyond that horizon
are **missing**, not artificially set to zero hazard or a forced failure year.
They still supply comparable next-year OOF predictions for stacking.

The 2019–2026 record cannot validate a 2030–2051 exact failure year. Next-year
Brier/log loss/AP and observable multi-year Brier scores evaluate the parts
that can be tested. MAE computed only among observed failures would selectively
exclude long-lived pipes and is not a valid primary all-pipe timing metric.
Never label these long-horizon estimates as observed or validated future years.

## Files and reproduction

```powershell
python -m pip install -r requirements.txt
python verify.py
python experiment.py --trials 40 --deployment-trials 60
python verify.py --exports
```

`01_.../2023/` through `10_.../2027/` contain per-model, per-origin configurations,
all optimization trials, selected feature names and compressed survival/year
predictions. `results/` contains annual comparisons, temporal OOF probabilities,
residual correlations, subgroup checks and shared-Doc-format TSVs. A completed
`finished.json` is the completion marker, not the presence of an empty folder.

`models/` holds local fitted estimators and `cache/` reproducible intermediates;
both are excluded from Git to avoid large binary artifacts. Code, inputs in the
repository, dependency versions, seeds and selected configurations reproduce
them. SQLite studies/logs are local; portable JSON/CSV trial records are tracked.
`predict.py` provides Pipe ID lookup with a selected saved deployment model.
Known replaced/native-polyurethane pipes receive an explicit ineligible status.
