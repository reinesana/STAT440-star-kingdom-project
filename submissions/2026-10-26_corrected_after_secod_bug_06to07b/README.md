# 06, 07a and 07b — corrected surface formula

October 26, 2026. Official public scores: pending.

    cost = round_to_cents(100 + surface_rate * area**0.85 + 200 * unreplaced_length)

Only area is exponentiated. The original circle records, original gross-benefit definitions and predictions are retained. Only initial construction costs and the resulting fixed priority order are recalculated.

| Candidate | Benefit |
| --- | --- |
| 06 | Team ensemble_stable predicted repair cost × old LightGBM probability of a first leak during 2027–2028, summed over contained initially unreplaced pipes |
| 07a | Original audit gross benefits: predicted repair cost × full-ensemble 2027 leak probability, summed over contained initially unreplaced pipes |
| 07b | Same gross-benefit approach, with CatBoost + LightGBM 2027 leak probabilities only |

All priorities divide circle benefit by max(initial corrected circle construction cost, $1); ties use original source Pipe ID. No $200/m charge is subtracted from benefits in these gross variants. No models are retrained. 06 uses the existing saved LightGBM model, metadata/category mapping and ensemble_stable severity predictions. 07a/07b use the published original audit benefits, preserving original fallback contributions.

Before export, the original ordering of each candidate is reproduced with the old cost formula. Each new submission preserves the original circle multiset exactly. Exported CSVs are reparsed and validated for full original-iron coverage and annual budget feasibility. Validation counts all contained surface types, current non-polyurethane length, historical replacements, copper/brass, unknown material conservatively, overlapping units, free skips, half-up cent rounding, carryover and head-of-line blocking.

## No-future-leak validation

| Candidate | Construction cost | Completion | Uncovered iron |
| --- | ---: | ---: | ---: |
| 06_corrected_stable_risk_per_work_cost.csv | $194,877,381.84 | 2046 | 0 |
| 07a_corrected_full_ensemble_gross.csv | $195,022,408.73 | 2046 | 0 |
| 07b_corrected_catboost_lightgbm_gross.csv | $194,814,617.24 | 2046 | 0 |

Each file has 38,455 work units. These construction figures exclude future leaks and repairs and are not official score predictions. Public scores remain pending, and private performance is unknown. Preserve original submissions for same-evaluator comparisons; cross-version score differences do not isolate algorithm gains.

## Reproduction

Source repository: https://github.com/reinesana/STAT440-star-kingdom-project

Inputs under source/ include pipes.csv, train.csv, original submissions, original 07a/07b audits, ensemble_stable.csv, risk.txt and metadata.json. Run build_06_07_corrected.py with numpy, pandas, lightgbm and its dependencies installed. LightGBM may require the platform OpenMP runtime. The saved model is used for inference only; all original order reproduction checks passed.

Supporting .audit.csv files show new ranks, original ranks, benefits and corrected initial costs. Matching .validation.json files contain annual execution and hashes. Submit only the three candidate CSVs, not audit files.

Prepared by Rion with Codex assistance. No uploads to GitHub, Google Doc or the evaluator were performed.
