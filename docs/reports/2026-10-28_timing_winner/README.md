# Timing-model winner: F168 / R029

[English](METHOD.en.md) · [日本語](METHOD.ja.md)

Follow-up cost comparison: [English](COST.en.md) · [日本語](COST.ja.md). Rounded user-reported public results and confirmed filename mappings are distinguished from exact official annual evidence.

This documents the best **official total cost among the 20 submitted timing candidates** in the October 27–28 experiment. It does not claim an untouched private-test win.

- Forecast: **F168**, ten survival models combined with standardized Ridge stacking.
- Order: **R029**, a constant 10% cumulative-failure threshold plus annual probability and budget allocation.
- Official construction cost: **$158,410,940.32**.
- Official leak repair cost: **$61,112,905.80**.
- Official total: **$219,523,846.12**; public score: **$30,476,153.88**.
- Canonical submitted CSV: `T27118.csv`; SHA-256: `a7268dab2a127f3b19a5c349433755aea1866e573b572cd09176c5d7960b6401`.

## Evidence

| File | Contents |
|---|---|
| [stacking_2027.json](evidence/stacking_2027.json) | Exact scaler, signed Ridge coefficients, intercept, alpha, reconstruction check |
| [base_models_2027.json](evidence/base_models_2027.json) | Each base model's selected settings, features and support |
| [winner_forecast_folds.csv](evidence/winner_forecast_folds.csv) | Four temporal forecast evaluations |
| [winner_order_folds.csv](evidence/winner_order_folds.csv) | Historical ordering metrics and selected thresholds |
| [public20_results.csv](evidence/public20_results.csv) | All 20 submitted candidates and exact official costs |
| [winner_official_annual.csv](evidence/winner_official_annual.csv) | Official 25-year CAPEX/OPEX detail |
| [winner_result.json](evidence/winner_result.json) | Winning submission identity and exact totals |
| [provenance.json](evidence/provenance.json) | Input/source hashes and experiment scope |

This is a methods and evidence package. Full fitted base models, feature caches, the large experiment bank and the submission CSV are not bundled. It is not a standalone training distribution. Historical experiments and the shared Google Doc are unchanged.
