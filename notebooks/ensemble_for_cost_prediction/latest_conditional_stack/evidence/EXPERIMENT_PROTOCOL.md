# Frozen three-model temporal ensemble experiment (2026-10-28)

Fixed before running: reuse saved predictions from CatBoost (`takito_catboost`), corrected log RF (`anthony_rf_log_mean`), and corrected log LightGBM (`rion_lgb_log_mean_refit`). Their 2024–2026 mean RMSE/MAE reproduce the three numbers in the user request. The target is repair cost conditional on leakage, in original dollars. IDs are join keys, not predictors. No base models are refitted.

For each origin 2024, 2025, 2026, train meta models using only 2023 through origin-1 OOF. Compare the same 6,766 events; report arithmetic mean of annual RMSE/MAE, annual mean total bias, and annual wins versus CatBoost. Pooled MSE/LAD are the convex training objectives, distinct from the reported annual means. 2023 is unavailable for honest meta-model evaluation because no earlier base OOF is available. Fit 2027 separately on all 8,592 OOF rows and export 27,240 supported pipe predictions.

Families:

- Nonnegative, sum-one convex combinations minimizing MSE and MAE. MSE uses multistart SLSQP with KKT checks; MAE uses sparse exact linear programming with primal objective checks.
- Ridge with intercept and training-only StandardScaler. Seven lambda values: 0, .001, .01, .1, 1, 10, 100; sklearn alpha=n*lambda. Negative dollar forecasts clipped to zero. Report fixed-grid sensitivities plus two forward-selected recipes (choose lowest mean annual prior-validation RMSE or MAE). Default lambda=1 when no earlier validation year exists.
- Conditional convex combinations for material, surface, and material×surface, each under MSE and MAE. Categories from prepared year-start features; no target/date/ID features. Groups with fewer than 50 prior OOF rows and unknown groups use global weights. Other groups use n/(n+tau)*local + tau/(n+tau)*global weights. Tau values 0,50,200,1000,1e9. Select tau on mean annual prior-forward RMSE/MAE matching the objective; default tau=200 in 2024. Every candidate forecast is calculated before scoring; later labels never select earlier recipes.

Controls: individual models, equal-three average, original 15-input optimized stack. Parameter selection is nested forward in time: 2025 selections can use only 2024 forward validation; 2026 selections 2024–25; final 2027 selections 2024–26. No outer-year labels are used in meta fitting/selection. No random folds. Fixed grid results are sensitivity analyses, not independently validated model selection.

Verification: unique joins and matching targets/years, upstream base model training/tuning cutoffs and hashes, exact controls, independent metrics from saved CSV, optimizer checks, deterministic parameter refit, future-label poisoning invariance for selected recipes, forecast reconstruction from JSON including reversed query order and unseen-group fallback. Save environment and SHA256 manifests. Checks cannot rule out every upstream feature defect. Historical years have been inspected repeatedly: results are reused-validation evidence, not untouched holdout or official competition scores.

Current construction cost model reference: `docs/current_rules.md`, version `2026-10-26-area-exponent`. This experiment does not simulate construction, create a work plan, or submit to an evaluator. Read current shared Doc on 2026-10-28; it prioritizes RMSE over MAE and uses separate annual MAE/RMSE tables. No edits to shared Doc.

Rerun from project root: `.venv-eda/Scripts/python.exe github-main/notebooks/Takito_Kobayashi/2026-10-28/ThreeModelStacking/run_experiment.py`, then the same interpreter with `verify_experiment.py`. Outputs are confined to this new dated directory.
