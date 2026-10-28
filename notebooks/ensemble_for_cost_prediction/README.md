# Repair cost model candidates for 2027

**Current recommendation, 28 October 2026:** [material × surface conditional stack — English / 日本語](latest_conditional_stack/README.md). This replaces the recommendation below for stability-oriented conditional repair-amount prediction. [Actual fitted model through 2026](latest_conditional_stack/fitted_model_2027.joblib) · [2027 forecasts](latest_conditional_stack/predictions_2027.csv). It has no official planning score. Historical prediction CSVs retain their original scope and values.

[English report](report_en.md) · [日本語レポート](report_ja.md)

The historical six CSVs are in `predictions/`: five model candidates plus one reference. Each contains all 28,926 inventory IDs absent from train.csv, with exactly `id,predicted_cost`, sorted by decreasing cost. See the reports before using the 1,686 extrapolated rows. The new model instead exports only 27,240 supported pipes as `pipe_id,predicted_repair_cost`; the two export contracts are not interchangeable.

[Export validation](evidence/export_validation.json) · [Model composition](evidence/candidate_weights.csv) · [Final settings](evidence/final_model_settings.csv) · [Sources](evidence/sources.html)

Training-code snapshots are provided for inspection. They retain their original workspace paths and are not a standalone execution package; the CSV exporter was run in the original reviewed workspace. Fitted model binaries and large intermediate feature tables are not included.

[Repair-all 2027 scenario totals](evidence/repair_all_2027_totals.csv) · [Inputs and outputs](evidence/model_input_contract.json) · [Per-pipe missing-value flags](evidence/input_quality_by_pipe.csv) · [Input-quality audit](evidence/input_quality_audit.json)

To regenerate the report additions and scenario sums from the committed CSVs, run `python notebooks/ensemble_for_cost_prediction/training_code/update_reports.py` from the repository root (pandas, numpy and tabulate required). This updates documentation and audit tables only; it does not retrain or change predictions.
