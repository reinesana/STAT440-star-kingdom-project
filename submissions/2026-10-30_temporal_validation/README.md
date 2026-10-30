# Temporal prediction validation

Read the [report](../../reports/2026-10-30_temporal_validation.md). Three temporal origins and six controlled one-year policy replays support retaining the existing $77.46M integrated-coverage submission. No new candidate or submission is recommended.

## Reproduce

From the repository root, install `numpy pandas scipy`, then run:

```powershell
python submissions/2026-10-30_temporal_validation/audit_predictions.py --output-dir outputs/temporal_validation_reproduction
python submissions/2026-10-30_temporal_validation/replay_policy.py --output-dir outputs/temporal_validation_reproduction
```

The first command requires a fresh output directory. The second uses its saved predictions. Saved ML out-of-time forecasts are reused; base ML estimators are not retrained. Ryota models are refit at each historical origin. Publication entry points were syntax-checked but not rerun after adapting paths.

`evidence/` contains compact metrics, protocols, input hashes, checks and retained-file validation. Large per-pipe predictions are regenerated rather than duplicated. This is a validation record, not a new submission package. Historical years were already inspected; 2026 is not an untouched holdout.
