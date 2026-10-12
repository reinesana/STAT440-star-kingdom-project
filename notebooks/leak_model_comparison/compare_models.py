"""Compare seven leak models and all equal-weight subsets of size 2-7.

Selection: lowest equally weighted mean annual Brier score on 2023-2025.
Tie-break: mean annual log loss, then candidate name. Weights are never fitted
on 2026. Freeze selection.json before computing any 2026 performance metrics.
Requires pandas, numpy, scikit-learn; run after xgboost_predictions.py exports
all four years to predictions_all_years/. Original contributors are unchanged.
"""
from pathlib import Path
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

BASE = Path(__file__).resolve().parent
NOTEBOOKS = BASE.parent
OUT = BASE / "comparison_results"


def score(y, p):
    return dict(brier_score=float(brier_score_loss(y, p)),
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                average_precision=float(average_precision_score(y, p)),
                actual_leaks=int(y.sum()), expected_leaks=float(p.sum()),
                leak_count_bias_pct=float(100 * (p.sum() / y.sum() - 1)))


def main():
    paths = {
        "xgboost_raw": NOTEBOOKS / "Amy_Sim/predictions/xgboost_oof_predictions.csv",
        "xgboost_calibrated": NOTEBOOKS / "Amy_Sim/predictions/xgboost_calibrated_oof_predictions.csv",
        "catboost": NOTEBOOKS / "Ryota_Sato/catboost_predictions.csv",
        "lightgbm_weighted_raw": NOTEBOOKS / "Mar_Nacianceno/lightgbm_oof_predictions.csv",
        "lightgbm_weighted_calibrated": NOTEBOOKS / "Mar_Nacianceno/lightgbm_calibrated_oof_predictions.csv",
        "lightgbm_baseline_raw": NOTEBOOKS / "Mar_Nacianceno/lightgbm_baseline_oof_predictions.csv",
        "lightgbm_baseline_calibrated": NOTEBOOKS / "Mar_Nacianceno/lightgbm_baseline_calibrated_oof_predictions.csv",
    }
    master = pd.read_csv(NOTEBOOKS.parent / "data/leak_prediction_master.csv", dtype={"pipe_id": "string"})
    reference = master.loc[master.year.between(2023, 2026), ["pipe_id", "year", "Y"]].set_index(["pipe_id", "year"]).sort_index()
    predictions = {}
    audit = []
    for name, path in paths.items():
        frame = pd.read_csv(path, dtype={"pipe_id": "string"})
        if list(frame.columns) != ["pipe_id", "year", "y_true", "predicted_probability"]:
            raise ValueError(f"Unexpected columns: {name}")
        if frame.isna().any().any() or frame.duplicated(["pipe_id", "year"]).any():
            raise ValueError(f"Missing values or duplicate keys: {name}")
        indexed = frame.set_index(["pipe_id", "year"]).sort_index()
        pd.testing.assert_index_equal(indexed.index, reference.index)
        np.testing.assert_array_equal(indexed.y_true, reference.Y)
        p = indexed.predicted_probability.to_numpy()
        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError(f"Invalid probabilities: {name}")
        predictions[name] = p
        audit.append(dict(model=name, path=str(path), rows=len(frame),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    year = reference.index.get_level_values("year").to_numpy()
    y = reference.Y.to_numpy()
    names = list(predictions)
    candidates = {name: (name,) for name in names}
    for size in range(2, len(names) + 1):
        for members in combinations(names, size):
            candidates["equal__" + "__".join(members)] = members
    # Only development-year labels enter candidate scoring and selection.
    development = []
    for name, members in candidates.items():
        for test_year in [2023, 2024, 2025]:
            mask = year == test_year
            p = np.mean([predictions[m][mask] for m in members], axis=0)
            development.append(dict(candidate=name, year=test_year, members=len(members),
                                    **score(y[mask], p)))
    dev = pd.DataFrame(development)
    leaderboard = dev.groupby("candidate").agg(
        mean_annual_brier=("brier_score", "mean"),
        mean_annual_log_loss=("log_loss", "mean"),
        mean_annual_average_precision=("average_precision", "mean"),
        worst_annual_brier=("brier_score", "max"),
        mean_absolute_leak_count_bias_pct=("leak_count_bias_pct", lambda s: s.abs().mean()),
        members=("members", "first")).reset_index().sort_values(
            ["mean_annual_brier", "mean_annual_log_loss", "candidate"])
    winner = leaderboard.iloc[0].candidate
    best_single = leaderboard.loc[leaderboard.members.eq(1)].iloc[0].candidate
    OUT.mkdir(parents=True, exist_ok=True)
    selection = dict(selected_candidate=winner, members=list(candidates[winner]),
                     weights={m: 1 / len(candidates[winner]) for m in candidates[winner]},
                     best_single=best_single, selection_years=[2023, 2024, 2025],
                     evaluation_year=2026, candidates=len(candidates),
                     rule="Mean annual Brier; ties: mean annual log loss, then name",
                     caveat="Prior exposure to 2026 in the team's development cannot be excluded.",
                     inputs=audit)
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    dev.to_csv(OUT / "development_annual_metrics.csv", index=False)
    leaderboard.to_csv(OUT / "development_leaderboard.csv", index=False)
    # Selection is now frozen. Evaluate seven singles and the chosen procedure.
    evaluation = []
    for name in dict.fromkeys([*names, winner]):
        mask = year == 2026
        p = np.mean([predictions[m][mask] for m in candidates[name]], axis=0)
        evaluation.append(dict(candidate=name, **score(y[mask], p)))
    test = pd.DataFrame(evaluation)
    test.to_csv(OUT / "evaluation_2026.csv", index=False)
    chosen = reference.reset_index().rename(columns={"Y": "y_true"})
    chosen["predicted_probability"] = np.mean([predictions[m] for m in candidates[winner]], axis=0)
    chosen.to_csv(OUT / "selected_oof_predictions.csv", index=False)
    # Calibration bins summarize probability reliability without fitting anything.
    calibration_rows = []
    for period, mask in [("development_2023_2025", year < 2026), ("evaluation_2026", year == 2026)]:
        for name in dict.fromkeys([best_single, winner]):
            p = np.mean([predictions[m][mask] for m in candidates[name]], axis=0)
            temp = pd.DataFrame({"probability": p, "actual": y[mask]})
            temp["bin"] = pd.cut(temp.probability, np.linspace(0, 1, 11), include_lowest=True)
            bins = temp.groupby("bin", observed=True).agg(rows=("actual", "size"),
                       mean_probability=("probability", "mean"), actual_rate=("actual", "mean")).reset_index()
            bins["period"], bins["candidate"] = period, name
            calibration_rows.append(bins)
    pd.concat(calibration_rows).to_csv(OUT / "calibration_bins.csv", index=False)
    selected_dev = leaderboard.iloc[0]
    single_dev = leaderboard.loc[leaderboard.candidate.eq(best_single)].iloc[0]
    gain = 100 * (single_dev.mean_annual_brier - selected_dev.mean_annual_brier) / single_dev.mean_annual_brier
    selected_test = test.loc[test.candidate.eq(winner)].iloc[0]
    report = ["# Leak probability model comparison", "",
              "Seven supplied single-model procedures and all 120 equal-weight ensembles were compared.",
              "Selection uses only 2023-2025; each year has equal weight. 2026 metrics were computed after selection.json was saved.", "",
              "Selected: " + winner, "", "Members: " + ", ".join(candidates[winner]),
              f"Equal weight: {1 / len(candidates[winner]):.6f} each.",
              f"Development mean annual Brier: {selected_dev.mean_annual_brier:.6f}.",
              f"Best single: {best_single}; Brier {single_dev.mean_annual_brier:.6f}.",
              f"Development Brier improvement over best single: {gain:.3f}%.", "",
              f"Selected procedure on 2026: Brier {selected_test.brier_score:.6f}; log loss {selected_test.log_loss:.6f}; average precision {selected_test.average_precision:.6f}.",
              f"2026 expected leaks: {selected_test.expected_leaks:.1f}; actual {selected_test.actual_leaks}; bias {selected_test.leak_count_bias_pct:.2f}%.", "",
              "## Single models on development years", "",
              leaderboard.loc[leaderboard.members.eq(1)].to_csv(index=False),
              "## Final 2026 evaluation (not used to reselect)", "", test.to_csv(index=False),
              "## Limits and deployment", "",
              "The candidate search is exploratory and its development scores are selection-optimistic. No claim of statistical superiority or global optimality is made.",
              "Prior team exposure to 2026 means this is a temporal evaluation, not a guaranteed untouched final test.",
              "New calibrated XGBoost holds the base estimator fixed and reserves the preceding year for calibration. Mar's calibrated LightGBM transfers a prior-year calibration mapping to a refitted estimator. These are distinct procedures.",
              "XGBoost retains Amy's weighted configuration; models are compared as supplied, not with equal tuning budgets. Correlated LightGBM variants can give that model family extra representation in an ensemble.",
              "No future probabilities were generated. Deploy the frozen member/weight choices with the same training/calibration schedule shifted to 2027 and reconstruct the eligible first-leak population.",
              "Long-horizon forecasts require further assumptions about aging, replacement status, and the changing annual leak rate."]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"selected": winner, "members": candidates[winner],
                      "best_single": best_single, "development_brier": selected_dev.mean_annual_brier,
                      "development_gain_pct": gain, "evaluation_2026": selected_test.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
