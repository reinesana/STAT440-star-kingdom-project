"""Single frozen two-target specification; run from raw supplied CSVs.

Backtest metrics are written only after all fit/predict operations. They are never
read by the training or deployment paths. PROTOCOL.json is the source of settings.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import catboost
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(data_dir):
    pipes = pd.read_csv(Path(data_dir) / "pipes.csv").rename(columns={
        "Pipe ID": "pipe_id", "Lay date": "lay_date", "Material": "material",
        "Surface": "surface"})
    events = pd.read_csv(Path(data_dir) / "train.csv").rename(columns={
        "Pipe ID": "pipe_id", "Date": "date", "Cost": "cost"})
    if not pipes.pipe_id.is_unique or not events.pipe_id.isin(pipes.pipe_id).all():
        raise ValueError("Inventory IDs must be unique and every event must join")
    pipes["lay_date"] = pd.to_datetime(pipes.lay_date, errors="coerce")
    for col in ["material", "surface"]:
        pipes[col] = pipes[col].fillna("__unknown__").astype(str)
    xy = pipes[["GPS x1", "GPS y1", "GPS x2", "GPS y2"]].apply(pd.to_numeric, errors="coerce")
    pipes["length_m"] = np.hypot(xy["GPS x2"] - xy["GPS x1"], xy["GPS y2"] - xy["GPS y1"])
    events["date"] = pd.to_datetime(events.date, errors="raise")
    events["event_year"] = events.date.dt.year
    if events.cost.isna().any() or not np.isfinite(events.cost).all() or (events.cost < 0).any():
        raise ValueError("Cost must be observed, finite and nonnegative")
    # Stable ordering also makes training reproducible after CSV row permutations.
    pipes = pipes.sort_values("pipe_id").reset_index(drop=True)
    events = events.sort_values(["date", "pipe_id", "Time", "cost"]).reset_index(drop=True)
    return pipes, events


def features(pipes, year):
    x = pipes[["pipe_id", "material", "surface", "length_m"]].copy()
    x["age"] = (pd.Timestamp(year, 1, 1) - pipes.lay_date).dt.days / 365.2425
    x.loc[x.age < 0, "age"] = np.nan
    x["age_missing"] = x.age.isna().astype("int8")
    x["length_missing"] = x.length_m.isna().astype("int8")
    return x


def risk_rows(pipes, events, year):
    cutoff = pd.Timestamp(year, 1, 1)
    # Only observations before cutoff determine eligibility. No future status.
    previous = set(events.loc[events.date < cutoff, "pipe_id"])
    eligible = ((pipes.material != "polyurethane") & ~pipes.pipe_id.isin(previous)
                & (pipes.lay_date.isna() | (pipes.lay_date <= cutoff)))
    x = features(pipes, year).loc[eligible].copy()
    current = set(events.loc[(events.date >= cutoff) & (events.date < pd.Timestamp(year + 1, 1, 1)), "pipe_id"])
    x["y"] = x.pipe_id.isin(current).astype("int8")
    x["year"] = year
    return x


def cost_rows(pipes, events, year):
    q = events.loc[events.event_year == year, ["pipe_id", "cost"]]
    x = q.merge(features(pipes, year), on="pipe_id", how="left", validate="many_to_one")
    x["year"] = year
    return x


def fit_pair(risk, costs, config):
    hazard = CatBoostClassifier(**config["parameters"], loss_function=config["hazard_loss"])
    severity = CatBoostRegressor(**config["parameters"], loss_function=config["cost_loss"])
    kwargs = {"cat_features": config["categorical"]}
    hazard.fit(risk[config["features"]], risk.y, **kwargs)
    severity.fit(costs[config["features"]], costs.cost, **kwargs)
    return hazard, severity


def forecast(pipes, events, year, hazard, severity, config):
    x = features(pipes, year)
    eligible = set(risk_rows(pipes, events.loc[events.date < pd.Timestamp(year, 1, 1)], year).pipe_id)
    active = x.pipe_id.isin(eligible)
    x["eligible_at_year_start"] = active
    x["p_leak"] = 0.0
    x["predicted_repair_cost"] = 0.0
    x.loc[active, "p_leak"] = hazard.predict_proba(x.loc[active, config["features"]])[:, 1]
    x.loc[active, "predicted_repair_cost"] = np.maximum(severity.predict(x.loc[active, config["features"]]), 0)
    x["input_incomplete"] = (x.age_missing.astype(bool) | x.length_missing.astype(bool)
                              | x.material.eq("__unknown__") | x.surface.eq("__unknown__"))
    x["expected_annual_repair_cost"] = x.p_leak * x.predicted_repair_cost
    return x


def run(data_dir, output_dir):
    config = json.loads((HERE / "PROTOCOL.json").read_text(encoding="utf-8"))
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new or empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    identity = {"protocol_sha256": sha256(HERE / "PROTOCOL.json"), "source_sha256": sha256(__file__),
                "input_sha256": {name: sha256(Path(data_dir) / name) for name in ["pipes.csv", "train.csv"]},
                "protocol": config, "python": platform.python_version(), "catboost": catboost.__version__,
                "numpy": np.__version__, "pandas": pd.__version__}
    (out / "frozen_run.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    pipes, events = load_data(data_dir)
    if events.event_year.min() < config["training_start"] or events.event_year.max() >= config["forecast_year"]:
        raise ValueError("Unexpected event years: never consume forecast-year labels")
    risk_parts, cost_parts, backtests, cost_tests, training_counts = [], [], [], [], []
    for year in range(config["training_start"], config["forecast_year"] + 1):
        if year in config["report_years"] or year == config["forecast_year"]:
            risk = pd.concat(risk_parts, ignore_index=True)
            costs = pd.concat(cost_parts, ignore_index=True)
            assert risk.year.max() < year and costs.year.max() < year
            hazard, severity = fit_pair(risk, costs, config)
            training_counts.append({"prediction_year": year, "training_end": int(risk.year.max()),
                                    "risk_rows": len(risk), "cost_rows": len(costs)})
            if year == config["forecast_year"]:
                hazard.save_model(str(out / "hazard.cbm"))
                severity.save_model(str(out / "severity.cbm"))
                pred = forecast(pipes, events, year, hazard, severity, config)
                pred.to_csv(out / "predictions_2027.csv", index=False, float_format="%.12g")
                pred[["pipe_id", "predicted_repair_cost"]].to_csv(out / "cost_predictions_2027.csv", index=False, float_format="%.12g")
                reloaded_h = CatBoostClassifier().load_model(str(out / "hazard.cbm"))
                reloaded_s = CatBoostRegressor().load_model(str(out / "severity.cbm"))
                again = forecast(pipes, events, year, reloaded_h, reloaded_s, config)
                np.testing.assert_array_equal(pred.p_leak, again.p_leak)
                np.testing.assert_array_equal(pred.predicted_repair_cost, again.predicted_repair_cost)
                shuffled = forecast(pipes.iloc[::-1], events.iloc[::-1], year, reloaded_h, reloaded_s, config).sort_values("pipe_id")
                np.testing.assert_array_equal(pred.p_leak.to_numpy(), shuffled.p_leak.to_numpy())
                np.testing.assert_array_equal(pred.predicted_repair_cost.to_numpy(), shuffled.predicted_repair_cost.to_numpy())
                assert pred.p_leak.between(0, 1).all() and np.isfinite(pred.predicted_repair_cost).all()
                verification = {"rows": len(pred), "eligible": int(pred.eligible_at_year_start.sum()),
                                "eligible_incomplete_input": int((pred.eligible_at_year_start & pred.input_incomplete).sum()),
                                "save_reload_exact": True, "reversed_input_exact": True,
                                "finite_nonnegative_cost_and_bounded_probability": True,
                                "training_counts": training_counts, "official_evaluation": None,
                                "new_untouched_holdout": False}
            else:
                q = risk_rows(pipes, events, year)
                q["prediction"] = hazard.predict_proba(q[config["features"]])[:, 1]
                backtests.append(q[["pipe_id", "year", "y", "prediction"]])
                c = cost_rows(pipes, events, year)
                c["prediction"] = np.maximum(severity.predict(c[config["features"]]), 0)
                cost_tests.append(c[["pipe_id", "year", "cost", "prediction"]])
                print(f"Completed fixed fit/prediction for {year}; metrics not yet computed", flush=True)
        if year < config["forecast_year"]:
            risk_parts.append(risk_rows(pipes, events, year))
            cost_parts.append(cost_rows(pipes, events, year))
    # Report-only code starts here; no branch above reads these statistics.
    rp, cp = pd.concat(backtests), pd.concat(cost_tests)
    rp.to_csv(out / "hazard_backtest.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    cp.to_csv(out / "cost_backtest.csv", index=False, float_format="%.12g")
    metrics = []
    for year in config["report_years"]:
        q, c = rp[rp.year == year], cp[cp.year == year]
        error = c.prediction - c.cost
        metrics.append({"year": year, "risk_rows": len(q), "leaks": int(q.y.sum()),
                        "brier": brier_score_loss(q.y, q.prediction),
                        "log_loss": log_loss(q.y, q.prediction, labels=[0, 1]),
                        "pr_auc": average_precision_score(q.y, q.prediction),
                        "predicted_leaks": q.prediction.sum(), "cost_rows": len(c),
                        "cost_rmse": float(np.sqrt(np.mean(error**2))), "cost_mae": float(np.mean(np.abs(error))),
                        "cost_total_bias": float(error.sum() / c.cost.sum())})
    pd.DataFrame(metrics).to_csv(out / "descriptive_backtest.csv", index=False)
    verification["output_sha256"] = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
