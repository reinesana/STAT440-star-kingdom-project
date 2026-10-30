"""Verify a frozen release without selecting or training any model."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from model import sha256


def verify(folder):
    root = Path(folder)
    report = json.loads((root / "verification.json").read_text())
    frozen = json.loads((root / "frozen_run.json").read_text())
    for name, digest in report["output_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError("Artifact checksum mismatch: " + name)
    # Canonicalize source text because a Git checkout can convert line endings.
    for filename, field in [("model.py", "source_sha256"), ("PROTOCOL.json", "protocol_sha256")]:
        raw = (root.parent / filename).read_bytes()
        if frozen[field] not in {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}:
            raise ValueError("Frozen source mismatch: " + filename)
    predictions = pd.read_csv(root / "predictions_2027.csv")
    assert len(predictions) == report["rows"] and predictions.pipe_id.is_unique
    assert predictions.p_leak.between(0, 1).all()
    assert (predictions.predicted_repair_cost >= 0).all()
    assert np.isfinite(predictions.predicted_repair_cost).all()
    assert predictions.loc[~predictions.eligible_at_year_start, "p_leak"].eq(0).all()
    metrics = pd.read_csv(root / "descriptive_backtest.csv")
    assert metrics.year.tolist() == frozen["protocol"]["report_years"]
    print("All artifact hashes, frozen source and forecast invariants verified.")


if __name__ == "__main__":
    verify(Path(__file__).parent / "artifacts")
