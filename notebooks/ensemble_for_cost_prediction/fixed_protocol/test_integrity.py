"""Temporal availability and risk-set invariants, independent of model accuracy."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from model import features, risk_rows, cost_rows, forecast


class FixedProtocolIntegrity(unittest.TestCase):
    def setUp(self):
        self.pipes = pd.DataFrame({
            "pipe_id": ["old", "now", "later", "poly", "unknown", "unbuilt"],
            "material": ["cast iron", "copper", "brass", "polyurethane", "__unknown__", "cast iron"],
            "surface": ["road"] * 6, "length_m": [3., 5., 7., 2., np.nan, 4.],
            "lay_date": pd.to_datetime(["1960-01-01"] * 4 + [None, "2030-01-01"])})
        self.events = pd.DataFrame({"pipe_id": ["old", "now", "later"],
                                    "date": pd.to_datetime(["2022-12-31", "2023-01-01", "2024-06-01"]),
                                    "cost": [10., 20., 30.], "event_year": [2022, 2023, 2024]})

    def test_year_start_risk_set_and_first_event_removal(self):
        q = risk_rows(self.pipes, self.events, 2023).set_index("pipe_id")
        self.assertEqual(set(q.index), {"now", "later", "unknown"})
        self.assertEqual(int(q.loc["now", "y"]), 1)
        self.assertEqual(int(q.loc["later", "y"]), 0)
        self.assertNotIn("now", set(risk_rows(self.pipes, self.events, 2024).pipe_id))

    def test_future_labels_costs_do_not_change_earlier_training(self):
        changed = self.events.copy()
        changed.loc[changed.event_year >= 2024, "cost"] = 1e12
        changed.loc[changed.event_year >= 2024, "pipe_id"] = "unknown"
        pd.testing.assert_frame_equal(risk_rows(self.pipes, self.events, 2023), risk_rows(self.pipes, changed, 2023))
        pd.testing.assert_frame_equal(cost_rows(self.pipes, self.events, 2023), cost_rows(self.pipes, changed, 2023))

    def test_missing_inputs_are_flagged_and_not_silently_dropped(self):
        x = features(self.pipes, 2023).set_index("pipe_id")
        self.assertEqual(int(x.loc["unknown", "age_missing"]), 1)
        self.assertEqual(int(x.loc["unknown", "length_missing"]), 1)
        self.assertIn("unknown", set(risk_rows(self.pipes, self.events, 2023).pipe_id))

    def test_forecast_does_not_consume_same_year_or_future_labels(self):
        class ConstantModel:
            def predict_proba(self, x):
                return np.tile([0.8, 0.2], (len(x), 1))
            def predict(self, x):
                return np.repeat(100., len(x))
        config = json.loads((Path(__file__).parent / "PROTOCOL.json").read_text())
        model = ConstantModel()
        a = forecast(self.pipes, self.events, 2023, model, model, config)
        b = forecast(self.pipes, self.events[self.events.event_year < 2023], 2023, model, model, config)
        pd.testing.assert_frame_equal(a, b)
        q = a.set_index("pipe_id")
        self.assertEqual(float(q.loc["old", "p_leak"]), 0.)
        self.assertEqual(float(q.loc["poly", "p_leak"]), 0.)
        self.assertEqual(float(q.loc["unknown", "p_leak"]), 0.2)


if __name__ == "__main__":
    unittest.main()
