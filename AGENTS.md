# Experiment rules

For all new STAT440 experiments, use `docs/current_rules.md` and the October 26, 2026 correction: `100 + surface_rate * area**0.85 + 200 * unreplaced_length`, rounded to cents. Only area is exponentiated. The October 14 specification and historical snapshots contain an obsolete formula.

Use current shared validators/simulator; record cost-model version and input hashes. Keep new outputs in a new dated directory. Preserve historical CSVs, reports and scores; compare only results with a confirmed matching evaluator version. `legacy_cost=True` exists solely for reproducing archived artifacts and must not be used in new experiments.

Read the current relevant shared experiment Google Doc before model/experiment work: https://docs.google.com/document/d/1N32Z_DC6Frw1OYwSNA9mIlXjgDPuKiOQXH4nJV2RZcQ/edit . Reading is authorized; do not edit it without user authorization. Coordinates and derived straight-line lengths in pipes.csv are metres.

For commits use author `takky-k <tk.tk.tk.20020115@gmail.com>`. Preserve unrelated local changes.
