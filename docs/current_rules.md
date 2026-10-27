# Current experiment rules (October 26, 2026 correction)

Effective for new experiments from October 27. Source: the instructor correction screenshot supplied by Takito, corroborated by the team's merged October 26 rebuild (commit 3e69668) and the current shared experiment Doc.

```text
area = pi * radius**2
surface_rate = sum of distinct surface rates among ALL wholly contained pipes
cost = round_to_cents(100 + surface_rate * area**0.85 + 200 * currently_non_polyurethane_length)
```

Only area is exponentiated. Surface types from already-replaced/polyurethane pipes remain included. Empty or polyurethane-only units cost zero. Both endpoints must be inside the closed circle; lengths and coordinates are metres. Historical leaks set initial polyurethane status. Overlapping units charge each replacement length once. Annual allocation, carryover, head-of-line blocking, 2027–2051 horizon and geometric coverage requirements remain as documented in the assignment.

## Experiment workflow

Use the shared `src/build_cost_submissions.py` validator and `src/simulate_score.py` simulator. New validation records carry `cost_model_version=2026-10-26-area-exponent`. Save new outputs in a new dated folder; old CSVs, validation JSON and reports remain historical evidence. Older generators have fixed dated output paths: change those paths before rerunning to avoid replacing archived results. Scripts under archived submission folders are historical snapshots and may retain the old formula.

The October 14 assignment snapshot is archived unchanged; its product-exponent formula is superseded by this correction. The raw `instruction.docx` outside the repository is likewise not the current formula reference.

Run from the repository root:

```powershell
python -m unittest discover -s tests -p 'test_current_rules.py'
python src/rebuild_current_21a.py
```

The second command keeps 21a geometry and prediction inputs, rebuilds its priority order with the corrected formula, and validates the exported CSV into `submissions/2026-10-27_current_rules/`. It does not retrain predictive models or submit to the official evaluator. The historical `.../reproduce.py` explicitly opts into `legacy_cost=True` solely to reproduce the frozen old 21a artifact. All new experiment calls default to corrected costs; do not set that archival flag.

## Score provenance

The team recorded corrected October 26 public scores of 69.89M (12a) and 70.24M (13a). These are reported official public results in the shared Doc and merged team report. Historical 20a/21a scores must not be assumed comparable without evaluator-version confirmation. New corrected 21a outputs need a fresh official evaluation; no simulator or no-future-leak balance is an official public score.
