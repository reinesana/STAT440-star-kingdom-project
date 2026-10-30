# Integrated historical coverage — public $77.46M

Read the [report](../../reports/2026-10-30_integrated_coverage.md) for method, result and limitations.

- [Submitted CSV](submission_integrated_coverage.csv) — 30,094 circles.
- [Source](build_submission.py) and [official result / input hashes](official_results.json).
- Current cost model: `2026-10-26-area-exponent`. Private score unknown.

From the repository root, install the root requirements and use a **new** output directory:

```sh
python submissions/2026-10-30_integrated_coverage/build_submission.py --output-dir work/reproduce_integrated_coverage
```

The source uses repository-relative frozen inputs and shared helpers. It refuses existing output directories and checks regenerated CSV bytes. No model is retrained.
