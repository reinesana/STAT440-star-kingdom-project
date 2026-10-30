# Corrected 21a comparison — public $74.95M

Read the [report](../../reports/2026-10-30_21a_comparison.md) for method, result and limitations.

- [Submitted CSV](21a_corrected_order.csv) — 36,375 circles.
- [Source](reproduce.py) and [official result / input hashes](official_results.json).
- Current cost model: `2026-10-26-area-exponent`. Private score unknown.

From the repository root, install the root requirements and use a **new** output directory:

```sh
python submissions/2026-10-30_21a_comparison/reproduce.py --output-dir work/reproduce_21a_control
```

The source uses repository-relative frozen inputs and shared helpers. It refuses existing output directories and checks regenerated CSV bytes. No model is retrained.
