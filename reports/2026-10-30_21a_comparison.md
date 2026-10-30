# Corrected 21a reference comparison — 30 October 2026

This is a controlled evaluation of the existing `submissions/2026-10-27_current_rules/21a_corrected_order.csv`, not a new model. It uses frozen 20a geometry, p17b probabilities and the existing Takito CatBoost / Anthony corrected-log-RF cost stack. Ordering uses the corrected area-only exponent cost model. The submitted 36,375-row CSV is preserved byte-for-byte.

| Recorded official result | Value |
|---|---:|
| Public display | $74.95M |
| Score from annual totals | $74,945,762.98 |
| Construction | $144,501,626.47 |
| Leak repair | $30,552,610.55 |
| Construction completion | 2041 |

Compared with the $77.46M integrated plan, 21a saved $804,407.65 in repairs but spent $3,315,140.82 more on construction, losing **$2,510,733.17** overall. Keep the integrated plan as the best recorded public result. The simulator had favored 21a under many assumptions; this official comparison shows that modeled rankings did not transfer to this public future. Private performance is unknown.

Full original-iron geometry and budget completion passed the current validator. No-future-leak construction was $187,750,619.70, completing in 2045. Do not compare this result with historical 21a scores under older evaluator versions.

[CSV, source and reproduction](../submissions/2026-10-30_21a_comparison/) · [Recorded result and hashes](../submissions/2026-10-30_21a_comparison/official_results.json).
