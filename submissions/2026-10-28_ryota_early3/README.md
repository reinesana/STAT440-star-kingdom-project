# STAT440 independent risk-priority circle optimizer

## Latest submitted result

**Integration validation note:** These are frozen, user-submitted CSVs. They cover
all initially unreplaced iron, but fail the shared validator's stricter requirement
to geometrically cover every original iron pipe (including historical repairs).
The shared active-iron budget validation and zero-hazard simulator cost cross-check
are recorded separately in `outputs_early3/shared_validation.json`. Do not describe
the submitted files as passing the full shared geometry requirements.

At original generation the assignment text was unavailable. During integration,
`docs/assignment_2026-10-14.md` and `docs/current_rules.md` were read. Their geometry
requirement is stricter than the original experiment's active-iron assumption.
Historical reports below are preserved as experiment provenance.

The user reported an official displayed score of **$76.91M** for
`outputs_early3/early3_adaptive.csv`, compared with **$71.40M** for the previous
`outputs/multi_pipe.csv`: an improvement of **$5.51M (7.72%)** using the rounded
displayed scores. This is user-provided leaderboard evidence, distinct from the
local simulations. Screenshots, submission hashes, and provenance are recorded
in `outputs_early3/official_results.json`.

The newer method emphasizes 2027–2029, uses finer age-based risk estimates,
and chooses grouped or isolated pipes by marginal expected avoided leak loss
per construction dollar. See `EARLY3_METHOD.md` and `outputs_early3/SUMMARY.md`.
Reproduce with `python early_three_years.py --sims 30` after the setup below.

This folder is a reproducible local experiment using the uploaded `data.zip`,
`build_baseline.py`, `risk_priority.py`, and `simulate_score.py`. The three supplied
scripts are preserved unchanged in `reference/`; `optimize.py` is a separate
implementation. It uses NumPy and pandas, without LightGBM or SciPy.

The packaged code automatically uses the repository-root `data/` when no local
`data/` is present. No raw data is duplicated in this submission folder.

## Reproduce

Use Python 3.10 or newer. Install `requirements.txt`, put the ZIP's `pipes.csv` and
`train.csv` in `data/`, then run from this folder:

```sh
python -m unittest discover -v
python optimize.py --sims 30
```

Results go to `outputs/`. CSV submissions have **no header** and two columns:
`"(x, y)",radius`, with nine decimal places. `multi_pipe.csv` is the experimental
multi-pipe submission; `pairwise.csv` and `single_pipe.csv` are comparison arms.
`report.json` includes data hashes, exported submission hashes, geometry audits,
chronological prediction checks, deterministic budget results and simulation
summaries. `simulations.csv` retains every trial; `comparison.csv` summarizes them.

## Rules verified and their provenance

The attached formula image confirms:

```
100 + (sum of distinct contained-pipe surface rates) * (pi * radius^2)^0.85
    + 200 * total length of contained non-polyurethane pipe
```

Rates from the supplied baseline: grassland 3, farmland 7, swamp 12, road 17,
structure 50, water 32. Geometry requires **both endpoints** inside the circle;
convexity then guarantees containment of each straight segment. Partially intersected
pipes incur no cost under the supplied code's interpretation. All contained pipes,
including historical replacements, contribute surface types; only pipes still
unreplaced contribute replacement length. Empty replacement units are skipped.

Other working rules come from the supplied scripts, **not an independently
available official assignment**: 2027–2051 inclusive, annual 10 million budget,
unused budget carries forward, circles execute in CSV order, an unaffordable next
circle blocks subsequent circles, and original polyurethane plus historical leak
pipes start replaced. Construction costs round half up to integer cents. Iron
means cast, wrought, or gray iron. Coverage is required for all initially unreplaced
iron, with historical replacements already satisfying the retirement objective.
The report separately records original-iron geometry coverage.

The official problem image was not available in the recovered attachments. The
synced Homework 1 PDF concerns train/test splits and diamonds, not this contest.
Therefore official timing of leaks versus construction, treatment of unknown
materials, radius restrictions, and any additional submission rules remain
unconfirmed. Recheck these against the actual assignment before submission.
No arbitrary maximum radius or minimum radius rule has been invented here.

## Search and comparison

The search considers the five nearest active iron neighbors within 125 coordinate
units of each active iron pipe. It tests all anchor pairs, anchor triples from the
four closest neighbors, and nearest-neighbor prefixes of size four through six.
Each proposed circle encloses all endpoints of the proposed group. A one-micrometre
margin protects serialization. A spatial grid finds **all** incidentally contained
pipes before evaluating the cost. The algorithm greedily selects disjoint active
iron groups with positive construction savings relative to their singleton costs;
unselected pipes receive singleton circles. This is a bounded local heuristic,
not an exact global optimization. Savings used in selection can overlap through
incidental pipes; final costs always use sequential replacement state.

Pairwise and multi-pipe arms use the **same** candidate construction, cost rule,
greedy selection criterion and risk ordering. The pairwise arm accepts candidates
containing at most two active iron pipes, whereas the multi-pipe arm accepts at
most six. A single pipe's own disk may incidentally contain additional active iron;
fallback singleton disks remain allowed in every arm. Thus group-size counts mean
selected design groups, not necessarily actual replacement counts on execution.
This pairwise baseline is independently constructed: the supplied baseline script
is actually one-circle-per-original-iron, oldest-first, not a pairwise optimizer.
The independent `single_pipe` arm instead uses the shared risk ordering for a fair
geometry comparison. No unavailable teammate pairwise CSV was assumed.

Risk priority is two-year expected leak cost divided by initial construction cost.
Annual leak probabilities use material, five-year age band and surface, shrunk
toward a material rate with 100 at-risk pipe-years of prior weight. Material rates
shrink toward the global rate with weight 100. Historical exposures stop at the
first leak. Missing dates use the median installation age; unknown material gets
the pooled fallback. Historical 2025 and 2026 checks train strictly on earlier
leak outcomes. Configuration is fixed, not tuned on these checks. Severity priority
uses historical material/surface mean costs with surface fallback. The model is
intentionally transparent and independent of the supplied LightGBM implementation.

## Simulation interpretation and limitations

All arms share identical leak events and cost draws for each trial (seed 440).
Three sensitivity scenarios multiply fitted annual hazards by 0.5, 1 and 2,
capped at 0.95. Each pipe leaks at most once. Construction executes at the start
of each year, followed by that year's leaks; leaked pipes are polyurethane next
year. Costs are bootstrapped by surface, so simulation severity is coarser than
the priority model. Repair charges use integer cents. The 25-year horizon continues
even if construction finishes, including leaks in unreplaced copper/brass/unknown
pipes; this fixes the supplied simulator's early-stop omission.

Score is 250 million minus construction and repair costs, reported only for
feasible trials. The paired Monte Carlo interval measures finite-simulation
uncertainty **conditional on the fitted model**. It does not measure model error,
and hazard extrapolation outside observed ages is especially uncertain. These
are simulated scores, never official contest scores. No-future-leak results
separately establish ordered budget feasibility without relying on leaks to
reduce replacement costs. A higher simulated score does not guarantee a better
hidden-test score.

Tests cover known enclosing disks including collinear/duplicate endpoints,
spatial queries against brute force, cent rounding, budget carryover, and leaks
after construction completion. Export validation re-reads the actual rounded CSV,
recomputes endpoint containment and costs, and checks active iron coverage.

## GitHub handoff

Commit `optimize.py`, `test_optimize.py`, `requirements.txt`, this README,
`SUMMARY.md`, `reference/`, and `outputs/` to your repository. `data/*.csv` is
ignored to keep source data out of GitHub by default; obtain and extract the
original ZIP to reproduce. Review redistribution permission before including
the supplied reference scripts. The original local handoff did not publish a remote repository; this dated folder
is the GitHub integration of that handoff.
