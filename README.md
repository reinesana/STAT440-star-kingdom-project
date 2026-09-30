# STAT 440 – Star Kingdom Pipe Replacement

Replace every iron pipe with polyurethane between **2027 and 2051**, with **$10M added to the budget each year**. We submit an ordered list of circular work units, one per line, in the form `"(x, y)",r`.

**Score = ($250M − work-unit costs) − leak repair costs during 2027–2051**

The assignment specification is authoritative. Strategy suggestions below should be tested using our scorer.

## Layout

```text
data/          pipes.csv, train.csv (leak history 2019–2026). Don't edit these.
notebooks/     exploration, named yourname_topic
src/           shared code, cost function, simulator and validator
submissions/   submission CSVs, named YYYY-MM-DD_description.csv
```

Setup: create a `.venv` and install the packages in `requirements.txt`.

The assignment calls the historical leak file `costs.csv`; our project uses `train.csv`. Confirm that these refer to the same supplied data.

## Submission format

Submit a CSV with **no header**, one work unit per line:

```csv
"(1234.5, 6789.0)",25.0
```

- Coordinates and radii are in **metres**.
- Each work unit describes a **closed circle**, so the boundary counts as inside.
- Every original iron pipe must be **wholly inside at least one submitted circle**.
- When a unit executes, every currently non-polyurethane pipe wholly inside its circle is replaced, including copper and brass.
- A pipe that only intersects the circle is not replaced.
- For straight pipes, check that **both endpoints** are inside. Checking only the midpoint is insufficient.
- Validate coverage again after exporting the CSV, accounting for numerical precision.

## Work-unit cost

| Component | Rate |
| --- | ---: |
| Groundbreaking | $100/work unit |
| Grass | $3/m² |
| Farm | $7/m² |
| Swamp | $12/m² |
| Road | $17/m² |
| Structure | $50/m² |
| Water | $32/m² |
| Replacement length | $200/m |

For a circle with radius `r`:

```text
area = pi * r²
surface_rate = sum of rates for distinct surface types among all wholly contained pipes
replacement_length = total length of currently non-polyurethane wholly contained pipes

cost = round_to_cents(
    100 + (surface_rate * area)^0.85 + 200 * replacement_length
)
```

### Important cost details

- Count each surface type **once**, regardless of the number or length of pipes.
- Apply the combined surface rate to the **whole circle area**.
- For example, grass and road give a rate of `3 + 17 = 20`.
- Surface types are counted from **all wholly contained pipes**, including polyurethane pipes and pipes already replaced by leaks or earlier work units.
- Only currently non-polyurethane pipes contribute the $200/m term.
- Overlapping circles must not charge replacement length twice.
- If a circle contains no pipes or only currently polyurethane pipes, it is **skipped at zero cost**, including no groundbreaking charge.
- Round each executed work unit's final cost to cents. Use consistent rounding and avoid floating-point budget errors.

## Budget and execution

1. At the beginning of 2027, start with $10M.
2. Execute work units in CSV order. Execution is instantaneous.
3. If the next unit would make the budget negative, stop work for that year. **Do not skip ahead to a cheaper unit.**
4. Carry the remaining budget forward and add $10M at the beginning of the next year.
5. Resume from the blocked unit, recalculating its cost using the current pipe materials.
6. Continue through 2051.

A unit costing exactly the available budget can execute. A unit costing more than $10M may execute after budget accumulates.

The specification is rejected if it cannot finish within the 25-year window or if the required iron replacement and coverage conditions are not satisfied.

Surplus uses the full **$250M allocation**, even if work finishes early. Under the stated rules, repair costs are deducted separately from the score rather than from the construction budget.

## Things to know

- **Order affects both repair costs and work-unit costs.** A pipe that leaks before scheduled replacement becomes polyurethane, reducing later replacement length and sometimes causing a unit to be skipped.
- Pipes that leaked in `train.csv` are already polyurethane when the project starts. `pipes.csv` records their original material.
- Keep original material and current material in separate fields.
- During the project, a leak incurs a repair cost and the pipe is then replaced by polyurethane.
- Evaluate leaks throughout **2027–2051**, even if scheduled work finishes early.
- Every currently non-polyurethane pipe fully inside a circle is replaced and charged $200/m. Avoid unnecessary copper and brass replacement when it increases total cost.
- One structure or water pipe can make a large circle expensive because its surface rate applies to the whole area.
- The `0.85` exponent and $100 charge can favour grouping, but larger circles may add area, expensive surface types and extra replacement length. **Fewer, bigger circles are not always cheaper.**
- Prioritize expected **avoidable repair cost**, using both leak probability and repair severity.

Confirm any unspecified details with the instructor or official evaluator, especially the ordering of leaks and work at exactly the start of a year and whether all pipe geometry is represented by straight segments.

## Baseline

Start with one circle per original iron pipe.

For a straight pipe:

```text
centre = midpoint of the two endpoints
radius = half the pipe length
```

Use a small, documented numerical margin if necessary, then recheck containment.

This gives geometric coverage, but **budget feasibility must still be checked**. Baseline circles can contain other pipes, including copper, brass and expensive surface types.

Order the baseline using estimated avoidable repair costs, then evaluate it with the simulator.

## Scorer and validation — first priority

Build the cost function, simulator and validator before optimizing circles.

Check:

- Full containment, boundary cases and overlapping circles.
- Distinct surface counting, including surfaces from polyurethane pipes.
- Historical leaks and material status at the start of 2027.
- Project-window leaks and their effects on later costs.
- Zero-cost skipped units.
- Exact-budget execution and budget carryover.
- Stopping at the first unaffordable work unit.
- Complete geometric coverage of every original iron pipe.
- Completion by 2051.
- Correct parsing and validation of the exported submission CSV.

For each candidate, report:

- Total work-unit costs.
- Estimated repair costs.
- Estimated score.
- Completion year.
- Uncovered iron count.
- Model/data version and random seed.

Unknown future leaks mean our score is an **estimate**, not the official future score. Compare candidates using the same leak scenarios and record uncertainty.

Also run a no-future-leak simulation as a construction-budget check.

## Data and backtesting

- Preserve pipe IDs as strings.
- Check duplicate records, missing values, dates, units and material/surface labels.
- Verify joins between `pipes.csv` and `train.csv`.
- Keep raw data unchanged.
- Fit models on **2019–2023** and evaluate on **2024–2026**.
- Reconstruct pipe status at each cutoff using only information available at that time.
- Do not use future leaks, future repair costs or future replacement status as training features.

Historical backtesting evaluates predictions. It does not directly observe the score of a proposed 2027–2051 replacement plan.

## Roles

| # | Role | Person |
| --- | --- | --- |
| 1 | Data cleaning and exploratory analysis | |
| 2 | Leak probability model (per pipe, per year) | |
| 3 | Leak cost model and backtest (fit on 2019–2023, test on 2024–2026) | |
| 4 | Cost function, scorer and validator (**first priority**) | |
| 5 | Circle design (grouping pipes cheaply) | |
| 6 | Ordering within the budget, final submission | |

## Steps

1. **Week 1:** everyone reads the specification. Build the scorer and validator, clean the data, and make a simple leak-risk estimate.
2. **Week 2:** create a baseline submission, check coverage and budget feasibility, and estimate its score.
3. **Weeks 3–4:** improve the risk model, circles and ordering. Keep changes supported by validation that outperform the best candidate so far.
4. **Week 5:** pick the final submission, have someone else independently recheck the exported CSV, and write the report.

Work on branches and open pull requests into `main`.

Record each candidate's method, model/data version, random seed, costs, completion year and validation result so results can be reproduced.
