# STAT 440 – Star Kingdom Pipe Replacement

Replace every iron pipe with polyurethane between 2027 and 2051, with $10M per year. We submit an ordered list of circular work units, one per line, in the form `"(x, y)",r`.

**Score = ($250M − work-unit costs) − leak repair costs during 2027–2051**

## Layout

```
data/          pipes.csv, train.csv (leak history 2019–2026). Don't edit these.
notebooks/     exploration, named yourname_topic
src/           shared code
submissions/   submission CSVs, named YYYY-MM-DD_description.csv
```

Setup: create a `.venv` and install the packages in `requirements.txt`.

## Things to know

- Surplus doesn't depend on order. **Order only matters for leaks**, so replace risky pipes first.
- Every non-polyurethane pipe fully inside a circle is replaced and charged $200/m, so **keep copper and brass out of circles**.
- The surface rate sums each surface type present once, and applies to the **whole circle area**. One structure or water pipe makes a big circle expensive.
- The ^0.85 exponent and the $100 charge per unit make fewer, bigger circles cheaper, up to a point.
- Pipes that leaked in `train.csv` are already polyurethane.
- Simplest valid plan: one small circle around each iron pipe. That's our baseline.

## Roles

| # | Role | Person |
|---|---|---|
| 1 | Data cleaning and exploratory analysis | |
| 2 | Leak probability model (per pipe, per year) | |
| 3 | Leak cost model and backtest (fit on 2019–2023, test on 2024–2026) | |
| 4 | Cost function and scorer (**first priority**) | |
| 5 | Circle design (grouping pipes cheaply) | |
| 6 | Ordering within the budget, final submission | |

## Steps

1. **Week 1:** everyone reads the spec. Build the scorer, clean the data, and make a simple leak-risk estimate.
2. **Week 2:** baseline submission (one circle per pipe, riskiest first), scored and valid.
3. **Weeks 3–4:** improve the risk model, circles and ordering. Keep a change only if it beats the best score so far.
4. **Week 5:** pick the final submission, have someone else re-check its validity, and write the report.

Work on branches and open pull requests into `main`.
