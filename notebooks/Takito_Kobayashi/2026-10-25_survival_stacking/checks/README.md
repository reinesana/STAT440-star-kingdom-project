# Preflight and independent checks

The AFT likelihoods are independently checked against lifelines; MTLR and PMF
likelihood indexing are checked against pycox. These synthetic checks are not
model-evaluation results.

`preflight_failure/` preserves the initial interface bug investigation: a
single-time-point StepFunction returned a scalar; this was corrected and a
single-versus-multiple-time regression check added before final studies.

`preliminary_one_year_only/` preserves an abandoned preflight contract. Before
any outer-year evaluation, selection was changed to also assess 2-year failure
probability where those labels were already available in the inner window.
Those studies are not used for final selections. Their local SQLite/log files
are ignored; only the old contract is retained as provenance.
