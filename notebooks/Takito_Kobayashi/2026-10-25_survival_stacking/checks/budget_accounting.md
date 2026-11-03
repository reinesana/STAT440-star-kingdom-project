# Operational correction: queued candidates versus attempted trials

The initial runner subtracted all five enqueued candidates before calling
Optuna optimize(n_trials=...). Optuna consumes queued candidates within that
budget, so the initial pass attempted 35/55 rather than the intended 40/60.
This was detected from study counts, before inspecting outer evaluation scores.

The runner now subtracts only non-WAITING trials and checks actual attempted
counts before treating a completed origin as finished. The same studies can
receive their five missing trials. Likelihoods, features, search ranges,
selection objectives, input data and outer labels are unchanged. The old
contract is retained when advancing the operational code hash. Final exports
must be regenerated after each origin reaches its full intended budget.
