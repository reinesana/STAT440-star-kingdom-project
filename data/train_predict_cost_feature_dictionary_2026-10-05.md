# Cost feature dictionary — additions on 2026-10-05

The 35 original columns of `train_predict_cost.csv` are unchanged. The following
501 columns are candidate features, not a universally recommended model input.
Rows and original values are unchanged (14,107 events; 536 total columns).
Original data/train.csv and pipes.csv remain unchanged.

All historical statistics exclude the current and future calendar years and
are frozen at January 1. Costs use the dataset's original monetary unit. Distances
and endpoint lengths are metres. Neighbours exclude the focal pipe and require
a known lay date no later than the event. Missing historical statistics are blank;
zero counts mean no recorded events. Means/tail rates use the documented prior
smoothing; nearest-event means, medians/maxima/std are not smoothed.

Historical cost-group features use the original 14,107 severity rows. The explicit
repair-state and route/section histories use all 14,113 raw train.csv records.
The inventory_* repair columns retain the archived severity-row history definition.
Do not treat event counts as exposure-adjusted leak probabilities. Route per-km
counts adjust for length but not time at risk or changes in replacement status.

Endpoint routes use exact coordinate equality (0m snapping tolerance), split at
branches/dead ends, and do not connect interior crossings. All known inventory lay
dates precede 2019 (latest 1985-12-28); 15 unknown-date pipes are excluded from
historical topology. They remain in pipe_route_lookup.csv with topology_eligible=0
and blank route fields. Physical connectivity, depth and a real continuous asset
identity are not established by this inferred geometry.

Direction change is 0 degrees for straight continuation. Local is_straight uses
<=10 degrees and is_bend uses >10 degrees at degree-2 endpoints. Straight sections
split when a member's direction differs by >10 degrees from the section's initial
direction, so accumulated small turns also split sections. Route_is_curved is 1
for accumulated turn >=30 degrees, length/end-distance >1.01, or a closed route.
A local straight flag and a curved route flag may both be 1. Isolated segments
have no continuation and neither local flag. Closed routes have blank tortuosity.
Closed-route straight sections do not merge across the deterministic start seam.

route_position measures distance to the segment midpoint from a deterministic
route endpoint (lexicographically smallest endpoint; same convention for loops).
The original endpoint order is retained for endpoint_degree_1/2. Route and section
IDs are stable grouping strings based on their smallest member Pipe ID, not
ordinal predictors. They change if inventory/topology changes. Use IDs for
grouping, validation and mapping; select numerical group summaries separately.

Event dates/hours are recorded incident attributes. For future cost scenarios,
specify a scenario date/time and recompute age/calendar/history features. Their
existence does not mean future leak times are known. Endpoint sums, segment
proximity and projected overlaps are not actual curved/excavation lengths, measured
surface areas, confirmed physical connections or observed incident mechanisms.
No model retraining or improvement claim accompanies this data extension.

## Added columns

| Column | Definition | Source / availability |
|---|---|---|
| `age_x_wet_proximity` | Pipe age * ln(1 + inventory_25m_water_count + inventory_25m_swamp_count). | Inventory/event attributes or endpoint topology |
| `event_hour` | Integer recorded event hour, 0 through 23. | Inventory/event attributes or endpoint topology |
| `event_weekday` | Event weekday: Monday=0 through Sunday=6. | Inventory/event attributes or endpoint topology |
| `event_weekend` | 1 when the event occurs on Saturday or Sunday. | Inventory/event attributes or endpoint topology |
| `hist_age_band_all_max` | Maximum cost for same age band (20/40/60/80 years), using all preceding years. | Prior-year severity history |
| `hist_age_band_all_max_log1p` | Maximum cost for same age band (20/40/60/80 years), using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_age_band_all_mean` | Mean cost for same age band (20/40/60/80 years), using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_age_band_all_mean_log1p` | Mean cost for same age band (20/40/60/80 years), using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_age_band_all_median` | Median cost for same age band (20/40/60/80 years), using all preceding years. | Prior-year severity history |
| `hist_age_band_all_median_log1p` | Median cost for same age band (20/40/60/80 years), using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_age_band_all_n` | Recorded event count for same age band (20/40/60/80 years), using all preceding years. | Prior-year severity history |
| `hist_age_band_all_std` | Sample standard deviation of cost for same age band (20/40/60/80 years), using all preceding years. | Prior-year severity history |
| `hist_age_band_all_std_log1p` | Sample standard deviation of cost for same age band (20/40/60/80 years), using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_age_band_all_tail_rate` | Fraction of high-cost events for same age band (20/40/60/80 years), using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_age_band_last1_max` | Maximum cost for same age band (20/40/60/80 years), using previous calendar year. | Prior-year severity history |
| `hist_age_band_last1_max_log1p` | Maximum cost for same age band (20/40/60/80 years), using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_age_band_last1_mean` | Mean cost for same age band (20/40/60/80 years), using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_age_band_last1_mean_log1p` | Mean cost for same age band (20/40/60/80 years), using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_age_band_last1_median` | Median cost for same age band (20/40/60/80 years), using previous calendar year. | Prior-year severity history |
| `hist_age_band_last1_median_log1p` | Median cost for same age band (20/40/60/80 years), using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_age_band_last1_n` | Recorded event count for same age band (20/40/60/80 years), using previous calendar year. | Prior-year severity history |
| `hist_age_band_last1_std` | Sample standard deviation of cost for same age band (20/40/60/80 years), using previous calendar year. | Prior-year severity history |
| `hist_age_band_last1_std_log1p` | Sample standard deviation of cost for same age band (20/40/60/80 years), using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_age_band_last1_tail_rate` | Fraction of high-cost events for same age band (20/40/60/80 years), using previous calendar year. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_age_band_last2_max` | Maximum cost for same age band (20/40/60/80 years), using previous two calendar years. | Prior-year severity history |
| `hist_age_band_last2_max_log1p` | Maximum cost for same age band (20/40/60/80 years), using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_age_band_last2_mean` | Mean cost for same age band (20/40/60/80 years), using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_age_band_last2_mean_log1p` | Mean cost for same age band (20/40/60/80 years), using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_age_band_last2_median` | Median cost for same age band (20/40/60/80 years), using previous two calendar years. | Prior-year severity history |
| `hist_age_band_last2_median_log1p` | Median cost for same age band (20/40/60/80 years), using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_age_band_last2_n` | Recorded event count for same age band (20/40/60/80 years), using previous two calendar years. | Prior-year severity history |
| `hist_age_band_last2_std` | Sample standard deviation of cost for same age band (20/40/60/80 years), using previous two calendar years. | Prior-year severity history |
| `hist_age_band_last2_std_log1p` | Sample standard deviation of cost for same age band (20/40/60/80 years), using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_age_band_last2_tail_rate` | Fraction of high-cost events for same age band (20/40/60/80 years), using previous two calendar years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_grid1000_all_max` | Maximum cost for same 1000m square grid cell, using all preceding years. | Prior-year severity history |
| `hist_grid1000_all_max_log1p` | Maximum cost for same 1000m square grid cell, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid1000_all_mean` | Mean cost for same 1000m square grid cell, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_grid1000_all_mean_log1p` | Mean cost for same 1000m square grid cell, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid1000_all_median` | Median cost for same 1000m square grid cell, using all preceding years. | Prior-year severity history |
| `hist_grid1000_all_median_log1p` | Median cost for same 1000m square grid cell, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid1000_all_n` | Recorded event count for same 1000m square grid cell, using all preceding years. | Prior-year severity history |
| `hist_grid1000_all_std` | Sample standard deviation of cost for same 1000m square grid cell, using all preceding years. | Prior-year severity history |
| `hist_grid1000_all_std_log1p` | Sample standard deviation of cost for same 1000m square grid cell, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid1000_all_tail_rate` | Fraction of high-cost events for same 1000m square grid cell, using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_grid1000_last1_max` | Maximum cost for same 1000m square grid cell, using previous calendar year. | Prior-year severity history |
| `hist_grid1000_last1_max_log1p` | Maximum cost for same 1000m square grid cell, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid1000_last1_mean` | Mean cost for same 1000m square grid cell, using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_grid1000_last1_mean_log1p` | Mean cost for same 1000m square grid cell, using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid1000_last1_median` | Median cost for same 1000m square grid cell, using previous calendar year. | Prior-year severity history |
| `hist_grid1000_last1_median_log1p` | Median cost for same 1000m square grid cell, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid1000_last1_n` | Recorded event count for same 1000m square grid cell, using previous calendar year. | Prior-year severity history |
| `hist_grid1000_last1_std` | Sample standard deviation of cost for same 1000m square grid cell, using previous calendar year. | Prior-year severity history |
| `hist_grid1000_last1_std_log1p` | Sample standard deviation of cost for same 1000m square grid cell, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid1000_last1_tail_rate` | Fraction of high-cost events for same 1000m square grid cell, using previous calendar year. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_grid1000_last2_max` | Maximum cost for same 1000m square grid cell, using previous two calendar years. | Prior-year severity history |
| `hist_grid1000_last2_max_log1p` | Maximum cost for same 1000m square grid cell, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid1000_last2_mean` | Mean cost for same 1000m square grid cell, using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_grid1000_last2_mean_log1p` | Mean cost for same 1000m square grid cell, using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid1000_last2_median` | Median cost for same 1000m square grid cell, using previous two calendar years. | Prior-year severity history |
| `hist_grid1000_last2_median_log1p` | Median cost for same 1000m square grid cell, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid1000_last2_n` | Recorded event count for same 1000m square grid cell, using previous two calendar years. | Prior-year severity history |
| `hist_grid1000_last2_std` | Sample standard deviation of cost for same 1000m square grid cell, using previous two calendar years. | Prior-year severity history |
| `hist_grid1000_last2_std_log1p` | Sample standard deviation of cost for same 1000m square grid cell, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid1000_last2_tail_rate` | Fraction of high-cost events for same 1000m square grid cell, using previous two calendar years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_grid250_all_max` | Maximum cost for same 250m square grid cell, using all preceding years. | Prior-year severity history |
| `hist_grid250_all_max_log1p` | Maximum cost for same 250m square grid cell, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid250_all_mean` | Mean cost for same 250m square grid cell, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_grid250_all_mean_log1p` | Mean cost for same 250m square grid cell, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid250_all_median` | Median cost for same 250m square grid cell, using all preceding years. | Prior-year severity history |
| `hist_grid250_all_median_log1p` | Median cost for same 250m square grid cell, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid250_all_n` | Recorded event count for same 250m square grid cell, using all preceding years. | Prior-year severity history |
| `hist_grid250_all_std` | Sample standard deviation of cost for same 250m square grid cell, using all preceding years. | Prior-year severity history |
| `hist_grid250_all_std_log1p` | Sample standard deviation of cost for same 250m square grid cell, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid250_all_tail_rate` | Fraction of high-cost events for same 250m square grid cell, using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_grid250_last1_max` | Maximum cost for same 250m square grid cell, using previous calendar year. | Prior-year severity history |
| `hist_grid250_last1_max_log1p` | Maximum cost for same 250m square grid cell, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid250_last1_mean` | Mean cost for same 250m square grid cell, using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_grid250_last1_mean_log1p` | Mean cost for same 250m square grid cell, using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid250_last1_median` | Median cost for same 250m square grid cell, using previous calendar year. | Prior-year severity history |
| `hist_grid250_last1_median_log1p` | Median cost for same 250m square grid cell, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid250_last1_n` | Recorded event count for same 250m square grid cell, using previous calendar year. | Prior-year severity history |
| `hist_grid250_last1_std` | Sample standard deviation of cost for same 250m square grid cell, using previous calendar year. | Prior-year severity history |
| `hist_grid250_last1_std_log1p` | Sample standard deviation of cost for same 250m square grid cell, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid250_last1_tail_rate` | Fraction of high-cost events for same 250m square grid cell, using previous calendar year. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_grid250_last2_max` | Maximum cost for same 250m square grid cell, using previous two calendar years. | Prior-year severity history |
| `hist_grid250_last2_max_log1p` | Maximum cost for same 250m square grid cell, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid250_last2_mean` | Mean cost for same 250m square grid cell, using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_grid250_last2_mean_log1p` | Mean cost for same 250m square grid cell, using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid250_last2_median` | Median cost for same 250m square grid cell, using previous two calendar years. | Prior-year severity history |
| `hist_grid250_last2_median_log1p` | Median cost for same 250m square grid cell, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid250_last2_n` | Recorded event count for same 250m square grid cell, using previous two calendar years. | Prior-year severity history |
| `hist_grid250_last2_std` | Sample standard deviation of cost for same 250m square grid cell, using previous two calendar years. | Prior-year severity history |
| `hist_grid250_last2_std_log1p` | Sample standard deviation of cost for same 250m square grid cell, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid250_last2_tail_rate` | Fraction of high-cost events for same 250m square grid cell, using previous two calendar years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_grid500_all_max` | Maximum cost for same 500m square grid cell, using all preceding years. | Prior-year severity history |
| `hist_grid500_all_max_log1p` | Maximum cost for same 500m square grid cell, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid500_all_mean` | Mean cost for same 500m square grid cell, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_grid500_all_mean_log1p` | Mean cost for same 500m square grid cell, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid500_all_median` | Median cost for same 500m square grid cell, using all preceding years. | Prior-year severity history |
| `hist_grid500_all_median_log1p` | Median cost for same 500m square grid cell, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid500_all_n` | Recorded event count for same 500m square grid cell, using all preceding years. | Prior-year severity history |
| `hist_grid500_all_std` | Sample standard deviation of cost for same 500m square grid cell, using all preceding years. | Prior-year severity history |
| `hist_grid500_all_std_log1p` | Sample standard deviation of cost for same 500m square grid cell, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid500_all_tail_rate` | Fraction of high-cost events for same 500m square grid cell, using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_grid500_last1_max` | Maximum cost for same 500m square grid cell, using previous calendar year. | Prior-year severity history |
| `hist_grid500_last1_max_log1p` | Maximum cost for same 500m square grid cell, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid500_last1_mean` | Mean cost for same 500m square grid cell, using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_grid500_last1_mean_log1p` | Mean cost for same 500m square grid cell, using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid500_last1_median` | Median cost for same 500m square grid cell, using previous calendar year. | Prior-year severity history |
| `hist_grid500_last1_median_log1p` | Median cost for same 500m square grid cell, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid500_last1_n` | Recorded event count for same 500m square grid cell, using previous calendar year. | Prior-year severity history |
| `hist_grid500_last1_std` | Sample standard deviation of cost for same 500m square grid cell, using previous calendar year. | Prior-year severity history |
| `hist_grid500_last1_std_log1p` | Sample standard deviation of cost for same 500m square grid cell, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid500_last1_tail_rate` | Fraction of high-cost events for same 500m square grid cell, using previous calendar year. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_grid500_last2_max` | Maximum cost for same 500m square grid cell, using previous two calendar years. | Prior-year severity history |
| `hist_grid500_last2_max_log1p` | Maximum cost for same 500m square grid cell, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid500_last2_mean` | Mean cost for same 500m square grid cell, using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_grid500_last2_mean_log1p` | Mean cost for same 500m square grid cell, using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid500_last2_median` | Median cost for same 500m square grid cell, using previous two calendar years. | Prior-year severity history |
| `hist_grid500_last2_median_log1p` | Median cost for same 500m square grid cell, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid500_last2_n` | Recorded event count for same 500m square grid cell, using previous two calendar years. | Prior-year severity history |
| `hist_grid500_last2_std` | Sample standard deviation of cost for same 500m square grid cell, using previous two calendar years. | Prior-year severity history |
| `hist_grid500_last2_std_log1p` | Sample standard deviation of cost for same 500m square grid cell, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_grid500_last2_tail_rate` | Fraction of high-cost events for same 500m square grid cell, using previous two calendar years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_length_band_all_max` | Maximum cost for same length band (10/25/50/100m), using all preceding years. | Prior-year severity history |
| `hist_length_band_all_max_log1p` | Maximum cost for same length band (10/25/50/100m), using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_length_band_all_mean` | Mean cost for same length band (10/25/50/100m), using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_length_band_all_mean_log1p` | Mean cost for same length band (10/25/50/100m), using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_length_band_all_median` | Median cost for same length band (10/25/50/100m), using all preceding years. | Prior-year severity history |
| `hist_length_band_all_median_log1p` | Median cost for same length band (10/25/50/100m), using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_length_band_all_n` | Recorded event count for same length band (10/25/50/100m), using all preceding years. | Prior-year severity history |
| `hist_length_band_all_std` | Sample standard deviation of cost for same length band (10/25/50/100m), using all preceding years. | Prior-year severity history |
| `hist_length_band_all_std_log1p` | Sample standard deviation of cost for same length band (10/25/50/100m), using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_length_band_all_tail_rate` | Fraction of high-cost events for same length band (10/25/50/100m), using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_length_band_last1_max` | Maximum cost for same length band (10/25/50/100m), using previous calendar year. | Prior-year severity history |
| `hist_length_band_last1_max_log1p` | Maximum cost for same length band (10/25/50/100m), using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_length_band_last1_mean` | Mean cost for same length band (10/25/50/100m), using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_length_band_last1_mean_log1p` | Mean cost for same length band (10/25/50/100m), using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_length_band_last1_median` | Median cost for same length band (10/25/50/100m), using previous calendar year. | Prior-year severity history |
| `hist_length_band_last1_median_log1p` | Median cost for same length band (10/25/50/100m), using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_length_band_last1_n` | Recorded event count for same length band (10/25/50/100m), using previous calendar year. | Prior-year severity history |
| `hist_length_band_last1_std` | Sample standard deviation of cost for same length band (10/25/50/100m), using previous calendar year. | Prior-year severity history |
| `hist_length_band_last1_std_log1p` | Sample standard deviation of cost for same length band (10/25/50/100m), using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_length_band_last1_tail_rate` | Fraction of high-cost events for same length band (10/25/50/100m), using previous calendar year. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_length_band_last2_max` | Maximum cost for same length band (10/25/50/100m), using previous two calendar years. | Prior-year severity history |
| `hist_length_band_last2_max_log1p` | Maximum cost for same length band (10/25/50/100m), using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_length_band_last2_mean` | Mean cost for same length band (10/25/50/100m), using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_length_band_last2_mean_log1p` | Mean cost for same length band (10/25/50/100m), using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_length_band_last2_median` | Median cost for same length band (10/25/50/100m), using previous two calendar years. | Prior-year severity history |
| `hist_length_band_last2_median_log1p` | Median cost for same length band (10/25/50/100m), using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_length_band_last2_n` | Recorded event count for same length band (10/25/50/100m), using previous two calendar years. | Prior-year severity history |
| `hist_length_band_last2_std` | Sample standard deviation of cost for same length band (10/25/50/100m), using previous two calendar years. | Prior-year severity history |
| `hist_length_band_last2_std_log1p` | Sample standard deviation of cost for same length band (10/25/50/100m), using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_length_band_last2_tail_rate` | Fraction of high-cost events for same length band (10/25/50/100m), using previous two calendar years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_material_all_max` | Maximum cost for same material, using all preceding years. | Prior-year severity history |
| `hist_material_all_max_log1p` | Maximum cost for same material, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_all_mean` | Mean cost for same material, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_material_all_mean_log1p` | Mean cost for same material, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_all_median` | Median cost for same material, using all preceding years. | Prior-year severity history |
| `hist_material_all_median_log1p` | Median cost for same material, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_all_n` | Recorded event count for same material, using all preceding years. | Prior-year severity history |
| `hist_material_all_std` | Sample standard deviation of cost for same material, using all preceding years. | Prior-year severity history |
| `hist_material_all_std_log1p` | Sample standard deviation of cost for same material, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_all_tail_rate` | Fraction of high-cost events for same material, using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_material_last1_max` | Maximum cost for same material, using previous calendar year. | Prior-year severity history |
| `hist_material_last1_max_log1p` | Maximum cost for same material, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_last1_mean` | Mean cost for same material, using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_material_last1_mean_log1p` | Mean cost for same material, using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_last1_median` | Median cost for same material, using previous calendar year. | Prior-year severity history |
| `hist_material_last1_median_log1p` | Median cost for same material, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_last1_n` | Recorded event count for same material, using previous calendar year. | Prior-year severity history |
| `hist_material_last1_std` | Sample standard deviation of cost for same material, using previous calendar year. | Prior-year severity history |
| `hist_material_last1_std_log1p` | Sample standard deviation of cost for same material, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_last1_tail_rate` | Fraction of high-cost events for same material, using previous calendar year. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_material_last2_max` | Maximum cost for same material, using previous two calendar years. | Prior-year severity history |
| `hist_material_last2_max_log1p` | Maximum cost for same material, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_last2_mean` | Mean cost for same material, using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_material_last2_mean_log1p` | Mean cost for same material, using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_last2_median` | Median cost for same material, using previous two calendar years. | Prior-year severity history |
| `hist_material_last2_median_log1p` | Median cost for same material, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_last2_n` | Recorded event count for same material, using previous two calendar years. | Prior-year severity history |
| `hist_material_last2_std` | Sample standard deviation of cost for same material, using previous two calendar years. | Prior-year severity history |
| `hist_material_last2_std_log1p` | Sample standard deviation of cost for same material, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_last2_tail_rate` | Fraction of high-cost events for same material, using previous two calendar years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_material_surface_all_max` | Maximum cost for same material/surface pair, using all preceding years. | Prior-year severity history |
| `hist_material_surface_all_max_log1p` | Maximum cost for same material/surface pair, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_surface_all_mean` | Mean cost for same material/surface pair, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_material_surface_all_mean_log1p` | Mean cost for same material/surface pair, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_surface_all_median` | Median cost for same material/surface pair, using all preceding years. | Prior-year severity history |
| `hist_material_surface_all_median_log1p` | Median cost for same material/surface pair, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_surface_all_n` | Recorded event count for same material/surface pair, using all preceding years. | Prior-year severity history |
| `hist_material_surface_all_std` | Sample standard deviation of cost for same material/surface pair, using all preceding years. | Prior-year severity history |
| `hist_material_surface_all_std_log1p` | Sample standard deviation of cost for same material/surface pair, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_surface_all_tail_rate` | Fraction of high-cost events for same material/surface pair, using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_material_surface_last1_max` | Maximum cost for same material/surface pair, using previous calendar year. | Prior-year severity history |
| `hist_material_surface_last1_max_log1p` | Maximum cost for same material/surface pair, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_surface_last1_mean` | Mean cost for same material/surface pair, using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_material_surface_last1_mean_log1p` | Mean cost for same material/surface pair, using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_surface_last1_median` | Median cost for same material/surface pair, using previous calendar year. | Prior-year severity history |
| `hist_material_surface_last1_median_log1p` | Median cost for same material/surface pair, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_surface_last1_n` | Recorded event count for same material/surface pair, using previous calendar year. | Prior-year severity history |
| `hist_material_surface_last1_std` | Sample standard deviation of cost for same material/surface pair, using previous calendar year. | Prior-year severity history |
| `hist_material_surface_last1_std_log1p` | Sample standard deviation of cost for same material/surface pair, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_surface_last1_tail_rate` | Fraction of high-cost events for same material/surface pair, using previous calendar year. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_material_surface_last2_max` | Maximum cost for same material/surface pair, using previous two calendar years. | Prior-year severity history |
| `hist_material_surface_last2_max_log1p` | Maximum cost for same material/surface pair, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_surface_last2_mean` | Mean cost for same material/surface pair, using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_material_surface_last2_mean_log1p` | Mean cost for same material/surface pair, using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_surface_last2_median` | Median cost for same material/surface pair, using previous two calendar years. | Prior-year severity history |
| `hist_material_surface_last2_median_log1p` | Median cost for same material/surface pair, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_surface_last2_n` | Recorded event count for same material/surface pair, using previous two calendar years. | Prior-year severity history |
| `hist_material_surface_last2_std` | Sample standard deviation of cost for same material/surface pair, using previous two calendar years. | Prior-year severity history |
| `hist_material_surface_last2_std_log1p` | Sample standard deviation of cost for same material/surface pair, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_material_surface_last2_tail_rate` | Fraction of high-cost events for same material/surface pair, using previous two calendar years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_nearest10_max` | Maximum cost for nearest 10 prior event pipe midpoints, using all preceding years. | Prior-year severity history |
| `hist_nearest10_max_log1p` | Maximum cost for nearest 10 prior event pipe midpoints, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_nearest10_mean` | Mean cost for nearest 10 prior event pipe midpoints, using all preceding years. | Prior-year severity history |
| `hist_nearest10_mean_log1p` | Mean cost for nearest 10 prior event pipe midpoints, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_nearest10_median` | Median cost for nearest 10 prior event pipe midpoints, using all preceding years. | Prior-year severity history |
| `hist_nearest10_median_log1p` | Median cost for nearest 10 prior event pipe midpoints, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_nearest25_max` | Maximum cost for nearest 25 prior event pipe midpoints, using all preceding years. | Prior-year severity history |
| `hist_nearest25_max_log1p` | Maximum cost for nearest 25 prior event pipe midpoints, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_nearest25_mean` | Mean cost for nearest 25 prior event pipe midpoints, using all preceding years. | Prior-year severity history |
| `hist_nearest25_mean_log1p` | Mean cost for nearest 25 prior event pipe midpoints, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_nearest25_median` | Median cost for nearest 25 prior event pipe midpoints, using all preceding years. | Prior-year severity history |
| `hist_nearest25_median_log1p` | Median cost for nearest 25 prior event pipe midpoints, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_nearest5_max` | Maximum cost for nearest 5 prior event pipe midpoints, using all preceding years. | Prior-year severity history |
| `hist_nearest5_max_log1p` | Maximum cost for nearest 5 prior event pipe midpoints, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_nearest5_mean` | Mean cost for nearest 5 prior event pipe midpoints, using all preceding years. | Prior-year severity history |
| `hist_nearest5_mean_log1p` | Mean cost for nearest 5 prior event pipe midpoints, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_nearest5_median` | Median cost for nearest 5 prior event pipe midpoints, using all preceding years. | Prior-year severity history |
| `hist_nearest5_median_log1p` | Median cost for nearest 5 prior event pipe midpoints, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_radius100_max` | Maximum cost for events within 100m midpoint distance, using all preceding years. | Prior-year severity history |
| `hist_radius100_max_log1p` | Maximum cost for events within 100m midpoint distance, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_radius100_mean` | Mean cost for events within 100m midpoint distance, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_radius100_mean_log1p` | Mean cost for events within 100m midpoint distance, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_radius100_median` | Median cost for events within 100m midpoint distance, using all preceding years. | Prior-year severity history |
| `hist_radius100_median_log1p` | Median cost for events within 100m midpoint distance, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_radius100_n` | Recorded event count for events within 100m midpoint distance, using all preceding years. | Prior-year severity history |
| `hist_radius100_tail_rate` | Fraction of high-cost events for events within 100m midpoint distance, using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_radius250_max` | Maximum cost for events within 250m midpoint distance, using all preceding years. | Prior-year severity history |
| `hist_radius250_max_log1p` | Maximum cost for events within 250m midpoint distance, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_radius250_mean` | Mean cost for events within 250m midpoint distance, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_radius250_mean_log1p` | Mean cost for events within 250m midpoint distance, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_radius250_median` | Median cost for events within 250m midpoint distance, using all preceding years. | Prior-year severity history |
| `hist_radius250_median_log1p` | Median cost for events within 250m midpoint distance, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_radius250_n` | Recorded event count for events within 250m midpoint distance, using all preceding years. | Prior-year severity history |
| `hist_radius250_tail_rate` | Fraction of high-cost events for events within 250m midpoint distance, using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_radius500_max` | Maximum cost for events within 500m midpoint distance, using all preceding years. | Prior-year severity history |
| `hist_radius500_max_log1p` | Maximum cost for events within 500m midpoint distance, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_radius500_mean` | Mean cost for events within 500m midpoint distance, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_radius500_mean_log1p` | Mean cost for events within 500m midpoint distance, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_radius500_median` | Median cost for events within 500m midpoint distance, using all preceding years. | Prior-year severity history |
| `hist_radius500_median_log1p` | Median cost for events within 500m midpoint distance, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_radius500_n` | Recorded event count for events within 500m midpoint distance, using all preceding years. | Prior-year severity history |
| `hist_radius500_tail_rate` | Fraction of high-cost events for events within 500m midpoint distance, using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_radius50_max` | Maximum cost for events within 50m midpoint distance, using all preceding years. | Prior-year severity history |
| `hist_radius50_max_log1p` | Maximum cost for events within 50m midpoint distance, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_radius50_mean` | Mean cost for events within 50m midpoint distance, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_radius50_mean_log1p` | Mean cost for events within 50m midpoint distance, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_radius50_median` | Median cost for events within 50m midpoint distance, using all preceding years. | Prior-year severity history |
| `hist_radius50_median_log1p` | Median cost for events within 50m midpoint distance, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_radius50_n` | Recorded event count for events within 50m midpoint distance, using all preceding years. | Prior-year severity history |
| `hist_radius50_tail_rate` | Fraction of high-cost events for events within 50m midpoint distance, using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_segment100_all_max` | Maximum cost for segments within 100m shortest planar distance, using all preceding years. | Prior-year severity history |
| `hist_segment100_all_mean` | Mean cost for segments within 100m shortest planar distance, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_segment100_all_median` | Median cost for segments within 100m shortest planar distance, using all preceding years. | Prior-year severity history |
| `hist_segment100_all_n` | Recorded event count for segments within 100m shortest planar distance, using all preceding years. | Prior-year severity history |
| `hist_segment100_all_std` | Sample standard deviation of cost for segments within 100m shortest planar distance, using all preceding years. | Prior-year severity history |
| `hist_segment100_all_tail_rate` | Fraction of high-cost events for segments within 100m shortest planar distance, using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_segment250_all_max` | Maximum cost for segments within 250m shortest planar distance, using all preceding years. | Prior-year severity history |
| `hist_segment250_all_mean` | Mean cost for segments within 250m shortest planar distance, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_segment250_all_median` | Median cost for segments within 250m shortest planar distance, using all preceding years. | Prior-year severity history |
| `hist_segment250_all_n` | Recorded event count for segments within 250m shortest planar distance, using all preceding years. | Prior-year severity history |
| `hist_segment250_all_std` | Sample standard deviation of cost for segments within 250m shortest planar distance, using all preceding years. | Prior-year severity history |
| `hist_segment250_all_tail_rate` | Fraction of high-cost events for segments within 250m shortest planar distance, using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_segment25_all_max` | Maximum cost for segments within 25m shortest planar distance, using all preceding years. | Prior-year severity history |
| `hist_segment25_all_mean` | Mean cost for segments within 25m shortest planar distance, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_segment25_all_median` | Median cost for segments within 25m shortest planar distance, using all preceding years. | Prior-year severity history |
| `hist_segment25_all_n` | Recorded event count for segments within 25m shortest planar distance, using all preceding years. | Prior-year severity history |
| `hist_segment25_all_std` | Sample standard deviation of cost for segments within 25m shortest planar distance, using all preceding years. | Prior-year severity history |
| `hist_segment25_all_tail_rate` | Fraction of high-cost events for segments within 25m shortest planar distance, using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_surface_all_max` | Maximum cost for same surface, using all preceding years. | Prior-year severity history |
| `hist_surface_all_max_log1p` | Maximum cost for same surface, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_surface_all_mean` | Mean cost for same surface, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_surface_all_mean_log1p` | Mean cost for same surface, using all preceding years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_surface_all_median` | Median cost for same surface, using all preceding years. | Prior-year severity history |
| `hist_surface_all_median_log1p` | Median cost for same surface, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_surface_all_n` | Recorded event count for same surface, using all preceding years. | Prior-year severity history |
| `hist_surface_all_std` | Sample standard deviation of cost for same surface, using all preceding years. | Prior-year severity history |
| `hist_surface_all_std_log1p` | Sample standard deviation of cost for same surface, using all preceding years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_surface_all_tail_rate` | Fraction of high-cost events for same surface, using all preceding years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_surface_last1_max` | Maximum cost for same surface, using previous calendar year. | Prior-year severity history |
| `hist_surface_last1_max_log1p` | Maximum cost for same surface, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_surface_last1_mean` | Mean cost for same surface, using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_surface_last1_mean_log1p` | Mean cost for same surface, using previous calendar year. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_surface_last1_median` | Median cost for same surface, using previous calendar year. | Prior-year severity history |
| `hist_surface_last1_median_log1p` | Median cost for same surface, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_surface_last1_n` | Recorded event count for same surface, using previous calendar year. | Prior-year severity history |
| `hist_surface_last1_std` | Sample standard deviation of cost for same surface, using previous calendar year. | Prior-year severity history |
| `hist_surface_last1_std_log1p` | Sample standard deviation of cost for same surface, using previous calendar year. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_surface_last1_tail_rate` | Fraction of high-cost events for same surface, using previous calendar year. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hist_surface_last2_max` | Maximum cost for same surface, using previous two calendar years. | Prior-year severity history |
| `hist_surface_last2_max_log1p` | Maximum cost for same surface, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_surface_last2_mean` | Mean cost for same surface, using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). | Prior-year severity history |
| `hist_surface_last2_mean_log1p` | Mean cost for same surface, using previous two calendar years. Smoothed as (local cost sum + 20 * prior mean)/(count + 20). Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_surface_last2_median` | Median cost for same surface, using previous two calendar years. | Prior-year severity history |
| `hist_surface_last2_median_log1p` | Median cost for same surface, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_surface_last2_n` | Recorded event count for same surface, using previous two calendar years. | Prior-year severity history |
| `hist_surface_last2_std` | Sample standard deviation of cost for same surface, using previous two calendar years. | Prior-year severity history |
| `hist_surface_last2_std_log1p` | Sample standard deviation of cost for same surface, using previous two calendar years. Apply ln(1+x) to the statistic. | Prior-year severity history |
| `hist_surface_last2_tail_rate` | Fraction of high-cost events for same surface, using previous two calendar years. Threshold is the 99th percentile of all earlier-year costs; rate uses 50 prior pseudo-observations. | Prior-year severity history |
| `hour_cos` | cos(2*pi*event_hour/24). | Recorded event calendar/time |
| `hour_sin` | sin(2*pi*event_hour/24). | Recorded event calendar/time |
| `inventory_100m_age_mean` | Within 100m shortest segment distance: mean age at the event date (years). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_100m_count` | Within 100m shortest segment distance: number of other pipes. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_100m_days_since_repair` | Within 100m shortest segment distance: days from latest previous repair to January 1. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_100m_iron_fraction` | Within 100m shortest segment distance: fraction originally iron. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_100m_length_sum` | Within 100m shortest segment distance: sum of whole endpoint lengths (m). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_100m_parallel_count` | Within 100m shortest segment distance: number of nearly parallel pipes (within 15 degrees, positive projected overlap). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_100m_parallel_projected_length` | Within 100m shortest segment distance: sum of projected overlap lengths for nearly parallel pipes (m). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_100m_prior_repair_fraction` | Within 100m shortest segment distance: fraction with a prior recorded repair. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_100m_prior_repairs` | Within 100m shortest segment distance: count of prior recorded repairs. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_100m_recent_repairs` | Within 100m shortest segment distance: count repaired in the 365 days preceding January 1. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_100m_road_count` | Within 100m shortest segment distance: road-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_100m_same_material_fraction` | Within 100m shortest segment distance: fraction sharing the focal original material. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_100m_structure_count` | Within 100m shortest segment distance: structure-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_100m_surface_diversity` | Within 100m shortest segment distance: number of distinct surface labels. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_100m_swamp_count` | Within 100m shortest segment distance: swamp-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_100m_water_count` | Within 100m shortest segment distance: water-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_10m_age_mean` | Within 10m shortest segment distance: mean age at the event date (years). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_10m_count` | Within 10m shortest segment distance: number of other pipes. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_10m_days_since_repair` | Within 10m shortest segment distance: days from latest previous repair to January 1. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_10m_iron_fraction` | Within 10m shortest segment distance: fraction originally iron. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_10m_length_sum` | Within 10m shortest segment distance: sum of whole endpoint lengths (m). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_10m_parallel_count` | Within 10m shortest segment distance: number of nearly parallel pipes (within 15 degrees, positive projected overlap). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_10m_parallel_projected_length` | Within 10m shortest segment distance: sum of projected overlap lengths for nearly parallel pipes (m). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_10m_prior_repair_fraction` | Within 10m shortest segment distance: fraction with a prior recorded repair. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_10m_prior_repairs` | Within 10m shortest segment distance: count of prior recorded repairs. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_10m_recent_repairs` | Within 10m shortest segment distance: count repaired in the 365 days preceding January 1. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_10m_road_count` | Within 10m shortest segment distance: road-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_10m_same_material_fraction` | Within 10m shortest segment distance: fraction sharing the focal original material. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_10m_structure_count` | Within 10m shortest segment distance: structure-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_10m_surface_diversity` | Within 10m shortest segment distance: number of distinct surface labels. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_10m_swamp_count` | Within 10m shortest segment distance: swamp-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_10m_water_count` | Within 10m shortest segment distance: water-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_250m_age_mean` | Within 250m shortest segment distance: mean age at the event date (years). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_250m_count` | Within 250m shortest segment distance: number of other pipes. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_250m_days_since_repair` | Within 250m shortest segment distance: days from latest previous repair to January 1. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_250m_iron_fraction` | Within 250m shortest segment distance: fraction originally iron. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_250m_length_sum` | Within 250m shortest segment distance: sum of whole endpoint lengths (m). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_250m_parallel_count` | Within 250m shortest segment distance: number of nearly parallel pipes (within 15 degrees, positive projected overlap). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_250m_parallel_projected_length` | Within 250m shortest segment distance: sum of projected overlap lengths for nearly parallel pipes (m). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_250m_prior_repair_fraction` | Within 250m shortest segment distance: fraction with a prior recorded repair. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_250m_prior_repairs` | Within 250m shortest segment distance: count of prior recorded repairs. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_250m_recent_repairs` | Within 250m shortest segment distance: count repaired in the 365 days preceding January 1. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_250m_road_count` | Within 250m shortest segment distance: road-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_250m_same_material_fraction` | Within 250m shortest segment distance: fraction sharing the focal original material. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_250m_structure_count` | Within 250m shortest segment distance: structure-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_250m_surface_diversity` | Within 250m shortest segment distance: number of distinct surface labels. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_250m_swamp_count` | Within 250m shortest segment distance: swamp-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_250m_water_count` | Within 250m shortest segment distance: water-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_25m_age_mean` | Within 25m shortest segment distance: mean age at the event date (years). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_25m_count` | Within 25m shortest segment distance: number of other pipes. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_25m_days_since_repair` | Within 25m shortest segment distance: days from latest previous repair to January 1. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_25m_iron_fraction` | Within 25m shortest segment distance: fraction originally iron. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_25m_length_sum` | Within 25m shortest segment distance: sum of whole endpoint lengths (m). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_25m_parallel_count` | Within 25m shortest segment distance: number of nearly parallel pipes (within 15 degrees, positive projected overlap). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_25m_parallel_projected_length` | Within 25m shortest segment distance: sum of projected overlap lengths for nearly parallel pipes (m). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_25m_prior_repair_fraction` | Within 25m shortest segment distance: fraction with a prior recorded repair. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_25m_prior_repairs` | Within 25m shortest segment distance: count of prior recorded repairs. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_25m_recent_repairs` | Within 25m shortest segment distance: count repaired in the 365 days preceding January 1. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_25m_road_count` | Within 25m shortest segment distance: road-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_25m_same_material_fraction` | Within 25m shortest segment distance: fraction sharing the focal original material. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_25m_structure_count` | Within 25m shortest segment distance: structure-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_25m_surface_diversity` | Within 25m shortest segment distance: number of distinct surface labels. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_25m_swamp_count` | Within 25m shortest segment distance: swamp-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_25m_water_count` | Within 25m shortest segment distance: water-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_50m_age_mean` | Within 50m shortest segment distance: mean age at the event date (years). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_50m_count` | Within 50m shortest segment distance: number of other pipes. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_50m_days_since_repair` | Within 50m shortest segment distance: days from latest previous repair to January 1. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_50m_iron_fraction` | Within 50m shortest segment distance: fraction originally iron. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_50m_length_sum` | Within 50m shortest segment distance: sum of whole endpoint lengths (m). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_50m_parallel_count` | Within 50m shortest segment distance: number of nearly parallel pipes (within 15 degrees, positive projected overlap). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_50m_parallel_projected_length` | Within 50m shortest segment distance: sum of projected overlap lengths for nearly parallel pipes (m). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_50m_prior_repair_fraction` | Within 50m shortest segment distance: fraction with a prior recorded repair. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_50m_prior_repairs` | Within 50m shortest segment distance: count of prior recorded repairs. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_50m_recent_repairs` | Within 50m shortest segment distance: count repaired in the 365 days preceding January 1. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_50m_road_count` | Within 50m shortest segment distance: road-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_50m_same_material_fraction` | Within 50m shortest segment distance: fraction sharing the focal original material. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_50m_structure_count` | Within 50m shortest segment distance: structure-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_50m_surface_diversity` | Within 50m shortest segment distance: number of distinct surface labels. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_50m_swamp_count` | Within 50m shortest segment distance: swamp-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_50m_water_count` | Within 50m shortest segment distance: water-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_5m_age_mean` | Within 5m shortest segment distance: mean age at the event date (years). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_5m_count` | Within 5m shortest segment distance: number of other pipes. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_5m_days_since_repair` | Within 5m shortest segment distance: days from latest previous repair to January 1. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_5m_iron_fraction` | Within 5m shortest segment distance: fraction originally iron. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_5m_length_sum` | Within 5m shortest segment distance: sum of whole endpoint lengths (m). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_5m_parallel_count` | Within 5m shortest segment distance: number of nearly parallel pipes (within 15 degrees, positive projected overlap). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_5m_parallel_projected_length` | Within 5m shortest segment distance: sum of projected overlap lengths for nearly parallel pipes (m). Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_5m_prior_repair_fraction` | Within 5m shortest segment distance: fraction with a prior recorded repair. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_5m_prior_repairs` | Within 5m shortest segment distance: count of prior recorded repairs. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_5m_recent_repairs` | Within 5m shortest segment distance: count repaired in the 365 days preceding January 1. Excludes focal pipe and unknown/future lay dates. | Prior-year severity-row repair history + inventory |
| `inventory_5m_road_count` | Within 5m shortest segment distance: road-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_5m_same_material_fraction` | Within 5m shortest segment distance: fraction sharing the focal original material. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_5m_structure_count` | Within 5m shortest segment distance: structure-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_5m_surface_diversity` | Within 5m shortest segment distance: number of distinct surface labels. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_5m_swamp_count` | Within 5m shortest segment distance: swamp-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_5m_water_count` | Within 5m shortest segment distance: water-labelled pipe count. Excludes focal pipe and unknown/future lay dates. | Inventory at event date |
| `inventory_collinear_overlap_count` | Count of collinear overlaps of positive length. | Inventory at event date; geometry only |
| `inventory_distance_to_road` | Shortest segment distance (m) to another road-labelled pipe; not distance to a mapped land boundary. | Inventory at event date |
| `inventory_distance_to_structure` | Shortest segment distance (m) to another structure-labelled pipe; not distance to a mapped land boundary. | Inventory at event date |
| `inventory_distance_to_water` | Shortest segment distance (m) to another water-labelled pipe; not distance to a mapped land boundary. | Inventory at event date |
| `inventory_intersection_count` | Count of intersecting other segments, including endpoint touches/overlaps. | Inventory at event date; geometry only |
| `inventory_mean5_midpoint_distance` | Mean midpoint distance (m) to five nearest eligible other pipes. | Inventory at event date; geometry only |
| `inventory_nearest_midpoint_distance` | Distance (m) to nearest eligible other pipe midpoint. | Inventory at event date; geometry only |
| `inventory_proper_crossing_count` | Count of proper interior segment crossings. | Inventory at event date; geometry only |
| `inventory_shared_endpoint_count` | Count of other pipes sharing at least one endpoint. | Inventory at event date; geometry only |
| `lay_month` | Original installation month. | Inventory/event attributes or endpoint topology |
| `lay_year` | Original installation year. | Inventory/event attributes or endpoint topology |
| `length_age` | Pipe length (m) multiplied by pipe age (years). | Inventory/event attributes or endpoint topology |
| `length_x_structure_exposure` | Pipe length * ln(1 + inventory_25m_structure_count). | Inventory/event attributes or endpoint topology |
| `material_surface` | Combined original material and surface category. | Inventory/event attributes or endpoint topology |
| `midpoint_x_squared` | Square of midpoint_x; distances in m, ages in years. | Derived from existing columns |
| `midpoint_y_squared` | Square of midpoint_y; distances in m, ages in years. | Derived from existing columns |
| `month_cos` | cos(2*pi*(event_month-1)/12). | Recorded event calendar/time |
| `month_sin` | sin(2*pi*(event_month-1)/12). | Recorded event calendar/time |
| `orientation_cos2` | cos(2*orientation angle): (dx^2-dy^2)/length^2; invariant to endpoint order. | Inventory/event attributes or endpoint topology |
| `orientation_sin2` | sin(2*orientation angle): 2*dx*dy/length^2; invariant to endpoint order. | Inventory/event attributes or endpoint topology |
| `pipe_length_squared` | Square of pipe_length; distances in m, ages in years. | Derived from existing columns |
| `road_x_structure_exposure` | Indicator that focal surface is road * ln(1 + inventory_25m_structure_count). | Inventory/event attributes or endpoint topology |
| `span_x` | Absolute difference between endpoint x coordinates (m). | Inventory/event attributes or endpoint topology |
| `span_y` | Absolute difference between endpoint y coordinates (m). | Inventory/event attributes or endpoint topology |
| `weekday_cos` | cos(2*pi*event_weekday/7). | Recorded event calendar/time |
| `weekday_sin` | sin(2*pi*event_weekday/7). | Recorded event calendar/time |
| `years_used_at_leak_squared` | Square of years_used_at_leak; distances in m, ages in years. | Derived from existing columns |
| `neighbor_count_50` | Other eligible pipes within 50m midpoint distance (different from segment distance). | Inventory at event date |
| `neighbor_count_100` | Other eligible pipes within 100m midpoint distance (different from segment distance). | Inventory at event date |
| `neighbor_length_sum_100` | Total endpoint length of other pipes within 100m midpoint distance (m). | Inventory/event attributes or endpoint topology |
| `neighbor_road_count_100` | Road-labelled other pipe count within 100m midpoint distance. | Inventory/event attributes or endpoint topology |
| `neighbor_count_250` | Other eligible pipes within 250m midpoint distance (different from segment distance). | Inventory at event date |
| `prior_repairs_25m` | Other eligible pipes within 25m shortest segment distance: number repaired before January 1. | Full raw prior-year repair history + inventory |
| `prior_repair_fraction_25m` | Other eligible pipes within 25m shortest segment distance: fraction repaired before January 1. | Full raw prior-year repair history + inventory |
| `no_recorded_repair_25m` | Other eligible pipes within 25m shortest segment distance: number with no recorded repair before January 1. | Full raw prior-year repair history + inventory |
| `current_pu_fraction_25m` | Other eligible pipes within 25m shortest segment distance: fraction originally polyurethane or repaired before January 1. | Full raw prior-year repair history + inventory |
| `remaining_iron_count_25m` | Other eligible pipes within 25m shortest segment distance: number originally iron and not repaired before January 1. | Full raw prior-year repair history + inventory |
| `remaining_iron_length_25m` | Other eligible pipes within 25m shortest segment distance: sum of lengths (m) of original iron not repaired before January 1. | Full raw prior-year repair history + inventory |
| `repairs_last365d_25m` | Other eligible pipes within 25m shortest segment distance: number repaired during the 365 days preceding January 1. | Full raw prior-year repair history + inventory |
| `days_since_neighbor_repair_25m` | Other eligible pipes within 25m shortest segment distance: days since latest repair before January 1, blank if no repairs. | Full raw prior-year repair history + inventory |
| `prior_repairs_100m` | Other eligible pipes within 100m shortest segment distance: number repaired before January 1. | Full raw prior-year repair history + inventory |
| `prior_repair_fraction_100m` | Other eligible pipes within 100m shortest segment distance: fraction repaired before January 1. | Full raw prior-year repair history + inventory |
| `no_recorded_repair_100m` | Other eligible pipes within 100m shortest segment distance: number with no recorded repair before January 1. | Full raw prior-year repair history + inventory |
| `current_pu_fraction_100m` | Other eligible pipes within 100m shortest segment distance: fraction originally polyurethane or repaired before January 1. | Full raw prior-year repair history + inventory |
| `remaining_iron_count_100m` | Other eligible pipes within 100m shortest segment distance: number originally iron and not repaired before January 1. | Full raw prior-year repair history + inventory |
| `remaining_iron_length_100m` | Other eligible pipes within 100m shortest segment distance: sum of lengths (m) of original iron not repaired before January 1. | Full raw prior-year repair history + inventory |
| `repairs_last365d_100m` | Other eligible pipes within 100m shortest segment distance: number repaired during the 365 days preceding January 1. | Full raw prior-year repair history + inventory |
| `days_since_neighbor_repair_100m` | Other eligible pipes within 100m shortest segment distance: days since latest repair before January 1, blank if no repairs. | Full raw prior-year repair history + inventory |
| `prior_repairs_250m` | Other eligible pipes within 250m shortest segment distance: number repaired before January 1. | Full raw prior-year repair history + inventory |
| `prior_repair_fraction_250m` | Other eligible pipes within 250m shortest segment distance: fraction repaired before January 1. | Full raw prior-year repair history + inventory |
| `no_recorded_repair_250m` | Other eligible pipes within 250m shortest segment distance: number with no recorded repair before January 1. | Full raw prior-year repair history + inventory |
| `current_pu_fraction_250m` | Other eligible pipes within 250m shortest segment distance: fraction originally polyurethane or repaired before January 1. | Full raw prior-year repair history + inventory |
| `remaining_iron_count_250m` | Other eligible pipes within 250m shortest segment distance: number originally iron and not repaired before January 1. | Full raw prior-year repair history + inventory |
| `remaining_iron_length_250m` | Other eligible pipes within 250m shortest segment distance: sum of lengths (m) of original iron not repaired before January 1. | Full raw prior-year repair history + inventory |
| `repairs_last365d_250m` | Other eligible pipes within 250m shortest segment distance: number repaired during the 365 days preceding January 1. | Full raw prior-year repair history + inventory |
| `days_since_neighbor_repair_250m` | Other eligible pipes within 250m shortest segment distance: days since latest repair before January 1, blank if no repairs. | Full raw prior-year repair history + inventory |
| `wet_segments_25m` | Count of other water/swamp-labelled segments within 25m shortest segment distance. | Inventory at event date |
| `wet_segments_100m` | Count of other water/swamp-labelled segments within 100m shortest segment distance. | Inventory at event date |
| `event_night` | 1 when recorded event time is before 06:00 or at/after 22:00. | Inventory/event attributes or endpoint topology |
| `topology_eligible` | 1 when original lay date is known and before 2019; 0 otherwise. | Inventory/event attributes or endpoint topology |
| `route_id` | Route: Common grouping ID; not an ordered numeric predictor. | Exact endpoint geometry, pre-2019 inventory |
| `straight_section_id` | Straight section: Common grouping ID; not an ordered numeric predictor. | Exact endpoint geometry, pre-2019 inventory |
| `endpoint_degree_1` | Number of eligible segments meeting original endpoint 1. | Inventory/event attributes or endpoint topology |
| `endpoint_degree_2` | Number of eligible segments meeting original endpoint 2. | Inventory/event attributes or endpoint topology |
| `is_branch_adjacent` | 1 when at least one endpoint has degree >=3. | Inventory/event attributes or endpoint topology |
| `has_continuation` | 1 when at least one endpoint has exactly two incident pipes. | Inventory/event attributes or endpoint topology |
| `turn_angle` | Maximum direction change at a degree-2 endpoint (degrees); blank with no continuation. | Inventory/event attributes or endpoint topology |
| `is_bend` | 1 when a degree-2 endpoint direction change exceeds 10 degrees. | Inventory/event attributes or endpoint topology |
| `is_straight` | 1 when there is a degree-2 continuation and its maximum direction change is <=10 degrees; local criterion only. | Inventory/event attributes or endpoint topology |
| `route_is_curved` | Route: 1 for closed route, accumulated turn >=30 degrees, or tortuosity >1.01. | Exact endpoint geometry, pre-2019 inventory |
| `route_is_closed` | Route: 1 for a closed endpoint chain. | Exact endpoint geometry, pre-2019 inventory |
| `route_length` | Route: Sum of member endpoint lengths (m). | Exact endpoint geometry, pre-2019 inventory |
| `route_pipe_count` | Route: Number of member segments. | Exact endpoint geometry, pre-2019 inventory |
| `route_total_turn_deg` | Route: Sum of direction changes at degree-2 nodes in degrees. | Exact endpoint geometry, pre-2019 inventory |
| `route_tortuosity` | Route: Route length / straight distance between ends; blank for closed routes. | Exact endpoint geometry, pre-2019 inventory |
| `route_position` | Route: Distance (m) from deterministic route start to the focal segment midpoint along the route. | Exact endpoint geometry, pre-2019 inventory |
| `route_position_fraction` | Route: Route position divided by route length. | Exact endpoint geometry, pre-2019 inventory |
| `straight_section_pipe_count` | Straight section: Number of member segments. | Exact endpoint geometry, pre-2019 inventory |
| `straight_section_length` | Straight section: Sum of member endpoint lengths (m). | Exact endpoint geometry, pre-2019 inventory |
| `route_prior_leak_count` | Route: Number of recorded leaks in all preceding calendar years. | Full raw prior-year leak history |
| `route_prior_leak_cost` | Route: Total recorded leak cost in all preceding calendar years. | Full raw prior-year leak history |
| `route_last1_leak_count` | Route: Number of recorded leaks in the previous calendar year. | Full raw prior-year leak history |
| `route_prior_leaks_per_km` | Route: Prior leak count divided by group length in kilometres; descriptive, not a failure probability. | Full raw prior-year leak history |
| `route_prior_cost_per_m` | Route: Prior leak cost divided by group length in metres. | Full raw prior-year leak history |
| `route_days_since_leak` | Route: Days from last recorded prior leak to January 1; blank if none. | Full raw prior-year leak history |
| `straight_section_prior_leak_count` | Straight section: Number of recorded leaks in all preceding calendar years. | Full raw prior-year leak history |
| `straight_section_prior_leak_cost` | Straight section: Total recorded leak cost in all preceding calendar years. | Full raw prior-year leak history |
| `straight_section_last1_leak_count` | Straight section: Number of recorded leaks in the previous calendar year. | Full raw prior-year leak history |
| `straight_section_prior_leaks_per_km` | Straight section: Prior leak count divided by group length in kilometres; descriptive, not a failure probability. | Full raw prior-year leak history |
| `straight_section_prior_cost_per_m` | Straight section: Prior leak cost divided by group length in metres. | Full raw prior-year leak history |
| `straight_section_days_since_leak` | Straight section: Days from last recorded prior leak to January 1; blank if none. | Full raw prior-year leak history |
