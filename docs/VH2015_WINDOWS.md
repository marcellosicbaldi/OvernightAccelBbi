# VH2015 and selected overnight analysis windows

The main notebook runs `offline_processing/vh2015_sleep.py` immediately after
the diary crop and before wrist bursts. Its input is the already-cropped raw
Garmin XYZ table. The adapter neither reads a diary nor estimates a second SPT.
It uses the preserved UTC timestamps for the cropped rows, avoiding the loss of
timezone information in the notebook's earlier local-time display tables.

## Controls

| Setting | Default | Meaning |
| --- | --- | --- |
| `ANALYSIS_WINDOW` | `"whole"` | All recorded portions of the diary-cropped night, including unclassified tails |
| Other modes | `"sleep_only"`, `"wake_only"` | Scored sleep or counted wake episodes only |
| `MIN_WAKE_EPISODE_SECONDS` | `None` | No minimum wake duration; a number filters raw wake episodes shorter than that many seconds |
| `SAMPLING_RATE_HZ` | `100.0` | Garmin regularization rate, shared with burst analysis |
| `MAX_ACCEL_GAP_SECONDS` | `0.25` | Larger timestamp gaps split acquisition runs |

A wake episode exactly as long as the minimum still counts. Short rejected wake
episodes become sleep for window selection. Both raw and filtered epoch labels
remain in `vh_epochs`; `vh_wake_episodes` retains every raw episode, duration and
`counted_wake` flag. Changing a setting requires rerunning the VH2015 cell and
the downstream analysis cells. Saved outputs remain stale until rerun.

## Algorithm and provenance

The numerical core is ported from the user's `sleep_menghini/vh2015_sleep.py`:
`_r_seq_round_indices`, `ggir_average_per_epoch`,
`_replace_edge_zeros_like_ggir`, `ggir_roll_median_axis_pandas`, and
`vh2015_sleep_wake_from_anglez`. `compute_anglez` retains its pandas calculation.
No local path or external copy of that project is needed at runtime.

The port retains rolling five-second medians of the three axes, mean z-angle in
five-second epochs, four-decimal rounding, and classification of stable runs
lasting at least five minutes as sleep. Adjacent angle changes must remain at
or below five degrees; edge runs count. Other scored epochs are wake. A common
XYZ scale (mg versus g) cancels in the angle calculation; burst magnitude still
requires the explicit unit conversion.

Garmin adaptation: sort/average duplicate XYZ timestamps and interpolate jitter
only inside acquisition runs. Each run starts its own five-second epoch grid.
The adapter uses the reference's `match_ggir_resize_bug=False` option so an
arbitrary crop length cannot accidentally bypass rolling-median preprocessing.
GENEActiv page resampling, 15-minute trimming and PSG aggregation are omitted.
Incomplete trailing epochs are unclassified, included in `whole` mode only.
Runs too short for the reference's full rolling-median window are also
unclassified (at 100 Hz, the odd window needs just over five seconds).
Nonfinite samples, zero vectors and undefined epoch angles cause explicit errors.
These are algorithmic sleep/wake estimates, not independently established sleep.

## Boundaries and downstream behavior

All intervals use UTC `[start, end)` bounds. `analysis_intervals` is the requested
selection. `burst_analysis_intervals` records the actual regular-grid coverage
that supports burst filtering; intervals with insufficient filter padding are
reported and omitted. Gaps are never joined to obtain a longer input.

- Acceleration regularization, band-pass filtering, envelope calculation and
  burst merging run independently inside each selected interval. Even excluded
  gaps shorter than the five-second burst merge rule remain boundaries.
- HR observations are selected before analysis. Each event uses only observations
  from its own interval, including all linear/spline interpolation anchors.
  Baselines or responses crossing a boundary receive `analysis_interval_edge`;
  samples outside that interval remain missing even if another interval has HR.
  Isolation guards must fit inside the same interval. AUC tertiles are computed
  across all selected candidates together, before HR exclusions.
- GP HRV intersects quiet segments with the same intervals before cleaning or
  constructing windows. BBI differences, artifact detection and interpolation
  cannot cross excluded periods. Short wake-only intervals may yield no complete
  HR-response or HRV windows; exclusions and empty tables remain inspectable.
- Plots draw intervals separately so line segments cannot imply continuity.

The original diary-cropped `accel_df` and recorded-HR source table are preserved.
Detection tables include `analysis_interval_id`. HR and HRV reports record the
chosen mode, VH2015 settings and interval count. The optional raw-data export
still exports the diary-cropped acceleration; it is not a selected-only table.

## Validation

Synthetic tests cover the five-minute and five-degree thresholds, edge bouts,
raw XYZ sleep/wake detection, timestamp jitter/duplicates/timezones, partial
epochs, wake-duration filtering, empty selections, and acquisition gaps.
Regression tests also verify that excluded acceleration cannot alter selected
bursts, that short exclusions prevent merging, that HR boundary epochs and
spline anchors cannot cross intervals, and that HRV cleaning/windows split.

Run `python -m unittest discover -s offline_processing -p "test_*.py" -v`.
