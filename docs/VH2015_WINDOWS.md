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
| Other modes | `"sleep_only"`, `"wake_only"` | Select complete bursts by state; restrict HR/HRV to the corresponding intervals |
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
state selection. `burst_analysis_intervals` intersects that selection with the
actual acceleration coverage. Acquisition runs with insufficient filter padding
are reported and omitted. Gaps are never joined to obtain a longer input.

- Acceleration regularization, band-pass filtering, envelope calculation and
  burst merging run once on each complete diary-cropped acquisition run, before
  state selection. Missing-data gaps remain boundaries. Complete burst identity,
  onset, offset, duration and AUC are therefore consistent between modes.
- **Any positive overlap with a counted wake episode labels the entire burst
  wake**, even 1 ms. Formally, `burst.start < wake.end` and
  `burst.end > wake.start`. Touching an endpoint alone is zero overlap. This uses
  timestamp precision without rounding to seconds or a percentage threshold.
  Sleep requires full containment in a scored sleep interval; all other bursts
  are unclassified and appear only in `whole` mode. The overlap test uses wake
  episodes remaining after `MIN_WAKE_EPISODE_SECONDS` filtering.
- No burst is split or redetected at a sleep/wake boundary. Thus, a wake-labelled
  burst's sleep-side fragment cannot appear in `sleep_only`. A burst spanning
  multiple wake intervals remains one wake burst. `all_bursts_df` retains all
  labels; `bursts_df` is the selected subset, with stable `candidate_id` values.
- HR observations are selected before analysis. Each event uses only observations
  from its first overlapping interval, including all linear/spline anchors.
  Crossing wake bursts remain visible but receive `burst_crosses_analysis_interval`
  and cannot enter HR summaries. Their onsets are never moved to the wake boundary.
  Baselines or responses crossing a boundary receive `analysis_interval_edge`;
  samples outside that interval remain missing even if another interval has HR.
  Isolation guards must fit inside the same interval. AUC tertiles are computed
  across all selected candidates together, before HR exclusions.
  All original movements remain in isolation/late-overlap checks, including the
  sleep-side tails of wake-labelled bursts excluded from sleep event summaries.
- GP HRV intersects quiet segments with the same intervals before cleaning or
  constructing windows. All complete movement candidates block quiet periods,
  including wake-labelled bursts extending into sleep, subject to the existing
  minimum-burst-duration policy. BBI differences, artifact detection and interpolation
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
Regression tests verify 1 ms and sub-millisecond overlap, zero-duration endpoint
contact, full-burst wake assignment without sleep fragments, wake-duration
filtering before assignment, and preservation of full burst metrics. HR boundary
epochs and spline anchors cannot cross intervals, and HRV cleaning/windows split.
The lower-level interval detector retains independent filtering/merging when
explicitly given separate intervals; the notebook now supplies acquisition runs
to it and then selects complete bursts by state.

Run `python -m unittest discover -s offline_processing -p "test_*.py" -v`.
