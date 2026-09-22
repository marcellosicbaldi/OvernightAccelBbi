# VH2015 wake episodes and analysis selection

In `offline_processing/load_fit_test.ipynb`, run the diary crop, then **VH2015
wake detection and analysis selection**, then the burst and cardiac sections.
The detector consumes the current raw `accel_df` with `sample_time`, `accel_x`,
`accel_y`, and `accel_z`. It does not read the diary, select a diary row, estimate
an SPT, or trim the crop to 15-minute boundaries. Rerun this section and every
following analysis cell after changing the diary or selection settings.

## Settings

| Notebook setting | Default | Meaning |
| --- | --- | --- |
| `ANALYSIS_WINDOW` | `"whole_spt"` | Entire diary-cropped night, including any unclassified tail; this name does not imply an estimated SPT |
| Alternative modes | `"sleep_only"`, `"wake_only"` | Effective VH2015 sleep or wake intervals, respectively |
| `MIN_WAKE_EPISODE_SECONDS` | `None` | No minimum wake-duration filtering; a non-negative number suppresses wake bouts strictly shorter than that many seconds |
| `SAMPLING_RATE_HZ` | `100.0` | Regular grid used for raw-axis preprocessing and burst preparation |
| `MAX_ACCEL_GAP_SECONDS` | `0.25` | Maximum acquisition gap allowed in acceleration interpolation |
| `ACCEL_INPUT_UNIT` | `"mg"` | Scale for acceleration magnitude and burst detection; z-angle is invariant to a common positive scale on all three axes |

Suppressed wake bouts count as effective sleep, including bouts at the crop
edges. Bouts equal to the minimum remain wake. `None` and zero retain the same
labels. The unfiltered labels and suppression flags remain available.

## Algorithm provenance and Garmin adaptation

`offline_processing/vh2015_wake.py` ports the algorithmic functions from the
user-supplied `sleep_menghini/vh2015_sleep.py`: `ggir_roll_median_axis_pandas`,
its GGIR edge handling and epoch-averaging helpers, `compute_anglez`, and
`vh2015_sleep_wake_from_anglez`.

1. Sort raw timestamps, average duplicate XYZ observations, and interpolate each
   axis onto the regular sample grid. Reject nonfinite inputs and acquisition
   gaps over the configured limit, as the existing burst preparation does.
2. Apply the source's centered five-second rolling-median axis preprocessing,
   including its approximately 10 Hz intermediate representation.
3. Compute `degrees(arctan(z / sqrt(x*x + y*y)))`, average over complete
   five-second epochs, and round to four decimals as in the source.
4. Mark contiguous runs of adjacent epoch-angle differences **<=5 degrees** as
   sleep when the run contains at least **five minutes**. Other complete epochs
   are wake. Runs touching either crop edge are included.
5. Optionally suppress short wake bouts, then form end-exclusive selected intervals.

The source's optional GGIR 3.3-6 resize bug is disabled: repeated rolling medians
are cropped to the signal length instead of unexpectedly returning raw axes when
the crop length is not divisible by the downsampling step. GENEActiv-specific
readers, resampling and GGIR 15-minute alignment are not used. Epochs begin at the
first sample of the supplied crop. There is no additional 30-second majority vote.

A final incomplete five-second epoch is unclassified and excluded from sleep-only
and wake-only analysis. The report gives its duration; whole-night analysis keeps
it. Recordings too short to support the centered median have no classified epochs.
An undefined rolling z-angle raises an error rather than becoming sleep. These
labels are accelerometer estimates; this port is not validation on Garmin or a
sleep-stage classifier.

## Keeping excluded periods separate

`analysis_windows.detect_bursts_in_intervals` selects raw rows **before** magnitude
regularization, band-pass filtering, envelope interpolation, and burst merging.
Each selected interval gets its own detector call. Even gaps shorter than the
five-second merge rule cannot join bursts. Intervals with too few samples for the
filter are reported as `too_short_for_filter`; they do not produce detections.
Only complete regular sample periods contribute to burst AUC at interval edges.

The recorded-HR analysis restricts observations and linear/cubic interpolation
anchors to the event's interval. Epoch queries outside it remain missing. The
entire -20 through +49 s response and observed isolation margins must fit in that
same interval to be included. Boundary exclusions remain inspectable as
`analysis_interval_edge`. AUC tertiles are calculated over all selected candidate
bursts together, before HR exclusions, rather than separately per interval.

GP HRV also inherits the selection: BBI rows, cleaning runs, burst guards, quiet
segments and windows cannot cross excluded periods. Its short-burst and quality
policies otherwise remain the same. Plots draw selected intervals separately.
An empty selection produces empty burst, HR and HRV tables without stale results.

## Inspectable outputs and checks

- `vh2015_epochs`: start/end, z-angle, `raw_wake`, effective `wake`/`sleep`, and
  `wake_suppressed` for every complete epoch.
- `wake_episodes` and `vh2015_result["raw_wake_episodes"]`: counted and original
  wake bouts, including duration in seconds.
- `analysis_intervals` and `analysis_intervals_utc`: selected [start, end) bounds
  in the acceleration clock and UTC. The notebook clips them to diary bounds.
- `bursts_df`: interval ID plus the existing burst metrics;
  `burst_signal_segments`: independent magnitude/filter/envelope Series;
  `acc_quality`: per-interval preparation and skipped-interval diagnostics.
- Existing HR/HRV tables remain available. Their reports record the selected
  intervals, mode, VH2015 thresholds and wake-duration setting. Optional Parquet
  export includes labels, wake bouts and selected intervals in the ignored output
  directory. Raw `accel_df` and `bbi_df` remain unchanged.

`test_vh2015_wake.py` uses synthetic data and reference-generated z-angle values
to check the port, threshold boundaries, all selection modes, duration suppression,
input preservation, empty selections, independent burst detection, global AUC
tertiles, HR interpolation/repair bounds and separated HRV cleaning/windows.
Run with the repository's standard unittest discovery command.
