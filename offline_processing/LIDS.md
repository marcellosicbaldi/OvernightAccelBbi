# Garmin envelope–LIDS

Run `lids.ipynb` for nightly plots, a parameter table, and the mean of four
period-normalized LIDS cycles. The input is the existing, already SPT-cropped
`<processed_root>/<night>/accelerometer_samples.parquet` layout. No diary or
additional sleep detection is used.

Set `OVERNIGHT_PROCESSED_ROOT`, set `PROCESSED_ROOT` in the notebook, or put
`{"processed_root": "/path/to/garmin_processed"}` in a private, Git-ignored
`offline_processing/lids.local.json`. Relative paths resolve against the
notebook's module directory. Outputs
default to the Git-ignored `offline_processing/outputs/lids_envelope/`. The command-line
equivalent is:

```sh
python offline_processing/lids_analysis.py /path/to/garmin_processed
```

## Activity and transformation

This is an **acceleration-envelope adaptation**, not a calibrated reproduction
of actigraph ZCM counts. The activity scale changes LIDS amplitude, offset, MRI
and potentially the chosen period. Do not directly compare these values to
ZCM-based results or the previous ENMO analysis.

1. Interpret Garmin XYZ as mg (configurable), nominally 100 Hz. Interpret naive
   timestamps in Europe/Rome (configurable). Use the existing burst-analysis
   `prepare_acceleration`: compute magnitude in g, average duplicate magnitudes,
   and interpolate onto a regular sampling grid.
2. Process continuous finite runs independently: split at invalid XYZ samples
   or gaps longer than `max_gap_s=0.25`. Skip runs too short for filter padding
   (51 samples or fewer). Do not filter across gaps; midnight itself does not split a continuous recording.
3. Use the shared burst-analysis `bandpass_acceleration`: 0.1–10 Hz, order-8,
   zero-phase Butterworth, using NeuroKit's SOS implementation. Then use the
   same `compute_envelope`: upper minus lower envelope, each interpolated from
   groups of ten extrema. No burst threshold or event selection is applied.
4. Convert the continuous envelope from g to **mg**, then average each minute.
   Minutes are anchored at the first saved SPT sample and labeled at their left
   edges. Require at least 90% coverage in both finite unique observed samples
   and processed regular samples. Missing samples are never treated as zero.
5. **Sum ten valid one-minute means** to obtain each 10-minute activity score:
   `activity_sum_mg = sum(minute_mean_envelope_mg)`.
   All ten minutes must be valid; do not sum a shorter or incomplete bin.
   This is a sum of minute-mean mg values, not an integral in mg·s or activity
   counts. The mg scale is explicit: using g would change the LIDS transform.
6. Convert `LIDS = 100 / (1 + activity_sum_mg)`, then smooth with a centered
   three-bin (30-minute) moving mean. At valid-run edges use the available one
   or two bins; never smooth through a missing bin. Higher LIDS means less
   movement. Both raw and smoothed values are retained.

Elapsed time uses actual timestamp differences across timezone transitions;
ambiguous naive DST timestamps are rejected. The high-pass component removes
constant magnitude offsets, avoiding ENMO's subtraction of a fixed 1-g baseline
and clipping below that baseline. This does not eliminate all calibration
issues: axis scale errors still affect the envelope. Filter-edge transients
and constant envelope extension beyond outer extrema follow the burst method.

Minute-level CSVs expose the means and coverage used for every activity sum.
The reference notebook transforms minute-level ZCM; this adaptation instead
sums minute-mean envelopes into the requested 10-minute timeline before LIDS.
Previous local ENMO results remain in `outputs/lids/`; new outputs are written
to `outputs/lids_envelope/`.

## Fit parameters

The default follows the projection fit and maximum-MRI selection in the user's
[reference analysis](https://github.com/marcellosicbaldi/lids-analysis), scanning
30–180 minutes in 5-minute steps:

`fit(t) = offset + amplitude * cos(2π*t/period - phase_rad)`

For each candidate period, `a = 2*mean(LIDS*cos(2π*t/period))`,
`b = 2*mean(LIDS*sin(2π*t/period))`, `offset = mean(LIDS)`,
`amplitude = hypot(a,b)`, and `phase_rad = atan2(b,a)`. Finite pairs only are
used; missing bins keep their actual elapsed positions.

- `period_min`: period selected by maximum `MRI = 2*amplitude*Pearson r`.
- `amplitude`: half the fitted peak-to-trough range, in adapted LIDS units.
- `phase_rad`, `phase_deg`: cosine phase relative to the first saved SPT sample,
  using the minus-phase convention above; these are not the separately defined
  onset/offset inflection phases used by some LIDS packages.
- `peak_time_min`: first fitted maximum at or after onset, modulo one period.
- `pearson_r`, `r2`, `rmse`, `mri`: descriptive goodness of fit. `r2` is
  `1-SSE/SST`, not Pearson r squared.
- `period_at_boundary`: the optimum hit an end of the search grid.

Projection is preserved for compatibility. For an incomplete number of cycles
or missing bins, the sine/cosine/intercept basis is not orthogonal; projection
can yield biased amplitudes and negative R². `LIDSConfig(fit_method="least_squares")`
(CLI: `--fit-method least_squares`) fits those three coefficients jointly and
uses the same maximum-MRI period selection. Every output records the method.
Fit curves are not clipped to 0–100, so any overshoot remains visible.

At least six finite bins spanning 60 minutes are required. Constant, too-short,
or unreadable nights remain in the summary with a reason and no invented
period. No significance filter is imposed on successful fits; boundary periods
and poor fits remain visible. No naive Pearson p-value is reported because
smoothing, autocorrelation, and period selection invalidate its usual inference.

## Average profile

For each successfully fitted night, use its **observed smoothed LIDS** and
convert elapsed time to `internal_min = external_min / period_min * 90`.
Linearly interpolate within contiguous valid stretches onto `0, 10, ..., 350`
internal minutes: the 36 ten-minute positions of four cycles. Do not shift
phase, normalize amplitude, extend short nights, or interpolate over missing
bins. Bin values are represented at their left edges; no right-edge value is
fabricated beyond the final observed bin.

The mean gives each available night equal weight in each bin. SEM uses the
sample standard deviation across those nights and is undefined for fewer than
two contributors. The plot includes each rescaled night faintly, mean ± SEM,
and the actual number contributing at every position. `lids_cycles.csv`
reports nights with any data, all nine positions, and minimum/maximum per-bin
counts for each cycle. The quoted ~50% fourth-cycle retention is a motivation
for limiting to four cycles, not an assumed property of these recordings.

Files include nightly PNGs, one-minute activity CSVs, and 10-minute binned CSVs, `lids_summary.csv`,
`lids_normalized.csv`, `lids_profile.csv`, `lids_cycles.csv`, `settings.json`,
and `lids_average.png`/`.svg`. Personal results and executed notebook copies
stay local. The committed notebook has no outputs.

Background: [Winnebeck et al. (2018)](https://doi.org/10.1016/j.cub.2017.11.063),
*Dynamics and Ultradian Structure of Human Sleep in Real Life*;
[pyActigraphy LIDS documentation](https://ghammad.github.io/pyActigraphy/LIDS.html).

Synthetic validation:

```sh
python -m unittest discover -s offline_processing -p "test_lids_analysis.py" -v
```
