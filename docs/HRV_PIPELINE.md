# GP HRV adaptation for Garmin BBIs

The HRV section in [load_fit_test.ipynb](../offline_processing/load_fit_test.ipynb)
now uses [gp_hrv.py](../offline_processing/gp_hrv.py), adapted from the author's
existing `GP_pipeline/src/gp_pipeline/wrist/heart_rate_variability` workflow.
The recorded-HR event averages and their movement isolation rules are separate
and unchanged by this adaptation.

## Processing and defaults

1. Decode the individually numbered Garmin BBIs. Preserve their values, sequence
   numbers, and approximate callback-arrival timestamps, including multiple BBIs
   delivered in one callback.
2. Reuse the notebook's detected movements and diary crop. For HRV segmentation,
   retain movements lasting at least **2 seconds**, add **1 second after** each,
   and take the complement inside the recorded diary interval. This matches GP's
   duration policy: shorter movements remain inside the resulting quiet periods.
   Set `HRV_MIN_BURST_DURATION_SECONDS=0` to retain them too.
3. Split each quiet segment at a missing sequence number or a callback gap over
   **5 seconds** with the notebook's `HRV_MAX_CALLBACK_GAP_SECONDS` setting.
   Classify and clean each resulting delivery run separately. Direct calls to
   `gp_hrv.interburst_hrv` retain a 3-second default unless overridden.
4. Apply GP's interval-based artifact classifier **once**, with intervals in
   seconds. It flags `ectopic`, `missed`, `extra`, and `longshort` indices. These
   are algorithm categories, not clinical diagnoses. Mask those intervals and
   values outside **300-2000 ms**, then interpolate linearly by interval order.
   Invalid endpoints take the nearest valid value. No usable anchors means no
   repair. No samples are added or removed, and raw values remain available.
5. Skip quiet segments shorter than **60 seconds**. Use the full segment for
   lengths of **60-300 seconds**. For longer segments, use **300-second windows**
   with **60-second steps**, adding a final window ending at the segment end
   when the steps do not land there. Cleaning precedes windowing, as in GP.
6. Screen each window and calculate the metrics below for accepted windows.
   Keep rejected windows with explicit reasons and missing metrics.

For example, a 430-second quiet segment produces windows `[0, 300)`, `[60, 360)`,
`[120, 420)`, and `[130, 430)`. A 180-second segment produces one 180-second
window. To use fixed 60-second windows with a 30-second step, set the notebook's
minimum and maximum to 60 and its step to 30.

## Metrics and quality

Metrics use cleaned intervals `b` in milliseconds:

| Output | Calculation |
| --- | --- |
| `mean_hr_bpm` | Mean of `60000 / b` |
| `rmssd_ms` | Square root of the mean squared successive differences |
| `sdnn_ms` | Sample standard deviation, `ddof=1` |
| `pip` | GP inflection-point count divided by the number of intervals |
| `pip_pct` | `100 * pip` |

GP's PIP counts adjacent differences whose product is `<= 1e-10`, including
zero differences. Therefore a constant sequence of N intervals has
`PIP=(N-2)/N`, despite zero RMSSD and SDNN. Quantization and interpolation affect
this definition. The formula is preserved rather than silently redefined.

Each window requires **at least 30 usable intervals**, no unresolved invalid
values, no missing sequence numbers, no callback or boundary silence over
the configured limit (5 seconds in the notebook), and a sum of cleaned BBIs within
**90-110%** of its wall-clock duration.
The last check screens timing consistency; it does not establish physiological
beat coverage. These delivery safeguards are additions for the Garmin data.

`HRV_MAX_INTERPOLATED_FRACTION=1.0` preserves GP's lack of rejection based on
the fraction repaired. All repaired intervals are flagged and each window reports
their count and fraction. A lower limit enables a stricter screen; no particular
limit has been validated here. A high repair fraction can strongly affect metrics.

## What was adapted

| Supplied GP workflow | Garmin implementation |
| --- | --- |
| Empatica PPG peak detection using MSPTDfast | Omitted: this recorder provides BBIs, without a raw PPG waveform |
| Invent a first IBI from the mean of the first intervals | Omitted: each delivered Garmin BBI already exists |
| GP `signal_fixpeaks(..., iterative=False)` artifact indices | Same classifier and constants; the unused corrected-peak return is not ported |
| Mask flags/range failures and interpolate linearly in each quiet period | Same rule inside each contiguous delivery run; never across sequence holes or long callback gaps |
| Inclusive timestamp slices | End-exclusive `[start, end)` slices avoid including the start of a movement or counting a shared boundary twice |
| Detect movements from the Empatica signal | Reuse the existing Garmin burst detector and its active threshold; no new detection is run for HRV |
| Sleep windows from the GP pipeline or diary | Use the notebook's matched lights-off-to-wakeup interval, clipped to the recording |
| Skip unusable windows | Preserve candidate windows and exclusion reasons for inspection |

For malformed nonfinite or nonpositive BBIs, the classifier itself runs on the
remaining contiguous positive finite subsequences; the invalid values are then
masked and repaired within their delivery run. Classification, interpolation,
and metric calculation never join across distinct quiet segments.

Window placement uses **callback arrival times, not exact beat times**. This
adapter does not establish normal-to-normal intervals, sensor origin, sleep
stages, or physiological validity of Garmin HRV. Variable-duration windows should
be compared with their duration in view; overlapping windows are not independent.

## Inspection and exports

The function returns:

- `intervals`: original BBIs, `bbi_clean_ms`, individual artifact-class flags,
  combined `artifact`, `invalid`, `interpolated`, and segment/delivery-run IDs.
  BBIs outside quiet periods are retained with unassigned IDs and no cleaned value.
- `segments`: the quiet intervals and their durations.
- `windows`: metrics, window lengths, artifact/repair counts, timing checks,
  inclusion decisions, and exclusion reasons.
- `blocked_bursts`: retained movements including the post-movement margin.
- `report`: active settings, counts, and interpretation notes.

The notebook plots RMSSD/SDNN, mean HR, PIP, and raw/cleaned BBIs. Times remain UTC
in tables and use `LOCAL_TZ` in plots. Set `SAVE_PARQUET=True` in the optional
export cell to write these tables as Parquet files and the settings as JSON to
the ignored `offline_processing/outputs/` directory. Parquet export uses the
`pyarrow` dependency listed in the analysis requirements.

The previous strict, uncorrected implementation remains in
[interburst_hrv.py](../offline_processing/interburst_hrv.py) for reference and
regression tests. It also supplies shared input validation, segmentation, and PIP
helpers. The notebook's active backend is `gp_hrv.py`.

## Provenance and verification

The following source files were supplied by the author and inspected on
2026-09-18. Their SHA-256 hashes identify the version used without introducing
a dependency on a private directory or publishing workstation paths:

| GP source | SHA-256 |
| --- | --- |
| `pipeline.py` | `c57311b75ba153f964cde3985e7f30999d7bdce9ce7e32a767c33e46538cb491` |
| `quiet_periods.py` | `7004bfd1c86eb1790fd7af14f5ef5a8bc448a57537b52e01d5d972207298b117` |
| `kubios.py` | `fb97e8f8c7e82036adebcaa8a6a9b649800b057be1ccb03bb57d49adc66ae19f` |
| `heart_rate_fragmentation.py` | `461676d7df91f43e2d32fde14362c67f63ec547690d141ac06d9be9a667b13d7` |

The `_find_artifacts` and `_compute_threshold` functions in
[gp_hrv_artifacts.py](../offline_processing/gp_hrv_artifacts.py) are copied from
the supplied `kubios.py` without changing their arithmetic. This code is derived
from NeuroKit; see [third-party notices](../THIRD_PARTY_NOTICES.md).

Direct comparisons against the supplied source checked artifact indices and
classifier arrays, cleaned intervals, PIP, and window geometry on deterministic
synthetic inputs. Repository tests include an injected-artifact fixture with
classifications from that source, metric formula checks, delivery-gap boundaries,
range screening, window placement, and empty/invalid inputs. These are software
regression checks, not physiological validation.

Run all tests from the repository root:

```sh
python -m unittest discover -s offline_processing -p "test_*.py" -v
```
