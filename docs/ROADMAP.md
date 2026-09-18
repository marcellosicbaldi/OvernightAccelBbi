# Roadmap

The next milestone is **import a night, inspect its events**. This is planned work,
not functionality available in the first public source release.

## v0.2.0: Night Explorer

### Import and recording quality

- Import one local FIT file through a browser interface backed by the Python pipeline.
- Report recording bounds, acceleration timing/units, BBI sequence recovery, and gaps.
- Let the user set the analysis interval and timezone or supply a sleep diary.
- Keep original signals and analysis settings available alongside derived results.

### Timeline and event inspection

- Display synchronized movement and recorded-HR timelines with detected bursts.
- Select a burst to inspect its duration, AUC, HR epoch, and exclusion reasons.
- Display quiet-period HRV windows, including excluded windows and their reasons.
- Clearly distinguish recorded HR from interval-derived estimates.

### Completion criteria

- A user can import a supported recording and inspect an event without editing Python.
- Import errors and missing BBI data have understandable messages.
- The UI reports the same events and quality decisions as the underlying pipeline.
- A clearly labeled synthetic demonstration can be explored without personal data.
- Processing initially runs locally; automated Garmin synchronization is a later investigation.

## Later directions

- Compare nights and export analysis reports with versioned settings.
- Add diary entries, morning ratings, and personal experiment tracking.
- Batch-process studies and support review annotations.
- Validate movement and cardiac measurements against reference sensors.
- Investigate Garmin synchronization, device compatibility, and a mobile companion.

Sleep-stage classification and clinical interpretation require separate validation.
