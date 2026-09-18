# Changelog

Repository releases version the combined recorder, Python analysis, and documentation.
The watch's on-screen firmware version is tracked separately.

## Unreleased

- Use 30-second movement isolation plus full-epoch subsequent-movement exclusion
  in the main HR-response notebook.
- Adopt the updated -20 through +49 s HR epoch and 15-s baseline [-20, -5).
  Derive report labels and plot bounds from the active analysis settings.
- Update regression checks to reject a subsequent movement at +49 s and allow
  one at +50 s when the 30-second gap rule is also satisfied.
- Planned: Night Explorer FIT import and interactive event inspection.

## 0.1.0 - 2026-09-18

First public source release. Includes watch recorder **v1.1.1** and the existing
offline analysis pipeline; no interactive web application is included yet.

### Included

- Connect IQ recorder for Instinct 3 AMOLED 45 mm and 50 mm, native acceleration
  logging, bounded numbered BBI snapshots, and a final session snapshot.
- BBI recovery diagnostics, movement-burst detection, movement-associated HR
  responses, quiet-period RMSSD/PIP, sleep-diary cropping, and wrist reorientation.
- Portable notebook configuration and notebooks without personal recordings or outputs.
- Setup documentation, development roadmap, and automated Python regression checks.

### Publication fixes

- Preserve the current HR epoch (-19 through +54 s) and 14-s baseline [-19, -5),
  and align documentation, report metadata, and regression tests with these settings.
- Extend the optional subsequent-movement overlap guard to the actual epoch end.
- Remove workstation-specific build defaults and use the SDK Manager configuration
  or explicit environment variables/arguments.
