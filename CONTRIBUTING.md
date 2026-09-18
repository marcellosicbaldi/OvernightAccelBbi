# Development

Use Python 3.12 and install `offline_processing/requirements.txt`. Run the existing
regression suite from the repository root:

```sh
python -m unittest discover -s offline_processing -p "test_*.py" -v
```

Analysis changes should document their units, timing assumptions, configurable
thresholds, and handling of missing data. Preserve meaningful exclusions instead
of silently filling gaps. Add regression coverage when changing analysis behavior.

Before committing notebooks, clear all outputs and save them. Keep recording paths
in `offline_processing/paths.local.json`, which is ignored by Git. Do not add raw
recordings, diaries, generated results, signing keys, or participant information.
Synthetic fixtures must be clearly identified as synthetic.

Use GitHub issues for work items and focused branches for changes. The first
implementation milestone is described in [the roadmap](docs/ROADMAP.md).

## Releases

Repository tags use `vMAJOR.MINOR.PATCH` and record the combined source release.
Update `CHANGELOG.md`, verify the Python CI run, and create a GitHub release from
the corresponding tag. Record the watch firmware version separately; the initial
repository release v0.1.0 contains watch firmware v1.1.1.

The Connect IQ simulator and physical-watch checks are separate from Python CI.
Do not describe a passing Python suite as device or physiological validation.
