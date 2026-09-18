"""Resolve notebook inputs from an optional, Git-ignored local configuration."""

import json
import os
from pathlib import Path


def input_path(name):
    """Resolve relative paths against offline_processing, regardless of notebook cwd.

    OVERNIGHT_<NAME> environment variables override paths.local.json. No private
    input path needs to be embedded in a committed notebook.
    """
    directory = Path(__file__).resolve().parent
    defaults = {
        "fit_path": "data/recording.fit",
        "reorientation_fit_path": "data/recording.fit",
        "diary_path": "data/diary.xlsx",
    }
    if name not in defaults:
        raise ValueError(f"Unknown notebook input: {name}")
    config_file = directory / "paths.local.json"
    config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
    path = Path(os.environ.get(f"OVERNIGHT_{name.upper()}", config.get(name, defaults[name]))).expanduser()
    return path if path.is_absolute() else directory / path
