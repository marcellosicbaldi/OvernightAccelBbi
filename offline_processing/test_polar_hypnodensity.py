"""Synthetic checks for Polar PPG input selection and hypnodensity timing.

Install requirements-wav2sleep.txt to run the optional inference-pipeline checks.
The fake model returns known logits; no checkpoint or recordings are downloaded.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    import wav2sleep
except ImportError:
    torch = wav2sleep = None


def notebook_source(marker):
    notebook = json.loads(Path(__file__).with_name("polar.ipynb").read_text(encoding="utf-8"))
    return next("".join(cell["source"]) for cell in notebook["cells"] if marker in "".join(cell["source"]))


class RawChannelTests(unittest.TestCase):
    def test_raw_channel_survives_later_ppg_replacement(self):
        optical = pd.DataFrame({"PPG_0": [-100_000, -99_000], "PPG_1": [1, 2], "Ambient": [7, 8]})
        namespace = {"ppg": optical}
        exec(notebook_source("ppg_raw_0 ="), namespace)
        namespace["ppg"].iloc[:] = 0
        pd.testing.assert_series_equal(namespace["ppg_raw_0"], optical["PPG_0"])


@unittest.skipIf(wav2sleep is None, "Optional wav2sleep dependencies are not installed")
class HypnodensityTimingTests(unittest.TestCase):
    def run_cell(self, duration_s, input_source="raw"):
        # A synthetic optical waveform with a large DC offset, crossing midnight.
        t = np.arange(int(duration_s * 50)) / 50
        index = pd.Timestamp("2020-01-01 23:59:40") + pd.to_timedelta(t, unit="s")
        raw = pd.Series(-100_000 + 300 * np.sin(2 * np.pi * 1.2 * t), index=index, name="PPG_0")
        filtered = np.cos(2 * np.pi * 1.2 * t)
        acceleration = pd.DataFrame({"x": t * 0, "y": t * 0 + 1000, "z": t * 0}, index=index)
        original = raw.copy()
        original_filtered = filtered.copy()
        logits = np.array([1.0, 2.0, -0.5, 0.5], dtype=np.float32)
        captured = {}

        class FakeModel:
            num_classes = 4
            valid_signals = ["PPG"]
            signal_encoders = SimpleNamespace(causal=False)

            def __call__(self, signals):
                captured.update({name: value.detach().cpu().numpy() for name, value in signals.items()})
                return torch.tensor(logits).reshape(1, 1, 4).repeat(1, 1200, 1)

        with TemporaryDirectory() as directory:
            raw_csv = Path(directory) / "outputs" / "polar_wav2sleep" / "20200101T235940_ppg0_hypnodensity.csv"
            if input_source == "filtered":
                raw_csv.parent.mkdir(parents=True)
                raw_csv.write_text("previous raw result\n", encoding="utf-8")
            namespace = {
                "np": np, "pd": pd, "plt": plt, "Path": Path, "PROCESSING": Path(directory),
                "ppg_raw_0": raw, "ppg_filtered": filtered, "acc": acceleration,
            }
            with patch.object(wav2sleep, "load_model", return_value=FakeModel()) as loader, patch.object(plt, "show"):
                source = notebook_source("WAV2SLEEP_MODEL_REVISION =")
                source = source.replace('WAV2SLEEP_INPUT = "filtered"', f'WAV2SLEEP_INPUT = {input_source!r}', 1)
                exec(source, namespace)
            self.assertEqual(loader.call_args.args, ("hf://joncarter/wav2sleep",))
            self.assertEqual(loader.call_args.kwargs["revision"], "8b6e46ab3ef6e3945282b81f631933c59db0b205")
            result = namespace["hypnodensity"]
            self.assertTrue(namespace["w2s_output_csv"].exists())
            self.assertEqual("_filtered" in namespace["w2s_output_csv"].name, input_source == "filtered")
            if input_source == "filtered":
                self.assertEqual(raw_csv.read_text(encoding="utf-8"), "previous raw result\n")
            exported = pd.read_csv(namespace["w2s_output_csv"])
            self.assertEqual(len(exported), len(result))

        pd.testing.assert_series_equal(raw, original)
        np.testing.assert_array_equal(filtered, original_filtered)
        self.assertEqual(set(captured), {"PPG"})
        # Independent reference for interpolation, zero-padding, and global z-score.
        target_seconds = np.arange(1, 10 * 3600 * 1024 // 30 + 1) * 30 / 1024
        selected_values = raw.to_numpy(dtype=np.float32) if input_source == "raw" else filtered.astype(np.float32)
        expected = np.interp(target_seconds, t, selected_values, left=0, right=0).astype(np.float32)
        expected = (expected.astype(np.float64) - expected.mean(dtype=np.float64)) / expected.std(dtype=np.float64, ddof=1)
        np.testing.assert_allclose(captured["PPG"][0], expected, atol=2e-6)

        stages = ["Wake", "Light", "Deep", "REM"]
        expected_probability = np.exp(logits.astype(np.float64))
        expected_probability /= expected_probability.sum()
        np.testing.assert_allclose(result[stages], np.tile(expected_probability, (len(result), 1)), atol=1e-7)
        self.assertEqual(result.index[0], index[0])
        self.assertEqual(result["observed_end"].iloc[-1], index[0] + pd.Timedelta(seconds=duration_s))
        self.assertEqual(len(result), int(np.ceil(duration_s / 30)))
        self.assertTrue((np.diff(result.index.asi8) == 30_000_000_000).all())

        axes = namespace["axes_hypnodensity"]
        self.assertTrue(all(axes[0].get_shared_x_axes().joined(axes[0], ax) for ax in axes[1:]))
        np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), filtered)
        self.assertEqual(len(axes[0].lines), 3)
        self.assertEqual(len(axes[2].collections), 4)
        self.assertIn(f"{input_source} PPG_0 input", axes[2].get_title())
        plt.close(namespace["fig_hypnodensity"])
        return result

    def test_partial_epoch_retains_only_observed_duration(self):
        result = self.run_cell(65)
        self.assertEqual(list(result["is_complete_epoch"]), [True, True, False])
        np.testing.assert_allclose(result["observed_duration_s"], [30, 30, 5])

    def test_exact_epoch_boundary_does_not_add_padding_epoch(self):
        result = self.run_cell(60)
        self.assertEqual(list(result["is_complete_epoch"]), [True, True])
        np.testing.assert_allclose(result["observed_duration_s"], [30, 30])

    def test_filtered_input_reaches_model_and_preserves_raw_result(self):
        self.run_cell(65, input_source="filtered")


if __name__ == "__main__":
    unittest.main()
