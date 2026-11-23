"""Synthetic regressions for export timing, missing nights, and raw-data zoom."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nightly_signals import Night, SIGNALS, load_nights, overview_indices, plot_night


class ExportTests(unittest.TestCase):
    def write_exports(self, root):
        windows = [
            {"diary_date": "2026-11-11", "start": "2026-11-11T23:45:00+01:00",
             "end": "2026-11-12T06:30:00+01:00"},
            # The diary date labels the night even when bedtime is after midnight.
            {"diary_date": "2026-11-14", "start": "2026-11-15T00:05:00+01:00",
             "end": "2026-11-15T08:00:00+01:00"},
        ]
        starts = [pd.Timestamp(window["start"]) for window in windows]
        ends = [pd.Timestamp(window["end"]) for window in windows]
        for key, (_, column, _) in SIGNALS.items():
            times = []
            for start, end in zip(starts, ends):
                times.extend([start - pd.Timedelta(seconds=1), start, start,
                              start + pd.Timedelta(seconds=1), end, end + pd.Timedelta(seconds=1)])
            if key == "verity_sense_ppg":
                times = times[6:]
            index = pd.DatetimeIndex(times).tz_convert("UTC")
            naive = key == "empatica_ppg"
            if naive:
                index = index.tz_convert("Europe/Rome").tz_localize(None)
            field = "index" if naive else "datetime"
            table = pa.table({field: index, column: np.arange(len(index))})
            metadata = {
                "subject": "synthetic", "signal": key,
                "diary_timezone": "Europe/Rome", "source_timestamp_field": field,
                "naive_source_timestamp_timezone": "Europe/Rome" if naive else None,
                "sleep_windows_local": windows,
            }
            table = table.replace_schema_metadata({
                b"night_data_export": json.dumps(metadata).encode(),
                b"PANDAS_ATTRS": json.dumps({"units": {column: "uV" if column == "ecg" else "a.u."}}).encode(),
            })
            pq.write_table(table, root / f"synthetic_{key}.parquet")
        return windows

    def test_timezone_alignment_inclusive_bounds_duplicates_and_missing_sensor(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_exports(root)
            (root / "._synthetic_h10_ecg.parquet").write_bytes(b"ignored AppleDouble file")
            nights = load_nights(root)
        self.assertEqual(len(nights), 2)
        first, second = nights
        self.assertEqual(str(first.signals["empatica_ppg"].index.tz), "Europe/Rome")
        pd.testing.assert_index_equal(first.signals["empatica_ppg"].index,
                                      first.signals["h10_ecg"].index)
        self.assertEqual(len(first.signals["h10_ecg"]), 4)
        self.assertTrue(first.signals["h10_ecg"].index.has_duplicates)
        self.assertTrue(first.signals["verity_sense_ppg"].empty)
        self.assertEqual(len(second.signals["verity_sense_ppg"]), 4)
        self.assertEqual(second.diary_date, "2026-11-14")
        self.assertEqual(second.start.day, 15)
        self.assertEqual(second.units["h10_ecg"], "uV")

    def test_inconsistent_windows_fail_instead_of_misaligning_signals(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_exports(root)
            path = root / "synthetic_empatica_ppg.parquet"
            table = pq.read_table(path)
            metadata = dict(table.schema.metadata)
            export = json.loads(metadata[b"night_data_export"])
            export["sleep_windows_local"][0]["start"] = "2026-11-11T23:46:00+01:00"
            metadata[b"night_data_export"] = json.dumps(export).encode()
            pq.write_table(table.replace_schema_metadata(metadata), path)
            with self.assertRaisesRegex(ValueError, "Sleep windows/timezone disagree"):
                load_nights(root)


class PlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def make_night(self, times, values):
        signal = pd.Series(values, index=times)
        return Night("synthetic", "2026-11-11", times[0], times[-1], "Europe/Rome",
                     {key: signal for key in SIGNALS}, {key: "a.u." for key in SIGNALS})

    def test_overview_preserves_peaks_troughs_and_nan_boundaries(self):
        values = np.zeros(50_000)
        values[137], values[4097] = 999, -888
        values[20_000:21_000] = np.nan
        selected = overview_indices(values, max_points=100)
        self.assertTrue({0, 137, 4097, 19_999, 20_000, 20_999, 21_000, 49_999} <= set(selected))
        self.assertTrue(np.all(np.diff(selected) > 0))
        self.assertLess(len(selected), 150)

    def test_shared_zoom_restores_original_samples_in_all_panels(self):
        times = pd.date_range("2026-11-11 23:45", periods=1001, freq="10ms", tz="Europe/Rome")
        values = np.sin(np.arange(len(times)))
        figure = plot_night(self.make_night(times, values), max_points=20)
        self.assertEqual(len(figure.axes), 3)
        self.assertLess(len(figure.axes[0].lines[0].get_ydata()), 30)
        bounds = mdates.date2num(times[[200, 204]].tz_convert("UTC").tz_localize(None).to_numpy())
        figure.axes[-1].set_xlim(*bounds)
        figure.canvas.draw()
        for axis in figure.axes:
            rendered = axis.lines[0].get_ydata()
            self.assertGreaterEqual(len(rendered), 5)
            self.assertLess(len(rendered), 10)
            self.assertTrue(set(values[200:205]) <= set(rendered))

    def test_gaps_break_lines_and_missing_sensor_has_explicit_label(self):
        start = pd.Timestamp("2026-11-11 23:45", tz="Europe/Rome")
        times = pd.DatetimeIndex(start + pd.to_timedelta([0, 0.01, 0.02, 5, 5.01], unit="s"))
        night = self.make_night(times, np.arange(5))
        night.signals["verity_sense_ppg"] = night.signals["verity_sense_ppg"].iloc[:0]
        figure = plot_night(night)
        y = figure.axes[0].lines[0].get_ydata()
        np.testing.assert_equal(y, [0, 1, 2, np.nan, 3, 4])
        self.assertIn("No samples", figure.axes[1].texts[0].get_text())


if __name__ == "__main__":
    unittest.main()
