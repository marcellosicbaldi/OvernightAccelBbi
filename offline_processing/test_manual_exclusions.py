"""Synthetic regression tests; no personal recordings or diary are required."""

import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis_windows import interval_ids, select_observations, subtract_intervals
from burst_hr_response import analyze_burst_hr
from detect_acc_bursts import select_bursts_by_state
from gp_hrv import interburst_hrv
from manual_exclusions import apply_exclusions, exclusion_report, prepare_exclusions


ORIGIN = pd.Timestamp("2025-01-01 23:00", tz="UTC")


def bounds(pairs):
    return pd.DataFrame({name: pd.to_datetime([ORIGIN + pd.Timedelta(seconds=p[i]) for p in pairs], utc=True)
                         for i, name in enumerate(("start", "end"))})


def synthetic_detection():
    bursts = bounds([(60, 62), (180, 182), (199, 201), (240, 242), (298, 302), (350, 352)])
    bursts["AUC"] = np.arange(1, len(bursts) + 1, dtype=float)
    bursts["duration"] = bursts.end - bursts.start
    bursts["analysis_interval_id"] = 0
    signal = pd.Series(1., index=pd.date_range(ORIGIN, periods=600, freq="s"))
    return {"bursts": bursts, "intervals": bounds([(0, 600)]), "quality": pd.DataFrame(),
            "signals": [{"magnitude": signal, "filtered": signal, "score": signal, "threshold": .04}]}


class ManualExclusionTests(unittest.TestCase):
    def test_local_dates_convert_to_utc_and_other_nights_are_ignored(self):
        periods = prepare_exclusions([
            ("2025-01-02 03:15", "2025-01-02 05:31", "awake"),
            ("2025-01-03 03:15", "2025-01-03 05:31", "other night"),
        ], ORIGIN, ORIGIN + pd.Timedelta(hours=8))
        self.assertEqual(periods.start.iloc[0], pd.Timestamp("2025-01-02 02:15Z"))
        self.assertEqual(periods.end.iloc[0], pd.Timestamp("2025-01-02 04:31Z"))
        self.assertEqual(len(periods), 1)
        self.assertEqual(exclusion_report(periods)["excluded_seconds"], 136 * 60)
        json.dumps(exclusion_report(periods))

    def test_overlap_union_clipping_and_midnight(self):
        periods = prepare_exclusions([
            ("2025-01-01 22:00Z", "2025-01-01 23:02Z", "awake"),
            ("2025-01-01 23:01Z", "2025-01-01 23:03Z", "invalid signal"),
            ("2025-01-01 23:03Z", "2025-01-01 23:04Z", "awake"),
        ], ORIGIN, ORIGIN + pd.Timedelta(minutes=10))
        pd.testing.assert_frame_equal(periods[["start", "end"]], bounds([(0, 240)]))
        self.assertEqual(periods.reason.iloc[0], "awake; invalid signal")
        midnight = prepare_exclusions([
            ("2025-01-01 23:55", "2025-01-02 00:05", "awake"),
        ], ORIGIN, ORIGIN + pd.Timedelta(minutes=10))
        pd.testing.assert_frame_equal(midnight[["start", "end"]], bounds([(0, 300)]))

    def test_invalid_or_undated_periods_fail_instead_of_guessing(self):
        for annotation in [("03:15", "05:31", "awake"),
                           ("2025-01-02 05:31", "2025-01-02 03:15", "awake"),
                           ("2025-01-02 03:15", "2025-01-02 03:15", "awake"),
                           ("2025-01-02 03:15", "2025-01-02 05:31", ""),
                           ("2025-01-02 03:15", "2025-01-02 05:31")]:
            with self.subTest(annotation=annotation), self.assertRaises(ValueError):
                prepare_exclusions([annotation], ORIGIN, ORIGIN + pd.Timedelta(hours=8))
        for date in ("2025-03-30", "2025-10-26"):
            with self.subTest(date=date), self.assertRaises(Exception):
                prepare_exclusions([(date + " 02:15", date + " 02:30", "awake")],
                                   ORIGIN, ORIGIN + pd.Timedelta(days=365))

    def test_subtraction_preserves_acquisition_and_touching_boundaries(self):
        original = bounds([(0, 100), (100, 150), (200, 300)])
        exclusions = bounds([(20, 40), (140, 210), (290, 400)])
        remaining = subtract_intervals(original, exclusions)
        pd.testing.assert_frame_equal(remaining.reset_index(drop=True),
                                      bounds([(0, 20), (40, 100), (100, 140), (210, 290)]))
        self.assertEqual(interval_ids(pd.DatetimeIndex([ORIGIN + pd.Timedelta(seconds=x)
                                                       for x in (19, 20, 39, 40, 99, 100)]),
                                      remaining).tolist(), [0, -1, -1, 1, 1, 2])
        self.assertTrue(subtract_intervals(original, bounds([(0, 400)])).empty)
        pd.testing.assert_frame_equal(subtract_intervals(original, bounds([])).reset_index(drop=True), original)

    def test_any_overlap_flags_whole_candidate_and_preserves_its_auc(self):
        raw = synthetic_detection()
        selected = select_bursts_by_state(raw, bounds([(0, 600)]), bounds([]))
        periods = bounds([(200, 300)]).assign(reason="awake")
        result = apply_exclusions(selected, periods)
        self.assertEqual(result["all_bursts"].excluded_manually.tolist(), [False, False, True, True, True, False])
        self.assertEqual(result["bursts"].candidate_id.tolist(), [0, 1, 5])
        self.assertEqual(result["bursts"].analysis_interval_id.tolist(), [0, 0, 1])
        pd.testing.assert_frame_equal(result["all_bursts"][["start", "end", "AUC"]],
                                      selected["all_bursts"][["start", "end", "AUC"]])
        self.assertNotIn("excluded_manually", selected["all_bursts"])
        self.assertEqual(len(result["signals"]), 2)
        for part in result["signals"]:
            self.assertTrue((interval_ids(part["score"].index, periods) < 0).all())
        # End touching and start touching are eligible; nanosecond overlap is excluded.
        touching = raw.copy()
        touching["bursts"] = bounds([(198, 200), (300, 302), (199, 200)]).assign(AUC=1., analysis_interval_id=0)
        touching["bursts"].loc[2, "end"] += pd.Timedelta(nanoseconds=1)
        touching = select_bursts_by_state(touching, bounds([(0, 600)]), bounds([]))
        self.assertEqual(apply_exclusions(touching, periods)["all_bursts"].excluded_manually.tolist(),
                         [False, False, True])

    def test_exclusions_override_whole_sleep_and_wake_selections(self):
        raw = synthetic_detection()
        for mode, expected_ids in [("whole", [0, 1, 5]), ("sleep_only", [0, 1]), ("wake_only", [5])]:
            with self.subTest(mode=mode):
                selected = select_bursts_by_state(raw, bounds([(0, 200)]), bounds([(200, 600)]), mode)
                result = apply_exclusions(selected, bounds([(200, 300)]).assign(reason="awake"))
                self.assertEqual(result["bursts"].candidate_id.tolist(), expected_ids)
                self.assertTrue(result["all_bursts"].loc[result["all_bursts"].excluded_manually,
                                                         "manual_exclusion_reason"].eq("awake").all())

    def test_no_exclusions_and_fully_excluded_night(self):
        selected = select_bursts_by_state(synthetic_detection(), bounds([(0, 600)]), bounds([]))
        unchanged = apply_exclusions(selected, bounds([]).assign(reason=pd.Series(dtype=str)))
        pd.testing.assert_frame_equal(unchanged["bursts"][selected["bursts"].columns], selected["bursts"])
        excluded = apply_exclusions(selected, bounds([(0, 600)]).assign(reason="awake"))
        self.assertTrue(excluded["intervals"].empty)
        self.assertTrue(excluded["bursts"].empty)
        self.assertEqual(excluded["signals"], [])
        self.assertTrue(excluded["all_bursts"].excluded_manually.all())
        empty = select_bursts_by_state(synthetic_detection(), bounds([]), bounds([]), "sleep_only")
        self.assertTrue(apply_exclusions(empty, bounds([]).assign(reason=pd.Series(dtype=str)))["bursts"].empty)

    def test_hr_epochs_and_hrv_cleaning_cannot_cross_exclusions(self):
        selected = select_bursts_by_state(synthetic_detection(), bounds([(0, 600)]), bounds([]))
        periods = bounds([(200, 300)]).assign(reason="awake")
        selected = apply_exclusions(selected, periods)
        observations = pd.DataFrame({"hr_bpm": 60., "break_before": False},
                                    index=pd.date_range(ORIGIN, periods=600, freq="s"))
        kept = select_observations(observations, selected["intervals"])
        self.assertEqual(len(kept), 500)
        self.assertTrue(kept.loc[ORIGIN + pd.Timedelta(seconds=300), "break_before"])
        hr = analyze_burst_hr(selected["bursts"], kept, ORIGIN, ORIGIN + pd.Timedelta(seconds=600),
                              analysis_intervals=selected["intervals"], movement_bursts=selected["all_bursts"])
        self.assertEqual(hr["events"].candidate_id.tolist(), [0, 1, 5])
        self.assertFalse(hr["events"].loc[1, "included"])
        self.assertIn("analysis_interval_edge", hr["events"].loc[1, "exclusion_reason"])
        self.assertTrue(hr["epochs_bpm"].loc[1, 20:49].isna().all())
        rows = [{"sequence": i + 1, "bbi_ms": 1000.,
                 "callback_time_utc_approx": (ORIGIN + pd.Timedelta(seconds=i)).isoformat()} for i in range(600)]
        hrv = interburst_hrv(rows, selected["all_bursts"], ORIGIN, ORIGIN + pd.Timedelta(seconds=600),
                             analysis_intervals=selected["intervals"])
        self.assertFalse(hrv["windows"].empty)
        self.assertEqual(len(hrv["intervals"]), 500)
        for table in (hrv["segments"], hrv["windows"]):
            self.assertFalse(((table.start < periods.end.iloc[0]) & (table.end > periods.start.iloc[0])).any())
        # The discarded crossing movement still blocks the first two seconds after wake.
        self.assertFalse(((hrv["segments"].start < ORIGIN + pd.Timedelta(seconds=303))
                          & (hrv["segments"].end > ORIGIN + pd.Timedelta(seconds=300))).any())

    def test_notebook_settings_load_private_annotations(self):
        notebook = json.loads(Path(__file__).with_name("load_fit_test.ipynb").read_text(encoding="utf-8"))
        settings = next(c for c in notebook["cells"] if c.get("id") == "manual-exclusion-settings")
        with tempfile.TemporaryDirectory() as directory:
            processing = Path(directory)
            (processing / "analysis_exclusions.local.json").write_text(json.dumps([
                ["2025-01-02 00:03:20", "2025-01-02 00:05:00", "awake"],
            ]), encoding="utf-8")
            context = {"pd": pd, "np": np, "importlib": importlib, "display": lambda *_: None,
                       "print": lambda *_: None,
                       "processing_dir": processing, "sleep_start_utc": ORIGIN,
                       "sleep_end_utc": ORIGIN + pd.Timedelta(seconds=600), "LOCAL_TZ": "Europe/Rome"}
            exec("".join(settings["source"]), context)
            pd.testing.assert_frame_equal(context["manual_exclusion_intervals"][["start", "end"]],
                                          bounds([(200, 300)]))


if __name__ == "__main__":
    unittest.main()
