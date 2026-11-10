"""Synthetic respiration modulation, quiet geometry and missing-data checks."""
from dataclasses import replace
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppg_respiration import (METHODS, RespirationConfig, analyze_respiration, derive_features,
                            extract_period_features, rate_spectrum, respiration_availability,
                            respiration_quiet_periods, respiratory_runs)
from respiration_plots import plot_comparison, plot_example, plot_waveforms, representative_window


START = pd.Timestamp("2020-01-01 23:50", tz="UTC")


def bounds(pairs):
    frame = pd.DataFrame([(START + pd.Timedelta(seconds=a), START + pd.Timedelta(seconds=b))
                          for a, b in pairs], columns=["start", "end"])
    for c in ("start", "end"):
        frame[c] = pd.to_datetime(frame[c], utc=True)
    return frame


def ppg_signal(duration=420, frequency=.2):
    t = np.arange(55 * duration) / 55
    # Independently modulate optical baseline, pulse amplitude, and cardiac phase.
    # Use resolvable PP modulation at the pipeline's 55 Hz detection grid.
    phase = 2 * np.pi * t + .30 * np.sin(2 * np.pi * frequency * t)
    waveform = (1000 + 30 * np.sin(2 * np.pi * frequency * t)
                + (50 + 10 * np.cos(2 * np.pi * frequency * t)) * np.sin(phase))
    return pd.Series(-waveform, index=START + pd.to_timedelta(t, unit="s"))


def features(duration=1400):
    pp = .9 + .025 * np.sin(2 * np.pi * .2 * np.arange(int(duration / .8)) * .9)
    ends = np.cumsum(pp)
    first = ends - pp
    keep = ends + .2 < duration
    pp, ends, first = pp[keep], ends[keep], first[keep]
    peak_t = ends + .18
    return pd.DataFrame({"pp_time": START + pd.to_timedelta((first + ends) / 2, unit="s"),
                         "pp_start": START + pd.to_timedelta(first, unit="s"),
                         "pp_end": START + pd.to_timedelta(ends, unit="s"),
                         "foot_time": START + pd.to_timedelta(ends, unit="s"),
                         "pwa_time": START + pd.to_timedelta(peak_t, unit="s"),
                         "pp_s": pp, "hr_bpm": 60 / pp, "valid": True, "artifact": False,
                         "RIIV": 1000 + 10 * np.sin(2 * np.pi * .2 * peak_t),
                         "RIAV": 50 + 5 * np.cos(2 * np.pi * .2 * peak_t), "RIFV": pp})


def fake_extractor(table):
    def extract(raw, config):
        part = table.loc[(table.pp_start >= raw.index[0]) & (table.pwa_time <= raw.index[-1])].copy()
        return part.reset_index(drop=True), {}
    return extract


class QuietPeriodTests(unittest.TestCase):
    def test_matches_hrv_short_burst_and_post_guard_rules(self):
        quiet, report = respiration_quiet_periods(bounds([(10, 12), (20, 21), (400, 402)]), bounds([(0, 1000)]))
        expected = bounds([(0, 10), (13, 400), (403, 1000)])
        self.assertEqual(list(quiet.start), list(expected.start))
        self.assertEqual(list(quiet.end), list(expected.end))
        self.assertEqual(report["ignored_short_bursts"], 1)
        self.assertEqual(report["retained_bursts"], 2)

    def test_recording_edges_are_explicitly_configurable(self):
        config = RespirationConfig(include_recording_edges=False)
        quiet, _ = respiration_quiet_periods(bounds([(10, 12), (400, 402)]), bounds([(0, 1000)]), config)
        self.assertEqual(len(quiet), 1)
        self.assertEqual(quiet.start.iloc[0], START + pd.Timedelta(seconds=13))
        empty, _ = respiration_quiet_periods(bounds([]), bounds([(0, 1000)]), config)
        self.assertTrue(empty.empty)

    def test_complement_with_no_movements_matches_hrv(self):
        quiet, _ = respiration_quiet_periods(bounds([]), bounds([(0, 1000)]))
        self.assertEqual(quiet.duration_s.sum(), 1000)

    def test_coverage_gaps_and_touching_boundaries_remain_separate(self):
        quiet, _ = respiration_quiet_periods(bounds([]), bounds([(0, 300), (300, 600), (700, 900)]))
        self.assertEqual(len(quiet), 3)
        self.assertEqual(list(quiet.coverage_id), [0, 1, 2])

    def test_configuration_rejects_impossible_settings(self):
        for args in ({"respiratory_band_hz": (.1, 2)}, {"max_window_s": 30},
                     {"post_burst_guard_s": -1}, {"max_bad_fraction": 1.1},
                     {"filter_edge_guard_s": -1}, {"ppg_fs": 10}, {"min_beats": 2}):
            with self.assertRaises(ValueError):
                RespirationConfig(**args)


class FeatureTests(unittest.TestCase):
    def fixture(self):
        t = np.arange(30 * 100) / 100
        waveform = pd.Series(1000 + 10 * np.cos(2 * np.pi * t), index=START + pd.to_timedelta(t, unit="s"))
        ends = np.arange(2, 29)
        table = pd.DataFrame({"pp_start": START + pd.to_timedelta(ends - 1.5, unit="s"),
                              "pp_end": START + pd.to_timedelta(ends - .5, unit="s"),
                              "pp_time": START + pd.to_timedelta(ends - 1, unit="s"),
                              "foot_time": START + pd.to_timedelta(ends - .5, unit="s"),
                              "pwa_time": START + pd.to_timedelta(ends, unit="s"),
                              "pp_s": 1., "valid": True})
        return waveform, table

    def test_baseline_amplitude_and_pp_are_separate_quantities(self):
        raw, table = self.fixture()
        result = derive_features(raw, table)
        np.testing.assert_allclose(result.RIIV, 1000)
        np.testing.assert_allclose(result.RIAV, 20)
        np.testing.assert_allclose(result.RIFV, 1)
        self.assertTrue(result.valid.all())
        pd.testing.assert_series_equal(result.pwa_time, table.pwa_time)
        pd.testing.assert_series_equal(result.pp_time, table.pp_time)

    def test_raw_dc_offset_survives_riiv_but_cancels_from_riav(self):
        raw, table = self.fixture()
        a, b = derive_features(raw, table), derive_features(raw + 10000, table)
        np.testing.assert_allclose(b.RIIV - a.RIIV, 10000)
        np.testing.assert_allclose(b.RIAV, a.RIAV)
        np.testing.assert_allclose(b.RIFV, a.RIFV)

    def test_fiducials_outside_observed_signal_are_not_extrapolated(self):
        raw, table = self.fixture()
        table.loc[0, "pp_start"] -= pd.Timedelta(seconds=2)
        with self.assertRaises(ValueError):
            derive_features(raw, table)

    def test_a_shifted_pp_clock_cannot_silently_misalign_rifv(self):
        raw, table = self.fixture()
        table["pp_time"] += pd.Timedelta(seconds=.1)
        with self.assertRaisesRegex(ValueError, "Inconsistent pulse pairing"):
            derive_features(raw, table)

    def test_full_shared_pulse_pipeline_retains_baseline_and_recovers_modulations(self):
        table, quality = extract_period_features(ppg_signal())
        self.assertGreater(len(table), 300)
        self.assertAlmostEqual(table.loc[table.valid, "RIIV"].mean(), 1000, delta=5)
        runs = list(respiratory_runs(table))
        self.assertTrue(runs)
        longest = max(runs, key=lambda run: len(run[0]))[0]
        self.assertGreaterEqual(len(longest), 240)
        for method in METHODS:
            metrics, _ = rate_spectrum(longest[method])
            self.assertAlmostEqual(metrics["candidate_rate_bpm"], 12, delta=1)


class SpectrumTests(unittest.TestCase):
    def test_known_frequency_and_actual_resolution_without_zero_padding(self):
        for duration in (60, 120, 300):
            t = np.arange(duration * 4) / 4
            m, _ = rate_spectrum(9000 + 2 * np.sin(2 * np.pi * .2 * t) + t * .01)
            self.assertAlmostEqual(m["candidate_rate_bpm"], 12)
            self.assertAlmostEqual(m["resolution_bpm"], 60 / duration)
            self.assertGreater(m["peak_power_fraction"], .95)

    def test_constant_modulation_cannot_generate_a_rate(self):
        with self.assertRaisesRegex(ValueError, "no_modulation_variance"):
            rate_spectrum(np.ones(1200) * 1000)

    def test_band_boundary_peak_is_identified(self):
        t = np.arange(1200) / 4
        m, _ = rate_spectrum(np.sin(2 * np.pi * .1 * t))
        self.assertTrue(m["peak_at_band_boundary"])


class AnalysisTests(unittest.TestCase):
    def analyze(self, table=None, config=RespirationConfig()):
        quiet, _ = respiration_quiet_periods(bounds([(0, 5), (650, 660), (1290, 1300)]), bounds([(0, 1300)]), config)
        return analyze_respiration(ppg_signal(1300), quiet, config,
                                   feature_extractor=fake_extractor(features() if table is None else table))

    def test_windows_remain_inside_one_period_and_continuous_run(self):
        result = self.analyze()
        self.assertFalse(result["windows"].empty)
        unique = result["windows"].drop_duplicates("window_id")
        for row in unique.itertuples():
            period = result["periods"].loc[row.quiet_period_id]
            self.assertGreaterEqual(row.start, period.start)
            self.assertLessEqual(row.end, period.end)
            self.assertGreaterEqual(row.duration_s, 60)
            self.assertLessEqual(row.duration_s, 300)
            self.assertEqual(row.n_grid_samples, round(row.duration_s * 4))
            signal = result["signals"].loc[result["signals"].signal_run_id == row.signal_run_id]
            self.assertGreaterEqual(row.start, signal.time.min())
            self.assertLessEqual(row.end, signal.time.max() + pd.Timedelta(seconds=.25))
        self.assertTrue((result["windows"].groupby("window_id").method.nunique() == 3).all())

    def test_long_bad_beat_gap_splits_filtering_and_windows(self):
        table = features()
        a, b = START + pd.Timedelta(seconds=200), START + pd.Timedelta(seconds=220)
        table.loc[(table.pp_time >= a) & (table.pp_time < b), "valid"] = False
        result = self.analyze(table)
        for window in result["windows"].itertuples():
            self.assertFalse(window.start < b and window.end > a)
        for _, run in result["signals"].groupby("signal_run_id"):
            self.assertFalse(run.time.min() < a and run.time.max() > b)

    def test_short_masked_gap_is_reported_and_quality_threshold_is_applied(self):
        table = features()
        table.loc[150, "valid"] = False
        result = self.analyze(table, RespirationConfig(max_bad_fraction=0))
        self.assertTrue(result["windows"].exclusion_reason.str.contains("flagged_beats").any())
        self.assertTrue((result["windows"].max_beat_gap_s > 1).any())

    def test_beat_sampling_alias_risk_is_flagged(self):
        table = features()
        t = (table.pwa_time - START).dt.total_seconds()
        table["RIIV"] = 1000 + 2 * np.sin(2 * np.pi * .48 * t)
        # At PP about 0.9s, safety*beat Nyquist ~=0.5Hz; 0.48 is below it.
        result = self.analyze(table, RespirationConfig(beat_nyquist_safety=.7))
        riiv = result["windows"].loc[result["windows"].method == "RIIV"]
        self.assertTrue(riiv.exclusion_reason.str.contains("nyquist").all())

    def test_weak_or_ambiguous_peak_keeps_candidate_but_withholds_accepted_rate(self):
        table = features()
        t = (table.pwa_time - START).dt.total_seconds()
        table["RIIV"] = 1000 + np.sin(2 * np.pi * .2 * t) + np.sin(2 * np.pi * .35 * t)
        result = self.analyze(table, RespirationConfig(min_peak_power_fraction=.95))
        rejected = result["windows"].loc[(result["windows"].method == "RIIV") & ~result["windows"].included]
        self.assertTrue(rejected.candidate_rate_bpm.notna().all())
        self.assertTrue(rejected.rate_bpm.isna().all())
        self.assertTrue(rejected.exclusion_reason.str.contains("weak_spectral").all())

    def test_equal_period_summary_and_unique_duration_are_distinct_from_window_counts(self):
        result = self.analyze()
        summary = respiration_availability(result)
        self.assertTrue((summary.unique_covered_min < summary.summed_window_min_counts_overlap).all())
        self.assertTrue((summary.contributing_quiet_periods > 0).all())
        self.assertEqual(result["period_rates"].groupby(["quiet_period_id", "method"]).size().max(), 1)

    def test_empty_results_and_plotting(self):
        quiet, _ = respiration_quiet_periods(bounds([]), bounds([]))
        result = analyze_respiration(ppg_signal(), quiet)
        self.assertTrue(result["windows"].empty)
        self.assertIsNone(representative_window(result))
        self.assertEqual(respiration_availability(result).included_windows.sum(), 0)
        for figure in (plot_example(ppg_signal(), result), plot_waveforms(result), plot_comparison(result)):
            figure.canvas.draw()
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
