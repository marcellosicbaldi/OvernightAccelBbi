"""Synthetic checks for direct raw-PPG RIIV, adaptive cutoffs and integration."""
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from direct_riiv import DIRECT_METHOD, DirectRIIVConfig, add_direct_riiv, direct_riiv_window
from ppg_respiration import METHODS, RespirationConfig, analyze_respiration, rate_spectrum, respiration_quiet_periods
from respiration_plots import plot_comparison, plot_direct_filter, plot_example
import test_ppg_respiration as fixtures
from test_ppg_respiration import START, bounds, ppg_signal


def raw_signal(duration=420, respiratory_hz=.2, cardiac_hz=1):
    t = np.arange(55 * duration) / 55
    values = 1000 + 30 * np.sin(2 * np.pi * respiratory_hz * t) + 100 * np.sin(2 * np.pi * cardiac_hz * t)
    return pd.Series(-values, index=START + pd.to_timedelta(t, unit="s"))


class RawFilterTests(unittest.TestCase):
    def extract(self, raw=None, hr=60, direct=DirectRIIVConfig()):
        return direct_riiv_window(raw_signal() if raw is None else raw,
                                  START + pd.Timedelta(seconds=40), START + pd.Timedelta(seconds=340),
                                  START, START + pd.Timedelta(seconds=420), np.full(200, hr), direct=direct)

    def test_preserves_respiration_and_suppresses_cardiac_carrier_without_phase_shift(self):
        frame, info, spectrum = self.extract()
        t = (frame.time - START).dt.total_seconds().to_numpy()
        expected = 30 * np.sin(2 * np.pi * .2 * t)
        self.assertGreater(np.corrcoef(frame.RIIV_direct_raw, expected)[0, 1], .999)
        self.assertAlmostEqual(np.std(frame.RIIV_direct_raw) / np.std(expected), 1, delta=.002)
        cardiac = np.argmin(abs(spectrum.frequency_hz.to_numpy() - 1))
        self.assertLess(spectrum.lowpass_power.iloc[cardiac] / spectrum.raw_power.iloc[cardiac], 1e-4)
        self.assertGreater(info["cardiac_attenuation_db_theoretical"], 40)
        self.assertEqual(rate_spectrum(frame.RIIV_direct_raw)[0]["candidate_rate_bpm"], 12)

    def test_cutoff_tracks_hr_in_hz_and_is_capped_before_downsampling(self):
        _, a, _ = self.extract(hr=45)
        _, b, _ = self.extract(hr=60)
        _, c, _ = self.extract(hr=120)
        self.assertAlmostEqual(a["lowpass_cutoff_hz"], .525)
        self.assertAlmostEqual(b["lowpass_cutoff_hz"], .7)
        self.assertAlmostEqual(c["lowpass_cutoff_hz"], 1.)

    def test_exact_four_hz_window_and_context_bounds(self):
        frame, info, _ = self.extract()
        self.assertEqual(len(frame), 1200)
        self.assertEqual(frame.time.iloc[0], START + pd.Timedelta(seconds=40))
        np.testing.assert_allclose(np.diff(pd.DatetimeIndex(frame.time).asi8) / 1e9, .25)
        self.assertGreaterEqual(info["raw_context_start"], START + pd.Timedelta(seconds=10))
        self.assertLess(info["raw_context_end"], START + pd.Timedelta(seconds=370))

    def test_other_quiet_periods_cannot_affect_filtering(self):
        raw = raw_signal()
        a, b = START + pd.Timedelta(seconds=25), START + pd.Timedelta(seconds=355)
        args = (START + pd.Timedelta(seconds=40), START + pd.Timedelta(seconds=340), a, b, np.full(200, 60))
        before = direct_riiv_window(raw, *args)[0]
        modified = raw.copy()
        modified.loc[(modified.index < a) | (modified.index >= b)] *= 1000
        after = direct_riiv_window(modified, *args)[0]
        pd.testing.assert_frame_equal(before, after)

    def test_raw_gaps_and_invalid_hr_fail_explicitly(self):
        raw = raw_signal().drop(raw_signal().index[55 * 100:55 * 101])
        with self.assertRaisesRegex(ValueError, "raw_ppg_gap"):
            self.extract(raw)
        for hr in (np.nan, 0, -1):
            with self.assertRaises(ValueError):
                self.extract(hr=hr)
        with self.assertRaisesRegex(ValueError, "no_separated"):
            self.extract(hr=6)

    def test_invalid_filter_options_are_rejected(self):
        for options in ({"cutoff_hr_fraction": 1}, {"hr_quantile": .9}, {"order": 0},
                        {"context_s": 0}, {"peak_cutoff_fraction": 1}, {"min_cardiac_attenuation_db": -1}):
            with self.assertRaises(ValueError):
                DirectRIIVConfig(**options)
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            self.extract(direct=DirectRIIVConfig(max_cutoff_hz=2))


class IntegratedTests(unittest.TestCase):
    def setUp(self):
        self.base = fixtures.AnalysisTests().analyze()
        self.base["features"]["raw_peak"] = self.base["features"].RIIV + self.base["features"].RIAV / 2
        self.base["features"]["raw_foot"] = self.base["features"].RIIV - self.base["features"].RIAV / 2
        self.raw = ppg_signal(1300)

    def test_appends_fourth_method_on_same_windows_without_altering_originals(self):
        result = add_direct_riiv(self.base, self.raw)
        original = self.base["windows"]
        preserved = result["windows"].loc[result["windows"].method.isin(METHODS), original.columns]
        pd.testing.assert_frame_equal(original, preserved.reset_index(drop=True))
        self.assertEqual(result["windows"].window_id.nunique(), original.window_id.nunique())
        self.assertTrue((result["windows"].groupby("window_id").method.nunique() == 4).all())
        self.assertTrue(result["comparison"].all_feature_methods_included.all())
        self.assertGreater(result["comparison"].all_methods_included.sum(), 0)

    def test_native_ppg_does_not_receive_beat_sampling_nyquist_rejection(self):
        base = fixtures.AnalysisTests().analyze(config=RespirationConfig(beat_nyquist_safety=.1))
        self.assertFalse(base["windows"].included.any())
        result = add_direct_riiv(base, self.raw)
        direct = result["windows"].loc[result["windows"].method == DIRECT_METHOD]
        self.assertTrue(direct.included.any())
        self.assertTrue(direct.beat_nyquist_hz_approx.isna().all())
        self.assertFalse(direct.exclusion_reason.str.contains("nyquist").any())

    def test_does_not_inherit_feature_baseline_spectral_rejection(self):
        base = dict(self.base)
        base["windows"] = base["windows"].copy()
        mask = base["windows"].method == "RIIV"
        base["windows"].loc[mask, "included"] = False
        base["windows"].loc[mask, "exclusion_reason"] = "weak_spectral_concentration"
        result = add_direct_riiv(base, self.raw)
        self.assertTrue(result["windows"].loc[result["windows"].method == DIRECT_METHOD, "included"].any())

    def test_near_cutoff_peaks_and_inadequate_cardiac_attenuation_are_flagged(self):
        result = add_direct_riiv(self.base, raw_signal(1300, respiratory_hz=.4),
                                  DirectRIIVConfig(cutoff_hr_fraction=.3))
        direct = result["windows"].loc[result["windows"].method == DIRECT_METHOD]
        self.assertTrue(direct.exclusion_reason.str.contains("near_lowpass_cutoff").any())
        result = add_direct_riiv(self.base, self.raw, DirectRIIVConfig(min_cardiac_attenuation_db=100))
        direct = result["windows"].loc[result["windows"].method == DIRECT_METHOD]
        self.assertTrue(direct.exclusion_reason.str.contains("insufficient_cardiac_attenuation").all())
        self.assertTrue(direct.rate_bpm.isna().all())
        self.assertTrue(direct.candidate_rate_bpm.notna().all())

    def test_repeated_extension_is_idempotent_and_overlapping_windows_stay_separate(self):
        once = add_direct_riiv(self.base, self.raw)
        twice = add_direct_riiv(once, self.raw)
        pd.testing.assert_frame_equal(once["windows"], twice["windows"])
        self.assertFalse(twice["direct_signals"].duplicated(["window_id", "time"]).any())
        self.assertTrue(twice["direct_signals"].time.duplicated().any())

    def test_empty_data_and_updated_plots(self):
        quiet, _ = respiration_quiet_periods(bounds([]), bounds([]))
        empty = add_direct_riiv(analyze_respiration(self.raw, quiet), self.raw)
        result = add_direct_riiv(self.base, self.raw)
        for target in (empty, result):
            for figure in (plot_direct_filter(self.raw, target), plot_example(self.raw, target), plot_comparison(target)):
                figure.canvas.draw()
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
