"""Synthetic tests for Captum window attribution; no checkpoints or recordings."""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from polar_interpretability import (MODEL_FS, STAGES, aggregate_attributions, candidate_epochs,
                                    checkpointed_encoder, epoch_table, explain_epoch, validate_ppg, window_bounds)

try:
    import torch
    from torch import nn
    import captum
except ImportError:
    torch = nn = captum = None


class TimingTests(unittest.TestCase):
    def test_partial_epoch_and_ranking_exclude_padding(self):
        p = np.tile([0.4, 0.3, 0.2, 0.1], (1200, 1))
        p[1] = [0.25] * 4
        table = epoch_table(p, pd.Timestamp("2020-01-01 23:59:40"), 65)
        self.assertEqual(list(table["observed_duration_s"]), [30, 30, 5])
        self.assertEqual(list(table["is_complete_epoch"]), [True, True, False])
        self.assertEqual(len(candidate_epochs(table)), 2)
        self.assertEqual(candidate_epochs(table).iloc[0]["epoch"], 1)
        raw = table.copy()
        raw.loc[raw.index[0], STAGES] = [0.1, 0.7, 0.1, 0.1]
        self.assertEqual(candidate_epochs(table, raw).iloc[0]["epoch"], 0)
        self.assertEqual(table.index[1].day, 2)

    def test_window_excludes_partial_and_padded_samples(self):
        self.assertEqual(window_bounds(1, 3 * 1024 + 17, 0), (1024, 2048))
        self.assertEqual(window_bounds(2, 3 * 1024 + 17, 30), (1024, 3 * 1024 + 17))
        with self.assertRaisesRegex(ValueError, "complete observed epoch"):
            window_bounds(3, 3 * 1024 + 17, 0)
        # The upstream right-edge sample at 60s may be padded even for a 60s record.
        self.assertEqual(window_bounds(1, 2047, 0, duration_s=60), (1024, 2047))

    def test_bad_timing_and_long_gaps_fail_explicitly(self):
        ppg = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="20ms"))
        np.testing.assert_allclose(validate_ppg(ppg), [0, 0.02, 0.04])
        bad = ppg.copy()
        bad.index = pd.DatetimeIndex([ppg.index[0], ppg.index[1], ppg.index[1]])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_ppg(bad)
        bad.index = pd.date_range("2020-01-01", periods=3, freq="2s")
        with self.assertRaisesRegex(ValueError, "gap"):
            validate_ppg(bad)

    def test_aggregation_preserves_ig_sum_but_does_not_sum_occlusion(self):
        samples = pd.DataFrame({"seconds": (np.arange(68) + 1) / MODEL_FS,
                                "integrated_gradients": np.linspace(-1, 1, 68),
                                "occlusion_logit_change": np.full(68, 3.0)})
        summary = aggregate_attributions(samples)
        self.assertAlmostEqual(summary["signed_ig"].sum(), samples["integrated_gradients"].sum())
        np.testing.assert_allclose(summary["occlusion_logit_change"], 3)


@unittest.skipIf(captum is None, "Optional Captum dependencies are not installed")
class AttributionTests(unittest.TestCase):
    def make_model(self, nonlinear=False):
        class ToyNight(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.seen = []

            def forward(self, signals):
                signal = signals["PPG"]
                self.seen.append(signal.detach().clone())
                epochs = signal.reshape(-1, 4, 1024)
                weights = torch.linspace(0.1, 1, 1024, device=signal.device)
                score = epochs.round().mean(-1) if nonlinear else (epochs * weights).mean(-1)
                # Context outside the explanation window also affects the output.
                score = self.scale * (score + 0.1 * signal.mean(-1, keepdim=True))
                return torch.stack([score, -score, score * 0.3, score * 0], dim=-1)

        return ToyNight().eval()

    def test_local_ig_completeness_occlusion_and_fixed_context_for_both_references(self):
        full = torch.sin(torch.arange(4096) / 37).reshape(1, -1)
        full[:, 3072:] = -0.75  # Nonzero normalized padding must remain exactly fixed.
        original = full.clone()
        for baseline in ["mean", "reversed"]:
            with self.subTest(baseline=baseline):
                model = self.make_model()
                result = explain_epoch(model, full, 1, 3072, stage="Wake", contrast="Light",
                                       baseline=baseline, context_seconds=0, occlusion_seconds=10,
                                       n_steps=4, max_steps=4, atol=1e-5, rtol=1e-5,
                                       checkpoint=False, progress=None)
                self.assertTrue(result.diagnostics["converged"])
                self.assertAlmostEqual(result.diagnostics["ig_sum"], result.diagnostics["score_difference"], places=5)
                self.assertTrue(model.scale.requires_grad)  # Original parameter state restored.
                self.assertEqual(result.samples["sample_index"].iloc[0], 1024)
                self.assertEqual(result.samples["sample_index"].iloc[-1], 2047)
                for seen in model.seen:
                    torch.testing.assert_close(seen[:, :1024], original[:, :1024])
                    torch.testing.assert_close(seen[:, 2048:], original[:, 2048:])
                # Independently replace the first occlusion block and check its score change.
                window = original[:, 1024:2048]
                reference = window.mean().expand_as(window) if baseline == "mean" else window.flip(-1)
                replacement = original.clone()
                block = result.diagnostics["occlusion_block_samples"]
                replacement[:, 1024:1024 + block] = reference[:, :block]
                with torch.no_grad():
                    a, b = model({"PPG": original}), model({"PPG": replacement})
                    expected = float(((a[0, 1, 0] - a[0, 1, 1]) - (b[0, 1, 0] - b[0, 1, 1])).item())
                np.testing.assert_allclose(result.samples["occlusion_logit_change"].iloc[:block], expected, atol=1e-6)
        torch.testing.assert_close(full, original)

    def test_failed_convergence_is_reported(self):
        full = torch.tensor([0.4, 1.3]).repeat(2048).reshape(1, -1)
        result = explain_epoch(self.make_model(nonlinear=True), full, 1, 3072,
                               baseline="mean", context_seconds=0, occlusion_seconds=30,
                               n_steps=2, max_steps=4, atol=1e-8, rtol=1e-8,
                               checkpoint=False, progress=None)
        self.assertFalse(result.diagnostics["converged"])
        self.assertEqual(result.diagnostics["n_steps"], 4)
        self.assertGreater(abs(result.diagnostics["convergence_delta"]), 0.1)

    def test_inference_mode_is_rejected_for_gradient_attribution(self):
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, "inference_mode"):
            explain_epoch(self.make_model(), torch.ones(1, 4096), 1, 3072,
                          checkpoint=False, progress=None)

    def test_checkpointing_preserves_forward_gradient_and_restores_module(self):
        encoder = nn.Module()
        encoder.cnn = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
        original = encoder.cnn
        model = SimpleNamespace(signal_encoders=SimpleNamespace(get_encoder=lambda name: encoder))
        x = torch.linspace(-1, 1, 8).reshape(1, -1).requires_grad_()
        expected = encoder.cnn(x)
        expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
        with checkpointed_encoder(model):
            actual = encoder.cnn(x)
            actual_gradient = torch.autograd.grad(actual.sum(), x)[0]
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertIs(encoder.cnn, original)
        with self.assertRaisesRegex(RuntimeError, "synthetic"), checkpointed_encoder(model):
            raise RuntimeError("synthetic failure")
        self.assertIs(encoder.cnn, original)


if __name__ == "__main__":
    unittest.main()
