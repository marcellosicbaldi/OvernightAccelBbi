"""Local counterfactual explanations of full-night, PPG-only wav2sleep scores.

Captum is imported only when attribution is requested. Model input is the fixed,
already normalized full night. Only a selected observed window is replaced;
normalization, surrounding observations, and trailing padding stay unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd

STAGES = ["Wake", "Light", "Deep", "REM"]
MODEL_URI = "hf://joncarter/wav2sleep"
MODEL_REVISION = "8b6e46ab3ef6e3945282b81f631933c59db0b205"
SAMPLES_PER_EPOCH = 1024
MODEL_FS = SAMPLES_PER_EPOCH / 30


def validate_ppg(ppg):
    if not isinstance(ppg, pd.Series) or not isinstance(ppg.index, pd.DatetimeIndex) or len(ppg) < 2:
        raise ValueError("PPG must be a Series with at least two datetime-indexed samples.")
    if ppg.index.hasnans or not ppg.index.is_unique or not ppg.index.is_monotonic_increasing:
        raise ValueError("PPG timestamps must be valid, unique, and increasing.")
    values = ppg.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or np.ptp(values) == 0:
        raise ValueError("PPG samples must be finite and nonconstant.")
    seconds = (ppg.index.asi8 - ppg.index[0].value) / 1e9
    if np.diff(seconds).max() > 1:
        raise ValueError("PPG contains a gap >1 s. Select a continuous recording before filtering/inference.")
    return seconds


def load_polar_recording(folder):
    """Read the CSV layout used in polar.ipynb; sensor time is ns since 2000-01-01."""
    folder = Path(folder)

    def read(kind, legacy):
        directory = folder / kind
        if not directory.is_dir():
            directory = folder / legacy
        files = sorted(directory.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No {kind} CSV files in {directory}")
        frame = pd.concat([pd.read_csv(path, index_col=0) for path in files])
        frame.index = pd.to_datetime(frame.index, origin="2000-01-01", unit="ns")
        return frame

    optical = read("verity_ppg", "ppg")
    ppg = optical.iloc[:, 0].rename("PPG_0")
    validate_ppg(ppg)
    acc = read("verity_acc", "acc").iloc[:, :3].copy()
    if acc.shape[1] != 3:
        raise ValueError("ACC needs three raw axes.")
    acc.columns = ["x", "y", "z"]
    if acc.index.hasnans or not acc.index.is_unique or not acc.index.is_monotonic_increasing:
        raise ValueError("ACC timestamps must be valid, unique, and increasing.")
    return ppg, acc


def prepare_input(ppg, max_hours=10):
    """Reuse upstream interpolation, padding, float32 conversion, and global z-score."""
    from wav2sleep.data.dataset import ParquetDataset
    from wav2sleep.data.preprocessing import process_waveform_dataframe

    seconds = validate_ppg(ppg)
    duration = float(seconds[-1] + np.median(np.diff(seconds)))
    if duration > max_hours * 3600:
        raise ValueError("Recording exceeds the configured model length; select a shorter recording.")
    processed = process_waveform_dataframe(
        pd.DataFrame({"PPG": ppg.to_numpy(dtype=np.float32)}, index=seconds),
        columns=["PPG"], max_length_hours=max_hours,
    )
    with TemporaryDirectory(prefix="polar_captum_") as directory:
        path = Path(directory) / "PPG.parquet"
        processed.to_parquet(path)
        dataset = ParquetDataset([str(path)], columns=["PPG"], require_labels=False, causal=False)
        signals, _ = dataset[0]
    # Only samples actually interpolated inside the recording may be perturbed.
    observed_samples = int(np.searchsorted(processed.index.to_numpy(), seconds[-1], side="right"))
    return signals["PPG"].unsqueeze(0), duration, observed_samples


def epoch_table(probabilities, start, duration):
    count = int(np.ceil(duration / 30))
    if probabilities.ndim != 2 or probabilities.shape[1] != 4 or len(probabilities) < count:
        raise ValueError("Expected enough four-class probability rows for the recording.")
    probabilities = probabilities[:count]
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative.")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    begins = np.arange(count) * 30.0
    observed_ends = np.minimum(begins + 30, duration)
    result = pd.DataFrame(probabilities, columns=STAGES,
                          index=pd.DatetimeIndex(start + pd.to_timedelta(begins, unit="s"), name="epoch_start"))
    result["epoch_end"] = start + pd.to_timedelta(begins + 30, unit="s")
    result["observed_end"] = start + pd.to_timedelta(observed_ends, unit="s")
    result["observed_duration_s"] = observed_ends - begins
    result["is_complete_epoch"] = np.isclose(observed_ends - begins, 30, rtol=0, atol=1e-6)
    return result


def candidate_epochs(hypnodensity, raw_comparison=None):
    """Rank complete epochs; disagreements first, then high entropy. Not a quality score."""
    p = hypnodensity[STAGES].to_numpy()
    table = pd.DataFrame({"epoch": np.arange(len(p)), "time": hypnodensity.index,
                          "predicted_stage": np.asarray(STAGES)[p.argmax(axis=1)],
                          "probability": p.max(axis=1),
                          "entropy": -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1),
                          "raw_disagrees": False})
    if raw_comparison is not None:
        pd.testing.assert_index_equal(raw_comparison.index, hypnodensity.index)
        table["raw_disagrees"] = raw_comparison[STAGES].to_numpy().argmax(axis=1) != p.argmax(axis=1)
    table = table.loc[hypnodensity["is_complete_epoch"].to_numpy()]
    return table.sort_values(["raw_disagrees", "entropy"], ascending=False, kind="stable").reset_index(drop=True)


def window_bounds(epoch, observed_samples, context_seconds=30, duration_s=None):
    if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 0:
        raise ValueError("Epoch must be a nonnegative integer.")
    complete = ((epoch + 1) * SAMPLES_PER_EPOCH <= observed_samples if duration_s is None
                else (epoch + 1) * 30 <= duration_s + 1e-8)
    if not complete or epoch * SAMPLES_PER_EPOCH >= observed_samples:
        raise ValueError("Select a complete observed epoch; partial/padded epochs are excluded.")
    if not np.isfinite(context_seconds) or context_seconds < 0:
        raise ValueError("Context must be finite and nonnegative.")
    extra = int(round(context_seconds * MODEL_FS))
    return max(0, int(epoch) * SAMPLES_PER_EPOCH - extra), min(observed_samples, (int(epoch) + 1) * SAMPLES_PER_EPOCH + extra)


def make_baseline(window, name):
    """References in fixed normalized model-input units; neither is a known sleep state."""
    if name == "mean":
        return window.mean(dim=-1, keepdim=True).expand_as(window).clone()
    if name == "reversed":
        return window.flip(-1).clone()
    raise ValueError('Baseline must be "mean" or "reversed".')


@contextmanager
def checkpointed_encoder(model, enabled=True):
    """Recompute CNN blocks during backward to reduce memory; restore the original module."""
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint_sequential

    class CheckpointCNN(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = layers

        def forward(self, x):
            if torch.is_grad_enabled() and x.requires_grad:
                return checkpoint_sequential(self.layers, len(self.layers), x, use_reentrant=False)
            return self.layers(x)

    encoder = model.signal_encoders.get_encoder("PPG") if enabled else None
    original = encoder.cnn if encoder is not None else None
    if encoder is not None:
        encoder.cnn = CheckpointCNN(original)
    try:
        yield
    finally:
        if encoder is not None:
            encoder.cnn = original


@dataclass
class Explanation:
    samples: pd.DataFrame
    diagnostics: dict


def explain_epoch(model, full_input, epoch, observed_samples, *, stage=None, contrast=None,
                  baseline="mean", context_seconds=30, occlusion_seconds=10,
                  n_steps=32, max_steps=128, atol=0.02, rtol=0.05,
                  checkpoint=True, duration_s=None, progress=print):
    """Explain a local replacement while preserving the exact full-night prediction.

    IG completeness: sum(attributions) ~= score(original) - score(local reference).
    Occlusion values are logit differences per replaced block, repeated per sample;
    they must NOT be summed across samples. No renormalization occurs here.
    """
    import torch
    from captum.attr import IntegratedGradients, Occlusion

    if torch.is_inference_mode_enabled():
        raise RuntimeError("Attribution must run outside torch.inference_mode().")
    if model.training:
        raise ValueError("Call model.eval() before explaining predictions.")
    if full_input.ndim != 2 or full_input.shape[0] != 1 or not torch.isfinite(full_input).all():
        raise ValueError("Expected one finite, normalized full-night tensor [1, samples].")
    if not 0 < observed_samples <= full_input.shape[-1]:
        raise ValueError("Observed sample count must lie inside the full-night input.")
    if n_steps < 2 or max_steps < n_steps:
        raise ValueError("Use 2 <= n_steps <= max_steps.")
    if not np.isfinite([atol, rtol]).all() or atol < 0 or rtol < 0:
        raise ValueError("Convergence tolerances must be finite and nonnegative.")
    if not np.isfinite(occlusion_seconds) or occlusion_seconds <= 0:
        raise ValueError("Occlusion duration must be positive and finite.")
    lo, hi = window_bounds(epoch, observed_samples, context_seconds, duration_s)
    full = full_input.detach()
    original_window = full[:, lo:hi].clone()
    reference = make_baseline(original_window, baseline)
    with torch.no_grad():
        original_logits = model({"PPG": full})[:, epoch, :]
    target = int(original_logits.argmax(dim=-1).item()) if stage is None else STAGES.index(stage)
    other = None if contrast is None else STAGES.index(contrast)
    if target == other:
        raise ValueError("Target and contrast must be different stages.")

    def forward(window):
        n = window.shape[0]
        signal = torch.cat((full[:, :lo].expand(n, -1), window, full[:, hi:].expand(n, -1)), dim=-1)
        scores = model({"PPG": signal})[:, epoch, :]
        return scores[:, target] if other is None else scores[:, target] - scores[:, other]

    started = time.perf_counter()
    parameter_states = [parameter.requires_grad for parameter in model.parameters()]
    try:
        model.requires_grad_(False)
        with checkpointed_encoder(model, checkpoint):
            with torch.no_grad():
                score = float(forward(original_window).item())
                reference_score = float(forward(reference).item())
            expected_score = float(original_logits[0, target].item())
            if other is not None:
                expected_score -= float(original_logits[0, other].item())
            if not np.isclose(score, expected_score, atol=1e-5, rtol=1e-5):
                raise RuntimeError("Window wrapper did not preserve the full-night prediction.")
            steps = int(n_steps)
            while True:
                if progress:
                    progress(f"Epoch {epoch}, {STAGES[target]}, reference={baseline}: IG {steps} steps", flush=True)
                with torch.enable_grad():
                    attribution, delta = IntegratedGradients(forward).attribute(
                        original_window, baselines=reference, n_steps=steps,
                        internal_batch_size=1, return_convergence_delta=True,
                    )
                delta_value = float(delta.item())
                converged = abs(delta_value) <= atol + rtol * abs(score - reference_score)
                if converged or steps >= max_steps:
                    break
                steps = min(steps * 2, max_steps)
            if progress:
                progress(f"Epoch {epoch}, reference={baseline}: occlusion", flush=True)
            block = min(hi - lo, max(1, int(round(occlusion_seconds * MODEL_FS))))
            with torch.no_grad():
                occlusion = Occlusion(forward).attribute(
                    original_window, baselines=reference, sliding_window_shapes=(block,),
                    strides=(block,), perturbations_per_eval=1,
                )
    finally:
        for parameter, state in zip(model.parameters(), parameter_states):
            parameter.requires_grad_(state)
    ig = attribution.detach().cpu().numpy()[0]
    occ = occlusion.detach().cpu().numpy()[0]
    if not np.isfinite(ig).all() or not np.isfinite(occ).all() or not np.isfinite(delta_value):
        raise RuntimeError("Non-finite attribution/diagnostic encountered.")
    samples = pd.DataFrame({"sample_index": np.arange(lo, hi),
                            "seconds": (np.arange(lo, hi) + 1) / MODEL_FS,
                            "normalized_ppg": original_window.cpu().numpy()[0],
                            "reference_ppg": reference.cpu().numpy()[0],
                            "integrated_gradients": ig, "occlusion_logit_change": occ})
    diagnostics = {"epoch": int(epoch), "stage": STAGES[target], "contrast": contrast,
                   "baseline": baseline, "score": score, "reference_score": reference_score,
                   "score_difference": score - reference_score, "ig_sum": float(ig.sum()),
                   "convergence_delta": delta_value, "converged": bool(converged),
                   "n_steps": steps, "convergence_atol": atol, "convergence_rtol": rtol,
                   "window_start_s": lo / MODEL_FS, "window_end_s": hi / MODEL_FS,
                   "occlusion_block_samples": block, "elapsed_s": time.perf_counter() - started}
    return Explanation(samples, diagnostics)


def aggregate_attributions(samples, seconds=1):
    """IG sums preserve signed completeness; occlusion uses a mean, never a sum."""
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Aggregation duration must be positive and finite.")
    groups = np.floor((samples["seconds"].to_numpy() - 1 / MODEL_FS) / seconds).astype(int)
    table = samples.assign(bin=groups).groupby("bin", sort=True).agg(
        seconds=("seconds", "min"), signed_ig=("integrated_gradients", "sum"),
        absolute_ig=("integrated_gradients", lambda x: np.abs(x).sum()),
        occlusion_logit_change=("occlusion_logit_change", "mean"),
    )
    return table.reset_index(drop=True)


def plot_overview(raw_ppg, filtered_ppg, acc, hypnodensity, input_name="filtered", stride=10):
    """Display-only stride keeps long-night plots small; inference uses every sample."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(17, 9))
    axes[0].plot(acc.index[::stride], acc.iloc[::stride].to_numpy(), linewidth=0.5)
    axes[0].set(ylabel="ACC (mg)", title=f"Raw acceleration (display stride: {stride} samples)")
    axes[0].legend(["x", "y", "z"], loc="upper right")
    axes[1].plot(filtered_ppg.index[::stride], filtered_ppg.to_numpy()[::stride], color="tab:red", linewidth=0.6)
    axes[1].set(ylabel="PPG (a.u.)", title="Filtered PPG: 0.5-8 Hz, existing sign reversal")
    density_panel(axes[2], hypnodensity)
    axes[2].set(title=f"wav2sleep hypnodensity ({input_name} PPG_0 input)", xlabel="Recording time",
                xlim=(raw_ppg.index[0], hypnodensity["observed_end"].iloc[-1]))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    return fig


def density_panel(ax, hypnodensity):
    edges = hypnodensity.index.append(pd.DatetimeIndex([hypnodensity["observed_end"].iloc[-1]]))
    values = hypnodensity[STAGES].to_numpy()
    ax.stackplot(edges, np.vstack([values, values[-1]]).T, labels=STAGES,
                 colors=["#8c8c8c", "#56b4e9", "#0072b2", "#d55e00"], step="post", alpha=0.9)
    ax.set(ylabel="Stage probability", ylim=(0, 1))
    ax.legend(loc="upper right", ncol=4)


def plot_explanations(explanations, start, acc, hypnodensity):
    """Show signed IG sums and occlusion logit changes with synchronized sensor context."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    first = next(iter(explanations.values()))
    epoch = first.diagnostics["epoch"]
    left = start + pd.Timedelta(seconds=first.diagnostics["window_start_s"])
    right = start + pd.Timedelta(seconds=first.diagnostics["window_end_s"])
    epoch_start = start + pd.Timedelta(seconds=epoch * 30)
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(17, 12))
    acc_window = acc.loc[left:right]
    axes[0].plot(acc_window.index, acc_window.to_numpy(), linewidth=0.7)
    axes[0].set(ylabel="ACC (mg)", title="Raw ACC context (not a model input)")
    axes[0].legend(["x", "y", "z"], loc="upper right")
    times = start + pd.to_timedelta(first.samples["seconds"], unit="s")
    axes[1].plot(times, first.samples["normalized_ppg"], color="black", linewidth=0.7)
    axes[1].set(ylabel="Normalized PPG", title="Actual resampled PPG input to the model")
    for name, result in explanations.items():
        summary = aggregate_attributions(result.samples)
        clock = start + pd.to_timedelta(summary["seconds"], unit="s")
        delta = result.diagnostics["convergence_delta"]
        status = "OK" if result.diagnostics["converged"] else "NOT CONVERGED"
        axes[2].plot(clock, summary["signed_ig"], label=f"{name}: delta={delta:.3g} ({status})")
        axes[3].step(times, result.samples["occlusion_logit_change"], where="post", label=name)
    axes[2].set(ylabel="IG / second", title="Signed contributions to the selected score (sum per second)")
    axes[3].set(ylabel="Logit change", title="Occlusion: original score minus score with one block replaced")
    for ax in axes[2:4]:
        ax.axhline(0, color="black", linewidth=0.6)
        ax.legend(loc="upper right")
    density_panel(axes[4], hypnodensity)
    axes[4].set(xlabel="Recording time", xlim=(left, right))
    axes[4].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    for ax in axes:
        ax.axvspan(epoch_start, epoch_start + pd.Timedelta(seconds=30), color="gold", alpha=0.15)
    score = first.diagnostics["stage"]
    if first.diagnostics["contrast"] is not None:
        score += " minus " + first.diagnostics["contrast"]
    fig.suptitle(f"Epoch {epoch}: {score} logit | shaded = explained 30-second output epoch", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_pulse_detail(explanations, start, detail_seconds=10):
    """Color the first seconds of the output epoch by per-sample IG, with one panel per reference."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    epoch = next(iter(explanations.values())).diagnostics["epoch"]
    lower = epoch * 30
    upper = lower + min(30, detail_seconds)
    fig, axes = plt.subplots(len(explanations), 1, sharex=True, sharey=True,
                             figsize=(17, 3 * len(explanations)), squeeze=False)
    for ax, (name, result) in zip(axes[:, 0], explanations.items()):
        samples = result.samples.loc[(result.samples["seconds"] > lower) & (result.samples["seconds"] <= upper)]
        clock = start + pd.to_timedelta(samples["seconds"], unit="s")
        x = mdates.date2num(clock)
        y = samples["normalized_ppg"].to_numpy()
        ig = samples["integrated_gradients"].to_numpy()
        magnitude = max(float(np.abs(ig).max()), 1e-12)
        points = np.column_stack([x, y]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # A gray zero keeps near-zero contributions visible against a white background.
        cmap = LinearSegmentedColormap.from_list("signed_ppg_ig", ["#2166ac", "#737373", "#b2182b"])
        collection = LineCollection(segments, cmap=cmap, norm=TwoSlopeNorm(vmin=-magnitude, vcenter=0, vmax=magnitude))
        collection.set_array((ig[:-1] + ig[1:]) / 2)
        collection.set_linewidth(2)
        ax.add_collection(collection)
        ax.set(xlim=(x[0], x[-1]), ylim=(y.min() - 0.2, y.max() + 0.2), ylabel="Normalized PPG",
               title=f"{name} reference | positive supports the target relative to this reference")
        fig.colorbar(collection, ax=ax, label="IG per sample")
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1, 0].set_xlabel("Recording time")
    fig.tight_layout()
    return fig
