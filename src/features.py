"""Per-epoch features from motion, heart rate and time.
Walch features follow github.com/ojwalch/sleep_classifiers; per-epoch features add no context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0

# Column order of the feature matrix, as (name, group). Groups drive the ablations.
FEATURES = (
    ("count_walch", "motion"),
    ("count_epoch", "motion"),
    ("hr_walch", "heart_rate"),
    ("hr_mean_z", "heart_rate"),
    ("hr_std_epoch", "heart_rate"),
    ("cosine", "time"),
    ("time_hours", "time"),
)


@dataclass(frozen=True)
class CountParams:
    fs_hz: int
    band_hz: Tuple[float, float]
    filter_order: int
    n_bins: int
    bin_range: Tuple[float, float]
    count_seconds: int
    offset: float
    scale: float


@dataclass(frozen=True)
class HeartRateParams:
    dog_sigmas_s: Tuple[float, float]
    dog_scalar: float
    dog_points: int
    norm_percentile: float

    def __post_init__(self) -> None:
        if self.dog_points % 2 == 0:
            raise ValueError("dog_points must be odd so the filter keeps the signal length")


@dataclass(frozen=True)
class WindowParams:
    half_width_s: int
    count_sigma_s: float


@dataclass(frozen=True)
class ClockParams:
    shift_hours: float


@dataclass(frozen=True)
class FeatureParams:
    epoch_s: int
    counts: CountParams
    heart: HeartRateParams
    window: WindowParams
    clock: ClockParams

    @classmethod
    def from_config(cls, section: Dict[str, Any]) -> "FeatureParams":
        return cls(
            epoch_s=section["epoch_s"],
            counts=CountParams(**section["counts"]),
            heart=HeartRateParams(**section["heart"]),
            window=WindowParams(**section["window"]),
            clock=ClockParams(**section["clock"]),
        )


def activity_counts(times: np.ndarray, z: np.ndarray, start: float, end: float,
                    params: CountParams) -> Tuple[np.ndarray, np.ndarray]:
    """Return bin centre times and activity counts from z acceleration; bins start at `start`."""
    fs = params.fs_hz
    grid = start + np.arange(int(round((end - start) * fs))) / fs
    resampled = np.interp(grid, times, z)

    nyquist = fs / 2
    numerator, denominator = butter(
        params.filter_order, [cutoff / nyquist for cutoff in params.band_hz], "bandpass")
    rectified = np.abs(filtfilt(numerator, denominator, resampled))

    edges = np.linspace(params.bin_range[0], params.bin_range[1], params.n_bins + 1)
    binned = np.digitize(rectified, edges)

    n_seconds = binned.size // fs
    peak_per_second = binned[: n_seconds * fs].reshape(n_seconds, fs).max(axis=1)
    n_counts = n_seconds // params.count_seconds
    summed = (peak_per_second[: n_counts * params.count_seconds]
              .reshape(n_counts, params.count_seconds).sum(axis=1))

    counts = np.maximum((summed - params.offset) * params.scale, 0.0)
    centres = start + (np.arange(n_counts) + 0.5) * params.count_seconds
    return centres, counts


def resample_1hz(times: np.ndarray, values: np.ndarray, start: float,
                 end: float) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate onto a 1 Hz grid over [start, end)."""
    grid = np.arange(start, end, 1.0)
    return grid, np.interp(grid, times, values)


def window_slice(grid: np.ndarray, epoch_start: float, epoch_s: float,
                 half_width_s: float) -> slice:
    """Grid indices strictly inside the epoch widened by half_width_s on each side."""
    first = np.searchsorted(grid, epoch_start - half_width_s, side="right")
    last = np.searchsorted(grid, epoch_start + epoch_s + half_width_s, side="left")
    return slice(int(first), int(last))


def epoch_slice(grid: np.ndarray, epoch_start: float, epoch_s: float) -> slice:
    """Grid indices inside the half-open epoch."""
    first = np.searchsorted(grid, epoch_start, side="left")
    last = np.searchsorted(grid, epoch_start + epoch_s, side="left")
    return slice(int(first), int(last))


def gaussian_mean(values: np.ndarray, sigma: float) -> float:
    """Gaussian-weighted mean, peaked at the middle of the window."""
    offsets = np.arange(values.size) - values.size // 2
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    return float(weights @ values / weights.sum())


def dog_filter(signal: np.ndarray, params: HeartRateParams) -> np.ndarray:
    """Band-pass a 1 Hz signal with a difference of Gaussians, mirroring the ends."""
    centred = signal - signal.mean()
    offsets = np.arange(params.dog_points) - params.dog_points // 2
    narrow, wide = params.dog_sigmas_s
    kernel = (np.exp(-0.5 * (offsets / narrow) ** 2)
              - params.dog_scalar * np.exp(-0.5 * (offsets / wide) ** 2))
    pad = params.dog_points // 2
    padded = np.concatenate((centred[:pad][::-1], centred, centred[-pad:][::-1]))
    return np.convolve(padded, kernel, mode="valid")


def normalise_filtered_hr(heart_rate: np.ndarray, params: HeartRateParams) -> np.ndarray:
    """DoG-filter heart rate and scale by the subject's own percentile of |signal|."""
    filtered = dog_filter(heart_rate, params)
    scale = np.percentile(np.abs(filtered), params.norm_percentile)
    if scale == 0:
        raise ValueError("Heart rate is constant, so it cannot be normalised")
    return filtered / scale


def cosine_proxy(seconds: np.ndarray, params: ClockParams) -> np.ndarray:
    """Walch's cosine clock proxy, given seconds since the first valid epoch."""
    shifted = seconds - params.shift_hours * SECONDS_PER_HOUR
    return -np.cos(shifted * 2 * np.pi / SECONDS_PER_DAY)


def motion_features(epoch_times: np.ndarray, motion_times: np.ndarray, z: np.ndarray,
                    start: float, end: float, params: FeatureParams) -> Dict[str, np.ndarray]:
    """Return count_walch (windowed) and count_epoch (this epoch only)."""
    centres, counts = activity_counts(motion_times, z, start, end, params.counts)
    grid, counts_1hz = resample_1hz(centres, counts, start, end)
    window = params.window
    return {
        "count_walch": np.array([
            gaussian_mean(counts_1hz[window_slice(grid, t, params.epoch_s, window.half_width_s)],
                          window.count_sigma_s)
            for t in epoch_times]),
        "count_epoch": np.array([
            counts[epoch_slice(centres, t, params.epoch_s)].sum() for t in epoch_times]),
    }


def heart_rate_features(epoch_times: np.ndarray, hr_times: np.ndarray, heart_rate: np.ndarray,
                        start: float, end: float, params: FeatureParams) -> Dict[str, np.ndarray]:
    """Return hr_walch (windowed) plus per-epoch mean and spread in subject z-units."""
    grid, hr_1hz = resample_1hz(hr_times, heart_rate, start, end)
    spread = hr_1hz.std()
    if spread == 0:
        raise ValueError("Heart rate is constant, so it cannot be normalised")
    z_scored = (hr_1hz - hr_1hz.mean()) / spread
    filtered = normalise_filtered_hr(hr_1hz, params.heart)

    windows = [window_slice(grid, t, params.epoch_s, params.window.half_width_s)
               for t in epoch_times]
    own_epoch = [epoch_slice(grid, t, params.epoch_s) for t in epoch_times]
    return {
        "hr_walch": np.array([filtered[w].std() for w in windows]),
        "hr_mean_z": np.array([z_scored[e].mean() for e in own_epoch]),
        "hr_std_epoch": np.array([z_scored[e].std() for e in own_epoch]),
    }


def time_features(epoch_times: np.ndarray, params: FeatureParams) -> Dict[str, np.ndarray]:
    """Return clock-proxy features; the dataset has no wall-clock time."""
    elapsed = epoch_times - epoch_times[0]
    return {
        "cosine": cosine_proxy(elapsed, params.clock),
        "time_hours": elapsed / SECONDS_PER_HOUR,
    }


def build_features(epoch_times: np.ndarray, motion_times: np.ndarray, z: np.ndarray,
                   hr_times: np.ndarray, heart_rate: np.ndarray, start: float, end: float,
                   params: FeatureParams) -> Tuple[np.ndarray, List[str], List[str]]:
    """Return the (n_epochs, n_features) matrix with feature names and groups."""
    columns = {
        **motion_features(epoch_times, motion_times, z, start, end, params),
        **heart_rate_features(epoch_times, hr_times, heart_rate, start, end, params),
        **time_features(epoch_times, params),
    }
    names = [name for name, _ in FEATURES]
    groups = [group for _, group in FEATURES]
    return np.column_stack([columns[name] for name in names]), names, groups
