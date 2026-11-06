"""Label mapping, cropping and epoch selection on the 30 s PSG grid."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

WAKE, NREM, REM = 0, 1, 2
UNSCORED = -1
CLASS_NAMES = ("wake", "nrem", "rem")

# Raw PSG codes. Stage 4 is the older R&K deep-sleep stage; Walch counts it as NREM.
STAGE_TO_CLASS = {-1: UNSCORED, 0: WAKE, 1: NREM, 2: NREM, 3: NREM, 4: NREM, 5: REM}


@dataclass(frozen=True)
class EpochSelection:
    """Epochs kept for modelling, and why the others were dropped."""

    times: np.ndarray
    labels: np.ndarray
    n_labelled: int
    n_unscored: int
    n_no_signal: int


def map_stages(stages: np.ndarray) -> np.ndarray:
    """Map raw PSG stages to wake/NREM/REM, leaving unscored epochs as UNSCORED."""
    unknown = sorted(set(np.unique(stages).tolist()) - set(STAGE_TO_CLASS))
    if unknown:
        raise ValueError(f"Unknown PSG stage values: {unknown}")
    return np.array([STAGE_TO_CLASS[int(stage)] for stage in stages], dtype=np.int64)


def crop_interval(label_times: np.ndarray, motion_times: np.ndarray, hr_times: np.ndarray,
                  epoch_s: float) -> Tuple[float, float]:
    """Return the span covered by the labels and both watch signals."""
    start = max(label_times[0], motion_times[0], hr_times[0])
    end = min(label_times[-1] + epoch_s, motion_times[-1], hr_times[-1])
    if start >= end:
        raise ValueError(f"Recordings do not overlap: start={start}, end={end}")
    return float(start), float(end)


def samples_per_epoch(sample_times: np.ndarray, epoch_starts: np.ndarray,
                      epoch_s: float) -> np.ndarray:
    """Count sorted samples falling in each half-open epoch [start, start + epoch_s)."""
    first = np.searchsorted(sample_times, epoch_starts, side="left")
    last = np.searchsorted(sample_times, epoch_starts + epoch_s, side="left")
    return last - first


def select_epochs(label_times: np.ndarray, stages: np.ndarray, motion_times: np.ndarray,
                  hr_times: np.ndarray, epoch_s: float) -> EpochSelection:
    """Keep scored epochs holding at least one motion and one heart rate sample."""
    classes = map_stages(stages)
    scored = classes != UNSCORED
    has_signal = ((samples_per_epoch(motion_times, label_times, epoch_s) > 0)
                  & (samples_per_epoch(hr_times, label_times, epoch_s) > 0))
    keep = scored & has_signal
    return EpochSelection(
        times=label_times[keep],
        labels=classes[keep],
        n_labelled=int(label_times.size),
        n_unscored=int((~scored).sum()),
        n_no_signal=int((scored & ~has_signal).sum()),
    )


def count_sequence_breaks(epoch_times: np.ndarray, epoch_s: float) -> int:
    """Count places where consecutive kept epochs are not adjacent in time."""
    return int((np.diff(epoch_times) > epoch_s).sum())
