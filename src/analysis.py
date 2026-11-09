"""Post-hoc analyses on saved predictions: transitions, signal gaps, disagreements."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.dataset import bounds_from_times


def load_predictions(run: str, predictions_dir: Path) -> Dict[str, pd.DataFrame]:
    """Return {subject: predictions} for one run, skipping longer run names."""
    frames = {}
    for path in sorted(Path(predictions_dir).glob(f"{run}_*.csv")):
        subject = path.stem[len(run) + 1:]
        if subject.isdigit():
            frames[subject] = pd.read_csv(path, dtype={"subject": str})
    return frames


def transition_mask(labels: np.ndarray, epoch_times: np.ndarray, epoch_s: int,
                    width: int = 2) -> np.ndarray:
    """True for the `width` epochs on each side of a stage change within a segment."""
    mask = np.zeros(labels.size, dtype=bool)
    for start, stop in bounds_from_times(epoch_times, epoch_s):
        changes = start + 1 + np.flatnonzero(np.diff(labels[start:stop]) != 0)
        for change in changes:
            mask[max(change - width, start):min(change + width, stop)] = True
    return mask


def gap_fraction(epoch_times: np.ndarray, sample_times: np.ndarray, epoch_s: int,
                 half_width_s: float, max_gap_s: float) -> float:
    """Share of epochs whose window overlaps a sampling gap longer than max_gap_s."""
    sample_times = np.sort(sample_times)
    long = np.flatnonzero(np.diff(sample_times) > max_gap_s)
    if long.size == 0:
        return 0.0
    gap_start, gap_end = sample_times[long], sample_times[long + 1]
    window_start = epoch_times - half_width_s
    window_end = epoch_times + epoch_s + half_width_s
    first_after = np.searchsorted(gap_end, window_start, side="right")
    candidate = np.minimum(first_after, gap_start.size - 1)
    overlaps = (first_after < gap_start.size) & (gap_start[candidate] < window_end)
    return float(overlaps.mean())


def disagreement_by_class(a: pd.DataFrame, b: pd.DataFrame, name_a: str,
                          name_b: str) -> pd.DataFrame:
    """Per true class: how often two models disagree, and which one is right then."""
    if not np.array_equal(a["epoch_time"].to_numpy(), b["epoch_time"].to_numpy()):
        raise ValueError("Predictions are not aligned on epoch_time")
    y_true = a["y_true"].to_numpy()
    pred_a, pred_b = a["y_pred"].to_numpy(), b["y_pred"].to_numpy()
    differ = pred_a != pred_b
    rows = []
    for code in np.unique(y_true):
        in_class = y_true == code
        both = in_class & differ
        rows.append({
            "true_class": int(code),
            "n_epochs": int(in_class.sum()),
            "n_disagree": int(both.sum()),
            "pct_disagree": 100.0 * both.sum() / max(in_class.sum(), 1),
            f"{name_a}_right": int((both & (pred_a == y_true)).sum()),
            f"{name_b}_right": int((both & (pred_b == y_true)).sum()),
            "both_wrong": int((both & (pred_a != y_true) & (pred_b != y_true)).sum()),
        })
    return pd.DataFrame(rows)
