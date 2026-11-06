"""Read the raw sleep-accel text files.

File formats (timestamps are seconds relative to the start of the PSG):
    motion/<id>_acceleration.txt   "time x y z"  (space separated, g)
    heart_rate/<id>_heartrate.txt  "time,bpm"
    labels/<id>_labeled_sleep.txt  "time stage"
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

MODALITIES = {
    "motion": "_acceleration.txt",
    "heart_rate": "_heartrate.txt",
    "labels": "_labeled_sleep.txt",
}


def clean_samples(samples: np.ndarray) -> np.ndarray:
    """Return finite rows sorted by time, keeping the first row per timestamp.

    The watch files contain out-of-order timestamps, and np.interp needs them
    strictly increasing.
    """
    finite = samples[np.isfinite(samples).all(axis=1)]
    if finite.shape[0] == 0:
        return finite
    ordered = finite[np.argsort(finite[:, 0], kind="stable")]
    is_new_time = np.concatenate(([True], np.diff(ordered[:, 0]) > 0))
    return ordered[is_new_time]


def _read(path: Path, separator: str) -> np.ndarray:
    return pd.read_csv(path, sep=separator, header=None, dtype=float).to_numpy()


def _subject_ids(folder: Path, suffix: str) -> set:
    return {path.name[: -len(suffix)] for path in folder.glob(f"*{suffix}")}


def list_subjects(raw_dir: Path) -> List[str]:
    """Return sorted subject ids, requiring every subject to have all modalities."""
    raw_dir = Path(raw_dir)
    per_modality = {name: _subject_ids(raw_dir / name, suffix)
                    for name, suffix in MODALITIES.items()}
    every_id = set().union(*per_modality.values())
    if not every_id:
        raise FileNotFoundError(f"No subject files under {raw_dir}")
    incomplete = sorted(every_id - set.intersection(*per_modality.values()))
    if incomplete:
        raise ValueError(f"Subjects missing a modality: {incomplete}")
    return sorted(every_id)


def load_motion(raw_dir: Path, subj: str) -> np.ndarray:
    """Return cleaned (n, 4) rows of time, x, y, z."""
    path = Path(raw_dir) / "motion" / f"{subj}{MODALITIES['motion']}"
    return clean_samples(_read(path, " "))


def load_heart_rate(raw_dir: Path, subj: str) -> np.ndarray:
    """Return cleaned (n, 2) rows of time, bpm."""
    path = Path(raw_dir) / "heart_rate" / f"{subj}{MODALITIES['heart_rate']}"
    return clean_samples(_read(path, ","))


def load_labels(raw_dir: Path, subj: str) -> np.ndarray:
    """Return (n, 2) rows of epoch start time, PSG stage."""
    path = Path(raw_dir) / "labels" / f"{subj}{MODALITIES['labels']}"
    return _read(path, " ")
