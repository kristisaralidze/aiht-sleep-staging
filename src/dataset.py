"""Load processed subjects and work with their epoch sequences."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SubjectData:
    """One night: epoch features and labels, restricted to a chosen feature set."""

    subject_id: str
    features: np.ndarray
    labels: np.ndarray
    epoch_times: np.ndarray
    feature_names: Tuple[str, ...]
    feature_groups: Tuple[str, ...]
    epoch_s: int


def _load_one(path: Path, feature_names: Sequence[str], epoch_s: int) -> SubjectData:
    with np.load(path, allow_pickle=False) as saved:
        available = [str(name) for name in saved["feature_names"]]
        missing = sorted(set(feature_names) - set(available))
        if missing:
            raise ValueError(f"Unknown features {missing} in {path.name}")
        columns = [available.index(name) for name in feature_names]
        return SubjectData(
            subject_id=str(saved["subject_id"]),
            features=saved["features"][:, columns],
            labels=saved["labels"],
            epoch_times=saved["epoch_times"],
            feature_names=tuple(feature_names),
            feature_groups=tuple(str(saved["feature_groups"][c]) for c in columns),
            epoch_s=epoch_s,
        )


def load_subjects(processed_dir: Path, feature_names: Sequence[str], epoch_s: int,
                  exclude: Sequence[str] = ()) -> List[SubjectData]:
    """Load every subject except `exclude`, keeping only `feature_names`."""
    paths = sorted(Path(processed_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No processed subjects in {processed_dir}")
    unknown = sorted(set(exclude) - {path.stem for path in paths})
    if unknown:
        raise ValueError(f"Cannot exclude unknown subjects: {unknown}")
    return [_load_one(path, feature_names, epoch_s) for path in paths
            if path.stem not in set(exclude)]


def segment_bounds(subj: SubjectData) -> List[Tuple[int, int]]:
    """Return [start, stop) index pairs of runs of time-adjacent epochs."""
    breaks = np.flatnonzero(np.diff(subj.epoch_times) > subj.epoch_s) + 1
    edges = [0, *breaks.tolist(), subj.epoch_times.size]
    return list(zip(edges[:-1], edges[1:]))


def stack(subjs: Sequence[SubjectData]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate subjects into one feature matrix and label vector."""
    return (np.vstack([subj.features for subj in subjs]),
            np.concatenate([subj.labels for subj in subjs]))
