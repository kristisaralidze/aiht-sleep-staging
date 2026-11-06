"""Turn one subject's raw files into an epoch-level record, and summarise records."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src import epochs, raw_io
from src.features import FeatureParams, build_features

SECONDS_PER_HOUR = 3600.0
COUNT_COLUMNS = ("n_labelled", "n_unscored", "n_no_signal", "n_epochs",
                 "n_wake", "n_nrem", "n_rem", "n_sequence_breaks")


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    features: np.ndarray
    labels: np.ndarray
    epoch_times: np.ndarray
    feature_names: Tuple[str, ...]
    feature_groups: Tuple[str, ...]
    epoch_s: int
    n_labelled: int
    n_unscored: int
    n_no_signal: int
    motion_hz: float


@dataclass(frozen=True)
class CroppedRecording:
    labels: np.ndarray
    motion: np.ndarray
    heart_rate: np.ndarray
    start: float
    end: float


def _crop(samples: np.ndarray, start: float, end: float) -> np.ndarray:
    return samples[(samples[:, 0] >= start) & (samples[:, 0] < end)]


def load_cropped(raw_dir: Path, subj: str, epoch_s: int) -> CroppedRecording:
    """Load one subject's raw files, cropped to the span all recordings share."""
    labels = raw_io.load_labels(raw_dir, subj)
    motion = raw_io.load_motion(raw_dir, subj)
    heart_rate = raw_io.load_heart_rate(raw_dir, subj)
    start, end = epochs.crop_interval(labels[:, 0], motion[:, 0], heart_rate[:, 0], epoch_s)
    return CroppedRecording(labels, _crop(motion, start, end), _crop(heart_rate, start, end),
                            start, end)


def process_subject(raw_dir: Path, subj: str, config: Dict[str, Any]) -> SubjectRecord:
    """Crop to the labelled recording, select valid epochs and build their features."""
    params = FeatureParams.from_config(config["features"])
    recording = load_cropped(raw_dir, subj, params.epoch_s)
    labels, motion, heart_rate = recording.labels, recording.motion, recording.heart_rate
    start, end = recording.start, recording.end

    selection = epochs.select_epochs(labels[:, 0], labels[:, 1].astype(int), motion[:, 0],
                                     heart_rate[:, 0], params.epoch_s)
    if selection.times.size == 0:
        raise ValueError(f"Subject {subj} has no valid epochs")

    matrix, names, groups = build_features(selection.times, motion[:, 0], motion[:, 3],
                                           heart_rate[:, 0], heart_rate[:, 1], start, end, params)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Subject {subj} has non-finite features")
    return SubjectRecord(
        subject_id=subj,
        features=matrix,
        labels=selection.labels,
        epoch_times=selection.times,
        feature_names=tuple(names),
        feature_groups=tuple(groups),
        epoch_s=params.epoch_s,
        n_labelled=selection.n_labelled,
        n_unscored=selection.n_unscored,
        n_no_signal=selection.n_no_signal,
        motion_hz=float(1.0 / np.median(np.diff(motion[:, 0]))),
    )


def save_record(record: SubjectRecord, out_dir: Path) -> Path:
    """Write one subject to <out_dir>/<subject>.npz (loadable without pickle)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record.subject_id}.npz"
    np.savez_compressed(
        path,
        subject_id=np.array(record.subject_id),
        features=record.features,
        labels=record.labels,
        epoch_times=record.epoch_times,
        feature_names=np.array(record.feature_names),
        feature_groups=np.array(record.feature_groups),
    )
    return path


def _percentages(counts: Dict[str, int], total: int) -> Dict[str, float]:
    return {f"pct_{name}": round(100.0 * counts[f"n_{name}"] / total, 1)
            for name in epochs.CLASS_NAMES}


def summary_row(record: SubjectRecord) -> Dict[str, Any]:
    """Summarise one subject: epochs kept, dropped, and class balance."""
    n_epochs = int(record.labels.size)
    class_counts = {f"n_{name}": int((record.labels == code).sum())
                    for code, name in enumerate(epochs.CLASS_NAMES)}
    return {
        "subject": record.subject_id,
        "n_labelled": record.n_labelled,
        "n_unscored": record.n_unscored,
        "n_no_signal": record.n_no_signal,
        "n_epochs": n_epochs,
        "hours": round(n_epochs * record.epoch_s / SECONDS_PER_HOUR, 2),
        **class_counts,
        **_percentages(class_counts, n_epochs),
        "n_sequence_breaks": epochs.count_sequence_breaks(record.epoch_times, record.epoch_s),
        "motion_hz": round(record.motion_hz, 1),
    }


def summary_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Stack subject rows and append a pooled ALL row."""
    table = pd.DataFrame(rows)
    totals = {column: int(table[column].sum()) for column in COUNT_COLUMNS}
    pooled = {
        "subject": "ALL",
        **totals,
        "hours": round(float(table["hours"].sum()), 2),
        **_percentages(totals, totals["n_epochs"]),
    }
    return pd.concat([table, pd.DataFrame([pooled])], ignore_index=True)[table.columns]
