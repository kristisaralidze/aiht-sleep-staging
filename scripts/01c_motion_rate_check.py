#!/usr/bin/env python
"""Measure how accelerometer sampling rate affects activity counts.

Comparing subjects confounds rate with behaviour, so the causal test thins each
full-rate night to about 10 Hz and recomputes counts on identical movement.

Outputs:
    results/tables/motion_rate_by_subject.csv   count statistics and rate per subject
    results/tables/motion_rate_thinning.csv     full-rate vs thinned counts
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import preprocess, raw_io  # noqa: E402
from src.features import FeatureParams, activity_counts  # noqa: E402
from src.utils import load_config  # noqa: E402

TARGET_HZ = 10.0
MIN_SOURCE_HZ = 45.0


def thinning_row(subj, recording, hz,
                 params):
    """Recompute counts on every n-th sample of the same night."""
    stride = int(round(hz / TARGET_HZ))
    motion = recording.motion
    args = (recording.start, recording.end, params)
    full = activity_counts(motion[:, 0], motion[:, 3], *args)[1]
    thin = activity_counts(motion[::stride, 0], motion[::stride, 3], *args)[1]
    return {
        "subject": subj,
        "source_hz": round(hz, 1),
        "thinned_hz": round(hz / stride, 1),
        "full_mean": float(full.mean()),
        "thinned_mean": float(thin.mean()),
        "ratio": float(thin.mean() / full.mean()),
        "r": float(np.corrcoef(full, thin)[0, 1]),
        "pct_zero_full": float(100 * (full == 0).mean()),
        "pct_zero_thinned": float(100 * (thin == 0).mean()),
    }


def subject_row(subj, processed_dir, hz):
    with np.load(processed_dir / f"{subj}.npz", allow_pickle=False) as saved:
        names = [str(name) for name in saved["feature_names"]]
        counts = saved["features"][:, names.index("count_epoch")]
        y = saved["labels"]
    return {
        "subject": subj,
        "motion_hz": round(hz, 1),
        "pct_zero": float(100 * (counts == 0).mean()),
        "mean_count": float(counts.mean()),
        "wake_mean": float(counts[y == 0].mean()),
        "sleep_mean": float(counts[y != 0].mean()),
    }


def main():
    config = load_config(ROOT / "configs" / "preprocess.yaml")
    params = FeatureParams.from_config(config["features"])
    raw_dir = ROOT / config["paths"]["raw_dir"]
    processed_dir = ROOT / config["paths"]["processed_dir"]
    tables_dir = ROOT / config["paths"]["tables_dir"]

    by_subject = []
    thinning = []
    for subj in tqdm(raw_io.list_subjects(raw_dir), desc="subjects"):
        recording = preprocess.load_cropped(raw_dir, subj, params.epoch_s)
        hz = float(1.0 / np.median(np.diff(recording.motion[:, 0])))
        by_subject.append(subject_row(subj, processed_dir, hz))
        if hz >= MIN_SOURCE_HZ:
            thinning.append(thinning_row(subj, recording, hz, params.counts))

    pd.DataFrame(by_subject).round(3).to_csv(tables_dir / "motion_rate_by_subject.csv", index=False)
    table = pd.DataFrame(thinning).round(3)
    table.to_csv(tables_dir / "motion_rate_thinning.csv", index=False)
    print(table[["ratio", "r"]].describe().loc[["count", "min", "50%", "max"]].round(3))
    print(f"\nSaved tables to {tables_dir}")


if __name__ == "__main__":
    main()
