#!/usr/bin/env python
"""Compare our Walch features with Walch's reference functions on every subject.

Output: results/tables/validate_vs_walch.csv
"""
import ast
import hashlib
import sys
import textwrap
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import preprocess, raw_io  # noqa: E402
from src.utils import load_config  # noqa: E402

COMMIT = "c357a9a26bdb26c3ca0c25fe3df374fdefe92564"
BASE_URL = f"https://raw.githubusercontent.com/ojwalch/sleep_classifiers/{COMMIT}/source/"
REFERENCE_FILES = {
    "utils.py": "6279457eda9bf26680e0f256dfc1ca83453e4d5535a79fe24b6c92e74c3808e9",
    "preprocessing/activity_count/activity_count_service.py":
        "7a2da4bedb370fbb2e06d6479440a7d60c3db01b5c864913de80258f1a322c80",
}
HALF_WINDOW_S = 285
EPOCH_S = 30
FS_HZ = 50
COUNT_SECONDS = 15


def fetch_reference(folder):
    """Download the pinned reference files once and verify their hashes."""
    folder.mkdir(parents=True, exist_ok=True)
    local = {}
    for relative, expected in REFERENCE_FILES.items():
        path = folder / Path(relative).name
        if not path.exists():
            with urllib.request.urlopen(BASE_URL + relative, timeout=30) as response:
                path.write_bytes(response.read())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{path.name} does not match the pinned hash")
        local[relative] = path
    return local


def extract_functions(path, names):
    """Run only the named function definitions from a reference file."""
    source = path.read_text(encoding="utf-8")
    namespace = {"np": np}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(textwrap.dedent(ast.get_source_segment(source, node)), namespace)
    return {name: namespace[name] for name in names}


def reference_counts(motion, max2epochs):
    """The steps of the reference build_activity_counts_without_matlab."""
    grid = np.arange(motion[:, 0].min(), motion[:, 0].max(), 1.0 / FS_HZ)
    z = np.interp(grid, motion[:, 0], motion[:, 3])
    numerator, denominator = butter(5, [3 / (FS_HZ / 2), 11 / (FS_HZ / 2)], "bandpass")
    rectified = np.abs(filtfilt(numerator, denominator, z))
    counts = max2epochs(np.digitize(rectified, np.linspace(0, 5, 129)), FS_HZ, COUNT_SECONDS)
    return np.maximum((counts - 18) * 3.07, 0)


def window(grid, epoch_start):
    return np.where((grid > epoch_start - HALF_WINDOW_S)
                    & (grid < epoch_start + EPOCH_S + HALF_WINDOW_S))[0]


def windowed_count(times, counts, epoch_times,
                   smooth_gauss):
    grid = np.arange(times.min(), times.max(), 1)
    values = np.interp(grid, times, counts)
    picks = [window(grid, t) for t in epoch_times]
    return np.array([smooth_gauss(values[p], p.size) for p in picks])


def windowed_hr(heart_rate, epoch_times, dog):
    grid = np.arange(heart_rate[:, 0].min(), heart_rate[:, 0].max(), 1)
    filtered = dog(np.interp(grid, heart_rate[:, 0], heart_rate[:, 1]), HALF_WINDOW_S)
    filtered = filtered / np.percentile(np.abs(filtered), 90)
    return np.array([filtered[window(grid, t)].std() for t in epoch_times])


def compare_subject(subj, raw_dir, config, utils,
                    max2epochs):
    record = preprocess.process_subject(raw_dir, subj, config)
    recording = preprocess.load_cropped(raw_dir, subj, EPOCH_S)
    names = list(record.feature_names)
    ours_count = record.features[:, names.index("count_walch")]
    ours_hr = record.features[:, names.index("hr_walch")]

    counts = reference_counts(recording.motion, max2epochs)
    first, last = recording.motion[:, 0].min(), recording.motion[:, 0].max()
    spread = np.linspace(first, last, counts.size)
    centres = first + (np.arange(counts.size) + 0.5) * COUNT_SECONDS
    smooth = utils["smooth_gauss"]
    reference_spread = windowed_count(spread, counts, record.epoch_times, smooth)
    reference_centres = windowed_count(centres, counts, record.epoch_times, smooth)
    reference_hr = windowed_hr(recording.heart_rate, record.epoch_times,
                               utils["convolve_with_dog"])
    return {
        "subject": subj,
        "n_epochs": record.labels.size,
        "r_hr": float(np.corrcoef(reference_hr, ours_hr)[0, 1]),
        "r_count_reference_stamps": float(np.corrcoef(reference_spread, ours_count)[0, 1]),
        "r_count_centre_stamps": float(np.corrcoef(reference_centres, ours_count)[0, 1]),
    }


def main():
    config = load_config(ROOT / "configs" / "preprocess.yaml")
    raw_dir = ROOT / config["paths"]["raw_dir"]
    files = fetch_reference(ROOT / "data" / "external" / f"walch_{COMMIT[:12]}")
    utils = extract_functions(files["utils.py"], {"smooth_gauss", "convolve_with_dog"})
    counts_file = files["preprocessing/activity_count/activity_count_service.py"]
    max2epochs = extract_functions(counts_file, {"max2epochs"})["max2epochs"]

    rows = [compare_subject(subj, raw_dir, config, utils, max2epochs)
            for subj in tqdm(raw_io.list_subjects(raw_dir), desc="subjects")]
    table = pd.DataFrame(rows)
    out_path = ROOT / config["paths"]["tables_dir"] / "validate_vs_walch.csv"
    table.round(5).to_csv(out_path, index=False)
    correlations = table.drop(columns=["subject", "n_epochs"])
    print(correlations.describe().loc[["min", "50%", "max"]].round(4))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
