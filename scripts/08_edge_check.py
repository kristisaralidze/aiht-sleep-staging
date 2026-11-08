#!/usr/bin/env python
"""Check how much of the window model's gain comes from epochs near segment edges.

Near a night's start or end, or a sequence break, a +/-k window repeats the edge
epoch, which a model could use as a position cue. Both the window model and its
baseline are scored on all epochs and on interior epochs only, where the whole
window is real data. k is the window size chosen in each subject's fold.

Output: results/tables/edge_check.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics  # noqa: E402
from src.context import interior_mask  # noqa: E402
from src.utils import load_config  # noqa: E402


def load_run(run, predictions_dir):
    files = sorted(predictions_dir.glob(f"{run}_*.csv"))
    return {path.stem[len(run) + 1:]: pd.read_csv(path) for path in files
            if path.stem[len(run) + 1:].isdigit()}


def score(frames, masks):
    """Pooled macro F1 and per-subject mean kappa over the masked epochs."""
    kept = {subj: frame[masks[subj]] for subj, frame in frames.items() if masks[subj].any()}
    y_true = np.concatenate([frame["y_true"].to_numpy() for frame in kept.values()])
    y_pred = np.concatenate([frame["y_pred"].to_numpy() for frame in kept.values()])
    kappas = [metrics.score(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())["kappa"]
              for frame in kept.values()]
    return {"n_subjects": len(kept), "n_epochs": int(y_true.size),
            "macro_f1": metrics.macro_f1(y_true, y_pred), "kappa": float(np.nanmean(kappas))}


def main():
    config = load_config(ROOT / "configs" / "baselines.yaml")
    tables_dir = ROOT / config["paths"]["tables_dir"]
    predictions_dir = ROOT / config["paths"]["predictions_dir"]
    selection = pd.read_csv(tables_dir / "baseline_selection.csv")
    best = selection[selection["selected"]].iloc[0]
    window_run = f"window_{best['model']}_{best['feature_set']}"
    folds = json.loads((ROOT / config["paths"]["logs_dir"] / f"folds_{window_run}.json")
                       .read_text(encoding="utf-8"))["folds"]

    window = load_run(window_run, predictions_dir)
    baseline = load_run(best["run"], predictions_dir)
    if sorted(window) != sorted(baseline):
        raise ValueError("Window and baseline predictions cover different subjects")
    everything = {subj: np.ones(len(frame), dtype=bool) for subj, frame in window.items()}
    interior = {subj: interior_mask(frame["epoch_time"].to_numpy(), config["epoch_s"],
                                    folds[subj]["params"]["k"])
                for subj, frame in window.items()}

    rows = []
    for subset, masks in (("all epochs", everything), ("interior epochs", interior)):
        base, win = score(baseline, masks), score(window, masks)
        rows.append({"subset": subset, "n_subjects": base["n_subjects"],
                     "n_epochs": base["n_epochs"],
                     "baseline_macro_f1": base["macro_f1"], "window_macro_f1": win["macro_f1"],
                     "gain_macro_f1": win["macro_f1"] - base["macro_f1"],
                     "baseline_kappa": base["kappa"], "window_kappa": win["kappa"],
                     "gain_kappa": win["kappa"] - base["kappa"]})
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "edge_check.csv", index=False)
    print(f"{window_run} vs {best['run']}")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
