#!/usr/bin/env python
"""Per-subject covariates, stage transitions and model disagreement, from saved predictions.

Outputs: results/tables/subject_analysis_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics  # noqa: E402
from src.analysis import (disagreement_by_class, gap_fraction, load_predictions,  # noqa: E402
                          transition_mask)
from src.preprocess import load_cropped  # noqa: E402
from src.utils import load_config  # noqa: E402

RUNS = ("forest_all", "window_forest_all", "hmm_forest_all", "gru_all")
WINDOW_RUN, GRU_RUN = "window_forest_all", "gru_all"
MOTION_GAP_S = 1.0
HR_GAP_S = 30.0
COVARIATES = ("pct_wake", "pct_rem", "hours", "motion_hz", "motion_gap_frac", "hr_gap_frac",
              "n_sequence_breaks")


def signal_gaps(subjects, raw_dir, epoch_s, half_width_s, epoch_times):
    rows = []
    for subj in subjects:
        recording = load_cropped(raw_dir, subj, epoch_s)
        times = epoch_times[subj]
        rows.append({
            "subject": subj,
            "motion_gap_frac": gap_fraction(times, recording.motion[:, 0], epoch_s,
                                            half_width_s, MOTION_GAP_S),
            "hr_gap_frac": gap_fraction(times, recording.heart_rate[:, 0], epoch_s,
                                        half_width_s, HR_GAP_S),
        })
    return pd.DataFrame(rows)


def subject_table(predictions, tables_dir, gaps):
    balance = pd.read_csv(tables_dir / "class_balance.csv", dtype={"subject": str})
    balance = balance[balance["subject"] != "ALL"]
    table = balance[["subject", "n_epochs", "pct_wake", "pct_rem", "hours", "motion_hz",
                     "n_sequence_breaks"]].merge(gaps, on="subject")
    for run, frames in predictions.items():
        kappa = {subj: metrics.score(frame["y_true"].to_numpy(),
                                     frame["y_pred"].to_numpy())["kappa"]
                 for subj, frame in frames.items()}
        table[f"kappa_{run}"] = table["subject"].map(kappa)
    return table


def correlations(table):
    rows = []
    for run in RUNS:
        for covariate in COVARIATES:
            rho, p = spearmanr(table[covariate], table[f"kappa_{run}"])
            rows.append({"run": run, "covariate": covariate, "n_subjects": len(table),
                         "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


def transitions(predictions, epoch_s):
    rows = []
    for run, frames in predictions.items():
        parts = {"transition": ([], []), "stable": ([], [])}
        for frame in frames.values():
            y_true, y_pred = frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()
            near = transition_mask(y_true, frame["epoch_time"].to_numpy(), epoch_s)
            for group, mask in (("transition", near), ("stable", ~near)):
                parts[group][0].append(y_true[mask])
                parts[group][1].append(y_pred[mask])
        for group, (true_parts, pred_parts) in parts.items():
            y_true, y_pred = np.concatenate(true_parts), np.concatenate(pred_parts)
            rows.append({"run": run, "epochs": group, "n_epochs": int(y_true.size),
                         **metrics.score(y_true, y_pred)})
    return pd.DataFrame(rows)


def disagreement(predictions):
    window, gru = predictions[WINDOW_RUN], predictions[GRU_RUN]
    by_class = [disagreement_by_class(window[s], gru[s], WINDOW_RUN, GRU_RUN) for s in window]
    per_class = pd.concat(by_class).groupby("true_class", as_index=False).sum()
    per_class["pct_disagree"] = 100.0 * per_class["n_disagree"] / per_class["n_epochs"]
    rows = []
    for subj, frame in window.items():
        y_true = frame["y_true"].to_numpy()
        pred_w, pred_g = frame["y_pred"].to_numpy(), gru[subj]["y_pred"].to_numpy()
        rows.append({"subject": subj, "n_epochs": int(y_true.size),
                     "pct_disagree": 100.0 * (pred_w != pred_g).mean(),
                     f"kappa_{WINDOW_RUN}": metrics.score(y_true, pred_w)["kappa"],
                     f"kappa_{GRU_RUN}": metrics.score(y_true, pred_g)["kappa"]})
    per_subject = pd.DataFrame(rows)
    per_subject["kappa_gru_minus_window"] = (per_subject[f"kappa_{GRU_RUN}"]
                                             - per_subject[f"kappa_{WINDOW_RUN}"])
    return per_class, per_subject.sort_values("pct_disagree", ascending=False)


def main():
    config = load_config(ROOT / "configs" / "baselines.yaml", ("paths", "epoch_s"))
    preprocess = load_config(ROOT / "configs" / "preprocess.yaml", ("paths", "features"))
    tables_dir = ROOT / config["paths"]["tables_dir"]
    epoch_s = config["epoch_s"]
    predictions = {run: load_predictions(run, ROOT / config["paths"]["predictions_dir"])
                   for run in RUNS}
    subjects = sorted(predictions[RUNS[0]])
    if any(sorted(frames) != subjects for frames in predictions.values()):
        raise ValueError("Runs do not cover the same subjects")

    epoch_times = {subj: predictions[RUNS[0]][subj]["epoch_time"].to_numpy() for subj in subjects}
    gaps = signal_gaps(subjects, ROOT / preprocess["paths"]["raw_dir"], epoch_s,
                       preprocess["features"]["window"]["half_width_s"], epoch_times)
    table = subject_table(predictions, tables_dir, gaps)
    outputs = {
        "subject_analysis_subjects.csv": table,
        "subject_analysis_correlations.csv": correlations(table),
        "subject_analysis_transitions.csv": transitions(predictions, epoch_s),
    }
    per_class, per_subject = disagreement(predictions)
    outputs["subject_analysis_disagreement_class.csv"] = per_class
    outputs["subject_analysis_disagreement_subject.csv"] = per_subject
    for name, frame in outputs.items():
        frame.to_csv(tables_dir / name, index=False)
        print(f"{name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
