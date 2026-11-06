"""Save what every LOSO run must leave behind: predictions, metrics, figures, timing."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import metrics  # noqa: E402
from src.epochs import CLASS_NAMES  # noqa: E402
from src.loso import LosoResult  # noqa: E402

CLASS_LABELS = ("Wake", "NREM", "REM")
DARK_TEXT = "#1f1f1f"
LIGHT_TEXT = "#ffffff"
LIGHT_TEXT_FROM = 0.55    # share of the panel maximum above which cell text turns white


def save_predictions(result: LosoResult, out_dir: Path) -> None:
    """Write one <run>_<subject>.csv per left-out subject."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fold in result.folds:
        frame = pd.DataFrame({"subject": fold.subject_id, "epoch_time": fold.epoch_times,
                              "y_true": fold.y_true, "y_pred": fold.y_pred})
        for code, name in enumerate(CLASS_NAMES):
            frame[f"p_{name}"] = fold.proba[:, code]
        frame.to_csv(out_dir / f"{result.name}_{fold.subject_id}.csv", index=False)


def evaluate(result: LosoResult, only_subjects: Optional[Sequence[str]] = None
             ) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Return one summary row (pooled + per-subject) and the per-subject table."""
    folds = [fold for fold in result.folds
             if only_subjects is None or fold.subject_id in set(only_subjects)]
    per_subject = pd.DataFrame([
        {"run": result.name, "subject": fold.subject_id, "n_epochs": fold.y_true.size,
         **metrics.score(fold.y_true, fold.y_pred)}
        for fold in folds])
    y_true = np.concatenate([fold.y_true for fold in folds])
    y_pred = np.concatenate([fold.y_pred for fold in folds])
    pooled = {f"{name}_pooled": value for name, value in metrics.score(y_true, y_pred).items()}
    row = {"run": result.name, "n_epochs": int(y_true.size), **pooled,
           **metrics.summarise_subjects(per_subject),
           "mean_inner_macro_f1": float(np.mean([fold.inner_score for fold in folds]))}
    return row, per_subject


def _panel(axis: Any, values: np.ndarray, text: Sequence[Sequence[str]], title: str) -> None:
    axis.pcolormesh(values, cmap="Blues", vmin=0, vmax=values.max(), edgecolors="white",
                    linewidth=2)
    axis.invert_yaxis()
    axis.set_aspect("equal")
    ticks = np.arange(len(CLASS_LABELS)) + 0.5
    axis.set_xticks(ticks, CLASS_LABELS)
    axis.set_yticks(ticks, CLASS_LABELS)
    axis.tick_params(length=0)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title, fontsize=10)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            dark_cell = values[row, column] > LIGHT_TEXT_FROM * values.max()
            axis.text(column + 0.5, row + 0.5, text[row][column], ha="center", va="center",
                      color=LIGHT_TEXT if dark_cell else DARK_TEXT, fontsize=10)


def save_confusion_figure(result: LosoResult, out_dir: Path) -> None:
    """Save the pooled confusion matrix, counts and row-normalised, as PDF and PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.concatenate([fold.y_true for fold in result.folds])
    y_pred = np.concatenate([fold.y_pred for fold in result.folds])
    counts, rates = metrics.confusion(y_true, y_pred)
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), constrained_layout=True)
    _panel(axes[0], counts.astype(float), [[f"{v:,}" for v in row] for row in counts], "Epochs")
    _panel(axes[1], rates, [[f"{v:.0%}" for v in row] for row in rates], "Share of true class")
    figure.suptitle(result.name, fontsize=11)
    for extension in ("pdf", "png"):
        figure.savefig(out_dir / f"confusion_{result.name}.{extension}", dpi=200)
    plt.close(figure)


def log_timing(result: LosoResult, path: Path, script: str) -> None:
    """Upsert this run's wall-clock and summed per-fold time into the timing log."""
    path = Path(path)
    row = {"run": result.name, "script": script, "n_folds": len(result.folds),
           "n_jobs": result.n_jobs, "cpu_count": os.cpu_count(),
           "wall_seconds": round(result.wall_seconds, 2),
           "fold_seconds_total": round(result.fold_seconds_total, 2)}
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=list(row))
    kept = existing[existing["run"] != result.name]
    combined = pd.DataFrame([row]) if kept.empty else pd.concat([kept, pd.DataFrame([row])])
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


def save_fold_log(result: LosoResult, logs_dir: Path) -> None:
    """Record what each fold chose, so later scripts can reuse tuned parameters."""
    log = {
        "run": result.name,
        "mean_inner_score": result.mean_inner_score,
        "folds": {fold.subject_id: {"params": fold.params, "inner_score": fold.inner_score,
                                    "seconds": round(fold.seconds, 2)}
                  for fold in result.folds},
    }
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / f"folds_{result.name}.json").write_text(json.dumps(log, indent=2) + "\n",
                                                        encoding="utf-8")


def load_fold_params(run: str, logs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Return {left-out subject: tuned params} from an earlier run."""
    log = json.loads((Path(logs_dir) / f"folds_{run}.json").read_text(encoding="utf-8"))
    return {subj: fold["params"] for subj, fold in log["folds"].items()}
