"""Evaluation metrics. Macro and per-class F1 are pooled; kappa is the per-subject mean.
Per-subject F1 skips classes absent from that subject's labels.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from src.epochs import CLASS_NAMES

CLASSES = tuple(range(len(CLASS_NAMES)))
PER_CLASS = tuple(f"f1_{name}" for name in CLASS_NAMES)
METRICS = ("accuracy", "kappa", "macro_f1", *PER_CLASS)


def _class_f1(y_true: np.ndarray, y_pred: np.ndarray, code: int) -> float:
    if not (y_true == code).any():
        return float("nan")
    return float(f1_score(y_true, y_pred, labels=[code], average="macro", zero_division=0.0))


def score(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Score one set of predictions; see the module docstring for conventions."""
    per_class = {name: _class_f1(y_true, y_pred, code) for code, name in zip(CLASSES, PER_CLASS)}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=list(CLASSES))),
        "macro_f1": float(np.nanmean(list(per_class.values()))),
        **per_class,
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 over classes present in y_true, the inner-CV selection metric."""
    return float(np.nanmean([_class_f1(y_true, y_pred, code) for code in CLASSES]))


def summarise_subjects(per_subject: pd.DataFrame) -> Dict[str, float]:
    """Mean and SD across subjects, skipping undefined values."""
    summary = {
        "n_subjects": int(len(per_subject)),
        "n_subjects_rem": int(per_subject["f1_rem"].notna().sum()),
    }
    for metric in METRICS:
        summary[f"{metric}_subj_mean"] = float(per_subject[metric].mean())
        summary[f"{metric}_subj_sd"] = float(per_subject[metric].std(ddof=1))
    return summary


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the confusion matrix as counts and as row-normalised rates."""
    counts = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
    row_totals = np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return counts, counts / row_totals
