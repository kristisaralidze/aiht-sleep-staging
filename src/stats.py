"""Paired per-subject comparisons between LOSO runs."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from scipy.stats import wilcoxon


def paired_comparison(per_subject: pd.DataFrame, baseline: str, model: str) -> Dict[str, Any]:
    """Compare per-subject kappa with a two-sided Wilcoxon signed-rank test."""
    wide = per_subject.pivot(index="subject", columns="run", values="kappa")[[baseline, model]]
    if wide.isna().any(axis=None):
        raise ValueError(f"{baseline} and {model} do not cover the same subjects")
    diff = wide[model] - wide[baseline]
    result = wilcoxon(diff)
    return {
        "baseline": baseline,
        "model": model,
        "n_subjects": int(len(diff)),
        "mean_baseline": float(wide[baseline].mean()),
        "mean_model": float(wide[model].mean()),
        "mean_diff": float(diff.mean()),
        "sd_diff": float(diff.std(ddof=1)),
        "median_diff": float(diff.median()),
        "n_improved": int((diff > 0).sum()),
        "n_worse": int((diff < 0).sum()),
        "n_tied": int((diff == 0).sum()),
        "wilcoxon_stat": float(result.statistic),
        "p_value": float(result.pvalue),
    }
