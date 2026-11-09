#!/usr/bin/env python
"""Paired per-subject kappa comparisons: context models, the GRU and feature ablations.

Output: results/tables/stats.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stats import paired_comparison  # noqa: E402

TABLES = ROOT / "results" / "tables"
LOCAL_SET = "local"
GRU_RUN = "gru_all"
ABLATION_MODELS = ("forest", "window_forest", "gru")


def comparisons(best):
    """Pairs of (baseline, context run), each differing only in the context."""
    model, feature_set = best["model"], best["feature_set"]
    pairs = [(best["run"], f"window_{model}_{feature_set}"), (best["run"], f"hmm_{best['run']}")]
    if feature_set != LOCAL_SET:
        pairs.append((f"{model}_{LOCAL_SET}", f"window_{model}_{LOCAL_SET}"))
    return pairs


def gru_comparisons(best):
    window = f"window_{best['model']}_{best['feature_set']}"
    return [(best["run"], GRU_RUN), (window, GRU_RUN), (f"hmm_{best['run']}", GRU_RUN)]


def ablation_comparisons():
    """For each model: what adding motion, HR, or time features changes."""
    pairs = []
    for model in ABLATION_MODELS:
        pairs += [(f"{model}_hr", f"{model}_motion_hr"), (f"{model}_motion", f"{model}_motion_hr"),
                  (f"{model}_motion_hr", f"{model}_all")]
    return pairs


def main():
    selection = pd.read_csv(TABLES / "baseline_selection.csv")
    best = selection[selection["selected"]].iloc[0]
    names = ["baselines_per_subject.csv", "context_per_subject.csv", "ablations_per_subject.csv"]
    per_subject = pd.concat([pd.read_csv(TABLES / name, dtype={"subject": str})
                             for name in names if (TABLES / name).exists()])
    pairs = comparisons(best)
    if GRU_RUN in set(per_subject["run"]):
        pairs += gru_comparisons(best)
    available = set(per_subject["run"])
    pairs += [pair for pair in ablation_comparisons() if set(pair) <= available]
    table = pd.DataFrame([paired_comparison(per_subject, baseline, model)
                          for baseline, model in pairs])
    table.to_csv(TABLES / "stats.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
