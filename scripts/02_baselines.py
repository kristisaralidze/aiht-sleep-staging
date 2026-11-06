#!/usr/bin/env python
"""Per-epoch baselines under leave-one-subject-out, plus sensitivity runs.

Outputs:
    results/tables/baselines.csv              Table 1, one row per model and feature set
    results/tables/baselines_per_subject.csv  every metric for every left-out subject
    results/tables/baseline_selection.csv     best baseline, chosen by inner-CV macro F1
    results/tables/sensitivity.csv            retrained runs vs the full run, same subjects
    results/predictions/<run>_<subject>.csv   predictions and probabilities
    results/figures/confusion_<run>.pdf/png   pooled confusion matrices
    results/logs/folds_<run>.json             tuned parameters per fold
    results/logs/timing.csv                   compute per run
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import reporting  # noqa: E402
from src.experiment import Experiment, baseline_spec, resolve_paths  # noqa: E402
from src.utils import load_config, set_seed, write_env_file, write_run_log  # noqa: E402

REQUIRED = ("seed", "paths", "epoch_s", "loso", "feature_sets", "models", "sensitivity")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baselines.yaml")
    return parser.parse_args()


def run_all_baselines(experiment):
    """Run every model on every feature set and write Table 1."""
    config = experiment.config
    results = {}
    rows = []
    per_subject = []
    for feature_set in config["feature_sets"]:
        subjs = experiment.subjs(feature_set)
        for model in config["models"]:
            name = f"{model}_{feature_set}"
            result, row, subject_rows = experiment.run(baseline_spec(name, model, config), subjs)
            results[name] = result
            rows.append({"model": model, "feature_set": feature_set, **row})
            per_subject.append(subject_rows)
    table = pd.DataFrame(rows)
    tables_dir = experiment.paths["tables_dir"]
    table.to_csv(tables_dir / "baselines.csv", index=False)
    pd.concat(per_subject).to_csv(tables_dir / "baselines_per_subject.csv", index=False)
    return results, table


def select_best(table, tables_dir):
    """Choose the best baseline without looking at any left-out subject's score."""
    selection = table[["run", "model", "feature_set", "mean_inner_macro_f1"]].copy()
    selection = selection.sort_values("mean_inner_macro_f1", ascending=False)
    selection["selected"] = [True] + [False] * (len(selection) - 1)
    selection.to_csv(tables_dir / "baseline_selection.csv", index=False)
    return selection.iloc[0]


def sensitivity_row(experiment, name, model, feature_set,
                    excluded, full):
    """Retrain without `excluded`, and score the full run on the same subjects."""
    spec = baseline_spec(name, model, experiment.config)
    subjs = experiment.subjs(feature_set, exclude=excluded)
    _, retrained, _ = experiment.run(spec, subjs)
    kept = [subj.subject_id for subj in subjs]
    same_subjects, _ = reporting.evaluate(full, only_subjects=kept)
    label = {"excluded": "+".join(excluded), "model": model, "feature_set": feature_set}
    return [{**label, "training": "retrained without excluded", **retrained},
            {**label, "training": "full run, same test subjects", **same_subjects}]


def run_sensitivity(experiment, best,
                    results):
    config = experiment.config
    feature_set = best["feature_set"]
    rows = []
    for model in config["models"]:
        name = f"{model}_{feature_set}_no_low_coverage"
        rows += sensitivity_row(experiment, name, model, feature_set,
                                config["sensitivity"]["low_coverage"],
                                results[f"{model}_{feature_set}"])
    rows += sensitivity_row(experiment, f"{best['run']}_no_low_rate", best["model"], feature_set,
                            config["sensitivity"]["low_rate"], results[best["run"]])
    pd.DataFrame(rows).to_csv(experiment.paths["tables_dir"] / "sensitivity.csv", index=False)


def main():
    args = parse_args()
    config = load_config(args.config, REQUIRED)
    set_seed(config["seed"])
    paths = resolve_paths(ROOT, config)
    paths["tables_dir"].mkdir(parents=True, exist_ok=True)
    write_env_file(paths["logs_dir"] / "env.txt")
    experiment = Experiment(config, paths, script="02_baselines")

    results, table = run_all_baselines(experiment)
    best = select_best(table, paths["tables_dir"])
    print(f"\nBest baseline by inner CV: {best['run']} "
          f"(macro F1 {best['mean_inner_macro_f1']:.3f})\n", flush=True)
    run_sensitivity(experiment, best, results)
    write_run_log(paths["logs_dir"] / "02_baselines.json", config,
                  {"best_baseline": best["run"]})


if __name__ == "__main__":
    main()
