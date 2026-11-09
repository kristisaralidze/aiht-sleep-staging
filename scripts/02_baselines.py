#!/usr/bin/env python
"""Per-epoch baselines under leave-one-subject-out, plus sensitivity runs.

Outputs: results/tables/baselines*.csv, baseline_selection.csv, sensitivity.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import reporting  # noqa: E402
from src.experiment import (Experiment, baseline_spec, resolve_paths,  # noqa: E402
                            sensitivity_rows)
from src.utils import load_config, set_seed, write_env_file, write_run_log  # noqa: E402

REQUIRED = ("seed", "paths", "epoch_s", "loso", "feature_sets", "models", "sensitivity")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baselines.yaml")
    parser.add_argument("--only", nargs="+", help="re-run these models, keep other saved rows")
    return parser.parse_args()


def run_baselines(experiment, model_names):
    """Run the given models on every feature set and update Table 1."""
    config = experiment.config
    results = {}
    rows = []
    per_subject = []
    for feature_set in config["feature_sets"]:
        subjs = experiment.subjs(feature_set)
        for model in model_names:
            name = f"{model}_{feature_set}"
            result, row, subject_rows = experiment.run(baseline_spec(name, model, config), subjs)
            results[name] = result
            rows.append({"model": model, "feature_set": feature_set, **row})
            per_subject.append(subject_rows)
    tables_dir = experiment.paths["tables_dir"]
    table = reporting.upsert_rows(tables_dir / "baselines.csv", pd.DataFrame(rows), ["run"])
    reporting.upsert_rows(tables_dir / "baselines_per_subject.csv", pd.concat(per_subject),
                          ["run", "subject"])
    return results, table


def select_best(table, tables_dir):
    """Choose the best baseline without looking at any left-out subject's score."""
    selection = table[["run", "model", "feature_set", "mean_inner_macro_f1"]].copy()
    selection = selection.sort_values("mean_inner_macro_f1", ascending=False)
    selection["selected"] = [True] + [False] * (len(selection) - 1)
    selection.to_csv(tables_dir / "baseline_selection.csv", index=False)
    return selection.iloc[0]


def run_sensitivity(experiment, best, results, model_names):
    config = experiment.config
    if best["model"] not in model_names:
        raise ValueError(f"Best baseline {best['run']} was not re-run; add it to --only")
    feature_set = best["feature_set"]
    rows = []
    for model in model_names:
        name = f"{model}_{feature_set}_no_low_coverage"
        rows += sensitivity_rows(experiment, baseline_spec(name, model, config), model,
                                 feature_set, config["sensitivity"]["low_coverage"],
                                 results[f"{model}_{feature_set}"])
    name = f"{best['run']}_no_low_rate"
    rows += sensitivity_rows(experiment, baseline_spec(name, best["model"], config), best["model"],
                             feature_set, config["sensitivity"]["low_rate"], results[best["run"]])
    table = reporting.upsert_rows(experiment.paths["tables_dir"] / "sensitivity.csv",
                                  pd.DataFrame(rows), ["excluded", "model", "training"])
    if table["feature_set"].nunique() != 1:
        raise ValueError("Sensitivity rows mix feature sets; re-run script 02 without --only")


def main():
    args = parse_args()
    config = load_config(args.config, REQUIRED)
    set_seed(config["seed"])
    paths = resolve_paths(ROOT, config)
    paths["tables_dir"].mkdir(parents=True, exist_ok=True)
    write_env_file(paths["logs_dir"] / "env.txt")
    experiment = Experiment(config, paths, script="02_baselines")

    model_names = args.only or list(config["models"])
    results, table = run_baselines(experiment, model_names)
    best = select_best(table, paths["tables_dir"])
    print(f"\nBest baseline by inner CV: {best['run']} "
          f"(macro F1 {best['mean_inner_macro_f1']:.3f})\n", flush=True)
    run_sensitivity(experiment, best, results, model_names)
    write_run_log(paths["logs_dir"] / "02_baselines.json", config,
                  {"best_baseline": best["run"]})


if __name__ == "__main__":
    main()
