#!/usr/bin/env python
"""Context baselines: +/-k windowed features and HMM smoothing.

Uses the best baseline chosen by script 02 (inner CV only). Its tuned parameters
are reused per fold, so each row changes one thing: the context.

Outputs:
    results/tables/context.csv               Table 2, references and context rows
    results/tables/context_per_subject.csv   every metric for every left-out subject
    plus predictions, confusion figures, fold logs and timing as in script 02
"""
import argparse
import sys
from functools import partial
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import context, hmm, loso, models, reporting  # noqa: E402
from src.experiment import Experiment, resolve_paths  # noqa: E402
from src.utils import load_config, set_seed, write_env_file, write_run_log  # noqa: E402

REQUIRED = ("seed", "paths", "epoch_s", "loso", "feature_sets", "models", "context")
LOCAL_SET = "local"
RUNS = ("window_local", "window_best", "hmm")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baselines.yaml")
    parser.add_argument("--only", nargs="+", choices=RUNS, help="re-run these, keep other rows")
    return parser.parse_args()


def window_spec(model, feature_set, experiment):
    """Tune k per fold, keeping the base model's tuned parameters for that fold."""
    return loso.ModelSpec(
        name=f"window_{model}_{feature_set}",
        build=partial(context.build_windowed, models.BUILDERS[model]),
        grid=[{"k": k} for k in experiment.config["context"]["k_grid"][feature_set]],
        fold_params=reporting.load_fold_params(f"{model}_{feature_set}",
                                               experiment.paths["logs_dir"]),
    )


def build_hmm(model, settings, n_inner, params,
              seed):
    return hmm.HmmSmoothed(models.BUILDERS[model], params, seed,
                           emission_weights=settings["emission_weights"],
                           pseudocount=settings["pseudocount"], n_inner=n_inner)


def hmm_spec(best, experiment):
    config = experiment.config
    return loso.ModelSpec(
        name=f"hmm_{best['run']}",
        build=partial(build_hmm, best["model"], config["context"], config["loso"]["n_inner"]),
        grid=[{}],
        fold_params=reporting.load_fold_params(best["run"], experiment.paths["logs_dir"]),
    )


def reference_rows(best, tables_dir):
    """Rows from Table 1 that the context models are compared against."""
    baselines = pd.read_csv(tables_dir / "baselines.csv")
    wanted = {best["run"], f"{best['model']}_{LOCAL_SET}"}
    return baselines[baselines["run"].isin(wanted)].assign(context="none (reference)")


def main():
    args = parse_args()
    config = load_config(args.config, REQUIRED)
    set_seed(config["seed"])
    paths = resolve_paths(ROOT, config)
    write_env_file(paths["logs_dir"] / "env.txt")
    experiment = Experiment(config, paths, script="03_context_baselines")

    selection = pd.read_csv(paths["tables_dir"] / "baseline_selection.csv")
    best = selection[selection["selected"]].iloc[0]
    model = best["model"]

    jobs = {
        "window_local": (window_spec(model, LOCAL_SET, experiment), LOCAL_SET, "window +/-k"),
        "window_best": (window_spec(model, best["feature_set"], experiment), best["feature_set"],
                        "window +/-k"),
        "hmm": (hmm_spec(best, experiment), best["feature_set"], "HMM"),
    }
    if best["feature_set"] == LOCAL_SET:
        del jobs["window_best"]
    rows = []
    per_subject = []
    for name in args.only or list(jobs):
        spec, feature_set, label = jobs[name]
        _, row, subject_rows = experiment.run(spec, experiment.subjs(feature_set))
        rows.append({"model": model, "feature_set": feature_set, "context": label, **row})
        per_subject.append(subject_rows)

    tables_dir = paths["tables_dir"]
    table = pd.concat([reference_rows(best, tables_dir), pd.DataFrame(rows)])
    reporting.upsert_rows(tables_dir / "context.csv", table, ["run"])
    reporting.upsert_rows(tables_dir / "context_per_subject.csv", pd.concat(per_subject),
                          ["run", "subject"])
    write_run_log(paths["logs_dir"] / "03_context_baselines.json", config,
                  {"best_baseline": best["run"]})


if __name__ == "__main__":
    main()
