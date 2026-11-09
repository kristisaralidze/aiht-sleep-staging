#!/usr/bin/env python
"""Whole-night bidirectional GRU under leave-one-subject-out, plus sensitivity runs.

Outputs: GRU rows in results/tables/context*.csv and sensitivity.csv
"""
import argparse
import sys
import time
from functools import partial
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import loso, reporting  # noqa: E402
from src.experiment import (Experiment, model_grid, resolve_paths,  # noqa: E402
                            sensitivity_rows)
from src.gru import build_gru_search  # noqa: E402
from src.utils import load_config, set_seed, write_env_file, write_run_log  # noqa: E402

REQUIRED = ("seed", "paths", "epoch_s", "loso", "feature_sets", "sensitivity")
MODEL = "gru"
CONTEXT = "BiGRU, whole night"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baselines.yaml")
    parser.add_argument("--gru-config", type=Path, default=ROOT / "configs" / "gru.yaml")
    parser.add_argument("--fold", help="time one outer fold for this subject and exit")
    return parser.parse_args()


def gru_spec(name, gru_config):
    """All candidates go to one search per fold, tuned on a fixed 24/6 subject split."""
    build = partial(build_gru_search, model_grid(gru_config["model"]))
    return loso.ModelSpec(name=name, build=build, grid=[{}])


def time_one_fold(experiment, gru_config, subject_id):
    """Time one outer fold and project the full run, without saving anything."""
    subjs = experiment.subjs(gru_config["feature_set"])
    test = next(subj for subj in subjs if subj.subject_id == subject_id)
    train = [subj for subj in subjs if subj is not test]
    settings = experiment.config["loso"]
    torch.set_num_threads(1)  # match one LOSO worker
    started = time.perf_counter()
    fold = loso.run_fold(gru_spec("timing", gru_config), train, test,
                         loso.fold_seed(experiment.config["seed"], subject_id), settings["n_inner"])
    seconds = time.perf_counter() - started
    n_folds = len(subjs) + (len(subjs) - 1) + (len(subjs) - 2)
    rounds = -(-len(subjs) // settings["n_jobs"]) + 2 * -(-(len(subjs) - 1) // settings["n_jobs"])
    print(f"one fold: {seconds / 60:.1f} min, chose {fold.params}")
    print(f"main run: {len(subjs)} folds on {settings['n_jobs']} workers, "
          f"about {seconds * -(-len(subjs) // settings['n_jobs']) / 3600:.1f} h")
    print(f"with sensitivity runs: {n_folds} folds, about {seconds * rounds / 3600:.1f} h")


def main():
    args = parse_args()
    config = load_config(args.config, REQUIRED)
    gru_config = load_config(args.gru_config, ("feature_set", "model"))
    set_seed(config["seed"])
    paths = resolve_paths(ROOT, config)
    experiment = Experiment(config, paths, script="04_gru")
    if args.fold:
        time_one_fold(experiment, gru_config, args.fold)
        return
    write_env_file(paths["logs_dir"] / "env.txt")

    feature_set = gru_config["feature_set"]
    name = f"{MODEL}_{feature_set}"
    full, row, per_subject = experiment.run(gru_spec(name, gru_config),
                                            experiment.subjs(feature_set))
    tables_dir = paths["tables_dir"]
    reporting.upsert_rows(tables_dir / "context.csv", pd.DataFrame(
        [{"model": MODEL, "feature_set": feature_set, "context": CONTEXT, **row}]), ["run"])
    reporting.upsert_rows(tables_dir / "context_per_subject.csv", per_subject,
                          ["run", "subject"])

    rows = []
    for suffix, key in (("no_low_coverage", "low_coverage"), ("no_low_rate", "low_rate")):
        spec = gru_spec(f"{name}_{suffix}", gru_config)
        rows += sensitivity_rows(experiment, spec, MODEL, feature_set,
                                 config["sensitivity"][key], full)
    reporting.upsert_rows(tables_dir / "sensitivity.csv", pd.DataFrame(rows),
                          ["excluded", "model", "training"])
    write_run_log(paths["logs_dir"] / "04_gru.json", {**config, "gru": gru_config},
                  {"run": name})


if __name__ == "__main__":
    main()
