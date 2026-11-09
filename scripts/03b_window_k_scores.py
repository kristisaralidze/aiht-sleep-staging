#!/usr/bin/env python
"""Inner-CV macro F1 for every window size k, per outer fold, for window_forest_all.

Output: results/tables/window_k_scores.csv
"""
import argparse
import sys
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import context, loso, models, reporting  # noqa: E402
from src.experiment import Experiment, resolve_paths  # noqa: E402
from src.utils import load_config  # noqa: E402

RUN, BASE_RUN, MODEL, FEATURE_SET = "window_forest_all", "forest_all", "forest", "all"
K_VALUES = (5, 10, 20, 30, 45, 60)


def fold_scores(spec, subjs, test_id, seed, n_inner):
    torch.set_num_threads(1)
    train = [subj for subj in subjs if subj.subject_id != test_id]
    fixed = dict(spec.fold_params[test_id])
    results = loso.inner_scores(spec, train, fixed, loso.fold_seed(seed, test_id), n_inner)
    return [{"subject": test_id, "k": result.params["k"], "inner_macro_f1": result.score}
            for result in results]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baselines.yaml")
    config = load_config(parser.parse_args().config,
                         ("seed", "paths", "epoch_s", "loso", "feature_sets", "context"))
    paths = resolve_paths(ROOT, config)
    subjs = Experiment(config, paths, script="03b_window_k_scores").subjs(FEATURE_SET)
    spec = loso.ModelSpec(
        name=RUN, build=partial(context.build_windowed, models.BUILDERS[MODEL]),
        grid=[{"k": k} for k in K_VALUES],
        fold_params=reporting.load_fold_params(BASE_RUN, paths["logs_dir"]))
    settings = config["loso"]
    parts = Parallel(n_jobs=settings["n_jobs"])(
        delayed(fold_scores)(spec, subjs, subj.subject_id, config["seed"], settings["n_inner"])
        for subj in subjs)
    table = pd.DataFrame([row for part in parts for row in part])

    tuned_grid = config["context"]["k_grid"][FEATURE_SET]
    table["in_tuned_grid"] = table["k"].isin(tuned_grid)
    logged = reporting.load_fold_params(RUN, paths["logs_dir"])
    best = (table[table["in_tuned_grid"]].sort_values(["subject", "inner_macro_f1"],
                                                      ascending=[True, False], kind="stable")
            .groupby("subject").head(1).set_index("subject")["k"])
    table["chosen"] = [k == logged[subj]["k"] for subj, k in zip(table["subject"], table["k"])]
    matches = sum(int(k == logged[subj]["k"]) for subj, k in best.items())
    table.to_csv(paths["tables_dir"] / "window_k_scores.csv", index=False)
    print(f"best k within the tuned grid matches script 03 in {matches} of {len(best)} folds")
    print(table.groupby("k")["inner_macro_f1"].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
