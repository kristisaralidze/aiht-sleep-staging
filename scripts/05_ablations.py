#!/usr/bin/env python
"""Feature ablations (motion, HR, motion + HR) for forest, window forest and GRU.

Outputs: results/tables/ablations.csv, ablations_per_subject.csv
"""
import argparse
import sys
from functools import partial
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import context, loso, models, reporting  # noqa: E402
from src.experiment import Experiment, baseline_spec, resolve_paths  # noqa: E402
from src.gru import build_gru  # noqa: E402
from src.utils import load_config, set_seed, write_env_file, write_run_log  # noqa: E402

REQUIRED = ("seed", "paths", "epoch_s", "loso", "feature_sets", "models")
ABLATION_REQUIRED = ("feature_sets", "reused_set", "k_grid", "gru")
MODELS = ("forest", "window_forest", "gru")
SUMMARY = ["run", "model", "feature_set", "source", "n_subjects", "macro_f1_pooled",
           "f1_wake_pooled", "f1_nrem_pooled", "f1_rem_pooled", "kappa_subj_mean",
           "kappa_subj_sd", "kappa_pooled", "accuracy_pooled", "mean_inner_macro_f1"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "baselines.yaml")
    parser.add_argument("--ablation-config", type=Path,
                        default=ROOT / "configs" / "ablations.yaml")
    parser.add_argument("--gru-config", type=Path, default=ROOT / "configs" / "gru.yaml")
    parser.add_argument("--only", nargs="+", choices=MODELS, help="re-run these, keep other rows")
    return parser.parse_args()


def window_spec(feature_set, experiment, k_grid):
    """Tune k per fold on top of the forest tuned for the same feature set."""
    return loso.ModelSpec(
        name=f"window_forest_{feature_set}",
        build=partial(context.build_windowed, models.BUILDERS["forest"]),
        grid=[{"k": k} for k in k_grid],
        fold_params=reporting.load_fold_params(f"forest_{feature_set}",
                                               experiment.paths["logs_dir"]),
    )


def gru_spec(feature_set, gru_config, chosen):
    """One fixed setting; the 24/6 inner split only decides when to stop."""
    params = {**gru_config["model"]["fixed"], **chosen}
    return loso.ModelSpec(name=f"gru_{feature_set}", build=build_gru, grid=[params])


def reused_rows(reused_set, tables_dir):
    """The all-features rows, taken from the existing runs."""
    runs = {"forest": f"forest_{reused_set}", "window_forest": f"window_forest_{reused_set}",
            "gru": f"gru_{reused_set}"}
    saved = pd.concat([pd.read_csv(tables_dir / "baselines.csv"),
                       pd.read_csv(tables_dir / "context.csv")]).drop_duplicates("run")
    rows = saved.set_index("run").loc[list(runs.values())].reset_index()
    rows["model"] = list(runs)
    rows["feature_set"] = reused_set
    rows["source"] = "existing run"
    return rows


def main():
    args = parse_args()
    config = load_config(args.config, REQUIRED)
    ablation = load_config(args.ablation_config, ABLATION_REQUIRED)
    gru_config = load_config(args.gru_config, ("model",))
    config = {**config, "feature_sets": {**config["feature_sets"], **ablation["feature_sets"]}}
    set_seed(config["seed"])
    paths = resolve_paths(ROOT, config)
    write_env_file(paths["logs_dir"] / "env.txt")
    experiment = Experiment(config, paths, script="05_ablations")

    specs = {
        "forest": lambda fs: baseline_spec(f"forest_{fs}", "forest", config),
        "window_forest": lambda fs: window_spec(fs, experiment, ablation["k_grid"]),
        "gru": lambda fs: gru_spec(fs, gru_config, ablation["gru"]),
    }
    rows, per_subject = [], []
    for model in args.only or MODELS:
        for feature_set in ablation["feature_sets"]:
            _, row, subject_rows = experiment.run(specs[model](feature_set),
                                                  experiment.subjs(feature_set))
            rows.append({"model": model, "feature_set": feature_set, "source": "ablation", **row})
            per_subject.append(subject_rows)

    tables_dir = paths["tables_dir"]
    reporting.upsert_rows(tables_dir / "ablations_per_subject.csv", pd.concat(per_subject),
                          ["run", "subject"])
    table = pd.concat([pd.DataFrame(rows), reused_rows(ablation["reused_set"], tables_dir)])
    reporting.upsert_rows(tables_dir / "ablations.csv", table[SUMMARY], ["run"])
    write_run_log(paths["logs_dir"] / "05_ablations.json",
                  {**config, "ablations": ablation, "gru": gru_config}, {"models": list(MODELS)})


if __name__ == "__main__":
    main()
