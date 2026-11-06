#!/usr/bin/env python
"""Build per-subject epoch features from data/raw into data/processed.

Outputs:
    data/processed/<subject>.npz        features, labels, feature names, epoch times
    results/tables/class_balance.csv    epochs per subject and class balance
    results/logs/01_preprocess.json     seed, config and versions for this run
    results/logs/env.txt                exact Python and package versions
"""
import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import preprocess, raw_io  # noqa: E402
from src.utils import load_config, set_seed, write_env_file, write_run_log  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "preprocess.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    paths = {name: ROOT / value for name, value in config["paths"].items()}
    subjs = raw_io.list_subjects(paths["raw_dir"])

    rows = []
    for subj in tqdm(subjs, desc="subjects"):
        record = preprocess.process_subject(paths["raw_dir"], subj, config)
        preprocess.save_record(record, paths["processed_dir"])
        rows.append(preprocess.summary_row(record))

    table = preprocess.summary_table(rows)
    paths["tables_dir"].mkdir(parents=True, exist_ok=True)
    table_path = paths["tables_dir"] / "class_balance.csv"
    table.to_csv(table_path, index=False)
    write_run_log(paths["logs_dir"] / "01_preprocess.json", config,
                  {"n_subjects": len(subjs), "subjects": subjs})
    write_env_file(paths["logs_dir"] / "env.txt")

    print(table.to_string(index=False))
    print(f"\nSaved {len(subjs)} subjects to {paths['processed_dir']} and {table_path}")


if __name__ == "__main__":
    main()
