#!/usr/bin/env bash
# Run the full pipeline from an empty results/ folder. Takes about 10 h on 8 CPU cores,
# most of it the GRU (script 04) and the ablations (script 05).
# 06 runs last because its figures and main_results.csv read the outputs of 03b, 05 and 07.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

if [ ! -d data/raw/motion ]; then
    bash scripts/00_download_data.sh
fi

mkdir -p results/logs
"$PY" scripts/01_preprocess.py
"$PY" scripts/01b_validate_vs_walch.py
"$PY" scripts/01c_motion_rate_check.py
"$PY" scripts/02_baselines.py | tee results/logs/02_baselines.log
"$PY" scripts/03_context_baselines.py | tee results/logs/03_context_baselines.log
"$PY" scripts/03b_window_k_scores.py
"$PY" scripts/04_gru.py | tee results/logs/04_gru.log
"$PY" scripts/05_ablations.py | tee results/logs/05_ablations.log
"$PY" scripts/07_stats.py | tee results/logs/07_stats.log
"$PY" scripts/08_edge_check.py
"$PY" scripts/09_subject_analysis.py
"$PY" scripts/06_figures.py
