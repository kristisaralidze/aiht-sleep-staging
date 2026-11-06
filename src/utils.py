"""Seeding, config loading and run logging shared by all scripts."""
from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import yaml

REQUIRED_CONFIG_KEYS = ("seed", "paths")


def set_seed(seed: int) -> None:
    """Seed every random number generator the project uses."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: Path | str,
                required: Sequence[str] = REQUIRED_CONFIG_KEYS) -> Dict[str, Any]:
    """Read a YAML config and fail fast if a top-level section is missing."""
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    missing = [key for key in required if key not in (config or {})]
    if missing:
        raise ValueError(f"{path} is missing config keys: {missing}")
    return config


def write_run_log(path: Path, config: Dict[str, Any], extra: Dict[str, Any]) -> None:
    """Record the seed, config and library versions behind a run."""
    log = {
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config["seed"],
        "config": config,
        "python": platform.python_version(),
        "numpy": np.__version__,
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")


def write_env_file(path: Path) -> None:
    """Record the exact Python and package versions behind the results."""
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                            capture_output=True, text=True).stdout
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"python {platform.python_version()}\n{platform.platform()}\n\n{frozen}",
                    encoding="utf-8")
