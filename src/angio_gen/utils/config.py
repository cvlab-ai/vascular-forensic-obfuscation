from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_weight_config_path() -> Path:
    return get_repo_root() / "src" / "angio_gen" / "config" / "weights.yaml"


@lru_cache(maxsize=1)
def load_weight_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config_file = Path(config_path) if config_path is not None else get_weight_config_path()

    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    repo_root = get_repo_root()
    weights = config.get("weights", {})
    config["weights"] = {
        name: str((repo_root / value).resolve()) if not Path(value).is_absolute() else value
        for name, value in weights.items()
    }

    return config
