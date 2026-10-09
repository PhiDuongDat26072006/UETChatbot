"""Repository data paths, with YAML defaults and environment overrides."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[3]


def _load_repo_paths(config_path: Path | None = None) -> dict[str, str]:
    """Read valid path strings; missing files use defaults, malformed values warn."""
    config_path = config_path or BASE_DIR / "config.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        logger.warning("Cannot read path configuration %s: %s", config_path, error)
        return {}
    if config is None:
        return {}
    if not isinstance(config, dict):
        logger.warning("Configuration must be a mapping: %s", config_path)
        return {}
    paths_config = config.get("paths", {})
    if not isinstance(paths_config, dict):
        logger.warning("paths must be a mapping: %s", config_path)
        return {}
    valid_paths = {}
    for key, value in paths_config.items():
        if isinstance(value, str) and value.strip():
            valid_paths[key] = value
        else:
            logger.warning("Invalid path value for %s; using its default", key)
    return valid_paths


_REPO_PATHS = _load_repo_paths()
DATA_DIR = os.getenv("CHATBOT_DATA_DIR") or str(BASE_DIR / _REPO_PATHS.get("data_dir", "data/raw_data"))
PROCESSED_DATA_DIR = os.getenv("CHATBOT_PROCESSED_DIR") or str(BASE_DIR / _REPO_PATHS.get("processed_dir", "data/processed_data"))
