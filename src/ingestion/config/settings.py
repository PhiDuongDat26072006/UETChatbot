"""Defaults are conservative and may be overridden from CLI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PROFILE_DIR = Path(__file__).parent / "sites"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "app.log"
PROCESSOR_VERSION = "html-v4-preserve"


def load_runtime():
    """Read editable operational defaults; invalid values fail before any HTTP."""
    import math
    import yaml

    path = Path(__file__).parent / "runtime.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "ingestion": {
            "max_pages",
            "max_depth",
            "limit",
            "download_workers",
            "site_workers",
            "flush_every",
            "per_host",
            "extract_workers",
            "stream_extract_workers",
            "timeout",
            "max_html_bytes",
            "retries",
            "delay",
        },
        "pdf": {"rpm", "retries"},
    }
    if not isinstance(config, dict) or set(config) != set(expected):
        raise ValueError(f"Invalid sections in {path}")
    for section, keys in expected.items():
        values = config[section]
        if not isinstance(values, dict) or set(values) != keys:
            raise ValueError(
                f"Invalid keys in {path}: {section}; expected {sorted(keys)}"
            )
        for key, value in values.items():
            if key in {"limit", "stream_extract_workers"} and value is None:
                continue
            number_type = (int, float) if key == "delay" else (int,)
            minimum = 0 if key in {"delay", "retries", "max_depth"} else 1
            if (
                isinstance(value, bool)
                or not isinstance(value, number_type)
                or not math.isfinite(value)
                or value < minimum
            ):
                raise ValueError(f"Invalid {section}.{key}: {value!r}")
    return config
