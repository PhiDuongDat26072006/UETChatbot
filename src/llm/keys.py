"""API key loading and log masking helpers for LLM providers."""

from __future__ import annotations

import os
from typing import List, Optional


def load_keys_for_provider(
    prefix: str,
    custom_key: Optional[str] = None,
    custom_keys: Optional[List[str]] = None,
) -> List[str]:
    """Load deduplicated API keys for ``prefix`` from arguments and environment variables."""
    keys: List[str] = []

    if custom_key == "":
        return []

    if custom_keys:
        for k in custom_keys:
            k_str = str(k).strip().strip("'\"")
            if k_str and k_str not in keys:
                keys.append(k_str)

    if custom_key:
        c = custom_key.strip().strip("'\"")
        if c and c not in keys:
            keys.append(c)

    for env_var in (f"{prefix}_API_KEYS", f"{prefix}_API_KEY", f"{prefix}_KEY"):
        raw_val = os.getenv(env_var, "")
        if raw_val:
            for item in raw_val.replace(";", "\n").replace(",", "\n").split("\n"):
                k = item.strip().strip("'\"")
                if k and not k.startswith("#") and k not in keys:
                    keys.append(k)

    for i in range(1, 21):
        k = os.getenv(f"{prefix}_API_KEY_{i}", "").strip().strip("'\"")
        if k and k not in keys:
            keys.append(k)

    return keys


def mask_key(key: str) -> str:
    """Mask an API key for safe logging."""
    if not key:
        return ""
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"
