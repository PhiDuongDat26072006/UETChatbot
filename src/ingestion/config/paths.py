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
DATA_SOURCE_DIR = os.getenv("CHATBOT_DATA_SOURCE_DIR") or str(BASE_DIR / _REPO_PATHS.get("data_source_dir", "data/data_source"))
DATA_DIR = os.getenv("CHATBOT_DATA_DIR") or str(BASE_DIR / _REPO_PATHS.get("data_dir", "data/raw_data"))
RAW_DATA_DIR = DATA_DIR
PROCESSED_DATA_DIR = os.getenv("CHATBOT_PROCESSED_DIR") or str(BASE_DIR / _REPO_PATHS.get("processed_dir", "data/processed_data"))
CHUNK_DATA_DIR = os.getenv("CHATBOT_CHUNK_DATA_DIR") or str(BASE_DIR / _REPO_PATHS.get("chunk_dir", "data/chunk_data"))


def to_project_relative_path(
    path: str | Path | None,
    base_dir: Path | None = None,
    strict: bool = False,
) -> str:
    """Chuyển đổi đường dẫn file cục bộ thành đường dẫn tương đối so với project root (POSIX format).
    
    Giữ nguyên các URL http/https. Ngăn chặn triệt để path traversal (../).
    Nếu đường dẫn tuyệt đối không thể ánh xạ tin cậy vào project root (Requirement 11),
    giữ nguyên để xem xét thủ công trừ khi strict=True.
    """
    if not path:
        return ""
    str_path = str(path).strip()
    if str_path.startswith("http://") or str_path.startswith("https://"):
        return str_path

    # Chuẩn hóa dấu phân cách thư mục sang POSIX (/)
    posix_str = str_path.replace("\\", "/")

    # Ngăn chặn path traversal (../)
    if "../" in posix_str or posix_str.endswith("/..") or posix_str == "..":
        raise ValueError(f"Path traversal detected: {str_path}")

    root = (base_dir or BASE_DIR).resolve()
    p = Path(str_path)

    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(root)
            return rel.as_posix()
        except ValueError:
            # Hỗ trợ ánh xạ đường dẫn lịch sử chứa thư mục data/ từ máy khác
            if "/data/raw_data/" in posix_str:
                idx = posix_str.index("/data/raw_data/")
                return posix_str[idx + 1:]
            elif "/data/" in posix_str:
                idx = posix_str.index("/data/")
                sub = posix_str[idx + 1:]  # data/...
                # Nếu là data/<domain>/files/ -> ánh xạ sang data/raw_data/<domain>/files/
                parts = sub.split("/")
                if len(parts) >= 3 and parts[0] == "data" and parts[1] not in ("raw_data", "data_source", "processed_data", "chunk_data"):
                    return f"data/raw_data/{'/'.join(parts[1:])}"
                return sub
            if strict:
                raise ValueError(f"Path outside project root: {str_path}")
            logger.warning("Đường dẫn không thuộc project root, giữ nguyên để kiểm tra thủ công: %s", str_path)
            return posix_str

    norm = os.path.normpath(posix_str).replace("\\", "/")
    if norm.startswith("../") or norm == "..":
        raise ValueError(f"Path traversal detected: {str_path}")
    return norm


def resolve_project_path(rel_path: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve một project-relative path thành absolute Path trên filesystem dựa trên BASE_DIR.
    
    Đảm bảo đường dẫn đích nằm hoàn toàn bên trong BASE_DIR, ngăn chặn path traversal.
    """
    if not rel_path:
        raise ValueError("Cannot resolve empty path")
    rel_str = str(rel_path).strip().replace("\\", "/")
    if "../" in rel_str or rel_str.endswith("/..") or rel_str == "..":
        raise ValueError(f"Path traversal detected: {rel_str}")

    root = (base_dir or BASE_DIR).resolve()
    p = Path(rel_str)
    if p.is_absolute():
        resolved = p.resolve()
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            raise ValueError(f"Absolute path outside project root: {rel_str}")

    target = (root / rel_str).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(f"Resolved path escapes project root: {target}")
    return target


