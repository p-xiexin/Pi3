from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


INDEX_VERSION = 1


def load_dataset_index(
    data_root: str | Path,
    index_file: str | Path,
    expected_dataset: str,
) -> tuple[Path, dict[str, Any]]:
    root = Path(data_root).expanduser().resolve()
    path = Path(index_file).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Index file not found: {path}. Build it with "
            f"python -m datasets.tools.build_index {expected_dataset} {root}"
        )
    payload = np.load(path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid dataset index payload in {path}")
    if payload.get("version") != INDEX_VERSION:
        raise ValueError(
            f"Unsupported index version {payload.get('version')} in {path}; "
            f"expected {INDEX_VERSION}"
        )
    if payload.get("dataset") != expected_dataset:
        raise ValueError(
            f"Index {path} is for {payload.get('dataset')!r}, "
            f"not {expected_dataset!r}"
        )
    sequences = payload.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError(f"Index {path} contains no sequences")
    return root, payload


def data_path(data_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


__all__ = ["INDEX_VERSION", "data_path", "load_dataset_index"]
