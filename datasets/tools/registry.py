"""Load and validate the declarative dataset catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROFILE_ALIASES = {
    "minimal": "minimal",
    "test": "minimal",
    "train": "train",
    "full": "train",
}


@dataclass(frozen=True)
class DatasetCatalog:
    path: Path
    datasets: dict[str, dict[str, Any]]

    def get(self, dataset_id: str) -> dict[str, Any]:
        key = dataset_id.lower()
        if key not in self.datasets:
            choices = ", ".join(sorted(self.datasets))
            raise KeyError(f"Unknown dataset {dataset_id}. Available datasets: {choices}")
        return self.datasets[key]

    def profile(self, dataset_id: str, profile: str) -> tuple[dict[str, Any], str]:
        dataset = self.get(dataset_id)
        normalized = PROFILE_ALIASES.get(profile.lower())
        if normalized is None:
            raise KeyError(f"Unknown profile {profile}. Use minimal, test, train, or full")
        try:
            return dataset["profiles"][normalized], normalized
        except KeyError as exc:
            raise KeyError(f"Dataset {dataset_id} has no {normalized} profile") from exc


def load_catalog(path: Path | None = None) -> DatasetCatalog:
    catalog_path = path or Path(__file__).with_name("catalog.json")
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"Catalog has no datasets: {catalog_path}")
    required = {"name", "homepage", "license_url", "profiles", "verify"}
    for dataset_id, record in datasets.items():
        missing = required.difference(record)
        if missing:
            raise ValueError(f"Catalog entry {dataset_id} misses {sorted(missing)}")
        if set(record["profiles"]) != {"minimal", "train"}:
            raise ValueError(f"Catalog entry {dataset_id} must define minimal and train")
        for profile_name, profile in record["profiles"].items():
            if profile.get("mode") not in {"automatic", "official-tool", "authorized-input"}:
                raise ValueError(
                    f"Unsupported mode for {dataset_id}/{profile_name}: {profile.get('mode')}"
                )
            if "default_output" not in profile or "recipe" not in profile:
                raise ValueError(f"Incomplete profile {dataset_id}/{profile_name}")
    return DatasetCatalog(catalog_path, datasets)
