"""Crash-safe per-window artifacts for sequence evaluation."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = 1


def _atomic_json(path: Path, value: Mapping) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_png(path: Path, image) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG")
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


class WindowArtifactStore:
    """Treat one window JSON as the authoritative commit marker."""

    def __init__(
        self,
        sequence_dir: Path,
        fingerprint: str,
        metadata: Mapping,
        planned_positions: Mapping[int, Sequence[int]],
        metric_fields: Sequence[str],
        timing_fields: Sequence[str],
        resolved_config: str,
    ) -> None:
        self.sequence_dir = Path(sequence_dir)
        self.sequence_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = str(fingerprint)
        self.metadata = dict(metadata)
        self.planned_positions = {
            int(index): tuple(int(position) for position in positions)
            for index, positions in planned_positions.items()
        }
        self.metric_fields = tuple(metric_fields)
        self.timing_fields = tuple(timing_fields)
        self.metrics_path = self.sequence_dir / "metrics.csv"
        self.timings_path = self.sequence_dir / "timings.csv"
        self.progress_path = self.sequence_dir / "progress.json"
        self.summary_path = self.sequence_dir / "summary.json"
        self.resolved_config = resolved_config
        self.commits: dict[int, dict] = {}

        self._load_commits()
        if self.commits:
            _atomic_text(
                self.sequence_dir / "resolved_config.yaml",
                self.resolved_config,
            )
            self._rebuild_csv_files()

    def _load_commits(self) -> None:
        for path in sorted(self.sequence_dir.glob("chunk_*.json")):
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(f"unsupported window artifact {path}")
            if payload.get("fingerprint") != self.fingerprint:
                raise RuntimeError(
                    f"existing window artifact uses another evaluation "
                    f"configuration: {path}"
                )
            index = int(payload["window_index"])
            expected_positions = self.planned_positions.get(index)
            positions = tuple(int(value) for value in payload["positions"])
            if expected_positions != positions:
                raise RuntimeError(f"window plan changed for existing artifact {path}")
            image_path = self.sequence_dir / str(payload["image"])
            if not image_path.is_file():
                raise RuntimeError(f"window artifact has no image {image_path}")
            if index in self.commits:
                raise RuntimeError(f"duplicate committed window index {index}")
            self.commits[index] = payload

    def _ordered_commits(self) -> list[dict]:
        return [self.commits[index] for index in sorted(self.commits)]

    def _metric_rows(self) -> list[dict]:
        return [
            row
            for commit in self._ordered_commits()
            for row in commit["metrics"]
        ]

    def _timing_rows(self) -> list[dict]:
        return [commit["timing"] for commit in self._ordered_commits()]

    def _rebuild_csv_files(self) -> None:
        _write_csv(self.metrics_path, self._metric_rows(), self.metric_fields)
        _write_csv(self.timings_path, self._timing_rows(), self.timing_fields)

    @property
    def completed_indices(self) -> frozenset[int]:
        return frozenset(self.commits)

    def is_completed(self, window_index: int) -> bool:
        return int(window_index) in self.commits

    def commit(
        self,
        window_index: int,
        positions: Sequence[int],
        image,
        metric_rows: Sequence[Mapping],
        timing_row: Mapping,
    ) -> None:
        window_index = int(window_index)
        if window_index in self.commits:
            raise RuntimeError(f"window {window_index} is already committed")
        positions = tuple(int(position) for position in positions)
        if self.planned_positions.get(window_index) != positions:
            raise RuntimeError(f"window {window_index} does not match the plan")
        first_commit = not self.commits

        stem = (
            f"chunk_{window_index:05d}_"
            f"{positions[0]:06d}_{positions[-1]:06d}"
        )
        image_path = self.sequence_dir / f"{stem}.png"
        commit_path = self.sequence_dir / f"{stem}.json"
        _atomic_png(image_path, image)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "window_index": window_index,
            "positions": list(positions),
            "image": image_path.name,
            "metrics": [dict(row) for row in metric_rows],
            "timing": dict(timing_row),
        }
        _atomic_json(commit_path, payload)
        self.commits[window_index] = payload
        if first_commit:
            _atomic_text(
                self.sequence_dir / "resolved_config.yaml",
                self.resolved_config,
            )
            _write_csv(self.metrics_path, [], self.metric_fields)
            _write_csv(self.timings_path, [], self.timing_fields)
        _append_csv(self.metrics_path, payload["metrics"], self.metric_fields)
        _append_csv(self.timings_path, [payload["timing"]], self.timing_fields)
        self.write_state("running")

    def aggregates(self) -> dict:
        metric_rows = self._metric_rows()
        timing_rows = self._timing_rows()
        valid_pixels = sum(int(row["valid_pixels"]) for row in metric_rows)
        loss_sum = sum(float(row["loss_sum"]) for row in metric_rows)
        timing_names = (
            "pi3_backbone_seconds",
            "pi3_point_depth_decode_seconds",
            "glob3r_seconds",
            "network_seconds",
        )
        timing = {"unit": "seconds", "windows": len(timing_rows)}
        for name in timing_names:
            total = sum(float(row[name]) for row in timing_rows)
            timing[name] = {
                "total": total,
                "mean": total / len(timing_rows) if timing_rows else None,
            }
        return {
            "windows_completed": len(self.commits),
            "directed_pairs": len(metric_rows),
            "valid_pixels": valid_pixels,
            "loss_sum": loss_sum,
            "mean_warp_loss": loss_sum / valid_pixels if valid_pixels else None,
            "timing": timing,
        }

    def write_state(
        self,
        status: str,
        error: Mapping | None = None,
        failed_window: int | None = None,
    ) -> None:
        aggregate = self.aggregates()
        completed = sorted(self.commits)
        pending = sorted(set(self.planned_positions) - set(self.commits))
        summary = {
            **self.metadata,
            "status": status,
            "windows_total": len(self.planned_positions),
            **aggregate,
            "error": dict(error) if error is not None else None,
        }
        progress = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "status": status,
            "windows_total": len(self.planned_positions),
            "windows_completed": len(completed),
            "completed_window_indices": completed,
            "next_window_index": pending[0] if pending else None,
            "failed_window_index": failed_window,
            "valid_pixels": aggregate["valid_pixels"],
            "loss_sum": aggregate["loss_sum"],
            "mean_warp_loss": aggregate["mean_warp_loss"],
            "error": dict(error) if error is not None else None,
        }
        _atomic_json(self.summary_path, summary)
        _atomic_json(self.progress_path, progress)


__all__ = ["WindowArtifactStore"]
