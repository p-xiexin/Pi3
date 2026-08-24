"""Evaluate trained Glob3R dense warps over complete driving sequences."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import hydra
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf

from eval.evaluation_utils import (
    build_sequence_windows,
    final_warp_and_confidence,
    pixelwise_warp_metrics,
    render_evaluation_matrix,
    shard_weighted_items,
)
from eval.artifact_store import WindowArtifactStore
from eval.window_adapters import create_window_adapter
from local_opt.inference import load_glob3r_for_sfm
from pi3.models.glob3r.geometry import build_ground_truth_warp


CSV_FIELDS = (
    "dataset",
    "sequence",
    "window_index",
    "window_first_position",
    "window_last_position",
    "reference_instance",
    "target_instance",
    "target_index",
    "valid_pixels",
    "loss_sum",
    "mean_loss",
)

TIMING_FIELDS = (
    "dataset",
    "sequence",
    "window_index",
    "window_first_position",
    "window_last_position",
    "frame_count",
    "pi3_backbone_seconds",
    "pi3_point_depth_decode_seconds",
    "glob3r_seconds",
    "network_seconds",
)


@dataclass(frozen=True)
class SequenceTask:
    dataset_offset: int
    dataset_name: str
    sequence_index: int
    sequence_id: str
    frame_count: int
    window_count: int


@dataclass(frozen=True)
class NetworkTiming:
    pi3_backbone_seconds: float
    pi3_point_depth_decode_seconds: float
    glob3r_seconds: float
    network_seconds: float


@dataclass(frozen=True)
class WindowResult:
    image: object
    metric_rows: list[dict]
    timing_row: dict
    valid_pixels: int
    loss_sum: float


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return normalized or "sequence"


def _append_rank_failure(
    output_dir: Path,
    rank: int,
    payload: Mapping,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"rank_{int(rank)}_failures.jsonl"
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rank": int(rank),
        **dict(payload),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _build_tasks(
    dataset_names: list[str],
    datasets: Mapping[str, object],
    chunk_size: int,
) -> list[SequenceTask]:
    tasks = []
    for dataset_offset, dataset_name in enumerate(dataset_names):
        dataset = datasets[dataset_name]
        for sequence_index, sequence_id in enumerate(dataset.sequences):
            sequence_id = str(sequence_id)
            frame_count = int(dataset.num_imgs[sequence_id])
            tasks.append(
                SequenceTask(
                    dataset_offset=dataset_offset,
                    dataset_name=dataset_name,
                    sequence_index=sequence_index,
                    sequence_id=sequence_id,
                    frame_count=frame_count,
                    window_count=len(
                        build_sequence_windows(
                            frame_count,
                            int(dataset.frame_step),
                            chunk_size,
                        )
                    ),
                )
            )
    return tasks


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _decode_point_depth(model, geometry_tokens, positions, frame_count, height, width):
    """Decode the Pi3 local point map and its confidence without camera poses."""

    backbone = model.backbone
    batch = geometry_tokens.shape[0] // frame_count
    with torch.amp.autocast(device_type=geometry_tokens.device.type, enabled=False):
        point_hidden = backbone.point_decoder(geometry_tokens, xpos=positions).float()
        point_output = backbone.point_head(
            [point_hidden[:, backbone.patch_start_idx :]], (height, width)
        ).reshape(batch, frame_count, height, width, -1)
        xy, depth = point_output.split((2, 1), dim=-1)
        depth = depth.exp()
        local_points = torch.cat((xy * depth, depth), dim=-1)

        confidence_hidden = backbone.conf_decoder(
            geometry_tokens, xpos=positions
        ).float()
        confidence = backbone.conf_head(
            [confidence_hidden[:, backbone.patch_start_idx :]], (height, width)
        ).reshape(batch, frame_count, height, width, -1)
    return {"local_points": local_points, "conf": confidence}


def _timed_network_forward(model, images, accelerator):
    """Run the three requested network stages with synchronized wall timing."""

    images_window = images.unsqueeze(0)
    batch, frame_count, _, height, width = images_window.shape
    device = images.device
    with torch.inference_mode(), accelerator.autocast():
        _synchronize(device)
        network_start = time.perf_counter()

        stage_start = network_start
        geometry_tokens, positions, encoder_features = model._extract_window_features(
            images_window
        )
        _synchronize(device)
        pi3_backbone_seconds = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        geometry = _decode_point_depth(
            model, geometry_tokens, positions, frame_count, height, width
        )
        _synchronize(device)
        pi3_point_depth_decode_seconds = time.perf_counter() - stage_start

        patch_start = int(model.backbone.patch_start_idx)
        patch_tokens = geometry_tokens.reshape(
            batch,
            frame_count,
            geometry_tokens.shape[1],
            geometry_tokens.shape[2],
        )[:, :, patch_start:]
        stage_start = time.perf_counter()
        output = model.match_pair(
            patch_tokens,
            encoder_features,
            images_window,
            reference_index=0,
        )
        _synchronize(device)
        glob3r_seconds = time.perf_counter() - stage_start
        network_seconds = time.perf_counter() - network_start

    timing = NetworkTiming(
        pi3_backbone_seconds=pi3_backbone_seconds,
        pi3_point_depth_decode_seconds=pi3_point_depth_decode_seconds,
        glob3r_seconds=glob3r_seconds,
        network_seconds=network_seconds,
    )
    return geometry, output, timing


def _path_signature(path_value) -> dict:
    path = Path(str(path_value)).expanduser()
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "size": None, "mtime_ns": None}
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _evaluation_fingerprint(
    cfg: DictConfig,
    task: SequenceTask,
    resolved_config: str,
    windows,
) -> str:
    payload = {
        "resolved_config": resolved_config,
        "dataset": task.dataset_name,
        "sequence": task.sequence_id,
        "sequence_index": task.sequence_index,
        "frame_count": task.frame_count,
        "windows": [list(window.positions) for window in windows],
        "backbone_checkpoint": _path_signature(cfg.model.backbone_checkpoint),
        "matching_checkpoint": _path_signature(cfg.model.matching_checkpoint),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _release_window_memory(model, device: torch.device) -> None:
    captured_features = getattr(model, "_captured_encoder_features", None)
    if hasattr(captured_features, "clear"):
        captured_features.clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _compute_window(
    cfg: DictConfig,
    accelerator: Accelerator,
    model,
    adapter,
    dataset,
    task: SequenceTask,
    window,
) -> WindowResult:
    loaded = adapter.load(
        dataset,
        task.sequence_index,
        int(cfg.evaluation.resolution_index),
        window.positions,
    )
    views = loaded.views
    frame_instances = loaded.frame_instances
    images = torch.stack([view["img"].float() for view in views]).to(
        accelerator.device
    )
    depths_gt = torch.stack(
        [torch.as_tensor(view["depthmap"], dtype=torch.float32) for view in views]
    ).to(accelerator.device)
    intrinsics = torch.stack(
        [
            torch.as_tensor(view["camera_intrinsics"], dtype=torch.float32)
            for view in views
        ]
    ).to(accelerator.device)
    poses = torch.stack(
        [torch.as_tensor(view["camera_pose"], dtype=torch.float32) for view in views]
    ).to(accelerator.device)

    geometry, output, timing = _timed_network_forward(model, images, accelerator)
    target_indices = [int(index) for index in output.target_indices]
    target_from_reference = (
        torch.linalg.inv(poses[target_indices]) @ poses[0]
    ).unsqueeze(0)
    supervision = build_ground_truth_warp(
        depths_gt[0].unsqueeze(0),
        depths_gt[target_indices].unsqueeze(0),
        intrinsics[0].unsqueeze(0),
        intrinsics[target_indices].unsqueeze(0),
        target_from_reference,
        depth_threshold=float(cfg.evaluation.depth_threshold),
    )
    predicted_warp, predicted_confidence = final_warp_and_confidence(
        output, tuple(images.shape[-2:])
    )
    _loss_map, _valid_map, target_metrics = pixelwise_warp_metrics(
        predicted_warp,
        supervision.warp[0],
        supervision.confidence[0],
        supervision.mask[0],
        epsilon=float(cfg.evaluation.charbonnier_epsilon),
        alpha=float(cfg.evaluation.charbonnier_alpha),
    )

    metric_rows = []
    for metric, target_index in zip(target_metrics, target_indices):
        metric_rows.append(
            {
                "dataset": task.dataset_name,
                "sequence": task.sequence_id,
                "window_index": window.index,
                "window_first_position": window.positions[0],
                "window_last_position": window.positions[-1],
                "reference_instance": frame_instances[0],
                "target_instance": frame_instances[target_index],
                "target_index": target_index,
                "valid_pixels": metric.valid_pixels,
                "loss_sum": metric.loss_sum,
                "mean_loss": metric.mean_loss,
            }
        )

    predicted_depth = geometry["local_points"].squeeze(0)[..., 2].float()
    depth_confidence = (
        geometry["conf"].squeeze(0).sigmoid().squeeze(-1).float()
    )
    overview = render_evaluation_matrix(
        images=images,
        frame_instances=frame_instances,
        reference_index=0,
        target_indices=target_indices,
        predicted_warp=predicted_warp,
        predicted_confidence=predicted_confidence,
        ground_truth_warp=supervision.warp[0],
        ground_truth_positive=supervision.confidence[0],
        training_mask=supervision.mask[0],
        depths=predicted_depth,
        depth_confidence=depth_confidence,
        confidence_threshold=float(cfg.visualization.confidence_threshold),
        cell_width=int(cfg.visualization.cell_width),
    )
    timing_row = {
        "dataset": task.dataset_name,
        "sequence": task.sequence_id,
        "window_index": window.index,
        "window_first_position": window.positions[0],
        "window_last_position": window.positions[-1],
        "frame_count": len(window.positions),
        "pi3_backbone_seconds": timing.pi3_backbone_seconds,
        "pi3_point_depth_decode_seconds": timing.pi3_point_depth_decode_seconds,
        "glob3r_seconds": timing.glob3r_seconds,
        "network_seconds": timing.network_seconds,
    }
    return WindowResult(
        image=overview,
        metric_rows=metric_rows,
        timing_row=timing_row,
        valid_pixels=sum(metric.valid_pixels for metric in target_metrics),
        loss_sum=sum(metric.loss_sum for metric in target_metrics),
    )


def _evaluate_sequence(
    cfg: DictConfig,
    accelerator: Accelerator,
    model,
    dataset,
    task: SequenceTask,
    resolved_config: str,
    adapter=None,
) -> tuple[float, int, str]:
    frame_step = int(dataset.frame_step)
    chunk_size = int(cfg.evaluation.chunk_size)
    overlap = chunk_size // 2
    stride = chunk_size - overlap
    windows = build_sequence_windows(task.frame_count, frame_step, chunk_size)
    sequence_dir = (
        Path(cfg.output_dir)
        / task.dataset_name
        / f"{task.sequence_index:06d}_{_safe_name(task.sequence_id)}"
    )
    adapter = adapter or create_window_adapter(dataset)
    metadata = {
        "dataset": task.dataset_name,
        "sequence": task.sequence_id,
        "sequence_index": task.sequence_index,
        "rank": accelerator.process_index,
        "world_size": accelerator.num_processes,
        "frame_count": task.frame_count,
        "sampled_frame_count": len(range(0, task.frame_count, frame_step)),
        "frame_step": frame_step,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "stride": stride,
        "backbone_checkpoint": str(cfg.model.backbone_checkpoint),
        "matching_checkpoint": str(cfg.model.matching_checkpoint),
    }
    fingerprint = _evaluation_fingerprint(cfg, task, resolved_config, windows)
    planned_positions = {
        window.index: window.positions for window in windows
    }

    def create_store() -> WindowArtifactStore:
        return WindowArtifactStore(
            sequence_dir=sequence_dir,
            fingerprint=fingerprint,
            metadata=metadata,
            planned_positions=planned_positions,
            metric_fields=CSV_FIELDS,
            timing_fields=TIMING_FIELDS,
            resolved_config=resolved_config,
        )

    def log_failure(
        event_status: str,
        stage: str,
        error=None,
        window_index: int | None = None,
    ) -> None:
        _append_rank_failure(
            Path(cfg.output_dir),
            accelerator.process_index,
            {
                "status": event_status,
                "stage": stage,
                "dataset": task.dataset_name,
                "sequence": task.sequence_id,
                "sequence_index": task.sequence_index,
                "frame_count": task.frame_count,
                "windows_total": len(windows),
                "window_index": window_index,
                "fingerprint": fingerprint,
                "error": error,
            },
        )

    store = None
    status = "complete"

    print(
        f"[rank {accelerator.process_index}] {task.dataset_name}/{task.sequence_id} "
        f"frames={task.frame_count} windows={len(windows)}",
        flush=True,
    )

    if not windows or len(windows[0].positions) < 2:
        status = "skipped"
        log_failure(
            status,
            "window_plan",
            error={
                "type": "InsufficientFrames",
                "message": "sequence has fewer than two sampled frames",
            },
        )
    else:
        has_commits = sequence_dir.is_dir() and any(
            sequence_dir.glob("chunk_*.json")
        )
        if has_commits:
            try:
                store = create_store()
                store.write_state("running")
            except Exception as exception:
                status = "failed"
                error = {
                    "type": type(exception).__name__,
                    "message": str(exception),
                    "traceback": traceback.format_exc(),
                }
                log_failure(status, "resume_initialization", error=error)
                print(
                    f"[rank {accelerator.process_index}] failed to resume "
                    f"{task.dataset_name}/{task.sequence_id}: {exception}",
                    flush=True,
                )
                return 0.0, 0, status

        for window in windows:
            if store is not None and store.is_completed(window.index):
                print(
                    f"  window={window.index} positions="
                    f"[{window.positions[0]},{window.positions[-1]}] resumed",
                    flush=True,
                )
                continue

            result = None
            stage = "window_compute"
            try:
                result = _compute_window(
                    cfg,
                    accelerator,
                    model,
                    adapter,
                    dataset,
                    task,
                    window,
                )
                if store is None:
                    stage = "artifact_initialization"
                    store = create_store()
                stage = "artifact_commit"
                store.commit(
                    window.index,
                    window.positions,
                    result.image,
                    result.metric_rows,
                    result.timing_row,
                )
                window_mean = (
                    result.loss_sum / result.valid_pixels
                    if result.valid_pixels
                    else float("nan")
                )
                print(
                    f"  window={window.index} positions="
                    f"[{window.positions[0]},{window.positions[-1]}] "
                    f"valid_pixels={result.valid_pixels} loss={window_mean:.6f} "
                    f"network_ms="
                    f"{1000.0 * result.timing_row['network_seconds']:.3f}",
                    flush=True,
                )
            except Exception as exception:
                status = "failed"
                error = {
                    "type": type(exception).__name__,
                    "message": str(exception),
                    "traceback": traceback.format_exc(),
                }
                if store is not None:
                    try:
                        store.write_state(status, error, failed_window=window.index)
                    except Exception as state_exception:
                        print(
                            f"[rank {accelerator.process_index}] could not update "
                            f"failure state for {task.dataset_name}/{task.sequence_id}: "
                            f"{state_exception}",
                            flush=True,
                        )
                log_failure(
                    status,
                    stage,
                    error=error,
                    window_index=window.index,
                )
                print(
                    f"[rank {accelerator.process_index}] failed "
                    f"{task.dataset_name}/{task.sequence_id} "
                    f"window={window.index}: {exception}",
                    flush=True,
                )
                break
            finally:
                if result is not None and hasattr(result.image, "close"):
                    result.image.close()
                result = None
                _release_window_memory(model, accelerator.device)
        else:
            try:
                store.write_state("complete")
            except Exception as exception:
                status = "failed"
                error = {
                    "type": type(exception).__name__,
                    "message": str(exception),
                    "traceback": traceback.format_exc(),
                }
                log_failure(status, "sequence_finalize", error=error)

    aggregate = (
        store.aggregates()
        if store is not None
        else {"loss_sum": 0.0, "valid_pixels": 0}
    )
    return aggregate["loss_sum"], aggregate["valid_pixels"], status


@hydra.main(version_base="1.2", config_path=".", config_name="valid")
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator(mixed_precision=str(cfg.inference.mixed_precision))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    dataset_names = list(cfg.datasets)
    datasets = {
        name: hydra.utils.instantiate(cfg.datasets[name])
        for name in dataset_names
    }
    adapters = {
        name: create_window_adapter(dataset)
        for name, dataset in datasets.items()
    }
    tasks = _build_tasks(
        dataset_names, datasets, int(cfg.evaluation.chunk_size)
    )
    local_tasks = shard_weighted_items(
        tasks,
        [task.window_count for task in tasks],
        accelerator.process_index,
        accelerator.num_processes,
    )
    resolved_config = OmegaConf.to_yaml(cfg, resolve=True)
    print(
        f"[rank {accelerator.process_index}] assigned sequences={len(local_tasks)} "
        f"windows={sum(task.window_count for task in local_tasks)}",
        flush=True,
    )

    model = None
    if local_tasks:
        model = load_glob3r_for_sfm(
            cfg.model.backbone_checkpoint,
            cfg.model.matching_checkpoint,
            device=accelerator.device,
        )

    local_statistics = {
        dataset_name: {
            "loss_sum": 0.0,
            "valid_pixels": 0,
            "complete": 0,
            "failed": 0,
            "skipped": 0,
        }
        for dataset_name in dataset_names
    }
    for task in local_tasks:
        loss_sum, valid_pixels, status = _evaluate_sequence(
            cfg,
            accelerator,
            model,
            datasets[task.dataset_name],
            task,
            resolved_config,
            adapters[task.dataset_name],
        )
        statistics = local_statistics[task.dataset_name]
        statistics["loss_sum"] += loss_sum
        statistics["valid_pixels"] += valid_pixels
        statistics[status] += 1

    for dataset_name, statistics in local_statistics.items():
        valid_pixels = statistics["valid_pixels"]
        mean_loss = (
            statistics["loss_sum"] / valid_pixels
            if valid_pixels
            else float("nan")
        )
        print(
            f"[rank {accelerator.process_index}] {dataset_name} "
            f"complete={statistics['complete']} failed={statistics['failed']} "
            f"skipped={statistics['skipped']} valid_pixels={valid_pixels} "
            f"mean_warp_loss={mean_loss:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
