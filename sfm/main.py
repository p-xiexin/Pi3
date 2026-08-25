"""Run full-sequence SfM with selectable Glob3R or VGGSfM tracks."""

import argparse
import gc
from pathlib import Path

import torch

from .config import load_config
from .dataset import ImageDataset
from .export import save_ply
from .factor_graph import FactorGraph, select_local_fixed_ids
from .features import (
    QUERY_BUCKET_GRID,
    QUERY_CANDIDATE_MULTIPLIER,
    SIFTFeatures,
)
from .model import load_models
from .reconstruction import reconstruct
from .tracks_export import save_ba_tracks
from .tracker import (
    KEYFRAME_MATCH_COVERAGE_RATIO,
    FrameStore,
    KEYFRAME_PROJECTION_RATIO,
    PI3_VALID_CONFIDENCE,
    WindowTracker,
)


def run(config):
    """Run factor construction, local BA, global initialization, and global BA."""
    dataset = ImageDataset(
        config["images"], config["image_size"], config.get("calibration"), config.get("mask")
    )
    frames = FrameStore(dataset)
    geometry_model, tracks_model = load_models(config)
    features = SIFTFeatures(config["points_per_keyframe"], config["device"])
    tracker = WindowTracker(geometry_model, tracks_model, features, frames, dataset, config)
    graph = FactorGraph(frames)
    windows = list(dataset.windows(config["window_size"]))
    for index, window_ids in enumerate(windows):
        # Measurements enter the persistent graph before the shared optimizer
        # creates a local view over this window.
        packet = tracker.track(window_ids)
        known = graph.add_factors(packet)
        # One overlapping camera removes the local gauge. Every other overlap
        # camera remains active and can absorb the new window measurements.
        fixed_ids = select_local_fixed_ids(index, window_ids, known)
        view = graph.local_view(window_ids)
        result = graph.optimize(view, fixed_ids, config["local_iterations"])
        print(
            f"local {index + 1}/{len(windows)} frames={window_ids[0]}..{window_ids[-1]} "
            f"keyframes={packet['keyframes'].tolist()} "
            f"fixed={len(fixed_ids)} observations={view['ii'].numel()} "
            f"valid={int(result['valid_observations'])}/{view['ii'].numel()} "
            f"loss={float(result['loss']):.6g} "
            f"loss_per_pixel={float(result['loss_per_pixel']):.6g}"
        )
    # Window inference is complete. Release both model frontends before the
    # all-camera graph and Schur workspaces are materialized for global BA.
    del packet, view, result
    del tracker, features, tracks_model, geometry_model
    gc.collect()
    if torch.device(config["device"]).type == "cuda":
        torch.cuda.empty_cache()
    # Global initialization consumes all relative-pose factors accumulated by
    # the sliding windows. Full BA then uses every eligible observation.
    graph.initialize_global()
    view = graph.full_view()
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    native_K = torch.as_tensor(
        dataset.native_K, device=view["K"].device, dtype=view["K"].dtype
    ).expand(view["K"].shape[0], -1, -1)
    processed_to_native = native_K @ torch.linalg.inv(view["K"])
    tracks_output = Path(config.get("tracks_output", output.parent / "tracks"))
    tracks_summary = save_ba_tracks(
        tracks_output, view, frames, pixel_transforms=processed_to_native
    )
    print(
        f"tracks cameras={tracks_summary['camera_count']} "
        f"points={tracks_summary['point_count']} "
        f"observations={tracks_summary['observation_count']} "
        f"path={tracks_summary['path']}"
    )
    # The external BA consumes the exported global graph input. This local
    # optimizer remains a numerical and geometric validation of that input.
    result = graph.optimize(
        view, [0], config["global_iterations"], backend=config["global_ba_backend"]
    )
    print(
        f"global backend={result['ba_backend']} "
        f"cameras={view['poses'].shape[0]} points={view['points'].shape[0]} "
        f"observations={view['ii'].numel()} "
        f"valid={int(result['valid_observations'])}/{view['ii'].numel()} "
        f"loss={float(result['loss']):.6g} "
        f"loss_per_pixel={float(result['loss_per_pixel']):.6g}"
    )
    reconstruction = reconstruct(graph, frames)
    frame_ids = sorted(graph.poses)
    poses = torch.stack([graph.poses[frame_id] for frame_id in frame_ids])
    edge_source = torch.tensor([edge[0] for edge in graph.edges])
    edge_target = torch.tensor([edge[1] for edge in graph.edges])
    edge_poses = torch.stack([edge[2] for edge in graph.edges]).cpu()
    edge_weight = torch.tensor([edge[3] for edge in graph.edges])
    torch.save(
        {
            "image_paths": [str(path) for path in dataset.paths],
            "frame_ids": torch.tensor(frame_ids),
            "world_to_camera": poses.cpu(),
            "camera_to_world": torch.linalg.inv(poses).cpu(),
            "intrinsics": torch.stack([graph.intrinsics[frame_id] for frame_id in frame_ids]).cpu(),
            "keyframes": torch.tensor(sorted(frames.keyframes)),
            "tracks_model": config["tracks_model"],
            "query_features": "lightglue_sift_bucketed",
            "image_size": torch.tensor(dataset.image_size),
            "window_size": int(config["window_size"]),
            "points_per_keyframe": int(config["points_per_keyframe"]),
            "manual_valid_mask": (
                None if dataset.valid_mask is None else dataset.valid_mask.cpu()
            ),
            "pi3_valid_confidence": PI3_VALID_CONFIDENCE,
            "keyframe_projection_ratio": KEYFRAME_PROJECTION_RATIO,
            "keyframe_match_coverage_ratio": KEYFRAME_MATCH_COVERAGE_RATIO,
            "query_bucket_grid": torch.tensor(QUERY_BUCKET_GRID),
            "query_candidate_multiplier": QUERY_CANDIDATE_MULTIPLIER,
            "pose_graph_source": edge_source,
            "pose_graph_target": edge_target,
            "pose_graph_relative_poses": edge_poses,
            "pose_graph_weight": edge_weight,
            "observation_frames": view["frame_ids"][view["ii"]].cpu(),
            "observation_points": view["point_ids"][view["jj"]].cpu(),
            "observation_uv": view["uv"].cpu(),
            "observation_weight": view["weight"].cpu(),
            "ba_backend": result["ba_backend"],
            "distortion": result.get("distortion", view["distortion"]).cpu(),
            "loss": result["loss"].cpu(),
            "loss_per_pixel": result["loss_per_pixel"].cpu(),
            "valid_observations": result["valid_observations"].cpu(),
            **{key: value.cpu() for key, value in reconstruction.items()},
        },
        output,
    )
    save_ply(
        output.parent / "sparse_tracks.ply",
        reconstruction["sparse_points"],
        # A sparse point is keyed by its reference frame and SIFT feature ID.
        scalar_fields={
            "frame_id": reconstruction["sparse_frame_ids"],
            "obs_cnt": reconstruction["sparse_obs_cnt"],
        },
    )
    save_ply(
        output.parent / "dense.ply", reconstruction["dense_points"],
        reconstruction["dense_colors"],
        scalar_fields={"frame_id": reconstruction["dense_frame_ids"]},
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sfm/config.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
