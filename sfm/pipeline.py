"""Orchestrate the cached SfM frontend and the persistent graph backend."""

import gc
from dataclasses import dataclass
from pathlib import Path

import torch

from .dataset import ImageDataset
from .factor_graph import FactorGraph
from .features import build_query_features
from .frontend_cache import (
    load_frontend_cache,
    load_loop_cache,
    packet_to_cpu,
    packet_to_device,
    restore_frames,
    save_frontend_cache,
    save_loop_cache,
)
from .loop import LoopDetector, sliding_keyframe_pairs
from .loop_visualization import save_loop_match_images
from .match_visualization import save_match_images
from .model import load_models
from .reconstruction import reconstruct
from .tracker import FrameStore, WindowTracker
from .tracks_export import save_ba_tracks


def release_device_memory(device):
    """Release unreachable model state before the next GPU-heavy stage."""
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


@dataclass
class FrontendPackets:
    """Verified measurements produced by the sliding and loop frontends."""

    sliding: list
    loop: list

    def consume(self):
        """Transfer packet ownership to the graph stage and clear frontend lists."""
        packets = [*self.sliding, *self.loop]
        self.sliding.clear()
        self.loop.clear()
        return packets


class CachedFrontend:
    """Own frontend model lifetimes, cache invalidation, and diagnostics."""

    def __init__(self, config, dataset, frames):
        self.config = config
        self.dataset = dataset
        self.frames = frames
        self.cache_path = Path(config["data_h5"])
        self.loop_cache_path = Path(config["data_loop_h5"])

    def _tracker(self):
        geometry_model, tracks_model = load_models(self.config)
        features = build_query_features(self.config)
        return WindowTracker(
            geometry_model,
            tracks_model,
            features,
            self.frames,
            self.dataset,
            self.config,
        )

    def _build_sliding_packets(self):
        windows = list(self.dataset.windows(self.config["window_size"]))
        tracker = self._tracker()
        packets = []
        for index, window_ids in enumerate(windows):
            packet = tracker.track(window_ids)
            packet["kind"] = "sliding"
            packets.append(packet_to_cpu(packet))
            relative_scale = (
                1.0
                if index == 0
                else float(self.frames.dense[int(window_ids[-1])][3])
            )
            print(
                f"frontend {index + 1}/{len(windows)} "
                f"frames={window_ids[0]}..{window_ids[-1]} "
                f"keyframes={packet['keyframes'].tolist()} "
                f"relative_scale={relative_scale:.6g}"
            )
        return packets

    def _save_sliding_diagnostics(self, packets):
        """Render sliding-window diagnostics before committing the frontend cache."""
        match_dir = self.cache_path.parent / "match"
        match_images = save_match_images(
            match_dir, packets, self.frames, self.dataset
        )
        print(f"sliding match images={len(match_images)} path={match_dir}")

    def _sliding_packets(self):
        if self.cache_path.exists():
            cached = load_frontend_cache(self.cache_path)
            packets = cached["packets"]
            restore_frames(self.frames, cached["frames"])
            print(
                f"frontend cache hit packets={len(packets)} path={self.cache_path}"
            )
            if not any(packet.get("visualization") for packet in packets):
                print(
                    "frontend cache has no raw match diagnostics; delete data.h5 "
                    "and data_loop.h5 to regenerate them with the current frontend"
                )
            return packets, False

        print(f"frontend cache miss path={self.cache_path}")
        if self.loop_cache_path.exists():
            self.loop_cache_path.unlink()
            print(f"stale loop cache removed path={self.loop_cache_path}")
        packets = self._build_sliding_packets()
        self._save_sliding_diagnostics(packets)
        save_frontend_cache(self.cache_path, packets, self.frames)
        print(
            f"frontend cache saved packets={len(packets)} path={self.cache_path}"
        )
        release_device_memory(self.config["device"])
        return packets, True

    def _detect_loop_windows(self, sliding_packets):
        local_pairs = sliding_keyframe_pairs(sliding_packets, self.frames.keyframes)
        detector = LoopDetector(
            self.config["dino_salad_checkpoint"],
            self.config["device"],
            self.config["image_size"],
            self.config["loop_batch_size"],
        )
        windows = detector.detect(
            self.frames,
            self.config["loop_similarity_threshold"],
            local_pairs,
        )
        del detector
        release_device_memory(self.config["device"])
        return windows, local_pairs

    def _track_loop_windows(self, windows):
        if not windows:
            return []
        tracker = self._tracker()
        packets = []
        for index, frame_ids in enumerate(windows):
            packet = tracker.track(
                frame_ids,
                references=[frame_ids[0]],
                minimum_edge_views=2,
            )
            if not packet["parts"]:
                continue
            packet["kind"] = "loop"
            packets.append(packet_to_cpu(packet))
            print(
                f"loop {index + 1}/{len(windows)} "
                f"reference={frame_ids[0]} retrieved={len(frame_ids) - 1} "
                f"edges={len(packet['edges'])}"
            )
        return packets

    def _save_loop_diagnostics(self, packets):
        """Render loop diagnostics before committing the loop cache."""
        loop_dir = self.cache_path.parent / "loop"
        loop_images = save_loop_match_images(loop_dir, packets, self.frames)
        print(f"loop match images={len(loop_images)} path={loop_dir}")

    def _loop_packets(self, sliding_packets, rebuilt_frontend):
        if not self.config["loop"]:
            return []
        if self.loop_cache_path.exists() and not rebuilt_frontend:
            packets = load_loop_cache(self.loop_cache_path)
            print(
                f"RGB SALAD loop cache hit packets={len(packets)} "
                f"path={self.loop_cache_path}"
            )
            return packets

        print(f"RGB SALAD loop cache miss path={self.loop_cache_path}")
        windows, local_pairs = self._detect_loop_windows(sliding_packets)
        packets = self._track_loop_windows(windows)
        print(
            f"loop detection keyframes={len(self.frames.keyframes)} "
            f"local_pairs={len(local_pairs) // 2} windows={len(windows)} "
            f"accepted={len(packets)}"
        )
        self._save_loop_diagnostics(packets)
        save_loop_cache(self.loop_cache_path, packets)
        print(
            f"RGB SALAD loop cache saved packets={len(packets)} "
            f"path={self.loop_cache_path}"
        )
        release_device_memory(self.config["device"])
        return packets

    def run(self):
        sliding_packets, rebuilt_frontend = self._sliding_packets()
        loop_packets = self._loop_packets(sliding_packets, rebuilt_frontend)
        return FrontendPackets(sliding_packets, loop_packets)


class GraphBackend:
    """Consume verified packets and own global graph initialization and export."""

    def __init__(self, config, dataset, frames, output_dir):
        self.config = config
        self.dataset = dataset
        self.frames = frames
        self.output_dir = Path(output_dir)

    def _build_graph(self, packets):
        graph = FactorGraph(self.frames)
        cached_packets = packets.consume()
        if not cached_packets:
            raise RuntimeError("frontend produced no factor packets")
        for index, cached_packet in enumerate(cached_packets):
            packet = packet_to_device(cached_packet, self.config["device"])
            graph.add_factors(packet)
            print(
                f"factors {index + 1}/{len(cached_packets)} "
                f"kind={packet.get('kind', 'sliding')} "
                f"frames={len(packet['frame_ids'])} "
                f"keyframes={packet['keyframes'].tolist()}"
            )
        cached_packets.clear()
        del packet, cached_packet
        graph.initialize_global()
        release_device_memory(self.config["device"])
        return graph

    def _export_tracks(self, view):
        native_K = torch.as_tensor(
            self.dataset.native_K,
            device=view["K"].device,
            dtype=view["K"].dtype,
        ).expand(view["K"].shape[0], -1, -1)
        processed_to_native = native_K @ torch.linalg.inv(view["K"])
        tracks_output = Path(
            self.config.get("tracks_output", self.output_dir / "tracks")
        )
        summary = save_ba_tracks(
            tracks_output,
            view,
            self.frames,
            pixel_transforms=processed_to_native,
        )
        print(
            f"tracks cameras={summary['camera_count']} "
            f"points={summary['point_count']} "
            f"observations={summary['observation_count']} path={summary['path']}"
        )

    def _optimize(self, graph, view):
        result = graph.optimize(
            view,
            [0],
            self.config["global_iterations"],
            backend=self.config["global_ba_backend"],
        )
        print(
            f"global backend={result['ba_backend']} "
            f"cameras={view['poses'].shape[0]} points={view['points'].shape[0]} "
            f"observations={view['ii'].numel()} "
            f"valid={int(result['valid_observations'])}/{view['ii'].numel()} "
            f"loss={float(result['loss']):.6g} "
            f"loss_per_pixel={float(result['loss_per_pixel']):.6g}"
        )
        return result

    def _export_reconstruction(self, graph):
        from .export import save_camera_wireframes, save_ply

        result = reconstruct(graph, self.frames)
        save_ply(
            self.output_dir / "sparse_tracks.ply",
            result["sparse_points"],
            scalar_fields={
                "frame_id": result["sparse_frame_ids"],
                "obs_cnt": result["sparse_obs_cnt"],
            },
        )
        save_ply(
            self.output_dir / "dense.ply",
            result["dense_points"],
            result["dense_colors"],
            scalar_fields={"frame_id": result["dense_frame_ids"]},
        )
        camera_frame_ids = sorted(graph.poses)
        (self.output_dir / "camera_poses.ply").unlink(missing_ok=True)
        save_camera_wireframes(
            self.output_dir / "camera_poses.obj",
            torch.stack([graph.poses[frame_id] for frame_id in camera_frame_ids]),
            result["sparse_points"],
            camera_frame_ids,
        )

    def run(self, packets):
        graph = self._build_graph(packets)
        view = graph.full_view()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._export_tracks(view)
        result = self._optimize(graph, view)
        self._export_reconstruction(graph)
        return result


class SfMPipeline:
    """Composition root for one complete full-sequence SfM run."""

    def __init__(self, config):
        self.config = config

    def run(self):
        torch.manual_seed(int(self.config.get("random_seed", 0)))
        dataset = ImageDataset(
            self.config["images"],
            self.config["image_size"],
            self.config.get("calibration"),
            self.config.get("mask"),
        )
        frames = FrameStore(dataset)
        frontend = CachedFrontend(self.config, dataset, frames)
        packets = frontend.run()
        backend = GraphBackend(
            self.config, dataset, frames, Path(self.config["data_h5"]).parent
        )
        return backend.run(packets)


__all__ = [
    "CachedFrontend",
    "FrontendPackets",
    "GraphBackend",
    "SfMPipeline",
    "release_device_memory",
]
