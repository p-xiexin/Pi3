"""One persistent factor graph shared by window and full-sequence optimization."""

import torch

from .geometric_verification import triangulate_tracks
from .geometry import average_rotations, camera_centers, maximum_spanning_tree
from .optimizer import optimize_view_two_rounds


def select_local_fixed_ids(window_index, window_ids, known):
    """Select one camera to remove the local BA gauge."""
    if window_index == 0:
        return [int(window_ids[0])]
    if not known:
        raise RuntimeError(
            f"window {window_index} has no camera shared with the factor graph"
        )
    return [int(known[0])]


class FactorGraph:
    """Own all cameras, landmarks, observations, and relative-pose constraints."""

    def __init__(self, frames):
        self.frames = frames
        self.poses = {}
        self.intrinsics = {}
        self.point_lookup = {}
        self.point_count = 0
        self.point_references = None
        self.point_anchors = None
        self.point_positions = None
        self.point_initialized = None
        self.observations = []
        self.observation_ids = []
        self.next_observation_id = 0
        self.inactive_observation_ids = set()
        self.anchor_observed = set()
        self.edges = []
        self.edge_lookup = {}
        self.pi3_edge_keys = set()
        self.first_sliding_root_frame_id = None
        self.first_sliding_midpoint_frame_id = None

    @property
    def frame_ids(self):
        """Return frame IDs already registered in the persistent graph."""
        return set(self.poses)

    def _reserve_points(self, required, example):
        """Grow landmark storage without changing preassigned stable IDs."""
        if self.point_anchors is not None and self.point_anchors.device != example.device:
            self.point_references = self.point_references.to(example.device)
            self.point_anchors = self.point_anchors.to(example.device)
            self.point_positions = self.point_positions.to(example.device)
            self.point_initialized = self.point_initialized.to(example.device)
        capacity = 0 if self.point_anchors is None else self.point_anchors.shape[0]
        if required <= capacity:
            return
        capacity = max(1024, required, capacity * 2)
        references = torch.empty(capacity, device=example.device, dtype=torch.long)
        anchors = torch.empty(capacity, 3, device=example.device, dtype=example.dtype)
        positions = torch.empty_like(anchors)
        initialized = torch.zeros(capacity, device=example.device, dtype=torch.bool)
        if self.point_count:
            references[:self.point_count] = self.point_references[:self.point_count]
            anchors[:self.point_count] = self.point_anchors[:self.point_count]
            positions[:self.point_count] = self.point_positions[:self.point_count]
            initialized[:self.point_count] = self.point_initialized[:self.point_count]
        self.point_references, self.point_anchors = references, anchors
        self.point_positions, self.point_initialized = positions, initialized

    def _sync_track_registry(self, example):
        """Materialize stable IDs assigned by FrameStore before packet ingestion."""
        if self.frames is None:
            return
        if not hasattr(self.frames, "track_ids") or not hasattr(
            self.frames, "next_track_id"
        ):
            raise RuntimeError("FrameStore has no stable track ID registry")
        required = int(self.frames.next_track_id)
        self._reserve_points(required, example)
        if required == self.point_count and len(self.point_lookup) == required:
            return
        if required < self.point_count:
            raise RuntimeError("stable track ID registry cannot shrink")
        seen_ids = set()
        id_lookup = {point_id: key for key, point_id in self.point_lookup.items()}
        for frame_id in sorted(self.frames.track_ids):
            if frame_id not in self.frames.anchors:
                raise RuntimeError(
                    f"stable tracks for keyframe {frame_id} have no cached anchors"
                )
            keys, _, anchors, _ = self.frames.anchors[frame_id]
            track_ids = self.frames.track_ids[frame_id]
            if keys.ndim != 1 or track_ids.ndim != 1:
                raise RuntimeError("stable track keys and IDs must be one-dimensional")
            if keys.numel() != track_ids.numel() or anchors.shape[0] != keys.numel():
                raise RuntimeError(
                    f"keyframe {frame_id} stable track registry is misaligned"
                )
            for feature_id, track_id in zip(keys.tolist(), track_ids.tolist()):
                track_id = int(track_id)
                key = (int(frame_id), int(feature_id))
                if track_id < 0 or track_id >= required:
                    raise RuntimeError(
                        f"keyframe {frame_id} has invalid stable track ID {track_id}"
                    )
                if track_id in seen_ids:
                    raise RuntimeError(f"stable track ID {track_id} is assigned twice")
                seen_ids.add(track_id)
                known_key = id_lookup.get(track_id)
                if known_key is not None and known_key != key:
                    raise RuntimeError(
                        f"stable track ID {track_id} changed identity"
                    )
                known_id = self.point_lookup.get(key)
                if known_id is not None and known_id != track_id:
                    raise RuntimeError(f"stable track {key} changed ID")
                self.point_lookup[key] = track_id
                id_lookup[track_id] = key
            device_ids = track_ids.to(device=example.device, dtype=torch.long)
            self.point_references[device_ids] = int(frame_id)
            self.point_anchors[device_ids] = anchors.to(
                device=example.device, dtype=example.dtype
            )
        expected_ids = set(range(required))
        if seen_ids != expected_ids:
            missing = sorted(expected_ids.difference(seen_ids))
            raise RuntimeError(
                f"stable track ID registry is not contiguous, missing={missing[:8]}"
            )
        self.point_count = required

    def add_factors(self, packet):
        """Merge one window packet before any local or global optimization."""
        self._sync_track_registry(packet["poses"])
        packet_kind = packet.get("kind", "sliding")
        resolved_parts = []
        for part in packet["parts"]:
            if "track_ids" not in part:
                raise RuntimeError("factor packet has no stable track IDs")
            if part["track_ids"].shape != part["keys"].shape:
                raise RuntimeError("factor packet track IDs and keys are misaligned")
            point_ids = part["track_ids"].to(
                device=packet["frame_ids"].device, dtype=torch.long
            )
            observation_indices = part.get("obs_points")
            if not isinstance(observation_indices, torch.Tensor) or observation_indices.ndim != 1:
                raise RuntimeError("factor packet observation indices must be a 1D tensor")
            if bool(
                (
                    (observation_indices < 0)
                    | (observation_indices >= point_ids.numel())
                ).any()
            ):
                raise RuntimeError(
                    "factor packet observation references an invalid stable track index"
                )
            for feature_id, point_id in zip(
                part["keys"].tolist(), point_ids.tolist()
            ):
                key = (int(part["reference"]), int(feature_id))
                registered = self.point_lookup.get(key)
                if registered is None:
                    raise RuntimeError(
                        f"{packet_kind} packet references unknown stable track {key}"
                    )
                if int(point_id) != registered:
                    raise RuntimeError(
                        f"{packet_kind} packet changed stable track {key} "
                        f"from {registered} to {int(point_id)}"
                    )
            resolved_parts.append((part, point_ids))
        frame_ids = packet["frame_ids"].tolist()
        if (
            packet_kind == "sliding"
            and self.first_sliding_midpoint_frame_id is None
            and len(frame_ids) > 1
        ):
            self.first_sliding_root_frame_id = int(frame_ids[0])
            self.first_sliding_midpoint_frame_id = int(
                frame_ids[len(frame_ids) // 2]
            )
        known = [frame_id for frame_id in frame_ids if frame_id in self.poses]
        if known:
            anchor = known[0]
            anchor_local = frame_ids.index(anchor)
            anchor_pose = packet["poses"][anchor_local]
            aligned = (
                packet["poses"]
                @ torch.linalg.inv(anchor_pose)[None]
                @ self.poses[anchor][None]
            )
        else:
            aligned = packet["poses"]
        for local, frame_id in enumerate(frame_ids):
            if frame_id not in self.poses:
                self.poses[frame_id] = aligned[local].clone()
            self.intrinsics[frame_id] = packet["K"][local].clone()
        for part, point_ids in resolved_parts:
            observation_points = point_ids[part["obs_points"]]
            keep = []
            for frame_id, point_id in zip(part["obs_frames"].tolist(), observation_points.tolist()):
                duplicate_anchor = (
                    frame_id == part["reference"]
                    and point_id in self.anchor_observed
                )
                keep.append(not duplicate_anchor)
                if frame_id == part["reference"]:
                    self.anchor_observed.add(point_id)
            keep = torch.tensor(keep, device=point_ids.device, dtype=torch.bool)
            if keep.any():
                self.observations.append((
                    int(part["reference"]), part["obs_frames"][keep].clone(),
                    observation_points[keep].clone(), part["obs_uv"][keep].clone(),
                    part["obs_weights"][keep].clone(),
                ))
                count = int(keep.sum())
                self.observation_ids.append(
                    torch.arange(
                        self.next_observation_id,
                        self.next_observation_id + count,
                        device=point_ids.device,
                        dtype=torch.long,
                    )
                )
                self.next_observation_id += count
        pi3_T_WCs = None
        pi3_T_CWs = None
        pi3_local = None
        if packet_kind == "sliding" and "pi3_T_WCs" in packet:
            pi3_T_WCs = packet["pi3_T_WCs"].clone()
            metric_scale = torch.as_tensor(
                packet["metric_scale"],
                device=pi3_T_WCs.device,
                dtype=pi3_T_WCs.dtype,
            )
            pi3_T_WCs[:, :3, 3] *= metric_scale
            pi3_T_CWs = torch.linalg.inv(pi3_T_WCs)
            pi3_local = {
                int(frame_id): index for index, frame_id in enumerate(frame_ids)
            }
        for source, target, relative, weight in packet["edges"]:
            pi3_relative = pi3_T_WCs is not None
            if pi3_relative:
                relative = (
                    pi3_T_CWs[pi3_local[int(target)]]
                    @ pi3_T_WCs[pi3_local[int(source)]]
                )
            if source > target:
                source, target, relative = target, source, torch.linalg.inv(relative)
            source, target, weight = int(source), int(target), float(weight)
            key = (source, target)
            edge = (source, target, relative.clone(), weight)
            # Sliding packets are consumed chronologically.  The first verified
            # measurement therefore owns an overlapping frame pair; later
            # windows extend the graph without rewriting its established part.
            if key not in self.edge_lookup:
                self.edge_lookup[key] = len(self.edges)
                self.edges.append(edge)
                if pi3_relative:
                    self.pi3_edge_keys.add(key)
        return known

    def _view(self, frame_ids, scope):
        """Materialize a compact optimizer view over selected cameras and landmarks."""
        frame_ids = list(dict.fromkeys(map(int, frame_ids)))
        local_frame = {frame_id: i for i, frame_id in enumerate(frame_ids)}
        device = next(iter(self.poses.values())).device
        selected_ids = torch.tensor(frame_ids, device=device, dtype=torch.long)
        chunks = []
        for chunk_index, (
            reference, observation_frames, observation_points, uv, weight
        ) in enumerate(self.observations):
            if reference not in local_frame:
                continue
            ids = self.observation_ids[chunk_index]
            active = torch.tensor(
                [int(obs_id) not in self.inactive_observation_ids for obs_id in ids.tolist()],
                device=ids.device,
                dtype=torch.bool,
            )
            mask = active & torch.isin(observation_frames, selected_ids)
            if mask.any():
                chunks.append(
                    (
                        observation_frames[mask],
                        observation_points[mask],
                        uv[mask],
                        weight[mask],
                        ids[mask],
                    )
                )
        if not chunks:
            raise RuntimeError(f"{scope} graph view has no observations")
        observation_frames = torch.cat([chunk[0] for chunk in chunks])
        observation_points = torch.cat([chunk[1] for chunk in chunks])
        uv = torch.cat([chunk[2] for chunk in chunks])
        weight = torch.cat([chunk[3] for chunk in chunks])
        observation_ids = torch.cat([chunk[4] for chunk in chunks])
        # BA receives only landmarks constrained by at least three selected
        # camera observations. The persistent graph keeps the remaining data.
        counts = torch.bincount(observation_points, minlength=self.point_count)
        keep_points = (counts >= 3) & self.point_initialized[:self.point_count]
        mask = keep_points[observation_points]
        observation_frames, observation_points = observation_frames[mask], observation_points[mask]
        uv, weight, observation_ids = uv[mask], weight[mask], observation_ids[mask]
        point_ids = torch.nonzero(keep_points, as_tuple=False).squeeze(-1)
        if not point_ids.numel():
            raise RuntimeError(f"{scope} graph view has no multi-view tracks")
        camera_map = torch.full((max(frame_ids) + 1,), -1, device=device, dtype=torch.long)
        camera_map[selected_ids] = torch.arange(len(frame_ids), device=device)
        point_map = torch.full((self.point_count,), -1, device=device, dtype=torch.long)
        point_map[point_ids] = torch.arange(point_ids.numel(), device=device)
        poses = torch.stack([self.poses[frame_id] for frame_id in frame_ids])
        references = camera_map[self.point_references[point_ids]]
        anchors = self.point_anchors[point_ids]
        points = self.point_positions[point_ids].clone()
        return {
            "scope": scope,
            "frame_ids": torch.tensor(frame_ids, device=device, dtype=torch.long),
            "point_ids": point_ids,
            "poses": poses,
            "points": points,
            "references": references,
            "anchors": anchors,
            "K": torch.stack([self.intrinsics[frame_id] for frame_id in frame_ids]),
            "distortion": torch.zeros(len(frame_ids), 4, device=device, dtype=poses.dtype),
            "ii": camera_map[observation_frames],
            "jj": point_map[observation_points],
            "uv": uv,
            "weight": weight,
            "observation_ids": observation_ids,
        }

    def local_view(self, frame_ids):
        """Build a window-sized view of the persistent factor graph."""
        return self._view(frame_ids, "local")

    def full_view(self):
        """Build the full-sequence view consumed by global BA."""
        view = self._view(sorted(self.poses), "global")
        if self.first_sliding_midpoint_frame_id is not None:
            view["scale_gauge_root_frame_id"] = self.first_sliding_root_frame_id
            view["scale_gauge_frame_id"] = self.first_sliding_midpoint_frame_id
        return view

    def initialize_global(self):
        """Initialize cameras from verified edges and landmarks by multiview DLT."""
        frame_ids = sorted(self.poses)
        local = {frame_id: i for i, frame_id in enumerate(frame_ids)}
        edges = [
            (local[source], local[target], relative, weight)
            for source, target, relative, weight in self.edges
            if source in local and target in local
        ]
        pi3_edges = [
            (local[source], local[target], relative, weight)
            for source, target, relative, weight in self.edges
            if source in local
            and target in local
            and (source, target) in self.pi3_edge_keys
        ]
        if len(frame_ids) > 1 and not edges:
            raise RuntimeError("global pose graph has no edges")
        device = self.poses[frame_ids[0]].device
        if len(frame_ids) == 1:
            poses = torch.stack([self.poses[frame_ids[0]]])
        else:
            mst_edges = pi3_edges or edges
            print(
                f"global pose initialization="
                f"{'pi3_metric' if pi3_edges else 'essential'} "
                f"mst_edges={len(mst_edges)} rotation_edges={len(edges)}"
            )
            mst_source = torch.tensor(
                [edge[0] for edge in mst_edges], device=device
            )
            mst_target = torch.tensor(
                [edge[1] for edge in mst_edges], device=device
            )
            mst_relative = torch.stack([edge[2] for edge in mst_edges])
            mst_weight = torch.tensor(
                [edge[3] for edge in mst_edges],
                device=device,
                dtype=mst_relative.dtype,
            )
            try:
                poses = maximum_spanning_tree(
                    len(frame_ids),
                    mst_source,
                    mst_target,
                    mst_relative,
                    mst_weight,
                )
            except RuntimeError as error:
                if pi3_edges:
                    raise RuntimeError(
                        "metric-scaled Pi3 sliding pose graph is disconnected"
                    ) from error
                raise
            source = torch.tensor([edge[0] for edge in edges], device=device)
            target = torch.tensor([edge[1] for edge in edges], device=device)
            relative = torch.stack([edge[2] for edge in edges])
            weight = torch.tensor(
                [edge[3] for edge in edges], device=device, dtype=relative.dtype
            )
            rotations = average_rotations(poses, source, target, relative, weight)
            centers = camera_centers(poses)
            poses[:, :3, :3] = rotations
            poses[:, :3, 3] = -torch.einsum("sij,sj->si", rotations, centers)
            for frame_id, pose in zip(frame_ids, poses):
                self.poses[frame_id] = pose
        self._triangulate_global(frame_ids, poses)

    def _triangulate_global(self, frame_ids, poses):
        """Triangulate every three-view stable track and retire invalid observations."""
        device = poses.device
        local_frame = {frame_id: index for index, frame_id in enumerate(frame_ids)}
        frame_parts, point_parts, uv_parts, id_parts = [], [], [], []
        for chunk_index, (_, obs_frames, obs_points, uv, _) in enumerate(
            self.observations
        ):
            ids = self.observation_ids[chunk_index]
            active = torch.tensor(
                [int(obs_id) not in self.inactive_observation_ids for obs_id in ids.tolist()],
                device=ids.device,
                dtype=torch.bool,
            )
            if active.any():
                frame_parts.append(obs_frames[active])
                point_parts.append(obs_points[active])
                uv_parts.append(uv[active])
                id_parts.append(ids[active])
        if not frame_parts:
            raise RuntimeError("global graph has no active observations to triangulate")
        observation_frames = torch.cat(frame_parts)
        observation_points = torch.cat(point_parts)
        uv = torch.cat(uv_parts)
        observation_ids = torch.cat(id_parts)
        counts = torch.bincount(observation_points, minlength=self.point_count)
        candidate_ids = torch.nonzero(counts >= 3, as_tuple=False).squeeze(-1)
        if not candidate_ids.numel():
            raise RuntimeError("global graph has no three-view tracks to triangulate")
        point_map = torch.full(
            (self.point_count,), -1, device=device, dtype=torch.long
        )
        point_map[candidate_ids] = torch.arange(candidate_ids.numel(), device=device)
        candidate_observations = point_map[observation_points] >= 0
        observation_frames = observation_frames[candidate_observations]
        observation_points = observation_points[candidate_observations]
        uv = uv[candidate_observations]
        observation_ids = observation_ids[candidate_observations]
        ii = torch.tensor(
            [local_frame[int(frame_id)] for frame_id in observation_frames.tolist()],
            device=device,
            dtype=torch.long,
        )
        jj = point_map[observation_points]
        K = torch.stack([self.intrinsics[frame_id] for frame_id in frame_ids])
        points, observation_mask, point_mask, angles = triangulate_tracks(
            poses, K, ii, jj, uv, candidate_ids.numel()
        )
        reference_local = torch.tensor(
            [
                local_frame[int(reference)]
                for reference in self.point_references[candidate_ids].tolist()
            ],
            device=device,
            dtype=torch.long,
        )
        reference_observation = observation_mask & (ii == reference_local[jj])
        has_reference = torch.bincount(
            jj[reference_observation], minlength=candidate_ids.numel()
        ) > 0
        point_mask &= has_reference
        self.point_initialized[:self.point_count] = False
        valid_ids = candidate_ids[point_mask]
        self.point_positions[valid_ids] = points[point_mask]
        self.point_initialized[valid_ids] = True
        accepted_observations = observation_mask & point_mask[jj]
        self.inactive_observation_ids.update(
            int(obs_id)
            for obs_id in observation_ids[~accepted_observations].tolist()
        )
        self.triangulation_angles = torch.full(
            (self.point_count,), float("nan"), device=device, dtype=points.dtype
        )
        self.triangulation_angles[candidate_ids] = angles

    def optimize(self, view, fixed_ids, iterations, backend="native"):
        """Run the shared BA backend and commit its state to the persistent graph."""
        local = {int(frame_id): i for i, frame_id in enumerate(view["frame_ids"].tolist())}
        fixed = [local[int(frame_id)] for frame_id in fixed_ids if int(frame_id) in local]
        if not fixed:
            raise ValueError("optimization requires at least one fixed camera")
        if backend == "native":
            scale_gauge_camera = None
            if "scale_gauge_frame_id" in view:
                scale_gauge_root = local[int(view["scale_gauge_root_frame_id"])]
                if scale_gauge_root not in fixed:
                    raise ValueError(
                        "global fixed cameras must include the first sliding frame"
                    )
                scale_gauge_camera = local[int(view["scale_gauge_frame_id"])]
            result = optimize_view_two_rounds(
                view,
                fixed,
                bearing_iterations=15,
                first_iterations=int(iterations),
                second_iterations=10,
                scale_gauge_camera=scale_gauge_camera,
            )
            result["ba_backend"] = "native"
        elif backend == "colmap":
            from .colmap_optimizer import optimize_view_colmap_two_rounds
            result = optimize_view_colmap_two_rounds(
                view,
                fixed,
                bearing_iterations=15,
                first_iterations=int(iterations),
                second_iterations=10,
            )
        else:
            raise ValueError(f"unsupported BA backend {backend}")
        for frame_id, pose in zip(view["frame_ids"].tolist(), result["poses"]):
            self.poses[int(frame_id)] = pose
        if "K" in result:
            for frame_id, K in zip(view["frame_ids"].tolist(), result["K"]):
                self.intrinsics[int(frame_id)] = K
        point_inliers = result.get(
            "point_inliers",
            torch.ones(view["point_ids"].shape, device=view["point_ids"].device, dtype=torch.bool),
        )
        self.point_positions[view["point_ids"]] = result["points"]
        self.point_initialized[view["point_ids"]] = point_inliers
        observation_inliers = result.get("observation_inliers")
        if observation_inliers is not None and "observation_ids" in view:
            self.inactive_observation_ids.update(
                int(obs_id)
                for obs_id in view["observation_ids"][~observation_inliers].tolist()
            )
        result["frame_ids"] = view["frame_ids"]
        result["point_ids"] = view["point_ids"]
        result["scope"] = view["scope"]
        return result


__all__ = ["FactorGraph", "select_local_fixed_ids"]
