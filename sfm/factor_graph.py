"""One persistent factor graph shared by window and full-sequence optimization."""

import torch

from .geometry import average_rotations, camera_centers, maximum_spanning_tree
from .optimizer import optimize_view


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
        self.anchor_observed = set()
        self.edges = []

    @property
    def frame_ids(self):
        """Return frame IDs already registered in the persistent graph."""
        return set(self.poses)

    def _reserve_points(self, required, example):
        """Grow contiguous landmark storage geometrically while preserving IDs."""
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

    def add_factors(self, packet):
        """Merge one window packet before any local or global optimization."""
        frame_ids = packet["frame_ids"].tolist()
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
        for part in packet["parts"]:
            point_ids = []
            for feature_id, anchor in zip(part["keys"].tolist(), part["anchors"]):
                # This key joins repeated observations of the same SIFT query
                # across overlapping windows into one persistent landmark.
                key = (int(part["reference"]), int(feature_id))
                if key not in self.point_lookup:
                    point_id = self.point_count
                    self._reserve_points(point_id + 1, anchor)
                    self.point_lookup[key] = point_id
                    self.point_references[point_id] = key[0]
                    self.point_anchors[point_id] = anchor
                    self.point_count += 1
                point_ids.append(self.point_lookup[key])
            point_ids = torch.tensor(point_ids, device=packet["frame_ids"].device, dtype=torch.long)
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
        for source, target, relative, weight in packet["edges"]:
            if source > target:
                source, target, relative = target, source, torch.linalg.inv(relative)
            self.edges.append((int(source), int(target), relative.clone(), float(weight)))
        return known

    def _view(self, frame_ids, scope):
        """Materialize a compact optimizer view over selected cameras and landmarks."""
        frame_ids = list(dict.fromkeys(map(int, frame_ids)))
        local_frame = {frame_id: i for i, frame_id in enumerate(frame_ids)}
        device = next(iter(self.poses.values())).device
        selected_ids = torch.tensor(frame_ids, device=device, dtype=torch.long)
        chunks = []
        for reference, observation_frames, observation_points, uv, weight in self.observations:
            if reference not in local_frame:
                continue
            mask = torch.isin(observation_frames, selected_ids)
            if mask.any():
                chunks.append(
                    (
                        observation_frames[mask],
                        observation_points[mask],
                        uv[mask],
                        weight[mask],
                    )
                )
        if not chunks:
            raise RuntimeError(f"{scope} graph view has no observations")
        observation_frames = torch.cat([chunk[0] for chunk in chunks])
        observation_points = torch.cat([chunk[1] for chunk in chunks])
        uv = torch.cat([chunk[2] for chunk in chunks])
        weight = torch.cat([chunk[3] for chunk in chunks])
        # BA receives only landmarks constrained by at least three selected
        # camera observations. The persistent graph keeps the remaining data.
        counts = torch.bincount(observation_points, minlength=self.point_count)
        keep_points = counts >= 3
        mask = keep_points[observation_points]
        observation_frames, observation_points = observation_frames[mask], observation_points[mask]
        uv, weight = uv[mask], weight[mask]
        point_ids = torch.nonzero(keep_points, as_tuple=False).squeeze(-1)
        if not point_ids.numel():
            raise RuntimeError(f"{scope} graph view has no multi-view tracks")
        camera_map = torch.full((max(frame_ids) + 1,), -1, device=device, dtype=torch.long)
        camera_map[selected_ids] = torch.arange(len(frame_ids), device=device)
        point_map = torch.full((self.point_count,), -1, device=device, dtype=torch.long)
        point_map[point_ids] = torch.arange(point_ids.numel(), device=device)
        poses = torch.stack([self.poses[frame_id] for frame_id in frame_ids])
        centers = camera_centers(poses)
        references = camera_map[self.point_references[point_ids]]
        anchors = self.point_anchors[point_ids]
        points = torch.einsum(
            "pji,pj->pi", poses[references, :3, :3], anchors
        ) + centers[references]
        initialized = self.point_initialized[point_ids]
        points[initialized] = self.point_positions[point_ids[initialized]]
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
        }

    def local_view(self, frame_ids):
        """Build a window-sized view of the persistent factor graph."""
        return self._view(frame_ids, "local")

    def full_view(self):
        """Build the full-sequence view consumed by global BA."""
        return self._view(sorted(self.poses), "global")

    def initialize_global(self):
        """Initialize all cameras from the maximum spanning tree and rotation averaging."""
        frame_ids = sorted(self.poses)
        local = {frame_id: i for i, frame_id in enumerate(frame_ids)}
        edges = [
            (local[source], local[target], relative, weight)
            for source, target, relative, weight in self.edges
            if source in local and target in local
        ]
        if len(frame_ids) > 1 and not edges:
            raise RuntimeError("global pose graph has no edges")
        if len(frame_ids) == 1:
            return
        source = torch.tensor([edge[0] for edge in edges], device=self.poses[frame_ids[0]].device)
        target = torch.tensor([edge[1] for edge in edges], device=source.device)
        relative = torch.stack([edge[2] for edge in edges])
        weight = torch.tensor(
            [edge[3] for edge in edges],
            device=source.device,
            dtype=relative.dtype,
        )
        poses = maximum_spanning_tree(len(frame_ids), source, target, relative, weight)
        rotations = average_rotations(poses, source, target, relative, weight)
        centers = camera_centers(poses)
        poses[:, :3, :3] = rotations
        poses[:, :3, 3] = -torch.einsum("sij,sj->si", rotations, centers)
        for frame_id, pose in zip(frame_ids, poses):
            self.poses[frame_id] = pose
        self.point_initialized[:self.point_count] = False

    def optimize(self, view, fixed_ids, iterations):
        """Run the shared BA backend and commit its state to the persistent graph."""
        local = {int(frame_id): i for i, frame_id in enumerate(view["frame_ids"].tolist())}
        fixed = [local[int(frame_id)] for frame_id in fixed_ids if int(frame_id) in local]
        if not fixed:
            raise ValueError("optimization requires at least one fixed camera")
        result = optimize_view(view, fixed, iterations)
        for frame_id, pose in zip(view["frame_ids"].tolist(), result["poses"]):
            self.poses[int(frame_id)] = pose
        self.point_positions[view["point_ids"]] = result["points"]
        self.point_initialized[view["point_ids"]] = True
        result["frame_ids"] = view["frame_ids"]
        result["point_ids"] = view["point_ids"]
        result["scope"] = view["scope"]
        return result


__all__ = ["FactorGraph", "select_local_fixed_ids"]
