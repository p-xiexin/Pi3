"""Calibrated pair verification and sparse geometric initialization."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import average_rotations, camera_centers, maximum_spanning_tree


MIN_TRACK_OBSERVATIONS = 3
MIN_PAIR_MATCHES = 16
MIN_PAIR_INLIERS = 12
ESSENTIAL_RANSAC_THRESHOLD_PX = 1.0
MIN_TRIANGULATION_ANGLE_DEG = 1.5


@dataclass(frozen=True)
class VerifiedTracks:
    """Observation mask and relative-pose graph after essential verification."""

    mask: torch.Tensor
    keep: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    relative: torch.Tensor
    weight: torch.Tensor
    pair_matches: torch.Tensor
    pair_inliers: torch.Tensor


def _normalized_points(points, K):
    """Convert pixel coordinates to calibrated normalized image coordinates."""
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=-1)
    rays = torch.einsum("ij,pj->pi", torch.linalg.inv(K), homogeneous)
    return rays[:, :2] / rays[:, 2:3]


def _estimate_relative_pose(
    source_points,
    target_points,
    source_K,
    target_K,
    cv2_module=None,
):
    """Estimate target-from-source pose with calibrated essential RANSAC."""
    if cv2_module is None:
        import cv2 as cv2_module

    source_normalized = _normalized_points(source_points, source_K)
    target_normalized = _normalized_points(target_points, target_K)
    source_np = source_normalized.detach().double().cpu().numpy()
    target_np = target_normalized.detach().double().cpu().numpy()
    focal = float(
        torch.stack(
            (
                source_K[0, 0],
                source_K[1, 1],
                target_K[0, 0],
                target_K[1, 1],
            )
        ).mean()
    )
    threshold = ESSENTIAL_RANSAC_THRESHOLD_PX / max(focal, 1.0)
    essential, ransac_mask = cv2_module.findEssentialMat(
        source_np,
        target_np,
        np.eye(3),
        method=cv2_module.RANSAC,
        prob=0.999,
        threshold=threshold,
    )
    if essential is None or ransac_mask is None:
        return None

    essential = np.asarray(essential)
    best = None
    for start in range(0, essential.shape[0], 3):
        candidate = essential[start : start + 3]
        if candidate.shape != (3, 3):
            continue
        count, rotation, translation, pose_mask = cv2_module.recoverPose(
            candidate,
            source_np,
            target_np,
            np.eye(3),
            mask=np.asarray(ransac_mask).copy(),
        )
        inliers = np.asarray(pose_mask).reshape(-1) > 0
        if inliers.shape[0] != source_np.shape[0]:
            raise RuntimeError("recoverPose returned a misaligned inlier mask")
        if best is None or int(count) > best[0]:
            best = int(count), rotation, np.asarray(translation).reshape(3), inliers
    if best is None or best[0] < MIN_PAIR_INLIERS:
        return None

    _, rotation, translation, inliers = best
    return (
        torch.from_numpy(np.asarray(rotation)).to(source_points),
        torch.from_numpy(translation).to(source_points),
        torch.from_numpy(inliers).to(device=source_points.device),
    )


def _candidate_pairs(frame_count, pairs):
    if pairs is None:
        return list(combinations(range(frame_count), 2))
    if isinstance(pairs, torch.Tensor):
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape [pair_count,2]")
        pairs = pairs.detach().cpu().tolist()
    result = []
    seen = set()
    for source, target in pairs:
        source, target = int(source), int(target)
        if source == target:
            continue
        if source < 0 or target < 0 or source >= frame_count or target >= frame_count:
            raise ValueError("pair index is outside the track window")
        pair = (min(source, target), max(source, target))
        if pair not in seen:
            result.append(pair)
            seen.add(pair)
    return result


def verify_tracks(
    tracks,
    mask,
    weights,
    K,
    minimum_track_observations=None,
    pairs=None,
):
    """Build a pose graph and retain observations supported by essential geometry."""
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError("tracks must have shape [frames,points,2]")
    if mask.shape != tracks.shape[:2] or weights.shape != mask.shape:
        raise ValueError("track masks and weights must have shape [frames,points]")
    if K.shape != (tracks.shape[0], 3, 3):
        raise ValueError("intrinsics must have shape [frames,3,3]")
    minimum_track_observations = (
        MIN_TRACK_OBSERVATIONS
        if minimum_track_observations is None
        else int(minimum_track_observations)
    )
    if minimum_track_observations < 2:
        raise ValueError("minimum_track_observations must be at least two")

    mask = (
        mask.to(dtype=torch.bool)
        & torch.isfinite(tracks).all(dim=-1)
        & torch.isfinite(weights)
        & (weights > 0)
    )
    multiview = mask.sum(dim=0) >= minimum_track_observations
    support = torch.zeros_like(mask, dtype=torch.int32)
    sources, targets, transforms, edge_weights = [], [], [], []
    pair_matches, pair_inliers = [], []
    for source, target in _candidate_pairs(tracks.shape[0], pairs):
        shared = torch.nonzero(
            mask[source] & mask[target] & multiview, as_tuple=False
        ).squeeze(dim=-1)
        if shared.numel() < MIN_PAIR_MATCHES:
            continue
        estimate = _estimate_relative_pose(
            tracks[source, shared], tracks[target, shared], K[source], K[target]
        )
        if estimate is None:
            continue
        rotation, translation, local_inliers = estimate
        local_inliers = torch.as_tensor(
            local_inliers, device=shared.device, dtype=torch.bool
        ).reshape(-1)
        if local_inliers.numel() != shared.numel():
            raise RuntimeError("relative pose estimator returned a misaligned inlier mask")
        inlier_ids = shared[local_inliers]
        if inlier_ids.numel() < MIN_PAIR_INLIERS:
            continue
        relative = torch.eye(4, device=K.device, dtype=K.dtype)
        relative[:3, :3] = rotation.to(device=K.device, dtype=K.dtype)
        relative[:3, 3] = translation.to(device=K.device, dtype=K.dtype)
        sources.append(source)
        targets.append(target)
        transforms.append(relative)
        edge_weights.append(
            (weights[source, inlier_ids] * weights[target, inlier_ids]).sum()
        )
        pair_matches.append(shared.numel())
        pair_inliers.append(inlier_ids.numel())
        support[source, inlier_ids] += 1
        support[target, inlier_ids] += 1

    if not transforms:
        raise RuntimeError("essential-matrix verification produced no pose edges")
    verified_mask = mask & (support > 0)
    keep = verified_mask.sum(dim=0) >= minimum_track_observations
    if not bool(keep.any()):
        raise RuntimeError("essential-matrix verification rejected every track")
    return VerifiedTracks(
        mask=verified_mask[:, keep],
        keep=keep,
        source=torch.tensor(sources, device=K.device, dtype=torch.long),
        target=torch.tensor(targets, device=K.device, dtype=torch.long),
        relative=torch.stack(transforms),
        weight=torch.stack(edge_weights).to(device=K.device, dtype=K.dtype),
        pair_matches=torch.tensor(pair_matches, device=K.device, dtype=torch.long),
        pair_inliers=torch.tensor(pair_inliers, device=K.device, dtype=torch.long),
    )


def initialize_poses(frame_count, verified):
    """Initialize world-to-camera poses from verified edges and rotation averaging."""
    poses = maximum_spanning_tree(
        frame_count,
        verified.source,
        verified.target,
        verified.relative,
        verified.weight,
    )
    rotations = average_rotations(
        poses,
        verified.source,
        verified.target,
        verified.relative,
        verified.weight,
    )
    centers = camera_centers(poses)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = -torch.einsum("sij,sj->si", rotations, centers)
    return poses


def _packet_pairs(packet, frame_lookup):
    edges = packet.get("edges", [])
    pairs = []
    if edges:
        for edge in edges:
            if len(edge) < 2:
                raise ValueError("packet edge must contain source and target frame IDs")
            source, target = int(edge[0]), int(edge[1])
            if source not in frame_lookup or target not in frame_lookup:
                raise ValueError("packet edge references a frame outside the packet")
            pairs.append((frame_lookup[source], frame_lookup[target]))
        return pairs
    for part in packet.get("parts", []):
        source = int(part["reference"])
        if source not in frame_lookup:
            raise ValueError("packet part reference is outside the packet")
        for target in part["obs_frames"].unique().tolist():
            target = int(target)
            if target != source:
                pairs.append((frame_lookup[source], frame_lookup[target]))
    return pairs


def _packet_tracks(packet):
    frame_ids = torch.as_tensor(packet["frame_ids"], dtype=torch.long)
    K = packet["K"]
    if frame_ids.ndim != 1 or frame_ids.numel() != K.shape[0]:
        raise ValueError("packet frame IDs and intrinsics are misaligned")
    frame_ids = frame_ids.to(device=K.device)
    frame_lookup = {frame_id: local for local, frame_id in enumerate(frame_ids.tolist())}
    parts = packet.get("parts", [])
    if not parts:
        raise RuntimeError("factor packet has no track parts to verify")

    offsets = []
    all_track_ids = []
    point_count = 0
    for part in parts:
        if "track_ids" not in part:
            raise RuntimeError("factor packet has no stable track IDs")
        track_ids = part["track_ids"].to(device=K.device, dtype=torch.long)
        keys = part.get("keys")
        if track_ids.ndim != 1 or (keys is not None and keys.shape != track_ids.shape):
            raise RuntimeError("factor packet track IDs and keys are misaligned")
        if bool((track_ids < 0).any()):
            raise RuntimeError("stable track IDs must be nonnegative")
        offsets.append((point_count, point_count + track_ids.numel()))
        point_count += track_ids.numel()
        all_track_ids.append(track_ids)
    stable_track_ids = torch.cat(all_track_ids)
    if stable_track_ids.unique().numel() != stable_track_ids.numel():
        raise RuntimeError("stable track ID occurs in more than one packet part")

    tracks = torch.zeros(
        frame_ids.numel(), point_count, 2, device=K.device, dtype=K.dtype
    )
    weights = torch.zeros(
        frame_ids.numel(), point_count, device=K.device, dtype=K.dtype
    )
    mask = torch.zeros(
        frame_ids.numel(), point_count, device=K.device, dtype=torch.bool
    )
    observation_locations = []
    for part, (start, stop) in zip(parts, offsets):
        obs_frames = part["obs_frames"].to(device=K.device, dtype=torch.long)
        obs_points = part["obs_points"].to(device=K.device, dtype=torch.long)
        obs_uv = part["obs_uv"].to(device=K.device, dtype=K.dtype)
        obs_weights = part["obs_weights"].to(device=K.device, dtype=K.dtype)
        observation_count = obs_frames.numel()
        if (
            obs_points.shape != obs_frames.shape
            or obs_uv.shape != (observation_count, 2)
            or obs_weights.shape != obs_frames.shape
        ):
            raise RuntimeError("factor packet observations are misaligned")
        if observation_count and (
            int(obs_points.min()) < 0 or int(obs_points.max()) >= stop - start
        ):
            raise RuntimeError("factor packet observation has an invalid point index")
        try:
            local_frames = torch.tensor(
                [frame_lookup[int(frame)] for frame in obs_frames.tolist()],
                device=K.device,
                dtype=torch.long,
            )
        except KeyError as error:
            raise RuntimeError(
                f"factor packet observation references unknown frame {error.args[0]}"
            ) from error
        point_columns = obs_points + start
        linear = local_frames * point_count + point_columns
        if linear.unique().numel() != linear.numel():
            raise RuntimeError("factor packet contains duplicate frame-track observations")
        tracks[local_frames, point_columns] = obs_uv
        weights[local_frames, point_columns] = obs_weights
        mask[local_frames, point_columns] = True
        observation_locations.append((local_frames, point_columns))
    return (
        frame_ids,
        frame_lookup,
        parts,
        offsets,
        stable_track_ids,
        tracks,
        mask,
        weights,
        observation_locations,
    )


def verify_packet(
    packet,
    minimum_track_observations=None,
    pairs=None,
    initialize=None,
):
    """Verify one factor packet without assigning or merging stable track IDs."""
    (
        frame_ids,
        frame_lookup,
        parts,
        offsets,
        stable_track_ids,
        tracks,
        mask,
        weights,
        observation_locations,
    ) = _packet_tracks(packet)
    packet_kind = packet.get("kind", "sliding")
    if minimum_track_observations is None:
        minimum_track_observations = 2 if packet_kind == "loop" else MIN_TRACK_OBSERVATIONS
    if pairs is None and packet_kind == "loop":
        pairs = _packet_pairs(packet, frame_lookup)
    verified = verify_tracks(
        tracks,
        mask,
        weights,
        packet["K"],
        minimum_track_observations=minimum_track_observations,
        pairs=pairs,
    )
    full_verified_mask = torch.zeros_like(mask)
    full_verified_mask[:, verified.keep] = verified.mask

    verified_parts = []
    for part, (start, stop), locations in zip(parts, offsets, observation_locations):
        reference_local = frame_lookup[int(part["reference"])]
        owner_supported = full_verified_mask[reference_local, start:stop]
        point_keep = verified.keep[start:stop] & owner_supported
        local_frames, point_columns = locations
        observation_keep = full_verified_mask[local_frames, point_columns]
        observation_keep &= point_keep[
            part["obs_points"].to(point_keep.device)
        ]
        if not bool(point_keep.any()) or not bool(observation_keep.any()):
            continue
        point_map = torch.full(
            (stop - start,), -1, device=point_keep.device, dtype=torch.long
        )
        point_map[point_keep] = torch.arange(
            int(point_keep.sum()), device=point_keep.device, dtype=torch.long
        )
        verified_part = dict(part)
        for name in ("keys", "track_ids", "anchors", "queries", "anchor_weights"):
            value = part.get(name)
            if isinstance(value, torch.Tensor) and value.shape[:1] == point_keep.shape:
                verified_part[name] = value[point_keep.to(value.device)]
        for name in ("obs_frames", "obs_uv", "obs_weights"):
            value = part[name]
            verified_part[name] = value[observation_keep.to(value.device)]
        kept_obs_points = part["obs_points"][observation_keep.to(part["obs_points"].device)]
        verified_part["obs_points"] = point_map.to(kept_obs_points.device)[kept_obs_points]
        if bool(
            (
                (verified_part["obs_points"] < 0)
                | (verified_part["obs_points"] >= int(point_keep.sum()))
            ).any()
        ):
            raise RuntimeError("verified observation references a rejected stable track")
        verified_part["edge_source"] = []
        verified_part["edge_target"] = []
        verified_part["edge_weight"] = []
        verified_parts.append(verified_part)

    if not verified_parts:
        raise RuntimeError(
            "essential-matrix verification rejected every owner observation"
        )

    global_source = frame_ids[verified.source]
    global_target = frame_ids[verified.target]
    output = dict(packet)
    output["frame_ids"] = frame_ids
    output["parts"] = verified_parts
    output["edges"] = [
        (int(source), int(target), relative, float(weight))
        for source, target, relative, weight in zip(
            global_source.tolist(),
            global_target.tolist(),
            verified.relative,
            verified.weight,
        )
    ]
    if initialize is None:
        initialize = packet_kind != "loop"
    if initialize:
        output["poses"] = initialize_poses(frame_ids.numel(), verified)
    output["geometry_pairs"] = verified.source.new_tensor(verified.source.numel())
    output["geometry_tracks"] = sum(
        part["track_ids"].numel() for part in verified_parts
    )
    output["geometry_observations"] = sum(
        part["obs_frames"].numel() for part in verified_parts
    )
    output["geometry_pair_matches"] = verified.pair_matches
    output["geometry_pair_inliers"] = verified.pair_inliers
    output["geometry_inlier_ratio"] = (
        verified.pair_inliers.sum().to(dtype=packet["K"].dtype)
        / verified.pair_matches.sum().clamp_min(1)
    )
    output["verified_track_ids"] = torch.cat(
        [part["track_ids"] for part in verified_parts]
    )
    return output


def triangulate_tracks(
    poses,
    K,
    ii,
    jj,
    uv,
    point_count,
    minimum_track_observations=None,
):
    """Triangulate tracks and apply cheirality and minimum-parallax checks."""
    minimum_track_observations = (
        MIN_TRACK_OBSERVATIONS
        if minimum_track_observations is None
        else int(minimum_track_observations)
    )
    if minimum_track_observations < 2:
        raise ValueError("minimum_track_observations must be at least two")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape [frames,4,4]")
    if K.shape != (poses.shape[0], 3, 3):
        raise ValueError("intrinsics must have shape [frames,3,3]")
    if ii.ndim != 1 or jj.shape != ii.shape or uv.shape != (ii.numel(), 2):
        raise ValueError("triangulation observations are misaligned")
    point_count = int(point_count)
    if ii.numel() and (
        int(ii.min()) < 0
        or int(ii.max()) >= poses.shape[0]
        or int(jj.min()) < 0
        or int(jj.max()) >= point_count
    ):
        raise ValueError("triangulation observation index is out of range")

    uv1 = torch.cat((uv, torch.ones_like(uv[:, :1])), dim=-1)
    normalized = torch.einsum("oij,oj->oi", torch.linalg.inv(K)[ii], uv1)
    normalized = normalized[:, :2] / normalized[:, 2:3]
    centers = camera_centers(poses)
    points = torch.full(
        (point_count, 3), float("nan"), device=poses.device, dtype=poses.dtype
    )
    observation_mask = torch.zeros_like(ii, dtype=torch.bool)
    point_mask = torch.zeros(point_count, device=poses.device, dtype=torch.bool)
    angles = torch.zeros(point_count, device=poses.device, dtype=poses.dtype)

    for point_id in range(point_count):
        observation_ids = torch.nonzero(jj == point_id, as_tuple=False).squeeze(-1)
        cameras = ii[observation_ids]
        if cameras.unique().numel() < minimum_track_observations:
            continue
        projection = poses[cameras, :3]
        xy = normalized[observation_ids]
        rows = torch.cat(
            (
                xy[:, 0:1] * projection[:, 2] - projection[:, 0],
                xy[:, 1:2] * projection[:, 2] - projection[:, 1],
            )
        )
        _, _, vh = torch.linalg.svd(rows)
        homogeneous = vh[-1]
        if homogeneous[3].abs() <= torch.finfo(homogeneous.dtype).eps:
            continue
        point = homogeneous[:3] / homogeneous[3]
        camera_points = torch.einsum(
            "cij,j->ci", poses[cameras, :3, :3], point
        ) + poses[cameras, :3, 3]
        positive = torch.isfinite(camera_points).all(dim=-1) & (camera_points[:, 2] > 0)
        positive_ids = observation_ids[positive]
        positive_cameras = ii[positive_ids]
        if positive_cameras.unique().numel() < minimum_track_observations:
            continue
        directions = F.normalize(point[None] - centers[positive_cameras], dim=-1)
        cosine = directions @ directions.transpose(0, 1)
        cosine.fill_diagonal_(1)
        angle = torch.rad2deg(torch.acos(cosine.min().clamp(-1, 1)))
        if not bool(torch.isfinite(angle) & (angle >= MIN_TRIANGULATION_ANGLE_DEG)):
            continue
        points[point_id] = point
        observation_mask[positive_ids] = True
        point_mask[point_id] = True
        angles[point_id] = angle
    return points, observation_mask, point_mask, angles


def validate_landmarks(
    poses,
    points,
    ii,
    jj,
    minimum_track_observations=None,
):
    """Validate existing landmarks with cheirality, support, and parallax."""
    minimum_track_observations = (
        MIN_TRACK_OBSERVATIONS
        if minimum_track_observations is None
        else int(minimum_track_observations)
    )
    camera_points = torch.einsum(
        "oij,oj->oi", poses[ii, :3, :3], points[jj]
    ) + poses[ii, :3, 3]
    positive = torch.isfinite(camera_points).all(dim=-1) & (camera_points[:, 2] > 0)
    centers = camera_centers(poses)
    point_mask = torch.zeros(points.shape[0], device=poses.device, dtype=torch.bool)
    angles = torch.zeros(points.shape[0], device=poses.device, dtype=poses.dtype)
    for point_id in range(points.shape[0]):
        observation_ids = torch.nonzero(
            (jj == point_id) & positive, as_tuple=False
        ).squeeze(dim=-1)
        cameras = ii[observation_ids]
        if cameras.unique().numel() < minimum_track_observations:
            continue
        directions = F.normalize(points[point_id][None] - centers[cameras], dim=-1)
        cosine = directions @ directions.transpose(0, 1)
        cosine.fill_diagonal_(1)
        angle = torch.rad2deg(torch.acos(cosine.min().clamp(-1, 1)))
        if bool(torch.isfinite(angle) & (angle >= MIN_TRIANGULATION_ANGLE_DEG)):
            point_mask[point_id] = True
            angles[point_id] = angle
    return positive & point_mask[jj], point_mask, angles


__all__ = [
    "ESSENTIAL_RANSAC_THRESHOLD_PX",
    "MIN_PAIR_INLIERS",
    "MIN_PAIR_MATCHES",
    "MIN_TRACK_OBSERVATIONS",
    "MIN_TRIANGULATION_ANGLE_DEG",
    "VerifiedTracks",
    "initialize_poses",
    "triangulate_tracks",
    "validate_landmarks",
    "verify_packet",
    "verify_tracks",
]
