"""Probe sliding-window scale degeneracy from a cached data.h5."""

import argparse
from collections import defaultdict
from types import SimpleNamespace

import torch

from .factor_graph import FactorGraph
from .frontend_cache import load_frontend_cache, packet_to_device, restore_frames
from .geometry import camera_centers
from .optimizer import (
    _bearing_adjust,
    _bundle_adjust,
    _fixed_camera_mask,
    _robust_weight,
    _scale_factor,
    _schur_step,
    _select_scale_gauge,
    _stage_fixed_camera_mask,
    filter_reprojection_observations,
)


def _mst_edges(edges):
    nodes = sorted({int(node) for edge in edges for node in edge[:2]})
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    weights = torch.tensor([float(edge[3]) for edge in edges])
    selected = []
    for index in torch.argsort(weights, descending=True).tolist():
        source, target = map(int, edges[index][:2])
        root_source, root_target = find(source), find(target)
        if root_source == root_target:
            continue
        parent[root_source] = root_target
        selected.append(edges[index])
        if len(selected) == len(nodes) - 1:
            break
    return selected


def _point_frames(graph, active_only=False):
    frames = defaultdict(set)
    for (_, observation_frames, observation_points, _, _), observation_ids in zip(
        graph.observations, graph.observation_ids
    ):
        for frame, point, observation_id in zip(
            observation_frames.tolist(),
            observation_points.tolist(),
            observation_ids.tolist(),
        ):
            if active_only and (
                int(observation_id) in graph.inactive_observation_ids
                or not bool(graph.point_initialized[int(point)])
            ):
                continue
            frames[int(point)].add(int(frame))
    return frames


def _view_point_frames(view, observation_mask):
    frames = defaultdict(set)
    ids = torch.nonzero(observation_mask, as_tuple=False).squeeze(-1)
    camera_frames = view["frame_ids"][view["ii"][ids]].tolist()
    point_ids = view["jj"][ids].tolist()
    for frame, point in zip(camera_frames, point_ids):
        frames[int(point)].add(int(frame))
    return frames


def _strong_bridges(point_frames, cut):
    return [
        point
        for point, frames in point_frames.items()
        if sum(frame < cut for frame in frames) >= 2
        and sum(frame >= cut for frame in frames) >= 2
    ]


def _window_path(poses, frame_ids, window):
    centers = camera_centers(poses)
    lookup = {int(frame): index for index, frame in enumerate(frame_ids.tolist())}
    selected = centers[[lookup[int(frame)] for frame in window]]
    return torch.linalg.vector_norm(selected[1:] - selected[:-1], dim=-1).sum()


def _cuts(windows):
    seen = set(windows[0])
    cuts = []
    for window in windows[1:]:
        new_frames = [frame for frame in window if frame not in seen]
        if new_frames:
            cuts.append(min(new_frames))
        seen.update(window)
    return cuts


def _summary(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    return tuple(float(torch.quantile(values, q)) for q in (0.1, 0.5, 0.9))


def _normalized_bearing_linearization(centers, points, ii, jj, projector):
    """Linearize P(X-C)/||X-C|| for the probe-only bearing A/B."""
    offset = points[jj] - centers[ii]
    distance = offset.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    direction = offset / distance
    eye = torch.eye(3, device=offset.device, dtype=offset.dtype)
    direction_projector = eye - direction[..., None] * direction[..., None, :]
    jacobian = (projector @ direction_projector) / distance[..., None]
    error = torch.einsum("oij,oj->oi", projector, direction)
    return error, -jacobian, jacobian


def _normalized_bearing_adjust(
    poses, points, view, fixed, iterations, scale_gauge=None
):
    """Run a scale-invariant bearing refinement without changing production code."""
    ii, jj = view["ii"], view["jj"]
    fixed, active = _stage_fixed_camera_mask(fixed, ii, poses.shape[0])
    centers = camera_centers(poses)
    if scale_gauge is None:
        scale_gauge = _select_scale_gauge(centers, fixed, active)
    rotations = poses[:, :3, :3].clone()
    uv1 = torch.cat((view["uv"], torch.ones_like(view["uv"][:, :1])), -1)
    rays_camera = torch.einsum("oij,oj->oi", torch.linalg.inv(view["K"])[ii], uv1)
    rays_world = torch.einsum(
        "oij,oj->oi", rotations[ii].transpose(-1, -2), rays_camera
    )
    rays = rays_world / rays_world.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    eye = torch.eye(3, device=poses.device, dtype=poses.dtype)
    projector = eye - rays[..., None] * rays[..., None, :]
    for _ in range(int(iterations)):
        error, Jc, Jx = _normalized_bearing_linearization(
            centers, points, ii, jj, projector
        )
        weight = _robust_weight(error, view["weight"], 1.0)
        dc, dpoints = _schur_step(
            Jc,
            Jx,
            -error,
            weight,
            ii,
            jj,
            poses.shape[0],
            points.shape[0],
            fixed,
            _scale_factor(scale_gauge, centers, poses, False),
        )
        centers += dc
        points += dpoints
    poses[:, :3, 3] = -torch.einsum("sij,sj->si", rotations, centers)
    return poses, points


def _edge_directions(poses, frame_ids, edges):
    """Convert target-from-source translations into oriented world directions."""
    local = {int(frame): index for index, frame in enumerate(frame_ids.tolist())}
    source = torch.tensor(
        [local[int(edge[0])] for edge in edges], device=poses.device, dtype=torch.long
    )
    target = torch.tensor(
        [local[int(edge[1])] for edge in edges], device=poses.device, dtype=torch.long
    )
    translation = torch.stack([edge[2][:3, 3] for edge in edges])
    translation = translation / translation.norm(dim=-1, keepdim=True)
    direction = -torch.einsum(
        "eji,ej->ei", poses[target, :3, :3], translation
    )
    weight = torch.tensor(
        [float(edge[3]) for edge in edges], device=poses.device, dtype=poses.dtype
    )
    gap = torch.tensor(
        [abs(int(edge[1]) - int(edge[0])) for edge in edges],
        device=poses.device,
        dtype=torch.long,
    )
    return source, target, direction, weight, gap


def _edge_direction_stats(poses, frame_ids, edges):
    source, target, direction, _, _ = _edge_directions(poses, frame_ids, edges)
    centers = camera_centers(poses)
    predicted = centers[target] - centers[source]
    predicted = predicted / predicted.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    cosine = (predicted * direction).sum(dim=-1).clamp(-1, 1)
    angles = torch.rad2deg(torch.acos(cosine))
    return (
        float(torch.quantile(angles, 0.5)),
        float(torch.quantile(angles, 0.9)),
        float((cosine > 0).float().mean()),
    )


def _translation_direction_average(poses, frame_ids, edges):
    """Build a probe-only spectral camera-center candidate from edge directions."""
    source, target, direction, weight, gap = _edge_directions(
        poses, frame_ids, edges
    )
    source = source.cpu()
    target = target.cpu()
    direction = direction.detach().cpu().double()
    weight = weight.detach().cpu().double()
    gap = gap.cpu()
    camera_count = poses.shape[0]
    A = torch.zeros(3 * len(edges), 3 * (camera_count - 1), dtype=torch.float64)
    eye = torch.eye(3, dtype=torch.float64)
    median_weight = torch.quantile(weight, 0.5)
    for edge_id, (i, j, d, w) in enumerate(
        zip(source.tolist(), target.tolist(), direction, weight)
    ):
        block = torch.sqrt(w / median_weight) * (eye - d[:, None] * d[None, :])
        rows = slice(3 * edge_id, 3 * edge_id + 3)
        if i:
            A[rows, 3 * (i - 1) : 3 * i] -= block
        if j:
            A[rows, 3 * (j - 1) : 3 * j] += block
    eigenvalues, eigenvectors = torch.linalg.eigh(A.T @ A)
    centers = torch.zeros(camera_count, 3, dtype=torch.float64)
    centers[1:] = eigenvectors[:, 0].reshape(-1, 3)
    shortest = gap == gap.min()
    projected = (
        direction[shortest]
        * (centers[target[shortest]] - centers[source[shortest]])
    ).sum(dim=-1)
    scale = (weight[shortest] * projected).sum() / weight[shortest].sum()
    centers /= scale
    spectrum = eigenvalues.clamp_min(0)
    maximum = float(spectrum[-1])
    threshold = maximum * 1.0e-8
    near_null = int((spectrum <= threshold).sum())
    denominator = max(float(spectrum[0]), maximum * 1.0e-15)
    eig_gap = float(spectrum[1]) / denominator
    candidate = poses.detach().cpu().double().clone()
    candidate[:, :3, 3] = -torch.einsum(
        "sij,sj->si", candidate[:, :3, :3], centers
    )
    return candidate.to(poses), near_null, eig_gap


def _ray_stats(poses, points, view):
    ii, jj = view["ii"], view["jj"]
    uv1 = torch.cat((view["uv"], torch.ones_like(view["uv"][:, :1])), dim=-1)
    rays_camera = torch.einsum("oij,oj->oi", torch.linalg.inv(view["K"])[ii], uv1)
    rays_world = torch.einsum(
        "oij,oj->oi", poses[ii, :3, :3].transpose(-1, -2), rays_camera
    )
    rays_world /= rays_world.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    centers = camera_centers(poses)
    predicted = points[jj] - centers[ii]
    predicted /= predicted.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    cosine = (rays_world * predicted).sum(dim=-1).clamp(-1, 1)
    angles = torch.rad2deg(torch.acos(cosine))
    camera_points = torch.einsum(
        "oij,oj->oi", poses[ii, :3, :3], points[jj]
    ) + poses[ii, :3, 3]
    return (
        float(torch.quantile(angles, 0.5)),
        float(torch.quantile(angles, 0.9)),
        float((camera_points[:, 2] > 0.2).float().mean()),
    )


def _pair_step(poses, frame_ids, source, target):
    lookup = {int(frame): index for index, frame in enumerate(frame_ids.tolist())}
    centers = camera_centers(poses)
    return float((centers[lookup[target]] - centers[lookup[source]]).norm())


def _cut_local_ratio(poses, frame_ids, cut):
    lookup = {int(frame): index for index, frame in enumerate(frame_ids.tolist())}
    centers = camera_centers(poses)
    local_steps = []
    for source in range(cut - 3, cut + 3):
        if source == cut - 1 or source not in lookup or source + 1 not in lookup:
            continue
        local_steps.append((centers[lookup[source + 1]] - centers[lookup[source]]).norm())
    local = float(torch.quantile(torch.stack(local_steps), 0.5))
    return _pair_step(poses, frame_ids, cut - 1, cut) / local


def _pi3_center_candidate(initial_poses, frame_ids, pi3_windows):
    """Chain metric-scaled Pi3 centers using each window anchor rotation."""
    pose_local = {
        int(frame): index for index, frame in enumerate(frame_ids.tolist())
    }
    centers = {}
    diagnostics = []
    for window_ids, T_WCs, metric_scale in pi3_windows:
        window_ids = list(map(int, window_ids.tolist()))
        local_centers = T_WCs[:, :3, 3].to(initial_poses) * float(metric_scale)
        anchor = window_ids[0]
        anchor_center = centers.get(
            anchor,
            torch.zeros(3, device=initial_poses.device, dtype=initial_poses.dtype),
        )
        anchor_rotation = initial_poses[pose_local[anchor], :3, :3]
        predicted = anchor_center + torch.einsum(
            "ij,nj->ni", anchor_rotation.transpose(0, 1), local_centers
        )
        overlap = [index for index, frame in enumerate(window_ids) if frame in centers]
        if overlap:
            reference = torch.stack([centers[window_ids[index]] for index in overlap])
            residual = (predicted[overlap] - reference).norm(dim=-1)
            if len(overlap) > 1:
                current_distances = torch.pdist(predicted[overlap])
                reference_distances = torch.pdist(reference)
                residual_scale = float(
                    (current_distances * reference_distances).sum()
                    / current_distances.square().sum()
                )
                local_step = torch.linalg.vector_norm(
                    reference[1:] - reference[:-1], dim=-1
                ).median()
            else:
                residual_scale = float("nan")
                local_step = residual.new_tensor(1.0)
            residual50 = float(torch.quantile(residual / local_step, 0.5))
            residual90 = float(torch.quantile(residual / local_step, 0.9))
        else:
            residual_scale = residual50 = residual90 = float("nan")
        diagnostics.append(
            (float(metric_scale), len(overlap), residual_scale, residual50, residual90)
        )
        for frame, center in zip(window_ids, predicted):
            if frame not in centers:
                centers[frame] = center
    ordered_centers = torch.stack([centers[int(frame)] for frame in frame_ids.tolist()])
    candidate = initial_poses.clone()
    candidate[:, :3, 3] = -torch.einsum(
        "sij,sj->si", candidate[:, :3, :3], ordered_centers
    )
    return candidate, diagnostics


def _retriangulate_probe_graph(graph, frame_ids, poses, inactive_observation_ids):
    """Rebuild probe landmarks from the supplied poses and raw graph tracks."""
    graph.inactive_observation_ids = set(inactive_observation_ids)
    ordered_ids = [int(frame) for frame in frame_ids.tolist()]
    for frame_id, pose in zip(ordered_ids, poses):
        graph.poses[frame_id] = pose.clone()
    graph._triangulate_global(ordered_ids, poses)
    return graph.full_view()


@torch.no_grad()
def probe(data_h5, device="auto", bearing_iterations=15, compare=False):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = load_frontend_cache(data_h5)
    packets = payload["packets"]
    windows = [list(map(int, packet["frame_ids"].tolist())) for packet in packets]
    pi3_windows = [
        (packet["frame_ids"], packet["pi3_T_WCs"], packet["metric_scale"])
        for packet in packets
    ]
    frames = SimpleNamespace(keyframes=set(), dense={}, anchors={}, track_ids={})
    restore_frames(frames, payload["frames"])
    frames.dense.clear()
    del payload

    graph = FactorGraph(frames)
    for packet in packets:
        graph_packet = {
            key: packet[key]
            for key in ("kind", "frame_ids", "poses", "K", "parts", "edges")
        }
        graph.add_factors(packet_to_device(graph_packet, device))
    del packets

    raw_point_frames = _point_frames(graph)
    initial_inactive_observation_ids = set(graph.inactive_observation_ids)
    selected_edges = _mst_edges(graph.edges)
    edge_norms = [float(edge[2][:3, 3].norm()) for edge in selected_edges]
    edge_gaps = [abs(int(edge[1]) - int(edge[0])) for edge in selected_edges]
    unit_edges = sum(abs(norm - 1.0) <= 1.0e-3 for norm in edge_norms)
    long_edges = [edge for edge in selected_edges if abs(int(edge[1]) - int(edge[0])) > 1]
    print(
        f"PROBE device={device} frames={len(graph.poses)} "
        f"packets={len(windows)} edges={len(graph.edges)}"
    )
    print(
        f"MST selected={len(selected_edges)} unit_t={unit_edges}/{len(selected_edges)} "
        f"long_gap={len(long_edges)} median_gap={_summary(edge_gaps)[1]:.6g} "
        f"max_gap={max(edge_gaps)}"
    )
    if long_edges:
        examples = sorted(
            long_edges,
            key=lambda edge: abs(int(edge[1]) - int(edge[0])),
            reverse=True,
        )[:8]
        print(
            "MST_LONG "
            + " ".join(
                f"{int(edge[0])}-{int(edge[1])}"
                f"(g{abs(int(edge[1]) - int(edge[0]))},"
                f"n{float(edge[2][:3, 3].norm()):.4g})"
                for edge in examples
            )
        )

    graph.initialize_global()
    triangulated_point_frames = _point_frames(graph, active_only=True)
    for cut in _cuts(windows):
        raw = _strong_bridges(raw_point_frames, cut)
        triangulated = _strong_bridges(triangulated_point_frames, cut)
        angles = graph.triangulation_angles[triangulated]
        angles = angles[torch.isfinite(angles)]
        angle50 = float(torch.quantile(angles, 0.5)) if angles.numel() else float("nan")
        retain = len(triangulated) / max(len(raw), 1)
        print(
            f"BRIDGE cut={cut} raw_2x2={len(raw)} tri_2x2={len(triangulated)} "
            f"retain={retain:.4f} angle50_deg={angle50:.4g}"
        )

    view = graph.full_view()
    camera_counts = torch.bincount(view["ii"], minlength=view["poses"].shape[0])
    camera_weights = torch.zeros(
        view["poses"].shape[0], device=view["weight"].device, dtype=view["weight"].dtype
    )
    camera_weights.index_add_(0, view["ii"], view["weight"])
    frame_lookup = {
        int(frame): index for index, frame in enumerate(view["frame_ids"].tolist())
    }
    for cut in _cuts(windows):
        entries = []
        for frame in range(cut - 2, cut + 3):
            if frame in frame_lookup:
                local = frame_lookup[frame]
                entries.append(
                    f"{frame}={int(camera_counts[local])}/{float(camera_weights[local]):.4g}"
                )
        print(f"OBS cut={cut} count/sum_weight " + " ".join(entries))

    initial_poses = view["poses"].clone()
    fixed = _fixed_camera_mask(initial_poses, [0])
    stage_fixed, active = _stage_fixed_camera_mask(
        fixed, view["ii"], initial_poses.shape[0]
    )
    gauge_frame_id = view.get("scale_gauge_frame_id")
    gauge_camera = (
        None if gauge_frame_id is None else frame_lookup[int(gauge_frame_id)]
    )
    scale_gauge = _select_scale_gauge(
        camera_centers(initial_poses),
        fixed if gauge_camera is not None else stage_fixed,
        active,
        camera=gauge_camera,
    )
    if scale_gauge is not None:
        root, camera, _, target = scale_gauge
        print(
            f"BASELINE_SCALE_GAUGE initialization=essential "
            f"root={int(view['frame_ids'][root])} "
            f"camera={int(view['frame_ids'][camera])} target={float(target):.6g}"
        )
    bearing_poses, bearing_points = _bearing_adjust(
        initial_poses.clone(),
        view["points"].clone(),
        view,
        fixed,
        bearing_iterations,
        scale_gauge=scale_gauge,
    )
    trajectories = []
    for window in windows:
        denominator = max(len(window) - 1, 1)
        mst_step = float(_window_path(initial_poses, view["frame_ids"], window)) / denominator
        bearing_step = float(_window_path(bearing_poses, view["frame_ids"], window)) / denominator
        trajectories.append((mst_step, bearing_step))
    first_mst, first_bearing = trajectories[0]
    for index, (window, (mst_step, bearing_step)) in enumerate(
        zip(windows, trajectories)
    ):
        print(
            f"TRAJ win={index} frames={window[0]}..{window[-1]} "
            f"mst_step={mst_step:.6g} mst_rel={mst_step / first_mst:.6g} "
            f"bearing_step={bearing_step:.6g} "
            f"bearing_rel={bearing_step / first_bearing:.6g} "
            f"bearing/mst={bearing_step / mst_step:.6g}"
        )

    initial_centers = camera_centers(initial_poses)
    bearing_centers = camera_centers(bearing_poses)
    ordered = torch.argsort(view["frame_ids"])
    ordered_frames = view["frame_ids"][ordered].tolist()
    initial_steps = torch.linalg.vector_norm(
        initial_centers[ordered][1:] - initial_centers[ordered][:-1], dim=-1
    )
    bearing_steps = torch.linalg.vector_norm(
        bearing_centers[ordered][1:] - bearing_centers[ordered][:-1], dim=-1
    )
    ratios = bearing_steps / initial_steps.clamp_min(1.0e-12)
    candidates = [
        (float(ratio), int(source), int(target))
        for ratio, source, target in zip(
            ratios.tolist(), ordered_frames[:-1], ordered_frames[1:]
        )
        if int(target) == int(source) + 1
    ]
    print(
        "WORST_STEPS "
        + " ".join(
            f"{source}-{target}={ratio:.4g}"
            for ratio, source, target in sorted(candidates)[:8]
        )
    )
    if not compare:
        return

    normalized_poses, normalized_points = _normalized_bearing_adjust(
        initial_poses.clone(),
        view["points"].clone(),
        view,
        fixed,
        bearing_iterations,
        scale_gauge=scale_gauge,
    )
    methods = (
        ("mst", initial_poses, view["points"]),
        ("linear", bearing_poses, bearing_points),
        ("unit_candidate", normalized_poses, normalized_points),
    )
    for name, poses, points in methods:
        angle50, angle90, positive = _ray_stats(poses, points, view)
        print(
            f"AB_RAY method={name} angle50_deg={angle50:.6g} "
            f"angle90_deg={angle90:.6g} positive_depth={positive:.6g}"
        )
    method_steps = {
        name: [
            float(_window_path(poses, view["frame_ids"], window)) / (len(window) - 1)
            for window in windows
        ]
        for name, poses, _ in methods
    }
    for index, window in enumerate(windows):
        mst = method_steps["mst"][index]
        linear = method_steps["linear"][index]
        unit = method_steps["unit_candidate"][index]
        print(
            f"AB_WIN win={index} frames={window[0]}..{window[-1]} "
            f"mst_rel={mst / method_steps['mst'][0]:.6g} "
            f"linear_rel={linear / method_steps['linear'][0]:.6g} "
            f"unit_rel={unit / method_steps['unit_candidate'][0]:.6g} "
            f"linear/mst={linear / mst:.6g} unit/mst={unit / mst:.6g}"
        )
    for cut in _cuts(windows):
        mst_cut = _pair_step(initial_poses, view["frame_ids"], cut - 1, cut)
        linear_cut = _pair_step(bearing_poses, view["frame_ids"], cut - 1, cut)
        unit_cut = _pair_step(normalized_poses, view["frame_ids"], cut - 1, cut)
        print(
            f"AB_CUT cut={cut} linear/mst={linear_cut / mst_cut:.6g} "
            f"unit/mst={unit_cut / mst_cut:.6g} "
            f"mst/local={_cut_local_ratio(initial_poses, view['frame_ids'], cut):.6g} "
            f"linear/local={_cut_local_ratio(bearing_poses, view['frame_ids'], cut):.6g} "
            f"unit/local={_cut_local_ratio(normalized_poses, view['frame_ids'], cut):.6g}"
        )

    print(
        f"BA_PROBE backend=native bearing={bearing_iterations} ba1=20 ba2=10"
    )
    ba1_poses, ba1_points = _bundle_adjust(
        bearing_poses.clone(),
        bearing_points.clone(),
        view,
        fixed,
        20,
        scale_gauge=scale_gauge,
    )
    first_filter = filter_reprojection_observations(view, ba1_poses, ba1_points)
    first_ids = torch.nonzero(
        first_filter["observation_inliers"], as_tuple=False
    ).squeeze(-1)
    ba2_poses, ba2_points = _bundle_adjust(
        ba1_poses.clone(),
        ba1_points.clone(),
        view,
        fixed,
        10,
        observation_ids=first_ids,
        scale_gauge=scale_gauge,
    )
    final_filter = filter_reprojection_observations(
        view, ba2_poses, ba2_points, observation_ids=first_ids
    )
    ba_stages = (
        ("bearing", bearing_poses, bearing_points),
        ("ba1", ba1_poses, ba1_points),
        ("ba2", ba2_poses, ba2_points),
    )
    ba_steps = {}
    for name, poses, points in ba_stages:
        angle50, angle90, positive = _ray_stats(poses, points, view)
        ba_steps[name] = [
            float(_window_path(poses, view["frame_ids"], window))
            / (len(window) - 1)
            for window in windows
        ]
        print(
            f"BA_RAY stage={name} angle50_deg={angle50:.6g} "
            f"angle90_deg={angle90:.6g} positive_depth={positive:.6g}"
        )
    for index, window in enumerate(windows):
        bearing = ba_steps["bearing"][index]
        ba1 = ba_steps["ba1"][index]
        ba2 = ba_steps["ba2"][index]
        print(
            f"BA_WIN win={index} frames={window[0]}..{window[-1]} "
            f"bearing_rel={bearing / ba_steps['bearing'][0]:.6g} "
            f"ba1_rel={ba1 / ba_steps['ba1'][0]:.6g} "
            f"ba2_rel={ba2 / ba_steps['ba2'][0]:.6g} "
            f"ba1/bearing={ba1 / bearing:.6g} ba2/ba1={ba2 / ba1:.6g}"
        )
    for cut in _cuts(windows):
        print(
            f"BA_CUT cut={cut} "
            f"bearing/local={_cut_local_ratio(bearing_poses, view['frame_ids'], cut):.6g} "
            f"ba1/local={_cut_local_ratio(ba1_poses, view['frame_ids'], cut):.6g} "
            f"ba2/local={_cut_local_ratio(ba2_poses, view['frame_ids'], cut):.6g}"
        )
    for name, result in (("first", first_filter), ("final", final_filter)):
        point_frames = _view_point_frames(view, result["observation_inliers"])
        bridges = " ".join(
            f"cut{cut}={len(_strong_bridges(point_frames, cut))}"
            for cut in _cuts(windows)
        )
        print(
            f"BA_FILTER stage={name} "
            f"obs={int(result['observation_inliers'].sum())}/{view['ii'].numel()} "
            f"points={int(result['point_inliers'].sum())}/{view['points'].shape[0]} "
            f"{bridges}"
        )

    direction_poses, near_null, eig_gap = _translation_direction_average(
        initial_poses, view["frame_ids"], graph.edges
    )
    mst_first = method_steps["mst"][0]
    direction_first = float(
        _window_path(direction_poses, view["frame_ids"], windows[0])
    ) / (len(windows[0]) - 1)
    direction_centers = camera_centers(direction_poses) * (mst_first / direction_first)
    direction_poses[:, :3, 3] = -torch.einsum(
        "sij,sj->si", direction_poses[:, :3, :3], direction_centers
    )
    mst_angle50, mst_angle90, mst_positive = _edge_direction_stats(
        initial_poses, view["frame_ids"], graph.edges
    )
    dir_angle50, dir_angle90, dir_positive = _edge_direction_stats(
        direction_poses, view["frame_ids"], graph.edges
    )
    print(
        f"DIRAVG_CANDIDATE near_null={near_null} eig_gap={eig_gap:.6g} "
        f"mst_angle50/90={mst_angle50:.4g}/{mst_angle90:.4g} "
        f"dir_angle50/90={dir_angle50:.4g}/{dir_angle90:.4g} "
        f"mst_positive={mst_positive:.4f} dir_positive={dir_positive:.4f}"
    )
    direction_steps = [
        float(_window_path(direction_poses, view["frame_ids"], window))
        / (len(window) - 1)
        for window in windows
    ]
    for index, window in enumerate(windows):
        print(
            f"DIRAVG_WIN win={index} frames={window[0]}..{window[-1]} "
            f"mst_rel={method_steps['mst'][index] / mst_first:.6g} "
            f"dir_rel={direction_steps[index] / direction_steps[0]:.6g}"
        )
    for cut in _cuts(windows):
        print(
            f"DIRAVG_CUT cut={cut} "
            f"mst/local={_cut_local_ratio(initial_poses, view['frame_ids'], cut):.6g} "
            f"dir/local={_cut_local_ratio(direction_poses, view['frame_ids'], cut):.6g}"
        )

    pi3_poses, pi3_diagnostics = _pi3_center_candidate(
        initial_poses, view["frame_ids"], pi3_windows
    )
    for index, (scale, overlap, residual_scale, residual50, residual90) in enumerate(
        pi3_diagnostics
    ):
        print(
            f"PI3_CENTER_CHUNK win={index} scale={scale:.6g} overlap={overlap} "
            f"residual_scale={residual_scale:.6g} "
            f"overlap50/local={residual50:.6g} overlap90/local={residual90:.6g}"
        )
    pi3_angle50, pi3_angle90, pi3_positive = _edge_direction_stats(
        pi3_poses, view["frame_ids"], graph.edges
    )
    print(
        f"PI3_CENTER_EDGE angle50/90={pi3_angle50:.4g}/{pi3_angle90:.4g} "
        f"positive={pi3_positive:.4f}"
    )
    pi3_steps = [
        float(_window_path(pi3_poses, view["frame_ids"], window))
        / (len(window) - 1)
        for window in windows
    ]
    for index, window in enumerate(windows):
        print(
            f"PI3_CENTER_WIN win={index} frames={window[0]}..{window[-1]} "
            f"linear_rel={method_steps['linear'][index] / method_steps['linear'][0]:.6g} "
            f"pi3_rel={pi3_steps[index] / pi3_steps[0]:.6g}"
        )
    for cut in _cuts(windows):
        print(
            f"PI3_CENTER_CUT cut={cut} "
            f"linear/local={_cut_local_ratio(bearing_poses, view['frame_ids'], cut):.6g} "
            f"pi3/local={_cut_local_ratio(pi3_poses, view['frame_ids'], cut):.6g}"
        )

    pi3_view = _retriangulate_probe_graph(
        graph,
        view["frame_ids"],
        pi3_poses,
        initial_inactive_observation_ids,
    )
    pi3_angle50, pi3_angle90, pi3_positive = _ray_stats(
        pi3_view["poses"], pi3_view["points"], pi3_view
    )
    print(
        f"PI3_TRI points={pi3_view['points'].shape[0]}/{graph.point_count} "
        f"obs={pi3_view['ii'].numel()}/{graph.next_observation_id} "
        f"angle50_deg={pi3_angle50:.6g} angle90_deg={pi3_angle90:.6g} "
        f"positive_depth={pi3_positive:.6g}"
    )
    pi3_point_frames = _point_frames(graph, active_only=True)
    for cut in _cuts(windows):
        raw = _strong_bridges(raw_point_frames, cut)
        triangulated = _strong_bridges(pi3_point_frames, cut)
        print(
            f"PI3_TRI_BRIDGE cut={cut} raw_2x2={len(raw)} "
            f"tri_2x2={len(triangulated)} "
            f"retain={len(triangulated) / max(len(raw), 1):.6g}"
        )

    pi3_fixed = _fixed_camera_mask(pi3_view["poses"], [0])
    pi3_stage_fixed, pi3_active = _stage_fixed_camera_mask(
        pi3_fixed, pi3_view["ii"], pi3_view["poses"].shape[0]
    )
    pi3_scale_gauge = _select_scale_gauge(
        camera_centers(pi3_view["poses"]),
        pi3_fixed if gauge_camera is not None else pi3_stage_fixed,
        pi3_active,
        camera=gauge_camera,
    )
    if pi3_scale_gauge is not None:
        root, camera, _, target = pi3_scale_gauge
        print(
            f"PI3_SCALE_GAUGE initialization=pi3_metric "
            f"root={int(pi3_view['frame_ids'][root])} "
            f"camera={int(pi3_view['frame_ids'][camera])} "
            f"target={float(target):.6g}"
        )
    pi3_bearing_poses, pi3_bearing_points = _bearing_adjust(
        pi3_view["poses"].clone(),
        pi3_view["points"].clone(),
        pi3_view,
        pi3_fixed,
        bearing_iterations,
        scale_gauge=pi3_scale_gauge,
    )
    pi3_ba1_poses, pi3_ba1_points = _bundle_adjust(
        pi3_bearing_poses.clone(),
        pi3_bearing_points.clone(),
        pi3_view,
        pi3_fixed,
        20,
        scale_gauge=pi3_scale_gauge,
    )
    pi3_first_filter = filter_reprojection_observations(
        pi3_view, pi3_ba1_poses, pi3_ba1_points
    )
    pi3_first_ids = torch.nonzero(
        pi3_first_filter["observation_inliers"], as_tuple=False
    ).squeeze(-1)
    pi3_ba2_poses, pi3_ba2_points = _bundle_adjust(
        pi3_ba1_poses.clone(),
        pi3_ba1_points.clone(),
        pi3_view,
        pi3_fixed,
        10,
        observation_ids=pi3_first_ids,
        scale_gauge=pi3_scale_gauge,
    )
    pi3_final_filter = filter_reprojection_observations(
        pi3_view,
        pi3_ba2_poses,
        pi3_ba2_points,
        observation_ids=pi3_first_ids,
    )
    pi3_stages = (
        ("fixed", pi3_view["poses"], pi3_view["points"]),
        ("bearing", pi3_bearing_poses, pi3_bearing_points),
        ("ba1", pi3_ba1_poses, pi3_ba1_points),
        ("ba2", pi3_ba2_poses, pi3_ba2_points),
    )
    pi3_stage_steps = {}
    for name, poses, points in pi3_stages:
        angle50, angle90, positive = _ray_stats(poses, points, pi3_view)
        pi3_stage_steps[name] = [
            float(_window_path(poses, pi3_view["frame_ids"], window))
            / (len(window) - 1)
            for window in windows
        ]
        print(
            f"PI3_OPT_RAY stage={name} angle50_deg={angle50:.6g} "
            f"angle90_deg={angle90:.6g} positive_depth={positive:.6g}"
        )
    for index, window in enumerate(windows):
        print(
            f"PI3_OPT_WIN win={index} frames={window[0]}..{window[-1]} "
            + " ".join(
                f"{name}_rel={pi3_stage_steps[name][index] / pi3_stage_steps[name][0]:.6g}"
                for name, _, _ in pi3_stages
            )
        )
    for cut in _cuts(windows):
        print(
            f"PI3_OPT_CUT cut={cut} "
            + " ".join(
                f"{name}/local={_cut_local_ratio(poses, pi3_view['frame_ids'], cut):.6g}"
                for name, poses, _ in pi3_stages
            )
        )
    for name, result in (
        ("first", pi3_first_filter),
        ("final", pi3_final_filter),
    ):
        point_frames = _view_point_frames(pi3_view, result["observation_inliers"])
        bridges = " ".join(
            f"cut{cut}={len(_strong_bridges(point_frames, cut))}"
            for cut in _cuts(windows)
        )
        print(
            f"PI3_OPT_FILTER stage={name} "
            f"obs={int(result['observation_inliers'].sum())}/{pi3_view['ii'].numel()} "
            f"points={int(result['point_inliers'].sum())}/{pi3_view['points'].shape[0]} "
            f"{bridges}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_h5")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--bearing-iterations", type=int, default=15)
    parser.add_argument(
        "--ab", action="store_true", help="compare probe-only scale candidates"
    )
    args = parser.parse_args()
    probe(args.data_h5, args.device, args.bearing_iterations, args.ab)


if __name__ == "__main__":
    main()
