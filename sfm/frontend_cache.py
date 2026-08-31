"""Minimal staged HDF5 caches for SfM frontend measurements."""

import io
from pathlib import Path

import h5py
import numpy as np
import torch


CACHE_FORMAT = "geometric_sfm_v7_pi3_centers"


def _map_tensors(value, transform):
    if torch.is_tensor(value):
        return transform(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, transform) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tensors(item, transform) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, transform) for item in value)
    return value


def packet_to_cpu(packet):
    """Detach one packet into the portable representation stored in HDF5."""
    return _map_tensors(packet, lambda tensor: tensor.detach().cpu())


def packet_to_device(packet, device):
    """Move a cached packet onto the active graph device."""
    return _map_tensors(packet, lambda tensor: tensor.to(device))


def snapshot_frames(frames):
    """Capture only the FrameStore state required after frontend inference."""
    return {
        "keyframes": sorted(frames.keyframes),
        "dense": packet_to_cpu(frames.dense),
        "anchors": packet_to_cpu(frames.anchors),
        "track_ids": packet_to_cpu(frames.track_ids),
        "next_track_id": int(frames.next_track_id),
    }


def restore_frames(frames, state):
    """Restore a complete FrameStore snapshot from a valid cache."""
    frames.keyframes = set(map(int, state["keyframes"]))
    frames.dense = {int(key): value for key, value in state["dense"].items()}
    frames.anchors = {int(key): value for key, value in state["anchors"].items()}
    frames.track_ids = {
        int(key): value for key, value in state["track_ids"].items()
    }
    frames.next_track_id = int(state["next_track_id"])


def _load_payload(path):
    """Load one opaque payload from an existing HDF5 stage cache."""
    with h5py.File(path, "r") as handle:
        raw = handle["payload"][()].tobytes()
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def _save_payload(path, payload):
    """Atomically replace one complete HDF5 stage cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    encoded = np.frombuffer(buffer.getbuffer(), dtype=np.uint8)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("payload", data=encoded, compression="lzf")
        handle.flush()
    temporary.replace(path)


def load_frontend_cache(path):
    """Load sliding packets and FrameStore state from data.h5."""
    payload = _load_payload(path)
    if payload.get("format") != CACHE_FORMAT:
        raise RuntimeError(
            f"frontend cache {path} uses an incompatible frontend protocol; "
            "delete data.h5 and data_loop.h5"
        )
    frames = payload.get("frames", {})
    missing = {"track_ids", "next_track_id"}.difference(frames)
    if missing:
        raise RuntimeError(
            f"frontend cache {path} has no stable track ID registry; delete it"
        )
    return payload


def save_frontend_cache(path, packets, frames):
    """Atomically save the completed sliding-window stage to data.h5."""
    if any(packet.get("kind", "sliding") != "sliding" for packet in packets):
        raise ValueError("data.h5 accepts sliding packets only")
    if any(
        "track_ids" not in part
        for packet in packets
        for part in packet["parts"]
    ):
        raise ValueError("data.h5 requires stable track IDs in every packet part")
    if any(
        "pi3_T_WCs" not in packet or "metric_scale" not in packet
        for packet in packets
    ):
        raise ValueError("data.h5 requires raw Pi3 window poses and metric scale")
    _save_payload(
        path,
        {
            "format": CACHE_FORMAT,
            "packets": [packet_to_cpu(packet) for packet in packets],
            "frames": snapshot_frames(frames),
        },
    )


def load_loop_cache(path):
    """Load only the incremental loop packets from data_loop.h5."""
    payload = _load_payload(path)
    if payload.get("format") != CACHE_FORMAT:
        raise RuntimeError(
            f"loop cache {path} uses an incompatible frontend protocol; "
            "delete data.h5 and data_loop.h5"
        )
    packets = payload["packets"]
    if any(packet.get("kind") != "loop" for packet in packets):
        raise ValueError("data_loop.h5 contains a non-loop packet")
    if any(
        "track_ids" not in part
        for packet in packets
        for part in packet["parts"]
    ):
        raise RuntimeError(
            f"loop cache {path} has no stable track IDs; delete it"
        )
    return packets


def save_loop_cache(path, packets):
    """Atomically save one complete incremental loop stage."""
    if any(packet.get("kind") != "loop" for packet in packets):
        raise ValueError("data_loop.h5 accepts loop packets only")
    if any(
        "track_ids" not in part
        for packet in packets
        for part in packet["parts"]
    ):
        raise ValueError("data_loop.h5 requires stable track IDs in every packet part")
    _save_payload(
        path,
        {
            "format": CACHE_FORMAT,
            "packets": [packet_to_cpu(packet) for packet in packets],
        },
    )


__all__ = [
    "CACHE_FORMAT",
    "load_frontend_cache",
    "load_loop_cache",
    "packet_to_cpu",
    "packet_to_device",
    "restore_frames",
    "save_frontend_cache",
    "save_loop_cache",
    "snapshot_frames",
]
