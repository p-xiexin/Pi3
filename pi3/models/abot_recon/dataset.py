"""Thin adapters from existing Pi3 datasets to the ABot-Recon sequence contract.

The parent datasets remain responsible for image, depth, intrinsics and pose IO.
These adapters only disable view shuffling, establish a deterministic stream
order and attach metadata consumed by ``prepare_abot_batch``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, MutableMapping

from datasets.blendmvs_dataset import BlendedMVSPi3XDataset
from datasets.kitti_dataset import KITTIPi3XDataset
from datasets.scannet_dataset import ScannetDataset
from datasets.tartanair_dataset import TarTanAirDataset
from datasets.waymo_processed_dataset import WaymoPi3XDataset


def _numeric_frame_id(view: Mapping[str, Any]) -> int:
    """Extract a stable integer frame id from a Pi3 view dictionary."""

    value = view.get("frame_id", view.get("instance"))
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"(\d+)(?!.*\d)", str(value))
        if match is None:
            raise ValueError(
                f"ABot-Recon cannot recover a numeric frame id from {value!r}"
            )
        return int(match.group(1))


class _ABotReconDatasetAdapter:
    """Mixin that adds the ordered-stream metadata shared by all adapters."""

    abot_dataset_label = "ABotRecon"
    sort_parent_views = False
    use_pseudo_sequence_index = False

    def __init__(
        self,
        *args,
        shuffle: bool = False,
        random_sample_thres: float = 0.0,
        min_sequence_frames: int | None = None,
        **kwargs,
    ):
        if shuffle:
            raise ValueError(
                f"{type(self).__name__} requires shuffle=false"
            )
        if float(random_sample_thres) != 0.0:
            raise ValueError(
                f"{type(self).__name__} requires random_sample_thres=0.0"
            )
        super().__init__(
            *args,
            shuffle=False,
            random_sample_thres=0.0,
            **kwargs,
        )
        self.min_sequence_frames = (
            None if min_sequence_frames is None else int(min_sequence_frames)
        )
        if self.min_sequence_frames is not None:
            self._filter_short_sequences(self.min_sequence_frames)
        self.dataset_label = self.abot_dataset_label

    @staticmethod
    def _record_frame_count(record: Mapping[str, Any]) -> int:
        for key in ("frames", "frame_ids"):
            if key in record:
                return len(record[key])
        raise ValueError(
            "min_sequence_frames requires records containing frames or frame_ids"
        )

    def _filter_short_sequences(self, minimum_frames: int) -> None:
        """Remove records that cannot produce a complete fixed-stride clip."""

        if minimum_frames <= 0:
            raise ValueError("min_sequence_frames must be positive")
        if not hasattr(self, "records"):
            raise ValueError(
                f"{type(self).__name__} does not expose sequence records for "
                "min_sequence_frames filtering"
            )

        frame_step = int(getattr(self, "frame_step", 1))
        required_span = (minimum_frames - 1) * frame_step + 1
        records = list(self.records)
        self.records = [
            record for record in records
            if self._record_frame_count(record) >= required_span
        ]
        if not self.records:
            raise ValueError(
                f"{type(self).__name__} has no sequence with at least "
                f"{required_span} source frames"
            )

        if hasattr(self, "sequences"):
            self.sequences = [record["sequence_id"] for record in self.records]
        if hasattr(self, "num_imgs"):
            self.num_imgs = {
                record["sequence_id"]: self._record_frame_count(record)
                for record in self.records
            }
        print(
            f"[{self.abot_dataset_label}] Kept {len(self.records)}/{len(records)} "
            f"sequences for min_sequence_frames={minimum_frames}, "
            f"frame_step={frame_step}, required_span={required_span}",
            flush=True,
        )

    def _get_views(self, index, resolution, rng, *args, **kwargs):
        views = super()._get_views(index, resolution, rng, *args, **kwargs)
        if self.sort_parent_views:
            views.sort(key=_numeric_frame_id)
            ordered_ids = [_numeric_frame_id(view) for view in views]
            if isinstance(getattr(self, "this_views_info", None), dict):
                for key in ("idxs", "pairs"):
                    if key in self.this_views_info:
                        self.this_views_info[key] = ordered_ids

        for temporal_index, view in enumerate(views):
            self._annotate_view(view, temporal_index)
        return views

    def _annotate_view(
        self,
        view: MutableMapping[str, Any],
        temporal_index: int,
    ) -> None:
        source_frame_id = _numeric_frame_id(view)
        if self.use_pseudo_sequence_index:
            view["source_frame_id"] = source_frame_id
            view["frame_id"] = temporal_index
        else:
            view["frame_id"] = source_frame_id

        view["temporal_index"] = temporal_index
        view["dataset"] = self.abot_dataset_label
        view["abot_recon_ordered"] = True


class KITTIABotReconDataset(_ABotReconDatasetAdapter, KITTIPi3XDataset):
    """Ordered KITTI Raw clips used for the current Stage I smoke test."""

    abot_dataset_label = "KITTIABotRecon"


class TartanAirABotReconDataset(_ABotReconDatasetAdapter, TarTanAirDataset):
    """TartanAir clips sorted by their original temporal frame ids."""

    abot_dataset_label = "TartanAirABotRecon"
    sort_parent_views = True


class ScanNetABotReconDataset(_ABotReconDatasetAdapter, ScannetDataset):
    """ScanNet clips sorted by their original temporal frame ids."""

    abot_dataset_label = "ScanNetABotRecon"
    sort_parent_views = True


class WaymoABotReconDataset(_ABotReconDatasetAdapter, WaymoPi3XDataset):
    """Ordered Waymo clips using the parent's sequence and depth processing."""

    abot_dataset_label = "WaymoABotRecon"


class BlendedMVSABotReconDataset(
    _ABotReconDatasetAdapter,
    BlendedMVSPi3XDataset,
):
    """BlendedMVS camera-graph paths represented as ordered pseudo-sequences."""

    abot_dataset_label = "BlendedMVSABotRecon"
    use_pseudo_sequence_index = True


__all__ = [
    "KITTIABotReconDataset",
    "TartanAirABotReconDataset",
    "ScanNetABotReconDataset",
    "WaymoABotReconDataset",
    "BlendedMVSABotReconDataset",
]
