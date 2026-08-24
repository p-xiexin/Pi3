"""Read exact evaluation windows without mutating source dataset instances."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MethodType
from typing import Sequence


@dataclass(frozen=True)
class LoadedWindow:
    views: list[dict]
    frame_instances: tuple[str, ...]


class WindowAdapter:
    """Base interface for exact window reads through native dataset __getitem__."""

    def load(
        self,
        dataset,
        sequence_index: int,
        resolution_index: int,
        positions: Sequence[int],
    ) -> LoadedWindow:
        raise NotImplementedError

    @staticmethod
    def _validate_views(
        views,
        expected_count: int,
        expected_sequence: str,
    ) -> None:
        if len(views) != expected_count:
            raise RuntimeError(
                f"{expected_sequence} returned {len(views)} views for "
                f"{expected_count} requested positions"
            )
        loaded_sequences = [str(view.get("sequence", "")) for view in views]
        if any(sequence != expected_sequence for sequence in loaded_sequences):
            raise RuntimeError(
                f"dataset retry replaced {expected_sequence} with "
                f"{loaded_sequences}"
            )
        loaded_indices = [view.get("idx") for view in views]
        if any(
            index is not None and int(index[0]) != 0
            for index in loaded_indices
        ):
            raise RuntimeError(
                f"isolated dataset returned another sequence index "
                f"{loaded_indices}"
            )


class SampledRecordWindowAdapter(WindowAdapter):
    """Use a fixed sampler on an isolated shallow copy of a record dataset."""

    def load(
        self,
        dataset,
        sequence_index: int,
        resolution_index: int,
        positions: Sequence[int],
    ) -> LoadedWindow:
        positions = tuple(int(position) for position in positions)
        sequence_index = int(sequence_index)
        source_sequence = dataset.sequences[sequence_index]
        expected_sequence = str(source_sequence)

        window_dataset = copy.copy(dataset)
        window_dataset.records = [dataset.records[sequence_index]]
        window_dataset.sequences = [source_sequence]
        window_dataset.num_imgs = {
            source_sequence: int(dataset.num_imgs[source_sequence])
        }
        window_dataset.frame_num = len(positions)
        window_dataset.shuffle = False

        def fixed_positions(_self, *_args, **_kwargs):
            return list(positions)

        window_dataset._sample_positions = MethodType(
            fixed_positions, window_dataset
        )
        views = window_dataset[(0, int(resolution_index), len(positions))]
        self._validate_views(views, len(positions), expected_sequence)
        return LoadedWindow(
            views=views,
            frame_instances=tuple(str(view["instance"]) for view in views),
        )


class SequenceListWindowAdapter(WindowAdapter):
    """Slice ADS-style per-sequence lists on a disposable dataset copy."""

    _SEQUENCE_ATTRIBUTES = (
        "sequences_files",
        "sequences_extrinsic",
        "sequences_intrinsic",
        "sequences_wh",
        "sequences_rel_pose",
        "sequences_cam_extrinsic",
    )

    def load(
        self,
        dataset,
        sequence_index: int,
        resolution_index: int,
        positions: Sequence[int],
    ) -> LoadedWindow:
        positions = tuple(int(position) for position in positions)
        sequence_index = int(sequence_index)
        source_sequence = dataset.sequences[sequence_index]
        expected_sequence = (
            str(source_sequence)
            .replace("\\", "/")
            .rstrip("/")
            .rsplit("/", 1)[-1]
        )

        window_dataset = copy.copy(dataset)
        window_dataset.sequences = [source_sequence]
        for name in self._SEQUENCE_ATTRIBUTES:
            values = getattr(dataset, name, None)
            if values is None:
                continue
            source = values[sequence_index]
            selected = (
                None
                if source is None
                else [source[position] for position in positions]
            )
            setattr(window_dataset, name, [selected])

        selected_files = window_dataset.sequences_files[0]
        window_dataset.num_imgs = {source_sequence: len(selected_files)}
        window_dataset.frame_num = len(positions)
        window_dataset.shuffle = False
        for name in ("trans_step", "rot_step"):
            if hasattr(window_dataset, name):
                setattr(window_dataset, name, 0)

        views = window_dataset[(0, int(resolution_index), len(positions))]
        self._validate_views(views, len(positions), expected_sequence)
        frame_instances = tuple(str(view.get("label", "")) for view in views)
        if any(not identity for identity in frame_instances):
            raise RuntimeError(
                f"{expected_sequence} did not return labels for all ADS frames"
            )
        for view, identity in zip(views, frame_instances):
            view["instance"] = identity
        return LoadedWindow(views=views, frame_instances=frame_instances)


def create_window_adapter(dataset) -> WindowAdapter:
    """Select a supported adapter from dataset capabilities."""

    if callable(getattr(dataset, "_sample_positions", None)) and hasattr(
        dataset, "records"
    ):
        return SampledRecordWindowAdapter()
    if hasattr(dataset, "sequences_files"):
        return SequenceListWindowAdapter()
    raise TypeError(
        f"{dataset.__class__.__name__} has no evaluation window adapter. "
        "Supported datasets expose records plus _sample_positions, or "
        "ADS-style sequences_files."
    )


__all__ = [
    "LoadedWindow",
    "SampledRecordWindowAdapter",
    "SequenceListWindowAdapter",
    "WindowAdapter",
    "create_window_adapter",
]
