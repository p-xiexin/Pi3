"""ABot-Recon sampling over an existing ordered Pi3 dataset."""

from __future__ import annotations

from math import ceil
import re

import numpy as np

from datasets.base.easy_dataset import EasyDataset


def _sequence_length(dataset, index):
    """Read the small length contract shared by Pi3 sequence datasets."""

    if hasattr(dataset, "abot_sequence_length"):
        return int(dataset.abot_sequence_length(index))
    if hasattr(dataset, "sequence_lengths"):
        lengths = dataset.sequence_lengths
        return int(lengths(index) if callable(lengths) else lengths[index])
    if hasattr(dataset, "records"):
        record = dataset.records[index]
        for key in ("frames", "frame_ids", "view_ids", "sample_tokens"):
            if key in record:
                return len(record[key])
    sequence = dataset.sequences[index]
    if hasattr(dataset, "num_imgs"):
        length = int(dataset.num_imgs[sequence])
        if hasattr(dataset, "invalid_list"):
            length -= sum(frame < length for frame in dataset.invalid_list[sequence])
        return length
    return len(dataset.frames[sequence])


def _minimum_length(target, minimum_frames, minimum_ratio):
    return min(target, max(minimum_frames, ceil(target * minimum_ratio)))


def _frame_number(view):
    value = view.get("source_frame_id", view.get("frame_id", view.get("instance")))
    try:
        return float(value)
    except (TypeError, ValueError):
        return int(re.search(r"(\d+)(?!.*\d)", str(value)).group(1))


def adaptive_forward(source_length, target_length, step_range, rng):
    """Sample a stride after clipping its upper bound to the feasible span."""

    if target_length == 1:
        return 1
    feasible = (source_length - 1) // (target_length - 1)
    lower, upper = step_range
    upper = min(upper, feasible)
    lower = min(lower, upper)
    return int(rng.integers(lower, upper + 1))


def foldback(
    source_length,
    target_length,
    rng,
    repeat_probability=0.05,
    max_consecutive_repeat=1,
):
    """Walk forward and backward without duplicating sequence boundaries."""

    if source_length == 1:
        return [0] * target_length

    position, direction, repeat_run = 0, 1, 0
    path = [position]
    while len(path) < target_length:
        if (
            repeat_run < max_consecutive_repeat
            and rng.random() < repeat_probability
        ):
            path.append(position)
            repeat_run += 1
            continue

        repeat_run = 0
        next_position = position + direction
        if next_position < 0 or next_position >= source_length:
            direction *= -1
            next_position = position + direction
        position = next_position
        path.append(position)
    return path


class ABotReconSequenceWrapper(EasyDataset):
    """Filter, adaptively stride, then fold short ordered sequences.

    ``dataset`` can be an initialized Pi3 dataset or a Hydra partial.  Configure
    ``frame_num`` as the largest frame count sampled by the dynamic batch
    sampler.  Parents exposing ``frame_step`` receive adaptive strides.  Other
    ordered parents retain their own forward sampling rule.
    """

    def __init__(
        self,
        dataset,
        frame_num,
        resolution=None,
        frame_step_range=(1, 1),
        min_source_frames=12,
        min_source_ratio=0.25,
        repeat_probability=0.05,
        max_consecutive_repeat=1,
        sort_views=False,
        dataset_label=None,
        seed=2024,
    ):
        self.frame_num = int(frame_num)
        if callable(dataset):
            kwargs = {"frame_num": self.frame_num}
            if resolution is not None:
                kwargs["resolution"] = resolution
            dataset = dataset(**kwargs)
        self.dataset = dataset
        self.frame_step_range = tuple(map(int, frame_step_range))
        self.min_source_frames = int(min_source_frames)
        self.min_source_ratio = float(min_source_ratio)
        self.repeat_probability = float(repeat_probability)
        self.max_consecutive_repeat = int(max_consecutive_repeat)
        self.sort_views = bool(sort_views)
        self.dataset_label = dataset_label
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        cutoff = _minimum_length(
            self.frame_num,
            self.min_source_frames,
            self.min_source_ratio,
        )
        lengths = np.fromiter(
            (_sequence_length(dataset, i) for i in range(len(dataset))),
            dtype=np.int64,
            count=len(dataset),
        )
        self.source_indices = np.flatnonzero(lengths >= cutoff)
        self.source_lengths = lengths[self.source_indices]
        if not len(self.source_indices):
            raise ValueError(
                f"No sequence has the required {cutoff} source frames"
            )
        print(
            f"[ABotSampler] kept {len(self.source_indices)}/{len(lengths)} "
            f"sequences, target={self.frame_num}, cutoff={cutoff}",
            flush=True,
        )

    def __len__(self):
        return len(self.source_indices)

    @property
    def _resolutions(self):
        return self.dataset._resolutions

    def convert_attributes(self):
        self.dataset.convert_attributes()

    def set_epoch(self, epoch, base_seed=None):
        self.dataset.set_epoch(epoch, base_seed)
        seed = self.seed if base_seed is None else int(base_seed)
        self.rng = np.random.default_rng(seed + int(epoch))

    def _load(self, index, resolution_index, frame_num, frame_step):
        old_num = self.dataset.frame_num
        has_step = hasattr(self.dataset, "frame_step")
        old_step = getattr(self.dataset, "frame_step", 1)
        try:
            self.dataset.frame_num = frame_num
            if has_step:
                self.dataset.frame_step = frame_step
            parent_index = (
                index
                if resolution_index is None
                else (index, resolution_index, frame_num)
            )
            views = self.dataset[parent_index]
            if self.sort_views:
                views.sort(key=_frame_number)
            return views, has_step
        finally:
            self.dataset.frame_num = old_num
            if has_step:
                self.dataset.frame_step = old_step

    def _annotate(self, view, frame, sample, resolution, policy, step):
        result = dict(view)
        result["source_frame_id"] = result.get(
            "source_frame_id", result.get("frame_id", result.get("instance"))
        )
        result["frame_id"] = frame
        result["temporal_index"] = frame
        result["abot_sampling_policy"] = policy
        result["abot_frame_step"] = step
        result["abot_recon_ordered"] = True
        if self.dataset_label is not None:
            result["dataset"] = self.dataset_label
        result["idx"] = (sample, 0 if resolution is None else resolution, frame)
        return result

    def __getitem__(self, index):
        if isinstance(index, tuple):
            sample, resolution, target = index
            sample, resolution, target = int(sample), int(resolution), int(target)
        else:
            sample, resolution, target = int(index), None, self.frame_num

        source_index = int(self.source_indices[sample])
        source_length = int(self.source_lengths[sample])
        cutoff = _minimum_length(
            target, self.min_source_frames, self.min_source_ratio
        )
        if source_length < cutoff:
            raise ValueError(
                "frame_num must cover the dynamic sampler's largest target"
            )

        if source_length >= target:
            if hasattr(self.dataset, "frame_step"):
                step = adaptive_forward(
                    source_length, target, self.frame_step_range, self.rng
                )
                policy = "adaptive_forward"
            else:
                step, policy = 1, "parent_ordered"
            views, _ = self._load(source_index, resolution, target, step)
            path = range(target)
        else:
            views, _ = self._load(source_index, resolution, source_length, 1)
            path = foldback(
                len(views),
                target,
                self.rng,
                self.repeat_probability,
                self.max_consecutive_repeat,
            )
            step, policy = 1, "foldback"

        return [
            self._annotate(
                views[source_position],
                frame,
                sample,
                resolution,
                policy,
                step,
            )
            for frame, source_position in enumerate(path)
        ]


__all__ = ["ABotReconSequenceWrapper", "adaptive_forward", "foldback"]
