"""Ordered KITTI training clips for the ABot-Recon Stage I prototype."""

from datasets.kitti_dataset import KITTIPi3XDataset


class KITTIABotReconDataset(KITTIPi3XDataset):
    """Add the chronological metadata required by ``prepare_abot_batch``."""

    def __init__(self, *args, shuffle=False, random_sample_thres=0.0, **kwargs):
        if shuffle:
            raise ValueError("KITTIABotReconDataset requires shuffle=false")
        super().__init__(
            *args,
            shuffle=False,
            random_sample_thres=random_sample_thres,
            **kwargs,
        )
        self.dataset_label = "KITTIABotRecon"

    def _get_views(self, index, resolution, rng, is_test=False):
        views = super()._get_views(index, resolution, rng, is_test=is_test)
        for temporal_index, view in enumerate(views):
            view["frame_id"] = int(view["instance"])
            view["temporal_index"] = temporal_index
            view["dataset"] = self.dataset_label
            view["abot_recon_ordered"] = True
        return views
