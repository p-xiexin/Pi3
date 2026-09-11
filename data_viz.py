"""Repository entry point for Hydra-configured dataset visualization."""

import hydra
from omegaconf import DictConfig

from tests.dataset_viz import run


@hydra.main(
    version_base="1.2",
    config_path="configs",
    config_name="dataset_viz.yaml",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
