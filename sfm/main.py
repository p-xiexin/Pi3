"""Run full-sequence SfM with selectable Glob3R or VGGSfM tracks."""

import argparse

from .config import load_config
from .pipeline import SfMPipeline


def run(config):
    """Run cached factor construction, global initialization, BA, and export."""
    return SfMPipeline(config).run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sfm/config.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
