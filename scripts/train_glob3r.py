import sys
sys.path.append('.')

import hydra
import trainers
from omegaconf import OmegaConf, open_dict


def run_stage(stage_cfg):
    trainer = eval(stage_cfg.trainer)(stage_cfg)
    trainer.train()

@hydra.main(version_base="1.2", config_path="../configs", config_name="default")
def main(hydra_cfg):
    stages = hydra_cfg.get("stages")
    if stages is None:
        run_stage(hydra_cfg)
        return

    for stage in stages:
        stage_cfg = OmegaConf.merge(hydra_cfg, stage)
        with open_dict(stage_cfg):
            del stage_cfg["stages"]
        run_stage(stage_cfg)

if __name__ == '__main__':
    main()
