# Glob3R sequence evaluation

This directory contains the standalone evaluation entry point and its Hydra
configuration. Existing dataset implementations remain unchanged. Exact window
positions are loaded through disposable shallow copies. Record-based datasets
receive a fixed sampler only on the copy. `AdsDataset` copies contain one
sequence with file and calibration lists sliced to the requested positions.
The source dataset instance remains read-only throughout evaluation.

The intended server command is

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 8 eval/dataset_eval.py --config-name valid
```

Choose the backbone from the `model` Hydra group. `model=pi3` loads a standard
Pi3 checkpoint through `model.backbone_checkpoint`. `model=pi3x` defaults to
the released inference class and loads its checkpoint through
`model.backbone.ckpt`. A checkpoint produced with `pi3x_training.py` selects
that existing class by overriding `_target_`. For example:

```bash
# Pi3
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 8 eval/dataset_eval.py --config-name valid \
  model=pi3 \
  model.backbone_checkpoint=/path/to/pi3.safetensors \
  model.matching_checkpoint=/path/to/pi3_glob3r.bin

# A checkpoint trained with pi3x_training.py
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 8 eval/dataset_eval.py --config-name valid \
  model=pi3x \
  model.backbone._target_=pi3.models.pi3x_training.Pi3X \
  +model.backbone.checkpoint_strategy=null \
  model.backbone.ckpt=/path/to/pi3x_training.bin \
  model.matching_checkpoint=/path/to/pi3x_glob3r.bin
```

Evaluation enforces `model.with_prior=false`. Pi3 and Pi3X therefore receive
the same RGB-only input, and the training Pi3X class cannot sample stochastic
conditioning masks while in evaluation mode. Set the dataset paths in
`valid.yaml`, or use Hydra command-line overrides. Delete or comment out unused
entries under `datasets`.

Evaluation-only model assembly lives in `eval/model/glob3r.py`. It reuses the
existing Pi3, Pi3X, and Glob3R modules and does not alter or depend on the
`local_opt` model wrapper.
Each sequence is owned by one rank. Every completed window atomically writes a
PNG and a matching `chunk_*.json` commit, appends `metrics.csv` and
`timings.csv`, and updates `progress.json` plus `summary.json`. A matching rerun
rebuilds both CSV files from committed window JSON files and skips completed
windows. A changed config, window plan, or checkpoint file is rejected instead
of being mixed into an existing sequence directory.

At startup only the evaluation root is created. A new sequence directory and
its `resolved_config.yaml` appear after the first window has completed loading,
inference, loss computation, and rendering. Short sequences and sequences that
fail before their first commit leave no sequence directory. Their records are
appended to `rank_<rank>_failures.jsonl` in the evaluation root. Failures after
an earlier successful window are recorded both in that rank log and in the
existing sequence progress files.

Whole sequences are assigned greedily using their expected window counts. This
keeps each sequence's artifacts under one rank while reducing the long-tail
imbalance caused by round-robin sequence assignment. Ranks finish independently;
the entry point has no final distributed barrier or metric gather. Per-sequence
files are authoritative, and the console summary is local to each rank.

`timings.csv` records synchronized wall time for the Pi3-family backbone,
geometry heads, Glob3R matching, and their combined network path. Pi3X geometry
uses its existing `forward_head`, including point, confidence, camera, and
metric heads; the legacy CSV column remains
`pi3_point_depth_decode_seconds`. Dataset loading, ground-truth construction,
loss computation, and rendering are outside the reported network time.

Window tensors live inside one computation scope. After the image and metrics
are committed, CPU objects are collected and the CUDA allocator cache is
released before the next window. The model remains resident on its rank's GPU.

Local development only validates compilation and CPU helpers. A successful
local test does not establish checkpoint loading, GPU inference, or real-data
correctness on the server.
