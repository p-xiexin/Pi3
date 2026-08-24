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

Set both checkpoint paths and dataset paths in `valid.yaml`, or use Hydra
command-line overrides. Delete or comment out unused entries under `datasets`.
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

`timings.csv` records synchronized wall time for the Pi3 backbone, Pi3 point and
depth-confidence decoding, Glob3R matching, and their combined network path.
Dataset loading, ground-truth construction, loss computation, and rendering are
outside the reported network time.

Window tensors live inside one computation scope. After the image and metrics
are committed, CPU objects are collected and the CUDA allocator cache is
released before the next window. The model remains resident on its rank's GPU.

Local development only validates compilation and CPU helpers. A successful
local test does not establish checkpoint loading, GPU inference, or real-data
correctness on the server.
