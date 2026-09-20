# ABot-Recon training migration

This package maps the public ABot-Recon inference network onto Pi3's existing
training stack.  It keeps the DINOv2 ViT-L/14 encoder, 36 alternating frame and
global decoder blocks, five pose tokens, local point head and optional cloned
confidence head.  Global blocks use causal PyTorch SDPA with a fixed 12-frame
key/value window.  The pose branch uses adjacent translation and scalar-last
quaternion prediction followed by the released gated temporal rotation refiner.

The public report defines point, normal, multi-gap pose, rotation smoothness and
confidence losses.  It does not publish the scalar weights, gap gamma, precise
smoothness equation or confidence targets.  `ABotReconLoss` exposes all of these
choices.  Its confidence label is the detached aligned point error threshold,
without the private sky-segmentation dependency used by Pi3 training.

Pi3 datasets return a list of view dictionaries in the order established by
`ABotReconSequenceWrapper`.  The trainer stacks only the input images, while
the loss stacks point maps, masks and camera poses directly from that original
list.  The wrapper filters very short sequences, selects a feasible forward
stride and folds a loaded short sequence in memory when it cannot fill a clip.

Stage I starts from a released Pi3 checkpoint and trains 32 frames at 504 by 280.
The model, loss, trainer, dataset adapters, entry points, and self-contained
Hydra configs all live in this directory for server migration.

```powershell
python -m pi3.models.abot_recon.train model.load_pi3=path/to/pi3.pth train_dataset.KITTIABotRecon.dataset.raw_root=path/to/kitti_raw train_dataset.KITTIABotRecon.dataset.depth_root=path/to/kitti_depth
```

The package-local `train_slurm.sh` follows the repository's single-node,
eight-GPU Accelerate launch.  All model and dataset settings come from
`stage1.yaml`.

Set `train.use_ema=true` to maintain the report's 0.999 EMA through training.  EMA
updates after every optimizer step and is used for validation.  Accelerate
stores it as `custom_checkpoint_0.pkl` beside the raw model and optimizer
states, so interrupted runs restore both versions.  Point the next stage's
`model.ckpt` at this custom checkpoint to initialize from EMA weights.  Keep
EMA disabled when resuming an older checkpoint that does not contain this
file.

`data_sampler.py` wraps existing Pi3 datasets through Hydra partials, so all
image, depth and geometry code remains in the parent loader.  Parent datasets
with `frame_step` use adaptive forward sampling.  Parents without it keep step
one and their native ordered sampling.  The multi-dataset examples live in
`dataset.yaml`; add a dataset to `weights` only after filling its local paths.

Stage II inherits the complete Stage I configuration and overrides only its
training schedule in `stage2.yaml`.  Start it with
`python -m pi3.models.abot_recon.train --config-name stage2 model.ckpt=...`.
Use the Stage I model file or EMA custom checkpoint for `model.ckpt`; do not use
`train.resume` for a stage transition because the optimizer parameter groups
change when the rotation refiner becomes trainable.

The 32-frame setting follows the report and can exceed available GPU memory.
Lower `train.image_num_range` and `train.max_img_per_gpu` together for a
smoke run.  The migrated path uses SDPA and does not include the released
FlashInfer paged-KV runtime, loop closure or global bundle adjustment components.

Inference uses the released width-locked preprocessing and returns the dense
point, confidence and pose dictionary produced by the training forward.

```python
from pi3.models.abot_recon import ABotRecon, infer_paths

model = ABotRecon(ckpt="checkpoints/abot_recon.safetensors").cuda().eval()
result = infer_paths(model, image_paths)
```

`stage1.yaml` wraps the existing KITTI loader as a concrete ordered-video
example with KITTI Raw RGB/OXTS and official Depth Completion supervision.  The
report uses a mixture of 30 datasets, dataset-specific temporal samplers,
filtering and augmentation.  Exact data reproduction requires those internal
datasets and policies.

```powershell
python -m pi3.models.abot_recon.train model.load_pi3=path/to/pi3.pth train_dataset.KITTIABotRecon.dataset.raw_root=path/to/kitti_raw train_dataset.KITTIABotRecon.dataset.depth_root=path/to/kitti_depth train_dataset.KITTIABotRecon.dataset.index_file=path/to/kitti.npy
```

TensorBoard logs `train/abot_reconstruction` and `val/abot_reconstruction`
using the interval and frame sampling settings under
`abot_recon.visualization`.
