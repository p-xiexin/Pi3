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

Pi3 datasets still return a list of view dictionaries.  `prepare_abot_batch`
orders each batch element independently using `frame_id` or the numeric suffix of
`instance`, then applies the same permutation to images, point maps, masks,
intrinsics and camera poses.  Datasets without numeric frame ids must preserve a
temporal or graph traversal order and set `shuffle: false`.

Stage I starts from a released Pi3 checkpoint and trains 32 frames at 504 by 280.
The model, loss, trainer, KITTI adapter, visualization, entry points, and one
self-contained Hydra config all live in this directory for server migration.
Install the package-local visualization dependency after copying the directory.

```powershell
pip install -r pi3/models/abot_recon/requirements.txt
```

```powershell
python -m pi3.models.abot_recon.train model.load_pi3=path/to/pi3.pth train_dataset.KITTIABotRecon.raw_root=path/to/kitti_raw train_dataset.KITTIABotRecon.depth_root=path/to/kitti_depth
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

`dataset.py` contains thin ABot-Recon adapters for the existing KITTI,
TartanAir, ScanNet, Waymo and BlendedMVS loaders.  The adapters reuse all parent
dataset IO and geometry code, disable view shuffling and attach a stable stream
order.  The inactive dataset blocks remain in `stage1.yaml`; add a dataset to
the train and test `weights` mappings only after filling its local paths.

Stage II inherits the complete Stage I configuration and overrides only its
training schedule in `stage2.yaml`.  Start it with
`python -m pi3.models.abot_recon.train --config-name stage2 model.ckpt=...`.
Use the Stage I model file or EMA custom checkpoint for `model.ckpt`; do not use
`train.resume` for a stage transition because the optimizer parameter groups
change when the rotation refiner becomes trainable.

`viz.py` writes one TensorBoard image grid every 500 optimizer steps by default.
`abot_reconstruction` aligns its columns to uniformly sampled frames and shows
world-frame pose, RGB, aligned predicted depth, ground-truth depth, relative
point error, confidence and the validity mask.  Every pose cell contains the
complete dashed GT path, the solid predicted prefix available at that frame,
and separate GT and predicted SLAM camera frustums.  Validation records the
first batch once per epoch.  The interval, frame count and panel sizes live
under `abot_recon.visualization` in `stage1.yaml`.

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

`stage1.yaml` uses the package-local KITTI adapter as a concrete ordered-video
example with KITTI Raw RGB/OXTS and official Depth Completion supervision.  The
report uses a mixture of 30 datasets, dataset-specific temporal samplers,
filtering and augmentation.  Exact data reproduction requires those internal
datasets and policies.

```powershell
python -m pi3.models.abot_recon.train model.load_pi3=path/to/pi3.pth train_dataset.KITTIABotRecon.raw_root=path/to/kitti_raw train_dataset.KITTIABotRecon.depth_root=path/to/kitti_depth train_dataset.KITTIABotRecon.index_file=path/to/kitti.npy
```

The dataset-only ABot-Recon contract can be rendered without loading a model.

```powershell
python -m pi3.models.abot_recon.data_viz train_dataset.KITTIABotRecon.raw_root=path/to/kitti_raw train_dataset.KITTIABotRecon.depth_root=path/to/kitti_depth train_dataset.KITTIABotRecon.index_file=path/to/kitti.npy
```
