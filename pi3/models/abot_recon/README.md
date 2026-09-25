# ABot-Recon training migration

This package maps the public ABot-Recon inference network onto Pi3's existing
training stack.  It keeps the DINOv2 ViT-L/14 encoder, 36 alternating frame and
global decoder blocks, five pose tokens, local point head and optional cloned
confidence head.  Global blocks use causal PyTorch SDPA with a fixed 12-frame
key/value window.  The pose branch uses adjacent translation and scalar-last
quaternion prediction followed by the released gated temporal rotation refiner.

## Supervision

For batch $b$, frame $i$, and pixel $p=(u,v)$, the dataset provides world point
$X^w_{bip}$, camera-to-world pose $T^w_{bi}\in\mathrm{SE}(3)$, and validity
$m_{bip}$.  The model predicts camera-frame point $\hat P_{bip}$, trajectory
$\hat T_{0\leftarrow i}$, rotation residual $r_{bi}$, and optional confidence
logit $\hat s_{bip}$.

### Target gauge and scene normalization

Frame zero defines the sequence gauge.

$$
T_{0\leftarrow w}=(T^{w}_{b0})^{-1},\qquad
G_{bip}=\pi_3\!\left(T_{0\leftarrow w}\bar X^{w}_{bip}\right),\qquad
T_{0\leftarrow i}=T_{0\leftarrow w}T^{w}_{bi},
$$

Here $\bar X=[X^\top,1]^\top$ and $\pi_3$ removes its homogeneous coordinate.
The sample scale is the mean valid point distance.

$$
d_b=\frac{\sum_{i,p}m_{bip}\lVert G_{bip}\rVert_2}
          {\sum_{i,p}m_{bip}},\qquad
G'_{bip}=\frac{G_{bip}}{d_b},\qquad
t'_{0\leftarrow i}=\frac{t_{0\leftarrow i}}{d_b}.
$$

$$
P^*_{bip}=\pi_3\!\left((T'_{0\leftarrow i})^{-1}\bar G'_{bip}\right).
$$

Thus point supervision uses each frame's camera coordinates.  World points
only construct $P^*$ and receive no direct loss.  The effective mask is
$m\land\operatorname{isfinite}(X^w)$.

### Scale-aligned local point loss

Pi3 aligns each sample with inverse-depth weights.

$$
w_{bip}=\frac{1}{\max\!\left(z^*_{bip},0.1\bar z^*_{bi}\right)+10^{-6}},
\qquad
\bar z^*_{bi}=\frac{\sum_p m_{bip}z^*_{bip}}{\sum_p m_{bip}}.
$$

$$
\tilde a_b=\arg\min_{a\in\mathbb{R}}
\sum_{i,p,c}m_{bip}w_{bip}
\left|a\hat P_{bipc}-P^*_{bipc}\right|,
\qquad a_b^*=|\tilde a_b|.
$$

The solver uses 4096 nearest-neighbor resamples of flattened valid points.
$a_b^*$ is detached from autograd.

$$
\mathcal{L}_{\mathrm{pts}}=
\operatorname{mean}_{b,i,p,c\,:\,m_{bip}=1}
\left[w_{bip}\left|a_b^*\hat P_{bipc}-P^*_{bipc}\right|\right].
$$

This loss is scale invariant.  Pose supervision multiplies predicted
translations by $a_b^*$ so point and trajectory scales agree.  Since $d_b$
removes dataset scale, the objective does not supervise absolute metric scale.

### Surface-normal loss

Each $2\times2$ cell yields four oriented triangle normals $n_q$.

$$
\theta_q=\operatorname{atan2}\!\left(
\lVert n_q(a_b^*\hat P)\times n_q(P^*)\rVert_2,
n_q(a_b^*\hat P)^\top n_q(P^*)
\right).
$$

Triangles touching invalid pixels or target depth edges with relative tolerance
$0.03$ are masked.  Valid angles are clamped to $[1^\circ,90^\circ]$.

$$
\rho_\beta(x)=
\begin{cases}
x^2/(2\beta), & x<\beta,\\
x-\beta/2, & x\geq\beta,
\end{cases}
\qquad \beta=3^\circ.
$$

The four penalties are averaged and divided by $4\max(H,W)$.  Pi3 enables this
term only for its high- and middle-quality dataset groups.  ScanNet, TartanAir,
Hypersim, and BlendedMVS qualify.  KITTI does not, so
$\mathcal{L}_{\mathrm{normal}}=0$ in the supplied configs.

### Composition-aware pose loss

$$
\hat T_{i\leftarrow j}=\hat T_{0\leftarrow i}^{-1}\hat T_{0\leftarrow j},
\qquad
T^*_{i\leftarrow j}=(T'_{0\leftarrow i})^{-1}T'_{0\leftarrow j}.
$$

Predicted translations are multiplied by $a_b^*$ before composition.  With
$g=j-i$, the supervised pair set is

$$
\mathcal{P}=\{(i,j)\mid 0\leq i<j<N,\ j-i\leq K\},
\qquad K=\min(11,N-1).
$$

$$
\alpha_{ij}=\frac{(j-i)^\gamma}
{\frac{1}{|\mathcal{P}|}\sum_{(u,v)\in\mathcal{P}}(v-u)^\gamma},
\qquad \gamma=0.5.
$$

Translation uses coordinate-averaged Huber loss with $\delta=0.1$.  Rotation
uses the $\mathrm{SO}(3)$ geodesic angle.

$$
\ell_{t,ij}=\frac{1}{3}\sum_{c=1}^{3}
\operatorname{Huber}_\delta(\hat t_{i\leftarrow j,c}-t^*_{i\leftarrow j,c}),
$$

$$
\ell_{R,ij}=\operatorname{atan2}\!\left(
\frac{1}{2}\lVert\operatorname{vee}(Q-Q^\top)\rVert_2,
\frac{\operatorname{tr}(Q)-1}{2}
\right),\qquad
Q=\hat R_{i\leftarrow j}^{\top}R^*_{i\leftarrow j}.
$$

The logged losses have the following exact relation.

$$
\texttt{trans\_loss}=\mathcal{L}_{\mathrm{trans}}
=\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}\ell_{t,ij},
$$

$$
\texttt{rot\_loss}=\mathcal{L}_{\mathrm{rot}}
=\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}
\alpha_{ij}\ell_{R,ij},
$$

$$
\texttt{pose\_loss}=\mathcal{L}_{\mathrm{pose}}
=\lambda_t\mathcal{L}_{\mathrm{trans}}
+\lambda_R\mathcal{L}_{\mathrm{rot}}
=100\,\texttt{trans\_loss}+\texttt{rot\_loss},
$$

where $\lambda_t=100$ and $\lambda_R=1$.  `rot_loss` already contains
$\alpha_{ij}$, while `trans_loss` has no gap weighting.  The total training
objective then applies `pose_weight = 0.1`.

$$
0.1\,\texttt{pose\_loss}
=10\,\texttt{trans\_loss}+0.1\,\texttt{rot\_loss}.
$$

### Rotation-refiner smoothness

$$
\mathcal{L}_{\mathrm{smooth}}=
\operatorname{mean}_{b,i,c}r_{bic}^{2}
+\lambda_\Delta\operatorname{mean}_{b,i,c}(r_{b,i+1,c}-r_{bic})^{2},
\qquad \lambda_\Delta=1.
$$

The report omits this exact equation.  Stage I disables the refiner, so the term
is zero.  Stage II enables it.

### Confidence supervision

$$
e_{bip}=\frac{w_{bip}}{3}\sum_c
\left|a_b^*\hat P_{bipc}-P^*_{bipc}\right|,
\qquad
y_{bip}=\mathbf{1}[e_{bip}<\tau],\qquad \tau=0.02,
$$

$$
\mathcal{L}_{\mathrm{conf}}=
\operatorname{mean}_{b,i,p\,:\,m_{bip}=1}
\operatorname{BCEWithLogits}(\hat s_{bip},y_{bip}).
$$

The label is detached.  It is an implementation choice because the report does
not publish confidence targets.  The supplied configs disable the confidence
head and set `confidence_weight = 0.0`.

### Total objective and active stage losses

$$
\mathcal{L}=
\lambda_{\mathrm{pts}}\mathcal{L}_{\mathrm{pts}}+
\lambda_{\mathrm{normal}}\mathcal{L}_{\mathrm{normal}}+
\lambda_{\mathrm{pose}}\mathcal{L}_{\mathrm{pose}}+
\lambda_{\mathrm{smooth}}\mathcal{L}_{\mathrm{smooth}}+
\lambda_{\mathrm{conf}}\mathcal{L}_{\mathrm{conf}}.
$$

$$
(\lambda_{\mathrm{pts}},\lambda_{\mathrm{normal}},
\lambda_{\mathrm{pose}},\lambda_{\mathrm{smooth}},
\lambda_{\mathrm{conf}})=(1,1,0.1,0.1,0).
$$

$$
\mathcal{L}_{\mathrm{stage1}}=
\mathcal{L}_{\mathrm{pts}}+0.1\mathcal{L}_{\mathrm{pose}}.
$$

$$
\mathcal{L}_{\mathrm{stage2}}=
\mathcal{L}_{\mathrm{pts}}+0.1\mathcal{L}_{\mathrm{pose}}+
0.1\mathcal{L}_{\mathrm{smooth}}.
$$

KITTI contributes no normal loss.  Stage I also has no smoothness loss.  Stage
II adds smoothness through the enabled rotation refiner.  The report does not
publish the scalar weights, $\gamma$, confidence targets, or exact smoothness
formula.  These values document this implementation.

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
