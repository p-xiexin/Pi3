# Glob3R / Pi3 Refine 实现说明

本实现先以冻结的 Pi3 作为几何 backbone，新增 Glob3R matching 与 refinement 分支。训练仍使用仓库原有入口 `scripts/train_pi3.py`，通过独立组合模型接入，不修改 Pi3 实现。

## 核心实现

| 文件 | 主要内容 | 论文对应部分 |
| --- | --- | --- |
| `pi3/models/glob3r/model.py` | 张量约束、5 层 matching decoder、多视图 Fourier embedding、DPT head、stride 4/2/1 refinement，以及 RoMaV2 权重映射 | Eq. (1)、(2)、(7)–(25) |
| `pi3/models/glob3r/geometry.py` | GT warp、深度一致性、关键帧计数、motion averaging 与 bundle adjustment 目标 | Eq. (4)–(6)、(26)–(30) |
| `pi3/models/glob3r/loss.py` | Patch NLL、Charbonnier warp loss、confidence BCE 与总损失 | Eq. (3)、(31)–(35) |
| `pi3/models/glob3r/glob3r_training.py` | 完整 Glob3R 训练模型：Pi3 encoder 特征捕获、geometry token 整理、matching head 组合与 backbone 冻结 | Appendix A/B 的 backbone 接入 |

## Pi3 接入

| 文件 | 作用 |
| --- | --- |
| `pi3/models/glob3r/glob3r_training.py` | `Glob3R` 显式调用冻结的 encoder 和 decoder，不执行 Pi3 原有 point、camera 和 confidence heads。 |
| `pi3/models/pi3_training.py` | 现有 Pi3 backbone；Glob3R 只调用并冻结它，不修改文件内容。 |
| `trainers/glob3r_trainer.py` | 阶段训练与优化器过滤；仅调用独立的可视化 monitor。 |
| `utils/glob3r_visualization.py` | 独立管理 TensorBoard 绘制、记录间隔和日志写入。 |

## 训练与数据配置

| 文件 | 作用 |
| --- | --- |
| `configs/glob3r_coarse.yaml` | 完整基础实验：Pi3、matching head、loss、数据、监控，以及 coarse 阶段训练参数。 |
| `configs/glob3r_refinement.yaml` | 继承 coarse 配置，只覆盖 refinement 阶段、帧数和学习率差异。 |
| `datasets/glob3r_transforms.py` | Color jitter、Gaussian blur、随机灰度等训练增强。 |
| `datasets/glob3r_scannet_dataset.py` | 为最小验证提供确定性的 ScanNet 连续帧窗口；不替代官方训练集实现。 |
| `utils/glob3r_scheduler.py` | 支持梯度累积换算的 linear-warmup + cosine scheduler。 |

### Glob3R ScanNet 验证集设计

`Glob3RScannetValidationDataset` 是供 `glob3r_test.yaml` 使用的确定性小窗口数据集，不替代官方 `ScannetDataset`。它取 color、depth、pose 文件的交集，过滤可选 invalid list，并按固定 seed 从排序后的帧中选取连续窗口；首帧作为 reference，其余帧保持时间顺序。

图像、米制深度、c2w pose 和内参仍通过 `BaseDataset` 的裁剪缩放流程输出。该实现仅假设相邻帧具有较高重叠，没有使用 GT overlap 或 Eq. (4) keyframe 筛选。

## 代码检查

当前阶段只沿 `scripts/train_pi3.py`、`Glob3RTrainer`、`Glob3R` 和 matching loss 做静态调用链审查，并核对 token 裁剪、张量维度、冻结范围、optimizer 参数前缀及 TensorBoard 输出字段；不执行模型前向、反向或训练测试。

## 训练

两个阶段均使用原训练入口 `scripts/train_pi3.py`。配置中的 64K micro-steps 和梯度累积 2 对应论文的 32K optimizer steps；2K warmup 同样按 optimizer steps 计算。

在 Linux 下下载 Pi3 官方权重：

```bash
mkdir -p ckpts/Pi3
wget -c "https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors?download=true" \
  -O ckpts/Pi3/model.safetensors
```

Coarse matching：

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  scripts/train_pi3.py \
  --config-name glob3r_coarse \
  glob3r.backbone_checkpoint=ckpts/Pi3/model.safetensors
```

单卡调试训练（5 epochs × 800 iterations）：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  scripts/train_pi3.py \
  --config-name glob3r_test \
  glob3r.backbone_checkpoint=ckpts/Pi3/model.safetensors \
  scannet_root=/path/to/scannet
```

`glob3r_test.yaml` 是不继承其他 YAML 的独立调试配置，共执行 5 个 epoch、每个 epoch
800 个 optimizer iterations，并每 100 step 写入 TensorBoard 可视化。训练和验证都只
使用 ScanNet；每个 epoch 结束后执行验证并保存 checkpoint。

Refinement 阶段加载 coarse matching checkpoint，并可导入尺寸兼容的 RoMaV2 refinement 权重：

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
  scripts/train_pi3.py \
  --config-name glob3r_refinement \
  glob3r.backbone_checkpoint=ckpts/Pi3/model.safetensors \
  glob3r.matching_checkpoint=/path/to/coarse_matching_checkpoint.pt \
  glob3r.romav2_refinement_checkpoint=/path/to/romav2_checkpoint.pt
```

配置只展示仓库接入方式。完整复现论文训练分布还需要 Appendix B 所列的外部数据集；这些数据集及预训练 checkpoint 不随仓库提供。

## TensorBoard

训练阶段每 500 个 optimizer steps 写入一张 `train/matching_overview`，验证阶段每个 epoch 取第一批写入 `val/matching_overview`。两者都是六行网格：`Images` 显示参考帧和目标帧，`Warp` 与 `GT Warp` 分别显示预测和真值 warp 的重采样结果，`Conf` 与 `GT Conf` 分别显示预测和真值置信度，`Mask` 显示几何监督区域。默认取 1 个样本和最多 7 个目标帧；可在 `glob3r.visualization` 中调整训练记录间隔、样本数、目标帧数、置信度阈值和单元格宽度。

默认日志目录是 `outputs/${name}`。在仓库根目录查看 coarse 训练日志：

```powershell
tensorboard --logdir outputs/glob3r_coarse
```

查看 refinement 训练日志：

```powershell
tensorboard --logdir outputs/glob3r_refinement
```

也可以同时查看 `outputs` 下的所有实验：

```powershell
tensorboard --logdir outputs
```

启动后在浏览器访问 `http://localhost:6006`，在 Images 面板中选择 `train/matching_overview` 或 `val/matching_overview`。

## 论文公式说明

Warp 仍表示从 reference 像素到 target 坐标的映射：

\[
W^{a\rightarrow b}(p_{\mathrm{ref}})=p_{\mathrm{target}}.
\tag{2}
\]

可视化相应计算 `output(p_ref)=target(p_target)`。

相似度矩阵以 reference patch \(n\) 为行、target patch \(m\) 为列，并沿 target 维 \(m\) 归一化：

\[
L_{nm}^{a\rightarrow b}
=
\frac{1}{\tau}\operatorname{cosim}\!\left(z_n^a,z_m^b\right),
\qquad
P_{nm}^{a\rightarrow b}
=
\operatorname{Softmax}_{m}\!\left(L_{n:}^{a\rightarrow b}\right).
\tag{12}
\]

代码按照 RoMaV2 [15] 直接保存 `cos/tau` logits，只进行一次 Softmax。

Fourier embedding 编码 target patch 坐标：

\[
\chi_m^b=\gamma\!\left(p_m^b\right).
\tag{15}
\]

随后使用匹配概率将 target 坐标编码聚合到 reference 网格：

\[
\chi_n^{a\rightarrow b}
=
\sum_m P_{nm}^{a\rightarrow b}\chi_m^b.
\tag{16}
\]

由于 \(\chi_n^{a\rightarrow b}\) 已位于 reference 网格，DPT 将其与 reference token \(Z^a\) 拼接：

\[
F^{a\rightarrow b}
=
\operatorname{Proj}\!\left(Z^a\oplus\chi^{a\rightarrow b}\right).
\tag{18}
\]

代码同时使用 reference view 的 encoder features，避免在 DPT 中混入 target 网格。

NLL 对每个 reference patch 行 \(n\) 监督其对应的 target patch \(m_n^*\)：

\[
\mathcal{L}_{\mathrm{NLL}}^{a\rightarrow b}
=
-\frac{1}{\lvert\Omega_{\mathrm{patch}}\rvert}
\sum_n
\log P_{n,m_n^*}^{a\rightarrow b}.
\tag{31}
\]

loss 直接对 \(L=\operatorname{cosim}/\tau\) 做 cross entropy，与坐标聚合使用同一行概率分布，避免对 `exp(cos/tau)` 再次 Softmax。

多尺度监督保持不变：GT warp、confidence 和 mask 使用 nearest 下采样；warp 数值仍是原图 target 像素坐标，不需要再次缩放内参。

## 待确认的实现问题

以下问题尚未修改代码：

1. **RoMaV2 refinement 坐标约定不兼容。** 当前 `WarpRefinement` 使用端点归一化坐标、`align_corners=True`，并直接执行 `warp + delta_warp`；RoMaV2 使用像素中心坐标、`align_corners=False`，且 residual 除以 `4 * [W_s, H_s]`，displacement/local correlation 还使用 `scale_factor`。当前虽然能够加载形状兼容的 RoMaV2 权重，但坐标和更新尺度并不完全兼容。
2. **Eq. (31) 的 patch 标签可能存在半个 patch 偏移。** `patch_nll_targets` 通过 `x / (W - 1) * (W_patch - 1)` 量化每个 reference-row 投影对应的 target patch；这相当于按图像端点对齐，而非按 ViT patch center 对齐，可能产生系统性标签偏移。
