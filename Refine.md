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
| `utils/glob3r_scheduler.py` | 支持梯度累积换算的 linear-warmup + cosine scheduler。 |

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

默认每 500 个 optimizer steps 写入一张 `train/matching_overview` 四行网格：`Images` 显示参考帧和目标帧，`Warp` 显示目标帧按预测 warp 重采样到参考视角的结果，`Conf` 显示预测置信度，`Mask` 显示几何监督 mask。默认取 1 个样本和最多 7 个目标帧；可在 `glob3r.visualization` 中调整记录间隔、样本数、目标帧数、置信度阈值和单元格宽度。

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

启动后在浏览器访问 `http://localhost:6006`，在 Images 面板中选择 `train/matching_overview`。
