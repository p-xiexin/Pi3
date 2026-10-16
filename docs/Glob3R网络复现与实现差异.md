# Glob3R 网络复现与实现差异

## 参考文献

- [1] Deng et al. [Glob3R: Global Structure-from-Motion with 3D Foundation Models](https://arxiv.org/html/2607.09225v1), 2026。本文引用的原论文公式编号固定为 v1。
- [2] Wang et al. [π³: Permutation-Equivariant Visual Geometry Learning](https://arxiv.org/abs/2507.13347), 2025；[Pi3 / Pi3X 代码](https://github.com/yyfz/Pi3)。本项目使用其几何 backbone。
- [3] Edstedt et al. [RoMa v2: Harder Better Faster Denser Feature Matching](https://arxiv.org/abs/2511.15706), 2025；[官方代码](https://github.com/Parskatt/RoMaV2)。用于 refinement 结构、坐标参数化与权重初始化。

下文“原式”指 [1] 的公式；标记为 R 的公式是按本地实现整理的表达，不沿用原论文编号。

## 网络结构与 Pi3 接入

输入为图像集合 $\mathcal I=\{I_i\}_{i=1}^{N}$，省略 batch 维。冻结 backbone 提取几何 token $\mathbf H$，5 层 matching decoder 生成匹配 token（原式 (7)–(9)）：

$$
\mathbf Z=\operatorname{Dec}_{\mathrm{match}}(\mathbf H),\qquad
\mathbf Z\in\mathbb R^{N\times M\times C},\qquad
M=\frac{H}{14}\frac{W}{14}.
\tag{R1}
$$

本地训练使用 Pi3，提取 encoder 第 `5/11/17/23` 层与 decoder 特征，不执行原有几何 heads。匹配 token 经坐标 embedding 和 DPT head 输出 stride 4 的 warp、confidence，再以 stride `4/2/1` refinement 恢复全分辨率。论文使用 Pi3X；本地评测和 SfM 可另外装配几何 heads，区别于 matching-only 训练路径。

实现见 [model.py](../pi3/models/glob3r/model.py) 和 [glob3r_training.py](../pi3/models/glob3r/glob3r_training.py)。

## 匹配定义与损失函数

### Warp 方向与原文错误

记 $a$ 为 reference、$b$ 为 target，$x^a$、$x^b$ 分别为两幅图像的像素坐标。统一定义为：

$$
W^{a\rightarrow b}:\mathcal D^a\rightarrow\mathbb R^2,
\qquad W^{a\rightarrow b}(x^a)=x^b.
\tag{R2}
$$

即 **warp 存在 reference 网格上，数值是 target 中的绝对像素坐标**，并非位移 $x^b-x^a$。其定义与原式 (28) 及 refinement 原式 (23) 一致。

**论文 v1 的方向叙述有误，已通过邮件与原作者确认。** 附录原式 (15)–(16) 聚合 reference 坐标到 target patch，原式 (18) 拼接 target token，原式 (31) 监督 target patch 对应的 reference 标签；DPT 输出段也写成 target→reference。这与 GT warp 的 reference→target 定义冲突，不能仅改箭头符号，必须同步修正 embedding、DPT 输入和 NLL 标签。现有 [邮件材料](../pi3/models/glob3r/email.md) 保存了问题说明和修正推导。

可视化将 target 重采样到 reference 网格：

$$
\widetilde I_b(x^a)=I_b\!\left(W^{a\rightarrow b}(x^a)\right).
\tag{R3}
$$

图像采样使用 target 的值生成 reference 网格上的图像，并不意味着 warp 是 target→reference；$W^{b\rightarrow a}$ 需独立预测，不能直接转置稠密 warp。

### 多视图匹配 embedding

令 $n$ 为 reference patch、$m$ 为 target patch。按修正后的原式 (12)、(15)–(18)，相似度以 reference 为行、target 为列，沿 target 维归一化：

$$
L_{nm}^{a\rightarrow b}=\frac{1}{\tau}
\frac{(z_n^a)^\top z_m^b}{\|z_n^a\|\|z_m^b\|},\qquad
P_{nm}^{a\rightarrow b}=\operatorname{Softmax}_{m}(L_{n:}^{a\rightarrow b}),
\qquad\tau=0.1.
\tag{R4}
$$

编码 **target 坐标**，再聚合回 reference 网格：

$$
\gamma(x)=\cos(2\pi A x)\oplus\sin(2\pi A x),\qquad
\chi_n^{a\rightarrow b}=\sum_m P_{nm}^{a\rightarrow b}\gamma(x_m^b),\qquad
F_n^{a\rightarrow b}=\operatorname{Proj}(z_n^a\oplus\chi_n^{a\rightarrow b}).
\tag{R5}
$$

$A$ 为固定高斯矩阵，$\oplus$ 为通道拼接。DPT 使用 $F^{a\rightarrow b}$ 和 reference 的多尺度 encoder 特征。代码仅对 $L=\mathrm{cosim}/\tau$ 做一次 softmax；避免照搬原式 (12)、(31) 而形成 $\operatorname{Softmax}(\exp(L))$。

### Coarse-to-fine refinement

原式 (23)–(25) 在 reference 网格拼接 reference 特征、warp 采样的 target 特征、位移编码及局部相关，预测残差。以下用归一化坐标表示当前实现：

$$
F_s=\phi_s^a\oplus\phi_s^b(\bar W_{\mathrm{in},s})
\oplus g_s\!\left(\beta\odot(\bar W_{\mathrm{in},s}-\bar x_s^a)\right)
\oplus\operatorname{Corr}_s,\qquad
\bar W_s=\bar W_{\mathrm{in},s}+\frac{d_s}{4(W_s,H_s)}.
\tag{R6}
$$

$(d_s,\Delta\ell_s)=\operatorname{Refine}_s(F_s)$，$\ell_s=\ell_{\mathrm{in},s}+\Delta\ell_s$ 是 confidence logit。$\bar W_{\mathrm{in},s}$ 为上一步 warp 插值到当前 reference 网格的结果；第一个 stride 4 阶段直接使用 coarse 预测。$W_s,H_s$ 为当前网格尺寸，$\beta=(W/512,H/512)$。

为兼容 RoMaV2，像素中心归一化为 $\bar u=2(u+1/2)/W-1$，采样和跨尺度插值用 `align_corners=False`。局部相关为 $\langle\phi_s^a,\phi_s^b\rangle/\sqrt{C_s}$，窗口大小为 `7/3/0`。各阶段停止梯度穿过上一阶段 warp/logit，输出再转换回原图 target 像素坐标；插值改变 reference 网格密度，不额外缩放坐标值。

### 几何真值与监督

设 $T_{wa},T_{wb}$ 为 c2w，$T_{ba}=T_{wb}^{-1}T_{wa}$。对 reference 深度 $D_a$，按原式 (26)–(28)：

$$
X_b=R_{ba}\bigl(D_a(x^a)K_a^{-1}\widetilde x^a\bigr)+t_{ba},\qquad
W^{*\,a\rightarrow b}(x^a)=\pi(K_bX_b).
\tag{R7}
$$

$\widetilde x^a$ 为齐次坐标，$\pi$ 为齐次除法。可见性标签 $y$ 按原式 (29) 检查投影在界内、投影/采样深度为正及相对深度误差小于 0.05；训练 mask $m$ 按原式 (30) 额外纳入 reference 深度有效但投影越界的负样本。

NLL 对每个 reference patch $n$ 监督其 GT 投影对应的 target patch $m_n^*$，是原式 (31) 的方向修正版：

$$
\mathcal L_{\mathrm{NLL}}
=-\frac{1}{|\Omega_{\mathrm{patch}}|}
\sum_{n\in\Omega_{\mathrm{patch}}}\log P_{n,m_n^*}^{a\rightarrow b}.
\tag{R8}
$$

密集监督对应原式 (32)–(34)，记 $\Omega_+=\{x:y(x)m(x)=1\}$、$\Omega_m=\{x:m(x)=1\}$：

$$
\mathcal L_{\mathrm{warp}}=
\frac{1}{|\Omega_+|}\sum_{x\in\Omega_+}
\left(\|\widehat W(x)-W^*(x)\|_2^2+\epsilon^2\right)^{\alpha/2},\qquad
\mathcal L_{\mathrm{conf}}=
\frac{1}{|\Omega_m|}\sum_{x\in\Omega_m}\operatorname{BCEWithLogits}(\ell(x),y(x)).
\tag{R9}
$$

默认 $\epsilon=10^{-3}$、$\alpha=0.5$。GT warp 与 mask 用 nearest 下采样，warp 值始终是原图 target 坐标。代码对有效 batch/target 元素取均值，再对 coarse/refinement 阶段取均值，区别于原式 (35) 显式按 target 求和。

当前配置还启用天空零置信度和 pose-ray 扩展：天空只在几何 mask 未覆盖处作负监督；pose-ray 将预测/GT warp 经 $K_b^{-1}$ 转为单位射线，以夹角的鲁棒惩罚约束匹配。两者分别由 `train_sky`、`loss.pose_ray.enabled` 控制，详见 [loss.py](../pi3/models/glob3r/loss.py)。

## 论文差异与关键修正

Warp 的方向错误已通过邮件与原作者确认；本地实现统一 reference→target 的 embedding、DPT、NLL 与 refinement。以下细节仍需结合坐标约定和源码理解：

- **Patch 中心位置**：以零起点整数像素中心计，边长 14 的 patch 几何中心为 $(14j+6.5,14i+6.5)$，整像素采样可能与其相差半像素。当前 NLL 用 nearest 将 GT warp 下采样到 patch 网格，再按图像端点量化 target 标签；这一过程并非显式 patch-center 采样。需分别核对 reference 采样点与 target 标签量化，不能把所有偏差都归为半像素。
- **坐标归一化**：横坐标按宽、纵坐标按高处理。当前 Fourier embedding 在 patch 网格上使用 `linspace(-1,1)`，即横纵分别按 $W'-1,H'-1$ 端点对齐；refinement 使用像素中心归一化 $(2(u+1/2)/W-1,2(v+1/2)/H-1)$。两种约定需通过明确转换衔接，不能直接混用，尤其要检查分辨率和宽高比变化。
- **预测头结构与激活函数**：原式 (18) 只给出 pair feature，完整预测路径见原式 (20)，还接收独立 encoder 特征。本地 DPT 融合 reference 的四层 encoder 特征，残差融合块使用 ReLU，末端为 `Conv → GELU → Conv`；coarse warp 经 sigmoid 后乘 $(W-1,H-1)$，confidence 保留 logits 参与 BCE、输出概率时才 sigmoid。refinement 通过残差更新 warp，不再截断到图像内。原文不足以确定所有内部层与激活位置，需与作者实现进一步核对。

此外，当前实现与完整论文设置有以下差异：

- **Loss reduction**：论文原式 (35) 按 target 求和；本地对有效 batch/target 元素取均值，再对 coarse/refinement 阶段取均值，因此数值尺度可能不同。
- **附加监督**：本地增加几何 mask 外的天空零置信度监督和 pose-ray 射线角度约束，均可通过配置关闭；

方向修正前后的定量结果及附加监督消融见后续 [实验记录](Glob3R训练实验与问题记录.md)

## 训练与评测实现

基于 Pi3 开源仓库复用训练框架，新增功能集中于独立模型、训练器与配置。`Glob3RTrainer` 继承 `Pi3Trainer`，模型通过 forward hook 捕获 encoder 中间特征，不修改 Pi3 主干源码；封装模型显式调用 encoder 和 `decode()`，跳过原有几何 heads。训练时冻结主干，仅更新对应阶段的匹配模块，优化器检查避免主干参数混入。

### 目录结构

```text
Pi3/
├── pi3/models/glob3r/
│   ├── model.py                    # 匹配网络与预测头
│   ├── glob3r_training.py          # 主干封装与分阶段控制
│   ├── geometry.py                 # 几何计算
│   └── loss.py                     # 匹配损失与几何约束
├── trainers/
│   ├── base_trainer_accelerate.py  # 通用训练框架
│   ├── pi3_trainer.py              # Pi3 训练器
│   └── glob3r_trainer.py           # Glob3R 训练器
├── scripts/
│   └── train_glob3r.py             # 训练入口
├── configs/
│   ├── glob3r_coarse.yaml          # 粗匹配训练
│   ├── glob3r_refinement.yaml      # 精细匹配训练
│   ├── glob3r_joint.yaml           # 可选联合微调
│   ├── train_glob3r.yaml           # 顺序两阶段模板，需补齐字段
│   └── accelerate/ddp.yaml         # 分布式配置
├── eval/
│   ├── dataset_eval.py             # 独立评测入口
│   ├── valid.yaml                  # 评测配置
│   ├── model/
│   │   ├── glob3r.py               # 评测模型组装
│   │   ├── pi3.yaml
│   │   └── pi3x.yaml
│   ├── evaluation_utils.py         # 指标与可视化
│   ├── window_adapters.py          # 确定窗口适配
│   └── artifact_store.py           # 结果保存与恢复
└── train.sh                        # Slurm 启动脚本
```

### 训练配置与入口

`scripts/train_glob3r.py` 通过 Hydra 加载配置并调用 `Glob3RTrainer`。训练始终以输入位置 0 为 reference。

| 阶段 | 更新模块 | 学习率 | 每组图像数范围 |
| --- | --- | --- | --- |
| Coarse | matching decoder、DPT 匹配头 | `1e-4` | 2–48 |
| Refinement | fine features、refinement | `5e-5` | 2–24 |
| Joint（可选） | 粗匹配与精细匹配模块 | `5e-5` | 2–24 |

Coarse 与 refinement 默认均为 80 epochs、每 epoch 800 个 micro-batches、梯度累积 2，对应各阶段计划 32K optimizer updates；使用 AdamW、BF16 和 warmup 余弦调度。Joint 默认 10 epochs、warmup 200。当前 loss 除 NLL、warp、confidence 外，还启用可关闭的天空与 pose-ray 监督。

Coarse 指定预训练 Pi3 权重；refinement 默认读取同一 `run_root` 下的 coarse best model。RoMaV2 权重通过 `glob3r.romav2_refinement_checkpoint` 可选加载，仅导入名称和尺寸兼容参数；Joint 需指定起始 matching 权重。冻结范围以 `configure_stage()` 为准。

`train_glob3r.yaml` 提供在同一进程内循环 `stages` 的模板，但当前 refinement 条目未设置 `glob3r.stage=refinement`，且其 backbone 权重为空；`name` 不会自动切换阶段。补齐对应字段后再使用，日常运行优先采用下列独立阶段命令。

### 启动命令

以下命令在仓库根目录执行。先填写训练/验证数据路径；评测另需填写 `eval/valid.yaml` 的数据根目录和索引，删除不用的数据源。权重与运行目录替换为实际路径。

```bash
# 第一阶段：粗匹配训练
accelerate launch --multi_gpu --num_processes 8 \
  scripts/train_glob3r.py --config-name glob3r_coarse \
  run_root=/path/to/run \
  glob3r.backbone_checkpoint=/path/to/pi3.safetensors
```

```bash
# 第二阶段：精细匹配训练
# 默认读取 /path/to/run/coarse/ckpts/best_model/pytorch_model.bin
accelerate launch --multi_gpu --num_processes 8 \
  scripts/train_glob3r.py --config-name glob3r_refinement \
  run_root=/path/to/run \
  glob3r.backbone_checkpoint=/path/to/pi3.safetensors
```

```bash
# 独立评测
accelerate launch --config_file configs/accelerate/ddp.yaml \
  --num_processes 8 \
  eval/dataset_eval.py --config-name valid \
  model=pi3 \
  model.backbone_checkpoint=/path/to/pi3.safetensors \
  model.matching_checkpoint=/path/to/matching_checkpoint.bin
```

评测模型在 `eval/model/glob3r.py` 独立组装，支持 Pi3/Pi3X，统一 RGB-only 输入；窗口、指标与恢复规则见 [评测说明](../eval/README.md)。

服务器上也可使用：

```bash
sbatch train.sh
```

`train.sh` 默认运行 coarse，通过 Slurm `afterok` 依赖在成功结束后运行 refinement。使用前在配置中填写主干权重和数据路径，并核对环境、仓库路径、GPU 资源及 `logs/` 目录。两个阶段共用 `RUN_DIR`，恢复方式见脚本注释。
