# ABot-Recon Stage II 训练数据与监督策略问题分析

本文基于 ABot-Recon Technical Report、当前 Pi3 迁移代码，以及本地配置进行静态审计。分析范围集中在 ARKitScenes、ASE 和 ScanNet 的数据组织、时序采样、监督信号与 Stage II 训练计数。网络结构和推理后端不在本次讨论范围内。

文中的 report 结论来自 [ABot-Recon Technical Report](https://arxiv.org/html/2608.27529v1)。当前代码事实来自本仓库对应实现。服务器上的最终行为仍应以 Hydra resolved config 和运行日志为准。

## 结论摘要

当前最值得优先处理的问题有四项。

1. ARKitScenes 的低分辨率 LiDAR 深度正在参与点图监督，与 report 明确排除这套深度监督的做法冲突。
2. ARKitScenes、ASE 和 ScanNet 的实际时序采样均未严格复现 report 的 Table 10。ScanNet 的偏差最大。
3. 在 8 卡和梯度累积 4 的设置下，当前 38K micro steps 仅产生 9.5K optimizer updates，同时学习率调度已经走完 38K 步。

这几项足以解释 Stage II 总 loss 很快变平、rotation refiner 学习不足、roll 误差持续累积，以及由旋转误差耦合产生的世界系 z 漂移。

## Report 中的数据要求

ABot-Recon 使用 30 个合成和真实数据集。完整采样分布中，合成数据占 62.05%，真实数据占 37.95%。作者提高了长时间连续轨迹的采样概率，同时保留外观、几何和运动类型的多样性。

与当前训练相关的原始权重如下。

| 数据集 | Report 比例 | 对应权重 |
| --- | ---: | ---: |
| ARKitScenes | 1.34% | 134 |
| ASE | 2.69% | 269 |
| ScanNet | 2.46% | 246 |

三个数据集在论文完整混合数据中合计占 6.49%。当前模板中的权重 `134:269:246` 与 Table 1 一致。只保留这三个数据集后，归一化采样比例变为下表。

| 数据集 | 三数据集内部比例 |
| --- | ---: |
| ARKitScenes | 20.65% |
| ASE | 41.45% |
| ScanNet | 37.90% |

内部比例保持了 report 的相对关系，但整体训练分布已经发生明显变化。当前数据主要覆盖室内手持、头戴式合成和室内 RGB-D 运动，缺少 Waymo、KITTI-360、DL3DV、TartanAir 等长轨迹和室外运动。对于车辆轨迹或大尺度闭环测试，这种数据分布无法复现 report 的运动先验与尺度覆盖范围。

## 时序采样审计

Report 的 Appendix A.1 将视频采样分为 forward fixed-stride、adaptive forward 和 foldback。Table 10 对三个数据集给出了明确策略。

| 数据集 | Report 策略 | Report 间隔 | 当前行为 |
| --- | --- | ---: | --- |
| ARKitScenes | foldback | 1 | 源序列足够长时使用 adaptive forward |
| ASE | foldback | 1 到 2 | 长序列使用 adaptive forward，短序列 foldback 固定为步长 1 |
| ScanNet | adaptive forward | 1 到 30 | 父数据集随机抽取局部帧，wrapper 只负责排序 |

### ARKitScenes

当前 wrapper 仅在源序列短于目标帧数时进入 foldback。长度大于等于 128 的 ARKitScenes 序列会采用固定步长 1 的 forward clip。输出虽然保持时间顺序，但不属于 report 指定的 foldback 采样策略。

### ASE

当前 ASE 配置声明 `frame_step_range: [1, 2]`。当源序列足够长时，wrapper 会在 1 和 2 中选择一个固定步长并执行 adaptive forward。当源序列不足时，`foldback()` 内部没有接收 `frame_step_range`，实际始终按步长 1 前后遍历。

因此，配置中的步长 2 只会出现在 forward clip 中，不会出现在真正的 foldback segment 中。这与 report 描述的 foldback 过程不一致。Report 要求每段从区间中采样步长，到达边界后反向并为下一段重新采样步长。

### ScanNet

当前 `ScannetDataset` 没有 `frame_step` 属性。wrapper 因此无法执行自己的 adaptive forward，只能调用父数据集的 `_get_views()`。

父数据集先在一个局部范围中随机或分层选择帧。`sort_views: true` 随后按 frame id 排序。这个过程能恢复时间递增顺序，却无法保证相邻帧间隔落在 `[1, 30]` 中。随机分支还可能产生重复帧和较大的相邻跨度。

Rotation refiner 依赖最近 12 个相邻运动和视觉特征。相邻间隔失控会改变它看到的运动速度、旋转幅度和视觉重叠分布，因此会直接影响 Stage II 的训练目标。

### 当前 foldback 的额外差异

当前 wrapper 的 foldback 还包含 0.05 的随机原帧重复概率，最多连续重复一次。这个参数来自迁移实现，report 没有给出对应策略。

一个 128 帧 foldback clip 平均会包含约 6 个额外 identity transition。它们的相对位姿接近单位变换，会降低有效运动监督密度。是否保留需要通过消融验证，不能视为论文默认配置。

## ARKitScenes 深度监督

Report 在 Appendix A.2 明确指出，ARKitScenes 的低分辨率 iPhone LiDAR 深度精度不足，因此不使用这套深度进行监督。

当前加载路径会执行以下过程。

1. 读取 `lowres_depth`。
2. 将 RGB 放入深度对应的 384 × 288 画布。
3. 根据深度和内参生成 `pts3d` 与 `valid_mask`。
4. 在 `ABotReconLoss` 中计算点图损失。

这与 report 的监督策略直接冲突。噪声深度会通过 `L_pts` 更新点图解码器，也会参与 Pi3 的尺度对齐。它可能损害局部几何、点云形状和预测点图与位姿平移之间的尺度一致性。

ARKitScenes 的 RGB 和 camera-to-world pose 仍然可以用于相邻位姿监督。如何在不使用深度残差的情况下处理 translation scale，需要单独设计数据集级监督 mask。Report 没有公开这部分训练代码，因此不能凭空指定作者的内部实现。

## 点图与法向监督路由

当前 ABot loss 复用了 Pi3 `PointLoss` 的数据集质量分组。三个数据集的实际路由如下。

| 数据集 | 点图监督 | 法向监督 |
| --- | --- | --- |
| ARKitScenes | 开启 | 关闭 |
| ASE | 开启 | 关闭 |
| ScanNet | 开启 | 开启 |

ScanNet 通过名称映射进入 Pi3 的 middle-quality 分组。ASE 的数据集名称为 `AriaSyntheticEnvironmentsPi3X`，不在 Pi3 原始质量列表中，因此 `normal_loss` 静默返回零。

Report 保留了 Pi3 的点图和法向监督，并没有说明 ASE 应当关闭法向损失。ASE 使用合成深度，当前静默关闭法向监督没有 report 依据。ARKitScenes 因低分辨率深度不参与几何监督，其法向监督关闭是合理结果。

## Stage II 更新次数

Report 的 Stage II 使用 128 帧序列，global batch 为 32，在 32 张 MI308 上训练 38K iterations。每张卡每次处理一条序列，没有给出梯度累积设置。对应的训练量是 38K optimizer updates 和约 121.6 万条训练序列。

当前 8 卡配置使用

```yaml
gradient_accumulation_steps: 4
num_epoch: 38
iters_per_epoch: 1000
```

global batch 的计算结果正确。

$$
B_{\mathrm{global}}=8\times 1\times 4=32
$$

当前 trainer 中的 `iters_per_epoch` 统计 dataloader micro steps。每四次前后向才进行一次有效参数更新，因此 optimizer update 数量为

$$
N_{\mathrm{update}}=\frac{38\times1000}{4}=9500
$$

当前训练只完成了 report 四分之一的参数更新，也只看到了约 30.4 万条序列。

此外，公共 trainer 设置 `step_scheduler_with_optimizer=False`，并在每个 micro step 后调用 scheduler。学习率会在 38K micro steps 内完成整条退火曲线，而模型只有 9.5K 次参数更新。后半程 refiner 得到的有效学习率会明显低于按 38K updates 设计的训练策略。

在不修改公共 trainer 的前提下，8 卡和累积 4 要获得 38K optimizer updates，需要 152K micro steps。若保持每个 epoch 1000 个 micro steps，则需要 152 个 epoch。学习率调度的 total steps 也应对应 152K micro steps，使每四个 micro steps 的更新点沿着等价的 38K-update 曲线前进。

## Stage II 参数更新范围

Report 指出 Stage II 对新加入的 rotation refiner 使用峰值学习率 $5\times10^{-5}$，其余模型参数使用峰值学习率 $2\times10^{-5}$。当前 `stage2.yaml` 设置 `freeze_encoder: true`，导致 encoder 不属于其余可训练参数。

Report 没有描述 Stage II 冻结 encoder。`remaining model parameters` 更自然地对应除新 refiner 外的全部原模型参数。因此当前冻结 encoder 缩小了论文描述的微调范围。这项偏差可能降低模型对 128 帧训练分布的适应能力，但它的影响优先级低于更新次数和数据监督错误。

## Loss 与尺度


当前 pose loss 使用最大时间间隔 11，与窗口大小 $K=12$ 对应。

$$
\mathcal{P}=\left\{(i,j)\mid 0\leq i<j<N,\ j-i\leq 11\right\}
$$

当前实现对 composed relative pose 计算 translation Huber loss 和 SO(3) geodesic rotation loss。间隔权重 $\alpha_{ij}$ 只作用于旋转项，与 report 的 Equation 14 一致。


### 原始点云尺度

Pi3 point loss 先求预测点图到 GT 点图的最优尺度 $S_{\mathrm{opt}}$，再计算几何误差。

$$
\mathcal{L}_{\mathrm{pts}}
=
\left\lVert
S_{\mathrm{opt}}P_{\mathrm{pred}}-P_{\mathrm{gt}}
\right\rVert
$$

Pose translation 也乘相同的 $S_{\mathrm{opt}}$ 后再接受监督。这个目标约束点图与位姿平移的相对尺度，却不会赋予原始输出唯一的 metric scale。Report 的轨迹评测使用 Umeyama Sim3 对齐，稠密重建也先执行 Umeyama 和 ICP 对齐。因此原始点云绝对尺度不准不能单独证明 Stage II 训练失败。

之前 ABot loss 复用了 `PointLoss`，但没有执行 Pi3 `Pi3Loss.normalize_pred()` 中的预测尺度归一化。最终对齐几何在理想情况下近似等价，反向传播的尺度梯度却不同。当前原始点图尺度更容易沿无约束方向漂移。这是观察 raw point scale 时需要处理的实现差异。
当前 ABot loss 已严格迁移 Pi3 `Pi3Loss.normalize_pred()` 的预测尺度归一化。PointLoss 前先用有效预测点的平均距离归一化 local point map，并以相同因子归一化 camera translation。无效点置零、归一化因子和原地更新顺序均与 Pi3 保持一致。

## Roll 与世界系 z 漂移

全局平移通过相邻位姿递推得到。

$$
t_{0\leftarrow i}
=
t_{0\leftarrow i-1}
+
R_{0\leftarrow i-1}t_{i-1\leftarrow i}
$$

持续的 roll 偏差会旋转后续所有局部平移，将原本处于水平面的运动投影到世界系 z 方向。因此当前测试中的 z 漂移可能主要由 roll 累积产生，不必先假设 translation head 存在独立的 z 偏置。

结合当前实现，较合理的因果链如下。

1. Stage II 仅完成 9.5K 次 optimizer updates。
2. ScanNet 的相邻帧跨度不受 `[1, 30]` 约束。
3. ARKitScenes 的噪声深度更新点图分支并参与尺度对齐。
4. ASE 和 ARKitScenes 没有按 report 的 foldback 规则生成训练轨迹。
5. Rotation refiner 没有充分学到稳定的局部旋转修正。
6. 小幅 roll 误差经过长序列 composition 后转化为轨迹不闭合和世界系 z 漂移。

## 为什么总 loss 看起来不下降

Stage II 从已经训练完成的 Stage I 初始化。新增加的 rotation refiner 只影响 pose rotation 和 smoothness，点图损失仍然占据总 loss 的主要部分。即使 refiner 在改善旋转，总 loss 的下降幅度也可能很小。

当前三数据集训练还会放大这种现象。ARKitScenes 的点图 loss 含有低质量深度噪声，ASE 的 normal loss 恒为零，ScanNet 的 sampling gap 分布不稳定。不同数据集之间的 loss 方差可能大于 refiner 带来的平均改进。

因此 Stage II 应重点观察下列指标，而不能只观察 total loss。

```text
每个数据集独立的 local_pts_loss
每个数据集独立的 normal_loss
相邻帧 rotation error
gap 2 到 11 的 composed rotation error
raw pose 与 refined pose 的 rotation error 差值
roll drift 与 world-z drift
optimizer update count
当前 optimizer update 对应的 learning rate
```
