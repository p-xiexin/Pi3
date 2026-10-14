# ABot-Recon 官方训练代码对照审阅

> **状态更新（2026-10-14）：按用户要求，本次审阅引入的代码、配置和测试修改已全部撤回，恢复至审阅前基线 `dbd9fe7`；本文保留为历史审阅记录。下文“当前修正”“已修正”及测试通过结果均描述撤回前的实现，不代表当前工作区状态。新增文件及测试已移除，相关链接和复验命令仅供历史追溯。其他独立工作区改动未撤回。**

审阅日期：2026-09-26。对象为本仓库 `pi3/models/abot_recon/` 及其实际调用的数据、训练器代码。修正前基线：`dbd9fe793ad4bec1cc60145d52228b58a10ddd27`；“当前修正”指本次代码审阅后工作区中的实现。

官方参照固定为 [`amap-cvlab/ABot-Recon` 的 `train` 分支，提交 `24d427bbd2eeb6072140de5683796ba626995e18`](https://github.com/amap-cvlab/ABot-Recon/tree/24d427bbd2eeb6072140de5683796ba626995e18)。以下结论针对这个提交，不随远端分支变化自动更新。

**模型主体复现基本正确；需要修正的内容主要是数据时序与增强、训练执行，以及 loss 监督。** 本文按两个部分展开：第一部分说明数据如何组织、采样和预处理；第二部分说明模型如何设计和训练，以及输出如何接受监督。原审阅的 R01–R18 编号保留用于追溯，跨主题的 R18 分别说明数据接口和训练阶段两部分。

官方公开的是训练实现、开发配置默认值和微调示例。[官方配置](https://github.com/amap-cvlab/ABot-Recon/blob/24d427bbd2eeb6072140de5683796ba626995e18/configs/finetune.yaml) 明确区分 `[SOURCE]` 和 `[RELEASE]`，并声明不是论文完整配方。因此，公开示例的学习率、数据混合与训练轮数不能直接解释为论文 Stage I–III 的完整设定。

P1 表示会显著影响监督、训练或推理正确性；P2 表示实现差异、边界问题或可复现性问题。与官方不同的选择并不都属于数学错误。前次代码修正已完成 **53 项测试，其中 12 项直接对照官方源码，另有双进程 Gloo 检查通过**；本文按主题归属列出这些证据。本次调整仅重组文档，不代表新增一次模型训练或基准评估。

## 一、数据集处理

### 1.1 数据来源、格式与训练/验证划分

先区分数据分布与读取格式，再判断预处理是否对齐。模型和 loss 一致，并不能消除训练数据来源、采样概率和验证划分不同带来的影响。

| 项目 | 官方公开实现 / 示例 | 本地当前实现 | 审阅结论 |
| --- | --- | --- | --- |
| 默认训练来源 | 公开微调示例为 processed DL3DV:TartanGround=4:3 | 阶段配置默认 KITTI；`dataset.yaml` 为本地多来源配置片段 | 数据混合尚未对齐，不能据此声称复现论文完整数据分布 |
| 数据格式和过滤 | 提供多种 CUT3R 适配器及黑名单 | 保留 Pi3 读取器、原始目录和索引格式 | 未把本地数据目录转换为官方 processed 格式，也未补齐全部 30 个数据集 |
| ARKitScenes | 支持显式 camera-only 的序列标志 | wrapper 输出 `camera_only`，但原始索引/读取器仍要求原有 RGB-D 文件布局 | 监督模式已可区分，读取器仍非纯 RGB 输入流程；平移监督详见 2.2 节 |
| 验证来源与片段 | held-out TartanGround；相同场景/起点的 128 帧 s1、s6 两组流式验证，并计算两组均值 | 保留本地验证配置；Stage II 限制为 8 个 128 帧批次 | 两边的 `val/loss` 不能直接比较 |

当前没有真实训练数据目录，无法确认本地 split、训练/验证泄漏、有效深度比例或数据质量。对齐训练分布还需要逐数据源核对索引、采样逻辑和 blacklist，不能只调整总样本数或一个 stride 参数。

参照：[官方数据基类](https://github.com/amap-cvlab/ABot-Recon/blob/24d427bbd2eeb6072140de5683796ba626995e18/abot_recon/training/datasets/base.py)、[官方公开配置](https://github.com/amap-cvlab/ABot-Recon/blob/24d427bbd2eeb6072140de5683796ba626995e18/configs/finetune.yaml)。本地入口为 [dataset.yaml](dataset.yaml)、[stage1.yaml](stage1.yaml) 和 [stage2.yaml](stage2.yaml)。

### 1.2 时序采样与序列构造

流式重建要求每个样本中的顺序代表一条采样轨迹。随机挑选若干图像再按文件名排序，不能保证连续运动或指定帧间隔；短序列补齐也需要区分允许往返的来源与驾驶数据。

| 编号 | 等级 | 修正前问题 / 影响 | 当前修正 |
| --- | --- | --- | --- |
| R09 | P1 | TartanAir、ScanNet 原读取器随机挑帧后排序，不能保证指定步长；短序列还可能从重复随机帧构造 foldback | 为两读取器增加可选 `frame_step`，ABot 配置启用真正的顺序采样；普通 Pi3 不启用时保持原采样方式 |
| R10 | P1 | 通用短序列倒放也作用于 KITTI/Waymo 等驾驶数据 | 增加 `forward_only`；KITTI、Waymo、NuScenes 配置过滤过短片段而不倒放。可行跨度不足时自适应降低步长，但不低于 1 |
| R12 | P2 | 包装器各 worker 的 NumPy RNG 初始状态相同；父包装层不向子数据集传 epoch；持久 worker 看不到普通 Python 属性更新 | 包装器按 worker seed 建立独立流，共享 epoch seed；ABot trainer 递归通知实际序列包装器；显式校验帧数、步长和 shuffle/random sampling 约束 |

修正后，TartanAir、ScanNet 的 ABot 配置通过 `frame_step` 进入真正的顺序采样路径；不启用该选项时，普通 Pi3 配置仍保留原行为。KITTI、Waymo、NuScenes 使用 `forward_only=true`，不足目标长度的序列被过滤，不再通过倒放补齐。

**采样概率分布仍有差异。** 官方按数据源使用随机起点、fixed/variable stride、折返时更换步长以及最近 stride 排除等策略；本地通用 wrapper 仍是可行固定步长加短序列往返。BlendedMVS 的图路径属于多视图伪序列，不能仅按文件名排序就视为真实时间序列。

本地实现：[data_sampler.py](data_sampler.py)、[TartanAir 读取器](../../../datasets/tartanair_dataset.py)、[ScanNet 读取器](../../../datasets/scannet_dataset.py)。

### 1.3 几何预处理与数据增强

RGB、深度和相机内参必须使用同一个几何变换。对于长时序样本，还需控制不同帧之间增强参数的相关性，避免每帧独立裁剪、颜色扰动改变模型实际看到的运动和外观。

| 编号 | 等级 | 修正前问题 / 影响 | 当前修正 |
| --- | --- | --- | --- |
| R11 | P1 | 本地训练始终走普通缩放裁剪；没有官方 90% 保留 FOV 与 20% 序列一致增强 | 数据基类增加默认关闭的可选增强；ABot 训练配置使用 0.9/0.2、aug_crop=16；RGB-D/内参同步变换；颜色增强只作用于真实内容，padding 恢复精确均值，padding 深度为 0 |

当前 ABot 训练配置使用 `preserve_fov_prob=0.9`、`sequence_consistent_aug_prob=0.2`、`aug_crop=16`。数据基类新增选项默认关闭，因此其他未启用这些选项的 Pi3 配置保留原预处理行为。

保留 FOV 的路径固定输出宽度，按比例缩放后仅在竖直方向裁剪或补边；内参随实际栅格尺寸和偏移同步更新。补边深度为 0，不作为有效几何监督；颜色增强只作用于真实图像内容，最后恢复精确的 ImageNet RGB 均值补边。

FOV 图像、深度和内参已与官方输出对照，序列共享增强也已实现。本地仍使用 Pi3 的颜色/JPEG/模糊变换和自己的 seed 管理，因此不保证相同数值 seed 在两套训练代码中产生逐样本、逐位相同的增强结果。

本地实现：[数据基类](../../../datasets/base/base_dataset.py)、[序列增强](../../../datasets/base/sequence_augmentation.py)。官方依据为[数据基类中的预处理与增强流程](https://github.com/amap-cvlab/ABot-Recon/blob/24d427bbd2eeb6072140de5683796ba626995e18/abot_recon/training/datasets/base.py)。

### 1.4 数据输出接口与可视化

当前训练器使用逐帧字典列表。collate 后，每帧包含 RGB `img`、世界坐标点 `pts3d`、`valid_mask`、camera-to-world 位姿 `camera_pose`、逐样本 `dataset`，以及由 wrapper 提供的 `camera_only`。

`source_frame_id` 保留来源编号，`frame_id` / `temporal_index` 表示当前采样轨迹中的位置。进入模型和 loss 前应保留列表顺序，不能按来源编号再次排序，否则会破坏 foldback 的往返轨迹。

**R18（数据接口部分，P2）**：旧测试和数据可视化配置引用了已删除的 `data.py` 或 dataset aliases。已将测试迁移到实际 list-of-views / wrapper API，恢复 [data_viz.py](data_viz.py) 入口，修正 KITTI 可视化配置，并补齐 matplotlib 依赖。`camera_only` 如何控制点图、法向、平移和置信度监督，在 2.2 节统一说明。

历史文档 [TRAINING_DATA_STRATEGY_ANALYSIS.md](TRAINING_DATA_STRATEGY_ANALYSIS.md) 保留供追溯；其中旧数据路由和增强结论不再作为当前实现规范。

### 1.5 数据处理的验证与剩余差异

| 已验证内容 | 证据 |
| --- | --- |
| 顺序步长、短序列过滤、foldback 顺序与标记、worker/epoch 随机流、验证采样可复现性、可视化深度坐标 | [数据回归测试](../../../tests/test_abot_recon_kitti_dataset.py) |
| 序列几何/颜色增强一致性、内参投影一致性、无效深度补边和精确均值颜色补边 | [增强回归测试](../../../tests/test_abot_recon_augmentation.py) |
| FOV 处理后的图像、深度和内参与官方一致 | [官方源码差分测试](../../../tests/test_abot_recon_official.py) |

这些测试验证的是数据处理操作本身。完整数据覆盖、逐来源采样分布、颜色增强随机流和官方 s1/s6 验证协议仍有差异；真实数据质量和划分尚未验证。

## 二、训练策略

### 2.1 模型设计

本节依次讨论网络结构、流式状态、预测精度与权重初始化，再说明训练阶段、参数更新和续训管理。数据如何生成已在第一部分说明；各预测量的监督规则统一放在 2.2 节。

#### 2.1.1 网络结构：已经确认一致的部分

| 检查项 | 本地与官方对照结论 |
| --- | --- |
| 图像编码 | DINOv2 ViT-L/14，ImageNet RGB 均值和标准差，一致 |
| 主解码器 | 36 个交替的帧内/跨帧块，1024 维、16 头，一致 |
| 特殊 token | 5 个 register/pose tokens；末两层特征拼接，一致 |
| 注意力门控 | 每个主解码层 `sigmoid(gate_proj(x))`，无 bias，一致。官方日志中出现的 `bias=True` 文案与其实际构造不一致，应以代码为准 |
| 位置编码 | 帧内 2D RoPE；跨帧 3D RoPE 的维度 `[20,22,22]`、theta=10000、特殊 token 空间偏移，一致 |
| 因果窗口 | 当前帧与前 K−1 帧；K=12，一致。每层的窗口相同，不意味着整网仅有 12 帧有效感受野 |
| 点图 | 预测 `xy` 和 `log_z`，`z=exp(min(log_z,10))`，再生成 `[xy*z,z]`，一致 |
| 相邻位姿 | pose token 的 MLP 后均值；关系描述符 `[prev,curr,curr-prev,curr*prev]`；直接平移、scalar-last quaternion，一致 |
| 位姿累计 | 首帧单位阵，按 `T_prev @ delta` 累积 c2w，一致 |
| 旋转修正 | 视觉交叉注意力、frame role/age embedding、kernel=10 的 depthwise gated TCN、逐分量 `2°*tanh` 残差、一致 |
| 修正乘法 | `R_corrected = R_raw @ Exp(residual)`，平移不修改，一致。2° 是各分量的界，向量模长不一定小于 2° |
| checkpoint 结构 | 在 meta device 上构造完整双方模型，包含 confidence 分支时，所有 state_dict 键和张量形状完全相同 |

参照：[官方模型构造](https://github.com/amap-cvlab/ABot-Recon/blob/24d427bbd2eeb6072140de5683796ba626995e18/abot_recon/model.py)、[流式网络](https://github.com/amap-cvlab/ABot-Recon/blob/24d427bbd2eeb6072140de5683796ba626995e18/abot_recon/modeling/streaming/network.py)、[相邻位姿头](https://github.com/amap-cvlab/ABot-Recon/blob/24d427bbd2eeb6072140de5683796ba626995e18/abot_recon/modeling/pi3/models/layers/adjacent_pose_head.py)。本地实现为 [model.py](model.py)、[pose_head.py](pose_head.py) 和 [rope3d.py](rope3d.py)。

以上结果说明架构与参数结构已经对齐。完整 state_dict 的键和形状一致，不等于已经运行过完整公开权重，也不等于实际重建精度已达到官方结果。

#### 2.1.2 预测精度、流式状态与初始化修正

| 编号 | 等级 | 修正前问题 / 影响 | 当前修正 |
| --- | --- | --- | --- |
| R01 | P1 | 预测特征 `.float()` 后仍在外层 autocast 中，Linear/矩阵乘法会再次降精度；长位姿链尤其敏感 | 点/置信度线性头、位姿头、累计和点图世界变换使用显式禁用 autocast 的 FP32 区域，损失同样如此 |
| R08 | P1 | `infer_paths` 一次编码整段；没有持久 K/V、绝对帧位置或相机状态，只是全段因果计算 | 增加 `inference_step` 与 `iter_stream_paths`；每个全局层保留最多 K−1 帧 K/V，携带上一帧视觉特征、描述符、累计位姿和 TCN 状态；绝对时间位置跨调用递增；缓存 clone 避免小切片占住整段存储 |
| R16 | P2 | ABot checkpoint 只检查两个 pose 参数后 `strict=False`，缺失 backbone/gate 可静默留下随机权重；Pi3 初始化 confidence clone 发生在点分支加载前 | 检查除显式增删 confidence 外的完整键集合和形状；Pi3 权重载入后再复制 confidence decoder；拒绝编码器维度不匹配的 small/base 解码配置 |

修正后的整段 `forward()` 保留可微训练路径；增量推理使用 `inference_step` 和显式状态，每个全局层保留最多 K−1 帧 K/V，同时保存上一帧描述符、视觉特征、累计位姿和 TCN 特征。时间 RoPE 的绝对帧位置跨调用递增，新场景将状态设为 `None`。

调用方式为 `model.eval()` 后执行 `model.inference_step(images, state)`，并把输出中的 `stream_state` 交给下一次调用。长目录序列可使用 `iter_stream_paths`，逐帧消费输出而不积累历史图像或点图；原 `infer_paths` 仍是整段接口。

补齐的是有界 SDPA 状态，没有移植 FlashInfer paged allocator、FlexAttention 编译加速、camera-only 快速解码路径或稀疏回环检索/PGO。前三者主要影响执行效率，回环属于推理后处理，当前测试不支持同等 FPS 或长程闭环精度的结论。

#### 2.1.3 训练阶段与参数更新范围

| 配置 | 初始化与训练范围 | 片段长度 | 学习率与训练计数 | EMA |
| --- | --- | --- | --- | --- |
| 本地 Stage I | 从 Pi3 初始化；禁用 rotation refiner 和 confidence | 32 帧 | 1e-4；32×1000 micro-batches，累积 1 | 默认关闭 |
| 本地 Stage II | 从完整 Stage I 初始化；冻结 encoder，启用 refiner | 128 帧 | base=2e-5、refiner=5e-5；38×1000 micro-batches，累积 4 | 开启 |
| 公开微调设置对应的本地 `finetune` | 从完整 ABot 权重初始化；冻结 encoder | 32 帧 | 1e-6；20×800 micro-batches，累积 1 | 开启 |
| 本地 `stage3` 示例 | 从 Stage II / 公开模型初始化，仅训练 confidence decoder/head | 继承 `finetune` | 继承公开微调的 LR/OneCycle 设置 | 开启 |

Stage II 的 38×1000 是 micro-batches，累积 4 后为 **9500 个更新窗口**，不是 38000 次 optimizer step；其中异常窗口会被跳过，实际成功更新数可能更少。公开微调示例也不能用于倒推论文所有阶段的完整训练量。

**R18（训练阶段部分，P2）**：原来没有可直接组合的 confidence-only 配置。当前新增 [stage3.yaml](stage3.yaml)，冻结其他参数，包括后创建的 attention gate；checkpoint 不含 confidence 分支时，从已加载的 point decoder 初始化 confidence decoder。该配置是可运行的本地示例，不是经验证复原的论文 Stage III 配方。

原有阶段配置保留为本地训练方案，另提供 [finetune.yaml](finetune.yaml) 对应官方公开微调的优化设置。其数据仍采用第一部分说明的本地接口，不能把这个配置名称理解为整套官方训练流程已经复现。

#### 2.1.4 训练执行：优化器、异常窗口、EMA 与续训

| 编号 | 等级 | 修正前问题 / 影响 | 当前修正 |
| --- | --- | --- | --- |
| R13 | P1 | 大 loss 乘零后仍执行 AdamW，动量/衰减/EMA 继续变化；多卡各 rank 独立判断，无限梯度未检查 | ABot 独立 epoch loop 同步任意 rank 的异常标志；任一 micro-batch 异常则丢弃完整累积窗口；检查非有限梯度/梯度范数；不执行该窗口 optimizer/EMA；按官方规则消耗对应 schedule slot |
| R14 | P1 | `auto_resume=false` 被公共基类忽略；保存 `checkpoint_9` 后续训从 epoch 9 开始，重跑已完成一轮 | ABot 自己处理恢复；尊重显式关闭；`checkpoint_9` 从 epoch 10 继续；旧无 EMA checkpoint 可从恢复后的 online 权重初始化 EMA |
| R15 | P2 | optimizer 配置中的 betas 实际被共享 builder 忽略；日志与配置不一致；schedule/训练计数混用 micro/update | 显式传入 betas，默认设置为官方有效 beta2=0.999；支持 gate/confidence LR 分组；ABotOneCycleLR 按 optimizer window 计数，总槽位为 updates+1；不相容旧 schedule 拒绝自动加载 |

当前 scheduler 按 optimizer window 计数，各参数组保留自己的峰值 LR，总槽位为 `updates+1`；与官方相同，异常窗口也消耗一个 schedule slot，但不会执行该窗口的 optimizer/EMA 更新。

本地 EMA 仍保存完整模型副本，官方默认只保存可训练参数。数学更新相同，但内存开销不同。已有运行检查覆盖 CPU、DDP/Gloo；未验证 FSDP，新异常窗口处理显式不支持 DeepSpeed。

权重和训练状态迁移应区分处理：

- 新 Stage I/II 训练继续使用原入口，命令见 [README](README.md)，使用新的实验名避免覆盖已有目录。
- 旧模型参数名保留，可通过 `model.ckpt=... train.auto_resume=false` 初始化新实验。旧 optimizer/scheduler state 不保证续训等价，因为损失、增强、epoch 和 schedule 语义已有修正；不相容 schedule 会明确拒绝恢复。
- 官方完整模型权重可用于 `--config-name finetune model.ckpt=...`。官方 trainable-only `ema.pt` 不是完整模型，应先按官方 merge EMA 工具合并。
- 修改实现不会自动修复已经产生的训练权重；既往结果需要重新训练，或评估后再微调。

参照：[官方训练器](https://github.com/amap-cvlab/ABot-Recon/blob/24d427bbd2eeb6072140de5683796ba626995e18/abot_recon/training/trainer.py)。本地实现为 [train.py](train.py)、[training_loop.py](training_loop.py) 和 [ema.py](ema.py)。

#### 2.1.5 模型与训练执行的验证

| 已验证内容 | 证据 |
| --- | --- |
| 非零旋转修正下的相邻位姿输出/梯度、多个绝对时间偏移的 RoPE、完整 checkpoint 键和形状 | [官方源码差分测试](../../../tests/test_abot_recon_official.py) |
| K=1/K=3、batch=2，整段/逐帧/不等长分块输出一致；缓存底层存储有界；场景重置与形状变化拒绝 | [模型回归测试](../../../tests/test_abot_recon_regressions.py) |
| BF16 autocast 下预测头 FP32、梯度 checkpoint、EMA 存取、confidence-only 冻结、坏 checkpoint 拒绝 | [模型回归测试](../../../tests/test_abot_recon_regressions.py)、[训练回归测试](../../../tests/test_abot_recon_training.py) |
| 异常窗口不触发 AdamW/EMA hook、schedule 更新计数、auto_resume 与 epoch 恢复 | [优化器回归测试](../../../tests/test_abot_recon_optimizer.py) |
| 仅 rank 0 一个 micro-batch 超阈值时，两 rank 同步跳过该窗口，随后均完成两次更新，最终参数一致 | [双进程 Gloo 检查](../../../tests/abot_recon_ddp_smoke.py) |

Stage I、Stage II、finetune、Stage III 的 Hydra 配置、loss 和 optimizer 组合构造也已检查。完整公开权重的图像重建、真实训练收敛、CUDA BF16/FP16 内核差异、GPU 内存与 FPS、论文基准精度均尚未验证；前次审阅没有下载大模型权重或启动真实训练。

### 2.2 loss 监督

本节按“尺度与坐标 → 点图/法向 → 位姿 → 旋转修正 → 置信度 → 总目标”说明监督。核心区别是：RGB-D 用点图对齐预测尺度，camera-only 用轨迹对齐尺度，但两者都保留相机平移监督。

官方依据为[训练损失实现](https://github.com/amap-cvlab/ABot-Recon/blob/24d427bbd2eeb6072140de5683796ba626995e18/abot_recon/training/loss.py)，本地实现为 [loss.py](loss.py)。

#### 2.2.1 坐标系、尺度归一化与 camera-only

RGB-D 的 GT 首先转换到首帧相机坐标系，再除以有效世界点到该原点的平均距离；模型预测局部点按有效点平均模长归一化，同时缩放预测相机平移。随后每个序列最近邻重采样 4096 个有效点，用逆深度加权 L1 求尺度。**归一化保留梯度，尺度求解不反传**，点图和位姿平移使用同一个对齐尺度。

| 编号 | 等级 | 修正前问题 / 影响 | 当前修正 |
| --- | --- | --- | --- |
| R02 | P1 | ARKitScenes 按名称跳过所有点和平移监督，只剩旋转；没有官方 camera-only 的轨迹归一化/尺度对齐 | 增加逐序列 `camera_only` 标志；轨迹归一化，邻接运动长度比的 log-MAD 稳健尺度估计；继续监督平移。旧 ARKit 标签作为无显式标志时的兼容默认，显式 false 可启用 RGB-D |

camera-only 不把占位深度当监督。GT 与预测各自按首帧坐标系中的相机平移模长均值归一化；邻接运动长度的 GT/pred 比值在 log 空间进行 median/MAD 异常值剔除，保留值的几何均值作为 detached 尺度。不足两对有效运动时尺度取 1。

因此，`camera_only=true` 意味着关闭点图、法向和置信度监督，并不意味着关闭平移监督。该标志与原始数据读取格式的区别见 1.1 节。

#### 2.2.2 点图与法向监督

点图使用尺度对齐后的逐坐标 L1，逆深度权重为：

```text
weight = 1 / max(max(z_gt, 1e-6), 0.1 * mean_valid_depth)
```

旧实现的分母额外加 `1e-6`；当前已与官方的 clamp 写法一致。无效点不参与点图残差和尺度求解。

| 编号 | 等级 | 修正前问题 / 影响 | 当前修正 |
| --- | --- | --- | --- |
| R06 | P2 | 法向监督用排除名单，Hypersim、BlendedMVS、ASE 等都可能被默认纳入；与官方当前训练名单不符 | 采用官方白名单：TartanGround、TartanAir、PointOdyssey、ScanNet、VKITTI2，映射本地别名；其他来源默认不监督法向，可显式配置 `normal_datasets` |

法向监督不是对所有有深度的数据自动启用。当前采用官方发布版本的白名单并映射本地别名；是否为其他数据增加该项，应通过 `normal_datasets` 明确配置，避免数据集换名后意外改变监督。

#### 2.2.3 组合位姿监督

默认先从累计 c2w 位姿构造相对变换，监督 `i<j` 且 `j-i<=11` 的 pair；预测平移先应用与几何一致的对齐尺度。平移采用 delta=0.1 的逐坐标 Huber，旋转采用 SO(3) 角度。

| 编号 | 等级 | 修正前问题 / 影响 | 当前修正 |
| --- | --- | --- | --- |
| R07 | P2 | pose gap 指数默认 0.5，官方开发默认为 0.75；原稳定 atan2 与官方截断 acos 的数值/梯度不完全一致 | 默认改为 0.75 和官方 acos；保留可设 gap 指数，支持 dense/causal_upper pair 与 `max_pair_gap<=0`。acos 截断使完美旋转也有约 0.001424 rad 下界，这是官方数值约定 |

旋转 pair 权重为：

```text
rotation_pair_weight = (j-i)^0.75 / mean_over_pairs((j-i)^0.75)
L_pose = 100 * L_translation + L_rotation
```

平移默认没有 gap 加权。总损失再对 `L_pose` 乘 0.1。acos 截断的正下界属于官方数值约定；旧 atan2 写法本身不能仅因不同就被判定为数学错误，这里改动的目的在于对齐公开实现的数值和梯度。

#### 2.2.4 旋转修正的幅值和平滑正则

令相邻旋转修正向量为 `r`：

```text
M = mean(sum_xyz(r²))
S = mean(sum_xyz((r[t+1]-r[t])²))
L_correction = 0.0001 * (M + S)
```

| 编号 | 等级 | 修正前问题 / 影响 | 当前修正 |
| --- | --- | --- | --- |
| R04 | P1 | 原旋转修正正则为 `0.1 * mean_xyz(r²)`，官方为 `camera_weight(0.1) * 1e-3 * mean(sum_xyz(r²))` | 改用向量平方模；外层 `smooth_weight=1e-4`，时间差项同理。对同一残差，旧有效强度约为官方的 **333.3 倍** |

关键差异同时涉及权重位置和归约维度：官方在 camera loss 内使用 `1e-3`，外部再乘 camera weight `0.1`；残差先对 xyz 求和，再对样本和时间求均值。只改外层系数而不核对 `mean` / `sum`，仍不能得到相同的正则强度。

#### 2.2.5 置信度监督

有效像素标签由尺度对齐后的逆深度加权 L1 误差定义：逐像素误差小于 0.02 时为正标签，标签本身不反传。无效像素的目标为 0。

| 编号 | 等级 | 修正前问题 / 影响 | 当前修正 |
| --- | --- | --- | --- |
| R05 | P1 | 置信度只训练有效深度像素，无效区域没有负样本；全无效深度时 BCE 为 NaN | 默认使用 `BCE_valid + BCE_invalid` 两个分别归一化的均值；invalid 标签为 0；camera-only 的两个分支均跳过；缺失 logits 且要求置信度训练时明确报错 |

当前使用 `BCE_valid + BCE_invalid`，两个分支分别求均值后相加，不是把所有像素拼在一起求一个 BCE 均值。camera-only 序列不参加这两项；可通过 `confidence_invalid_as_zero=false` 关闭无效区域负样本项。confidence-only 阶段的初始化与可训练参数范围见 2.1.3 节。

#### 2.2.6 边界情况与日志/可视化一致性

| 编号 | 等级 | 修正前问题 / 影响 | 当前修正 |
| --- | --- | --- | --- |
| R03 | P1 | 无有效点时预测归一化除零，空 L1/BCE 的 mean 产生 NaN；无效点 NaN 可进入法向算术 | 有效点计数为零时使用尺度 1；零长度张量构造可微零；无效坐标在参与运算前屏蔽；单帧与静止轨迹有定义 |
| R17 | P2 | 原 loss 原地修改 prediction，重复求值和可视化依赖调用顺序；可视化还用另一套最小二乘尺度 | loss 使用独立字典及姿态副本；图中采用训练相同的归一化和尺度求解，包括 camera-only |

纯位姿序列使用 2.2.1 节的轨迹归一化；普通 RGB-D 样本没有有效点时，不通过零分母或空均值产生 NaN。单帧、静止轨迹和无效点占位也有明确定义。

预测输入不再被 loss 原地修改，可视化使用与训练相同的归一化和尺度求解。因此，重复求值或先后调用可视化不会改变后续 loss 的输入含义。

#### 2.2.7 总目标、阶段开关与验证

默认各项组合为：

```text
L = L_points + L_normal
  + 0.1 * (100 * L_translation + L_rotation)
  + 0.0001 * (M + S)
  + confidence_weight * (BCE_valid + BCE_invalid)
```

官方把两个旋转修正正则包含在 camera loss 内，本地保留历史 `pose_loss`、`smooth_loss` 分开记录；**默认总目标及梯度一致**，比较日志时需考虑这一拆分。

Stage I/II 的 confidence 默认关闭，Stage I 的 rotation refiner 也关闭；本地 Stage III 示例将点图、法向、位姿及旋转修正的外层权重置零，仅保留 confidence weight=0.05。因此，上式描述的是各项的默认组合规则，不意味着每个阶段都同时训练所有分支。

| 已验证内容 | 证据 |
| --- | --- |
| RGB-D/稀疏/camera-only 混合 batch，confidence 开/关；点、法向、平移、旋转、两类修正正则的数值与输入梯度 | [官方源码差分测试](../../../tests/test_abot_recon_official.py) |
| 单帧、空 mask、NaN 占位、camera-only 平移梯度、静止/异常运动尺度、置信度负样本、loss 不修改输入 | [损失回归测试](../../../tests/test_abot_recon_regressions.py) |
| 完美预测的官方 acos 下界、可视化尺度和无效批次处理 | [训练回归测试](../../../tests/test_abot_recon_training.py)、[可视化回归测试](../../../tests/test_abot_recon_viz.py) |

前次联合验证在 Python 3.12、PyTorch 2.5.1、torchvision 0.20.1、Accelerate 1.2.1 的独立临时环境中完成；53 项测试全部通过，唯一 warning 来自既有 SciPy deprecated import。`compileall` 和 `git diff --check` 也已通过，用户系统 Python 未被修改。

复验两个部分的全部检查，可在已安装仓库训练依赖的环境中运行：

```bash
git clone --branch train https://github.com/amap-cvlab/ABot-Recon.git /tmp/abot-reference
git -C /tmp/abot-reference checkout 24d427bbd2eeb6072140de5683796ba626995e18
ABOT_OFFICIAL_ROOT=/tmp/abot-reference python -m pytest -q tests/test_abot_recon_*.py
OMP_NUM_THREADS=1 python -m torch.distributed.run --standalone --nproc_per_node=2 tests/abot_recon_ddp_smoke.py
```

不设置 `ABOT_OFFICIAL_ROOT` 时，仅官方源码差分测试跳过，本地回归仍可运行。获取参照仓库需要网络，已有参照源码后的测试不依赖互联网或真实训练数据。数值与梯度对齐不替代真实数据上的收敛和最终精度验证。
