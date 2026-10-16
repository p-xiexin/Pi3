# Glob3R 训练实验与问题记录

## 模型训练记录

以下按 2026 年的开发时间段整理。

| 时间段 | 问题与观察 | 定位与处理过程 |
| --- | --- | --- |
| 7 月下旬（coarse 阶段，问题定位） | 训练无法收敛，warp 重采样图像几乎全黑；conf 可视化初期有部分正确响应，但随着训练进行也完全变黑 | 对照论文、RoMaV2 和匹配可视化，检查相似度归一化、坐标 embedding、DPT 输入与 NLL 标签，定位归一化及 reference/target 方向不一致的问题；方向问题已与作者确认 |
| 7 月下旬（coarse 阶段，修正验证） | 修改完成后，在 ScanNet 数据集上训练，得到初步正确的匹配结果 | 对 cosine logits 只做一次 softmax，将输入与监督统一到 reference→target 方向，同步修正坐标 embedding、DPT 输入与 NLL 标签，并调整 GT warp 采样与 ScanNet 验证（`4a42c9f`、`26dd57a`） |
| 7 月底（refinement 阶段） | coarse 修正后，在 ScanNet 上完成 refinement 的初步训练；随后接入单窗口 SfM 时，匹配质量仍不足，重建点云出现分层 | 参考原论文流程，将匹配结果用于 track 构建与单窗口重建，检查匹配在几何优化中的表现（`3677654`、`2ee5526`） |
| 8 月（数据适配与扩大训练） | 初步训练后的匹配质量不足；适配多种数据集并重新进行大规模训练后，匹配质量明显改善 | 扩展数据读取与采样适配，准备集群训练，重新进行大规模训练；结合匹配可视化观察训练前后的变化。相关代码节点包括集群入口与数据集适配提交（`5e09e04`、`a79342e`、`592f3bd`、`ad467af`） |
| 9 月上旬 | 将模型推送到 *** 后，在自采数据上从零训练，warp 出现大面积空洞，有效匹配面积不足，无法完成匹配任务 | 检查发现窗口内共视不足，且仅使用稀疏或半稠密点云数据；仅用稀疏数据的消融复现了同类现象。问题仍待解决，详见下节 |
| 9 月上旬 | 天空区域被几何有效掩码排除，置信度计算存在低精度 sigmoid 饱和风险 | 为天空添加零置信度监督，改用 FP32 logits 与 BCE-with-logits。同步整理阶段输出目录，新增 joint 配置用于匹配与 refinement 联合训练（`cf254cb`、`ba6edf0`） |
| 9 月中旬 | 探索在像素 warp 监督之外引入射线方向约束 | 新增 pose-ray 几何损失，比较预测 warp 对应的目标相机射线与 GT 深度、位姿确定的射线方向。保留独立开关与权重，用于和原 matching 损失进行消融对照（`55fd52a`） |
| 9 月下旬 | Pi3X 与 Glob3R 的评测需要接入同一入口 | 新增 Pi3X 模型适配和 Glob3R 评测适配器，调整模型配置与结果保存方式，为后续在相同数据上比较几何与匹配结果提供入口（`526305d`） |

训练方向问题的现象、推导与修改过程见 [邮件记录](../pi3/models/glob3r/email.md)，网络与监督定义见 [Glob3R 网络复现与实现差异](Glob3R网络复现与实现差异.md)。

## 当前主要问题：自采数据训练的匹配覆盖不足

9 月上旬，将 Glob3R 模型推送到 ***，在其自行采集的数据上从零训练，未加载此前训练好的 Glob3R 权重。训练结果中的 warp 出现大面积空洞，有效匹配面积不足，无法完成匹配任务。这是当前尚未解决的主要问题。

此前在 ScanNet 上得到初步正确结果，扩展数据并进行大规模训练后，匹配质量明显改善。对自采数据的检查及后续消融发现以下问题：

**视角共视不足。** 自采数据的视角采样稀疏，相邻帧之间的空间距离较远，训练窗口又设置得过大。窗口内大部分视角缺少重叠，无法提供有效的对应监督，无重叠视角的 warp 结果全黑。后续需结合实际共视范围调整采样与窗口大小。

**深度监督稀疏。** 自采数据仅提供稀疏或半稠密点云，训练时未混合具有高质量稠密标注的开源数据。针对这一点，已在本地做数据组成的消融对照：

| 数据组成 | 观察结果 |
| --- | --- |
| 仅使用稀疏点云数据 | 同样出现 warp 大面积空洞、有效匹配面积不足的问题 |
| 混合其他数据 | 未出现上述问题 |
| 完全不使用稀疏点云数据 | 未出现上述问题 |

这一对照说明，仅使用稀疏点云数据的训练设置能够复现该问题，混合数据的设置未出现同样现象。后续调整应优先考虑共视采样与高质量稠密数据的混合，并在服务器训练中确认匹配覆盖是否恢复。

## 主要问题二：Glob3R 匹配的时间与空间效率

以下结合服务器统计与代码分析 Glob3R head 的匹配开销。Pi3 主干、points head 和 camera head 属于两条路径的公共计算。

### 空间效率

设窗口有 $N$ 帧，patch 数 $M=HW/196$，特征维度 $C=1024$，batch 为 $B$，每个元素占 $b$ 字节。Glob3R 需要保留四层编码特征与 $2C$ 维几何 token；按每帧 $P$ 个 SIFT 点、128 维 FP32 描述子及 28 字节元数据比较：

$$
S_{\mathrm{G,input}}=6bBNMC,\qquad
S_{\mathrm{SIFT}}=540BNP.
$$

在 $672\times378$、每帧 20,000 个 SIFT 点和 BF16 特征的假设下，Glob3R 与 SIFT 保留特征的存储比约为 $1.47$。存储按独立张量计算，RGB 作为公共输入。

head 还会生成 $[B,N-1,M,M]$ 的相似度与概率，以及稠密 refinement 特征和输出。缓存特征可减少重复计算，并延长特征驻留时间；分批处理 target 和减少非必要输出，可用于控制显存。

同批提供的整模型统计如下，作为几何模型的资源参考：

| 统计项 | Pi3 | Pi3X |
| --- | --- | --- |
| 数据 | `data/chunk20` | ADS 序列 `0c8a90a51d25421d85137391a4961f64` |
| 帧数 | 20 | 60 |
| 分辨率 / 精度 | 672×378 / BF16 | 672×378 / BF16 |
| 多模态条件 | — | 未使用，分支关闭 |
| 推理耗时 | 3.393 s | 10.394 s |
| FPS / 每帧耗时 | 5.894 / 169.665 ms | 5.773 / 173.227 ms |
| points decoder | 65.816 ms | 193.444 ms |
| points head | 106.939 ms | 1111.409 ms |
| 推理前 allocated | 3.629 GiB | 3.871 GiB |
| forward 后 allocated | 3.788 GiB | 4.447 GiB |
| forward 峰值 allocated | 6.370 GiB | 22.697 GiB |
| forward 峰值 reserved | 6.662 GiB | 46.027 GiB |
| pipeline 峰值 allocated | 6.370 GiB | 22.697 GiB |
| pipeline 峰值 reserved | 7.348 GiB | 46.027 GiB |
| 导出点数 | 2,503,435 | 9,699,448 |

Pi3 测试使用 20 帧，Pi3X 使用另一序列的 60 帧。allocated 表示已分配给张量的显存，reserved 表示分配器保留的显存。

### 时间效率

服务器使用 `data/chunk20` 的 20 帧，分别以 `0/5/10` 为 reference，各记录 3 次 Glob3R head 计时。Pi3 权重为 `ckpts/Pi3/model.safetensors`，matching 权重为 `/starmap/nas/workspace/pxx/Pi3/outputs/glob3r_18532/refinement/ckpts/best_model/pytorch_model.bin`。下表为 Glob3R head 的计时结果。

| Reference | 平均耗时 | 最小～最大耗时 |
| --- | --- | --- |
| 0 | 714.430 ms | 712.278～716.059 ms |
| 5 | 724.274 ms | 722.723～725.526 ms |
| 10 | 699.510 ms | 697.970～701.133 ms |

汇总 9 次记录的分模块均值：

| 模块 | 平均耗时 | 约占 head 耗时 |
| --- | --- | --- |
| Matching decoder | 91.549 ms | 12.84% |
| Matching embedding | 3.827 ms | 0.54% |
| DPT matching | 51.580 ms | 7.24% |
| Refinement 特征提取 | 64.937 ms | 9.11% |
| Warp refinement | 500.850 ms | 70.27% |
| Glob3R head 整体 | 712.738 ms | 100% |

Warp refinement 平均耗时 500.850 ms，约占 head 总耗时的 70.27%，是主要优化对象。

同一窗口更换 reference 时，matching decoder 和 refinement 图像特征提取不依赖 reference，却会重复执行。设 reference 数为 $K$，两部分耗时分别为 $T_Z,T_\phi$，其余计算为 $T_p$，则：

$$
T_{\mathrm{current}}\approx K(T_Z+T_\phi+T_p),\qquad
T_{\mathrm{reuse}}\approx T_Z+T_\phi+KT_p.
$$

本次 $K=3$，三个 reference 的 head 耗时合计平均约为 2.138 s。按均值计算，两次重复的 decoder 与特征提取耗时约为 313 ms，复用的预计节省量为 14.6%。

| 优化方向 | 代码依据与处理思路 |
| --- | --- |
| 复用 matching tokens 与 refinement 图像特征 | 同一窗口提取一次；输入预处理和权重需保持不变 |
| 减少 warp refinement 开销 | 当前 stride $4/2$ 的局部相关分别逐偏移执行 49/9 次采样，可进一步拆分计时并探索融合算子；采样与卷积更新依赖当前 pair 和 warp，按 pair 执行 |
| 复用 DPT 的 reference 图像分支 | 当前对各 target 重复投影、金字塔与融合；该分支可先按 reference 计算，pair-specific 输出仍需分别执行 |

实现见 [Glob3R model.py](../pi3/models/glob3r/model.py)。后续可拆分 warp refinement 内部耗时，并测量特征复用后的性能。

## SfM 流水线记录

| 时间段 | 问题与观察 | 定位与处理过程 |
| --- | --- | --- |
| 7 月底 | 使用 ScanNet 初步训练的 refinement 模型，匹配效果不足，单窗口重建点云出现分层 | 参考原论文实现简化的单窗口 SfM，将匹配用于多视图 track 构建与几何优化，发现匹配问题在重建结果中表现为点云分层（`2ee5526`、`8bd4cdb`） |
| 8 月上旬 | 需要进一步区分匹配与优化后端对重建问题的影响 | 按多帧观测对应同一共享三维点组织优化，过滤支持不足的 track-frame 观测。增加 GT warp 与 VGGSfM tracks 对照入口，比较三角化与 Pi3 位姿初始化；Torch BA 参考 DROID 求解结构，保留可微分优化接入流式训练的探索方向（`ed71a45`、`32c9a29`、`3b6fc9b`、`d938274`） |
| 8 月中下旬 | 单窗口流程扩展到全序列后，需要检查跨窗口关联、裁剪区域与优化结果 | 增加滑动窗口因子图、前端缓存、几何验证、关键帧与回环处理；补充 mask 和轨迹导出。引入 COLMAP BA 用于后端正确性对照，适配 pycolmap 接口；VGGSfM 用于比较 Glob3R 前端的重建效果（`1cef0cf`、`3970e19`、`4bdc3ff`、`e36e6e5`） |
| 8 月底至 9 月初 | 全序列重建中的点云分层与尺度问题继续排查 | 增加跨窗口尺度估计、尺度探测与诊断输出，比较初始化路径；当时提交记录将分层现象定位为退化问题。随后使用 Pi3 位姿初始化并固定尺度规约，补充前后端对齐检查，修正 track 与匹配可视化（`add7303`、`30da5e6`、`0d710d5`、`bd8691f`） |
| 9 月下旬 | 稠密点图与优化后稀疏几何的对齐方式需要调整 | 将逐帧视差仿射拟合改为 chunk 内共享变换，并按帧平衡拟合权重，调整重建输出。结合对应序列比较调整前后的重建效果（`d4ae784`） |

SfM 当前流程、后端与运行方式见 [SfM 流水线与优化策略](SfM流水线与优化策略.md)。提交号对应各阶段的问题处理与对照入口。

## 实验配置与训练流程

当前入口为 `scripts/train_glob3r.py`。先填写 [coarse 配置](../configs/glob3r_coarse.yaml) 中的数据路径，并提供 Pi3 权重和 `run_root`：

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml   scripts/train_glob3r.py --config-name glob3r_coarse   run_root=/path/to/run   glob3r.backbone_checkpoint=/path/to/pi3.safetensors
```

输出按 `${run_root}/${glob3r.stage}` 分目录，包含训练日志和 `ckpts/`。多卡集群使用 [train.sh](../train.sh)，coarse 成功后提交依赖任务启动 refinement；迁移前需填写权重并替换环境、路径和资源。

## Coarse / Refinement / Joint 实验

| 配置 | 当前默认配方 |
| --- | --- |
| [Coarse](../configs/glob3r_coarse.yaml) | 2–48 视图，LR `1e-4`，80×800 micro-batches，累积 2 |
| [Refinement](../configs/glob3r_refinement.yaml) | 2–24 视图，LR `5e-5`，继承训练量；加载 coarse best model |
| [Joint](../configs/glob3r_joint.yaml) | 继承 refinement，10 epoch，LR `5e-5`，warmup 200 |

coarse/refinement 配置各对应 64K micro-batches、计划 32K 更新；实际有效更新数还受中断或跳过影响。Joint 同时训练匹配和 refinement，需显式指定起始 `glob3r.matching_checkpoint`。历史实验参数见各次保存的实际配置。

## 结果与推荐权重

推荐权重结合匹配可视化与独立评测结果选择，实验记录按下表整理：

| 实验 | 提交 / 实际配置 | 数据与训练量 | 结果 / 结论 | 日志 / 权重 |
| --- | --- | --- | --- | --- |
| 待补 | 待补 | 待补 | 待补 | 待补 |

TensorBoard 主要图像为 `train/matching_overview` 和 `val/matching_overview`，显示图像、预测/GT warp、置信度与 mask。

**待补：**coarse/refinement/joint 实际实验、曲线、指标、恢复记录和推荐 checkpoint。
