# SfM 流水线与优化策略

## 参考文献

- [1] Deng et al. [Glob3R: Global Structure-from-Motion with 3D Foundation Models](https://arxiv.org/html/2607.09225v1), 2026。前端匹配、关键帧与几何优化的主要参考。
- [2] Wang et al. [π³: Permutation-Equivariant Visual Geometry Learning](https://arxiv.org/abs/2507.13347), 2025；[Pi3 / Pi3X 代码](https://github.com/yyfz/Pi3)。本项目使用 Pi3 提供点图、位姿和置信度。
- [3] Teed and Deng. [DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras](https://arxiv.org/abs/2108.10869), NeurIPS 2021；[官方代码](https://github.com/princeton-vl/DROID-SLAM)。可微分 Torch BA 的实现参考。
- [4] Wang et al. [VGGSfM: Visual Geometry Grounded Deep Structure From Motion](https://arxiv.org/abs/2312.04563), CVPR 2024；[官方代码](https://github.com/facebookresearch/vggsfm)。匹配前端与重建效果的对照。

## 流水线与前端接入

全序列入口位于 `sfm/`，由滑动窗口前端与全局图优化后端组成：

```text
图像序列
  → 滑动窗口几何与匹配
  → 关键帧、轨迹及跨窗口尺度对齐
  → 回环检索与匹配因子
  → 全局位姿与三维点初始化
  → 几何优化与重建导出
```

Pi3 提供局部点图与 c2w 位姿，Glob3R 或 VGGSfM 提供二维观测。VGGSfM [4] 用作匹配前端对照，比较 Glob3R 的轨迹质量与重建效果。查询点来自 SuperPoint/SIFT，观测经置信度、可见性和几何检查过滤；关键帧按投影覆盖、置信度与最大间隔选择。Glob3R 路径共享 Pi3 编码结果用于几何与匹配。

同一 track 的多帧观测关联到共享三维点，滑动窗口与回环观测加入统一因子图。前端负责构建因子，后端完成全局初始化与优化；`local_opt/` 保留单窗口优化及 VGGSfM、GT warp 对照入口。

实现见 [pipeline.py](../sfm/pipeline.py)、[model.py](../sfm/model.py)和 [tracker.py](../sfm/tracker.py)。匹配网络说明见 [Glob3R 文档](Glob3R网络复现与实现差异.md)。

## 单窗口轨迹与优化公式

`local_opt/` 的单窗口流程对应 [1] 的几何、关键帧、匹配与优化步骤：

```text
Pi3 geometry（原式 (1)）
  → keyframes（原式 (4)）
  → dense warps（原式 (2)）
  → Tracks(rs, Xs_Cr, us, mask, ws)
  → ray-consistency optimization（原式 (5)）
  → sparse bundle adjustment（原式 (6)）
  → keyframe dense reconstruction
```

`Tracks` 按 `[S,P]` 组织观测，`S` 为窗口帧数，`P` 为轨迹数；第 `j` 列表示同一条 track。像素观测、权重和有效性分别保存在 `us[i,j]`、`ws[i,j]` 和 `mask[i,j]`：

$$
u_i^j=\texttt{us[i,j]},\qquad
\omega_{ij}=\texttt{ws[i,j]}.
$$

`rs[j]` 指定轨迹的 keyframe，`Xs_Cr[j]` 保存 Pi3 在该帧相机系中的 anchor point。单窗口共享世界点由 anchor 初始化：

$$
X_j^0=T_{WC,r_j}X_{r_j}(u_{r_j}^j).
$$

每条 track 使用一个共享三维变量，其他帧通过 Glob3R warp 提供二维观测。记 $\mathcal O$ 为有效观测集合，$R_i$ 为 w2c 旋转，$c_i$ 为世界系相机中心，$v_{ij}=K_i^{-1}\widetilde u_i^j$ 为相机系射线。原式 (5) 在固定旋转下优化相机中心、共享点和逐观测射线深度 $d_{ij}$：

$$
\min_{\{c_i\},\{X_j\},\{d_{ij}\}}
\sum_{(i,j)\in\mathcal O}
\omega_{ij}\rho\!\left(
\left\|X_j-\left(c_i+d_{ij}R_i^\top v_{ij}\right)\right\|_2^2
\right).
\tag{5}
$$

$\rho$ 为鲁棒惩罚。原式 (6) 继续优化同一组共享点与 w2c 位姿 $T_i$，$\pi$ 表示由内参 $K_i$ 和畸变参数 $\delta_i$ 定义的投影：

$$
\min_{\{T_i\},\{X_j\}}
\sum_{(i,j)\in\mathcal O}
\omega_{ij}\rho\!\left(
\left\|\pi(K_i,\delta_i,T_i,X_j)-u_i^j\right\|_2^2
\right).
\tag{6}
$$

优化后端分为 Torch BA 与 COLMAP 两条路径。可微分 Torch BA 最初用于参考 DROID-SLAM [3] 探索流式模型训练，将几何优化接入 PyTorch 计算流程；COLMAP 用作对照，验证优化实现的正确性。两者的对比应使用相同的观测、初始化与相机设置。优化后的共享点用于调整 keyframe 稠密点图的尺度并导出重建。

实现见 [matching.py](../local_opt/matching.py)、[sfm.py](../local_opt/sfm.py)及后端 [proj.py](../local_opt/backend/proj.py)、[chol.py](../local_opt/backend/chol.py)、[ba.py](../local_opt/backend/ba.py)。

## 跨窗口对齐与全局优化

重叠窗口通过共享帧的 Pi3 深度与置信度估计相对尺度，将各窗口几何对齐到已有尺度。全局位姿优先使用尺度对齐后的 Pi3 边构建最大生成树，再结合图中的相对旋转进行旋转平均；三维点由全局观测通过多视图 DLT 三角化初始化。

Pi3 输出采用 c2w，SfM 后端采用 w2c。记 $T_{WC}$ 为相机到世界的变换，则后端位姿为 $T_{CW}=T_{WC}^{-1}$。内参与图像裁剪、缩放保持一致；未提供标定时，读取器以原图长边估计焦距并将主点设在图像中心。

`global_ba_backend` 中，`native` 对应 Torch BA，`colmap` 对应 COLMAP 对照后端。当前全局后端先执行 15 次 bearing 调整，再执行两轮 BA，中间与末尾过滤重投影异常观测；首轮迭代数由 `global_iterations` 控制，第二轮为 10 次。native 后端固定参考相机并设置尺度规约。`local_iterations` 保留在配置校验中，当前全序列路径没有调用逐窗口局部优化。

实现见 [factor_graph.py](../sfm/factor_graph.py)、[optimizer.py](../sfm/optimizer.py)和 [colmap_optimizer.py](../sfm/colmap_optimizer.py)。

## 配置、运行与结果导出

使用前填写 [config.yaml](../sfm/config.yaml) 中的图像、权重与输出路径，并核对以下选项：

| 配置项 | 用途 |
| --- | --- |
| `images`、`calibration`、`mask` | 图像目录、可选相机标定与共享图像掩码 |
| `backbone_checkpoint`、`matching_checkpoint` | Pi3 与 Glob3R 权重 |
| `tracks_model` | `glob3r` 或 `vgg`；后者另需 `vggsfm_root` 与 `vggsfm_checkpoint` |
| `image_size`、`window_size` | 图像尺寸按 `[高,宽]` 填写，两个维度为 14 的倍数；窗口至少 3 帧 |
| `loop`、`dino_salad_checkpoint` | 是否启用回环及 SALAD 权重 |
| `global_ba_backend`、`global_iterations` | 全局优化后端与首轮 BA 迭代数 |
| `data_h5`、`data_loop_h5`、`tracks_output` | 滑窗缓存、回环缓存与外部 BA 轨迹包路径 |

以下命令在仓库根目录执行：

```bash
python -m sfm --config sfm/config.yaml
```

当前配置使用 16 帧窗口、`[378,672]` 图像尺寸、Glob3R tracks 和 native 后端。图像按文件名自然排序，窗口半重叠并覆盖序列尾部。回环通过 RGB DINOv2 SALAD 检索，默认余弦相似度阈值为 `0.5`，检索候选随后生成匹配与因子。

`data.h5` 保存滑窗前端结果，`data_loop.h5` 保存回环结果，缓存命中时复用前端数据。更换图像、权重或前端参数后，使用新的缓存路径，或清除旧缓存重新生成。

重建结果写入 `data_h5` 所在目录，轨迹包写入 `tracks_output`：

| 产物 | 内容 |
| --- | --- |
| `match/`、`loop/` | 滑窗匹配与回环诊断图 |
| `sparse_tracks.ply` | 稀疏轨迹点，包含来源帧与观测数 |
| `dense.ply` | 稠密点云，包含来源帧 |
| `camera_poses.obj` | 相机线框 |
| `tracks/` | 外部 BA 兼容包，包含 XML 与二进制 PLY，在全局优化前导出 |

缓存与导出实现见 [frontend_cache.py](../sfm/frontend_cache.py)、[tracks_export.py](../sfm/tracks_export.py)和 [export.py](../sfm/export.py)。
