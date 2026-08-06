# Glob3R DROID-style local BA

```text
Pi3 window inference
  → Eq. (4) keyframe selection
  → Eq. (2) directed PairMatch(r → t)
  → explicit DroidFactorGraph
  → matching overview with exact BA factors
  → MoBA or full BA
  → keyframe dense reconstruction
```

`Frames` 保存单窗口的 `Is/Xs_C/Cs/Ks/deltas/T_WCs`。`matching.match_batch()` 对每个 reference keyframe 运行 matching head，并为每条有向边返回：

```python
PairMatch(
    r,             # reference frame index
    t,             # target frame index
    W_r2t,         # [Hm, Wm, 2], target original-pixel coordinates
    valid_r2t,     # [Hm, Wm]
    Q_r2t,         # [Hm, Wm]
)
```

Warp 遵循论文 Eq. (2)：

\[
W^{r\rightarrow t}(p_r)=p_t.
\tag{2}
\]

`W_r2t` 全程保持浮点坐标。`factor_graph.build_droid_factor_graph()` 将其 reference 空间网格双线性重采样到 DROID 的 stride-8 网格，坐标值只连续除以 8，不做 `round()`。目标帧置信度也在连续的 $p_t$ 上使用双线性 `grid_sample`：

\[
C_t^{r\rightarrow t}(p_r)
=
C_t\!\left(W^{r\rightarrow t}(p_r)\right).
\]

每条稠密因子的权重由 `Q_r2t`、有效范围、reference/target Pi3 confidence 和有效深度共同确定。matching 总览只在 `Images` 行叠加最终 BA factor：reference 使用 DROID 规则二维 stride-8 source grid，并绘制当前页面所有 outgoing edges 的有效并集；每个 target 列只使用本 edge 的 `graph.weight > 0` mask，在 `graph.target * stride` 的亚像素位置绘制同色点。`Warp` 与 `Conf` 行保持原始 matching head 输出不变。每个 target 列顶部标注 edge 和实际 factor 数量。`sfm.py` 在调用 solver 前显式构建 factor graph 和可视化 matching；adapter 只负责 SE(3) 表示转换、调用 vendored `MoBA/BA` 和返回结果。

`moba` 固定 Pi3 depth，只更新相机位姿；`ba` 联合更新相机位姿和 keyframe inverse depth。两种模式均固定内参与畸变，vendored `ba.py`、`chol.py`、`projective_ops.py` 保持上游源码不变。

full BA 的 stride-8 深度通过比例场反馈到 Pi3 全分辨率深度：

\[
\frac{D_{\mathrm{opt}}}{D_{\mathrm{init}}}
=
\frac{d_{\mathrm{init}}}{d_{\mathrm{opt}}},
\qquad d=\frac{1}{D}.
\]

`sfm.py` 双线性上采样该比例场并乘回 Pi3 depth，避免直接上采样低分辨率 depth 导致细节平滑。

优化前先把所有 Pi3 `T_WC` 变换到首帧坐标系，因此 `pi3_raw.ply` 与 `pi3_sfm.ply` 使用同一个世界坐标系。两份点云都只融合 Eq. (4) 选出的 keyframes。

当前实现是 DROID 风格的稠密 MoBA/full BA 路径，不构造论文 Eq. (5)–(6) 的 sparse tracks、landmarks 或完整 Glob3R BA。
